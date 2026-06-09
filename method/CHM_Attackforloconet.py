import torch, math, random
from assist import *
from . import vggish_params
import torchaudio.transforms as T
import torch.nn.functional as F
def waveform_to_examples(data: torch.Tensor, sample_rate: int, numFrames: int, fps: float):
	if data.ndim == 1:
		data = data.unsqueeze(0) # [T] -> [1, T] (B=1)
	elif data.ndim == 3:
		data = torch.mean(data, dim=1) # [B, C, T] -> [B, T]

	if sample_rate != vggish_params.SAMPLE_RATE:
		resampler = T.Resample(
			orig_freq=sample_rate, 
			new_freq=vggish_params.SAMPLE_RATE,
			dtype=data.dtype
		).to(data.device)
		data = resampler(data) # [B, T] -> [B, T_new]

	sample_rate_float = float(vggish_params.SAMPLE_RATE)
	
	window_length_seconds = vggish_params.STFT_WINDOW_LENGTH_SECONDS * 25. / fps
	hop_length_seconds = vggish_params.STFT_HOP_LENGTH_SECONDS * 25. / fps

	win_length_samples = int(round(sample_rate_float * window_length_seconds))
	hop_length_samples = int(round(sample_rate_float * hop_length_seconds))
	
	n_fft = int(2**numpy.ceil(numpy.log2(win_length_samples)))

	mel_spectrogram_op = T.MelSpectrogram(
		sample_rate=vggish_params.SAMPLE_RATE,
		n_fft=n_fft,
		win_length=win_length_samples,
		hop_length=hop_length_samples,
		f_min=vggish_params.MEL_MIN_HZ,
		f_max=vggish_params.MEL_MAX_HZ,
		n_mels=vggish_params.NUM_MEL_BINS,
		window_fn=torch.hann_window,
		power=1.0, 
		mel_scale='htk',
		norm=None,
		center=False
	).to(data.device)
	mel_spec = mel_spectrogram_op(data)
	log_mel = torch.log(mel_spec + vggish_params.LOG_OFFSET)

	log_mel = log_mel.permute(0, 2, 1)

	maxAudio = int(numFrames * 4)
	n_frames_audio = log_mel.shape[1] # [B, T_frames, n_mels]

	if n_frames_audio < maxAudio:
		shortage = maxAudio - n_frames_audio

		log_mel = F.pad(log_mel, (0, 0, 0, shortage), mode='circular')
	log_mel = log_mel[:, :maxAudio, :] # [B, maxAudio, n_mels]

	return log_mel

def preprocess_loconet(model, visualFeature, audioFeature, labels, masks, device):

	b, s, t = visualFeature.shape[0], visualFeature.shape[1], visualFeature.shape[2]
	visualFeature = visualFeature.view(b * s, *visualFeature.shape[2:])
	labels = labels.view(b * s, *labels.shape[2:])
	masks = masks.view(b * s, *masks.shape[2:])
	audioEmbed = model.model.forward_audio_frontend(audioFeature)
	visualEmbedo = model.model.forward_visual_frontend(visualFeature)
	audioEmbedo = audioEmbed.repeat(s, 1, 1)
	audioEmbed, visualEmbed = model.model.forward_cross_attention(
		audioEmbedo, visualEmbedo)
	outsAV = model.model.forward_audio_visual_backend(audioEmbed, visualEmbed, b, s)
	labels = labels.reshape((-1))
	masks = masks.reshape((-1))
	outsAV_all = outsAV.view(b, s, t, -1)  # [b, s, t, 2] 保留所有speaker
	outsAV = outsAV_all[:, 0, :, :].view(b * t, -1)
	labels = labels.view(b, s, t)[:, 0, :].view(b * t).to(device)
	masks = masks.view(b, s, t)[:, 0, :].view(b * t)
	return outsAV, labels, masks, visualEmbedo, audioEmbedo, outsAV_all
class CHM:
	def __init__(self, mu=1.0, epsilon_v=6, epsilon_a=0.01, steps=20,
				 p_vflip=0.1, p_mod_drop=0.05, eot_K=5,
				 max_shift=1, p_occ=0.3, occ_max_ratio=0.3, device="cuda:0"):
		self.mu = mu
		self.epsilon_v = epsilon_v
		self.epsilon_a = epsilon_a
		self.steps = steps
		self.device = torch.device(device)
		
		# EOT 参数
		self.p_vflip = p_vflip
		self.p_mod_drop = p_mod_drop
		self.max_shift = max_shift
		self.p_occ = p_occ
		self.occ_max_ratio = occ_max_ratio
		self.eot_K = int(eot_K)

		# InfoNCE 参数
		self.win = 13
		self.delta_t = 7
		self.tau = 0.05        # 保持低温
		self.impossible_margin = 2.0
		
		self._v_clean_target = None

	@torch.no_grad()
	def _set_eval_except_bn_dropout(self, model):
		model.train()
		for m in model.modules():
			if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm3d, nn.Dropout)):
				m.eval()

	def _eot_transform_visual(self, v_stack):
		B, S, T, H, W, C = v_stack.shape
		v_t = v_stack[:, 0:1, ...] 
		
		if random.random() < self.p_vflip:
			v_t = torch.flip(v_t, dims=[4])

		if self.p_mod_drop > 0:
			drop = (torch.rand(T, device=v_t.device) > self.p_mod_drop).float()
			v_t = v_t * drop.view(1, 1, T, 1, 1, 1)

		if self.max_shift > 0:
			s = random.randint(-self.max_shift, self.max_shift)
			if s != 0:
				v_t = torch.roll(v_t, shifts=s, dims=2)
		
		if self.p_occ > 0:
			for b in range(B):
				for t_idx in range(T):
					if random.random() < self.p_occ:
						h_occ = int(H * random.random() * self.occ_max_ratio)
						w_occ = int(W * random.random() * self.occ_max_ratio)
						if h_occ > 0 and w_occ > 0:
							top = random.randint(0, H - h_occ)
							left = random.randint(0, W - w_occ)
							v_t[b, 0, t_idx, top:top+h_occ, left:left+w_occ, :] = 0.0
		
		v_stack_new = v_stack.clone()
		v_stack_new[:, 0:1, ...] = v_t
		return v_stack_new

	def smooth(self, sim_1x1T, win):
		return F.avg_pool1d(sim_1x1T, kernel_size=win, stride=1, padding=win // 2).squeeze(0).squeeze(0)

	def rgb_to_grey(self, x):
		if x.shape[-1] == 3:
			return (x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114).unsqueeze(-1)
		return x

	def loss_visual(self, v_T, v_cleaner, att):
		impersonation_loss = -torch.norm(v_T - att, p=2, dim=-1).mean()
		
		explosion_loss = torch.norm(v_T - v_cleaner, p=2, dim=-1).mean()
		loss_v = impersonation_loss + explosion_loss
		return loss_v
	def calculate_tolerant_overlap(self, target_seq, bystander_seq, tolerance=5):

		bystander_unsqueezed = bystander_seq.unsqueeze(1).float()
		target_float = target_seq.float()


		k_size = tolerance * 2 + 1

		bystander_dilated = F.max_pool1d(
			bystander_unsqueezed, 
			kernel_size=k_size, 
			stride=1, 
			padding=tolerance
		).squeeze(1) # 变回 [Batch, Time]

		overlap = target_float * bystander_dilated

		overlap_ratio = overlap.sum(dim=1) / (target_float.sum(dim=1) + 1e-6)
		
		return overlap_ratio
	def attack(self, model, feature, audio, labels, masks, numframes, attc, num_real_bystanders):

		mu = self.mu
		epsilon_v = self.epsilon_v
		epsilon_a = self.epsilon_a
		steps = self.steps
		device = self.device

		feature = feature.clone()
		audio = audio.clone()
		labels = labels.to(device)
		masks = masks.to(device)

		alpha_v = epsilon_v / steps
		alpha_a = epsilon_a / steps

		self._set_eval_except_bn_dropout(model)

		# 三条轨迹
		v_clean_target = feature[:, 0, ...].to(device)
		v_b = feature[:, 1, ...].to(device)
		v_c = feature[:, 2, ...].to(device)
		a_clean = audio.to(device)

		b, s, t = feature.shape[0], feature.shape[1], feature.shape[2]

		with torch.no_grad():
			# 视觉前端干净特征 (3条轨迹)
			v_clean = feature.view(b * s, *feature.shape[2:])
			v_clean = model.model.forward_visual_frontend(rgb_to_grey(v_clean))
			v_B = v_clean[1, ...]                # [T, D]
			v_C = v_clean[2, ...] 
			v_clean_t = v_clean[0, ...]          # [T, D] target 干净 embedding
			if num_real_bystanders > 0:
				active_B_ratio = self.calculate_tolerant_overlap(labels[:, 0, ...].to(device), labels[:, 1, ...].to(device))
				active_C_ratio = self.calculate_tolerant_overlap(labels[:, 0, ...].to(device), labels[:, 2, ...].to(device))
				att = v_B if active_B_ratio < active_C_ratio else v_C
				ratio = active_B_ratio if active_B_ratio < active_C_ratio else active_C_ratio
				silence_threshold = 0.6 
				is_mostly_silent = ratio < silence_threshold
				if not is_mostly_silent:
					att = feature.view(b * s, *attc.shape[2:])
					att = model.model.forward_visual_frontend(rgb_to_grey(att))[0, ...]
			else:
				att = feature.view(b * s, *attc.shape[2:])
				att = model.model.forward_visual_frontend(rgb_to_grey(att))[0, ...]
			
		# 音频归一化到 [-1,1]
		a_clean = a_clean.float() / 32768.0

		# 初始化扰动 & 动量
		delta_v = torch.zeros_like(v_clean_target, requires_grad=True, device=device)
		delta_a = torch.zeros_like(a_clean,        requires_grad=True, device=device)
		momentum_v = torch.zeros_like(v_clean_target, device=device)
		momentum_a = torch.zeros_like(a_clean,        device=device)

		for step in range(steps):

			if delta_v.grad is not None:
				delta_v.grad.zero_()
			if delta_a.grad is not None:
				delta_a.grad.zero_()

			g_v_accum = torch.zeros_like(delta_v)
			g_a_accum = torch.zeros_like(delta_a)
			if step < 10:
				K = self.eot_K
			else:
				K = 1

			for _ in range(self.eot_K):
				model.zero_grad()
				delta_v.requires_grad_()
				delta_a.requires_grad_()

				# --- 构造 EOT 视图 ---
				v_adv_target = v_clean_target + delta_v
				with torch.no_grad():
					v_adv_target.clamp_(0, 255)

				v_adv = torch.stack([v_adv_target, v_b, v_c], dim=1)  # [B,3,T,H,W,C]
				v_adv = self._eot_transform_visual(v_adv)
				v_adv = rgb_to_grey(v_adv)

				a_adv = a_clean + delta_a
				a_adv = a_adv * 32768.0
				a_adv = waveform_to_examples(a_adv, 16000, numframes, 25.0)  # [B, Tm, nm]
				a_adv = a_adv.unsqueeze(0)

				# --- 前向传播 ---
				outs_A, lab_A, mask_A, v_emb_r_target, a_emb_r, _ = preprocess_loconet(
					model, v_adv, a_adv, labels, masks, device
				)

				bce_loss_A, _, _, _ = model.lossAV.forward(outs_A, lab_A, mask_A)
				v_T = v_emb_r_target[0, ...]    # [T, D]

				
				vs = self.loss_visual(v_T, v_clean_t, att)
				
				
				# 总 loss（仍然只有 3 项）
				loss = (bce_loss_A + 0.05 * vs) / float(K)
				loss.backward()
				
				if delta_v.grad is not None:
					g_v_accum += delta_v.grad
					delta_v.grad.zero_()
				if delta_a.grad is not None:
					g_a_accum += delta_a.grad
					delta_a.grad.zero_()

			# --- MI-FGSM 更新 ---
			with torch.no_grad():
				gv, ga = g_v_accum, g_a_accum

				if gv is None:
					gv = torch.zeros_like(delta_v)
				if ga is None:
					ga = torch.zeros_like(delta_a)

				# video
				denom_v = gv.abs().mean(dim=tuple(range(1, gv.dim())), keepdim=True) + 1e-8
				momentum_v.mul_(mu).add_(gv / denom_v)
				delta_v.add_(alpha_v * momentum_v.sign()).clamp_(-epsilon_v, epsilon_v)

				# audio
				if ga.dim() >= 2:
					denom_a = ga.abs().mean(dim=1, keepdim=True) + 1e-8
				else:
					denom_a = ga.abs().mean() + 1e-8
				momentum_a.mul_(mu).add_(ga / denom_a)
				delta_a.add_(alpha_a * momentum_a.sign()).clamp_(-epsilon_a, epsilon_a)

			delta_v = delta_v.detach()
			delta_a = delta_a.detach()
			if delta_v.grad is not None:
				delta_v.grad.zero_()
			if delta_a.grad is not None:
				delta_a.grad.zero_()

		pad_b = torch.zeros_like(v_b, device=device)
		pad_c = torch.zeros_like(v_c, device=device)
		delta = torch.stack([delta_v.detach(), pad_b, pad_c], dim=1)
		return delta, delta_a.detach()
