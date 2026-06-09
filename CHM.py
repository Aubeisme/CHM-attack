import time, os, torch, argparse, warnings, glob, re, random, pandas, torchaudio, pytorch_wavelets as ptwt
from pathlib import Path
from assist import *
# from AttackASD.Loader.ContextualValLoader import ContextualValLoader as val_loader
from AttackASD.Loader.GlobalLoader import ContextualValLoader as val_loader
from AttackASD.utils.tools import *
import torch.nn.functional as F
from AttackASD.Loss.AttackLRloss import lossAV
import soundfile as sf
import wave
import torchvision.transforms.functional as TF
import importlib
from create_loconet_metadata import create_loconet_metadata
seed = 42
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description = "Attack")
parser.add_argument('--nDataLoaderThread', type=int, default=64,  help='Number of loader threads')  # pyright: ignore[reportUnusedCallResult]
parser.add_argument('--DEVICE',     type=str, default="cuda:0", help='GPU device, e.g. cuda:0')
# Data path
# [NOTE] 数据集路径需根据实际环境配置，可通过 --dataPathAVA 传入，或设置环境变量 DATA_PATH_AVA
parser.add_argument('--dataPathAVA',  type=str, default=os.environ.get("DATA_PATH_AVA", ""), help='Root path of AVA/UniTalk dataset')
parser.add_argument('--savePath',     type=str, default="exps/expRes")
parser.add_argument('--pretrainModel',         type=str, default="weight/pretrain_AVA.model",   help='ASD模型权重')
# Data selection
parser.add_argument('--evalDataType', type=str, default="val", help='Only for AVA, to choose the dataset for evaluation, val or test')
# For download dataset only, for evaluation only
# parser.add_argument('--downloadAVA',     dest='downloadAVA', action='store_true', help='Only download AVA dataset and do related preprocess')
parser.add_argument('--evaluation',      dest='evaluation', action='store_true', help='Only do evaluation by using pretrained model [pretrain_AVA.model]')
parser.add_argument('--evaNum', type=int, default=1000,  help='Number of samples')# Max is 8015
parser.add_argument('--modelName', type=str, default="LRASD", help='pretrain model')
parser.add_argument('--ablation', type=str, default="CHM_Final", help='Attack ablation mode')
parser.add_argument('--datatype', type=str, default="AVA", help='Dataset type: AVA or Uni')
parser.add_argument('--beta', type=float, default=0.5, help='Beta weight for visual loss')
# [NOTE] 对抗样本输出路径，默认 ./ADVset，可通过 --advpath 覆盖
parser.add_argument('--advpath', type=str, default="./ADVset", help='Base path for adversarial samples')
# [NOTE] 模型权重根目录，默认 ./AttackASD/ASDs，可按需修改
parser.add_argument('--modelRoot', type=str, default="./AttackASD/ASDs", help='Root directory of model weights')
# [NOTE] 评估工具脚本目录，默认 ./AttackASD/utils
parser.add_argument('--utilsDir', type=str, default="./AttackASD/utils", help='Directory of evaluation utility scripts')
args = parser.parse_args()

# [NOTE] 根据 datatype 自动设置数据集路径；如果 --dataPathAVA 已通过命令行或环境变量指定则优先使用
# 否则需手动设置 DATA_PATH_AVA 环境变量，或直接通过 --dataPathAVA 传入
if not args.dataPathAVA:
    if args.datatype == 'Uni':
        # UniTalk 数据集路径，请按实际环境修改
        args.dataPathAVA = os.environ.get("DATA_PATH_UNITALK", "")
    else:
        # AVA 数据集路径，请按实际环境修改
        args.dataPathAVA = os.environ.get("DATA_PATH_AVA_AVA", "")
# 将 advpath 与 modelName 联系起来
attackName = args.ablation + "_" + args.datatype + "_"
args.advpath = os.path.join(args.advpath, args.modelName, attackName)

ASD_module = importlib.import_module(name = "AttackASD.ASDs." + args.modelName + ".ASD")
ASD = ASD_module.ASD
# Data loader
args = init_args(args)
adv_path = args.advpath
if not os.path.exists(adv_path):
	os.makedirs(adv_path)

def preprocess(model, v_adv, a_adv, labels_full, numframes):
	v_emb_gray  = rgb_to_grey(v_adv)
	
	a_emb_mfcc = mfcc_torch(a_adv*32768.0, numframes)
	a_emb_r = model.model.forward_audio_frontend(a_emb_mfcc)   
	v_emb_r = model.model.forward_visual_frontend(v_emb_gray)
	if args.modelName == 'TalkNet':
		a_emb, v_emb = model.model.forward_cross_attention(a_emb_r, v_emb_r)
	else:
		a_emb, v_emb = a_emb_r, v_emb_r
	out = model.model.forward_audio_visual_backend(a_emb, v_emb)

	lab = labels_full[0].reshape((-1))
	return out, lab, a_emb_r, v_emb_r


def _eot_transform_visual(v_stack):
		"""
		单候选人版 EOT 变换
		v_stack: [B, T, H, W, C]，彩色视频，只包含目标说话人一条轨迹
		"""
		# 解包维度
		B, T, H, W, C = v_stack.shape

		# 拷贝一份，避免原地改动
		v_t = v_stack.clone()

		# 1) 水平翻转（W 轴）
		# dims=3 对应 W 维，因为形状是 [B, T, H, W, C]
		if random.random() < 0.05:
			v_t = torch.flip(v_t, dims=[3])

		# 2) 帧随机置零（frame drop）
		if 0.1 > 0:
			# 每一帧独立决定是否保留，1=保留，0=置零
			drop = (torch.rand(T, device=v_t.device) > 0.1).float()  # [T]
			v_t = v_t * drop.view(1, T, 1, 1, 1)

		# 3) 轻微时间平移（沿 T 维 roll）
		if 1 > 0:
			s = random.randint(-1, 1)
			if s != 0:
				# dims=1 对应 T 维（[B, T, H, W, C]）
				v_t = torch.roll(v_t, shifts=s, dims=1)

		# 4) 帧内块级遮挡（occlusion）
		if 0.3 > 0:
			occ_max_ratio = 0.3  # 最大遮挡比例（相对 H/W）

			for b in range(B):
				for t_idx in range(T):
					if random.random() < 0.3:
						# 随机块大小
						h_occ = int(H * random.random() * occ_max_ratio)
						w_occ = int(W * random.random() * occ_max_ratio)
						if h_occ <= 0 or w_occ <= 0:
							continue

						top  = random.randint(0, H - h_occ)
						left = random.randint(0, W - w_occ)

						# 该帧该块置零（所有颜色通道一起遮挡）
						v_t[b, t_idx, top:top + h_occ, left:left + w_occ, :] = 0.0

		return v_t
def _f1_prec_rec_from_arrays(y_true, y_score, thr):
	y_pred = (y_score >= thr).astype(numpy.int32)
	tp = int(((y_pred == 1) & (y_true == 1)).sum())
	fp = int(((y_pred == 1) & (y_true == 0)).sum())
	fn = int(((y_pred == 0) & (y_true == 1)).sum())
	prec = tp / (tp + fp + 1e-12)
	rec  = tp / (tp + fn + 1e-12)
	f1   = 2 * prec * rec / (prec + rec + 1e-12)
	return f1, prec, rec

def compute_f1_scores(gt_csv, pred_csv, sweep=True):
	import pandas as pd
	cols = ["video_id","frame_timestamp","entity_id"]
	gt   = pd.read_csv(gt_csv)
	pred = pd.read_csv(pred_csv)
	merged = gt.merge(pred[cols + ["score"]], on=cols, how="inner")
	y_true  = (merged["label"] == "SPEAKING_AUDIBLE").astype(numpy.int32).to_numpy()
	y_score = merged["score"].astype(float).to_numpy()
	f1_05, p_05, r_05 = _f1_prec_rec_from_arrays(y_true, y_score, 0.5)
	best_f1, best_thr, best_p, best_r = f1_05, 0.5, p_05, r_05
	if sweep:
		for t in numpy.linspace(0.0, 1.0, 101):
			f1, p, r = _f1_prec_rec_from_arrays(y_true, y_score, float(t))
			if f1 > best_f1:
				best_f1, best_thr, best_p, best_r = f1, float(t), p, r
	return {
		"F1@0.5": f1_05, "P@0.5": p_05, "R@0.5": r_05,
		"F1_best": best_f1, "thr_best": best_thr, "P_best": best_p, "R_best": best_r
	}

def calculate_tolerant_overlap(target_seq, bystander_seq, tolerance=5):

    # 1. 维度调整：MaxPool1d 需要 [Batch, Channel, Time] 格式
    # 我们把 bystander 变成 [Batch, 1, Time]
    bystander_unsqueezed = bystander_seq.unsqueeze(1).float()
    target_float = target_seq.float()

    # 2. 构造膨胀核 (Dilation Kernel)
    # kernel_size = tolerance * 2 + 1 (左5 + 自己 + 右5 = 11)
    k_size = tolerance * 2 + 1
    
    # 3. 形态学膨胀 (Dilation) -> 使用 MaxPool1d
    # padding=tolerance 保证输出的时间长度不变 (Time维度不变)
    # stride=1 保证逐帧扫描
    bystander_dilated = F.max_pool1d(
        bystander_unsqueezed, 
        kernel_size=k_size, 
        stride=1, 
        padding=tolerance
    ).squeeze(1) # 变回 [Batch, Time]

    # 4. 计算重叠 (Intersection)
    # 逻辑：Target在说话 AND (Bystander在附近5帧内说过话)
    overlap = target_float * bystander_dilated

    # 5. 计算重叠率 (Overlap Ratio)
    # 这里计算的是：Target 的说话帧里，有多少受到了干扰？
    # 分母加 1e-6 防止除以 0
    overlap_ratio = overlap.sum(dim=1) / (target_float.sum(dim=1) + 1e-6)
    
    return overlap_ratio

def attack(model, feature, audio, labels, numframes, num_real_bystanders, attc,
		   mu=1.0, epsilon_v=6, epsilon_a=0.001,
		   steps=20, device="cuda:0",
		   lambda_sync=0.5, lambda_vis=0.7, margin_sync=0.1, margin_vis_sim=0.9, margin_vis_dist=10.0):

	alpha_v = epsilon_v / steps
	alpha_a = epsilon_a / steps

	model.train()
	for m in model.modules():
		if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
			m.eval()
		if isinstance(m, torch.nn.Dropout):
			m.eval()

	device = torch.device(device)
	if args.modelName == "LightASD" or args.modelName == "LRASD":
		ifeature = 128
	else:
		ifeature = 256

	loss_av = lossAV(ifeature).to(device)

	labels = labels.to(device)
	v_clean_target = feature[:, 0, ...].to(device)
	v_b = feature[:, 1, ...].to(device)
	v_c = feature[:, 2, ...].to(device)
	wo_b = attc[:, 0, ...].to(device)
	a_clean = audio.to(device)

	b, s, t = feature.shape[0], feature.shape[1], feature.shape[2]
	labels_full_target = labels[:, 0, ...].to(device)

	# ====== 预先计算干净 embedding 和 说话帧权重 w、干净 target 视觉特征 v_clean_t ======
	with torch.no_grad():
		_, _, _, v_clean = preprocess(
			model,
			v_clean_target,
			align_audio_length(a_clean.view(a_clean.shape[0], -1), numframes),
			labels_full_target,
			numframes
		)                            # v_clean: [T, D]                        # [T, D]
		active_B_ratio = calculate_tolerant_overlap(labels_full_target, labels[:, 1, ...].to(device))
		active_C_ratio = calculate_tolerant_overlap(labels_full_target, labels[:, 2, ...].to(device))
		att = v_b if active_B_ratio < active_C_ratio else v_c
		ratio = active_B_ratio if active_B_ratio < active_C_ratio else active_C_ratio
		silence_threshold = 0.5 
		
		is_mostly_silent = ratio < silence_threshold

		if is_mostly_silent and num_real_bystanders:
			_, _, _, att = preprocess(
				model,
				att.to(device),
				align_audio_length(a_clean.view(a_clean.shape[0], -1), numframes),
				labels[:, 2, ...].to(device),
				numframes
			) 
		else:
			_, _, _, att = preprocess(
				model,
				wo_b.to(device),
				align_audio_length(a_clean.view(a_clean.shape[0], -1), numframes),
				labels[:, 2, ...].to(device),
				numframes
			)
		v_clean_t = v_clean                        # [T, D]
	
	def loss_visual(v_T, v_cleaner, attr):
		
		impersonation_loss = - torch.norm(v_T - attr, p=2, dim=-1).mean()

		explosion_loss = torch.norm(v_T - v_cleaner, p=2, dim=-1).mean()

		loss_v = args.beta * impersonation_loss + explosion_loss
			
		return loss_v
	

	# ====== 初始化扰动与动量 ======
	delta_v = torch.zeros_like(v_clean_target, requires_grad=True, device=device)
	delta_a = torch.zeros_like(a_clean,       requires_grad=True, device=device)
	momentum_v = torch.zeros_like(v_clean_target, device=device)
	momentum_a = torch.zeros_like(a_clean,        device=device)

	# ====== 迭代攻击 ======
	for step in range(steps):
		model.zero_grad()
		delta_v.requires_grad_()
		delta_a.requires_grad_()

		if delta_v.grad is not None:
			delta_v.grad.zero_()
		if delta_a.grad is not None:
			delta_a.grad.zero_()

		g_ac = torch.zeros_like(a_clean,       device=device)
		g_vc = torch.zeros_like(v_clean_target, device=device)
		if step < 10:
			K = 5
		else:
			K = 1
		for _ in range(5):
			v_adv_target = v_clean_target + delta_v
			v_adv_target = _eot_transform_visual(v_adv_target)
			a_adv = a_clean + delta_a
			a_adv_aligned = align_audio_length(a_adv[0], numframes)

			with torch.no_grad():
				v_adv_target.clamp_(0, 255)
				a_adv_aligned.clamp_(-1, 1)

			# 前向
			out_A, lab_A, a_emb_r, adv_A_v = preprocess(
				model, v_adv_target, a_adv_aligned, labels_full_target, numframes
			)

			bce_loss_A, _, _, _, logits_A = loss_av.forward(out_A, lab_A, r=1)
			bce_loss = bce_loss_A

			v_T = adv_A_v         # [T, D]

			vis_loss = loss_visual(v_T, v_clean_t, att)
			
			# 总损失（三项）：BCE + av_sim + loss_visual
			loss = (bce_loss + 0.05 * vis_loss) / float(K)

			loss.backward()
			if delta_v.grad is not None:
				g_vc += delta_v.grad
				delta_v.grad.zero_()
			if delta_a.grad is not None:
				g_ac += delta_a.grad
				delta_a.grad.zero_()

		# ===== MI-FGSM 更新 =====
		with torch.no_grad():
			gv = g_vc
			ga = g_ac
			if gv is None:
				gv = torch.zeros_like(delta_v)
			if ga is None:
				ga = torch.zeros_like(delta_a)

			denom_v = gv.abs().mean(dim=tuple(range(1, gv.dim())), keepdim=True) + 1e-8
			momentum_v = mu * momentum_v + (gv / denom_v)
			delta_v += alpha_v * momentum_v.sign()
			delta_v.clamp_(-epsilon_v, epsilon_v)

			denom_a = (ga.abs().mean(dim=1, keepdim=True) + 1e-8) if ga.dim() >= 2 \
					  else (ga.abs().mean() + 1e-8)
			momentum_a = mu * momentum_a + (ga / denom_a)
			delta_a += alpha_a * momentum_a.sign()
			delta_a.clamp_(-epsilon_a, epsilon_a)

		delta_v = delta_v.detach()
		delta_a = delta_a.detach()
		if delta_v.grad is not None:
			delta_v.grad.zero_()
		if delta_a.grad is not None:
			delta_a.grad.zero_()

	return delta_v.detach(), delta_a.detach()

def attack_network(model, loader, device, evalCsvSave, evalOrig, max_samples=None, **kwargs):
	print(evalOrig)
	print(evalCsvSave)
	if max_samples is not None:
		df_gt = pandas.read_csv(evalOrig)
		unique_entities = df_gt["entity_id"].unique()[:max_samples]
		df_gt = df_gt[df_gt["entity_id"].isin(unique_entities)]
		evalOrig_tmp = evalOrig.replace('.csv', '_tmp.csv')
		df_gt.to_csv(evalOrig_tmp, index=False)
		evalOrig = evalOrig_tmp

	predScores = []
	df = pandas.read_csv(evalOrig)
	print(df)
	grouped = df.groupby("entity_id")
	entity_audio_cache = {}
	rows = []

	# 确保 CSV 目录存在并清空之前的文件
	csv_dir = os.path.join(adv_path, 'csv')
	if not os.path.exists(csv_dir):
		os.makedirs(csv_dir)

	# 清空之前的 val_orig.csv 文件（如果存在）
	val_orig_path = os.path.join(adv_path, 'csv', 'val_orig.csv')
	if os.path.exists(val_orig_path):
		os.remove(val_orig_path)

	for idx, (audioFeature, visualFeatures_context, labels_context, masks_context, ei, numframes, num_real_bystanders,_,att) in enumerate(tqdm.tqdm(loader)):
		entity_id = ei if isinstance(ei, str) else ei[0]
		df_entity = grouped.get_group(entity_id)
		numframes = numframes.item()

		target_labels = labels_context[:, 0, ...].clone()
		audioFeature = audioFeature.float() / 32768.0
		# 添加批次维度以匹配 align_audio_length 的期望
		audioFeature = audioFeature.unsqueeze(0)  # [1, T] -> [B, T]
		per_v, per_a = attack(model, visualFeatures_context, audioFeature, labels_context, numframes, num_real_bystanders, att, device=device)
		# per_v, per_a = torch.zeros_like(visualFeatures_context[:, 0, ...], device=device), torch.zeros_like(audioFeature, device=device)
		adv_target_visual = visualFeatures_context[:, 0, ...].clone().to(device) + per_v
		adv_target_visual.clamp_(0, 255)

		# Preprocess for model input (convert to grayscale)
		adv_target_visual_gray = rgb_to_grey(adv_target_visual)

		# Apply perturbation to audio
		adv_audio = audioFeature.to(device) + per_a
		# 处理音频维度以匹配 align_audio_length 的期望
		adv_audio_aligned = align_audio_length(adv_audio[0], numframes)
		adv_audio_aligned.clamp_(-1, 1)
		adv_audio_mfcc = mfcc_torch(adv_audio_aligned*32768.0, numframes)
		model.eval()
		with torch.no_grad():
			audioEmbed  = model.model.forward_audio_frontend(adv_audio_mfcc)
			visualEmbed = model.model.forward_visual_frontend(adv_target_visual_gray)
			if args.modelName == 'TalkNet':
				audioEmbed, visualEmbed = model.model.forward_cross_attention(audioEmbed, visualEmbed)
			outsAV= model.model.forward_audio_visual_backend(audioEmbed, visualEmbed)
			labels_eval = target_labels[0].reshape((-1)).to(device)
			_, predScore, _, _ = model.lossAV.forward(outsAV, labels_eval)
			predScore = predScore[:,1].detach().cpu().numpy()
			predScores.extend(predScore)

		adv_target_visual = adv_target_visual.squeeze(0)  # [T, H, W, C]
		adv_audio = adv_audio.clamp(-1,1).squeeze(0).detach().cpu()
		# 2. 保存
		if entity_id not in entity_audio_cache:
			entity_audio_cache[entity_id] = []
		entity_audio_cache[entity_id].append(adv_audio)
		for i, (_, row) in enumerate(df_entity.iterrows()):
			row_dict = row.to_dict()
			# adv_target_visual[i] 是 [H, W, C]，需要转换为 [C, H, W]
			frame = adv_target_visual[i].permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
			new_row = save_for_loader_csv(row_dict, adv_audio, frame, dst_root=adv_path)
			rows.append(row_dict)

	# 循环结束后，一次性写入所有数据到 CSV 文件
	df_out = pandas.DataFrame(rows)
	# 使用我们刚生成的 adv groundtruth 进行评估，避免与原始 AVA groundtruth 不匹配
	val_orig_path = os.path.join(adv_path, 'csv', 'val_orig.csv')
	df_out.to_csv(val_orig_path, index=False)

	# 保存拼接的音频片段
	for entity_id, segs in entity_audio_cache.items():
		full_audio = torch.cat(segs, dim=0)  # 拼接
		if args.datatype == "Uni":
			video_id = entity_id.split(":")[0]
		else:
			video_id = entity_id.split("_")[0]
		
		audio_dir = Path(adv_path + "/clips_audios/val") / video_id
		audio_dir.mkdir(parents=True, exist_ok=True)
		write_wav_with_colon_safe(audio_dir / f"{entity_id}.wav", full_audio, sr=16000)

	# 生成 loader 辅助 CSV
	if args.datatype == "Uni":
		cmd = "python -O ori2lor.py --input_csv %s --output_csv %s "%(val_orig_path, os.path.join(adv_path, 'csv', 'val_loader.csv'))
	else:
		cmd = "python -O toloader.py --input_csv %s --output_csv %s "%(val_orig_path, os.path.join(adv_path, 'csv', 'val_loader.csv'))
	# cmd = "python -O toloader.py --input_csv %s --output_csv %s "%(val_orig_path, os.path.join(adv_path, 'csv', 'val_loader.csv'))
	subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

	# 构造预测结果，保证长度一致，避免 NaN
	evalRes = pandas.read_csv(val_orig_path)
	n_pred = len(predScores)
	n_gt = len(evalRes)
	if n_pred != n_gt:
		print(f"[WARN] predScores length ({n_pred}) != groundtruth rows ({n_gt}), truncating to min length.")
	min_len = min(n_pred, n_gt)
	evalRes = evalRes.iloc[:min_len].copy()
	evalRes['score'] = pandas.Series(predScores[:min_len])
	evalRes['label'] = pandas.Series(['SPEAKING_AUDIBLE'] * min_len)
	for col in ['label_id', 'instance_id']:
		if col in evalRes.columns:
			evalRes.drop([col], axis=1, inplace=True)

	# 将预测写到 adv 路径下，避免覆盖原始实验路径
	pred_path = os.path.join(adv_path, 'csv', 'val_res.csv')
	evalRes.to_csv(pred_path, index=False)

	# 运行评估脚本
	
	# [NOTE] 评估脚本路径: {utilsDir}/get_ava_active_speaker_performance.py
	eval_script = os.path.join(args.utilsDir, "get_ava_active_speaker_performance.py")
	cmd = "python -O %s -g %s -p %s "%(eval_script, val_orig_path, pred_path)
	out = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	stdout = out.stdout.decode('utf-8').strip()
	stderr = out.stderr.decode('utf-8').strip()
	print("RAW stdout:", stdout)
	print("RAW stderr:", stderr)

	# 尝试从 stdout/stderr 提取 mAP；失败则回退为 0.0，避免 AttributeError
	mAP_match = re.search(r'average\s+precision:\s*([0-9]*\.[0-9]+)', stdout) or re.search(r'average\s+precision:\s*([0-9]*\.[0-9]+)', stderr)
	if mAP_match:
		mAP = float(mAP_match.group(1))
	else:
		print("[ERROR] Failed to parse mAP. See logs above. Returning 0.0.")
		mAP = 0.0
	f1 = compute_f1_scores(val_orig_path, pred_path, sweep=True)
	print(f"[F1] @0.5={f1['F1@0.5']:.4f}  (P={f1['P@0.5']:.4f}, R={f1['R@0.5']:.4f});  "
		  f"bestF1={f1['F1_best']:.4f} @ thr={f1['thr_best']:.2f} "
		  f"(P={f1['P_best']:.4f}, R={f1['R_best']:.4f})")
	return mAP

def main():
	# This code is modified based on this [repository](https://github.com/TaoRuijie/TalkNet-ASD).

	print(args)
	DEVICE = args.DEVICE
	# loader = train_loader(trialFileName = args.trainTrialAVA, \
	#                       audioPath      = os.path.join(args.audioPathAVA , 'train'), \
	#                       visualPath     = os.path.join(args.visualPathAVA, 'train'), \
	#                       **vars(args))
	# trainLoader = torch.utils.data.DataLoader(loader, batch_size = 1, shuffle = True, num_workers = args.nDataLoaderThread, pin_memory = True, prefetch_factor = 2)

	csv_dir = os.path.dirname(args.evalTrialAVA)
	entity_json = os.path.join(csv_dir, 'val_entity.json')
	ts_json = os.path.join(csv_dir, 'val_ts.json')
	
	# 如果元数据文件不存在，自动生成
	if not os.path.exists(entity_json) or not os.path.exists(ts_json):
		print(f"LoCoNet metadata files not found in {csv_dir}")
		print("Generating metadata files from val_orig.csv...")
		try:
			create_loconet_metadata(args.evalOrig, csv_dir)
			print("✓ Metadata files created successfully")
		except Exception as e:
			print(f"Error creating metadata: {e}")
			raise
	else:
		print(f"✓ Found existing metadata files in {csv_dir}")

	loader = val_loader(trialFileName = args.evalTrialAVA, \
						audioPath     = os.path.join(args.audioPathAVA , args.evalDataType), \
						visualPath    = os.path.join(args.visualPathAVA, args.evalDataType), \
						num_speakers=3,
						**vars(args))
	subset = torch.utils.data.Subset(loader, range(args.evaNum))
	valLoader = torch.utils.data.DataLoader(subset, batch_size = 1, shuffle = False, num_workers = 64, pin_memory = True)
	model = ASD(device=DEVICE)
	# [NOTE] 模型权重路径: {modelRoot}/{modelName}/{pretrainModel}
	model.loadParameters(os.path.join(args.modelRoot, args.modelName, args.pretrainModel))

	mAPs = []

	mAP = attack_network(model=model, loader = valLoader, device = DEVICE, max_samples=args.evaNum, **vars(args))
	def upsert_result_csv(key_method, gen_model, map_value, csv_path="result.csv"):
		import pandas as pd, os
		if os.path.exists(csv_path):
			df = pd.read_csv(csv_path)
		else:
			df = pd.DataFrame(columns=["model-attackmethod","mAP","LRASD","LightASD","TalkNet","LoCoNet_ASD"])
		for c in ["model-attackmethod","mAP","LRASD","LightASD","TalkNet","LoCoNet_ASD"]:
			if c not in df.columns:
				df[c] = "" if c!="mAP" else pd.NA
		key = f"{key_method}-{gen_model}"
		if (df["model-attackmethod"] == key).any():
			idx = df.index[df["model-attackmethod"] == key][0]
		else:
			idx = len(df)
			df.loc[idx, "model-attackmethod"] = key
		df.loc[idx, "mAP"] = map_value
		df.to_csv(csv_path, index=False)
	upsert_result_csv(attackName, args.modelName, mAP, "result.csv")
	print("mAP %2.2f%%"%(mAP))

if __name__ == '__main__':
	main()