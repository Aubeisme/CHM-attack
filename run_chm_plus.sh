#!/bin/bash
# CHM_Plus (CAV-ReWire) batch evaluation script
#
# Usage:
#   ./run_chm_plus_ava.sh                           # 使用下方默认参数
#   修改下方变量后运行，或通过环境变量覆盖:
#     DATA_PATH_AVA=/path/to/ava ./run_chm_plus_ava.sh

set -e

# ========== 可配置参数（按需修改） ==========
SCRIPT="CHM.py"
# [NOTE] 模型列表，可扩展: ("LRASD" "LightASD" "TalkNet")
MODELS=("TalkNet")
# [NOTE] GPU 设备，按实际环境修改
DEVICE="${CUDA_DEVICE:-cuda:0}"
EVA_NUM=1000
# [NOTE] 数据集类型: AVA 或 Uni
DATATYPE="${DATATYPE:-Uni}"
ABLATION="CHM_PLUS"
# ============================================

for MODEL in "${MODELS[@]}"; do
    echo "============================================"
    echo "Running CHM_Plus on ${MODEL} | ${DATATYPE} | evaNum=${EVA_NUM}"
    echo "============================================"
    nohup python -O ${SCRIPT} \
        --modelName ${MODEL} \
        --DEVICE ${DEVICE} \
        --evaNum ${EVA_NUM} \
        --datatype ${DATATYPE} \
        --ablation ${ABLATION} \
        --beta_cbr 0.5 \
        --p_ghost 0.3 \
        --p_aeca 0.3 \
        --noise_std_aeca 0.05 > output.log 2>&1 &
    echo ""
    echo "[Done] ${MODEL} finished"
    echo ""
done

echo "All models completed."
