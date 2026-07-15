#!/bin/bash
# TurboQuant 验证启动脚本 — Qwen3-32B
# 用法: bash run_turboquant_qwen3.sh <model_path> [tp_size]

set -e

MODEL_PATH=${1:-"/path/to/Qwen3-32B"}
TP_SIZE=${2:-4}
VLLM_ASCEND_PATH="/你的本地路径/vllm-ascend-0.20.2rc1"

echo "========================================"
echo " TurboQuant 验证"
echo " 模型: $MODEL_PATH"
echo " TP:   $TP_SIZE"
echo "========================================"

# 关键：让 Python 优先使用本地修改的 vllm-ascend
export PYTHONPATH="${VLLM_ASCEND_PATH}:${PYTHONPATH}"

# 验证路径生效
python -c "import vllm_ascend; print('vllm_ascend:', vllm_ascend.__file__)"

# 验证 TQ 模块可导入
python -c "
from vllm_ascend.quantization.methods.turboquant import AscendTurboQuantAttentionMethod
from vllm_ascend.quantization.methods.turboquant_ops import store_turboquant_kv, dequant_paged_kv
from vllm_ascend.attention.attention_turboquant_v1 import AscendTurboQuantAttentionBackendImpl
from vllm_ascend.quantization.methods.registry import get_scheme_class
cls = get_scheme_class('turboquant', 'attention')
assert cls is not None, 'turboquant scheme not registered!'
print('TQ modules OK, scheme registered:', cls.__name__)
"

echo ""
echo ">>> 启动 vLLM serve..."
echo ""

vllm serve "$MODEL_PATH" \
  --kv-cache-dtype turboquant_3bit_nc \
  --tensor-parallel-size "$TP_SIZE" \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --port 8000
