#!/usr/bin/env bash
# 自动续跑训练脚本（支持崩溃后从上次 checkpoint 恢复）
# 放在项目根目录运行： ./auto_resume.sh

set -u   # 未定义变量时报错；不要加 -e（需要循环重试）

set +u
# 1) 激活 conda 环境（优先用系统里的 conda，其次尝试 /opt/conda）
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  # shellcheck disable=SC1091
  source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || {
    echo "[ERROR] 找不到 conda.sh，请确认已安装 Anaconda/Miniconda 并把其加入 PATH。"
    exit 1
  }
fi
conda activate heal-stamp || {
  echo "[ERROR] 激活 conda 环境 heal-stamp 失败，请检查环境是否存在。"
  exit 1
}
set -u

# 2) 以脚本所在目录为基准，构造相对路径
DIR="$(cd -- "$(dirname "$0")" >/dev/null 2>&1 && pwd -P)"

# 固定参数（相对脚本根目录）
PYTHON=python
TRAIN_SCRIPT="$DIR/opencood/tools/train.py"
YAML="$DIR/opencood/hypes_yaml/opv2v/task_agnostic_segmentation/local/m1/config.yaml"
MODEL_DIR="$DIR/opencood/logs/m1_CoBevT_both_OPV2V_2025_08_25_15_54_40"

# 日志目录 = MODEL_DIR 的上一级目录 + /training_logs
PARENT_DIR="$(dirname "$MODEL_DIR")"
LOGDIR="$PARENT_DIR/training_logs"
mkdir -p "$LOGDIR"

# 环境：及时刷新 stdout
export PYTHONUNBUFFERED=1

# 重试控制
MAX_RETRIES=9999
SLEEP_SECS=5

i=1
while [ $i -le $MAX_RETRIES ]; do
  now="$(date '+%F %T')"
  echo "[$now] Run #$i starting..."
  echo "[$now] YAML=$YAML"
  echo "[$now] MODEL_DIR=$MODEL_DIR"
  echo "[$now] LOGDIR=$LOGDIR"

  # 开始训练；将 stdout/stderr 追加到统一日志
  $PYTHON "$TRAIN_SCRIPT" -y "$YAML" --model_dir="$MODEL_DIR" 2>&1 | tee -a "$LOGDIR/train.log"
  rc=${PIPESTATUS[0]}   # tee 的第一个进程（训练）的返回码

  if [ $rc -eq 0 ]; then
    echo "[`date '+%F %T'`] Training finished successfully."
    exit 0
  else
    echo "[`date '+%F %T'`] Training crashed (exit code $rc). Resume after ${SLEEP_SECS}s..."
    sleep "$SLEEP_SECS"
    i=$((i+1))
  fi
done

echo "[`date '+%F %T'`] Reached MAX_RETRIES=$MAX_RETRIES, giving up."
exit 1
