device="${1:-0}"
config="${2:-garden}"
comment="${3:-''}"
CUDA_VISIBLE_DEVICES=$1 python main.py --config-name=$2
