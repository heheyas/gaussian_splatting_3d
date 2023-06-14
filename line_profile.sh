device="${1:-0}"
config="${2:-garden}"
comment="${3:-''}"
CUDA_LAUNCHE_BLOCKING=1 CUDA_VISIBLE_DEVICES=$1 kernprof -l -v main.py --config-name=$2 max_iteration=100 >profile_outputs/line_profile_${config}_${comment}.txt
