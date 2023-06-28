device="${1:-0}"
config="${2:-generative}"
comment="${3:-''}"
CUDA_LAUNCHE_BLOCKING=1 CUDA_VISIBLE_DEVICES=$1 kernprof -l -v generative.py max_iteration=300 >profile_outputs/line_profile_${config}_${comment}.txt
# CUDA_LAUNCHE_BLOCKING=1 CUDA_VISIBLE_DEVICES=$1 kernprof -l -v main_sh.py --config-name=${config} wandb=False warm_up=1000 eval_iteration=1000 log_iteration=200 pos_grad_thresh=2e-4 adaptive_control_iteration=300 max_iteration=4000 mean_lr=1e-2 alpha_reset_period=2000 remove_low_alpha_period=2000 alpha_lr=5e-2 ssim_loss_mult=0.0 >profile_outputs/line_profile_${config}_${comment}.txt
# main_sh.py --config-name=$2 max_iteration=50 warm_up=0 >profile_outputs/line_profile_${config}_${comment}.txt
