device="${1:-0}"
comment="${2:-''}"
CUDA_VISIBLE_DEVICES=${device} nsys nvprof \
    --profile-from-start off \
    --export-profile profile_outputs/nvprof_${comment}.nvvp \
    -e shared_ld_bank_conflict,shared_st_bank_conflict \
    --metrics gld_throughput,gst_throughput,shared_load_throughput,shared_store_throughput \
    --log-file profile_outputs/nvprof_${comment}.log \
    python main_sh.py --config-name=garden_sh max_iteration=10 wandb=False
