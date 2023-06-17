device="${1:-0}"
comment="${2:-''}"
CUDA_VISIBLE_DEVICES=${device} sudo /usr/local/cuda/bin/ncu \
    --profile-from-start off \
    \
    --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct,smsp__sass_average_data_bytes_per_wavefront_mem_shared.pct,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
    /opt/czl/anaconda3/envs/gs/bin/python main_sh.py wandb=False max_iteration=1 only_forward=True # -f -o profile_outputs/ncu_${comment} \
