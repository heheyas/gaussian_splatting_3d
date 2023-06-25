# 3d gaussian splatting

This branch contains another unofficial implementation (?) of the paper [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/). Currently **Work In Process**. Welcome to PR and discussions. For a better implementation, you can refers to [Feng Wang's repo](https://github.com/WangFeng18/3d-gaussian-splatting).


## This is currently a *Work In Process*
I roughly implement the functions required in the original paper, all of which reside in the `src/include` folder. **I have no idea on how to speed up renderering and backward.** So if you have any idea or suggestions, welcome to post a issue or email me, I will be grateful and honored to discuss with you ! (Please concat [Feng Wang](mailto:wang-f20@mails.tsinghua.edu.cn) or [Zilong Chen](mailto:chenzl22@mails.tsinghua.edu.cn))

## Performance
| Scene  | PSNR from paper | PSNR from this repo | Rendering Speed (official) | Rendering Speed (Ours) |
| ------ | --------------- | ------------------- | -------------------------- | :--------------------: |
| Garden | 25.82(5k)       | 24.08 (7k)          | 160 FPS (avg MIPNeRF360)   |         60 FPS         |

command : `python main_sh.py --config-name=garden_full pos_grad_thresh=2e-4 adaptive_control_iteration=100 alpha_reset_period=3000 remove_low_alpha_period=3000 alpha_scheduler=nothing svec_scheduler=nothing warmup_steps=300 use_train_sample=True split_scale_thresh=0.004 alpha_lr=0.1`
Ours 7K:
psnr: 24.17 ssim: 0.7611 ssim2: 0.8422 lpips: 0.1645
Ours 8K:
psnr: 24.14 ssim: 0.7604 ssim2: 0.8414 lpips: 0.1548
12K:
psnr: 24.59 ssim: 0.7792 ssim2: 0.8532 lpips: 0.1347

command: `python main_sh.py --config-name=garden_full pos_grad_thresh=2e-4 adaptive_control_iteration=100 alpha_reset_period=3000 remove_low_alpha_period=3000 alpha_scheduler=nothing svec_scheduler=nothing warmup_steps=300 use_train_sample=True split_scale_thresh=0.004 alpha_lr=0.1 sh_coeffs_lr=0.1 sh_coeffs_scheduler=nothing`
8K:
psnr: 24.37 ssim: 0.7764 ssim2: 0.8508 lpips: 0.139

## To run the code
1. first install all the requirements:
`pip install -e requirements.txt`
and addtional cuda extension:
`cd gs && ./build.sh`

2. prepare your data in `{workspace}/data`, make sure you have the outputs of colmap.

3. run the training script
`python main_sh.py`
using `viewer=True` to enable a Viser based viewer.

The configurations are managed with hydra and some of the configurations I have tested are in `conf`.

## TODOs:

- [ ] correct adaptive control
- [ ] train eval split and do complete evaluation
- [ ] profiling
- [ ] faster rendering !!


## Updates:
See update log [here](./update_log.md)