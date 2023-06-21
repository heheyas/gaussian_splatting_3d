# 3d gaussian splatting

This branch contains another unofficial implementation (?) of the paper [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/). Currently **Work In Process**. Welcome to PR and discussions. For a better implementation, you can refers to [Feng Wang's repo](https://github.com/WangFeng18/3d-gaussian-splatting).


## This is currently a *Work In Process*
I roughly implement the functions required in the original paper, all of which reside in the `src/include` folder. **I have no idea on how to speed up renderering and backward.** So if you have any idea or suggestions, welcome to post a issue or email me, I will be grateful and honored to discuss with you ! (Please concat [Feng Wang](mailto:wang-f20@mails.tsinghua.edu.cn) or [Zilong Chen](mailto:chenzl22@mails.tsinghua.edu.cn))

## Performance
| Scene  | PSNR from paper | PSNR from this repo | Rendering Speed (official) | Rendering Speed (Ours) |
| ------ | --------------- | ------------------- | -------------------------- | :--------------------: |
| Garden | 25.82(5k)       | 24.08 (7k)          | 160 FPS (avg MIPNeRF360)   |         60 FPS         |

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