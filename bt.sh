cd gs
python setup.py clean
pip install -e .
cd ..
CUDA_LAUNCH_BLOCKING=1 python test/test.py $1
