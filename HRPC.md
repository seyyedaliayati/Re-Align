If you are using the HRPC cluster, please follow these instructions to set up your environment.
```bash
conda create -n re-align python=3.10 -y
conda activate re-align
pip install --upgrade pip  
pip install -e .
module load CUDA/12.2.0
pip install -e ".[train]"

cd ~
git clone git@github.com:Dao-AILab/flash-attention.git
cd flash-attention
git checkout v2.3.6
mkdir -p tmp
TMPDIR=$PWD/tmp pip install . --no-build-isolation
pip install trl
pip install protobuf==3.20.*
```


Download the models:
```bash
huggingface-cli login
huggingface-cli download liuhaotian/llava-v1.6-vicuna-7b 
huggingface-cli download liuhaotian/llava-v1.6-vicuna-7b --local-dir $SCRATCH/hf_cache/liuhaotian/llava-v1.6-vicuna-7b --local-dir-use-symlinks False
```
