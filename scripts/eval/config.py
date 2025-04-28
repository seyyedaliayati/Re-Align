import subprocess
import os

# Environment setup
print("Setting up environment...")

# HPRC environment setup
subprocess.run('ml purge', shell=True, check=True)
subprocess.run('ml Anaconda3/2021.11', shell=True, check=True)
os.chdir('$SCRATCH/Re-Align')

# Print Python Version
subprocess.run('python --version', shell=True, check=True)
# Print PWD
subprocess.run('pwd', shell=True, check=True)

# Exporting environment variables
os.environ['HOME'] = '/scratch/user/ali.a'
os.environ['TRITON_CACHE_DIR'] = '/scratch/user/ali.a/triton_cache'
os.environ['HF_HOME'] = '/scratch/user/ali.a/hf_home'
os.environ['HF_DATASETS_CACHE'] = '/scratch/user/ali.a/hf_datasets'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DEBUG'] = '1'

os.environ['WANDB_DIR'] = '/scratch/user/ali.a/wandb'
os.environ['WANDB_CACHE_DIR'] = '/scratch/user/ali.a/wandb'
os.environ['WANDB_CONFIG_DIR'] = '/scratch/user/ali.a/wandb'
os.environ['WANDB_SETTINGS_DIR'] = '/scratch/user/ali.a/wandb'
os.environ['WANDB_MODE'] = 'offline'

# GPUs setup
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

print("Environment setup complete.")

MODELS_DIR = "./output"
DPO_NO_SFT = "llava-vicuna-7b-rdpo-lora-1e-6-beta-0.1-nosft"
DPO_SFT = "llava-vicuna-7b-rdpo-lora-1e-6-beta-0.1-sft"
SIMPO_NO_SFT = "" # TODO
SIMPO_SFT = "llava-vicuna-7b-rsimpo-lora-1e-6-beta-0.1-sft"

BASE_MODEL = "liuhaotian/llava-v1.6-vicuna-7b"

MODELS = {
    "base": BASE_MODEL,
    "dpo_nosft": os.path.join(MODELS_DIR, DPO_NO_SFT),
    "dpo_sft": os.path.join(MODELS_DIR, DPO_SFT),
    # "simpo_nosft": os.path.join(MODELS_DIR, SIMPO_NO_SFT),
    "simpo_sft": os.path.join(MODELS_DIR, SIMPO_SFT),
}
