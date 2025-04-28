import subprocess
import os

# Environment setup
print("Setting up environment...")

# Print Python Version
subprocess.run('python --version', shell=True, check=True)
# Print PWD
subprocess.run('pwd', shell=True, check=True)

assert os.environ.get('HOME') == '/scratch/user/ali.a', f"HOME is not set correctly: {os.environ.get('HOME')}"
assert os.environ.get('TRITON_CACHE_DIR') == '/scratch/user/ali.a/triton_cache', f"TRITON_CACHE_DIR is not set correctly: {os.environ.get('TRITON_CACHE_DIR')}"
assert os.environ.get('HF_HOME') == '/scratch/user/ali.a/hf_home', f"HF_HOME is not set correctly: {os.environ.get('HF_HOME')}"
assert os.environ.get('HF_DATASETS_CACHE') == '/scratch/user/ali.a/hf_datasets', f"HF_DATASETS_CACHE is not set correctly: {os.environ.get('HF_DATASETS_CACHE')}"
assert os.environ.get('HF_HUB_OFFLINE') == '1', f"HF_HUB_OFFLINE is not set correctly: {os.environ.get('HF_HUB_OFFLINE')}"
assert os.environ.get('HF_DEBUG') == '1', f"HF_DEBUG is not set correctly: {os.environ.get('HF_DEBUG')}"
assert os.environ.get('WANDB_DIR') == '/scratch/user/ali.a/wandb', f"WANDB_DIR is not set correctly: {os.environ.get('WANDB_DIR')}"
assert os.environ.get('WANDB_CACHE_DIR') == '/scratch/user/ali.a/wandb', f"WANDB_CACHE_DIR is not set correctly: {os.environ.get('WANDB_CACHE_DIR')}"
assert os.environ.get('WANDB_CONFIG_DIR') == '/scratch/user/ali.a/wandb', f"WANDB_CONFIG_DIR is not set correctly: {os.environ.get('WANDB_CONFIG_DIR')}"
assert os.environ.get('WANDB_SETTINGS_DIR') == '/scratch/user/ali.a/wandb', f"WANDB_SETTINGS_DIR is not set correctly: {os.environ.get('WANDB_SETTINGS_DIR')}"
assert os.environ.get('WANDB_MODE') == 'offline', f"WANDB_MODE is not set correctly: {os.environ.get('WANDB_MODE')}"
assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0', f"CUDA_VISIBLE_DEVICES is not set correctly: {os.environ.get('CUDA_VISIBLE_DEVICES')}"

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
