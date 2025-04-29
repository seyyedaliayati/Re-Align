from config import *
import os
import subprocess

split = "mmbench_dev_20230712"

question_file = f'./playground/data/eval/mmbench/{split}.tsv'
answers_dir = f'./playground/data/eval/mmbench/answers/{split}'
upload_dir = f'./playground/data/eval/mmbench/answers_upload/{split}'

# Ensure the upload directory exists
os.makedirs(upload_dir, exist_ok=True)

for model_name, model_path in MODELS.items():
    print(f"Running MMBench for model: {model_name}...")

    answers_file = os.path.join(answers_dir, f"{model_name}.jsonl")

    # Run inference
    cmd = [
        "python", "-m", "llava.eval.model_vqa_mmbench",
        "--model-path", model_path,
        "--question-file", question_file,
        "--answers-file", answers_file,
        "--single-pred-prompt",
        "--temperature", "0",
        "--conv-mode", "vicuna_v1"
    ]
    if model_name != "base":
        cmd += ["--model-base", BASE_MODEL]
    print("Running command:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # Convert answers for submission
    cmd = [
        "python", "scripts/convert_mmbench_for_submission.py",
        "--annotation-file", question_file,
        "--result-dir", answers_dir,
        "--upload-dir", upload_dir,
        "--experiment", model_name
    ]
    subprocess.run(cmd, check=True)
    print(f"TODO: Upload {upload_dir} to the evaluation server.")
    print(f"Finished MMBench for model: {model_name}.\n")

print("MMBench evaluation complete for all models.")
