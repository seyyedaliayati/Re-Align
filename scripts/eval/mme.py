from config import *
import os
import subprocess

question_file = './playground/data/eval/MME/llava_mme.jsonl'
image_folder = './playground/data/eval/MME/MME_Benchmark_release_version'
answers_dir = './playground/data/eval/MME/answers'
mme_dir = './playground/data/eval/MME'

for model_name, model_path in MODELS.items():
    print(f"Running MME for model: {model_name}...")
    answers_file = os.path.join(answers_dir, f"{model_name}.jsonl")

    # Run inference
    cmd = [
        "python", "-m", "llava.eval.model_vqa_loader",
        "--model-path", model_path,
        "--question-file", question_file,
        "--image-folder", image_folder,
        "--answers-file", answers_file,
        "--temperature", "0",
        "--conv-mode", "vicuna_v1"
    ]
    if model_name != "base":
        cmd += ["--model-base", BASE_MODEL]
    print("Running command:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # Convert answers for MME
    cmd = [
        "python", "convert_answer_to_mme.py",
        "--experiment", model_name
    ]
    print("Running conversion command:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=mme_dir)

    # Run evaluation tool
    cmd = [
        "python", "calculation.py",
        "--results_dir", os.path.join(answers_dir, model_name)
    ]
    print("Running evaluation command:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=os.path.join(mme_dir, "eval_tool"))

    print(f"Finished MME for model: {model_name}.\n")

print("MME evaluation complete for all models.")
