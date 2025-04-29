from config import *
import os
import subprocess

question_file = './playground/data/eval/mm-vet/llava-mm-vet.jsonl'
image_folder = './playground/data/eval/mm-vet/images'
answers_dir = './playground/data/eval/mm-vet/answers'
results_dir = './playground/data/eval/mm-vet/results'
# --> Evaluate the predictions in results_dir using the official jupyter notebook.

# Ensure the results directory exists
os.makedirs(results_dir, exist_ok=True)

for model_name, model_path in MODELS.items():
    print(f"Running MM-Vet for model: {model_name}...")
    answers_file = os.path.join(answers_dir, f"{model_name}.jsonl")
    result_file = os.path.join(results_dir, f"{model_name}.json")

    # Run inference
    cmd = [
        "python", "-m", "llava.eval.model_vqa",
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

    # Convert answers for evaluation
    cmd = [
        "python", "scripts/convert_mmvet_for_eval.py",
        "--src", answers_file,
        "--dst", result_file
    ]
    subprocess.run(cmd, check=True)
    print(f"TODO: Upload {result_file} to the evaluation server.")
    print(f"Finished MM-Vet for model: {model_name}.\n")

print("MM-Vet evaluation complete for all models.")
