from config import *
import os
import subprocess

question_file = './playground/data/eval/llava-bench-in-the-wild/questions.jsonl'
image_folder = './playground/data/eval/llava-bench-in-the-wild/images'
answers_dir = './playground/data/eval/llava-bench-in-the-wild/answers'

for model_name, model_path in MODELS.items():
    print(f"Running Llava Bench-in-the-Wild for model: {model_name}...")
    answers_file = os.path.join(answers_dir, f"{model_name}.jsonl")

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
    print(f"TODO: Run OpenAI on {answers_file}.")
    print(f"Finished Llava Bench-in-the-Wild for model: {model_name}.\n")

print("Llava Bench-in-the-Wild evaluation complete for all models.")
