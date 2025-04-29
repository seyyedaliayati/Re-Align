from config import *
import os
import subprocess

question_file = './playground/data/eval/pope/llava_pope_test.jsonl'
image_folder = './playground/data/eval/pope/val2014'
answers_dir = './playground/data/eval/pope/answers'
annotation_dir = './playground/data/eval/pope/coco'

for model_name, model_path in MODELS.items():
    print(f"Running Pope for model: {model_name}...")
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

    # Run evaluation
    cmd = [
        "python", "llava/eval/eval_pope.py",
        "--annotation-dir", annotation_dir,
        "--question-file", question_file,
        "--result-file", answers_file
    ]
    print("Running evaluation command:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    print(f"Finished Pope for model: {model_name}.\n")

print("Pope evaluation complete for all models.")
