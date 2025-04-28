import subprocess
import os
from config import *

# TextVQA paths
question_file = './playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl'
image_folder = './playground/data/eval/textvqa/train_images'
answers_dir = './playground/data/eval/textvqa/answers'

# Loop through models and run the corresponding commands for TextVQA
for model_name, model_path in MODELS.items():
    print(f"Running TextVQA for model: {model_name}...")
    answers_file = os.path.join(answers_dir, f"{model_name}.jsonl")
    
    # Command for the model evaluation
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

    # Command for evaluation after model prediction
    eval_cmd = [
        "python", "-m", "llava.eval.eval_textvqa",
        "--annotation-file", "./playground/data/eval/textvqa/TextVQA_0.5.1_val.json",
        "--result-file", answers_file
    ]
    
    subprocess.run(eval_cmd, check=True)

    print(f"Finished TextVQA for model: {model_name}.\n")

print("TextVQA evaluation complete for all models.")
