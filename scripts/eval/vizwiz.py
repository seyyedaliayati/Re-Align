from config import *
import os
import subprocess

question_file = './playground/data/eval/vizwiz/llava_test.jsonl'
image_folder = './playground/data/eval/vizwiz/test'
answers_dir = './playground/data/eval/vizwiz/answers'
upload_dir = './playground/data/eval/vizwiz/answers_upload'
# --> Submit the results to the evaluation server: upload_dir

# Ensure the upload directory exists
os.makedirs(upload_dir, exist_ok=True)

for model_name, model_path in MODELS.items():
    print(f"Running VizWiz for model: {model_name}...")
    answers_file = os.path.join(answers_dir, f"{model_name}.jsonl")
    upload_file = os.path.join(upload_dir, f"{model_name}.json")

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

    # Convert answers for submission
    cmd = [
        "python", "scripts/convert_vizwiz_for_submission.py",
        "--annotation-file", question_file,
        "--result-file", answers_file,
        "--result-upload-file", upload_file
    ]
    subprocess.run(cmd, check=True)

    print(f"Finished VizWiz for model: {model_name}.\n")

print("VizWiz evaluation complete for all models.")
