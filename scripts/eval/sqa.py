from config import *

question_file = './playground/data/eval/scienceqa/llava_test_CQM-A.json'
image_folder = './playground/data/eval/scienceqa/images/test'
answers_dir = './playground/data/eval/scienceqa/answers'

"""
python -m llava.eval.model_vqa_science \
    --model-path liuhaotian/llava-v1.6-vicuna-7b \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/base.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/base.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/base_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/base_result.json

"""

for model_name, model_path in MODELS.items():
    print(f"Running ScienceQA for model: {model_name}...")
    answers_file = os.path.join(answers_dir, f"{model_name}.jsonl")
    output_file = os.path.join(answers_dir, f"{model_name}_output.jsonl")
    result_file = os.path.join(answers_dir, f"{model_name}_result.json")
    
    cmd = [
        "python", "-m", "llava.eval.model_vqa_science",
        "--model-path", model_path,
        "--question-file", question_file,
        "--image-folder", image_folder,
        "--answers-file", answers_file,
        "--single-pred-prompt",
        "--temperature", "0",
        "--conv-mode", "vicuna_v1"
    ]
    
    if model_name != "base":
        cmd += ["--model-base", BASE_MODEL]
    print("Running command:", " ".join(cmd))
    
    subprocess.run(cmd, check=True)
    
    cmd = [
        'python', 'llava/eval/eval_science_qa.py',
        '--base-dir', './playground/data/eval/scienceqa',
        '--result-file', answers_file,
        '--output-file', output_file,
        '--output-result', result_file
    ]
    subprocess.run(cmd, check=True)
    
    print(f"Finished ScienceQA for model: {model_name}.\n")

print("ScienceQA evaluation complete for all models.")
