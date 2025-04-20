from transformers import AutoModel, AutoTokenizer

# Set the model name
model_name = "liuhaotian/llava-v1.6-vicuna-7b"

# Download and cache the model
print("Downloading model...")
AutoModel.from_pretrained(model_name)

# Download and cache the tokenizer
print("Downloading tokenizer...")
AutoTokenizer.from_pretrained(model_name)

print("Download complete!")
