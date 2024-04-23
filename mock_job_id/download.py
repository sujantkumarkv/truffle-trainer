from transformers import AutoTokenizer, AutoModelForCausalLM, PhiForCausalLM

# model = PhiForCausalLM.from_pretrained("microsoft/phi-1_5", torch_dtype="auto", trust_remote_code=True, cache_dir="./model/")
# tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True, cache_dir="./tokenizer/")

# model = PhiForCausalLM.from_pretrained("./model/models--microsoft--phi-1_5/snapshots/474b29ef617e857673769e3266cf9a4385ab5737", local_files_only=True)
# tokenizer = AutoTokenizer.from_pretrained("./tokenizer/models--microsoft--phi-1_5/snapshots/474b29ef617e857673769e3266cf9a4385ab5737", local_files_only=True)

# text = "who is krishna? Answer:"

# Specify the directory where you want to save/load the model and tokenizer
model_dir = "./model/phi-1_5"
tokenizer_dir = "./tokenizer/phi-1_5"

# Download and save the model and tokenizer
# model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5")
# tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")

# model.save_pretrained(model_dir)
# tokenizer.save_pretrained(tokenizer_dir)

# Load the model and tokenizer from the specified directory
model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)

# # Example usage
text = "who is krishna? Answer:"
tokens = tokenizer(text, return_tensors="pt")
output = model.generate(**tokens)
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

print(decoded_output)