from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load tokenizer and add special token if missing
tokenizer = AutoTokenizer.from_pretrained("sarvamai/sarvam-1", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained("sarvamai/sarvam-1", trust_remote_code=True)

# Resize the model embeddings to match the tokenizer’s vocabulary size
base_model.resize_token_embeddings(len(tokenizer))

# Load the fine-tuned adapter
model = PeftModel.from_pretrained(base_model, "./final_model")

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define your prompt
prompt = "Write a short Indian Folklore with moral"

# Tokenize the prompt
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# Generate text with specified parameters
generated_ids = model.generate(
    input_ids,
    max_new_tokens=1000,   # Adjust this value to control the length of the output
    no_repeat_ngram_size=2,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    eos_token_id=tokenizer.eos_token_id
)

# Decode the generated tokens
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print("Generated Story:\n")
print(generated_text)
