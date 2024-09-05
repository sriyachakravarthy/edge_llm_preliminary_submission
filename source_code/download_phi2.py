import torch
from transformers import AutoModelForCausalLM

# Load the model
model_name = 'microsoft/phi-2'
model = AutoModelForCausalLM.from_pretrained(model_name)

# Convert model to FP16 manually
def convert_to_fp16(model):
    for param in model.parameters():
        param.data = param.data.half()
    return model

# Convert the model to FP16
#model = convert_to_fp16(mod
save_dir_32='./phi-2-fp32'
model.save_pretrained(save_dir_32)
# Save the model locally
save_dir = './phi-2-fp16'
model_fp16 = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model_fp16.save_pretrained(save_dir)

# Verify FP16 Conversion
def check_fp16(model):
    for name, param in model.named_parameters():
        if param.dtype != torch.float16:
            print(f"{name}: {param.dtype}")
        else:
            print(f"{name}: correctly in FP16")


print(f"Model saved to {save_dir} in FP16 format.")
