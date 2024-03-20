import yaml

# Your YAML string
yaml_string = """
base_model: NousResearch/Llama-2-7b-hf
model_type: LlamaForCausalLM
tokenizer_type: LlamaTokenizer

load_in_8bit: false
load_in_4bit: true
strict: false

datasets:
  - path: mhenrichsen/alpaca_2k_test
    type: alpaca
dataset_prepared_path:
val_set_size: 0.05
output_dir: ./qlora-out

adapter: qlora
lora_model_dir:

sequence_len: 4096
sample_packing: true
pad_to_sequence_len: true

lora_r: 32
lora_alpha: 16
lora_dropout: 0.05
lora_target_modules:
lora_target_linear: true
lora_fan_in_fan_out:

wandb_project:
wandb_entity:
wandb_watch:
wandb_name:
wandb_log_model:

gradient_accumulation_steps: 4
micro_batch_size: 2
num_epochs: 4
optimizer: paged_adamw_32bit
lr_scheduler: cosine
learning_rate: 0.0002

train_on_inputs: false
group_by_length: false
bf16: auto
fp16:
tf32: false

gradient_checkpointing: true
early_stopping_patience:
resume_from_checkpoint:
local_rank:
logging_steps: 1
xformers_attention:
flash_attention: true

warmup_steps: 10
evals_per_epoch: 4
eval_table_size:
saves_per_epoch: 1
debug:
deepspeed:
weight_decay: 0.0
fsdp:
fsdp_config:
special_tokens:
"""

# Method to read and parse the YAML string
def read_and_parse_yaml(yaml_str):
    return yaml.safe_load(yaml_str)

# Method to update hyperparameters
def update_hyperparams(config, new_hyperparams):
    config.update(new_hyperparams)
    return config

# Method to make the final YAML string
def make_final_yaml_string(config):
    return yaml.dump(config, sort_keys=False)

# Read and parse the YAML string
config = read_and_parse_yaml(yaml_string)

# New hyperparameter values (example)
new_hyperparams = {
    'sequence_len': 2048,
    'lora_r': 64,
    'lora_alpha': 32,
    'warmup_steps': 20
}

# Update the hyperparameters
updated_config = update_hyperparams(config, new_hyperparams)

# Make the final YAML string
final_yaml_string = make_final_yaml_string(updated_config)

# Output the final YAML string
print(final_yaml_string)