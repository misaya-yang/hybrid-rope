import transformers, peft, torch
from transformers import TrainingArguments, Trainer
import inspect

# 1. TrainingArguments signature
ta_sig = inspect.signature(TrainingArguments.__init__)
ta_params = list(ta_sig.parameters.keys())
print("=== TrainingArguments ===")
print("eval_strategy:", "eval_strategy" in ta_params)
print("evaluation_strategy:", "evaluation_strategy" in ta_params)
print("gradient_checkpointing_kwargs:", "gradient_checkpointing_kwargs" in ta_params)
print("bf16:", "bf16" in ta_params)
print("logging_steps:", "logging_steps" in ta_params)
print("save_strategy:", "save_strategy" in ta_params)

# 2. Check Trainer.train signature
tr_sig = inspect.signature(Trainer.train)
tr_params = list(tr_sig.parameters.keys())
print("\n=== Trainer.train ===")
print("params:", tr_params)

# 3. Check LLaMA rotary embedding
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct", local_files_only=True, trust_remote_code=True)
print("\n=== LLaMA Config ===")
print("rope_theta:", getattr(cfg, "rope_theta", "N/A"))
print("rope_scaling:", getattr(cfg, "rope_scaling", "N/A"))
print("max_position_embeddings:", getattr(cfg, "max_position_embeddings", "N/A"))

# 4. Check rotary embedding class
from transformers import AutoModelForCausalLM
import importlib
mod_name = cfg.model_type
print(f"\n=== Model type: {mod_name} ===")

# Find rotary embedding module
import transformers.models.llama.modeling_llama as llama_mod
rotary_classes = [name for name in dir(llama_mod) if 'rotar' in name.lower()]
print("Rotary classes:", rotary_classes)

# Check if inv_freq is a buffer
for cls_name in rotary_classes:
    cls = getattr(llama_mod, cls_name, None)
    if cls and hasattr(cls, '__init__'):
        src = inspect.getsource(cls.__init__)
        has_inv_freq = 'inv_freq' in src
        has_register_buffer = 'register_buffer' in src
        print(f"  {cls_name}: inv_freq={has_inv_freq}, register_buffer={has_register_buffer}")

# 5. Check PEFT LoraConfig
from peft import LoraConfig, get_peft_model
lora_sig = inspect.signature(LoraConfig.__init__)
lora_params = list(lora_sig.parameters.keys())
print("\n=== LoraConfig ===")
print("target_modules:", "target_modules" in lora_params)
print("task_type:", "task_type" in lora_params)
print("modules_to_save:", "modules_to_save" in lora_params)
