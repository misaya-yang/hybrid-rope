import os
import torch
import argparse
from datasets import load_from_disk
from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments

def get_hybrid_inv_freq(dim, base=10000, alpha=20, j0_ratio=0.93):
    """
    Computes the Anchored Hybrid (GS-RoPE) inv_freq.
    Formula: theta_i = b^{-2i/d} [1 + (alpha-1) * sigmoid(k * (i - x0))]
    """
    k = 16.05 / dim
    i = torch.arange(0, dim // 2, dtype=torch.float32)
    x0 = j0_ratio * (dim // 2)
    
    # Standard Geometric frequencies
    freqs = 1.0 / (base ** (2 * i / dim))
    
    # Hybrid Modification
    scale = 1.0 + (alpha - 1.0) * torch.sigmoid(k * (i - x0))
    hybrid_freqs = freqs * scale
    return hybrid_freqs

def apply_custom_rope(model, rope_type, dim, base=10000):
    if rope_type == "standard":
        print("Using Standard Geometric RoPE.")
        return model
        
    elif rope_type == "hybrid":
        print("Applying Anchored Hybrid RoPE.")
        hybrid_freqs = get_hybrid_inv_freq(dim, base=base)
        
        for name, module in model.named_modules():
            if "rotary_emb" in name and hasattr(module, 'inv_freq'):
                # Override the register_buffer for inv_freq
                module.inv_freq.copy_(hybrid_freqs)
        return model
    else:
        raise ValueError(f"Unknown rope_type {rope_type}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the synthetic HF dataset")
    parser.add_argument("--rope_type", type=str, choices=["standard", "hybrid"], required=True)
    parser.add_argument("--output_dir", type=str, default="./results/phase_transition")
    parser.add_argument("--max_steps", type=int, default=200)
    args = parser.parse_args()
    
    print(f"Loading dataset from {args.data_path}")
    dataset = load_from_disk(args.data_path)
    
    # Trim for fast local testing
    if len(dataset) > 500:
        dataset = dataset.select(range(500))

    # Initialize a tiny Llama model (~30M) to fit in 12GB VRAM
    config = LlamaConfig(
        vocab_size=50257, # GPT2 vocab size
        hidden_size=512,
        intermediate_size=1376,
        num_hidden_layers=6,
        num_attention_heads=8,
        num_key_value_heads=8,
        max_position_embeddings=4096,
        attention_bias=False
    )
    
    model = LlamaForCausalLM(config)
    print(f"Model params: {model.num_parameters() / 1e6:.2f} M")
    
    # Apply RoPE injection
    dim = config.hidden_size // config.num_attention_heads
    model = apply_custom_rope(model, args.rope_type, dim, base=10000.0)
    
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, args.rope_type),
        max_steps=args.max_steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-4,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=200,
        fp16=True, # Use FP16 for 4070 Super
        report_to="none"
    )

    # Basic collator (dataset already has input_ids)
    def collate_fn(batch):
        input_ids = torch.tensor([ex["input_ids"] for ex in batch], dtype=torch.long)
        return {"input_ids": input_ids, "labels": input_ids.clone()}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
    )

    print(f"Starting training for {args.rope_type} on {os.path.basename(args.data_path)}...")
    trainer.train()
    
    # Extract training loss history
    history = trainer.state.log_history
    if len(history) > 0:
        # get the last valid loss
        losses = [log.get("loss", 0.0) for log in history if "loss" in log]
        final_loss = losses[-1] if len(losses) > 0 else "Unknown"
        print(f"==== RESULT: [{args.rope_type}] Final Loss on {os.path.basename(args.data_path)}: {final_loss} ====")

if __name__ == "__main__":
    main()
