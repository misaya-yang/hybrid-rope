import os
import sys
import json
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from datasets import load_dataset
import time
import gc

# Enable HF mirror
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"

# Results directory
RESULTS_DIR = "/opt/dfrope/results/350m_hybrid_final"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Setup logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(f"{RESULTS_DIR}/training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 350M Model config (1024 hidden, 24 layers, 16 heads)
MODEL_CONFIG = {
    "vocab_size": 50257,
    "n_positions": 32768,
    "n_embd": 1024,
    "n_layer": 24,
    "n_head": 16,
}

# Training config
BATCH_SIZE = 8
MAX_SEQ_LENGTH = 2048
TOTAL_TOKENS = 500_000_000
LEARNING_RATE = 1e-4
WARMUP_STEPS = 1000
EVAL_LENGTHS = [2048, 4096, 8192, 12288, 16384]
EVAL_INTERVAL = 5000
SAVE_INTERVAL = 10000

# Hybrid frequency config
ALPHA = 0.2  # Weight for anchpoly
GEO_FREQ = 10000
ANCHPOLY_P = 3.9
ANCHPOLY_OMF = 0.3

class HybridRoPE(nn.Module):
    """Hybrid RoPE: (1-alpha) * Geo + alpha * AnchPoly"""
    def __init__(self, head_dim, max_seq_len=32768, alpha=0.2, geo_freq=10000, 
                 anchpoly_p=3.9, anchpoly_omf=0.3):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.alpha = alpha
        self.geo_freq = geo_freq
        self.anchpoly_p = anchpoly_p
        self.anchpoly_omf = anchpoly_omf
        
        # Geometric inv_freq (standard)
        inv_freq_geo = 1.0 / (geo_freq ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq_geo", inv_freq_geo)
        
    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).float()
        
        # Geometric RoPE embeddings
        freqs_geo = torch.outer(t, self.inv_freq_geo)
        emb_geo = torch.cat([freqs_geo, freqs_geo], dim=-1)  # [seq_len, head_dim]
        
        # AnchPoly RoPE - compute base frequencies
        d_anch = int(self.head_dim * (1 - self.anchpoly_omf))
        if d_anch % 2 == 1:
            d_anch -= 1
        if d_anch < 2:
            d_anch = 2
            
        # Create frequency grid for AnchPoly
        freq_indices = torch.arange(0, d_anch, 2, device=device).float()
        inv_freq_anch = 1.0 / (torch.arange(1, seq_len + 1, device=device).float().unsqueeze(1) ** 
                               (freq_indices / d_anch))
        
        # Compute AnchPoly embeddings
        emb_anch = torch.cat([inv_freq_anch, inv_freq_anch], dim=-1)  # [seq_len, d_anch]
        
        # Pad AnchPoly to match head_dim
        if d_anch < self.head_dim:
            pad_size = self.head_dim - d_anch
            emb_anch = torch.cat([emb_anch, torch.zeros(seq_len, pad_size, device=device)], dim=-1)
        elif d_anch > self.head_dim:
            emb_anch = emb_anch[:, :self.head_dim]
        
        # Hybrid combination
        cos = (1 - self.alpha) * torch.cos(emb_geo) + self.alpha * torch.cos(emb_anch)
        sin = (1 - self.alpha) * torch.sin(emb_geo) + self.alpha * torch.sin(emb_anch)
        return cos, sin

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class HybridGPT2Attention(nn.Module):
    def __init__(self, config, rope):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.rope = rope
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
    def forward(self, hidden_states, attention_mask=None):
        bsz, seq_len, _ = hidden_states.shape
        qkv = self.c_attn(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rope(seq_len, hidden_states.device)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, is_causal=True
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.c_proj(attn_output)

class HybridGPT2Block(nn.Module):
    def __init__(self, config, rope):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = HybridGPT2Attention(config, rope)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
        )
        
    def forward(self, hidden_states, attention_mask=None):
        hidden_states = hidden_states + self.attn(self.ln_1(hidden_states), attention_mask)
        hidden_states = hidden_states + self.mlp(self.ln_2(hidden_states))
        return hidden_states

class HybridGPT2Model(nn.Module):
    def __init__(self, config, alpha=0.2, geo_freq=10000, anchpoly_p=3.9, anchpoly_omf=0.3):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        
        head_dim = config.n_embd // config.n_head
        self.rope = HybridRoPE(
            head_dim=head_dim,
            max_seq_len=config.n_positions,
            alpha=alpha,
            geo_freq=geo_freq,
            anchpoly_p=anchpoly_p,
            anchpoly_omf=anchpoly_omf
        )
        
        self.h = nn.ModuleList([HybridGPT2Block(config, self.rope) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
    def forward(self, input_ids, labels=None):
        hidden_states = self.wte(input_ids)
        
        for block in self.h:
            hidden_states = block(hidden_states)
            
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

def load_tinystories_tokens():
    logger.info("Loading TinyStories dataset...")
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return ds, tokenizer

class TokenIterator:
    def __init__(self, dataset, tokenizer, seq_length=2048):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.buffer = []
        
    def __iter__(self):
        for item in self.dataset:
            text = item["text"]
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            self.buffer.extend(tokens)
            
            while len(self.buffer) >= self.seq_length + 1:
                yield torch.tensor(self.buffer[:self.seq_length + 1])
                self.buffer = self.buffer[self.seq_length:]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def evaluate_at_length(model, tokenizer, length, num_batches=10, device="cuda"):
    model.eval()
    ds = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
    total_loss = 0
    count = 0
    
    iterator = TokenIterator(ds, tokenizer, seq_length=length)
    
    with torch.no_grad():
        for batch in iterator:
            if count >= num_batches:
                break
            batch = batch.unsqueeze(0).to(device)
            labels = batch.clone()
            
            outputs = model(batch, labels=labels)
            total_loss += outputs["loss"].item()
            count += 1
            
            if count % 5 == 0:
                torch.cuda.empty_cache()
    
    model.train()
    return math.exp(total_loss / count) if count > 0 else float("inf")

def main():
    logger.info("=" * 60)
    logger.info("FINAL 350M Hybrid Training (Alpha=0.2)")
    logger.info("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create model
    config = GPT2Config(**MODEL_CONFIG)
    model = HybridGPT2Model(
        config=config,
        alpha=ALPHA,
        geo_freq=GEO_FREQ,
        anchpoly_p=ANCHPOLY_P,
        anchpoly_omf=ANCHPOLY_OMF
    )
    
    param_count = count_parameters(model)
    logger.info(f"Model parameters: {param_count / 1e6:.1f}M")
    logger.info(f"Alpha (AnchPoly weight): {ALPHA}")
    logger.info(f"Geometric freq: {GEO_FREQ}")
    logger.info(f"AnchPoly p: {ANCHPOLY_P}, omf: {ANCHPOLY_OMF}")
    
    model = model.to(device)
    
    # Load dataset
    ds, tokenizer = load_tinystories_tokens()
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    steps_per_epoch = TOTAL_TOKENS // (BATCH_SIZE * MAX_SEQ_LENGTH)
    logger.info(f"Training for {steps_per_epoch} steps ({TOTAL_TOKENS/1e6:.0f}M tokens)")
    logger.info(f"Batch size: {BATCH_SIZE}, Seq length: {MAX_SEQ_LENGTH}")
    
    iterator = TokenIterator(ds, tokenizer, seq_length=MAX_SEQ_LENGTH)
    step = 0
    total_tokens = 0
    epoch_loss = 0
    
    start_time = time.time()
    batch_buffer = []
    
    for tokens in iterator:
        batch_buffer.append(tokens)
        
        if len(batch_buffer) < BATCH_SIZE:
            continue
            
        batch = torch.stack(batch_buffer).to(device)
        batch_buffer = []
        
        labels = batch.clone()
        
        optimizer.zero_grad()
        outputs = model(batch, labels=labels)
        loss = outputs["loss"]
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        step += 1
        tokens_processed = BATCH_SIZE * MAX_SEQ_LENGTH
        total_tokens += tokens_processed
        epoch_loss += loss.item()
        
        if step % 100 == 0:
            elapsed = time.time() - start_time
            tps = total_tokens / elapsed
            avg_loss = epoch_loss / 100
            ppl = math.exp(avg_loss)
            logger.info(f"Step {step} | Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | TPS: {tps:.0f}")
            epoch_loss = 0
            
        if step % EVAL_INTERVAL == 0:
            logger.info(f"Evaluation at step {step}")
            results = {}
            for eval_len in EVAL_LENGTHS:
                logger.info(f"  Evaluating at length {eval_len}...")
                ppl = evaluate_at_length(model, tokenizer, eval_len, num_batches=5, device=device)
                results[f"length_{eval_len}"] = ppl
                logger.info(f"    Length {eval_len}: PPL = {ppl:.2f}")
                torch.cuda.empty_cache()
            
            # Save results
            with open(f"{RESULTS_DIR}/eval_step_{step}.json", "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Evaluation complete")
            
        if step % SAVE_INTERVAL == 0:
            checkpoint_path = f"{RESULTS_DIR}/checkpoint_step_{step}.pt"
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            
        if total_tokens >= TOTAL_TOKENS:
            break
    
    # Final evaluation
    logger.info("" + "=" * 60)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 60)
    final_results = {}
    for eval_len in EVAL_LENGTHS:
        ppl = evaluate_at_length(model, tokenizer, eval_len, num_batches=10, device=device)
        final_results[f"length_{eval_len}"] = ppl
        logger.info(f"Length {eval_len}: PPL = {ppl:.2f}")
    
    with open(f"{RESULTS_DIR}/final_results.json", "w") as f:
        json.dump(final_results, f, indent=2)
    
    # Save final model
    torch.save(model.state_dict(), f"{RESULTS_DIR}/final_model.pt")
    logger.info(f"Training complete! Results saved to {RESULTS_DIR}")
    logger.info(f"Total steps: {step}")
    logger.info(f"Total tokens: {total_tokens / 1e6:.1f}M")

if __name__ == "__main__":
    main()
