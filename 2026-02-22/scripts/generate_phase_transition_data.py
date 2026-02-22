import argparse
import random
import numpy as np
import os
import datasets
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

def generate_shuffled_dataset(
    source_dataset_name: str,
    tokenizer_path: str,
    output_dir: str,
    num_samples: int = 10000,
    seq_length: int = 4096,
    gamma_prior: float = 1.0,
    seed: int = 42
):
    """
    Generates synthetic datasets to simulate different attention distance priors γ.
    
    Args:
        source_dataset_name: HF dataset name (e.g., 'wikitext', 'wikitext-103-v1').
        tokenizer_path: Path/name to the tokenizer (e.g., 'meta-llama/Llama-2-7b-hf').
        output_dir: Where to save the generated dataset.
        num_samples: Number of sequences to generate.
        seq_length: Target sequence length L.
        gamma_prior: Controls the snippet length distribution. 
                     γ -> 1.0 (Power-law, Continuous text)
                     γ -> 0.0 (Uniform, highly fragmented snippets)
    """
    print(f"Initializing data generation with target γ ≈ {gamma_prior}")
    random.seed(seed)
    np.random.seed(seed)
    
    # 1. Load Tokenizer
    print(f"Loading tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 2. Load Source Dataset (streaming to handle large corpora)
    print(f"Loading source dataset: {source_dataset_name}")
    try:
        if source_dataset_name == "wikitext":
            ds = load_dataset(source_dataset_name, 'wikitext-103-v1', split='train', streaming=True)
        else:
            ds = load_dataset(source_dataset_name, split='train', streaming=True)
    except Exception as e:
        print(f"Failed to load as streaming, falling back to full load: {e}")
        if source_dataset_name == "wikitext":
            ds = load_dataset(source_dataset_name, 'wikitext-103-v1', split='train')
        else:
            ds = load_dataset(source_dataset_name, split='train')

    # Convert full corpus to a long token stream
    def token_stream_generator():
        for example in ds:
            text = example.get('text', example.get('content', ''))
            if not text.strip(): continue
            tokens = tokenizer.encode(text, add_special_tokens=False)
            yield from tokens

    stream = token_stream_generator()
    
    # 3. Define Snippet Length Distribution based on γ
    # For γ ≈ 1 (Power-law/Natural): We want long contiguous chunks.
    # For γ ≈ 0 (Uniform/Random): We want many tiny chunks spliced together.
    
    def get_next_snippet_length(gamma):
        if gamma >= 0.9:
            # Full sequence length (pure continuous)
            return seq_length
        elif gamma <= 0.1:
            # Extreme fragmentation (snippets of 10-50 tokens)
            return random.randint(10, 50)
        else:
            # Intermediate mixing 
            # Mean length scales with gamma
            mean_len = int(seq_length * gamma)
            # Ensure it's between a small fragment and the full seq
            return min(seq_length, max(20, int(np.random.exponential(scale=mean_len))))

    print("Generating sequences...")
    generated_sequences = []
    
    # 4. Assembly Loop
    for i in range(num_samples):
        if i % 1000 == 0:
            print(f"Generated {i}/{num_samples} sequences...")
            
        current_seq = []
        while len(current_seq) < seq_length:
            target_chunk_len = get_next_snippet_length(gamma_prior)
            tokens_needed = seq_length - len(current_seq)
            chunk_len = min(target_chunk_len, tokens_needed)
            
            chunk = []
            try:
                for _ in range(chunk_len):
                    chunk.append(next(stream))
            except StopIteration:
                print("Source stream exhausted. Re-initializing...")
                stream = token_stream_generator()
                for _ in range(chunk_len - len(chunk)):
                    chunk.append(next(stream))
                    
            # If simulating low gamma, we optionally inject 'noise' or separator to break context
            if gamma_prior < 0.5 and len(current_seq) > 0:
                 # Inject EOS to destroy syntax continuation formally
                 current_seq.append(tokenizer.eos_token_id)
                 chunk = chunk[:-1] if len(chunk) > 0 else chunk
                 
            current_seq.extend(chunk)
            
        generated_sequences.append(current_seq)

    # 5. Save Artifact
    print("Formatting and saving to HuggingFace dataset structure...")
    out_ds = Dataset.from_dict({
        "input_ids": generated_sequences,
    })
    
    # Save to disk
    save_path = os.path.join(output_dir, f"gamma_{gamma_prior:.2f}_seq{seq_length}")
    os.makedirs(save_path, exist_ok=True)
    out_ds.save_to_disk(save_path)
    print(f"Successfully saved dataset to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic phase-transition datasets.")
    parser.add_argument("--dataset", type=str, default="wikitext", help="Source dataset")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer name/path")
    parser.add_argument("--out_dir", type=str, default="./data/phase_transition", help="Output directory")
    parser.add_argument("--samples", type=int, default=5000, help="Number of sequences")
    parser.add_argument("--seq_len", type=int, default=4096, help="Target sequence length L")
    parser.add_argument("--gamma", type=float, default=1.0, help="Attention prior gamma (1.0=continuous, 0.0=fragmented)")
    
    args = parser.parse_args()
    
    # Example generation for the two extremes
    print("\n--- Generating Continuous Baseline (γ ≈ 1.0) ---")
    generate_shuffled_dataset(
        source_dataset_name=args.dataset,
        tokenizer_path=args.tokenizer,
        output_dir=args.out_dir,
        num_samples=args.samples,
        seq_length=args.seq_len,
        gamma_prior=1.0
    )
    
    print("\n--- Generating Uniform Fragmented Baseline (γ ≈ 0.0) ---")
    generate_shuffled_dataset(
        source_dataset_name=args.dataset,
        tokenizer_path=args.tokenizer,
        output_dir=args.out_dir,
        num_samples=args.samples,
        seq_length=args.seq_len,
        gamma_prior=0.0
    )
