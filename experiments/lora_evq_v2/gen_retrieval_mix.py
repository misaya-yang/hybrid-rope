#!/usr/bin/env python3
"""
Generate synthetic retrieval training data for EVQ-LoRA stage2.

Generates a mix of:
  1. Single-needle passkey retrieval (S-NIAH)
  2. Multi-needle retrieval (MK-NIAH)
  3. Key-value association retrieval (KV-Retr)

All samples are within 8K tokens (training length).
Output: JSONL with messages format, ready for tokenization.

Usage:
    python gen_retrieval_mix.py --output retrieval_mix.jsonl --n_samples 500
"""

import argparse
import json
import random
import string


# ---------------------------------------------------------------------------
# Filler text pool (realistic-looking padding)
# ---------------------------------------------------------------------------

FILLER_SENTENCES = [
    "The development of renewable energy sources has become a critical priority for many nations around the world.",
    "Recent advances in machine learning have enabled new applications across healthcare, finance, and autonomous systems.",
    "The study of ocean currents reveals complex patterns that influence global climate and weather systems.",
    "Urban planning in the 21st century must balance population growth with environmental sustainability.",
    "Historical analysis of trade routes shows how commerce shaped cultural exchange between civilizations.",
    "The biochemistry of protein folding remains one of the most challenging problems in molecular biology.",
    "Advances in quantum computing promise to revolutionize cryptography and materials science.",
    "The relationship between sleep quality and cognitive performance has been extensively documented.",
    "Modern telecommunications infrastructure relies on fiber optic networks spanning continents.",
    "Archaeological evidence suggests that ancient civilizations developed sophisticated mathematical systems.",
    "The economics of space exploration require careful cost-benefit analysis across multiple decades.",
    "Biodiversity conservation efforts must address habitat loss, climate change, and invasive species.",
    "The philosophy of mind explores fundamental questions about consciousness and subjective experience.",
    "Statistical methods for causal inference have transformed how we evaluate policy interventions.",
    "The engineering challenges of building earthquake-resistant structures vary significantly by region.",
    "Nutritional science continues to revise recommendations based on new longitudinal studies.",
    "The sociology of digital communities reveals new patterns of social organization and identity.",
    "Atmospheric chemistry plays a crucial role in understanding air quality and pollution dynamics.",
    "The ethics of artificial intelligence development require input from diverse stakeholders.",
    "Comparative linguistics reveals deep structural similarities across seemingly unrelated languages.",
]


def random_filler(n_words: int) -> str:
    """Generate filler text of approximately n_words."""
    words = []
    while len(words) < n_words:
        words.extend(random.choice(FILLER_SENTENCES).split())
    return " ".join(words[:n_words])


def random_passkey() -> str:
    """Generate a random 5-digit passkey."""
    return str(random.randint(10000, 99999))


def random_city() -> str:
    cities = [
        "Tokyo", "London", "Paris", "Berlin", "Sydney", "Toronto",
        "Mumbai", "Cairo", "Seoul", "Madrid", "Rome", "Vienna",
        "Prague", "Dublin", "Oslo", "Helsinki", "Warsaw", "Lisbon",
        "Bangkok", "Jakarta", "Manila", "Hanoi", "Lima", "Bogota",
    ]
    return random.choice(cities)


def random_color() -> str:
    colors = [
        "red", "blue", "green", "yellow", "purple", "orange",
        "pink", "brown", "silver", "gold", "cyan", "magenta",
    ]
    return random.choice(colors)


# ---------------------------------------------------------------------------
# Task generators
# ---------------------------------------------------------------------------

def gen_single_needle(target_words: int = 3000) -> dict:
    """S-NIAH: Find a single passkey hidden in filler text."""
    passkey = random_passkey()

    # Split filler into before/after, place needle at random depth
    depth = random.uniform(0.1, 0.9)
    before_words = int(target_words * depth)
    after_words = target_words - before_words

    before = random_filler(before_words)
    after = random_filler(after_words)
    needle = f"The special passkey is: {passkey}. Remember this number."

    context = f"{before}\n\n{needle}\n\n{after}"

    return {
        "messages": [
            {"role": "user", "content": f"{context}\n\nWhat is the special passkey mentioned in the text above? Reply with only the number."},
            {"role": "assistant", "content": passkey},
        ]
    }


def gen_multi_needle(n_needles: int = 3, target_words: int = 3500) -> dict:
    """MK-NIAH: Find multiple passkeys hidden at different positions."""
    passkeys = [random_passkey() for _ in range(n_needles)]
    labels = [f"Key-{chr(65+i)}" for i in range(n_needles)]  # Key-A, Key-B, ...

    # Distribute needles evenly across the text
    segment_words = target_words // (n_needles + 1)
    parts = []
    for i in range(n_needles):
        parts.append(random_filler(segment_words))
        parts.append(f"[{labels[i]}] The secret code is {passkeys[i]}.")
    parts.append(random_filler(segment_words))

    context = "\n\n".join(parts)
    answer = ", ".join(f"{labels[i]}: {passkeys[i]}" for i in range(n_needles))

    return {
        "messages": [
            {"role": "user", "content": f"{context}\n\nList all the secret codes found in the text with their labels. Format: Label: Number"},
            {"role": "assistant", "content": answer},
        ]
    }


def gen_kv_retrieval(n_pairs: int = 10, target_words: int = 3000) -> dict:
    """KV-Retr: Associate keys with values, then query one."""
    cities = random.sample([
        "Tokyo", "London", "Paris", "Berlin", "Sydney", "Toronto",
        "Mumbai", "Cairo", "Seoul", "Madrid", "Rome", "Vienna",
        "Prague", "Dublin", "Oslo", "Helsinki", "Warsaw", "Lisbon",
        "Bangkok", "Jakarta",
    ], n_pairs)
    values = [str(random.randint(100, 999)) for _ in range(n_pairs)]

    segment_words = target_words // (n_pairs + 1)
    parts = []
    for i in range(n_pairs):
        parts.append(random_filler(segment_words))
        parts.append(f"The identification number for {cities[i]} is {values[i]}.")
    parts.append(random_filler(segment_words))

    context = "\n\n".join(parts)

    # Query a random key
    query_idx = random.randint(0, n_pairs - 1)

    return {
        "messages": [
            {"role": "user", "content": f"{context}\n\nWhat is the identification number for {cities[query_idx]}? Reply with only the number."},
            {"role": "assistant", "content": values[query_idx]},
        ]
    }


def gen_variable_tracking(n_vars: int = 3, n_updates: int = 5, target_words: int = 3000) -> dict:
    """VT: Track variable assignments across the text."""
    var_names = [f"VAR_{chr(88+i)}" for i in range(n_vars)]  # VAR_X, VAR_Y, VAR_Z
    current_vals = {v: random.randint(10, 99) for v in var_names}

    segment_words = target_words // (n_updates + 2)
    parts = [random_filler(segment_words)]

    # Initial assignment
    init_lines = [f"{v} = {current_vals[v]}" for v in var_names]
    parts.append("Initial values: " + ", ".join(init_lines) + ".")

    # Random updates
    for _ in range(n_updates):
        parts.append(random_filler(segment_words))
        v = random.choice(var_names)
        new_val = random.randint(10, 99)
        current_vals[v] = new_val
        parts.append(f"Update: {v} is now set to {new_val}.")

    parts.append(random_filler(segment_words))
    context = "\n\n".join(parts)

    answer = ", ".join(f"{v} = {current_vals[v]}" for v in var_names)

    return {
        "messages": [
            {"role": "user", "content": f"{context}\n\nWhat are the final values of all variables? Format: VAR_X = value, VAR_Y = value, ..."},
            {"role": "assistant", "content": answer},
        ]
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="retrieval_mix.jsonl")
    parser.add_argument("--n_samples", type=int, default=500,
                        help="Total number of samples to generate")
    parser.add_argument("--min_words", type=int, default=1500,
                        help="Minimum filler words per sample")
    parser.add_argument("--max_words", type=int, default=5000,
                        help="Maximum filler words per sample (stay within 8K tokens)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    generators = [
        ("S-NIAH", 0.30, lambda w: gen_single_needle(w)),
        ("MK-NIAH", 0.25, lambda w: gen_multi_needle(random.choice([2, 3, 4]), w)),
        ("KV-Retr", 0.25, lambda w: gen_kv_retrieval(random.choice([5, 8, 10]), w)),
        ("VT", 0.20, lambda w: gen_variable_tracking(random.choice([2, 3]), random.choice([3, 4, 5]), w)),
    ]

    samples = []
    counts = {}

    for i in range(args.n_samples):
        # Select task type by weight
        r = random.random()
        cumulative = 0.0
        for name, weight, gen_fn in generators:
            cumulative += weight
            if r < cumulative:
                target_words = random.randint(args.min_words, args.max_words)
                sample = gen_fn(target_words)
                sample["task_type"] = name
                samples.append(sample)
                counts[name] = counts.get(name, 0) + 1
                break

    # Shuffle
    random.shuffle(samples)

    # Write
    with open(args.output, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Generated {len(samples)} samples → {args.output}")
    for name, count in sorted(counts.items()):
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
