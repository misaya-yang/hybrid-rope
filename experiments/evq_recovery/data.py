"""Small streaming primitives shared by preparation, training, and evaluation."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import string


def sha_text(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def chat_ids(tokenizer, messages, generation=False):
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=generation)
    if isinstance(ids, dict) or hasattr(ids, 'input_ids'):
        ids = ids['input_ids']
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def supervised_chat(tokenizer, question, answer):
    messages = [{'role': 'user', 'content': question}]
    prompt = chat_ids(tokenizer, messages, generation=True)
    full = chat_ids(tokenizer, messages + [{'role': 'assistant', 'content': answer}])
    if full[:len(prompt)] != prompt:
        raise ValueError('native assistant prefix is not an exact training prefix')
    if len(full) <= len(prompt) or full[-1] != tokenizer.eos_token_id:
        raise ValueError('complete answer and native EOS required')
    return dict(input_ids=full, target_start=len(prompt), prompt_tokens=len(prompt),
                answer_tokens=len(full)-len(prompt))


def normalize(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    return ' '.join(text.split())


def qa_scores(prediction, references):
    if not references:
        raise ValueError('answer references required')
    pred = normalize(prediction).split()
    f1 = 0.
    for reference in references:
        gold = normalize(reference).split()
        shared = sum((Counter(pred) & Counter(gold)).values())
        value = 2*shared/(len(pred)+len(gold)) if pred and gold else float(pred == gold)
        f1 = max(f1, value)
    return dict(f1=f1, exact=float(any(normalize(prediction) == normalize(x) for x in references)))


class JsonlIndex:
    """Keep byte offsets, not Python lists for millions of token IDs."""
    def __init__(self, path):
        self.path = Path(path)
        self.offsets = []
        with self.path.open('rb') as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        with self.path.open('rb') as f:
            f.seek(self.offsets[index])
            return json.loads(f.readline())
