"""Prepare paired occurrence tests from pinned original MRCR conversations.

No model inference. Derived worlds are development diagnostics, not original
MRCR benchmark rows. Original labels are checked against the conversation.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':')).encode()).hexdigest()


def ordinal(n):
    suffix = 'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return str(n) + suffix


def locate(row):
    messages = json.loads(row['prompt'])
    marker = row['random_string_to_prepend']
    if not row['answer'].startswith(marker):
        raise ValueError('original target is missing its output marker')
    target_text = row['answer'][len(marker):]
    hits = [i for i, m in enumerate(messages) if m['role'] == 'assistant' and m['content'] == target_text]
    if len(hits) != 1:
        raise ValueError('target answer must identify exactly one assistant message')
    target = hits[0]
    if target < 1 or messages[target-1]['role'] != 'user':
        raise ValueError('target must follow a user request')
    request = messages[target-1]['content']
    occurrences = [i+1 for i, m in enumerate(messages[:-1])
        if m['role'] == 'user' and m['content'] == request
        and messages[i+1]['role'] == 'assistant']
    if len(occurrences) != row['n_needles']:
        raise ValueError('actual number of identical requests differs from source metadata')
    rank = occurrences.index(target)+1
    query = messages[-1]['content']
    matches = list(re.finditer(r'\b\d+(?:st|nd|rd|th)\b', query))
    if len(matches) != 1 or matches[0].group() != ordinal(rank):
        raise ValueError('query ordinal does not match its actual target occurrence')
    if messages[-1]['role'] != 'user' or marker not in query:
        raise ValueError('original final request is malformed')
    # This release stores the preceding user-message index, not the assistant.
    if row['desired_msg_index'] != target-1:
        raise ValueError('source desired_msg_index contract changed')
    return messages, occurrences, rank, matches[0].span()


def family(row, source_index):
    messages, occurrences, rank, span = locate(row)
    alternate = rank % len(occurrences)+1
    a, b = occurrences[rank-1], occurrences[alternate-1]
    if messages[a]['content'] == messages[b]['content']:
        raise ValueError('paired answers are identical')
    group = digest({'source_index': source_index, 'row': row})
    marker = row['random_string_to_prepend']
    records = []
    for world in (0, 1):
        history = copy.deepcopy(messages[:-1])
        if world:
            history[a]['content'], history[b]['content'] = history[b]['content'], history[a]['content']
        for requested in (rank, alternate):
            original_query = messages[-1]['content']
            query = original_query[:span[0]] + ordinal(requested) + original_query[span[1]:]
            answer = marker + history[occurrences[requested-1]]['content']
            records.append(dict(family_id=group, row_id=f'{group[:12]}_w{world}_q{requested}',
                variant='original' if world == 0 and requested == rank else 'counterfactual',
                world=world, requested_ordinal=requested,
                messages=history+[{'role': 'user', 'content': query}], references=[answer],
                random_string_to_prepend=marker, occurrence_message_indices=occurrences,
                target_message_index=occurrences[requested-1],
                history_sha256=digest(history), source_row_index=source_index,
                source_row_sha256=digest(row), n_needles=len(occurrences)))
    # The four long variants contain exactly the same historical message texts.
    inventory = sorted((m['role'], m['content']) for m in records[0]['messages'][:-1])
    for record in records:
        assert sorted((m['role'], m['content']) for m in record['messages'][:-1]) == inventory
    assert records[0]['references'] == records[3]['references']
    assert records[1]['references'] == records[2]['references']
    assert records[0]['references'] != records[1]['references']
    # Compact controls retain every competing occurrence and the source's
    # original introductory instructions/examples, not just the gold response.
    compact = copy.deepcopy(messages[:1])
    compact_occurrences = []
    for index in occurrences:
        compact.extend(copy.deepcopy(messages[index-1:index+1]))
        compact_occurrences.append(len(compact)-1)
    for long_record in records[:2]:
        compact_record = copy.deepcopy(long_record)
        compact_record.update(row_id=long_record['row_id']+'_compact', variant='compact_control',
            messages=copy.deepcopy(compact)+[copy.deepcopy(long_record['messages'][-1])],
            occurrence_message_indices=compact_occurrences,
            target_message_index=compact_occurrences[long_record['requested_ordinal']-1],
            history_sha256=digest(compact))
        records.append(compact_record)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--families', type=int, default=8)
    parser.add_argument('--max-context', type=int, default=32768)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    receipt = json.loads(args.receipt.read_text())
    if hashlib.sha256(args.source.read_bytes()).hexdigest() != receipt['expected_hub_metadata']['lfs']['oid']:
        raise ValueError('original source failed LFS hash verification')
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    table = pq.ParquetFile(args.source)
    candidates, rejected = [], {}
    for index, batch in enumerate(table.iter_batches(batch_size=1)):
        row = batch.to_pylist()[0]
        if row['n_chars'] > 120000 or len(row['answer']) > 1800:
            continue
        try:
            records = family(row, index)
            for record in records:
                ids = tok.apply_chat_template(record['messages'], tokenize=True, add_generation_prompt=True)
                prefix = tok.apply_chat_template(record['messages'][:-1], tokenize=True, add_generation_prompt=False)
                if ids[:len(prefix)] != prefix:
                    raise ValueError('chat template does not permit this exact query-blind prefix')
                answer_tokens = len(tok.encode(record['references'][0], add_special_tokens=False))
                cap = answer_tokens + 64
                if len(ids)+cap > args.max_context:
                    raise ValueError('whole prompt and answer exceed native context')
                record.update(input_ids=ids, prefix_tokens=len(prefix), input_tokens=len(ids),
                    max_new_tokens=cap, expected_answer_tokens=answer_tokens,
                    input_ids_sha256=digest(ids))
            # Every counterfactual in a family receives the same decode budget;
            # the requested occurrence must not be signalled by its token cap.
            family_cap = max(r['expected_answer_tokens'] for r in records)+64
            if any(r['input_tokens']+family_cap > args.max_context for r in records):
                raise ValueError('shared family answer budget exceeds native context')
            for record in records:
                record['max_new_tokens'] = family_cap
            candidates.append(records)
        except ValueError as error:
            reason=str(error);rejected[reason]=rejected.get(reason,0)+1
    candidates.sort(key=lambda records:records[0]['family_id'])
    selected = candidates[:args.families]
    if len(selected) != args.families:
        raise ValueError(f'only {len(selected)} eligible families; rejected={rejected}')
    args.output.mkdir(parents=True)
    rows = [r for group in selected for r in group]
    payload = ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows)
    (args.output/'inputs.jsonl').write_text(payload)
    tokenizer_hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in args.tokenizer.iterdir() if p.is_file()}
    summary = dict(status='PREPARED_CPU_ONLY', source_receipt=receipt,
        families=len(selected), rows=len(rows), eligible_families=len(candidates), rejected=rejected,
        selection='first family hashes among intact source rows <=120000 chars and <=1800 answer chars, fitting every derived full prompt and answer within native context; no model outputs used',
        native_context_limit=args.max_context, tokenizer_files=tokenizer_hashes,
        derived_inputs_sha256=hashlib.sha256(payload.encode()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        long_input_tokens=[r['input_tokens'] for r in rows if r['variant']!='compact_control'],
        generation_caps=[r['max_new_tokens'] for r in rows],
        scope='MRCR-derived development causal pairs, NOT an official full MRCR benchmark result; no inference or quality qualification has run')
    (args.output/'manifest.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:summary[k] for k in ('status','families','rows','eligible_families','rejected','derived_inputs_sha256')}))


if __name__ == '__main__':
    main()
