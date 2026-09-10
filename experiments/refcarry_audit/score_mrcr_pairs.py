"""Score complete generated token streams for original/derived MRCR rows.

Official SequenceMatcher and strict whole-string-plus-EOS are separate metrics.
This script performs no generation and never extracts a first answer/substring.
"""
import argparse
from collections import defaultdict
from difflib import SequenceMatcher
import json
from pathlib import Path


def score_text(text, ended_eos, row):
    marker = row['random_string_to_prepend']
    answer = row['references'][0]
    prefix_ok = text.startswith(marker)
    ratio = SequenceMatcher(None, text[len(marker):], answer[len(marker):]).ratio() if prefix_ok else 0.
    exact = text == answer
    other_answers = {marker+row['messages'][i]['content'] for i in row['occurrence_message_indices']}
    other_answers.discard(answer)
    return dict(prefix_ok=prefix_ok, official_sequence_ratio=ratio,
        whole_string_exact=exact, full_exact_and_eos=exact and ended_eos,
        ended_eos=ended_eos, wrong_occurrence_whole_string=text in other_answers)


def aggregate(rows):
    methods = defaultdict(list)
    for row in rows:
        methods[row['method']].append(row)
    result = {}
    for method, items in methods.items():
        by_variant = defaultdict(list)
        by_family = defaultdict(list)
        for item in items:
            by_variant[item['variant']].append(item)
            if item['variant'] != 'compact_control':
                by_family[item['family_id']].append(item)
        complete = [x for x in by_family.values() if len(x) == 4]
        result[method] = dict(rows=len(items),
            variants={key:dict(n=len(group),
                sequence_ratio=sum(r['official_sequence_ratio'] for r in group)/len(group),
                full_exact_and_eos=sum(r['full_exact_and_eos'] for r in group),
                ended_eos=sum(r['ended_eos'] for r in group),
                wrong_occurrence_exact=sum(r['wrong_occurrence_whole_string'] for r in group))
                for key, group in by_variant.items()},
            complete_four_way_families=len(complete),
            four_way_exact_and_eos=sum(all(r['full_exact_and_eos'] for r in group) for group in complete),
            incomplete_families=len(by_family)-len(complete))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    inputs = {r['row_id']:r for r in map(json.loads, args.inputs.read_text().splitlines())}
    results, seen = [], set()
    for prediction in map(json.loads, args.predictions.read_text().splitlines()):
        key = (prediction['method'], prediction['row_id'])
        if key in seen:raise ValueError('duplicate method/row prediction')
        seen.add(key)
        row = inputs[prediction['row_id']]
        if row['input_ids_sha256'] != prediction['input_ids_sha256']:
            raise ValueError('prediction was produced from different input tokens')
        ids, eos = prediction['generated_ids'], set(prediction['eos_token_ids'])
        if not ids or any(token in eos for token in ids[:-1]):
            raise ValueError('missing output or generation continued after EOS')
        ended = ids[-1] in eos
        # Remove only terminal EOS. Any other generated special token stays
        # visible and can make whole-string exactness fail.
        text = tokenizer.decode(ids[:-1] if ended else ids, skip_special_tokens=False)
        score = score_text(text, ended, row)
        results.append(dict(method=prediction['method'], row_id=row['row_id'],
            family_id=row['family_id'], variant=row['variant'], world=row['world'],
            requested_ordinal=row['requested_ordinal'], output_text=text,
            generated_ids=ids, **score))
    args.output.mkdir(parents=True)
    (args.output/'scores.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results))
    summary = aggregate(results)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':main()
