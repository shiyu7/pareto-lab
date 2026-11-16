#!/usr/bin/env python3
"""Tokenize an officially resolved SPEED-Bench throughput JSONL/parquet for SGLang.

Run the official source-resolution script first. This helper never fetches data,
executes remote tokenizer code, pads/truncates prompts, or silently fills missing text.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from speed_bench_data import CONFIG, ENTROPIES, manifest_path, sha256


def source_rows(path):
    if path.suffix == '.jsonl':
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)
    elif path.suffix == '.parquet':
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=16):
            yield from batch.to_pylist()
    else:
        raise ValueError('input must be resolved JSONL or parquet')


def encode_row(row, tokenizer, prompt_format):
    turns = row.get('turns')
    if not isinstance(turns, list) or len(turns) != 1 or not isinstance(turns[0], str):
        raise ValueError('throughput input must have exactly one resolved text turn; do not flatten conversations')
    text = turns[0]
    if not text.strip() or 'FULL BENCHMARK DATA SHOULD BE FETCHED' in text.upper():
        raise ValueError('unresolved SPEED-Bench source placeholder or empty prompt')
    category, qid = row.get('category'), row.get('question_id')
    if category not in ENTROPIES or not isinstance(qid, str) or not qid:
        raise ValueError('missing/invalid question_id or entropy category')
    if prompt_format == 'chat':
        ids = tokenizer.apply_chat_template([dict(role='user', content=text)],
                                            tokenize=True, add_generation_prompt=True)
    else:
        ids = tokenizer.encode(text, add_special_tokens=True)
    if not ids or any(type(x) is not int or x < 0 for x in ids):
        raise ValueError('tokenizer returned invalid token IDs')
    return dict(question_id=qid, category=category, input_ids=ids,
                source=row.get('source'), src_id=row.get('src_id'),
                prompt_sha256=hashlib.sha256(text.encode()).hexdigest())


def prepare(args, tokenizer):
    if not args.dataset_revision.strip():
        raise ValueError('dataset revision must not be blank')
    if not CONFIG.fullmatch(args.config):
        raise ValueError('only official throughput configs are supported')
    if args.output.exists() or manifest_path(args.output).exists():
        raise ValueError('output already exists')
    if args.context_limit <= 0 or args.max_output_tokens <= 0:
        raise ValueError('context limit and output reserve must be positive')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + '.partial')
    counts, seen, stats = Counter(), set(), {}
    try:
        with partial.open('x') as stream:
            for raw in source_rows(args.input):
                row = encode_row(raw, tokenizer, args.prompt_format)
                qid, cat, n = row['question_id'], row['category'], len(row['input_ids'])
                if qid in seen:
                    raise ValueError('duplicate question_id: ' + qid)
                if n + args.max_output_tokens > args.context_limit:
                    raise ValueError(f'{qid}: input {n} + output reserve exceeds context limit; no truncation performed')
                seen.add(qid)
                counts[cat] += 1
                s = stats.setdefault(cat, dict(min=n, max=n, total=0))
                s.update(min=min(s['min'], n), max=max(s['max'], n), total=s['total'] + n)
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        if set(counts) != set(ENTROPIES):
            raise ValueError('prepared file must contain all three entropy categories')
        for cat, s in stats.items():
            s['mean'] = s.pop('total') / counts[cat]
        files = {}
        for pattern in ('tokenizer*', '*vocab*', 'merges.txt', 'special_tokens_map.json', 'added_tokens.json', 'config.json'):
            for p in args.tokenizer.glob(pattern):
                if p.is_file():
                    files[p.name] = sha256(p)
        meta = dict(schema='speed-bench-tokenized', version=1,
                    dataset='nvidia/SPEED-Bench', config=args.config,
                    declared_dataset_revision=args.dataset_revision,
                    source_file=args.input.name, source_sha256=sha256(args.input),
                    tokenizer=str(args.tokenizer.resolve()), tokenizer_file_hashes=files,
                    prompt_format=args.prompt_format,
                    chat_template_sha256=hashlib.sha256(str(getattr(tokenizer, 'chat_template', None)).encode()).hexdigest(),
                    context_limit=args.context_limit, output_reserve=args.max_output_tokens,
                    counts=dict(counts), actual_input_tokens=stats, sha256=sha256(partial),
                    transforms='target tokenizer and declared template only; no padding/truncation')
        # Publish the manifest last; interrupted pairs fail validation on next use.
        partial.rename(args.output)
        with manifest_path(args.output).open('x') as stream:
            json.dump(meta, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        return meta
    finally:
        partial.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--dataset-revision', required=True, help='record the actual HF revision used during source resolution')
    p.add_argument('--tokenizer', type=Path, required=True, help='existing local tokenizer directory')
    p.add_argument('--prompt-format', choices=('chat', 'completion'), default='chat')
    p.add_argument('--context-limit', type=int, required=True)
    p.add_argument('--max-output-tokens', type=int, default=1536)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if not args.tokenizer.is_dir():
        p.error('--tokenizer must be an existing local directory')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True, trust_remote_code=False)
    try:
        meta = prepare(args, tokenizer)
    except (ValueError, OSError) as exc:
        p.exit(1, str(exc) + '\n')
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
