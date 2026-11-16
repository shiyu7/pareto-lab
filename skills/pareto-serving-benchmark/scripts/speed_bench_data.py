"""Validated, pre-tokenized SPEED-Bench inputs. No downloads or synthetic fallback."""
import hashlib
import json
from pathlib import Path
import random
import re

ENTROPIES = ('low_entropy', 'mixed', 'high_entropy')
CONFIG = re.compile(r'throughput_(1k|2k|8k|16k|32k|100k)\Z')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def manifest_path(path):
    return Path(path).with_suffix('.manifest.json')


def load_corpus(path, entropy):
    path = Path(path)
    meta = json.loads(manifest_path(path).read_text())
    if meta.get('schema') != 'speed-bench-tokenized' or meta.get('version') != 1:
        raise ValueError('unsupported SPEED-Bench corpus manifest')
    if not CONFIG.fullmatch(meta.get('config', '')) or entropy not in ENTROPIES:
        raise ValueError('expected a throughput config and one entropy category')
    if sha256(path) != meta.get('sha256'):
        raise ValueError('tokenized corpus SHA-256 mismatch')
    rows, seen, counts = [], set(), dict.fromkeys(ENTROPIES, 0)
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            category, qid = row.get('category'), row.get('question_id')
            if category not in ENTROPIES or not isinstance(qid, str) or not qid or qid in seen:
                raise ValueError('invalid category or duplicate/missing question_id')
            seen.add(qid)
            counts[category] += 1
            ids = row.get('input_ids')
            if not isinstance(ids, list) or not ids or any(type(x) is not int or x < 0 for x in ids):
                raise ValueError('invalid pre-tokenized input IDs')
            if category == entropy:
                rows.append(row)
    if counts != meta.get('counts') or not all(counts.values()):
        raise ValueError('manifest counts mismatch or a missing entropy category')
    if len(rows) < 2:
        raise ValueError('need disjoint warmup and measurement prompts')
    return rows, meta


class PromptPool:
    def __init__(self, rows, seed, warmup_prompts=16, replay_policy='cycle'):
        if not 1 <= warmup_prompts < len(rows):
            raise ValueError('warmup-prompts must leave a nonempty measured corpus')
        self.rows = list(rows)
        random.Random(seed).shuffle(self.rows)
        self.warmup = self.rows[:warmup_prompts]
        self.measured = self.rows[warmup_prompts:]
        self.policy = replay_policy
        self.cursor = self.warmup_cursor = 0

    def next(self, warmup=False):
        if warmup:
            row = self.warmup[self.warmup_cursor % len(self.warmup)]
            self.warmup_cursor += 1
            return row, 0
        epoch, index = divmod(self.cursor, len(self.measured))
        if epoch and self.policy == 'stop':
            raise ValueError('SPEED-Bench corpus exhausted before collection finished')
        self.cursor += 1
        return self.measured[index], epoch

    def summary(self):
        return dict(warmup_prompts=len(self.warmup), measured_prompts=len(self.measured),
                    replay_policy=self.policy, requests_selected=self.cursor,
                    unique_prompts_selected=min(self.cursor, len(self.measured)),
                    repeated_requests=max(0, self.cursor - len(self.measured)))
