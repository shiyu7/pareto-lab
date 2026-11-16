#!/usr/bin/env python3
"""Safely fetch completed points from a still-running Pareto collection."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
import uuid

from collect_and_archive import fetch_remote, positive, valid_point_file


def snapshot(manifest_path, length):
    manifest_path = manifest_path.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    remote = manifest.get('remote_directory')
    if not isinstance(remote, str) or not re.fullmatch(r'/tmp/pareto-skill-\d{8}T\d{6}Z-[0-9a-f]{8}', remote):
        raise ValueError('manifest has an invalid remote run directory')
    settings = manifest.get('settings') or {}
    if length not in settings.get('input_lengths', []):
        raise ValueError('input length is not part of this collection')
    transport = manifest.get('transport')
    if transport == 'kubernetes':
        kube = manifest.get('kubernetes') or {}
        if not all(isinstance(kube.get(k), str) and kube[k] for k in ('context', 'namespace', 'pod')):
            raise ValueError('manifest is missing Kubernetes context, namespace or pod')
    elif transport == 'ssh':
        kube = {}
        target = manifest.get('ssh_target')
        if not isinstance(target, str) or not target or target.startswith('-'):
            raise ValueError('manifest has an invalid SSH target')
    else:
        raise ValueError('manifest has an invalid transport')
    args = SimpleNamespace(transport=transport, context=kube.get('context'), namespace=kube.get('namespace'),
                           pod=kube.get('pod'), container=kube.get('container'), ssh_target=manifest.get('ssh_target'),
                           ssh_port=manifest.get('ssh_port'), ssh_proxyjump=manifest.get('ssh_proxyjump'),
                           python=settings.get('client_python') or 'python3')
    root = manifest_path.parent / 'live'
    root.mkdir(mode=0o700, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    destination = root / f'input-{length}-partial-{stamp}-{uuid.uuid4().hex[:6]}.json'
    ok, error = fetch_remote(args, f'{remote}/input-{length}.points.json', destination)
    if not ok:
        raise ValueError(f'could not fetch current point file: {error}')
    count, complete = valid_point_file(destination)
    if not count:
        destination.unlink()
        raise ValueError('current workload has no importable points yet')
    print(f'Snapshot: {destination}\nPoints: {count}; complete: {complete}')
    return destination


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--input-length', type=positive, required=True)
    args = p.parse_args(argv)
    try:
        snapshot(args.manifest, args.input_length)
    except (OSError, ValueError, TypeError, subprocess.SubprocessError) as exc:
        p.exit(1, f'snapshot failed: {exc}\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
