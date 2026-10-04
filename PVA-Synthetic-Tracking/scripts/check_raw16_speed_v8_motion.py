"""Repeat the frozen 48 PVA controls with exact CPU optimization adapters."""
import argparse
from pathlib import Path
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from raw16_speed_v8_common import cpu_motion_execution, package_identity
from profile_raw16_efficiency import write_json, sha, compact
import check_raw16_motion_v5 as generated
from run_raw16_speed_v8 import read
import run_raw16_cpu_v6 as v6


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for seed in (75316,129827,85723):
        path=args.output/f'controls_{seed}.json'
        with cpu_motion_execution() as cache, patch.object(generated,'CONFIG',v6.CONFIG):
            code=generated.run(path,seed)
        current=read(path);reference=read(args.evidence/f'status_controls_seed{seed}.json')
        exact=compact(current['cases'])==compact(reference['cases'])
        rows.append(dict(seed=seed,returncode=code,exact=exact,sha256=sha(path),
                         reference_sha256=sha(args.evidence/f'status_controls_seed{seed}.json'),
                         cache_hits=cache.hits,cache_misses=cache.misses))
        if code or not exact:
            write_json(args.output/'gate.json',dict(passed=False,rows=rows))
            return 2
    write_json(args.output/'gate.json',dict(passed=True,rows=rows,package_sha256=package_identity(),
                                           script_sha256=sha(__file__),real_media_read=False))
    return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--evidence',type=Path,required=True)
    raise SystemExit(run(parser.parse_args()))
