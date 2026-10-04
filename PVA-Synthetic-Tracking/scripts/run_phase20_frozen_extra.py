"""Run extra CPU media from an archived runtime, not the changing working tree."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from score_phase20_accuracy import digest
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--clip',choices=['0055','0082'],required=True);a=p.parse_args()
    a.root=a.root.resolve();a.runtime=a.runtime.resolve()
    frozen=json.loads((a.root/'freeze.json').read_text())
    for name,sha in frozen['code_sha256'].items():
        if digest(a.runtime/name)!=sha:raise ValueError('Archived runtime mismatch')
    if digest(a.root/'config.json')!=frozen['config_sha256']:raise ValueError('Changed configuration')
    packet=json.loads((ROOT/'results/tiny_target/phase20/encounter_accuracy_v2_20260914/scoring_packet.json').read_text())
    source=next(s for s in packet['plan']['sources'] if s['clip_id']==a.clip)
    if digest(source['path'])!=source['sha256']:raise ValueError('Changed source')
    subprocess.run([sys.executable,'-m','tiny_target.visible_baseline','--source',source['path'],
        '--config',str(a.root/'config.json'),'--output',str(a.root/('cpu_'+a.clip))],cwd=a.runtime,check=True)


if __name__=='__main__':main()
