"""Bundle explicit runtime and speed tools only; no media, holdouts or credentials."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tarfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tiny_target.visible_baseline import VisibleConfig,sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    base=ROOT/'results/tiny_target/phase20/maturity_validation_v5_20260914/config.json'
    config=asdict(VisibleConfig(**json.loads(base.read_text())))
    cp=a.output/'reference_config.json';cp.write_text(json.dumps(config,indent=2))
    paths=list(sorted((ROOT/'tiny_target').rglob('*.py')))+[
        ROOT/'scripts'/s for s in ['phase20_cuda_median.cu','benchmark_phase20_cuda_median.py','benchmark_phase20_state_update.py']]
    with tarfile.open(a.output/'runtime.tar.gz','w:gz') as tar:
        for path in paths:tar.add(path,arcname=str(path.relative_to(ROOT)))
        tar.add(cp,arcname='reference_config.json')
    (a.output/'freeze.json').write_text(json.dumps(dict(
        files_sha256={str(p.relative_to(ROOT)):sha256(p) for p in paths},
        reference_config_sha256=sha256(cp),archive_sha256=sha256(a.output/'runtime.tar.gz'),
        allowed_clip_ids=['0126'],scope='Exact-output speed experiments only; no detector threshold changes'),indent=2))
    print(a.output/'runtime.tar.gz')


if __name__=='__main__':main()
