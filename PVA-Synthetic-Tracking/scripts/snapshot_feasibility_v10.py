"""Read-only code/environment provenance for the generated-data experiment."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def paths():
    files=set((ROOT/'tiny_target').rglob('*.py'))
    files.update((ROOT/'tiny_target/detection/cuda').glob('*.cu'))
    files.update((ROOT/'scripts').glob('*v10.py'))
    files.update((ROOT/'scripts').glob('*v10.cu'))
    files.update((ROOT/'tests/unit').glob('*v10.py'))
    files.update(ROOT/p for p in ('scripts/raw16_speed_v8_common.py','scripts/profile_raw16_efficiency.py',
        'docs/raw16_feasibility_v10_plan.md','configs/evaluation/raw16_background_v7.json'))
    # macOS archive sidecars are metadata, not Python/CUDA source modules.
    return sorted(p for p in files if not p.name.startswith('._'))


def source_hashes():
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths()}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    record=dict(captured_utc=datetime.now(timezone.utc).isoformat(),source_sha256=source_hashes(),
        platform=platform.platform(),python=platform.python_version(),real_media_read=False)
    for name,file in (('hardware_model','/proc/device-tree/model'),('l4t','/etc/nv_tegra_release'),('memory','/proc/meminfo')):
        path=Path(file)
        record[name]=path.read_text().replace('\x00','').strip() if path.exists() else None
    nvcc=Path('/usr/local/cuda/bin/nvcc')
    record['nvcc']=subprocess.check_output([str(nvcc),'--version'],text=True) if nvcc.exists() else None
    with a.output.open('x') as f:json.dump(record,f,indent=2,sort_keys=True);f.write('\n')
