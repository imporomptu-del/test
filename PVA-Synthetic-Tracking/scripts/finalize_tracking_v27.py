"""Save post-run unit evidence and artifact hashes without overwriting results."""
from pathlib import Path
import subprocess
import sys
from profile_visible_v17 import read,sha,write
from check_tracking_batch_v27 import FILES

HERE=Path(__file__).resolve().parent


def main():
    names=('unit_gate.log','unit_gate.json','evidence_manifest.json')
    if any((HERE/n).exists() for n in names):raise FileExistsError('Fresh evidence manifest only')
    gate=read(HERE/'checked_replays_01.json')
    if not gate['passed'] or gate['error'] is not None or len(gate['replays'])!=8:
        raise ValueError('Complete exact replay gate required')
    if any(sha(HERE/n)!=d for n,d in gate['source_sha256'].items()):raise ValueError('Changed replay source')
    command=[sys.executable,'-m','unittest','-v','test_tracking_batch_v27']
    with (HERE/'unit_gate.log').open('x') as handle:
        done=subprocess.run(command,cwd=HERE,stdout=handle,stderr=subprocess.STDOUT)
    write(HERE/'unit_gate.json',dict(passed=done.returncode==0,returncode=done.returncode,command=command,
        log_sha256=sha(HERE/'unit_gate.log'),test_sha256=sha(HERE/'test_tracking_batch_v27.py')))
    if done.returncode:raise RuntimeError('Final Jetson unit gate failed')
    artifacts=FILES+('profile_0126_01.json','profile_0082_01.json','checked_replays_01.json',
        'tracking_geometry_v20.cpp','tracking_geometry_v20.py','check_tracking_geometry_v20.py',
        'build_tracking_geometry_v20.py','profile_visible_v17.py','finalize_tracking_v27.py',
        'build_01/build.json','build_01/libtracking_batch_v27.so','build_01/source/tracking_batch_v27.cpp',
        'build_01/source/tracking_geometry_v20.cpp','unit_gate.json','unit_gate.log')
    write(HERE/'evidence_manifest.json',dict(schema='seaqr.tracking-v27.evidence.v1',post_run_manifest=True,
        files_sha256={n:sha(HERE/n) for n in artifacts},media_decoded=False,production_changed=False))


if __name__=='__main__':main()
