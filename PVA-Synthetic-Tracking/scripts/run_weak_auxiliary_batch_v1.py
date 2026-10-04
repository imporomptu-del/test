"""Serial frozen metadata replay/audit/score harness; stop at the first failure."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)


def run(freeze, freeze_sha):
    root=Path(freeze).resolve().parent
    if sha(freeze)!=freeze_sha:
        raise ValueError('Frozen bundle differs')
    doc=json.loads(Path(freeze).read_text())
    for name,digest in doc['files_sha256'].items():
        if Path(name).name!=name or sha(root/name)!=digest:
            raise ValueError('Changed bundle source '+name)
    if sha(__file__)!=doc['files_sha256'][Path(__file__).name]:
        raise ValueError('Unfrozen batch harness')
    output=root/'results_01'
    output.mkdir(exist_ok=False)
    write(output/'started.json',dict(freeze_sha256=freeze_sha,started_utc=datetime.now(timezone.utc).isoformat(),
        workers=1,source_media_accessed=False))
    rows=[]
    def stage(name,command):
        print(name,'starting',flush=True)
        with (output/(name+'.log')).open('x') as stream:
            result=subprocess.run(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT,check=False)
        row=dict(stage=name,command=command,returncode=result.returncode,
            ended_utc=datetime.now(timezone.utc).isoformat(),log_sha256=sha(output/(name+'.log')))
        rows.append(row);write(output/(name+'.json'),row)
        if result.returncode:
            raise RuntimeError('Stopped on failed stage '+name)
        print(name,'passed',flush=True)
    passed=False
    try:
        for clip in ('0029','0126','0055'):
            stage('replay_'+clip,[sys.executable,str(root/'run_weak_auxiliary_replay_v1.py'),
                '--directory',doc['original_root'],'--clip',clip,'--audit-sha256',doc['original_audits_sha256'][clip],
                '--freeze',str(freeze),'--freeze-sha256',freeze_sha,'--output',str(output/clip)])
            stage('audit_'+clip,[sys.executable,str(root/'audit_weak_auxiliary_replay_v1.py'),
                '--directory',str(output/clip),'--freeze',str(freeze),'--freeze-sha256',freeze_sha,
                '--output',str(output/clip/'independent_audit.json')])
        stage('references',[sys.executable,str(root/'score_weak_auxiliary_references_v1.py'),
            '--directory',str(output),'--references',str(root/'references.json'),'--freeze',str(freeze),
            '--freeze-sha256',freeze_sha,'--output',str(output/'reference_score.json')])
        passed=True
    finally:
        write(output/'batch_status.json',dict(passed=passed,stages=rows,freeze_sha256=freeze_sha,
            ended_utc=datetime.now(timezone.utc).isoformat(),production_changed=False,source_media_accessed=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',type=Path,required=True);parser.add_argument('--freeze-sha256',required=True)
    args=parser.parse_args();run(args.freeze,args.freeze_sha256)
