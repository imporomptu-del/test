"""One worker, four predeclared reference/staged development diagnostics."""
from pathlib import Path
import subprocess
import sys
import time
from attribute_visible_native_v25 import sha,write,read

HERE=Path(__file__).resolve().parent


def main():
    if (HERE/'attribution_batch.json').exists():raise FileExistsError('Preserve prior diagnostics')
    rows=[];error=None
    try:
        for clip,mode in (('0126','reference'),('0126','staged'),('0082','staged'),('0082','reference')):
            name=f'attribute_{clip}_{mode}';print('START '+name,flush=True)
            command=[sys.executable,str(HERE/'attribute_visible_native_v25.py'),'--clip',clip,
                     '--mode',mode,'--output',str(HERE/name)]
            start=time.monotonic()
            with (HERE/(name+'.log')).open('x') as log:
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,cwd=HERE)
            row=dict(name=name,clip=clip,mode=mode,command=command,returncode=result.returncode,
                log_sha256=sha(HERE/(name+'.log')),wall_s=time.monotonic()-start)
            rows.append(row)
            if result.returncode:raise RuntimeError('Diagnostic failed: '+name)
            receipt=HERE/(name+'.attribution.json');r=read(receipt)
            if not r['passed']:raise AssertionError('Diagnostic gate failed')
            row['receipt_sha256']=sha(receipt);print('DONE '+name,flush=True)
    except BaseException as exc:error=repr(exc);raise
    finally:write(HERE/'attribution_batch.json',dict(passed=error is None,error=error,rows=rows,
        script_sha256=sha(__file__),attribution_sha256=sha(HERE/'attribute_visible_native_v25.py'),
        plan_sha256=sha(HERE/'visible_native_v25_plan.md')))


if __name__=='__main__':main()
