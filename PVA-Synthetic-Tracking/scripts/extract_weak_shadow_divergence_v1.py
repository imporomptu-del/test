"""Post-hoc metadata-only frames 0..40 from one fixed audited chunk0126 run.

Hash entire parent journals, but decode and copy only the first 41 raw lines.
No detector/tracker/native/media modules or captured pixel arrays are accessed.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import re

FIXED_DIRECTORY = Path('/tmp/seaqr_weak_shadow_v1_LveJSx')
CLIP = '0126'
FRAMES = 674
PREFIX_FRAMES = 41
SOURCE_SHA = 'c5302b873656793da47f1da3c03f05df595f17c3f9bc407ce0bfd99b7e718344'
CONFIG_SHA = '7c473765048e8e7f8c87042a421e0b22daf6280bb4591fd50f1d438ba2597d2f'
MOTION_SHA = 'fe450546af91f01a0fb090d76df3ba4db24081b0077c6a220d5194990fdda5b1'
SCHEMA = 'seaqr.weak-shadow.divergence-prefix.v1'
MAX_METADATA = 64*1024*1024
MAX_PREFIX = 256*1024*1024


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest_ok(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}',value) is not None


def regular(path):
    path=Path(path)
    require(path.is_absolute() and path.resolve()==path and path.is_file() and not path.is_symlink(),
            'Literal canonical regular metadata/code file required')
    require(path.suffix in ('.json','.jsonl','.py'), 'Only metadata/code inputs permitted')
    return path


def sha(path):
    h=hashlib.sha256()
    with regular(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def bytes_sha(raw):
    return hashlib.sha256(raw).hexdigest()


def decode(raw):
    def pairs(items):
        result={}
        for key,value in items:
            require(key not in result,'Duplicate JSON key')
            result[key]=value
        return result
    def number(text):
        value=float(text)
        require(math.isfinite(value),'Nonfinite JSON number')
        return value
    def invalid(_):
        raise ValueError('Nonfinite JSON constant')
    return json.loads(raw,object_pairs_hook=pairs,parse_float=number,parse_constant=invalid)


class BoundInputs:
    def __init__(self):
        self.hashes={}

    def bind(self,path,expected):
        path=regular(path)
        require(digest_ok(expected) and sha(path)==expected,'Changed pinned parent: '+str(path))
        require(str(path) not in self.hashes or self.hashes[str(path)]==expected,'Conflicting parent binding')
        self.hashes[str(path)]=expected
        return path

    def read(self,path,expected):
        path=self.bind(path,expected)
        require(path.suffix=='.json' and path.stat().st_size<=MAX_METADATA,'Bounded JSON metadata required')
        raw=path.read_bytes();value=decode(raw)
        require(type(value) is dict,'Metadata JSON object required')
        require(bytes_sha(raw)==expected,'Input changed while read')
        self.bind(path,expected)
        return value,raw

    def unchanged(self):
        for path,digest in list(self.hashes.items()):
            self.bind(path,digest)


def prefix(path,kind):
    chunks=[];rows=[];size=0
    with regular(path).open('rb') as stream:
        for index in range(PREFIX_FRAMES):
            line=stream.readline(MAX_METADATA+1)
            require(0<len(line)<=MAX_METADATA and line.endswith(b'\n') and bool(line.strip()),
                    'Missing/oversized/nonterminated prefix line')
            row=decode(line)
            require(type(row) is dict and type(row.get('frame_index')) is int and row['frame_index']==index
                    and type(row.get('timestamp_ns')) is int and row['timestamp_ns']==index*100000000
                    and type(row.get('segment')) is int and row['segment']>=0,'Wrong causal prefix frame')
            fields=('tracks','candidates') if kind=='clean' else ('records','strong_proposals','prior_forecasts')
            require(all(type(row.get(name)) is list for name in fields),'Missing complete causal tracker metadata')
            chunks.append(line);rows.append((row['frame_index'],row['timestamp_ns'],row['segment']))
            size+=len(line)
            require(size<=MAX_PREFIX,'Prefix exceeds fixed metadata memory bound')
    return b''.join(chunks),rows


def launch_valid(launch):
    require(launch.get('source') == '/home/serg/project/camera_reader_sky/srcsky/chunks/chunk_0126.avi'
            and launch.get('source_sha256')==SOURCE_SHA,'Wrong original source identity')
    require(launch.get('config_sha256')==CONFIG_SHA and launch.get('motion_config_sha256')==MOTION_SHA,
            'Wrong frozen configuration')
    require(type(launch.get('expected_frames')) is int and launch['expected_frames']==FRAMES
            and launch.get('max_frames') is None and launch.get('fps')==10
            and launch.get('annotations_supplied_to_detector') is False,'Changed full causal replay launch')
    require(type(launch.get('configuration')) is dict and launch['configuration'].get('input_bit_depth')==8,
            'Original8bit configuration required')
    for key in ('package_sha256','code_sha256'):
        require(type(launch.get(key)) is dict and bool(launch[key])
                and all(type(p) is str and digest_ok(d) for p,d in launch[key].items()),'Missing original source hashes')


def extract(directory,audit_sha256,freeze_sha256,output):
    directory,output=Path(directory),Path(output)
    require(directory==FIXED_DIRECTORY and directory.resolve()==directory and directory.is_dir(),
            'Only the fixed original experiment directory is authorized')
    require(output.is_absolute() and output.resolve()==output and not output.exists() and not output.is_symlink()
            and output.parent.is_dir() and directory not in output.parents,
            'Fresh canonical output directory outside original experiment required')
    inputs=BoundInputs()
    frozen,freeze_raw=inputs.read(directory/'freeze.json',freeze_sha256)
    require(frozen.get('schema')=='seaqr.weak-continuation-shadow.freeze.v1' and frozen.get('pre_run') is True,
            'Original prerun freeze required')
    plan,plan_raw=inputs.read(directory/'plan.json',frozen.get('plan_sha256'))
    require(plan.get('schema')=='seaqr.weak-continuation-shadow.plan.v1' and plan.get('full_causal_replay') is True
            and plan.get('clips',{}).get(CLIP,{}).get('frames')==FRAMES
            and plan['clips'][CLIP].get('source_sha256')==SOURCE_SHA,'Original fullclip plan differs')
    audit,audit_raw=inputs.read(directory/CLIP/'independent_audit.json',audit_sha256)
    require(audit.get('schema')=='seaqr.weak-continuation-shadow.audit.v1' and audit.get('passed') is True
            and audit.get('clip')==CLIP and type(audit.get('frames')) is int and audit['frames']==FRAMES,
            'Full674frame passed original audit required')
    for key,value in (('freeze_sha256',freeze_sha256),('plan_sha256',frozen['plan_sha256']),('source_sha256',SOURCE_SHA),
        ('baseline_journal_non_timing_exact',True),('baseline_output_state_learning_digests_exact',True),
        ('native_state_guards_unchanged',True),('production_changed',False),('weak_learning_enabled',False)):
        require(type(audit.get(key)) is type(value) and audit[key]==value,'Original audit guard differs: '+key)
    hashes=audit.get('files_sha256',{})
    names=('clean/frames.jsonl','shadow/shadow_trace.jsonl','clean/launch.json','shadow/launch.json')
    require(type(hashes) is dict and all(name in hashes for name in names),'Audit lacks parent bindings')
    parents={name:inputs.bind(directory/CLIP/name,hashes[name]) for name in names}
    launches={};copies={'freeze.json':freeze_raw,'plan.json':plan_raw,'original_audit.json':audit_raw}
    for arm in ('clean','shadow'):
        launch,raw=inputs.read(parents[arm+'/launch.json'],hashes[arm+'/launch.json'])
        launch_valid(launch);launches[arm]=launch;copies[arm+'_launch.json']=raw
    require(launches['clean']==launches['shadow'],'Original clean/shadow launch differs')
    reference,raw=inputs.read(directory/'reference_0126.json',frozen.get('files_sha256',{}).get('reference_0126.json'))
    for key in ('configuration','package_sha256','code_sha256','source_sha256','config_sha256','motion_config_sha256','fps'):
        require(reference.get(key)==launches['clean'].get(key),'Frozen reference/launch mismatch: '+key)
    copies['reference_0126.json']=raw
    raw_clean,clean_frames=prefix(parents['clean/frames.jsonl'],'clean')
    raw_shadow,shadow_frames=prefix(parents['shadow/shadow_trace.jsonl'],'shadow')
    require(clean_frames==shadow_frames,'Prefix baseline/shadow frame alignment differs')
    source=Path(__file__).resolve();test=source.with_name('test_extract_weak_shadow_divergence_v1.py')
    if not test.is_file():test=source.parents[1]/'tests/unit/test_extract_weak_shadow_divergence_v1.py'
    for path in (source,test):inputs.bind(path,sha(path))
    inputs.unchanged()
    receipt=dict(schema=SCHEMA,passed=False,error=None,clip=CLIP,parent_directory=str(directory),
        original_full_frames=FRAMES,frame_start=0,frame_end_inclusive=40,frames=PREFIX_FRAMES,
        audit_sha256=audit_sha256,freeze_sha256=freeze_sha256,plan_sha256=frozen['plan_sha256'],
        source_sha256=SOURCE_SHA,configuration_sha256=CONFIG_SHA,motion_configuration_sha256=MOTION_SHA,
        parent_and_code_sha256=inputs.hashes,artifacts={},metadata_only=True,posthoc_diagnostic=True,
        source_media_accessed=False,native_arrays_accessed=False,detector_or_tracker_rerun=False,
        raw_prefix_lines_preserved=True,rows_filtered=False,
        scope='Only fixed chunk0126 frames0..40 inclusive. Complete parent bytes are hashed; only prefix JSON lines decoded and copied.')
    output.mkdir()
    try:
        for name,raw in copies.items():
            with (output/name).open('xb') as stream:stream.write(raw)
            receipt['artifacts'][name]=dict(sha256=bytes_sha(raw),bytes=len(raw))
        for name,raw,parent in (('clean_prefix.jsonl.gz',raw_clean,'clean/frames.jsonl'),
                                ('shadow_prefix.jsonl.gz',raw_shadow,'shadow/shadow_trace.jsonl')):
            with (output/name).open('xb') as stream:
                with gzip.GzipFile(filename='',mode='wb',fileobj=stream,mtime=0,compresslevel=6) as compressed:
                    compressed.write(raw)
            receipt['artifacts'][name]=dict(sha256=bytes_sha((output/name).read_bytes()),
                raw_prefix_sha256=bytes_sha(raw),raw_bytes=len(raw),frames=PREFIX_FRAMES,
                full_parent_sha256=hashes[parent],full_parent_path=str(parents[parent]))
        inputs.unchanged()
        receipt['passed']=True
    except Exception as exc:
        receipt['error']=repr(exc)
        raise
    finally:
        with (output/'receipt.json').open('x') as stream:
            json.dump(receipt,stream,indent=2,allow_nan=False);stream.write('\n')
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--audit-sha256',required=True)
    parser.add_argument('--freeze-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=extract(args.directory,args.audit_sha256,args.freeze_sha256,args.output)
    print(json.dumps({k:result[k] for k in ('schema','passed','clip','frames','frame_start','frame_end_inclusive')}))
