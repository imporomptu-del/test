"""Cross-workspace array audit: normalize ONLY the two verified library paths."""
import json
from pathlib import Path

from profile_raw16_efficiency import sha
from summarize_raw16_cpu_v6 import difference, digest

SYNTHETIC_LIBRARY_SHA='e29dc8bae949e41497aff82cfa2fe07d52e1c039b5c337187bdb8c2e87b1fc65'
LIBRARY_PATHS=frozenset({
    '/tmp/seaqr_raw16_background_v7_vvwfye/build/cuda/libtiny_target_cuda.so',
    '/tmp/seaqr_raw16_speed_v8_boh3Kh/build/cuda/libtiny_target_cuda.so',
})
COUNTS={'source_frame':64,'pva_motion':63,'global_fit':63,'full_resolution_warp':64,'crop':64,
        'background_and_filter':64,'background_whitened':63,'cuda_shift_stack':6,'candidate_ranking':6,
        'candidate_extract':6,'synthetic_association':6,'finalize':1}


def normalized(value):
    if isinstance(value,dict):
        result={}
        for key,item in value.items():
            if key=='library_path':
                if item not in LIBRARY_PATHS:raise ValueError('Unknown library identity/path in audit')
                result[key]='<verified-sha256:'+SYNTHETIC_LIBRARY_SHA+'>'
            else:result[key]=normalized(item)
        return result
    if isinstance(value,list):return [normalized(item) for item in value]
    return value


def compare_records(a,b):
    counts=lambda rows:{name:sum(r['stage']==name for r in rows) for name in {r['stage'] for r in rows}}
    complete=counts(a)==counts(b)==COUNTS
    left,right=normalized(a),normalized(b)
    return dict(passed=complete and left==right,complete=complete,exact_after_path_normalization=left==right,
        first_difference=difference(left,right),event_count=len(a),counts=counts(a),
        normalized_left_sha256=digest(left),normalized_right_sha256=digest(right),
        only_normalization='The two separately SHA-verified synthetic-library workspace paths; no arrays or numbers changed.')


def compare_audits(left,right):
    a,b=[[json.loads(line) for line in Path(path).read_text().splitlines()] for path in (left,right)]
    return dict(**compare_records(a,b),raw_left_sha256=sha(left),raw_right_sha256=sha(right))
