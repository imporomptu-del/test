"""Exact visible journal gate with one explicit directional GPU-binary change."""
from copy import deepcopy
from profile_visible_v17 import read,sha
from compare_phase20_exact_runs import validate_decode,shape_accelerator,validate_gpu_transition
from repeat_phase20_kernel_speed import check_prefix


def check(reference,output,count,full,mode,config,config_sha,transition):
    left,right=[read(p/'launch.json') for p in (reference,output)]
    old,report=[read(p/'report.json') for p in (reference,output)]
    expected=deepcopy(left['configuration'])
    if mode=='candidate':expected['cuda_median_library']=config['cuda_median_library']
    if config!=expected or right['configuration']!=expected or report['configuration']!=expected:
        raise AssertionError('Only the explicit CUDA library path may change')
    if right['config_sha256']!=config_sha:raise AssertionError('Actual launch configuration hash changed')
    for key in ('source_sha256','fps','motion_config_sha256','package_sha256'):
        if left[key]!=right[key]:raise AssertionError('Provenance changed: '+key)
    after=transition['after_library_sha256'] if mode=='candidate' else transition['before_library_sha256']
    validate_gpu_transition(left,right,transition if mode=='candidate' else None)
    exact=deepcopy(left['exact_cuda_stabilization']);exact['library_sha256']=after
    if exact!=right['exact_cuda_stabilization']:raise AssertionError('Stabilization contract changed')
    external=deepcopy(left['external_accelerators'])
    external['median'].update(library_path=expected['cuda_median_library'],library_sha256=after)
    if external!=right['external_accelerators']:raise AssertionError('Accelerator provenance changed')
    if (not report['completed'] or report['frames']!=count or report['full_clip']!=full
            or report['faint_target_synthetic_branch_enabled'] is not False):
        raise AssertionError('Incomplete or changed visible branch')
    validate_decode(right,report);shape_accelerator(right);check_prefix(reference,output,count)
    motion=read(output.with_suffix('.execution.json'));original=read(reference.with_suffix('.execution.json'))
    for key in ('adapter_sha256','method_sha256','wrapper_sha256','runtime_sha256'):
        if motion[key]!=original[key]:raise AssertionError('Motion provenance changed: '+key)
    if (not motion['passed'] or motion['error'] is not None or not motion['closed']
            or motion['branch']!='visible' or motion['mode']!='reuse' or motion['processed_frames']!=count
            or motion['reuse_hits']!=count-2 or motion['reuse_misses']!=1):
        raise AssertionError('Motion lifecycle failed')
    if [(r['frame'],r['identity']) for r in motion['motion']] != [(r['frame'],r['identity']) for r in original['motion'][:count-1]]:
        raise AssertionError('Complete motion outputs changed')
    if full:
        for key in ('counts','qualified_tracks','qualified_track_count','availability','detection_status'):
            if old[key]!=report[key]:raise AssertionError('Aggregate changed: '+key)
    return dict(exact=True,frames=count,reference_journal_sha256=sha(reference/'frames.jsonl'),
                journal_sha256=sha(output/'frames.jsonl'),execution_sha256=sha(output.with_suffix('.execution.json')))
