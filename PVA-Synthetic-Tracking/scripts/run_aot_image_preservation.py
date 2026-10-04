#!/usr/bin/env python3
"""Frozen image-domain regional compensation study; never a production detector."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import platform

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PILOT = REPO.parent / 'outputs/seaqr_aot_pilot_20260927'
OUTPUT = PILOT / 'image_preservation_01'
ARMS = ('global_translation', 'local_translation', 'local_affine')
PREVIOUS = (0, 42, 85, 127, 170, 212, 255, 298)
WIDTH, HEIGHT = 2448, 2048
SCHEMA = 'seaqr.aot.image-preservation.v1'
PLAN_SCHEMA = 'seaqr.aot.image-preservation-plan.v1'
PINS = {
    'regional_result_sha256': (PILOT / 'regional_motion_01/result.json', '3161a56132bd5de41273ed495b360050bec286e43108259785307be431d7f822'),
    'download_validation_sha256': (PILOT / 'download_validation.json', 'b602b89755e60122f1cac200808d19bc58d55443697ff5d16f83da8afde530d9'),
    'image_loader_sha256': (REPO / 'scripts/verify_aot_image_patches.py', '491717743c1bc76fdb547b3c172425424d66432d0eab503f5cc13d330222a0f3'),
    'regional_runner_sha256': (REPO / 'scripts/compare_aot_regional_motion.py', 'f724e4b47a17a6c05e97c4eb793cf61bd97374bc2443123de0c39b93bda77d39'),
    'visible_baseline_sha256': (REPO / 'tiny_target/visible_baseline.py', 'd059ef60c9436bbf942a458586d9b546b8f1806db080f8fd20b22b768bf96df2'),
}
ARTIFACTS = {
    'script_sha256': Path(__file__),
    'core_sha256': REPO / 'scripts/aot_image_compensation.py',
    'generated_sha256': REPO / 'scripts/aot_image_temporal_controls.py',
    'summary_sha256': REPO / 'scripts/summarize_aot_image_preservation.py',
    'core_tests_sha256': REPO / 'tests/test_aot_image_compensation.py',
    'runner_tests_sha256': REPO / 'tests/test_aot_image_preservation.py',
    'plan_sha256': REPO / 'docs/aot_image_preservation_plan_20260928.md',
}
DESIGN = dict(previous_indices=list(PREVIOUS), arms=list(ARMS), width=WIDTH, height=HEIGHT,
    interpolation='OpenCV INTER_CUBIC float32, quantized convertMaps CV_16SC2',
    source_footprint='all 16 taps', support_erosion_radius=2, grid=[6, 8],
    main_sites=48, main_sigmas=[.6, 1.2], main_amplitudes=[-16., 16.], main_offsets=[[.5, 0.], [2., -1.]],
    boundary_anchors=[[1224., 853.25], [1224., 1194.25], [1071.25, 1024.], [1377.25, 1024.]],
    boundary_steps=[-1., -.5, 0., .5, 1.], boundary_sigma=.6,
    border_sites=[[1.25, 1.25], [2446.25, 1.25], [1.25, 2046.25], [2446.25, 2046.25]],
    radius=32, gaussian_truncation_sigma=6., intended_peak_dn=16.,
    counts=dict(main=3072, boundary=320, border=128, probes=3520, arm_evaluations=10560),
    generated_trajectories=83, generated_frames=747, numerical_tolerance_dn=1e-4,
    ratio_epsilon=1e-12, significant_oracle_fraction=1e-4,
    production_changed=False, detector_run=False, refitting=False, automatic_promotion=False)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_exclusive(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, allow_nan=False, separators=(',', ':'))
        stream.write('\n')


def hashes():
    result = {}
    for key, (path, expected) in PINS.items():
        require(path.is_file() and not path.is_symlink(), 'missing/linked pinned input')
        result[key] = sha(path)
        require(result[key] == expected, f'pinned artifact changed: {key}')
    for key, path in ARTIFACTS.items():
        require(path.is_file() and not path.is_symlink(), f'missing/linked artifact: {key}')
        result[key] = sha(path)
    return result


def load_module(path, digest, name):
    require(sha(path) == digest and not Path(path).is_symlink(), 'module identity changed')
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, 'module unavailable')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def approved_inventory(loader):
    metadata = loader.image_metadata(read(PINS['download_validation_sha256'][0]))
    allowed = sorted({i for p in PREVIOUS for i in (p, p+1)})
    rows = [metadata[i] for i in allowed]
    require([r['frame_index'] for r in rows] == allowed and len(rows) == 16, 'image scope differs')
    return rows


def check_scope(output):
    output = Path(output)
    require(output == OUTPUT and output.resolve() == OUTPUT and not output.is_symlink(), 'outside fixed image-preservation scope')
    return output


def freeze(output):
    output = check_scope(output)
    require(not output.exists(), 'existing output; refuse to replace freeze')
    before = hashes()
    loader = load_module(PINS['image_loader_sha256'][0], before['image_loader_sha256'], '_aot_preservation_loader_freeze')
    manifest = dict(schema=PLAN_SCHEMA, design=DESIGN, hashes=before, image_inventory=approved_inventory(loader),
                    source_images_opened=False, note='All sites/controls fixed before image-level outcomes.')
    require(before == hashes(), 'code/input changed during freeze')
    output.mkdir()
    write_exclusive(output/'manifest.json', manifest)
    print(json.dumps(dict(frozen=str(output/'manifest.json'), sha256=sha(output/'manifest.json'), images=16)))


def bound_hashes(output):
    output = check_scope(output)
    path = output/'manifest.json'
    require(path.is_file() and not path.is_symlink(), 'missing/linked frozen manifest')
    manifest = read(path)
    current = hashes()
    require(manifest.get('schema') == PLAN_SCHEMA and manifest.get('design') == DESIGN
            and manifest.get('hashes') == current, 'frozen design or code/input differs')
    return dict(**current, manifest_sha256=sha(path))


def cell_id(point):
    x, y = point
    require(0 <= x < WIDTH and 0 <= y < HEIGHT, 'anchor outside image')
    return int(y*6//HEIGHT)*8 + int(x*8//WIDTH)


def nominal_displacement(row, anchor):
    fitted = row['cells'][cell_id(anchor)]['arms']['global_translation']
    require(fitted['training_eligible'] and fitted['fit'] is not None, 'fixed nominal global fit unavailable')
    matrix = np.asarray(fitted['fit']['native_matrix'], dtype=float)
    p = np.asarray(anchor, dtype=float)
    return (matrix @ [*p, 1.])[:2] - p


def probe_specs(row):
    """Geometry only. No source-pixel or output dependence."""
    items = []
    def add(group, site_id, center, anchor, sigma, amp, delta, step=None):
        d = nominal_displacement(row, anchor)
        items.append(dict(id=f"p{row['previous_index']:03d}_{group}_{site_id:02d}_s{sigma:g}_a{amp:+g}_d{delta[0]:g}_{delta[1]:g}_t{step}",
            group=group, site_id=site_id, previous_center_xy=list(center), nominal_anchor_xy=list(anchor),
            current_center_xy=(np.asarray(center)+d+delta).tolist(), nominal_displacement_xy=d.tolist(),
            sigma=float(sigma), amplitude=float(amp), independent_offset_xy=list(delta), sweep_step=step))
    for i in range(48):
        rr, cc = divmod(i, 8)
        p = [(cc+.5)*WIDTH/8+.25, (rr+.5)*HEIGHT/6+.25]
        for sigma in (.6, 1.2):
            for amp in (-16., 16.):
                for delta in ([.5, 0.], [2., -1.]):
                    add('main', i, p, p, sigma, amp, delta)
    for i, anchor in enumerate(DESIGN['boundary_anchors']):
        for step in DESIGN['boundary_steps']:
            p = np.array(anchor) + ([step, 0] if i < 2 else [0, step])
            for amp in (-16., 16.):
                add('boundary', i, p.tolist(), anchor, .6, amp, [.5, 0.], step)
    for i, p in enumerate(DESIGN['border_sites']):
        for sigma in (.6, 1.2):
            for amp in (-16., 16.):
                add('border', i, p, p, sigma, amp, [.5, 0.])
    require(len(items) == 440 and len({r['id'] for r in items}) == 440, 'probe inventory differs')
    return items


def statistics(values):
    v = np.asarray(values, dtype=float)
    require(np.isfinite(v).all(), 'nonfinite statistic values')
    if not v.size:
        return dict(count=0, median=None, p90=None, maximum=None, rms=None)
    a = np.abs(v)
    return dict(count=int(v.size), median=float(np.median(a)), p90=float(np.quantile(a, .9)),
        maximum=float(a.max()), rms=float(np.sqrt(np.mean(v*v))))


def dense_summary(field, residual):
    valid = field['valid']
    cells = []
    for i in range(48):
        rr, cc = divmod(i, 8)
        x0, x1 = int(np.ceil(cc*WIDTH/8)), int(np.ceil((cc+1)*WIDTH/8))
        y0, y1 = int(np.ceil(rr*HEIGHT/6)), int(np.ceil((rr+1)*HEIGHT/6))
        sl = np.s_[y0:y1, x0:x1]
        cells.append(dict(cell_id=i, native_pixels=(x1-x0)*(y1-y0), valid=int(valid[sl].sum()),
                          support_counts={k:int(field[k][sl].sum()) for k in
                              ('numerical','model_support','kernel_valid','valid_pre','valid')},
                          residual=statistics(residual[sl][valid[sl]])))
    mask_keys = ('numerical', 'model_support', 'kernel_valid', 'valid_pre', 'valid')
    out = dict(native_pixels=WIDTH*HEIGHT, support_counts={k:int(field[k].sum()) for k in mask_keys},
        field_hashes={k:array_sha(field[k]) for k in ('qx', 'qy', *mask_keys)},
        upper_half_valid=int(valid[:HEIGHT//2].sum()), lower_half_valid=int(valid[HEIGHT//2:].sum()),
        half_support_counts={half:{k:int(field[k][sl].sum()) for k in mask_keys}
            for half,sl in (('upper',np.s_[:HEIGHT//2,:]),('lower',np.s_[HEIGHT//2:,:]))},
        residual_on_own_support=statistics(residual[valid]), cells=cells)
    jumps=[]
    for direction in ('vertical','horizontal'):
        records=[]
        limits=range(1,8) if direction=='vertical' else range(1,6)
        for i in limits:
            boundary = int(np.ceil(i*(WIDTH/8 if direction=='vertical' else HEIGHT/6)))
            if direction=='vertical':
                left,right=np.s_[:,boundary-1],np.s_[:,boundary]
                step=np.array([1.,0.])
            else:
                left,right=np.s_[boundary-1,:],np.s_[boundary,:]
                step=np.array([0.,1.])
            ok=field['valid'][left]&field['valid'][right]
            # Difference of displacement, not raw coordinates one pixel apart.
            dx=field['qx'][right]-field['qx'][left]-step[0]
            dy=field['qy'][right]-field['qy'][left]-step[1]
            records.append(dict(boundary_index=i, coordinate=boundary,
                possible_count=int(ok.size), supported_count=int(ok.sum()),
                adjacent_displacement_difference_px=statistics(np.hypot(dx[ok],dy[ok]))))
        jumps.append(dict(direction=direction,boundaries=records))
    out['cell_seams']=jumps
    return out


def visual_selected(spec, previous):
    canonical = spec['sigma']==.6 and spec['amplitude']==16. and spec['independent_offset_xy']==[.5,0.]
    return canonical and ((spec['group']=='main' and previous in (0,212) and spec['site_id'] in (0,27,47))
        or (spec['group']=='boundary' and previous==0 and spec['site_id'] in (0,2)))


def render_visual(path, spec, arm, arrays):
    """Display only; fixed ranges and no scientific rescaling of saved inputs."""
    from PIL import Image, ImageDraw
    keys=('previous','current_aligned','clean_residual','injected_residual','delta','oracle')
    tiles=[]
    for key in keys:
        value=np.asarray(arrays[key],dtype=float)
        mask=np.isfinite(value)
        if key in ('previous','current_aligned'):
            clipped=np.clip(np.nan_to_num(value),0,255).astype(np.uint8)
            rgb=np.repeat(clipped[:,:,None],3,axis=2)
        else:
            v=np.clip(np.nan_to_num(value)/16.,-1,1)
            rgb=np.empty((*v.shape,3),np.uint8)
            rgb[:,:,0]=np.rint(127.5*(1+v));rgb[:,:,1]=np.rint(127.5*(1-np.abs(v)));rgb[:,:,2]=np.rint(127.5*(1-v))
        rgb[~mask]=[255,0,255]
        tile=Image.fromarray(rgb).resize((195,195),Image.Resampling.NEAREST)
        tiles.append((key,tile))
    canvas=Image.new('RGB',(195*3,240*2+45),(245,245,245));draw=ImageDraw.Draw(canvas)
    draw.text((6,4),f"{spec['id']} | {arm}",fill='black')
    draw.text((6,19),'3x nearest | residual +/-16 DN | magenta=unavailable',fill='black')
    for i,(key,tile) in enumerate(tiles):
        xx=(i%3)*195; yy=45+(i//3)*240
        draw.text((xx+4,yy+4),key,fill='black');canvas.paste(tile,(xx,yy+24))
    require(not path.exists(), 'refusing visual overwrite')
    canvas.save(path)


def run(output):
    output=check_scope(output)
    require(output.is_dir() and all(not (output/n).exists() and not (output/n).is_symlink()
                                  for n in ('result.json','failure.json','review')),
            'output missing or already executed')
    before=bound_hashes(output)
    receipt=dict(schema=SCHEMA,passed=False,passed_interpretation='Protocol execution only, not model promotion',
        design=DESIGN,hashes_before=before,rows=[],generated=None,annotations_used=False,detector_run=False,
        production_changed=False,remote_accessed=False,refitting=False)
    try:
        core=load_module(ARTIFACTS['core_sha256'],before['core_sha256'],'_aot_image_core')
        controls=load_module(ARTIFACTS['generated_sha256'],before['generated_sha256'],'_aot_image_controls')
        loader=load_module(PINS['image_loader_sha256'][0],before['image_loader_sha256'],'_aot_image_loader')
        import cv2
        receipt['runtime']=dict(python=platform.python_version(),numpy=np.__version__,opencv=cv2.__version__)
        manifest=read(output/'manifest.json');inventory=approved_inventory(loader)
        require(inventory==manifest['image_inventory'],'approved image inventory changed')
        by_index={r['frame_index']:r for r in inventory}
        source=read(PINS['regional_result_sha256'][0])
        require(source['passed'] and source['hashes_before']==source['hashes_after']
                and [r['previous_index'] for r in source['rows']]==list(PREVIOUS),'regional provenance differs')
        (output/'review').mkdir()
        for row in source['rows']:
            prev=loader.load_approved_image(by_index[row['previous_index']])
            cur=loader.load_approved_image(by_index[row['current_index']])
            original_hashes=[array_sha(prev),array_sha(cur)]
            fields={arm:core.build_field(row,arm) for arm in ARMS}
            aligned={arm:core.pull(cur,field) for arm,field in fields.items()}
            residuals={arm:aligned[arm]-prev.astype(np.float32) for arm in ARMS}
            common=np.logical_and.reduce([fields[a]['valid'] for a in ARMS])
            entry=dict(previous_index=row['previous_index'],current_index=row['current_index'],
                dense={arm:dense_summary(fields[arm],residuals[arm]) for arm in ARMS},
                all_three_common=dict(count=int(common.sum()),native_pixels=WIDTH*HEIGHT,
                    residuals={arm:statistics(residuals[arm][common]) for arm in ARMS}),pairwise_common={},probes=[])
            for left,right in itertools.combinations(ARMS,2):
                matched=fields[left]['valid'] & fields[right]['valid']
                entry['pairwise_common'][f'{left}__{right}']=dict(count=int(matched.sum()),
                    native_pixels=WIDTH*HEIGHT,residuals={a:statistics(residuals[a][matched]) for a in (left,right)})
            for spec in probe_specs(row):
                probe=dict(**spec,arms={})
                for arm in ARMS:
                    show=visual_selected(spec,row['previous_index'])
                    result=core.probe(prev,cur,fields[arm],spec['previous_center_xy'],spec['current_center_xy'],
                        spec['sigma'],spec['amplitude'],roi_radius=32,return_arrays=show)
                    arrays=result.pop('arrays',None)
                    if show:
                        require(arrays is not None,'missing predeclared visual')
                        name=f"{spec['id']}__{arm}.png"
                        render_visual(output/'review'/name,spec,arm,arrays)
                        result['visual']=f'review/{name}'
                    probe['arms'][arm]=result
                entry['probes'].append(probe)
            require([array_sha(prev),array_sha(cur)]==original_hashes,'source image array mutated')
            for arm in ARMS:
                require(all(array_sha(fields[arm][key]) == digest
                    for key,digest in entry['dense'][arm]['field_hashes'].items()),
                    'probe changed a frozen map or support mask')
            entry['source_pixel_hashes']=original_hashes
            receipt['rows'].append(entry)
            print(json.dumps(dict(previous_index=row['previous_index'],completed=len(receipt['rows']),total=8)),flush=True)
            del fields,aligned,residuals,prev,cur,common
        receipt['generated']=controls.run_generated(core)
        generated=receipt['generated']
        require(generated.get('passed') is True
            and [len(generated[k]) for k in ('trajectories','support_dropouts','static_repeats')]==[72,3,8]
            and sum(len(t['frames']) for k in ('trajectories','support_dropouts','static_repeats')
                    for t in generated[k])==747
            and generated['counts']['total_frames']==747,'generated inventory or conformance incomplete')
        counts=Counter(p['group'] for row in receipt['rows'] for p in row['probes'])
        require(dict(counts)=={k:DESIGN['counts'][k] for k in ('main','boundary','border')},'actual probe counts differ')
        receipt['completed_counts']=dict(**dict(counts),probes=sum(counts.values()),arm_evaluations=sum(counts.values())*3)
        receipt['image_png_sha256_after']={r['img_name']:sha(loader.SOURCE_DIR/r['img_name']) for r in inventory}
        require(all(receipt['image_png_sha256_after'][r['img_name']]==r['png_sha256'] for r in inventory),'source PNG changed')
        receipt['hashes_after']=bound_hashes(output)
        require(before==receipt['hashes_after'],'frozen code/input changed')
        receipt['passed']=True
        write_exclusive(output/'result.json',receipt)
    except BaseException as exc:
        receipt['error']=repr(exc)
        write_exclusive(output/'failure.json',receipt)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('freeze','run'))
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args()
    (freeze if args.phase=='freeze' else run)(args.output)


if __name__=='__main__':
    main()
