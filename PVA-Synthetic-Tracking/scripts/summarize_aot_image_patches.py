#!/usr/bin/env python3
"""Descriptive summary and frozen-sample contact sheets; not a motion estimator."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def distribution(values):
    values = np.asarray(values, dtype=float)
    return dict(count=len(values), median=None if not len(values) else float(np.median(values)),
                p90=None if not len(values) else float(np.percentile(values, 90)))


def describe(rows):
    supported = [r for r in rows if r['comparison']['two_sizes_qualified_and_consistent']]
    scales = {}
    for i, size in enumerate((33, 65)):
        failures = Counter()
        for row in rows:
            scale = row['scales'][i]
            if not scale['available']:
                failures[scale['unavailable_reason']] += 1
            else:
                failures.update(key for key, passed in scale['qualification'].items() if not passed)
        offsets = np.array([r['scales'][i]['best_offset_xy'] for r in supported], dtype=float).reshape(-1, 2)
        global_residuals = []
        for row in supported:
            if 'saved_candidate_residual_xy' in row:
                candidate = np.array(row['saved_lk_displacement_xy']) - row['saved_candidate_residual_xy']
                global_residuals.append(np.linalg.norm(np.array(row['scales'][i]['best_offset_xy']) - candidate))
        scales[str(size)] = dict(qualified=sum(r['scales'][i]['qualified'] for r in rows),
            failure_counts_nonexclusive=dict(sorted(failures.items())),
            supported_subset_residual_to_original_global_candidate_px=distribution(global_residuals),
            supported_subset_offset_component_min_xy=None if not len(offsets) else offsets.min(axis=0).tolist(),
            supported_subset_offset_component_max_xy=None if not len(offsets) else offsets.max(axis=0).tolist(),
            supported_subset_offset_component_range_xy=None if not len(offsets) else np.ptp(offsets, axis=0).tolist(),
            supported_subset_difference_to_saved_lk_px=distribution(
                [r['comparison']['saved_lk_error_by_size_px'][i] for r in supported]))
    return dict(count=len(rows), mutually_supported=len(supported),
                verdict_counts=dict(Counter(r['comparison']['verdict'] for r in rows)), scales=scales)


def make_summary(rows, result_sha):
    actual = [r for r in rows if r['source_kind'] == 'actual_residual_extremum']
    controls = [r for r in rows if r['source_kind'] != 'actual_residual_extremum']
    return dict(schema='seaqr.aot.image-patch-summary.v1', result_sha256=result_sha,
        interpretation='Purposive residual-extreme scene-motion sample; not target accuracy or a population LK error rate.',
        actual=describe(actual),
        pairs=[dict(previous_index=index, **describe([r for r in actual if r['previous_index'] == index]))
               for index in (0, 42, 85, 127, 170, 212, 255, 298)],
        roles={role: describe([r for r in actual if role in r['selection_roles']]) for role in ('minimum', 'maximum')},
        roles_note='Deduplicated points bearing both roles appear in both role tables, only once in actual denominator.',
        counterexamples=[dict(id=r['id'], comparison=r['comparison'],
            known_shift_xy=r['synthetic_current']['shift_xy'], saved_lk_truth_error_px=float(np.linalg.norm(
                np.array(r['saved_lk_displacement_xy']) - r['synthetic_current']['shift_xy'])),
            scales=[dict(size=s['template_size'], qualified=s['qualified'], offset_xy=s.get('best_offset_xy'),
                ncc=s.get('best_ncc'), gap=s.get('gap'), truth_error_px=None if not s['available'] else
                float(np.linalg.norm(np.array(s['best_offset_xy']) - r['synthetic_current']['shift_xy']))) for s in r['scales']])
            for r in controls])


def font(size=15, bold=False):
    name = 'Arial Bold.ttf' if bold else 'Arial.ttf'
    return ImageFont.truetype('/System/Library/Fonts/Supplemental/' + name, size)


def native_crop(image, point, size):
    if point is None:
        return None
    x, y = np.floor(np.asarray(point) + .5).astype(int)
    h = size // 2
    if x-h < 0 or y-h < 0 or x+h >= image.shape[1] or y+h >= image.shape[0]:
        return None
    return Image.fromarray(image[y-h:y+h+1, x-h:x+h+1])


def exact_shift(previous, shift):
    dx, dy = map(int, shift)
    current = np.full_like(previous, 128)
    height, width = previous.shape
    x0, x1, y0, y1 = max(0, -dx), min(width, width-dx), max(0, -dy), min(height, height-dy)
    current[y0+dy:y1+dy, x0+dx:x1+dx] = previous[y0:y1, x0:x1]
    return current


def render_sheet(rows, images, title, destination):
    # Each record shows both native template sizes with fixed integer display
    # enlargement. No interpolation, contrast stretch, image-derived sorting,
    # qualification-based omission, or mark on the center pixel.
    width, block = 1260, 360
    canvas = Image.new('RGB', (width, 98 + block*len(rows)), '#f4f6f8')
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 13), title, font=font(22, True), fill='#162437')
    draw.text((20, 44), 'Frozen spatial sample. U8 unchanged; nearest-neighbor 4x (33px) / 2x (65px). LK display endpoint rounded half-up.', font=font(15), fill='#344154')
    draw.text((20, 67), 'Patch agreement is not object identity. Ambiguous = insufficient evidence. Counterexamples use synthetic shifted current images.', font=font(15), fill='#344154')
    for j, row in enumerate(rows):
        y = 98 + block*j
        previous = images[row['previous_index']]
        current = images[row['current_index']] if row['source_kind'] == 'actual_residual_extremum' else exact_shift(previous, row['synthetic_current']['shift_xy'])
        if row['source_kind'] != 'actual_residual_extremum':
            assert hashlib.sha256(current.tobytes()).hexdigest() == row['synthetic_current']['expected_pixel_sha256']
        p = np.array(row['previous_xy'])
        tag = row['comparison']['verdict']
        draw.rectangle((10, y, width-10, y+block-8), fill='white', outline='#ccd4dc')
        draw.text((20, y+8), f"{row['id']} | {tag}", font=font(15, True), fill='#162437')
        draw.text((20, y+30), f"p={p.tolist()}  saved LK d={np.round(row['saved_lk_displacement_xy'], 3).tolist()}  roles={row['selection_roles']}", font=font(14), fill='#344154')
        for k, size in enumerate((33, 65)):
            top = y + 55 + k*146
            positions = [p, row['current_xy']] + [None if not s['available'] else p + s['best_offset_xy'] for s in row['scales']]
            labels = ['previous', 'current @ LK', 'current @ NCC33', 'current @ NCC65']
            for col, (point, label) in enumerate(zip(positions, labels)):
                x = 20 + col*157
                crop = native_crop(previous if col == 0 else current, point, size)
                draw.text((x, top), f'{size}px {label}', font=font(12), fill='#344154')
                if crop is None:
                    draw.text((x, top+50), 'unavailable', font=font(14), fill='#7d3640')
                else:
                    factor = 4 if size == 33 else 2
                    canvas.paste(crop.resize((size*factor, size*factor), Image.Resampling.NEAREST), (x, top+14))
            s = row['scales'][k]
            x = 665
            draw.text((x, top+10), f"{size}px match: d={s.get('best_offset_xy')} | qualified={s['qualified']}", font=font(15, True), fill='#162437')
            if s['available']:
                values = [('NCC', s['best_ncc']), ('runner-up', s['runner_up_ncc']), ('gap', s['gap']),
                          ('std previous', s['template_std']), ('std current', s['best_current_std'])]
                text = '  '.join(f'{name}={value:.4f}' if value is not None else f'{name}=none' for name, value in values)
                draw.text((x, top+36), text, font=font(13), fill='#344154')
                failed = [key for key, passed in s['qualification'].items() if not passed]
                draw.text((x, top+59), 'Failed: ' + (', '.join(failed) or 'none'), font=font(13), fill='#7d3640' if failed else '#344154')
                draw.text((x, top+82), f"offset bounds={s['geometry']['offset_bounds']}  clipped={s['geometry']['search_clipped']}", font=font(13), fill='#344154')
                draw.text((x, top+104), f"distance to LK={row['comparison']['saved_lk_error_by_size_px'][k]:.4f}px", font=font(13), fill='#344154')
            else:
                draw.text((x, top+36), s['unavailable_reason'], font=font(14), fill='#7d3640')
    canvas.save(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', required=True, type=Path)
    parser.add_argument('--selection', required=True, type=Path)
    parser.add_argument('--metadata', required=True, type=Path)
    parser.add_argument('--source-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    # Input schema is checked here after the frozen runner's explicit audit.
    input_hashes = {str(p): digest(p) for p in (args.result, args.selection, args.metadata)}
    result = json.loads(args.result.read_text())
    selection = json.loads(args.selection.read_text())
    rows = result['rows']
    assert result['passed'] is True
    assert result['selection_sha256'] == input_hashes[str(args.selection)]
    assert result['hashes_before'] == result['hashes_after'] == selection['hashes_before'] == selection['hashes_after']
    lookup = {r['id']: r for r in rows}
    assert len(lookup) == len(rows) == len(selection['points'])
    for point in selection['points']:
        assert all(lookup[point['id']][key] == value for key, value in point.items())
    visual_ids = [entry['point_id'] for entry in selection['visual_selection']]
    assert len(visual_ids) == len(set(visual_ids)) and all(identifier in lookup for identifier in visual_ids)
    args.output.mkdir(parents=False, exist_ok=False)
    summary = make_summary(rows, digest(args.result))
    with (args.output/'summary.json').open('x') as out:
        json.dump(summary, out, indent=2, allow_nan=False)
        out.write('\n')
    metadata_hash = digest(args.metadata)
    assert metadata_hash == 'b602b89755e60122f1cac200808d19bc58d55443697ff5d16f83da8afde530d9'
    metadata = json.loads(args.metadata.read_text())['images']
    images, sources = {}, {}
    for index in (0, 1, 42, 43, 85, 86, 127, 128, 170, 171, 212, 213, 255, 256, 298, 299):
        entry = metadata[index]
        assert Path(entry['img_name']).name == entry['img_name']
        path = args.source_dir/entry['img_name']
        sources[str(path)] = digest(path)
        assert sources[str(path)] == entry['png_sha256']
        with Image.open(path) as source:
            values = np.asarray(source.convert('L'))
        assert values.shape == (2048, 2448) and values.dtype == np.uint8
        assert hashlib.sha256(values.tobytes()).hexdigest() == entry['pixel_sha256']
        images[index] = values
    # Runtime selector records fixed visual IDs before matching; no sorting or
    # choosing by new match quality is allowed here.
    visual = [lookup[identifier] for identifier in visual_ids]
    pages = []
    for index in (0, 42, 85, 127, 170, 212, 255, 298):
        subset = [r for r in visual if r['source_kind'] == 'actual_residual_extremum' and r['previous_index'] == index]
        # Four records per page avoids overly tall unreadable contact sheets.
        for start in range(0, len(subset), 4):
            filename = f'pair_{index:03d}_{index+1:03d}_{start//4+1:02d}.png'
            render_sheet(subset[start:start+4], images, f'Actual pair {index:03d} to {index+1:03d} | fixed visual review', args.output/filename)
            pages.append(dict(file=filename, point_ids=[r['id'] for r in subset[start:start+4]]))
    controls = [r for r in visual if r['source_kind'] != 'actual_residual_extremum']
    filename = 'known_shift_counterexamples.png'
    render_sheet(controls, images, 'Known-shift counterexamples | not actual next frames', args.output/filename)
    pages.append(dict(file=filename, point_ids=[r['id'] for r in controls]))
    assert {key: digest(Path(key)) for key in sources} == sources
    assert {key: digest(Path(key)) for key in input_hashes} == input_hashes
    receipt = dict(schema='seaqr.aot.image-patch-rendering.v1', script_sha256=digest(__file__),
        result_sha256=digest(args.result), selection_sha256=digest(args.selection), metadata_sha256=metadata_hash,
        input_png_sha256=sources, input_artifact_sha256=input_hashes, source_files_unchanged=True, input_artifacts_unchanged=True, pages=pages,
        generated_files_sha256={p.name: digest(p) for p in sorted(args.output.iterdir())})
    with (args.output/'rendering_receipt.json').open('x') as out:
        json.dump(receipt, out, indent=2, allow_nan=False)
        out.write('\n')
    print(json.dumps(dict(output=str(args.output), actual=summary['actual'], visual_pages=len(pages)), indent=2))


if __name__ == '__main__':
    main()
