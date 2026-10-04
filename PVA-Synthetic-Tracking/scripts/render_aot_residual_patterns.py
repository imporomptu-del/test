#!/usr/bin/env python3
"""Render numerical residual summaries only; never opens source imagery."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PILOT = Path(__file__).resolve().parents[2] / "outputs/seaqr_aot_pilot_20260927"
ROOT = PILOT / "residual_patterns_01"
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
INK, BLUE, LIGHT = "#182733", "#006b86", "#e5eaee"


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def label(draw, xy, text, size=19, bold=False, fill=INK):
    draw.text(xy, text, font=font(size, bold), fill=fill)


def arrow(draw, start, end, fill=BLUE, width=3):
    x, y = end
    dx, dy = x-start[0], y-start[1]
    length = math.hypot(dx, dy)
    if length < .8:
        draw.ellipse((x-2, y-2, x+2, y+2), fill=fill)
        return
    draw.line((start, end), fill=fill, width=width)
    ux, uy = dx/length, dy/length
    tip = min(7, length*.45)
    draw.polygon([(x, y), (x-tip*ux-tip*.5*uy, y-tip*uy+tip*.5*ux),
                  (x-tip*ux+tip*.5*uy, y-tip*uy-tip*.5*ux)], fill=fill)


def residual_panel(row, arrow_scale):
    canvas = Image.new("RGB", (1120, 1120), "white")
    draw = ImageDraw.Draw(canvas)
    label(draw, (45, 26), f"Actual pair {row['previous']:03d} to {row['previous']+1:03d}", 29, True)
    label(draw, (45, 69), "Scene-feature residuals, not detected targets", 21)
    label(draw, (45, 106), f"Saved translation {row['translation'][0]:+.3f}, {row['translation'][1]:+.3f} px  |  fit REJECTED", 20)
    label(draw, (45, 138), f"Accepted {row['accepted']} / selected {row['selected']}  |  residual p50 {row['p50']:.2f}, p90 {row['p90']:.2f} px", 20)
    x0, y0, scale = 72, 211, 976/2448
    gw, gh = 2448*scale, 2048*scale
    cw, ch = gw/8, gh/6
    for cell in row['cells']:
        x, y = x0+cell['column']*cw, y0+cell['row']*ch
        draw.rectangle((x, y, x+cw, y+ch), fill="#f6f8fa" if cell['median_vector'] is None else "#ffffff", outline=LIGHT)
    for px, py in row['points']:
        x, y = x0+px*scale, y0+py*scale
        draw.ellipse((x-1.5, y-1.5, x+1.5, y+1.5), fill="#a4acb3")
    for cell in row['cells']:
        x, y = x0+cell['column']*cw, y0+cell['row']*ch
        label(draw, (x+5, y+5), f"n={cell['count']}", 15, fill="#697580")
        v = cell['median_vector']
        if v is None:
            label(draw, (x+cw/2-13, y+ch/2-9), "n/a", 15, fill="#8c979f")
        else:
            start = (x+cw/2, y+ch/2)
            arrow(draw, start, (start[0]+arrow_scale*v[0], start[1]+arrow_scale*v[1]))
    for value in (0, 612, 1224, 1836, 2448):
        label(draw, (x0+value*scale-16, y0-28), str(value), 16)
    for value in (0, 512, 1024, 1536, 2048):
        label(draw, (9, y0+value*scale-10), str(value), 15)
    label(draw, (72, 1043), f"Native x-right / y-down. One residual px = {arrow_scale:.2f} display px (same across all 8 panels).", 18)
    label(draw, (72, 1073), "Arrows: cell median residual; n/a: fewer than 5 accepted points. Gray dots: accepted feature locations.", 17)
    return canvas


def comparison_plot(actual, natural):
    canvas = Image.new("RGB", (1260, 900), "white")
    draw = ImageDraw.Draw(canvas)
    label(draw, (42, 26), "Local agreement versus joint-shuffled residuals", 30, True)
    label(draw, (42, 76), "Ratio < 1: nearby accepted features agree more closely than after spatial shuffling.", 21)
    label(draw, (42, 108), "Descriptive only: different accepted cohorts/graphs; not a p-value or model-selection test.", 20)
    values = [r['ratio'] for r in actual+natural if r['ratio'] is not None]
    xmax = max(1.1, max(values)*1.08)
    left, right, top = 180, 940, 210
    def xpos(value):
        return left + value/xmax*(right-left)
    for tick in (0, .25, .5, .75, 1):
        x = xpos(tick)
        draw.line((x, top-20, x, 745), fill="#bcc8cf" if tick == 1 else LIGHT, width=2)
        label(draw, (x-14, 759), f"{tick:g}", 18)
    label(draw, (965, 168), "Actual anchor support", 18, True)
    for i, real in enumerate(actual):
        y = top + i*72
        label(draw, (42, y-9), f"{real['previous']:03d} to {real['previous']+1:03d}", 21, True)
        refs = [r['ratio'] for r in natural if r['previous'] == real['previous'] and r['ratio'] is not None]
        if refs:
            draw.line((xpos(min(refs)), y, xpos(max(refs)), y), fill="#adb7be", width=7)
            for val in refs:
                x = xpos(val)
                draw.ellipse((x-3, y-3, x+3, y+3), fill="#8797a2")
        if real['ratio'] is not None:
            x = xpos(real['ratio'])
            draw.ellipse((x-8, y-8, x+8, y+8), fill=BLUE)
            label(draw, (x+13, y+10), f"{real['ratio']:.3f}", 16, fill=BLUE)
        label(draw, (972, y-9), f"{real['anchors']}/{real['accepted']} ({real['anchor_fraction']:.0%})", 19)
    label(draw, (180, 799), "Observed local disagreement / median shuffled disagreement", 20)
    label(draw, (42, 842), "Blue: actual pair. Gray: six nonzero synthetic shifts per source. Eight zero-shift ratios are undefined (0/0).", 18)
    return canvas


def normalized(row):
    coherence = row['coherence']
    assert row['residual_available'] and coherence is not None
    return dict(previous=row['previous_index'], translation=row['saved_candidate_translation'],
        accepted=row['accepted_count'], selected=row['selected_count'],
        p50=row['candidate_residual_summary']['norm_quantiles']['p50'],
        p90=row['candidate_residual_summary']['norm_quantiles']['p90'],
        cells=row['grid']['cells'], points=row['points']['accepted_previous_xy'],
        ratio=coherence['observed_to_reference_median'], anchors=coherence['anchor_count'],
        anchor_fraction=coherence['anchor_fraction'])


def main():
    result_path = ROOT / 'result.json'
    raw = result_path.read_bytes()
    result = json.loads(raw)
    assert result['passed'] and len(result['rows']) == 64
    real_rows = result['rows'][:8]
    assert all(r['source_kind'] == 'aot_adjacent' and r['original_fit']['quality_status'] == 'rejected' for r in real_rows)
    actual = [normalized(r) for r in real_rows]
    natural = [normalized(r) for r in result['rows'][8:]]
    maximum = max(math.hypot(*c['median_vector']) for r in actual for c in r['cells'] if c['median_vector'] is not None)
    arrow_scale = 48.8 / maximum if maximum else 1.0
    output = ROOT / 'figures'
    output.mkdir(exist_ok=False)
    names = []
    overview = Image.new('RGB', (1680, 4*840), 'white')
    for i, row in enumerate(actual):
        panel = residual_panel(row, arrow_scale)
        name = f"pair_{row['previous']:03d}_{row['previous']+1:03d}_residuals.png"
        panel.save(output / name)
        overview.paste(panel.resize((840, 840), Image.Resampling.LANCZOS), ((i%2)*840, (i//2)*840))
        names.append(name)
    overview.save(output / 'all_eight_residual_maps.png')
    comparison_plot(actual, natural).save(output / 'coherence_comparison.png')
    receipt = dict(schema='seaqr.aot.residual-figures.v1', result_sha256=hashlib.sha256(raw).hexdigest(),
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_media_opened=False, actual_pair_count=8, cell_median_residual_max_px=maximum,
        arrow_display_pixels_per_residual_pixel=arrow_scale, common_arrow_scale=True,
        image_axes='native x-right/y-down; 2448x2048', no_model_fitted=True,
        display_only=True, files=names+['all_eight_residual_maps.png', 'coherence_comparison.png'])
    with (output / 'receipt.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
