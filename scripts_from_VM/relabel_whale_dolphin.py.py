#!/usr/bin/env python3
"""
relabel_whale_dolphin.py

Rewrite YOLO .txt labels (class ids) in /all_labels/ according to species in labels.csv:
- 7 = humpback_whale
- 5 = dolphins
- 6 = whales
"""

import argparse
import pandas as pd
from pathlib import Path

# --- Class maps ---
WHALES = {
    "humpback_whale","beluga","minke_whale","fin_whale","blue_whale",
    "gray_whale","southern_right_whale","sei_whale","brydes_whale","cuviers_beaked_whale"
}
DOLPHINS = {
    "melon_headed_whale","false_killer_whale","bottlenose_dolphin","dusky_dolphin",
    "spinner_dolphin","common_dolphin","killer_whale","pilot_whale",
    "long_finned_pilot_whale","short_finned_pilot_whale","white_sided_dolphin",
    "spotted_dolphin","pantropical_spotted_dolphin","pygmy_killer_whale",
    "rough_toothed_dolphin","commersons_dolphin","globicephala_sp","frasier_s_dolphin"
}
ALIASES = {
    "pantropic_spotted_dolphin":"pantropical_spotted_dolphin",
    "bottlenose_dolpin":"bottlenose_dolphin",
    "kiler_whale":"killer_whale",
    "globis":"globicephala_sp",
    "frasiers_dolphin":"frasier_s_dolphin",
}

def normalize_species(s: str) -> str:
    s = (s or "").strip().lower()
    return ALIASES.get(s, s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV file with columns: image,species,individual_id")
    ap.add_argument("--labels-dir", required=True, help="Folder with all_labels/*.txt")
    ap.add_argument("--dry-run", action="store_true", help="Preview changes without writing files")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    lbl_dir = Path(args.labels_dir)

    if not csv_path.exists():
        raise SystemExit(f"[ERROR] CSV not found: {csv_path}")
    if not lbl_dir.exists():
        raise SystemExit(f"[ERROR] labels-dir not found: {lbl_dir}")

    # Load CSV into basename -> species map
    df = pd.read_csv(csv_path)
    if "image" not in df.columns or "species" not in df.columns:
        raise SystemExit("[ERROR] CSV must contain columns: image,species")
    df["basename"] = df["image"].apply(lambda x: Path(str(x)).stem)
    df["species_n"] = df["species"].apply(normalize_species)
    img2species = dict(zip(df["basename"], df["species_n"]))

    rewritten, skipped_no_csv, skipped_unknown = 0, 0, 0

    for txt in lbl_dir.glob("*.txt"):
        stem = txt.stem  # e.g. 00021adfb725ed
        sp = img2species.get(stem)
        if sp is None:
            skipped_no_csv += 1
            continue

        if sp == "humpback_whale":
            cid = 7
        elif sp in DOLPHINS:
            cid = 5
        elif sp in WHALES:
            cid = 6
        else:
            skipped_unknown += 1
            continue

        lines = [ln.strip() for ln in txt.read_text().splitlines() if ln.strip()]
        if not lines:
            continue
        new_lines = []
        changed = False
        for ln in lines:
            parts = ln.split()
            if len(parts) < 5:
                continue
            if parts[0] != str(cid):
                parts[0] = str(cid)
                changed = True
            new_lines.append(" ".join(parts))
        if changed and not args.dry_run:
            txt.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            rewritten += 1
        elif changed and args.dry_run:
            print(f"[DRY-RUN] {txt} → class {cid}")

    print(f"[DONE] rewritten: {rewritten}")
    print(f"[DONE] skipped (no CSV match): {skipped_no_csv}")
    print(f"[DONE] skipped (unknown species): {skipped_unknown}")

if __name__ == "__main__":
    main()
