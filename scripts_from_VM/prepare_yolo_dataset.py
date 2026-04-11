import argparse
import hashlib
import json
import math
import sys
from collections import Counter, OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


VENDOR_PATH = Path(__file__).resolve().parents[1] / ".vendor" / "google-cloud-storage"
if VENDOR_PATH.exists():
    sys.path.insert(0, str(VENDOR_PATH))


DEFAULT_BUCKET = "seaqr-data"
DEFAULT_CATALOG = f"gs://{DEFAULT_BUCKET}/catalog/manifests/combined_manifest.jsonl"
DEFAULT_REGISTRY = f"gs://{DEFAULT_BUCKET}/catalog/registry/canonical_dataset_registry.json"
VALID_SPLITS = ("train", "val", "test")
VALID_MODES = ("count", "even", "ratio")
SOURCE_SPLIT_PREFERENCE = {
    "train": ("train", "val", "test"),
    "val": ("val", "test", "train"),
    "test": ("test", "val", "train"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a YOLO-ready prepared dataset from the canonical SEAQR catalog."
    )
    parser.add_argument("--name", required=True, help="Base name for the prepared dataset.")
    parser.add_argument(
        "--bucket",
        default=DEFAULT_BUCKET,
        help="Canonical bucket name. Default: seaqr-data.",
    )
    parser.add_argument(
        "--catalog",
        default=DEFAULT_CATALOG,
        help="Path or gs:// URI to combined_manifest.jsonl.",
    )
    parser.add_argument(
        "--registry",
        default=DEFAULT_REGISTRY,
        help="Path or gs:// URI to canonical_dataset_registry.json.",
    )
    parser.add_argument(
        "--mode",
        choices=VALID_MODES,
        default="even",
        help="Selection mode. Default: even.",
    )
    parser.add_argument(
        "--class",
        action="append",
        dest="class_specs",
        default=[],
        help="Count mode only. Format: class_name:train=<n|all>,val=<n|all>,test=<n|all>",
    )
    parser.add_argument(
        "--ratio",
        action="append",
        dest="ratios",
        default=[],
        help="Ratio mode only. Format: class_name=percent",
    )
    parser.add_argument(
        "--include-class",
        action="append",
        dest="include_classes",
        default=[],
        help="Even mode only. Repeat to include classes in even balancing.",
    )
    parser.add_argument(
        "--total-images",
        type=int,
        help="Required for even and ratio modes. Total target record count.",
    )
    parser.add_argument(
        "--split-ratio",
        default="train=80,val=10,test=10",
        help="Split ratio for even/ratio modes. Default: train=80,val=10,test=10",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        default=[],
        help="Optional source dataset filter. Repeatable.",
    )
    parser.add_argument(
        "--split",
        action="append",
        dest="source_splits",
        default=[],
        help="Optional source split filter. Repeatable.",
    )
    parser.add_argument(
        "--exclude-manifest",
        action="append",
        dest="exclude_manifests",
        default=[],
        help="Manifest path or gs:// URI containing records to exclude.",
    )
    parser.add_argument(
        "--class-order",
        action="append",
        dest="class_order",
        default=[],
        help="Optional explicit output YOLO class order. Repeatable. Do not include background.",
    )
    parser.add_argument(
        "--download-local",
        help="Base directory where the local YOLO-ready dataset will be created.",
    )
    parser.add_argument(
        "--upload-prepared",
        action="store_true",
        help="Upload prepared metadata files to gs://<bucket>/prepared/<dataset_name>/.",
    )
    parser.add_argument(
        "--timestamp-name",
        action="store_true",
        help="Append a timestamp to the dataset name.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic selection seed.")
    parser.add_argument(
        "--notes",
        default="",
        help="Optional free-text note written into summary.json and README.md.",
    )
    parser.add_argument(
        "--overwrite-local",
        action="store_true",
        help="Allow overwriting an existing local output directory.",
    )
    parser.add_argument(
        "--allow-shortfall",
        action="store_true",
        help="Allow proceeding when a class cannot meet the requested quota.",
    )
    parser.add_argument(
        "--min-available-fraction",
        type=float,
        default=1.0,
        help="Minimum achieved fraction of numeric requested records. Default: 1.0",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan and summarize the prepared dataset without downloading or uploading files.",
    )
    return parser.parse_args()


def parse_gs_uri(uri: str) -> Tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Unsupported URI: {uri}")
    remainder = uri[5:]
    bucket, blob = remainder.split("/", 1)
    return bucket, blob


def stable_rank(key: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp_suffix() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def split_ratio_dict(raw: str) -> Dict[str, int]:
    ratio = {}
    for part in raw.split(","):
        name, value = part.split("=", 1)
        name = name.strip()
        value = int(value.strip())
        if name not in VALID_SPLITS:
            raise SystemExit(f"Unsupported split in --split-ratio: {name}")
        ratio[name] = value
    missing = set(VALID_SPLITS) - set(ratio)
    if missing:
        raise SystemExit(f"--split-ratio must include train,val,test. Missing: {sorted(missing)}")
    if sum(ratio.values()) != 100:
        raise SystemExit("--split-ratio percentages must sum to 100.")
    return ratio


def distributed_counts(total: int, weights: Sequence[Tuple[str, float]]) -> Dict[str, int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    total_weight = sum(weight for _, weight in weights)
    if total_weight <= 0:
        raise ValueError("weights must sum to a positive value")
    raw = []
    allocated = 0
    for key, weight in weights:
        exact = total * (weight / total_weight)
        floor = math.floor(exact)
        raw.append((key, floor, exact - floor))
        allocated += floor
    remainder = total - allocated
    raw.sort(key=lambda item: (-item[2], item[0]))
    counts = {key: floor for key, floor, _ in raw}
    for idx in range(remainder):
        counts[raw[idx][0]] += 1
    return counts


def read_text(path_or_uri: str) -> str:
    if path_or_uri.startswith("gs://"):
        from google.cloud import storage

        client = storage.Client()
        bucket, blob = parse_gs_uri(path_or_uri)
        return client.bucket(bucket).blob(blob).download_as_text()
    return Path(path_or_uri).read_text(encoding="utf-8")


def load_json(path_or_uri: str):
    return json.loads(read_text(path_or_uri))


def load_jsonl(path_or_uri: str) -> List[dict]:
    text = read_text(path_or_uri)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=False) + "\n")


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_count_value(raw: str):
    if raw == "all":
        return "all"
    value = int(raw)
    if value < 0:
        raise SystemExit("Counts must be non-negative.")
    return value


def parse_class_spec(raw: str) -> Tuple[str, Dict[str, object]]:
    if ":" not in raw:
        raise SystemExit(f"Invalid --class spec: {raw}")
    class_name, payload = raw.split(":", 1)
    split_values = {}
    for part in payload.split(","):
        split_name, split_value = part.split("=", 1)
        split_name = split_name.strip()
        if split_name not in VALID_SPLITS:
            raise SystemExit(f"Invalid split in --class spec: {split_name}")
        split_values[split_name] = parse_count_value(split_value.strip())
    missing = set(VALID_SPLITS) - set(split_values)
    if missing:
        raise SystemExit(f"--class spec missing splits {sorted(missing)}: {raw}")
    return class_name.strip(), split_values


def parse_ratio_spec(raw: str) -> Tuple[str, float]:
    class_name, value = raw.split("=", 1)
    pct = float(value)
    if pct <= 0:
        raise SystemExit("Ratios must be positive.")
    return class_name.strip(), pct


def membership(record: dict, class_name: str) -> bool:
    if class_name == "background":
        return bool(record["is_background"])
    return class_name in record["classes_present"]


def requested_class_order(args) -> List[str]:
    if args.mode == "count":
        return [parse_class_spec(raw)[0] for raw in args.class_specs]
    if args.mode == "ratio":
        return [parse_ratio_spec(raw)[0] for raw in args.ratios]
    return list(args.include_classes)


def validate_mode_inputs(args):
    if args.mode == "count":
        if not args.class_specs:
            raise SystemExit("--mode count requires at least one --class argument.")
        if args.ratios or args.include_classes or args.total_images is not None:
            raise SystemExit("--mode count accepts only --class selections.")
    elif args.mode == "ratio":
        if not args.ratios or args.total_images is None:
            raise SystemExit("--mode ratio requires --total-images and at least one --ratio.")
        if args.class_specs or args.include_classes:
            raise SystemExit("--mode ratio accepts only --ratio selections.")
        total_pct = sum(parse_ratio_spec(raw)[1] for raw in args.ratios)
        if abs(total_pct - 100.0) > 1e-6:
            raise SystemExit("--ratio percentages must sum to 100.")
    elif args.mode == "even":
        if not args.include_classes or args.total_images is None:
            raise SystemExit("--mode even requires --total-images and at least one --include-class.")
        if args.class_specs or args.ratios:
            raise SystemExit("--mode even accepts only --include-class selections.")
    if not args.dry_run and not args.download_local:
        raise SystemExit("--download-local is required unless --dry-run is used.")
    if args.min_available_fraction <= 0 or args.min_available_fraction > 1:
        raise SystemExit("--min-available-fraction must be in the interval (0, 1].")


def build_requests(args) -> OrderedDict:
    requests = OrderedDict()
    if args.mode == "count":
        for raw in args.class_specs:
            class_name, split_values = parse_class_spec(raw)
            requests[class_name] = split_values
        return requests

    split_ratio = split_ratio_dict(args.split_ratio)
    split_totals = distributed_counts(args.total_images, list(split_ratio.items()))
    if args.mode == "even":
        classes = list(args.include_classes)
        per_split_requests = {class_name: {split: 0 for split in VALID_SPLITS} for class_name in classes}
        for split in VALID_SPLITS:
            counts = distributed_counts(split_totals[split], [(class_name, 1.0) for class_name in classes])
            for class_name, count in counts.items():
                per_split_requests[class_name][split] = count
        return OrderedDict((class_name, per_split_requests[class_name]) for class_name in classes)

    ratio_specs = OrderedDict(parse_ratio_spec(raw) for raw in args.ratios)
    requests = OrderedDict((class_name, {split: 0 for split in VALID_SPLITS}) for class_name in ratio_specs)
    class_totals = distributed_counts(args.total_images, list(ratio_specs.items()))
    for class_name, total in class_totals.items():
        split_counts = distributed_counts(total, list(split_ratio.items()))
        for split, count in split_counts.items():
            requests[class_name][split] = count
    return requests


def derive_output_class_order(request_order: List[str], explicit_order: List[str]) -> List[str]:
    requested_non_background = [name for name in request_order if name != "background"]
    if explicit_order:
        deduped = []
        seen = set()
        for name in explicit_order:
            if name == "background":
                raise SystemExit("Do not include background in --class-order.")
            if name in seen:
                raise SystemExit(f"Duplicate class in --class-order: {name}")
            seen.add(name)
            deduped.append(name)
        missing = [name for name in requested_non_background if name not in seen]
        if missing:
            raise SystemExit(
                f"--class-order is missing requested non-background classes: {missing}"
            )
        extras = [name for name in deduped if name not in requested_non_background]
        if extras:
            raise SystemExit(f"--class-order contains classes not requested: {extras}")
        return deduped
    deduped = []
    seen = set()
    for name in requested_non_background:
        if name not in seen:
            seen.add(name)
            deduped.append(name)
    return deduped


def filter_records(records: List[dict], datasets: List[str], splits: List[str], excluded_image_uris: set) -> List[dict]:
    dataset_filter = set(datasets) if datasets else None
    split_filter = set(splits) if splits else None
    filtered = []
    for record in records:
        if dataset_filter and record["dataset_id"] not in dataset_filter:
            continue
        if split_filter and record["split"] not in split_filter:
            continue
        if record["image_uri"] in excluded_image_uris:
            continue
        filtered.append(record)
    return filtered


def excluded_uris(paths: Sequence[str]) -> set:
    excluded = set()
    for item in paths:
        for record in load_jsonl(item):
            excluded.add(record["image_uri"])
    return excluded


def selection_tasks(requests: OrderedDict) -> List[Tuple[str, str, object]]:
    numeric = []
    all_tasks = []
    for class_name, split_values in requests.items():
        for split in VALID_SPLITS:
            target = split_values[split]
            task = (class_name, split, target)
            if target == "all":
                all_tasks.append(task)
            else:
                numeric.append(task)
    return numeric + all_tasks


def choose_records(records: List[dict], requests: OrderedDict, seed: int, allow_shortfall: bool):
    selected = []
    selected_uris = set()
    selection_counts = {class_name: {split: 0 for split in VALID_SPLITS} for class_name in requests}
    shortfalls = []

    for class_name, prepared_split, target in selection_tasks(requests):
        split_preference = {
            source_split: idx for idx, source_split in enumerate(SOURCE_SPLIT_PREFERENCE[prepared_split])
        }
        pool = []
        same_split_pool = []
        for source_split in SOURCE_SPLIT_PREFERENCE[prepared_split]:
            for record in records:
                if record["split"] != source_split:
                    continue
                if not membership(record, class_name):
                    continue
                if record["image_uri"] in selected_uris:
                    continue
                candidate = dict(record)
                candidate["source_split"] = record["split"]
                candidate["prepared_split"] = prepared_split
                pool.append(candidate)
                if source_split == prepared_split:
                    same_split_pool.append(candidate)
        pool.sort(
            key=lambda record: (
                split_preference[record["source_split"]],
                stable_rank(record["image_uri"], seed),
                record["image_uri"],
            )
        )
        same_split_pool.sort(
            key=lambda record: (
                stable_rank(record["image_uri"], seed),
                record["image_uri"],
            )
        )
        if target == "all":
            chosen = same_split_pool if same_split_pool else pool
        else:
            chosen = pool[:target]
            if len(chosen) < target:
                shortfalls.append(
                    {
                        "class_name": class_name,
                        "prepared_split": prepared_split,
                        "requested": target,
                        "selected": len(chosen),
                        "available_unselected": len(pool),
                    }
                )
                if not allow_shortfall:
                    raise SystemExit(
                        f"Not enough records for class={class_name} prepared_split={prepared_split}. "
                        f"Requested {target}, available {len(pool)}."
                    )
        for record in chosen:
            selected_uris.add(record["image_uri"])
            record["selection_class"] = class_name
            selected.append(record)
            selection_counts[class_name][prepared_split] += 1
    return selected, selection_counts, shortfalls


def load_registry_maps(registry: dict):
    dataset_to_local = {}
    dataset_to_class_to_local = defaultdict(dict)
    for dataset in registry["datasets"]:
        dataset_id = dataset["dataset_id"]
        local_to_global = dataset.get("global_classes", {})
        dataset_to_local[dataset_id] = local_to_global
        for local_id, class_name in local_to_global.items():
            dataset_to_class_to_local[dataset_id][class_name] = local_id
    return dataset_to_local, dataset_to_class_to_local


def output_file_stem(record: dict) -> str:
    image_uri = record["image_uri"]
    filename = Path(parse_gs_uri(image_uri)[1]).stem
    digest = hashlib.sha1(image_uri.encode("utf-8")).hexdigest()[:8]
    return f"{record['dataset_id']}__{filename}__{digest}"


def remap_label_text(record: dict, label_text: str, local_to_global: Dict[str, Dict[str, str]], output_class_ids: Dict[str, int]) -> str:
    lines = []
    dataset_map = local_to_global.get(record["dataset_id"], {})
    for raw in label_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        source_local_id = parts[0]
        class_name = dataset_map.get(source_local_id)
        if class_name not in output_class_ids:
            continue
        parts[0] = str(output_class_ids[class_name])
        lines.append(" ".join(parts))
    return "\n".join(lines) + ("\n" if lines else "")


def build_prepared_readme(summary: dict) -> str:
    lines = [
        f"# Prepared Dataset: {summary['dataset_name']}",
        "",
        f"Created at: {summary['created_at']}",
        f"Mode: {summary['mode']}",
        f"Bucket: {summary['bucket']}",
        f"Source catalog: {summary['source_catalog']}",
        f"Seed: {summary['seed']}",
        "",
        "Requested classes:",
    ]
    for class_name, payload in summary["selection_request"].items():
        lines.append(f"- {class_name}: {payload}")
    lines.extend(["", "Selected counts by class and split:"])
    for class_name, payload in summary["selected_counts"].items():
        lines.append(f"- {class_name}: {payload}")
    if summary.get("notes"):
        lines.extend(["", "Notes:", summary["notes"]])
    return "\n".join(lines) + "\n"


def summary_payload(
    resolved_name: str,
    args,
    requests: OrderedDict,
    selected_records: List[dict],
    selection_counts: Dict[str, Dict[str, int]],
    output_class_order: List[str],
    shortfalls: List[dict],
    local_output_dir: Optional[Path],
) -> dict:
    prepared_split_counts = dict(Counter(record["prepared_split"] for record in selected_records))
    source_split_counts = dict(Counter(record["source_split"] for record in selected_records))
    dataset_counts = dict(Counter(record["dataset_id"] for record in selected_records))
    requested_numeric = 0
    selected_numeric = 0
    for class_name, split_values in requests.items():
        for split, target in split_values.items():
            if isinstance(target, int):
                requested_numeric += target
                selected_numeric += selection_counts[class_name][split]
    return {
        "dataset_name": resolved_name,
        "created_at": now_utc_iso(),
        "mode": args.mode,
        "bucket": args.bucket,
        "source_catalog": args.catalog,
        "source_registry": args.registry,
        "seed": args.seed,
        "local_output_dir": str(local_output_dir) if local_output_dir else None,
        "prepared_uri": f"gs://{args.bucket}/prepared/{resolved_name}/" if args.upload_prepared else None,
        "selection_request": requests,
        "selected_counts": selection_counts,
        "selected_records": len(selected_records),
        "selected_prepared_split_counts": prepared_split_counts,
        "selected_source_split_counts": source_split_counts,
        "selected_dataset_counts": dataset_counts,
        "requested_numeric_records": requested_numeric,
        "selected_numeric_records": selected_numeric,
        "numeric_coverage_fraction": (selected_numeric / requested_numeric) if requested_numeric else None,
        "output_class_order": output_class_order,
        "output_class_id_map": {class_name: idx for idx, class_name in enumerate(output_class_order)},
        "dataset_filters": args.datasets,
        "source_split_filters": args.source_splits,
        "exclude_manifests": args.exclude_manifests,
        "allow_shortfall": args.allow_shortfall,
        "shortfalls": shortfalls,
        "notes": args.notes,
    }


def check_min_fraction(summary: dict, threshold: float):
    fraction = summary.get("numeric_coverage_fraction")
    if fraction is None:
        return
    if fraction < threshold:
        raise SystemExit(
            f"Selected numeric coverage fraction {fraction:.3f} is below "
            f"--min-available-fraction {threshold:.3f}."
        )


def materialize_local_dataset(
    output_dir: Path,
    selected_records: List[dict],
    output_class_order: List[str],
    local_to_global: Dict[str, Dict[str, str]],
):
    from google.cloud import storage

    client = storage.Client()
    output_class_ids = {class_name: idx for idx, class_name in enumerate(output_class_order)}
    split_to_source_image_uris = defaultdict(list)
    split_to_source_label_uris = defaultdict(list)
    manifest_lines = []

    for record in selected_records:
        split = record["prepared_split"]
        stem = output_file_stem(record)
        image_bucket, image_blob = parse_gs_uri(record["image_uri"])
        image_suffix = Path(image_blob).suffix or ".jpg"
        local_image = output_dir / "images" / split / f"{stem}{image_suffix}"
        local_label = output_dir / "labels" / split / f"{stem}.txt"
        local_image.parent.mkdir(parents=True, exist_ok=True)
        local_label.parent.mkdir(parents=True, exist_ok=True)

        client.bucket(image_bucket).blob(image_blob).download_to_filename(str(local_image))
        split_to_source_image_uris[split].append(record["image_uri"])

        if record["label_uri"]:
            label_bucket, label_blob = parse_gs_uri(record["label_uri"])
            label_text = client.bucket(label_bucket).blob(label_blob).download_as_text()
            remapped = remap_label_text(record, label_text, local_to_global, output_class_ids)
            local_label.write_text(remapped, encoding="utf-8")
            split_to_source_label_uris[split].append(record["label_uri"])
        else:
            local_label.write_text("", encoding="utf-8")

        manifest_record = dict(record)
        manifest_record["prepared_local_image"] = str(local_image)
        manifest_record["prepared_local_label"] = str(local_label)
        manifest_record["prepared_output_stem"] = stem
        manifest_lines.append(manifest_record)

    return manifest_lines, split_to_source_image_uris, split_to_source_label_uris


def data_yaml_text(output_dir: Path, output_class_order: List[str]) -> str:
    names_block = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(output_class_order))
    return (
        f"path: {output_dir}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        f"names:\n{names_block}\n"
    )


def upload_prepared_metadata(bucket: str, resolved_name: str, output_dir: Path):
    from google.cloud import storage

    client = storage.Client()
    target_bucket = client.bucket(bucket)
    upload_files = [
        "data.yaml",
        "manifest.jsonl",
        "summary.json",
        "README.md",
        "train_images.txt",
        "train_labels.txt",
        "val_images.txt",
        "val_labels.txt",
        "test_images.txt",
        "test_labels.txt",
    ]
    for relative in upload_files:
        path = output_dir / relative
        if not path.exists():
            continue
        target_blob = target_bucket.blob(f"prepared/{resolved_name}/{relative}")
        target_blob.upload_from_filename(str(path))


def ensure_local_output(args, resolved_name: str) -> Optional[Path]:
    if not args.download_local:
        return None
    output_dir = Path(args.download_local) / resolved_name
    if output_dir.exists():
        if not args.overwrite_local:
            raise SystemExit(
                f"Local output directory already exists: {output_dir}. "
                "Use --overwrite-local to replace it."
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main():
    args = parse_args()
    validate_mode_inputs(args)

    request_order = requested_class_order(args)
    output_class_order = derive_output_class_order(request_order, args.class_order)
    resolved_name = f"{args.name}_{timestamp_suffix()}" if args.timestamp_name else args.name

    records = load_jsonl(args.catalog)
    registry = load_json(args.registry)
    excluded = excluded_uris(args.exclude_manifests)
    filtered = filter_records(records, args.datasets, args.source_splits, excluded)
    requests = build_requests(args)
    selected, selection_counts, shortfalls = choose_records(
        filtered, requests, args.seed, args.allow_shortfall
    )

    local_output_dir = ensure_local_output(args, resolved_name) if not args.dry_run else None
    summary = summary_payload(
        resolved_name,
        args,
        requests,
        selected,
        selection_counts,
        output_class_order,
        shortfalls,
        local_output_dir,
    )
    check_min_fraction(summary, args.min_available_fraction)

    print(json.dumps(summary, indent=2, sort_keys=False))

    if args.dry_run:
        return

    if not output_class_order:
        raise SystemExit("Prepared dataset contains no non-background classes. YOLO training would be invalid.")

    local_to_global, _ = load_registry_maps(registry)
    manifest_records, split_to_image_uris, split_to_label_uris = materialize_local_dataset(
        local_output_dir, selected, output_class_order, local_to_global
    )

    write_json(local_output_dir / "summary.json", summary)
    write_jsonl(local_output_dir / "manifest.jsonl", manifest_records)
    write_text(local_output_dir / "data.yaml", data_yaml_text(local_output_dir, output_class_order))
    write_text(local_output_dir / "README.md", build_prepared_readme(summary))

    for split in VALID_SPLITS:
        write_text(
            local_output_dir / f"{split}_images.txt",
            "\n".join(split_to_image_uris.get(split, [])) + ("\n" if split_to_image_uris.get(split) else ""),
        )
        write_text(
            local_output_dir / f"{split}_labels.txt",
            "\n".join(split_to_label_uris.get(split, [])) + ("\n" if split_to_label_uris.get(split) else ""),
        )

    if args.upload_prepared:
        upload_prepared_metadata(args.bucket, resolved_name, local_output_dir)

    print(f"Prepared dataset written to {local_output_dir}")
    if args.upload_prepared:
        print(f"Prepared metadata uploaded to gs://{args.bucket}/prepared/{resolved_name}/")


if __name__ == "__main__":
    main()
