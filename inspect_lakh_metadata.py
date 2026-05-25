"""
inspect_lakh_metadata.py — diagnostic only, no files are moved or modified.

Walks data/human/lmd_matched/ and reports:
  - Every .json, .csv, .txt, and .h5 file found (filename, size, preview, fields)
  - The directory structure and MSD track ID naming convention embedded in the paths
  - Summary counts
"""

import os
import sys
import json
import csv

ROOT = os.path.join(os.path.dirname(__file__), "data", "human", "lmd_matched")
METADATA_EXTS = {".json", ".csv", ".txt", ".h5"}


# ── helpers ──────────────────────────────────────────────────────────────────

def fmt_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def inspect_json(path):
    with open(path, "r", errors="replace") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as e:
            print(f"    [JSON parse error: {e}]")
            return

    if isinstance(data, dict):
        keys = list(data.keys())
        print(f"  Type        : dict with {len(keys)} top-level keys")
        print(f"  Keys        : {keys[:20]}{'...' if len(keys) > 20 else ''}")
        print("  First 10 entries:")
        for i, (k, v) in enumerate(data.items()):
            if i >= 10:
                break
            print(f"    {k!r}: {str(v)[:120]}")

    elif isinstance(data, list):
        print(f"  Type        : list with {len(data)} entries")
        if data and isinstance(data[0], dict):
            print(f"  Fields      : {list(data[0].keys())}")
        print("  First 10 entries:")
        for i, item in enumerate(data[:10]):
            print(f"    [{i}] {str(item)[:120]}")

    else:
        print(f"  Type        : {type(data).__name__}")
        print(f"  Value       : {str(data)[:200]}")


def inspect_csv(path):
    with open(path, "r", errors="replace", newline="") as fh:
        reader = csv.reader(fh)
        rows = list(reader)

    if not rows:
        print("  [empty file]")
        return

    headers = rows[0]
    print(f"  Total rows  : {len(rows)} (including header)")
    print(f"  Columns ({len(headers)}): {headers}")
    print("  First 10 data rows:")
    for i, row in enumerate(rows[1:11]):
        print(f"    [{i+1}] {row}")


def inspect_txt(path):
    with open(path, "r", errors="replace") as fh:
        lines = fh.readlines()
    print(f"  Total lines : {len(lines)}")
    print("  First 10 lines:")
    for i, line in enumerate(lines[:10]):
        print(f"    {i+1}: {line.rstrip()}")


def inspect_h5(path):
    try:
        import h5py  # type: ignore
    except ImportError:
        print("  [h5py not installed — cannot inspect .h5 file]")
        return
    with h5py.File(path, "r") as f:
        def _walk(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"    Dataset: {name}  shape={obj.shape}  dtype={obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"    Group:   {name}/")
        print("  Contents:")
        f.visititems(_walk)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    if not os.path.isdir(ROOT):
        print(f"ERROR: directory not found: {ROOT}", file=sys.stderr)
        sys.exit(1)

    print("=" * 72)
    print(f"Scanning: {ROOT}")
    print("=" * 72)

    # ── 1. Count total MIDIs and collect MSD track IDs ───────────────────────
    total_mid = 0
    track_ids = set()
    artists_from_path = []

    for dirpath, dirnames, filenames in os.walk(ROOT):
        # The immediate parent of MIDI files is the MSD track ID directory
        # Path structure: .../<L1>/<L2>/<L3>/<TRACK_ID>/<hash>.mid
        for fname in filenames:
            if fname.lower().endswith(".mid"):
                total_mid += 1
                # The directory containing the .mid file is the track ID
                track_id = os.path.basename(dirpath)
                if track_id.startswith("TR"):
                    track_ids.add(track_id)

    print(f"\nMIDI files found : {total_mid:,}")
    print(f"Unique MSD Track IDs: {len(track_ids):,}")
    if track_ids:
        sample_ids = sorted(track_ids)[:10]
        print(f"Sample track IDs : {sample_ids}")

    # ── 2. Search for metadata files (.json, .csv, .txt, .h5) ───────────────
    print("\n" + "=" * 72)
    print("Searching for metadata files (.json, .csv, .txt, .h5)...")
    print("=" * 72)

    found = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in METADATA_EXTS:
                full_path = os.path.join(dirpath, fname)
                found.append(full_path)

    if not found:
        print("\n  *** No .json, .csv, .txt, or .h5 files found under lmd_matched/ ***")
        print()
        print("  The lmd_matched tree contains ONLY .mid files.")
        print("  Artist/title metadata is encoded in the MSD Track ID directory names")
        print("  (e.g. TRXXXXXX128F...) and requires an external MSD summary file")
        print("  (msd_summary_file.h5 or track_metadata.db) to resolve.")
    else:
        print(f"\n  Found {len(found)} metadata file(s):\n")
        for path in found:
            size = os.path.getsize(path)
            rel  = os.path.relpath(path, ROOT)
            print("-" * 72)
            print(f"  File    : {rel}")
            print(f"  Size    : {fmt_bytes(size)}")
            ext = os.path.splitext(path)[1].lower()
            if ext == ".json":
                inspect_json(path)
            elif ext == ".csv":
                inspect_csv(path)
            elif ext == ".txt":
                inspect_txt(path)
            elif ext == ".h5":
                inspect_h5(path)
            print()

    # ── 3. Also check one level above (data/human/) for companion files ──────
    parent = os.path.dirname(ROOT)
    print("\n" + "=" * 72)
    print(f"Checking sibling directories of lmd_matched/ for metadata: {parent}")
    print("=" * 72)

    sibling_meta = []
    for fname in os.listdir(parent):
        fpath = os.path.join(parent, fname)
        if os.path.isfile(fpath):
            ext = os.path.splitext(fname)[1].lower()
            if ext in METADATA_EXTS:
                sibling_meta.append(fpath)

    if not sibling_meta:
        print("  None found.")
    else:
        for path in sibling_meta:
            size = os.path.getsize(path)
            print(f"  {os.path.basename(path)}  ({fmt_bytes(size)})")

    # ── 4. Directory structure summary ───────────────────────────────────────
    print("\n" + "=" * 72)
    print("Directory structure (first 3 levels under lmd_matched/):")
    print("=" * 72)
    print(f"  lmd_matched/")
    l1_dirs = sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)))
    print(f"    ├── {len(l1_dirs)} L1 dirs: {l1_dirs[:8]}{'...' if len(l1_dirs) > 8 else ''}")
    if l1_dirs:
        l1_sample = os.path.join(ROOT, l1_dirs[0])
        l2_dirs = sorted(d for d in os.listdir(l1_sample) if os.path.isdir(os.path.join(l1_sample, d)))
        print(f"    └── e.g. {l1_dirs[0]}/")
        print(f"        ├── {len(l2_dirs)} L2 dirs: {l2_dirs[:8]}{'...' if len(l2_dirs) > 8 else ''}")
        if l2_dirs:
            l2_sample = os.path.join(l1_sample, l2_dirs[0])
            l3_dirs = sorted(d for d in os.listdir(l2_sample) if os.path.isdir(os.path.join(l2_sample, d)))
            print(f"        └── e.g. {l2_dirs[0]}/")
            print(f"            ├── {len(l3_dirs)} L3 dirs: {l3_dirs[:4]}{'...' if len(l3_dirs) > 4 else ''}")
            if l3_dirs:
                l3_sample = os.path.join(l2_sample, l3_dirs[0])
                track_id_dirs = sorted(d for d in os.listdir(l3_sample) if os.path.isdir(os.path.join(l3_sample, d)))
                print(f"            └── e.g. {l3_dirs[0]}/")
                print(f"                ├── {len(track_id_dirs)} MSD track ID dir(s): {track_id_dirs[:2]}")
                if track_id_dirs:
                    tid_sample = os.path.join(l3_sample, track_id_dirs[0])
                    mid_files = [f for f in os.listdir(tid_sample) if f.endswith(".mid")]
                    print(f"                └── e.g. {track_id_dirs[0]}/")
                    print(f"                    └── {len(mid_files)} .mid file(s): {mid_files[:3]}")

    print()
    print("=" * 72)
    print("Conclusion:")
    print("  The lmd_matched dataset is keyed by MSD Track IDs (TRXXXXXX...).")
    print("  No artist/title metadata is bundled in this directory.")
    print("  To filter by artist you would need one of:")
    print("    - msd_summary_file.h5  (Million Song Dataset summary)")
    print("    - match_scores.json    (LMD official metadata, ~120 MB)")
    print("    - track_metadata.db    (SQLite export of MSD)")
    print("  Available at: http://millionsongdataset.com/pages/getting-the-data/")
    print("=" * 72)


if __name__ == "__main__":
    main()
