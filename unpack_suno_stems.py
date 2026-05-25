"""
unpack_suno_stems.py — unpack Suno stem zips into stems/<track_name>/.

Source: ~/Suno Stems/*.zip
Naming rules:
    "Suncream Confetti Stems.zip"    → stems/Suncream Confetti/
    "Suncream Confetti Stems (1).zip" → stems/Suncream Confetti_v2/

Does not move or delete the original zip files.
"""

import os
import re
import sys
import zipfile

HOME       = os.path.expanduser("~")
ZIPS_DIR   = os.path.join(HOME, "Downloads", "Suno Stems")
STEMS_ROOT = os.path.join(os.path.dirname(__file__), "stems")

PREVIEW_FOLDERS = 2   # print internal structure of this many folders


def zip_to_track_name(filename: str) -> str:
    """
    "Suncream Confetti Stems.zip"     → "Suncream Confetti"
    "Suncream Confetti Stems (1).zip" → "Suncream Confetti_v2"
    """
    name = filename
    if name.lower().endswith(".zip"):
        name = name[:-4]
    # Detect and remove " (1)" BEFORE stripping " Stems", so the order doesn't matter.
    is_v2 = bool(re.search(r"\s+\(1\)$", name))
    if is_v2:
        name = re.sub(r"\s+\(1\)$", "", name)
    # Strip " Stems" (or " stems") suffix.
    name = re.sub(r"\s+[Ss]tems$", "", name).strip()
    if is_v2:
        name += "_v2"
    return name


def main() -> None:
    if not os.path.isdir(ZIPS_DIR):
        print(f"ERROR: source directory not found: {ZIPS_DIR}", file=sys.stderr)
        sys.exit(1)

    zip_files = sorted(
        f for f in os.listdir(ZIPS_DIR)
        if f.lower().endswith(".zip")
    )

    if not zip_files:
        print(f"No .zip files found in {ZIPS_DIR}")
        sys.exit(0)

    print(f"Found {len(zip_files)} zip file(s) in {ZIPS_DIR}\n")

    os.makedirs(STEMS_ROOT, exist_ok=True)

    unpacked:  list[str] = []
    skipped:   list[str] = []
    failed:    list[str] = []

    for fname in zip_files:
        track_name  = zip_to_track_name(fname)
        out_dir     = os.path.join(STEMS_ROOT, track_name)
        zip_path    = os.path.join(ZIPS_DIR, fname)

        if os.path.isdir(out_dir) and os.listdir(out_dir):
            print(f"  SKIP  {fname!r:50s} → already exists: {track_name}/")
            skipped.append(track_name)
            continue

        try:
            os.makedirs(out_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(out_dir)
            print(f"  OK    {fname!r:50s} → stems/{track_name}/")
            unpacked.append(track_name)
        except Exception as exc:
            print(f"  FAIL  {fname!r}: {exc}")
            failed.append(fname)

    # ── summary ───────────────────────────────────────────────────────────────
    SEP = "=" * 60
    print(f"\n{SEP}")
    print("SUMMARY")
    print(SEP)
    print(f"  Unpacked : {len(unpacked)}")
    print(f"  Skipped  : {len(skipped)}  (already existed)")
    print(f"  Failed   : {len(failed)}")
    print(f"\n  Output folders in stems/:")
    all_folders = sorted(unpacked + skipped)
    for name in all_folders:
        print(f"    stems/{name}/")

    # ── preview internal structure ────────────────────────────────────────────
    preview_targets = all_folders[:PREVIEW_FOLDERS]
    if preview_targets:
        print(f"\n{SEP}")
        print(f"INTERNAL STRUCTURE (first {len(preview_targets)} folder(s))")
        print(SEP)
        for track_name in preview_targets:
            track_dir = os.path.join(STEMS_ROOT, track_name)
            print(f"\n  stems/{track_name}/")
            try:
                entries = sorted(os.listdir(track_dir))
                for entry in entries:
                    full = os.path.join(track_dir, entry)
                    size_kb = os.path.getsize(full) / 1024
                    # Mark nested dirs
                    if os.path.isdir(full):
                        print(f"    [dir]  {entry}/")
                        for sub in sorted(os.listdir(full)):
                            sub_path = os.path.join(full, sub)
                            sub_kb = os.path.getsize(sub_path) / 1024
                            print(f"             {sub}  ({sub_kb:.0f} KB)")
                    else:
                        print(f"    {entry}  ({size_kb:.0f} KB)")
            except Exception as exc:
                print(f"    (could not list: {exc})")


if __name__ == "__main__":
    main()
