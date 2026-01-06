"""
ip_parse.py — minimal AUC Sedimentation Equilibrium (.IPn) parser + preview/trim/plot.

Usage examples:
  python ip_parse.py average_18k_pos.IP1
  python ip_parse.py average_18k_pos.IP1 --drop-head 2 --drop-tail 2 --plot --outdir outputs/
  python ip_parse.py average_18k_pos.IP1 --min-radius 6.72 --max-radius 7.24 --save-csv --plot

Outputs (if requested): CSVs and PNG plot in --outdir (default: ./ip_outputs)
"""

import os
import re
import argparse
from typing import Dict, Tuple, Optional

import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Helpers
# ----------------------------
def _safe_float(tok: str, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(tok)
    except Exception:
        return default

def _safe_int(tok: str, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(float(tok))
    except Exception:
        return default


# ----------------------------
# Core parser
# ----------------------------
def parse_ip_file(path: str) -> Tuple[Dict, pd.DataFrame]:
    """
    Parse an AUC Sedimentation Equilibrium .IP[n] file.

    Returns:
        meta: dict with header info (sample_name, rotor_position, temperature_c, rpm, rpm_1000x, raw_header_line)
        df: pandas DataFrame with columns ['radius_cm', 'fringe']
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        # keep non-empty lines only
        lines = [ln.strip() for ln in f.readlines() if ln.strip() != ""]

    if len(lines) < 3:
        raise ValueError("File does not look like a valid .IP file (need >= 3 non-empty lines).")

    # Line 1: sample name
    sample_name = lines[0]

    # Line 2: metadata line (instrument-specific tokens)
    header_line = lines[1]
    header_tokens = re.split(r"\s+", header_line)

    rotor_position = None
    temperature_c = None
    rpm = None
    rpm_1000x = None

    # Typical format per supervisor: "P [pos] [tempC] [rpm] [...] 655 [rpm/1000]"
    if len(header_tokens) >= 4 and header_tokens[0].upper().startswith("P"):
        rotor_position = _safe_int(header_tokens[1])
        temperature_c = _safe_float(header_tokens[2])
        rpm = _safe_int(header_tokens[3])
        # Try to find trailing "1000× rpm" (often after a token '655')
        if "655" in header_tokens:
            rpm_1000x = _safe_int(header_tokens[-1])
    else:
        # Fallback: infer by ranges
        nums = [_safe_float(t) for t in header_tokens if re.match(r"^-?\d+(\.\d+)?$", t)]
        for val in nums:
            if temperature_c is None and val is not None and 0 <= val <= 50:
                temperature_c = val
            if rpm is None and val is not None and 1000 <= val <= 100000:
                rpm = int(val)

    meta = {
        "sample_name": sample_name,
        "rotor_position": rotor_position,
        "temperature_c": temperature_c,
        "rpm": rpm,
        "rpm_1000x": rpm_1000x,
        "raw_header_line": header_line,
    }

    # Lines 3+: radius, fringe
    data_rows = []
    for ln in lines[2:]:
        parts = re.split(r"\s+", ln)
        if len(parts) >= 2:
            r = _safe_float(parts[0])
            conc = _safe_float(parts[1])
            if r is not None and conc is not None:
                data_rows.append([r, conc])

    if len(data_rows) == 0:
        raise ValueError("No numeric data rows found after header.")

    df = pd.DataFrame(data_rows, columns=["radius_cm", "fringe"])
    return meta, df


# ----------------------------
# Trimming utilities
# ----------------------------
def top_and_tail(df: pd.DataFrame, drop_head: int = 0, drop_tail: int = 0) -> pd.DataFrame:
    """
    Drop N rows from start (head) and end (tail), returning a copy.
    """
    n = len(df)
    head = drop_head if drop_head >= 0 else 0
    tail = drop_tail if drop_tail >= 0 else 0
    if head + tail >= n:
        return df.iloc[0:0].copy()
    return df.iloc[head:n - tail].reset_index(drop=True)

def trim_by_radius(df: pd.DataFrame,
                   min_radius: Optional[float] = None,
                   max_radius: Optional[float] = None) -> pd.DataFrame:
    """
    Keep only rows with min_radius <= radius_cm <= max_radius.
    If a bound is None, it is ignored.
    """
    out = df
    if min_radius is not None:
        out = out[out["radius_cm"] >= min_radius]
    if max_radius is not None:
        out = out[out["radius_cm"] <= max_radius]
    return out.reset_index(drop=True)


# ----------------------------
# Plot
# ----------------------------
def save_profile_plot(df: pd.DataFrame, meta: Dict, out_path: str, title_suffix: str = "") -> None:
    plt.figure()
    plt.plot(df["radius_cm"], df["fringe"], linestyle="-")
    plt.xlabel("Radius (cm)")
    plt.ylabel("Fringe (a.u.)")
    ttl = meta.get("sample_name") or "SE Profile"
    if title_suffix:
        ttl = f"{ttl} — {title_suffix}"
    plt.title(ttl)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Parse and preview AUC SE .IP[n] files.")
    ap.add_argument("file", help="Path to .IP[n] file")
    ap.add_argument("--drop-head", type=int, default=0, help="Drop N rows from the start")
    ap.add_argument("--drop-tail", type=int, default=0, help="Drop N rows from the end")
    ap.add_argument("--min-radius", type=float, default=None, help="Keep rows with radius >= this")
    ap.add_argument("--max-radius", type=float, default=None, help="Keep rows with radius <= this")
    ap.add_argument("--outdir", default="ip_outputs", help="Directory to write outputs")
    ap.add_argument("--save-csv", action="store_true", help="Save raw/trimmed CSVs")
    ap.add_argument("--plot", action="store_true", help="Save a PNG plot of the (trimmed) profile")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    meta, df = parse_ip_file(args.file)

    # Apply trimming
    df2 = top_and_tail(df, args.drop_head, args.drop_tail)
    df3 = trim_by_radius(df2, args.min_radius, args.max_radius)

    # Console summary
    print("\n== Metadata ==")
    for k, v in meta.items():
        print(f"{k}: {v}")
    print("\n== Data summary ==")
    print(f"rows_raw: {len(df)}, rows_after_trim: {len(df3)}")
    if len(df) > 0:
        print(f"radius_min_raw: {df['radius_cm'].min():.5f}  radius_max_raw: {df['radius_cm'].max():.5f}")
    if len(df3) > 0:
        print(f"radius_min_trim: {df3['radius_cm'].min():.5f}  radius_max_trim: {df3['radius_cm'].max():.5f}")

    # Save CSVs if requested
    if args.save_csv:
        raw_csv = os.path.join(args.outdir, "parsed_raw.csv")
        trim_csv = os.path.join(args.outdir, "parsed_trimmed.csv")
        df.to_csv(raw_csv, index=False)
        df3.to_csv(trim_csv, index=False)
        print(f"\nSaved: {raw_csv}")
        print(f"Saved: {trim_csv}")

    # Plot if requested
    if args.plot:
        plot_path = os.path.join(args.outdir, "profile_trimmed.png")
        save_profile_plot(df3, meta, plot_path, title_suffix="Trimmed")
        print(f"Saved plot: {plot_path}")

    # Also drop a tiny parse log
    log_path = os.path.join(args.outdir, "parse_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("IP file parse summary\n")
        for k, v in meta.items():
            f.write(f"{k}: {v}\n")
        f.write(f"rows_raw: {len(df)}\nrows_after_trim: {len(df3)}\n")
    print(f"Saved log: {log_path}")


if __name__ == "__main__":
    main()
