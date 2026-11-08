#!/usr/bin/env python3
"""Score a backtest submission and convert the metrics into points.

Since AI 2026 - Turku Energia challenge: 10-hour imbalance price forecast.
Starter package file 9, the scoring script (decision D14). Run the validation
script validate_submission.py first; this script performs only the minimum
checks that the metrics need.

Mode 1 - metrics for one submission:

  python score.py --submission forecast_backtest.csv --actuals actuals_validation.csv \
      --spike-margin 50 --output-dir results/

  Writes score_summary.csv, score_summary.md and score_by_horizon.csv.

Mode 2 - points for all scored submissions (after every submission has been scored):

  python score.py --points results/ --baseline results/baseline/score_summary.csv \
      --output results/points_summary.csv

  Reads results/<team_id>/score_summary.csv for every team folder (the baseline
  folder is excluded) and writes points_summary.csv.

Metrics (lower is better for every scored metric):
  accuracy        MAE of p50 against the imbalance price
  calibration     mean of the p10 and p90 pinball losses
  coverage        share of actuals inside [p10, p90] (supporting metric, reported for information)
  exceptional     mean of the Brier scores of p_spike_high and p_spike_low; the actual
                  quarter-hour is high when imbalance_price > spot_price + margin and
                  low when imbalance_price < spot_price - margin (D5)
  direction       three-class Brier score of p_up, p_down and p_none (range 0-2)

Pinball loss for quantile q: mean(max(q * (y - yhat), (q - 1) * (y - yhat))).
Two-class Brier: mean((p - o)^2). Three-class Brier: mean(sum_k (p_k - o_k)^2).

Points (D14), for each automatic sub-criterion k:
  points_k = weight_k * clip((B_k - T_k) / (B_k - best_k), 0, 1)
  B = baseline metric, T = team metric, best = best metric among the teams.
  If no team beats the baseline by more than the tolerance
  (B_k - best_k <= tolerance x |B_k|, default tolerance 1e-6), every team gets 0 points for k.
There are no minimum requirements (D16): MAE versus the baseline and the
P10-P90 coverage are reported for information only.

Actuals file columns: delivery_start_utc, imbalance_price, spot_price,
direction (up, down or none). Other columns (e.g. first_usable_origin_utc)
are ignored. Source of the direction: Fingrid dataset 369, dominating mFRR
direction (1 = up, -1 = down, 0 = none; decision D17).
Each metric uses the rows for which the actual values it needs are available;
the row counts are reported per metric.

No network access is used.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCORER_VERSION = "0.1.1"
SCHEMA_VERSION = "1.0"

REQUIRED_COLUMNS = [
    "forecast_origin_utc",
    "delivery_start_utc",
    "step",
    "p10",
    "p50",
    "p90",
    "p_up",
    "p_down",
    "p_none",
    "p_spike_high",
    "p_spike_low",
]
NUMERIC_COLUMNS = ["p10", "p50", "p90", "p_up", "p_down", "p_none", "p_spike_high", "p_spike_low"]
ACTUALS_COLUMNS = ["delivery_start_utc", "imbalance_price", "spot_price", "direction"]
DIRECTIONS = ["up", "down", "none"]

# criterion name -> (metric column in score_summary.csv, CLI weight attribute)
CRITERIA = {
    "accuracy": ("mae_p50", "weight_accuracy"),
    "calibration": ("calibration_pinball_mean", "weight_calibration"),
    "exceptional_prices": ("brier_spike_mean", "weight_spikes"),
    "direction": ("brier_direction", "weight_direction"),
}

TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
_UTC_TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(Z|\+00:00)$")


class ScoringError(Exception):
    """Input problem that prevents scoring."""


# --------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------

def read_csv_text(path: Path) -> pd.DataFrame:
    """Read a CSV as text: BOM removed, CRLF accepted, empty cells as ''."""
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except Exception as exc:  # noqa: BLE001
        raise ScoringError(f"Cannot read {path}: {exc}") from exc


def parse_utc(values: pd.Series, name: str, path: Path, allow_empty: bool = False) -> pd.Series:
    text = values.fillna("").astype(str).str.strip()
    core = text.str.extract(_UTC_TIMESTAMP_RE)[0]
    parsed = pd.to_datetime(core, format="%Y-%m-%dT%H:%M:%S", errors="coerce", utc=True)
    invalid = parsed.isna() & ~((text == "") & allow_empty)
    if invalid.any():
        first = invalid.idxmax()
        raise ScoringError(
            f"{path.name}: {name} has {int(invalid.sum())} invalid UTC timestamp(s); "
            f"first '{values[first]}' on line {first + 2}. Expected YYYY-MM-DDTHH:MM:SSZ."
        )
    return parsed


def parse_number(values: pd.Series, name: str, path: Path, allow_empty: bool) -> pd.Series:
    text = values.fillna("").astype(str).str.strip()
    numbers = pd.to_numeric(text.where(text != ""), errors="coerce")
    bad = numbers.isna() & (text != "")
    bad |= ~np.isfinite(numbers.fillna(0.0))
    if not allow_empty:
        bad |= text == ""
    if bad.any():
        first = bad.idxmax()
        raise ScoringError(
            f"{path.name}: {name} has {int(bad.sum())} missing or non-numeric value(s); "
            f"first '{values[first]}' on line {first + 2}. Run validate_submission.py first."
        )
    return numbers.astype(float)


def load_submission(path: Path) -> pd.DataFrame:
    raw = read_csv_text(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ScoringError(f"{path.name}: required column(s) missing: {', '.join(missing)}. Run validate_submission.py first.")
    sub = pd.DataFrame(index=raw.index)
    sub["forecast_origin_utc"] = parse_utc(raw["forecast_origin_utc"], "forecast_origin_utc", path)
    sub["delivery_start_utc"] = parse_utc(raw["delivery_start_utc"], "delivery_start_utc", path)
    step = parse_number(raw["step"], "step", path, allow_empty=False)
    if (step != np.floor(step)).any() or (step < 1).any():
        raise ScoringError(f"{path.name}: step must be a positive integer. Run validate_submission.py first.")
    sub["step"] = step.astype(int)
    for column in NUMERIC_COLUMNS:
        sub[column] = parse_number(raw[column], column, path, allow_empty=False)
    if sub.duplicated(["forecast_origin_utc", "step"]).any():
        raise ScoringError(f"{path.name}: duplicate forecast_origin_utc + step rows. Run validate_submission.py first.")
    return sub


def load_actuals(path: Path) -> pd.DataFrame:
    raw = read_csv_text(path)
    missing = [c for c in ACTUALS_COLUMNS if c not in raw.columns]
    if missing:
        raise ScoringError(f"{path.name}: actuals column(s) missing: {', '.join(missing)}.")
    act = pd.DataFrame(index=raw.index)
    act["delivery_start_utc"] = parse_utc(raw["delivery_start_utc"], "delivery_start_utc", path)
    act["imbalance_price"] = parse_number(raw["imbalance_price"], "imbalance_price", path, allow_empty=True)
    act["spot_price"] = parse_number(raw["spot_price"], "spot_price", path, allow_empty=True)
    direction = raw["direction"].str.strip().str.lower()
    bad = ~direction.isin(DIRECTIONS + [""])
    if bad.any():
        first = bad.idxmax()
        raise ScoringError(
            f"{path.name}: direction must be up, down, none or empty; first invalid value "
            f"'{raw.at[first, 'direction']}' on line {first + 2}."
        )
    act["direction"] = direction.where(direction != "")
    if act["delivery_start_utc"].duplicated().any():
        raise ScoringError(f"{path.name}: duplicate delivery_start_utc values in the actuals file.")
    return act


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def pinball(y: pd.Series, yhat: pd.Series, q: float) -> float:
    diff = y - yhat
    return float(np.mean(np.maximum(q * diff, (q - 1.0) * diff)))


def _mean_or_nan(values: pd.Series) -> float:
    return float(values.mean()) if len(values) else math.nan


def compute_metrics(df: pd.DataFrame, spike_margin: float) -> dict:
    """Compute all metrics for a merged submission + actuals table."""
    price = df[df["imbalance_price"].notna()]
    y = price["imbalance_price"]
    metrics: dict = {"rows_scored_price": len(price)}
    if len(price):
        metrics["mae_p50"] = float((y - price["p50"]).abs().mean())
        metrics["pinball_p10"] = pinball(y, price["p10"], 0.10)
        metrics["pinball_p50"] = pinball(y, price["p50"], 0.50)
        metrics["pinball_p90"] = pinball(y, price["p90"], 0.90)
        metrics["calibration_pinball_mean"] = (metrics["pinball_p10"] + metrics["pinball_p90"]) / 2.0
        metrics["coverage_p10_p90"] = float(((y >= price["p10"]) & (y <= price["p90"])).mean())
    else:
        for key in ("mae_p50", "pinball_p10", "pinball_p50", "pinball_p90", "calibration_pinball_mean", "coverage_p10_p90"):
            metrics[key] = math.nan

    spike = df[df["imbalance_price"].notna() & df["spot_price"].notna()]
    metrics["rows_scored_spike"] = len(spike)
    high = (spike["imbalance_price"] > spike["spot_price"] + spike_margin).astype(float)
    low = (spike["imbalance_price"] < spike["spot_price"] - spike_margin).astype(float)
    metrics["brier_spike_high"] = _mean_or_nan((spike["p_spike_high"] - high) ** 2)
    metrics["brier_spike_low"] = _mean_or_nan((spike["p_spike_low"] - low) ** 2)
    metrics["brier_spike_mean"] = (metrics["brier_spike_high"] + metrics["brier_spike_low"]) / 2.0
    metrics["observed_spike_high_share"] = _mean_or_nan(high)
    metrics["observed_spike_low_share"] = _mean_or_nan(low)

    direction = df[df["direction"].notna()]
    metrics["rows_scored_direction"] = len(direction)
    squared = sum((direction[f"p_{k}"] - (direction["direction"] == k).astype(float)) ** 2 for k in DIRECTIONS)
    metrics["brier_direction"] = _mean_or_nan(squared) if len(direction) else math.nan
    return metrics


def score_submission(args: argparse.Namespace) -> int:
    submission_path = Path(args.submission)
    actuals_path = Path(args.actuals)
    output_dir = Path(args.output_dir)

    submission = load_submission(submission_path)
    actuals = load_actuals(actuals_path)
    merged = submission.merge(actuals, on="delivery_start_utc", how="left")
    steps_per_hour = 60 // args.step_minutes
    merged["horizon_hour"] = (merged["step"] - 1) // steps_per_hour + 1

    overall = compute_metrics(merged, args.spike_margin)
    summary = {
        "scorer_version": SCORER_VERSION,
        "schema_version": SCHEMA_VERSION,
        "submission_file": submission_path.name,
        "actuals_file": actuals_path.name,
        "spike_margin": args.spike_margin,
        "rows_in_submission": len(merged),
        "rows_skipped_price": len(merged) - overall["rows_scored_price"],
    }
    summary.update(overall)

    baseline_mae = math.nan
    if args.baseline_summary:
        baseline = read_csv_text(Path(args.baseline_summary))
        if "mae_p50" not in baseline.columns or len(baseline) != 1:
            raise ScoringError(f"{args.baseline_summary}: expected one row with the column mae_p50.")
        baseline_mae = float(baseline.at[0, "mae_p50"])
    summary["baseline_mae_p50"] = baseline_mae
    summary["mae_diff_vs_baseline"] = summary["mae_p50"] - baseline_mae
    summary["mae_ratio_vs_baseline"] = summary["mae_p50"] / baseline_mae if baseline_mae else math.nan

    by_horizon_rows = []
    for hour, group in merged.groupby("horizon_hour", sort=True):
        row = {"horizon_hour": int(hour), "step_first": int(group["step"].min()), "step_last": int(group["step"].max())}
        row.update(compute_metrics(group, args.spike_margin))
        by_horizon_rows.append(row)
    by_horizon = pd.DataFrame(by_horizon_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(output_dir / "score_summary.csv", index=False, encoding="utf-8", lineterminator="\n")
    by_horizon.to_csv(output_dir / "score_by_horizon.csv", index=False, encoding="utf-8", lineterminator="\n")
    (output_dir / "score_summary.md").write_text(render_markdown(summary, by_horizon), encoding="utf-8", newline="\n")

    print(render_markdown(summary, by_horizon))
    print(f"Wrote {output_dir / 'score_summary.csv'}")
    print(f"Wrote {output_dir / 'score_summary.md'}")
    print(f"Wrote {output_dir / 'score_by_horizon.csv'}")
    return 0


def fmt(value, digits: int = 4) -> str:
    if isinstance(value, float):
        return "n/a" if math.isnan(value) else f"{value:.{digits}f}"
    return str(value)


def render_markdown(summary: dict, by_horizon: pd.DataFrame) -> str:
    lines = [
        "# Score summary",
        "",
        f"- Scorer version: {summary['scorer_version']}",
        f"- Schema version: {summary['schema_version']}",
        f"- Submission: {summary['submission_file']}",
        f"- Actuals: {summary['actuals_file']}",
        f"- Spike margin: {fmt(float(summary['spike_margin']), 2)} EUR/MWh",
        f"- Rows in submission: {summary['rows_in_submission']}",
        f"- Rows scored (price metrics): {summary['rows_scored_price']}; skipped (no actual): {summary['rows_skipped_price']}",
        f"- Rows scored (exceptional prices): {summary['rows_scored_spike']}",
        f"- Rows scored (direction): {summary['rows_scored_direction']}",
        "",
        "## Overall results",
        "",
        "| Metric | Value | Use |",
        "|---|---|---|",
        f"| MAE (p50) | {fmt(summary['mae_p50'])} | Level 1: accuracy |",
        f"| Pinball loss p10 | {fmt(summary['pinball_p10'])} | |",
        f"| Pinball loss p90 | {fmt(summary['pinball_p90'])} | |",
        f"| Mean pinball loss p10 and p90 | {fmt(summary['calibration_pinball_mean'])} | Level 1: calibration |",
        f"| P10-P90 coverage | {fmt(summary['coverage_p10_p90'])} | supporting metric (target about 0.80) |",
        f"| Brier p_spike_high | {fmt(summary['brier_spike_high'])} | |",
        f"| Brier p_spike_low | {fmt(summary['brier_spike_low'])} | |",
        f"| Mean Brier, exceptional prices | {fmt(summary['brier_spike_mean'])} | Level 1: exceptional prices |",
        f"| Three-class Brier, direction | {fmt(summary['brier_direction'])} | Level 1: direction |",
        f"| Pinball loss p50 | {fmt(summary['pinball_p50'])} | for information |",
        f"| Observed share of high / low quarter-hours | {fmt(summary['observed_spike_high_share'])} / {fmt(summary['observed_spike_low_share'])} | for information |",
        f"| Baseline MAE (p50) | {fmt(summary['baseline_mae_p50'])} | for information |",
        f"| MAE difference vs baseline (team - baseline) | {fmt(summary['mae_diff_vs_baseline'])} | for information |",
        f"| MAE ratio vs baseline (team / baseline) | {fmt(summary['mae_ratio_vs_baseline'])} | for information |",
        "",
        "## Results by forecast horizon",
        "",
        "| Hour | Steps | Rows | MAE | Pinball p10/p90 mean | Coverage | Brier spikes | Brier direction |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, row in by_horizon.iterrows():
        lines.append(
            f"| {int(row['horizon_hour'])} | {int(row['step_first'])}-{int(row['step_last'])} | {int(row['rows_scored_price'])} "
            f"| {fmt(row['mae_p50'])} | {fmt(row['calibration_pinball_mean'])} | {fmt(row['coverage_p10_p90'])} "
            f"| {fmt(row['brier_spike_mean'])} | {fmt(row['brier_direction'])} |"
        )
    lines += [
        "",
        "Lower is better for MAE, pinball loss and Brier scores. Points are calculated with "
        "`score.py --points` after all submissions have been scored.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Points (D14)
# --------------------------------------------------------------------------

def read_summary(path: Path) -> dict:
    table = read_csv_text(path)
    if len(table) != 1:
        raise ScoringError(f"{path}: expected exactly one data row.")
    row = table.iloc[0].to_dict()
    for metric, _ in CRITERIA.values():
        if metric not in row:
            raise ScoringError(f"{path}: column {metric} is missing.")
        value = float(row[metric]) if row[metric] != "" else math.nan
        if not math.isfinite(value):
            raise ScoringError(f"{path}: {metric} is missing or not a number.")
        row[metric] = value
    return row


def no_team_beats_baseline(baseline: float, best: float, tolerance: float) -> bool:
    """True when B - best <= tolerance x |B| (decision D14, 31.10.2026)."""
    return baseline - best <= tolerance * abs(baseline)


def criterion_points(baseline: float, team: float, best: float, weight: float, tolerance: float) -> float:
    """weight * clip((B - T) / (B - best), 0, 1); 0 for everyone if B - best <= tolerance x |B|."""
    if no_team_beats_baseline(baseline, best, tolerance):
        return 0.0
    denominator = baseline - best
    return weight * min(max((baseline - team) / denominator, 0.0), 1.0)


def compute_points(args: argparse.Namespace) -> int:
    results_dir = Path(args.points)
    baseline_path = Path(args.baseline)
    baseline = read_summary(baseline_path)

    team_files = sorted(p for p in results_dir.glob("*/score_summary.csv")
                        if p.resolve() != baseline_path.resolve())
    if not team_files:
        raise ScoringError(f"No team score_summary.csv files found under {results_dir}.")
    teams = {p.parent.name: read_summary(p) for p in team_files}

    warnings = []
    for team_id, row in teams.items():
        if float(row["spike_margin"]) != float(baseline["spike_margin"]):
            raise ScoringError(f"{team_id}: spike margin {row['spike_margin']} differs from the baseline ({baseline['spike_margin']}).")
        if row.get("scorer_version") != baseline.get("scorer_version"):
            raise ScoringError(f"{team_id}: scorer version {row.get('scorer_version')} differs from the baseline ({baseline.get('scorer_version')}).")
        if row.get("actuals_file") != baseline.get("actuals_file"):
            raise ScoringError(f"{team_id}: actuals file {row.get('actuals_file')} differs from the baseline ({baseline.get('actuals_file')}).")
        for count in ("rows_scored_price", "rows_scored_spike", "rows_scored_direction"):
            if row.get(count) != baseline.get(count):
                warnings.append(f"{team_id}: {count} = {row.get(count)}, baseline {baseline.get(count)}.")

    weights = {name: getattr(args, attr) for name, (_, attr) in CRITERIA.items()}
    best = {name: min(row[metric] for row in teams.values()) for name, (metric, _) in CRITERIA.items()}

    records = []
    for team_id, row in teams.items():
        record = {"team_id": team_id}
        total = 0.0
        for name, (metric, _) in CRITERIA.items():
            points = criterion_points(baseline[metric], row[metric], best[name], weights[name], args.zero_tolerance)
            record[metric] = row[metric]
            record[f"points_{name}"] = points
            total += points
        record["points_automatic_total"] = total
        records.append(record)

    result = pd.DataFrame(records).sort_values("points_automatic_total", ascending=False, kind="stable")
    meta = {
        "scorer_version": SCORER_VERSION,
        "baseline_file": str(baseline_path),
        "spike_margin": baseline["spike_margin"],
    }
    for name, (metric, _) in CRITERIA.items():
        meta[f"baseline_{metric}"] = baseline[metric]
        meta[f"best_{metric}"] = best[name]
        meta[f"weight_{name}"] = weights[name]
    meta["zero_tolerance"] = args.zero_tolerance
    for key, value in meta.items():
        result[key] = value

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8", lineterminator="\n")

    print(f"Points summary (scorer {SCORER_VERSION}), {len(teams)} team(s), baseline: {baseline_path}")
    print(f"Weights: " + ", ".join(f"{k} {v:g}" for k, v in weights.items()))
    for name, (metric, _) in CRITERIA.items():
        print(f"  {name}: baseline {metric} = {baseline[metric]:.6f}, best = {best[name]:.6f}"
              + ("  (no team beats the baseline by more than the tolerance: 0 points for everyone)" if no_team_beats_baseline(baseline[metric], best[name], args.zero_tolerance) else ""))
    columns = ["team_id"] + [f"points_{n}" for n in CRITERIA] + ["points_automatic_total"]
    print(result[columns].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("Reasoning chain (20 points) and Level 2 are scored by the jury and combined in the evaluation record.")
    for warning in warnings:
        print(f"WARNING: {warning}")
    print(f"Wrote {output}")
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score a backtest submission against actuals (mode 1), or convert the scored "
            "metrics of all submissions into points (mode 2, --points). Decision D14."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Mode 1 example:\n"
            "  python score.py --submission forecast_backtest.csv --actuals actuals_validation.csv --spike-margin 50 --output-dir results/\n"
            "Mode 2 example:\n"
            "  python score.py --points results/ --baseline results/baseline/score_summary.csv --output results/points_summary.csv\n\n"
            "Points per sub-criterion: weight x clip((B - T) / (B - best), 0, 1);\n"
            "0 for every team if B - best <= zero-tolerance x |B|.\n"
            "Exit code: 0 = success, 1 = input error."
        ),
    )
    mode1 = parser.add_argument_group("mode 1: score one submission")
    mode1.add_argument("--submission", help="Path to forecast_backtest.csv (validate it first).")
    mode1.add_argument("--actuals", help="Path to the actuals file (actuals_validation.csv or actuals_test.csv).")
    mode1.add_argument("--spike-margin", type=float, default=50.0,
                       help="Exceptional price margin in EUR/MWh around the spot price (default: %(default)s, D5).")
    mode1.add_argument("--output-dir", help="Directory for score_summary.csv, score_summary.md and score_by_horizon.csv.")
    mode1.add_argument("--baseline-summary",
                       help="Optional baseline score_summary.csv; MAE versus the baseline is reported for information.")
    mode1.add_argument("--step-minutes", type=int, default=15,
                       help="Length of one step in minutes, used for the horizon hours (default: %(default)s).")

    mode2 = parser.add_argument_group("mode 2: convert metrics into points")
    mode2.add_argument("--points", metavar="RESULTS_DIR",
                       help="Directory containing <team_id>/score_summary.csv for every scored submission.")
    mode2.add_argument("--baseline", help="Baseline score_summary.csv (its folder is excluded from the teams).")
    mode2.add_argument("--output", help="Output CSV, e.g. results/points_summary.csv.")
    mode2.add_argument("--weight-accuracy", type=float, default=30.0, help="Weight of accuracy (default: %(default)s, D7).")
    mode2.add_argument("--weight-calibration", type=float, default=20.0, help="Weight of calibration (default: %(default)s, D7).")
    mode2.add_argument("--weight-spikes", type=float, default=20.0, help="Weight of exceptional prices (default: %(default)s, D7).")
    mode2.add_argument("--weight-direction", type=float, default=10.0, help="Weight of direction (default: %(default)s, D7).")
    mode2.add_argument("--zero-tolerance", type=float, default=1e-6,
                       help="Relative tolerance: if B - best <= tolerance x |B|, every team gets 0 points "
                            "for the sub-criterion (default: %(default)s, D14).")

    parser.add_argument("--version", action="version", version=f"%(prog)s {SCORER_VERSION} (schema {SCHEMA_VERSION})")
    args = parser.parse_args(argv)

    if args.points:
        if not (args.baseline and args.output):
            parser.error("--points requires --baseline and --output.")
        if args.submission or args.actuals:
            parser.error("--points cannot be combined with --submission or --actuals.")
        if args.zero_tolerance < 0:
            parser.error("--zero-tolerance must not be negative.")
    else:
        if not (args.submission and args.actuals and args.output_dir):
            parser.error("mode 1 requires --submission, --actuals and --output-dir (or use --points for mode 2).")
        if args.step_minutes <= 0 or 60 % args.step_minutes != 0:
            parser.error("--step-minutes must be a positive divisor of 60.")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return compute_points(args) if args.points else score_submission(args)
    except ScoringError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
