#!/usr/bin/env python3
"""Validate the format of a backtest submission (forecast_backtest.csv).

Since AI 2026 - Turku Energia challenge: 10-hour imbalance price forecast.
Starter package file 8, the validation script. It checks the structure and
internal consistency of the submission only. It does not evaluate forecast
quality (that is score.py) and it cannot detect a breach of the data usage
rule (first_usable_origin_utc).

Rules (starter package specification, section 7.1):
  1. All 13 columns are present and the values have the schema types.
  2. Every forecast origin in forecast_origins.csv is present with exactly
     the expected number of steps; no unexpected forecast origins.
  3. No duplicate forecast_origin_utc + step combinations.
  4. No empty cells in required columns (data_cutoff_utc and top_drivers are
     optional and may be empty).
  5. p10 <= p50 <= p90, probabilities between 0 and 1, and
     p_up + p_down + p_none = 1 within the tolerance.
  6. p_spike_high + p_spike_low <= 1 within the tolerance.
  7. If data_cutoff_utc is given, data_cutoff_utc <= forecast_origin_utc.
  8. step is an integer 1..N, delivery_start_utc = origin + (step - 1) x step
     length, and timestamps are UTC ISO 8601 timestamps.

Timestamps: YYYY-MM-DDTHH:MM:SS followed by "Z" or "+00:00". Both UTC forms are
accepted and normalised to the "Z" form internally. Timestamps without a time
zone, with another UTC offset or with fractional seconds are rejected.

File handling: UTF-8, comma separator, dot as the decimal separator. A UTF-8
byte order mark is removed and CRLF line endings are accepted.

Exit code 0 when all checks pass (warnings allowed), otherwise 1.
No network access is used.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

VALIDATOR_VERSION = "0.1.0"
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
OPTIONAL_COLUMNS = ["data_cutoff_utc", "top_drivers"]
ALL_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS
QUANTILE_COLUMNS = ["p10", "p50", "p90"]
DIRECTION_COLUMNS = ["p_up", "p_down", "p_none"]
SPIKE_COLUMNS = ["p_spike_high", "p_spike_low"]
PROBABILITY_COLUMNS = DIRECTION_COLUMNS + SPIKE_COLUMNS
NUMERIC_COLUMNS = QUANTILE_COLUMNS + PROBABILITY_COLUMNS
TIMESTAMP_COLUMNS = ["forecast_origin_utc", "delivery_start_utc", "data_cutoff_utc"]

TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
_UTC_TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(Z|\+00:00)$")
_FRACTION_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[.,]\d+")
_OFFSET_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?[+-]\d{2}:?\d{2}$")
_NAIVE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2})?$")
_INTEGER_RE = re.compile(r"^[0-9]+$")
_NUMBER_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")

GROUPS = [
    "File",
    "Required columns",
    "Data types",
    "Forecast origins",
    "Rows",
    "Duplicate rows",
    "Missing required values",
    "Steps",
    "Timestamps",
    "Quantiles",
    "Probabilities",
    "Direction probabilities",
    "Spike probabilities",
    "Data cut-off rule",
]


@dataclass
class Finding:
    level: str  # "ERROR" or "WARNING"
    code: str
    group: str
    lines: list[str]


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    notes: dict[str, str] = field(default_factory=dict)

    def error(self, code: str, group: str, *lines: str) -> None:
        self.findings.append(Finding("ERROR", code, group, list(lines)))

    def warning(self, code: str, group: str, *lines: str) -> None:
        self.findings.append(Finding("WARNING", code, group, list(lines)))

    def status(self, group: str) -> str:
        levels = {f.level for f in self.findings if f.group == group}
        if "ERROR" in levels:
            return "FAILED"
        if "WARNING" in levels:
            return "WARNING"
        return "OK"

    @property
    def n_errors(self) -> int:
        return sum(1 for f in self.findings if f.level == "ERROR")

    @property
    def n_warnings(self) -> int:
        return sum(1 for f in self.findings if f.level == "WARNING")


def timestamp_problem(value: str) -> str:
    """Explain why a value is not an accepted UTC timestamp."""
    if value == "":
        return "empty value"
    if _FRACTION_RE.match(value):
        return "fractional seconds are not allowed"
    if _NAIVE_RE.match(value):
        return "time zone is missing (use the suffix Z, e.g. 2026-10-05T06:00:00Z)"
    if _OFFSET_RE.match(value):
        return "only UTC is allowed (suffix Z or +00:00)"
    return "unrecognised format (expected YYYY-MM-DDTHH:MM:SSZ, e.g. 2026-10-05T06:00:00Z)"


def parse_utc_timestamps(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Parse strict UTC timestamps.

    Returns (parsed, invalid_mask). Empty cells are NaT and not invalid.
    Accepted forms: YYYY-MM-DDTHH:MM:SSZ and YYYY-MM-DDTHH:MM:SS+00:00.
    """
    text = values.fillna("").astype(str)
    match = text.str.extract(_UTC_TIMESTAMP_RE)
    core = match[0]
    parsed = pd.to_datetime(core, format="%Y-%m-%dT%H:%M:%S", errors="coerce", utc=True)
    invalid = (text != "") & parsed.isna()
    return parsed, invalid


def row_ref(df: pd.DataFrame, index) -> str:
    """Human-readable reference to a row: file line, origin and step."""
    line = int(index) + 2  # header is line 1
    origin = df.at[index, "forecast_origin_utc"] if "forecast_origin_utc" in df.columns else "?"
    step = df.at[index, "step"] if "step" in df.columns else "?"
    return f"line {line} (forecast_origin_utc={origin}, step={step})"


def read_csv_text(path: Path) -> pd.DataFrame:
    """Read a CSV as text: BOM removed, CRLF accepted, empty cells as ''."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def load_origins(path: Path, report: Report) -> list[pd.Timestamp] | None:
    try:
        origins_df = read_csv_text(path)
    except Exception as exc:  # noqa: BLE001 - report any read problem clearly
        report.error("INVALID_ORIGINS_FILE", "File", f"Cannot read origins file {path}: {exc}")
        return None
    if "forecast_origin_utc" not in origins_df.columns:
        report.error("INVALID_ORIGINS_FILE", "File", f"Origins file {path} has no column forecast_origin_utc.")
        return None
    parsed, invalid = parse_utc_timestamps(origins_df["forecast_origin_utc"])
    if invalid.any() or parsed.isna().any():
        report.error("INVALID_ORIGINS_FILE", "File", f"Origins file {path} contains invalid or empty timestamps.")
        return None
    if parsed.duplicated().any():
        report.error("INVALID_ORIGINS_FILE", "File", f"Origins file {path} contains duplicate origins.")
        return None
    not_full_hour = parsed != parsed.dt.floor("h")
    if not_full_hour.any():
        report.error("INVALID_ORIGINS_FILE", "File", f"Origins file {path} contains origins that are not on the full hour.")
        return None
    return list(parsed)


def validate(backtest_path: Path, origins_path: Path, horizon_steps: int, step_minutes: int,
             tolerance: float) -> tuple[Report, dict]:
    report = Report()
    counts = {"origins_found": 0, "origins_expected": 0, "rows": 0, "rows_expected": 0, "cutoff_given": 0}

    origins = load_origins(origins_path, report)
    if origins is not None:
        counts["origins_expected"] = len(origins)
        counts["rows_expected"] = len(origins) * horizon_steps

    try:
        df = read_csv_text(backtest_path)
    except Exception as exc:  # noqa: BLE001
        report.error("UNREADABLE_FILE", "File", f"Cannot read {backtest_path} as a comma-separated UTF-8 CSV file: {exc}")
        return report, counts
    counts["rows"] = len(df)

    # Rule 1: columns ------------------------------------------------------
    missing_cols = [c for c in ALL_COLUMNS if c not in df.columns]
    if missing_cols:
        report.error("MISSING_COLUMNS", "Required columns",
                     f"{len(missing_cols)} column(s) missing: {', '.join(missing_cols)}.",
                     "All 13 columns must be present; optional columns may contain empty cells.")
    extra_cols = [c for c in df.columns if c not in ALL_COLUMNS]
    if extra_cols:
        report.warning("UNEXPECTED_COLUMNS", "Required columns",
                       f"{len(extra_cols)} unexpected column(s) are ignored: {', '.join(extra_cols)}.")
    if df.columns.duplicated().any():
        report.error("DUPLICATE_COLUMNS", "Required columns", "The header contains duplicate column names.")
        return report, counts
    present = set(df.columns)

    # Rule 4: missing required values -------------------------------------
    for column in REQUIRED_COLUMNS:
        if column not in present:
            continue
        empty = df[column].str.strip() == ""
        n_empty = int(empty.sum())
        if n_empty:
            extra = " - the file has not been filled in." if n_empty == len(df) else "."
            report.error("MISSING_VALUES", "Missing required values",
                         f"{column} contains {n_empty} empty value(s){extra}",
                         f"First empty value: {row_ref(df, empty.idxmax())}")

    # Rule 1: numeric types -------------------------------------------------
    numbers: dict[str, pd.Series] = {}
    for column in NUMERIC_COLUMNS:
        if column not in present:
            continue
        text = df[column].str.strip()
        bad = (text != "") & ~text.str.match(_NUMBER_RE)
        if bad.any():
            first = bad.idxmax()
            report.error("INVALID_NUMBER", "Data types",
                         f"{column}: {int(bad.sum())} value(s) are not decimal numbers (use a dot as the decimal separator).",
                         f"First invalid value: '{df.at[first, column]}' on {row_ref(df, first)}")
        values = pd.to_numeric(text.where(~bad & (text != "")), errors="coerce")
        numbers[column] = values

    # Rule 8: timestamps ----------------------------------------------------
    stamps: dict[str, pd.Series] = {}
    for column in TIMESTAMP_COLUMNS:
        if column not in present:
            continue
        parsed, invalid = parse_utc_timestamps(df[column].str.strip())
        stamps[column] = parsed
        if invalid.any():
            first = invalid.idxmax()
            value = df.at[first, column]
            report.error("INVALID_TIMESTAMP", "Timestamps",
                         f"{column}: {int(invalid.sum())} value(s) are not valid UTC timestamps.",
                         f"First invalid value: '{value}' on {row_ref(df, first)} - {timestamp_problem(value.strip())}.",
                         "Expected format: YYYY-MM-DDTHH:MM:SSZ (the suffix +00:00 is also accepted).")

    # Rule 8: step ----------------------------------------------------------
    step = None
    if "step" in present:
        text = df["step"].str.strip()
        is_int = text.str.match(_INTEGER_RE)
        step = pd.to_numeric(text.where(is_int), errors="coerce")
        bad_step = (text != "") & (step.isna() | (step < 1) | (step > horizon_steps))
        if bad_step.any():
            first = bad_step.idxmax()
            report.error("INVALID_STEP", "Steps",
                         f"{int(bad_step.sum())} row(s) have a step that is not an integer from 1 to {horizon_steps}.",
                         f"First invalid value: '{df.at[first, 'step']}' on {row_ref(df, first)}")
        step = step.where(~bad_step)

    # Rules 2 and 3: origins, rows, duplicates ---------------------------------
    origin = stamps.get("forecast_origin_utc")
    if origin is not None and step is not None:
        key_ok = origin.notna() & step.notna()
        keys = pd.DataFrame({"origin": origin[key_ok], "step": step[key_ok].astype(int)})

        duplicated = keys.duplicated(keep="first")
        if duplicated.any():
            first = duplicated.idxmax()
            report.error("DUPLICATE_ROWS", "Duplicate rows",
                         f"{int(duplicated.sum())} duplicate forecast_origin_utc + step combination(s) found.",
                         f"First duplicate: {row_ref(df, first)}")

        found = set(keys["origin"].unique())
        counts["origins_found"] = len(found)
        if origins is not None:
            expected = set(origins)
            counts["origins_found"] = len(found & expected)
            missing_origins = sorted(expected - found)
            if missing_origins:
                report.error("MISSING_ORIGINS", "Forecast origins",
                             f"{len(missing_origins)} forecast origin(s) are missing.",
                             f"First missing origin: {missing_origins[0].strftime(TIMESTAMP_FORMAT)}")
            unexpected = keys["origin"].isin(found - expected)
            if unexpected.any():
                first = unexpected.idxmax()
                n_unexpected = len(found - expected)
                report.error("UNEXPECTED_ORIGINS", "Forecast origins",
                             f"{n_unexpected} forecast origin(s) are not in the origins file ({int(unexpected.sum())} row(s)).",
                             f"First unexpected row: {row_ref(df, first)}")
            # Missing (origin, step) rows among the expected origins that are present
            unique_keys = keys.drop_duplicates()
            present_expected = unique_keys[unique_keys["origin"].isin(expected)]
            n_missing_rows = len(found & expected) * horizon_steps - len(present_expected)
            if n_missing_rows > 0:
                per_origin = present_expected.groupby("origin")["step"].apply(set)
                first_missing = None
                for o in sorted(per_origin.index):
                    gaps = sorted(set(range(1, horizon_steps + 1)) - per_origin[o])
                    if gaps:
                        first_missing = (o, gaps[0])
                        break
                report.error("MISSING_ROWS", "Rows",
                             f"{n_missing_rows} row(s) are missing: every forecast origin must have exactly {horizon_steps} steps.",
                             f"First missing row: forecast_origin_utc={first_missing[0].strftime(TIMESTAMP_FORMAT)}, step={first_missing[1]}")
        if counts["rows_expected"] and counts["rows"] != counts["rows_expected"]:
            report.error("ROW_COUNT", "Rows",
                         f"The file has {counts['rows']} data rows; expected {counts['rows_expected']} "
                         f"({counts['origins_expected']} origins x {horizon_steps} steps).")

        # Rule 8: delivery_start_utc matches origin and step
        delivery = stamps.get("delivery_start_utc")
        if delivery is not None:
            expected_delivery = origin + pd.to_timedelta((step - 1) * step_minutes, unit="min")
            comparable = origin.notna() & step.notna() & delivery.notna()
            wrong = comparable & (delivery != expected_delivery)
            if wrong.any():
                first = wrong.idxmax()
                report.error("INVALID_DELIVERY_START", "Timestamps",
                             f"{int(wrong.sum())} row(s) have a delivery_start_utc that does not equal "
                             f"forecast_origin_utc + (step - 1) x {step_minutes} min.",
                             f"First invalid row: {row_ref(df, first)}, delivery_start_utc={df.at[first, 'delivery_start_utc']}, "
                             f"expected {expected_delivery[first].strftime(TIMESTAMP_FORMAT)}")

    # Rule 5: quantile order ------------------------------------------------
    if all(c in numbers for c in QUANTILE_COLUMNS):
        p10, p50, p90 = (numbers[c] for c in QUANTILE_COLUMNS)
        for lower, upper, a, b in (("p10", "p50", p10, p50), ("p50", "p90", p50, p90)):
            bad = a.notna() & b.notna() & (a > b)
            if bad.any():
                first = bad.idxmax()
                report.error("INVALID_QUANTILE_ORDER", "Quantiles",
                             f"{int(bad.sum())} row(s) have {lower} > {upper}. Required: p10 <= p50 <= p90.",
                             f"First invalid row: {row_ref(df, first)}, {lower}={df.at[first, lower]}, {upper}={df.at[first, upper]}")

    # Rule 5: probability range ---------------------------------------------
    for column in PROBABILITY_COLUMNS:
        if column not in numbers:
            continue
        values = numbers[column]
        bad = values.notna() & ((values < 0) | (values > 1))
        if bad.any():
            first = bad.idxmax()
            report.error("PROBABILITY_OUT_OF_RANGE", "Probabilities",
                         f"{column}: {int(bad.sum())} value(s) are outside the range 0-1.",
                         f"First invalid value: {df.at[first, column]} on {row_ref(df, first)}")

    # Rule 5: direction probabilities sum to 1 --------------------------------
    if all(c in numbers for c in DIRECTION_COLUMNS):
        total = numbers["p_up"] + numbers["p_down"] + numbers["p_none"]
        bad = total.notna() & ((total - 1.0).abs() > tolerance + 1e-12)
        if bad.any():
            first = bad.idxmax()
            report.error("INVALID_PROBABILITY_SUM", "Direction probabilities",
                         f"{int(bad.sum())} row(s) have p_up + p_down + p_none outside 1 +/- {tolerance}.",
                         f"First invalid row: {row_ref(df, first)}, sum={total[first]:.6f}")

    # Rule 6: spike probabilities --------------------------------------------
    if all(c in numbers for c in SPIKE_COLUMNS):
        total = numbers["p_spike_high"] + numbers["p_spike_low"]
        bad = total.notna() & (total > 1.0 + tolerance + 1e-12)
        if bad.any():
            first = bad.idxmax()
            report.error("INVALID_SPIKE_SUM", "Spike probabilities",
                         f"{int(bad.sum())} row(s) have p_spike_high + p_spike_low greater than 1 (tolerance {tolerance}).",
                         f"First invalid row: {row_ref(df, first)}, sum={total[first]:.6f}")

    # Rule 7: data cut-off --------------------------------------------------
    cutoff = stamps.get("data_cutoff_utc")
    if cutoff is not None:
        counts["cutoff_given"] = int(cutoff.notna().sum())
        if origin is not None:
            bad = cutoff.notna() & origin.notna() & (cutoff > origin)
            if bad.any():
                first = bad.idxmax()
                report.error("DATA_CUTOFF_AFTER_ORIGIN", "Data cut-off rule",
                             f"{int(bad.sum())} row(s) have data_cutoff_utc later than forecast_origin_utc.",
                             f"First invalid row: {row_ref(df, first)}, data_cutoff_utc={df.at[first, 'data_cutoff_utc']}")
    return report, counts


def print_report(path: Path, report: Report, counts: dict) -> None:
    passed = report.n_errors == 0
    print("VALIDATION PASSED" if passed else "VALIDATION FAILED")
    print()
    print(f"File: {path.name}")
    print(f"Validator version: {VALIDATOR_VERSION}")
    print(f"Schema version: {SCHEMA_VERSION}")
    print(f"Forecast origins: {counts['origins_found']} / {counts['origins_expected']}")
    print(f"Rows: {counts['rows']} / {counts['rows_expected']}")
    for group in GROUPS:
        status = report.status(group)
        if group == "Data cut-off rule" and status == "OK":
            status = f"OK (optional column, {counts['cutoff_given']} values given)"
        print(f"{group}: {status}")
    print(f"Errors: {report.n_errors}")
    print(f"Warnings: {report.n_warnings}")
    for finding in report.findings:
        print()
        print(f"{finding.level} [{finding.code}]")
        for line in finding.lines:
            print(line)
    print()
    print("The submission is ready for scoring." if passed else "The submission is not ready for scoring.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the format of forecast_backtest.csv (schema version "
            f"{SCHEMA_VERSION}). Checks structure and internal consistency only; "
            "forecast quality is evaluated by score.py."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Required columns:\n  " + ", ".join(REQUIRED_COLUMNS) + "\n"
            "Optional columns (may be left empty):\n  " + ", ".join(OPTIONAL_COLUMNS) + "\n\n"
            "Timestamps: UTC, YYYY-MM-DDTHH:MM:SSZ (suffix +00:00 also accepted).\n"
            "Exit code: 0 = validation passed, 1 = validation failed.\n\n"
            "Example:\n  python validate_submission.py --backtest forecast_backtest.csv --origins forecast_origins.csv"
        ),
    )
    parser.add_argument("--backtest", required=True, help="Path to the submission file forecast_backtest.csv.")
    parser.add_argument("--origins", required=True, help="Path to forecast_origins.csv.")
    parser.add_argument("--horizon-steps", type=int, default=40,
                        help="Number of steps per forecast origin (default: %(default)s).")
    parser.add_argument("--step-minutes", type=int, default=15,
                        help="Length of one step in minutes (default: %(default)s).")
    parser.add_argument("--tolerance", type=float, default=0.001,
                        help="Tolerance for the probability sums (default: %(default)s).")
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {VALIDATOR_VERSION} (schema {SCHEMA_VERSION})")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    backtest = Path(args.backtest)
    origins = Path(args.origins)
    report, counts = validate(backtest, origins, args.horizon_steps, args.step_minutes, args.tolerance)
    print_report(backtest, report, counts)
    return 0 if report.n_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
