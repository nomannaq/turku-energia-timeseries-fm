#!/usr/bin/env python3
"""Generate the forecast origins and the sample submission template.

Since AI 2026 - Turku Energia challenge: 10-hour imbalance price forecast.
Organiser tool (starter package file 12). Produces:

  forecast_origins.csv   one column, forecast_origin_utc: every full-hour
                         forecast origin of the test period (UTC, ISO 8601, Z)
  sample_submission.csv  one row per forecast origin and step; identifier
                         columns filled, forecast columns left as empty cells

The test period is given as local dates (default time zone Europe/Helsinki).
The first origin is 00:00 local time on the start date and the last origin is
one origin interval before 00:00 local time on the day after the end date.
The origins are generated as an evenly spaced UTC series, so a daylight saving
time change produces neither a gap nor a duplicate.

Defaults follow the starter package specification v5 (decisions P1, P2):
step 1 is the quarter-hour that starts at the forecast origin, and quarter-hours
after the end of the test period are included.

No network access is used.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_VERSION = "0.1.0"
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
FORECAST_COLUMNS = [c for c in ALL_COLUMNS if c not in ("forecast_origin_utc", "delivery_start_utc", "step")]

TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def build_origins(test_start: str, test_end: str, timezone: str, frequency_minutes: int) -> pd.DatetimeIndex:
    """Return the forecast origins of the test period as a UTC DatetimeIndex."""
    start_date = pd.Timestamp(test_start)
    end_date = pd.Timestamp(test_end)
    if start_date != start_date.normalize() or end_date != end_date.normalize():
        raise ValueError("--test-start and --test-end must be dates (YYYY-MM-DD) without a time of day.")
    if end_date < start_date:
        raise ValueError("--test-end must not be earlier than --test-start.")
    if frequency_minutes <= 0 or (24 * 60) % frequency_minutes != 0:
        raise ValueError("--origin-frequency-minutes must be a positive divisor of 1440.")

    first_local = start_date.tz_localize(timezone)
    period_end_local = (end_date + pd.Timedelta(days=1)).tz_localize(timezone)
    first_utc = first_local.tz_convert("UTC")
    end_utc = period_end_local.tz_convert("UTC")
    frequency = pd.Timedelta(minutes=frequency_minutes)
    origins = pd.date_range(first_utc, end_utc - frequency, freq=frequency)
    return origins


def build_template(origins: pd.DatetimeIndex, horizon_steps: int, step_minutes: int) -> pd.DataFrame:
    """Return the sample submission: identifiers filled, forecast columns empty."""
    if horizon_steps <= 0:
        raise ValueError("--horizon-steps must be positive.")
    if step_minutes <= 0:
        raise ValueError("--step-minutes must be positive.")
    steps = pd.RangeIndex(1, horizon_steps + 1)
    origin_rep = origins.repeat(horizon_steps)
    step_rep = list(steps) * len(origins)
    offsets = pd.to_timedelta([(s - 1) * step_minutes for s in step_rep], unit="min")
    delivery = origin_rep + offsets

    template = pd.DataFrame(
        {
            "forecast_origin_utc": origin_rep.strftime(TIMESTAMP_FORMAT),
            "delivery_start_utc": delivery.strftime(TIMESTAMP_FORMAT),
            "step": step_rep,
        }
    )
    for column in FORECAST_COLUMNS:
        template[column] = ""
    return template[ALL_COLUMNS]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate forecast_origins.csv and sample_submission.csv for the "
            "Since AI 2026 imbalance price challenge (schema version "
            f"{SCHEMA_VERSION})."
        ),
        epilog=(
            "Required columns: " + ", ".join(REQUIRED_COLUMNS) + ". "
            "Optional columns (may be left empty): " + ", ".join(OPTIONAL_COLUMNS) + ". "
            "Example: python make_templates.py --output-dir ."
        ),
    )
    parser.add_argument("--test-start", default="2026-10-05",
                        help="First day of the test period, local date YYYY-MM-DD (default: %(default)s).")
    parser.add_argument("--test-end", default="2026-11-01",
                        help="Last day of the test period, local date YYYY-MM-DD (default: %(default)s).")
    parser.add_argument("--timezone", default="Europe/Helsinki",
                        help="Time zone of the test period dates (default: %(default)s).")
    parser.add_argument("--origin-frequency-minutes", type=int, default=60,
                        help="Interval between forecast origins in minutes (default: %(default)s).")
    parser.add_argument("--horizon-steps", type=int, default=40,
                        help="Number of quarter-hour steps per forecast origin (default: %(default)s).")
    parser.add_argument("--step-minutes", type=int, default=15,
                        help="Length of one step in minutes (default: %(default)s).")
    parser.add_argument("--output-dir", default=".",
                        help="Directory for the output files (default: current directory).")
    parser.add_argument("--version", action="version", version=f"%(prog)s {SCRIPT_VERSION} (schema {SCHEMA_VERSION})")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        origins = build_origins(args.test_start, args.test_end, args.timezone, args.origin_frequency_minutes)
        template = build_template(origins, args.horizon_steps, args.step_minutes)
    except (ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    origins_path = output_dir / "forecast_origins.csv"
    template_path = output_dir / "sample_submission.csv"

    pd.DataFrame({"forecast_origin_utc": origins.strftime(TIMESTAMP_FORMAT)}).to_csv(
        origins_path, index=False, encoding="utf-8", lineterminator="\n"
    )
    template.to_csv(template_path, index=False, encoding="utf-8", lineterminator="\n")

    last_delivery = origins[-1] + pd.Timedelta(minutes=(args.horizon_steps - 1) * args.step_minutes)
    print(f"make_templates.py {SCRIPT_VERSION}, schema version {SCHEMA_VERSION}")
    print(f"Test period (local, {args.timezone}): {args.test_start} - {args.test_end}")
    print(f"Forecast origins: {len(origins)}")
    print(f"  first: {origins[0].strftime(TIMESTAMP_FORMAT)}")
    print(f"  last:  {origins[-1].strftime(TIMESTAMP_FORMAT)}")
    print(f"Steps per origin: {args.horizon_steps} x {args.step_minutes} min")
    print(f"Template rows: {len(template)}")
    print(f"Last delivery quarter-hour start: {last_delivery.strftime(TIMESTAMP_FORMAT)}")
    print("Required columns: " + ", ".join(REQUIRED_COLUMNS))
    print("Optional columns: " + ", ".join(OPTIONAL_COLUMNS))
    print(f"Wrote {origins_path}")
    print(f"Wrote {template_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
