#!/usr/bin/env python3
"""Fetch the input data and build the data files of the starter package.

Since AI 2026 - Turku Energia challenge: 10-hour imbalance price forecast.
Organiser tool (starter package file 10). Produces, in long format
(one row per series and quarter-hour, specification chapter 6):

  <out-dir>/inputs.parquet, inputs.csv   open data inputs (Fingrid, ENTSO-E,
                                         calendar) with first_usable_origin_utc
  <out-dir>/spot_nordpool.csv            Nord Pool spot price (only with
                                         --spot-file; licence terms D3)
  <out-dir>/actuals_validation.csv       validation period actuals
  <out-dir>/quality_report.csv, .md      row counts, gaps and removed identical duplicates per series
  <organizer-dir>/actuals_test.csv       test period actuals (organiser only)

Data usage rule (D1): a row may be used at forecast origin t only if
first_usable_origin_utc <= t. The column is computed from the per-source
delays in availability.csv (specification chapters 6.2 and 8).

API keys (never stored in this script):
  FINGRID_API_KEY    Fingrid Open Data API key
  ENTSOE_API_TOKEN   ENTSO-E Transparency Platform security token
Each key is read from the environment variable of the same name or, if that
is not set, from the file given with --env-file (KEY=VALUE lines). There is no
default key file. Keys are masked as *** in all messages.

Raw API responses are cached in --cache-dir, so an interrupted run can be
restarted and only the missing chunks are fetched again.

Example:
  python build_dataset.py --start 2026-06-01 --end 2026-11-02T06:45:00Z --env-file keys.env --spot-file spot.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import sys
import time
import traceback
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

SCRIPT_VERSION = "0.1.2"
SCHEMA_VERSION = "1.0"

TZ = ZoneInfo("Europe/Helsinki")
UTC = timezone.utc
QUARTER = timedelta(minutes=15)

# Specification v5.1 chapter 2
DATA_PERIOD_START = datetime(2026, 5, 31, 21, 0, tzinfo=UTC)
DATA_PERIOD_LAST = datetime(2026, 11, 2, 6, 45, tzinfo=UTC)  # start of the last quarter-hour
VALIDATION_PERIOD = (datetime(2026, 9, 6, 21, 0, tzinfo=UTC), datetime(2026, 10, 4, 20, 45, tzinfo=UTC))
TEST_PERIOD = (datetime(2026, 10, 4, 21, 0, tzinfo=UTC), datetime(2026, 11, 2, 6, 45, tzinfo=UTC))

FINGRID_URL = "https://data.fingrid.fi/api/datasets/{dataset_id}/data"
ENTSOE_URL = "https://web-api.tp.entsoe.eu/api"
FINGRID_MIN_INTERVAL_S = 2.1   # Fingrid limit: 1 call / 2 s, 10 000 calls / day
ENTSOE_MIN_INTERVAL_S = 1.0    # ENTSO-E limit: 400 calls / min
MAX_RETRIES = 5
PROVISIONAL_HOURS = 48          # chunks ending closer than this to the fetch time may still change
PROVISIONAL_CACHE_MAX_AGE_H = 6  # such chunks are re-fetched when the cached copy is older than this

ENTSOE_EIC = {
    "FI": "10YFI-1--------U",
    "SE1": "10Y1001A1001A44P",
    "SE3": "10Y1001A1001A46L",
    "EE": "10Y1001A1001A39I",
}

OUTPUT_COLUMNS = ["series_id", "source", "dataset_id", "timestamp_utc", "value", "unit", "first_usable_origin_utc"]
ACTUALS_COLUMNS = ["delivery_start_utc", "imbalance_price", "spot_price", "direction", "first_usable_origin_utc"]
DIRECTION_MAP = {1: "up", -1: "down", 0: "none"}  # Fingrid 369 (D17)


# --------------------------------------------------------------------------------------
# Series catalogue (specification 7.3, decisions D4 and D17)
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class FingridSeries:
    series_id: str
    dataset_id: int
    unit: str          # unit as given in the Fingrid metadata
    note: str = ""


FINGRID_SERIES = [
    FingridSeries("fi_imbalance_price", 319, "EUR/MWh"),
    FingridSeries("fi_mfrr_up_price", 244, "EUR/MWh", "mFRR energy marginal price, up"),
    FingridSeries("fi_mfrr_down_price", 106, "EUR/MWh", "mFRR energy marginal price, down"),
    FingridSeries("fi_mfrr_up_activated", 375, "MW", "activated mFRR balancing regulation, up (max power of the quarter)"),
    FingridSeries("fi_mfrr_down_activated", 376, "MW", "activated mFRR balancing regulation, down (max power of the quarter)"),
    FingridSeries("fi_mfrr_direction", 369, "-1/0/1", "dominating mFRR direction: 1 up, -1 down, 0 none"),
    FingridSeries("fi_afrr_up_price", 347, "EUR/MWh", "sparse by source: empty when there is no aFRR activation"),
    FingridSeries("fi_afrr_down_price", 348, "EUR/MWh", "sparse by source: empty when there is no aFRR activation"),
    FingridSeries("fi_consumption", 124, "MWh/h"),
    FingridSeries("fi_production", 74, "MWh/h"),
    FingridSeries("fi_consumption_fc_da", 165, "MW", "day-ahead forecast, updated once a day"),
    FingridSeries("fi_wind_fc_da", 246, "MWh/h", "day-ahead forecast, updated once a day"),
    FingridSeries("fi_solar_fc_da", 247, "MWh/h", "day-ahead forecast, updated once a day"),
    FingridSeries("fi_se3_flow", 61, "MW", "positive = Finland to SE3"),
    FingridSeries("fi_no4_flow", 57, "MW", "positive = Finland to NO4"),
    FingridSeries("fi_se1_flow", 60, "MW", "positive = Finland to SE1"),
    FingridSeries("fi_ee_flow", 55, "MWh/h", "positive = Finland to Estonia"),
]


@dataclass(frozen=True)
class EntsoeSeries:
    key: str
    article: str           # Transparency Regulation article, matched against availability.csv
    area: str
    params: dict
    unit: str
    series_id: str = ""    # single series
    split_category: bool = False  # 17.1.g: one series per imbalance price category (D17)
    note: str = ""

    @property
    def dataset_id(self) -> str:
        return f"{self.article}:{self.area}"


def _imbalance(area: str) -> EntsoeSeries:
    return EntsoeSeries(
        key=f"imbalance_price_{area}", article="17.1.g", area=area,
        params={"documentType": "A85", "controlArea_Domain": ENTSOE_EIC[area]},
        unit="EUR/MWh", split_category=True,
        note="imbalance price; _long = category A04 (excess balance), _short = category A05 (insufficient balance)")


def _load_fc(area: str) -> EntsoeSeries:
    return EntsoeSeries(
        key=f"load_fc_da_{area}", article="6.1.b", area=area,
        params={"documentType": "A65", "processType": "A01", "outBiddingZone_Domain": ENTSOE_EIC[area]},
        unit="MW", series_id=f"{area.lower()}_load_fc_da", note="day-ahead total load forecast")


def _capacity(out_area: str, in_area: str) -> EntsoeSeries:
    return EntsoeSeries(
        key=f"capacity_da_{out_area}_{in_area}", article="11.1", area=f"{out_area}-{in_area}",
        params={"documentType": "A31", "auction.Type": "A01", "contract_MarketAgreement.Type": "A01",
                "out_Domain": ENTSOE_EIC[out_area], "in_Domain": ENTSOE_EIC[in_area]},
        unit="MW", series_id=f"{out_area.lower()}_{in_area.lower()}_capacity_da",
        note=f"offered day-ahead implicit capacity {out_area} to {in_area}")


# Only data on the ENTSO-E list of data available for free re-use (231018).
# NO4: no imbalance price or load forecast is published (checked 31.10.2026).
# Transfer capacities: only FI-EE is published (Nordic flow-based), D17.
# Outages (10.1) are not in the package (D17).
ENTSOE_SERIES = [
    _imbalance("SE1"), _imbalance("SE3"), _imbalance("EE"),
    _load_fc("SE1"), _load_fc("SE3"), _load_fc("EE"),
    _capacity("FI", "EE"), _capacity("EE", "FI"),
]

IMBALANCE_CATEGORY_SUFFIX = {"A04": "long", "A05": "short"}

CALENDAR_SERIES = {
    "cal_weekday_local": "0=Mon..6=Sun",
    "cal_hour_local": "hour",
    "cal_is_holiday_fi": "0/1",
    "cal_is_dst": "0/1",
}


# --------------------------------------------------------------------------------------
# Errors, logging and secret masking
# --------------------------------------------------------------------------------------

class BuildError(Exception):
    """A problem that stops the run with a clear message."""


class SpotFileError(BuildError):
    """The spot file does not meet the format rules."""


class SecretMasker:
    """Replaces API keys and security tokens with *** in any text."""

    _TOKEN_RE = re.compile(r"(securityToken=)[^&\s\"'<>]+", re.IGNORECASE)
    _HEADER_RE = re.compile(r"(x-api-key['\"]?\s*[:=]\s*['\"]?)[^\s'\",}]+", re.IGNORECASE)

    def __init__(self) -> None:
        self._secrets: list[str] = []

    def add(self, secret: str | None) -> None:
        if secret and len(secret) >= 4 and secret not in self._secrets:
            self._secrets.append(secret)

    def mask(self, text: object) -> str:
        out = str(text)
        for secret in self._secrets:
            out = out.replace(secret, "***")
        out = self._TOKEN_RE.sub(r"\1***", out)
        out = self._HEADER_RE.sub(r"\1***", out)
        return out


MASKER = SecretMasker()
WARNINGS: list[str] = []


def log(message: str) -> None:
    print(MASKER.mask(message), flush=True)


def warn(message: str) -> None:
    text = MASKER.mask(message)
    WARNINGS.append(text)
    print(f"WARNING: {text}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------------------
# Keys
# --------------------------------------------------------------------------------------

def read_env_file(path: Path) -> dict[str, str]:
    """Read KEY=VALUE lines. Comments (#) and blank lines are ignored."""
    if not path.is_file():
        raise BuildError(f"--env-file not found: {path}")
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        values[key] = value.strip().strip('"').strip("'")
    return values


class KeyStore:
    """Looks up a key from the environment first, then from --env-file."""

    def __init__(self, env_file: Path | None) -> None:
        self._file_values = read_env_file(env_file) if env_file else {}
        self._env_file = env_file
        for name in ("FINGRID_API_KEY", "ENTSOE_API_TOKEN"):
            MASKER.add(os.environ.get(name))
            MASKER.add(self._file_values.get(name))

    def get(self, name: str) -> str:
        value = os.environ.get(name) or self._file_values.get(name)
        if not value:
            where = f"environment variable {name}"
            where += f" or {name} in {self._env_file}" if self._env_file else " (or use --env-file)"
            raise BuildError(f"API key missing: set {where}.")
        return value


# --------------------------------------------------------------------------------------
# HTTP with rate limiting, retries and a raw response cache
# --------------------------------------------------------------------------------------

class RateLimitedClient:
    def __init__(self, name: str, min_interval_s: float, timeout_s: float = 120.0) -> None:
        import requests  # imported here so that the offline functions work without requests

        self._requests = requests
        self._session = requests.Session()
        self.name = name
        self.min_interval_s = min_interval_s
        self.timeout_s = timeout_s
        self._last_call = 0.0
        self.calls = 0

    def _wait(self) -> None:
        delay = self._last_call + self.min_interval_s - time.monotonic()
        if delay > 0:
            time.sleep(delay)

    def get(self, url: str, params: dict, headers: dict, label: str, accept_400_no_data: bool = False) -> bytes:
        for attempt in range(MAX_RETRIES + 1):
            self._wait()
            backoff = min(120.0, 5.0 * 2 ** attempt)
            try:
                response = self._session.get(url, params=params, headers=headers, timeout=self.timeout_s)
            except self._requests.RequestException as exc:
                self._last_call = time.monotonic()
                if attempt < MAX_RETRIES:
                    warn(f"{self.name} {label}: {type(exc).__name__}: {MASKER.mask(exc)}; retry in {backoff:.0f} s")
                    time.sleep(backoff)
                    continue
                raise BuildError(f"{self.name} {label}: request failed after {MAX_RETRIES} retries: "
                                 f"{type(exc).__name__}: {MASKER.mask(exc)}") from None
            self._last_call = time.monotonic()
            self.calls += 1
            status = response.status_code
            if status == 429 or 500 <= status < 600:
                retry_after = response.headers.get("Retry-After", "")
                delay = float(retry_after) if retry_after.strip().isdigit() else backoff
                if attempt < MAX_RETRIES:
                    warn(f"{self.name} {label}: HTTP {status}; retry in {delay:.0f} s")
                    time.sleep(delay)
                    continue
                raise BuildError(f"{self.name} {label}: HTTP {status} after {MAX_RETRIES} retries.")
            if status == 400 and accept_400_no_data and b"No matching data" in response.content:
                return response.content
            if status != 200:
                body = response.content[:400].decode("utf-8", "replace")
                raise BuildError(f"{self.name} {label}: HTTP {status}: {MASKER.mask(body)}")
            return response.content
        raise BuildError(f"{self.name} {label}: request failed.")  # not reached


class RawCache:
    """Stores raw responses as <cache-dir>/<source>/<name> with a .meta.json sidecar."""

    def __init__(self, root: Path, refresh: bool) -> None:
        self.root = root
        self.refresh = refresh
        self.hits = 0
        self.misses = 0

    def _paths(self, source: str, name: str) -> tuple[Path, Path]:
        base = self.root / source
        return base / name, base / (name + ".meta.json")

    def load(self, source: str, name: str) -> bytes | None:
        if self.refresh:
            return None
        data_path, meta_path = self._paths(source, name)
        if not data_path.is_file() or not meta_path.is_file():
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        fetched = datetime.fromisoformat(meta["fetched_utc"])
        chunk_end = datetime.fromisoformat(meta["chunk_end_utc"])
        provisional = fetched - chunk_end < timedelta(hours=PROVISIONAL_HOURS)
        if provisional and datetime.now(UTC) - fetched > timedelta(hours=PROVISIONAL_CACHE_MAX_AGE_H):
            return None
        self.hits += 1
        return data_path.read_bytes()

    def store(self, source: str, name: str, content: bytes, chunk_end: datetime) -> None:
        data_path, meta_path = self._paths(source, name)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_bytes(data_path, content)
        meta = {"fetched_utc": datetime.now(UTC).isoformat(), "chunk_end_utc": chunk_end.isoformat()}
        _atomic_write_bytes(meta_path, json.dumps(meta).encode("utf-8"))
        self.misses += 1


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(content)
    os.replace(tmp, path)


# --------------------------------------------------------------------------------------
# Time helpers
# --------------------------------------------------------------------------------------

def fmt_utc(ts: datetime | pd.Timestamp) -> str:
    return pd.Timestamp(ts).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_window_arg(text: str, is_end: bool) -> datetime:
    """Return the window start, or the exclusive window end.

    YYYY-MM-DD  = a local (Europe/Helsinki) date; an end date is included completely.
    ISO with Z  = a UTC quarter-hour start; an end timestamp is the start of the last quarter-hour.
    """
    text = text.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        day = date.fromisoformat(text)
        if is_end:
            day += timedelta(days=1)
        return datetime(day.year, day.month, day.day, tzinfo=TZ).astimezone(UTC)
    match = re.fullmatch(r"(\d{4}-\d{2}-\d{2})T(\d{2}):(\d{2})(?::(\d{2}))?(Z|\+00:00)", text)
    if not match:
        raise BuildError(f"invalid --start/--end value '{text}': use YYYY-MM-DD (local date) or "
                         "YYYY-MM-DDTHH:MM:SSZ (UTC).")
    ts = datetime.fromisoformat(f"{match.group(1)}T{match.group(2)}:{match.group(3)}:{match.group(4) or '00'}+00:00")
    if ts.minute % 15 or ts.second:
        raise BuildError(f"--start/--end '{text}' is not on a quarter-hour boundary.")
    return ts + QUARTER if is_end else ts


def quarter_index(start: datetime, end_excl: datetime) -> pd.DatetimeIndex:
    return pd.date_range(start, end_excl - QUARTER, freq="15min", tz="UTC")


def local_month_chunks(start: datetime, end_excl: datetime) -> list[tuple[datetime, datetime]]:
    """Split the window at local (Europe/Helsinki) month boundaries."""
    chunks = []
    cursor = start
    while cursor < end_excl:
        local = cursor.astimezone(TZ)
        year, month = (local.year + 1, 1) if local.month == 12 else (local.year, local.month + 1)
        boundary = datetime(year, month, 1, tzinfo=TZ).astimezone(UTC)
        chunk_end = min(boundary, end_excl)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end
    return chunks


def compact(ts: datetime) -> str:
    return ts.astimezone(UTC).strftime("%Y%m%dT%H%MZ")


# --------------------------------------------------------------------------------------
# availability.csv and first_usable_origin_utc (specification 6.2 and 8)
# --------------------------------------------------------------------------------------

@dataclass
class AvailabilityRule:
    source: str
    series_group: str
    dataset_ids: list[str]
    availability_type: str
    lag_minutes: int | None
    publish_day_offset: int | None
    publish_time_local: str | None


def load_availability(path: Path, capacity_time_override: str | None) -> list[AvailabilityRule]:
    if not path.is_file():
        raise BuildError(f"availability file not found: {path}")
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    needed = ["source", "series_group", "dataset_id", "availability_type", "lag_minutes",
              "publish_day_offset", "publish_time_local"]
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        raise BuildError(f"{path.name}: column(s) missing: {', '.join(missing)}")
    rules = []
    for line_no, row in enumerate(raw.to_dict("records"), start=2):
        kind = row["availability_type"].strip()
        publish_time = row["publish_time_local"].strip() or None
        if (row["source"] == "entsoe" and row["series_group"] == "transfer_capacity"
                and capacity_time_override):
            if publish_time != capacity_time_override:
                warn(f"--entsoe-capacity-publish-time {capacity_time_override} overrides {path.name} "
                     f"({publish_time}); update {path.name} before release.")
            publish_time = capacity_time_override
        if kind not in ("lag_after_period_end", "daily_local_time", "always"):
            raise BuildError(f"{path.name} line {line_no}: unknown availability_type '{kind}'.")
        lag = row["lag_minutes"].strip()
        offset = row["publish_day_offset"].strip()
        if kind == "lag_after_period_end" and not lag.isdigit():
            raise BuildError(f"{path.name} line {line_no}: lag_minutes must be an integer, got '{lag}'.")
        if kind == "daily_local_time":
            if not re.fullmatch(r"-?\d+", offset):
                raise BuildError(f"{path.name} line {line_no}: publish_day_offset must be an integer, got '{offset}'.")
            if not publish_time or not re.fullmatch(r"\d{2}:\d{2}", publish_time):
                raise BuildError(f"{path.name} line {line_no}: publish_time_local must be HH:MM, got "
                                 f"'{publish_time}'.")
        rules.append(AvailabilityRule(
            source=row["source"].strip(),
            series_group=row["series_group"].strip(),
            dataset_ids=[d.strip() for d in row["dataset_id"].split(";") if d.strip()],
            availability_type=kind,
            lag_minutes=int(lag) if lag.isdigit() else None,
            publish_day_offset=int(offset) if re.fullmatch(r"-?\d+", offset) else None,
            publish_time_local=publish_time,
        ))
    return rules


def find_rule(rules: list[AvailabilityRule], source: str, dataset_key: str | None) -> AvailabilityRule:
    """dataset_key: Fingrid dataset number, ENTSO-E article, or None for sources without IDs."""
    for rule in rules:
        if rule.source != source:
            continue
        if dataset_key is None and not rule.dataset_ids:
            return rule
        if dataset_key is not None and dataset_key in rule.dataset_ids:
            return rule
    raise BuildError(f"availability.csv has no rule for source '{source}', dataset '{dataset_key}'.")


def compute_first_usable(timestamps: pd.Series, period_end: pd.Series, rule: AvailabilityRule) -> pd.Series:
    """Specification 6.2: publication time, rounded up to the next full hour (same if already full)."""
    if rule.availability_type == "always":
        return pd.Series(pd.Timestamp(DATA_PERIOD_START), index=timestamps.index)
    if rule.availability_type == "lag_after_period_end":
        published = period_end + pd.Timedelta(minutes=rule.lag_minutes)
        return published.dt.ceil("h")
    # daily_local_time: delivery day = local date of the quarter-hour start
    local_day = timestamps.dt.tz_convert(TZ).dt.normalize().dt.tz_localize(None)
    hours, minutes = (int(x) for x in rule.publish_time_local.split(":"))
    publish_local = local_day + pd.Timedelta(days=rule.publish_day_offset) + pd.Timedelta(hours=hours, minutes=minutes)
    published = publish_local.dt.tz_localize(TZ, ambiguous="raise", nonexistent="raise").dt.tz_convert("UTC")
    return published.dt.ceil("h")


# --------------------------------------------------------------------------------------
# Fingrid
# --------------------------------------------------------------------------------------

@dataclass
class SeriesFrame:
    """Expanded quarter-hour rows of one series before deduplication."""
    series_id: str
    source: str
    dataset_id: str
    unit: str
    rule_key: tuple[str, str | None]
    note: str = ""
    rows: list[tuple[datetime, float, datetime]] = field(default_factory=list)  # (quarter start, value, period end)
    info: dict = field(default_factory=dict)


def _iso_z(ts: datetime) -> str:
    return ts.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_fingrid(series: FingridSeries, chunks, client_factory, keys: KeyStore, cache: RawCache) -> list[dict]:
    rows: list[dict] = []
    for start, end in chunks:
        name = f"{series.dataset_id}_{compact(start)}_{compact(end)}.json"
        content = cache.load("fingrid", name)
        if content is None:
            client = client_factory("fingrid")
            api_key = keys.get("FINGRID_API_KEY")
            pages: list[dict] = []
            page = 1
            while True:
                params = {"startTime": _iso_z(start), "endTime": _iso_z(end), "format": "json",
                          "pageSize": 20000, "page": page, "sortBy": "startTime", "sortOrder": "asc"}
                label = f"dataset {series.dataset_id} {_iso_z(start)}..{_iso_z(end)} page {page}"
                raw = client.get(FINGRID_URL.format(dataset_id=series.dataset_id), params,
                                 {"x-api-key": api_key, "Accept": "application/json"}, label)
                body = json.loads(raw)
                pages.extend(body.get("data", []))
                pagination = body.get("pagination") or {}
                last_page = pagination.get("lastPage") or 1
                if page >= last_page:
                    break
                page += 1
            content = json.dumps({"data": pages}).encode("utf-8")
            cache.store("fingrid", name, content, end)
            log(f"  fingrid {series.dataset_id} {_iso_z(start)}..{_iso_z(end)}: {len(pages)} rows fetched")
        rows.extend(json.loads(content)["data"])
    return rows


def expand_fingrid_rows(series: FingridSeries, raw_rows: list[dict]) -> SeriesFrame:
    frame = SeriesFrame(series.series_id, "fingrid", str(series.dataset_id), series.unit,
                        ("fingrid", str(series.dataset_id)), series.note)
    durations: dict[int, int] = {}
    nulls = 0
    bad = 0
    for item in raw_rows:
        start = datetime.fromisoformat(item["startTime"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(item["endTime"].replace("Z", "+00:00"))
        minutes = int((end - start).total_seconds() // 60)
        durations[minutes] = durations.get(minutes, 0) + 1
        value = item.get("value")
        if value is None:
            nulls += 1
            continue
        if minutes <= 0 or minutes % 15 or start.minute % 15:
            bad += 1
            continue
        # A row longer than 15 min (e.g. hourly) is repeated for each quarter-hour, and
        # first_usable_origin_utc is computed from the end of the original period.
        for k in range(minutes // 15):
            frame.rows.append((start + k * QUARTER, float(value), end))
    frame.info = {"source_durations_min": durations, "null_values": nulls, "invalid_periods": bad,
                  "time_field": "startTime"}
    if bad:
        warn(f"fingrid {series.dataset_id}: {bad} row(s) with a period that is not a multiple of 15 min skipped")
    return frame


# --------------------------------------------------------------------------------------
# ENTSO-E
# --------------------------------------------------------------------------------------

def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child(element: ET.Element, name: str) -> ET.Element | None:
    for child in element:
        if _local(child.tag) == name:
            return child
    return None


def _children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in element if _local(child.tag) == name]


def _duration_minutes(text: str) -> int:
    match = re.fullmatch(r"P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?)?", text.strip())
    if not match or not any(match.groups()):
        raise BuildError(f"ENTSO-E: unsupported resolution '{text}'")
    days, hours, minutes = (int(g) if g else 0 for g in match.groups())
    return days * 1440 + hours * 60 + minutes


def _xml_documents(content: bytes) -> list[bytes]:
    if content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            return [archive.read(name) for name in sorted(archive.namelist())]
    return [content]


def parse_entsoe(content: bytes) -> tuple[list[tuple[datetime, float, datetime, str]], dict]:
    """Return (quarter start, value, original period end, imbalance price category) rows.

    curveType A03 (variable sized blocks): a position without a point repeats the
    previous point until the next point or the end of the period. curveType A01:
    every position has its own point, a missing position stays missing.
    Resolutions longer than 15 min are repeated for each quarter-hour, and the
    original period end is kept for first_usable_origin_utc.
    """
    rows: list[tuple[datetime, float, datetime, str]] = []
    info = {"documents": 0, "timeseries": 0, "points": 0, "resolutions": set(), "curve_types": set(), "no_data": False}
    for document in _xml_documents(content):
        root = ET.fromstring(document)
        info["documents"] += 1
        if _local(root.tag).startswith("Acknowledgement"):
            reason = " ".join(e.text or "" for e in root.iter() if _local(e.tag) == "text").strip()
            if "No matching data" in reason:
                info["no_data"] = True
                continue
            raise BuildError(f"ENTSO-E acknowledgement: {MASKER.mask(reason)[:300]}")
        for ts in (e for e in root.iter() if _local(e.tag) == "TimeSeries"):
            info["timeseries"] += 1
            curve = (_child(ts, "curveType").text if _child(ts, "curveType") is not None else "A01").strip()
            info["curve_types"].add(curve)
            for period in _children(ts, "Period"):
                interval = _child(period, "timeInterval")
                p_start = datetime.fromisoformat(_child(interval, "start").text.replace("Z", "+00:00"))
                p_end = datetime.fromisoformat(_child(interval, "end").text.replace("Z", "+00:00"))
                resolution = _child(period, "resolution").text.strip()
                info["resolutions"].add(resolution)
                step = _duration_minutes(resolution)
                if step % 15:
                    raise BuildError(f"ENTSO-E: resolution {resolution} is not a multiple of 15 min")
                n_slots = int((p_end - p_start).total_seconds() // 60) // step
                points: dict[int, tuple[float, str]] = {}
                for point in _children(period, "Point"):
                    info["points"] += 1
                    position = int(_child(point, "position").text)
                    # explicit None checks: an Element without children is falsy
                    value_el = None
                    for value_name in ("quantity", "imbalance_Price.amount", "price.amount"):
                        value_el = _child(point, value_name)
                        if value_el is not None:
                            break
                    if value_el is None or value_el.text is None:
                        continue
                    category_el = _child(point, "imbalance_Price.category")
                    category = category_el.text.strip() if category_el is not None and category_el.text else ""
                    points[position] = (float(value_el.text), category)
                current: tuple[float, str] | None = None
                for position in range(1, n_slots + 1):
                    if position in points:
                        current = points[position]
                    elif curve != "A03":
                        current = None
                    if current is None:
                        continue
                    slot_start = p_start + timedelta(minutes=(position - 1) * step)
                    slot_end = slot_start + timedelta(minutes=step)
                    for k in range(step // 15):
                        rows.append((slot_start + k * QUARTER, current[0], slot_end, current[1]))
    info["resolutions"] = sorted(info["resolutions"])
    info["curve_types"] = sorted(info["curve_types"])
    return rows, info


def fetch_entsoe(series: EntsoeSeries, chunks, client_factory, keys: KeyStore, cache: RawCache) -> list[SeriesFrame]:
    collected: list[tuple[datetime, float, datetime, str]] = []
    summary = {"documents": 0, "timeseries": 0, "points": 0, "resolutions": set(), "curve_types": set(), "no_data_chunks": 0}
    for start, end in chunks:
        param_text = "_".join(f"{k}-{v}" for k, v in sorted(series.params.items()))
        name = f"{re.sub(r'[^A-Za-z0-9._-]+', '', param_text)}_{compact(start)}_{compact(end)}.bin"
        content = cache.load("entsoe", name)
        if content is None:
            client = client_factory("entsoe")
            token = keys.get("ENTSOE_API_TOKEN")
            params = dict(series.params)
            params["periodStart"] = start.astimezone(UTC).strftime("%Y%m%d%H%M")
            params["periodEnd"] = end.astimezone(UTC).strftime("%Y%m%d%H%M")
            label = f"{series.article} {series.area} {_iso_z(start)}..{_iso_z(end)}"
            content = client.get(ENTSOE_URL, {"securityToken": token, **params}, {}, label, accept_400_no_data=True)
            cache.store("entsoe", name, content, end)
            log(f"  entsoe {series.article} {series.area} {_iso_z(start)}..{_iso_z(end)}: {len(content)} bytes fetched")
        rows, info = parse_entsoe(content)
        collected.extend(rows)
        for key in ("documents", "timeseries", "points"):
            summary[key] += info[key]
        summary["resolutions"].update(info["resolutions"])
        summary["curve_types"].update(info["curve_types"])
        summary["no_data_chunks"] += int(info["no_data"])
    summary["resolutions"] = sorted(summary["resolutions"])
    summary["curve_types"] = sorted(summary["curve_types"])
    if summary["no_data_chunks"]:
        warn(f"ENTSO-E {series.dataset_id}: no data in {summary['no_data_chunks']} chunk(s)")

    frames: list[SeriesFrame] = []
    if series.split_category:
        categories_seen = sorted({row[3] for row in collected})
        unknown = [c for c in categories_seen if c not in IMBALANCE_CATEGORY_SUFFIX]
        if unknown:
            raise BuildError(f"ENTSO-E {series.dataset_id}: unexpected imbalance price category {unknown}")
        for category, suffix in IMBALANCE_CATEGORY_SUFFIX.items():
            frame = SeriesFrame(f"{series.area.lower()}_imbalance_price_{suffix}", "entsoe", series.dataset_id,
                                series.unit, ("entsoe", series.article), series.note)
            frame.rows = [(q, v, e) for q, v, e, c in collected if c == category]
            frame.info = dict(summary, category=category)
            frames.append(frame)
    else:
        frame = SeriesFrame(series.series_id, "entsoe", series.dataset_id, series.unit,
                            ("entsoe", series.article), series.note)
        frame.rows = [(q, v, e) for q, v, e, _ in collected]
        frame.info = summary
        frames.append(frame)
    for frame in frames:
        if any(r != "PT15M" for r in summary["resolutions"]):
            frame.note = (frame.note + "; " if frame.note else "") + \
                f"source resolution {'/'.join(summary['resolutions'])} repeated to 15 min, first_usable from the original period end"
    return frames


# --------------------------------------------------------------------------------------
# Calendar (D17: official Finnish public holidays + Midsummer Eve, known in advance)
# --------------------------------------------------------------------------------------

def easter_sunday(year: int) -> date:
    """Anonymous Gregorian algorithm."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _saturday_between(year: int, month: int, first_day: int) -> date:
    start = date(year, month, first_day)
    return start + timedelta(days=(5 - start.weekday()) % 7)


def finnish_holidays(year: int) -> dict[date, str]:
    easter = easter_sunday(year)
    midsummer_day = _saturday_between(year, 6, 20)
    return {
        date(year, 1, 1): "Uudenvuodenpaiva (New Year's Day)",
        date(year, 1, 6): "Loppiainen (Epiphany)",
        easter - timedelta(days=2): "Pitkaperjantai (Good Friday)",
        easter: "Paasiaispaiva (Easter Sunday)",
        easter + timedelta(days=1): "2. paasiaispaiva (Easter Monday)",
        date(year, 5, 1): "Vappu (May Day)",
        easter + timedelta(days=39): "Helatorstai (Ascension Day)",
        easter + timedelta(days=49): "Helluntaipaiva (Whit Sunday)",
        midsummer_day - timedelta(days=1): "Juhannusaatto (Midsummer Eve)",
        midsummer_day: "Juhannuspaiva (Midsummer Day)",
        _saturday_between(year, 10, 31): "Pyhainpaiva (All Saints' Day)",
        date(year, 12, 6): "Itsenaisyyspaiva (Independence Day)",
        date(year, 12, 25): "Joulupaiva (Christmas Day)",
        date(year, 12, 26): "Tapaninpaiva (St. Stephen's Day)",
    }


def build_calendar(index: pd.DatetimeIndex) -> pd.DataFrame:
    local = index.tz_convert(TZ)
    years = sorted(set(local.year))
    holidays = set()
    for year in years:
        holidays.update(finnish_holidays(year))
    local_dates = [ts.date() for ts in local]
    offsets_h = [ts.utcoffset().total_seconds() / 3600 for ts in local]
    values = {
        "cal_weekday_local": [float(ts.weekday()) for ts in local],
        "cal_hour_local": [float(ts.hour) for ts in local],
        "cal_is_holiday_fi": [1.0 if d in holidays else 0.0 for d in local_dates],
        "cal_is_dst": [1.0 if h == 3 else 0.0 for h in offsets_h],
    }
    parts = []
    for series_id, unit in CALENDAR_SERIES.items():
        parts.append(pd.DataFrame({
            "series_id": series_id, "source": "calendar", "dataset_id": "",
            "timestamp_utc": index, "value": values[series_id], "unit": unit,
        }))
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------------------
# Nord Pool spot file (K6)
# --------------------------------------------------------------------------------------

_LOCAL_TS_PATTERNS = [
    (re.compile(r"(\d{4})-(\d{2})-(\d{2})[T ](\d{1,2}):(\d{2})(?::(\d{2}))?"), "ymd"),
    (re.compile(r"(\d{1,2})\.(\d{1,2})\.(\d{4})[ T](\d{1,2}):(\d{2})(?::(\d{2}))?"), "dmy"),
]


def _parse_local_naive(text: str, line_no: int) -> datetime:
    text = text.strip()
    for pattern, order in _LOCAL_TS_PATTERNS:
        match = pattern.fullmatch(text)
        if match:
            g = match.groups()
            if order == "ymd":
                year, month, day = int(g[0]), int(g[1]), int(g[2])
            else:
                day, month, year = int(g[0]), int(g[1]), int(g[2])
            try:
                return datetime(year, month, day, int(g[3]), int(g[4]), int(g[5] or 0))
            except ValueError:
                break
    raise SpotFileError(f"spot file line {line_no}: timestamp '{text}' is not a local time without time zone "
                        "(expected e.g. 2026-10-25 03:15 or 25.10.2026 03:15).")


def local_to_utc_by_order(naive: list[datetime], line_numbers: list[int]) -> tuple[list[datetime], list[bool]]:
    """Convert Europe/Helsinki local times to UTC using the row order.

    In the repeated hour of the autumn change (25.10.2026 03:00-03:45) the first
    occurrence is UTC+3 (summer time) and the second UTC+2 (winter time).
    """
    seen: dict[datetime, int] = {}
    out: list[datetime] = []
    ambiguous_flags: list[bool] = []
    for ts, line_no in zip(naive, line_numbers):
        first = ts.replace(tzinfo=TZ, fold=0)
        second = ts.replace(tzinfo=TZ, fold=1)
        if first.utcoffset() != second.utcoffset():
            back = first.astimezone(UTC).astimezone(TZ).replace(tzinfo=None)
            if back != ts:
                raise SpotFileError(f"spot file line {line_no}: local time {ts:%Y-%m-%d %H:%M} does not exist "
                                    "(spring daylight saving time change).")
            count = seen.get(ts, 0)
            if count >= 2:
                raise SpotFileError(f"spot file line {line_no}: local time {ts:%Y-%m-%d %H:%M} occurs more than twice.")
            seen[ts] = count + 1
            out.append(ts.replace(tzinfo=TZ, fold=count).astimezone(UTC))
            ambiguous_flags.append(True)
        else:
            out.append(first.astimezone(UTC))
            ambiguous_flags.append(False)
    return out, ambiguous_flags


def _expected_quarters(local_day: date) -> int:
    start = datetime(local_day.year, local_day.month, local_day.day, tzinfo=TZ).astimezone(UTC)
    nxt = local_day + timedelta(days=1)
    end = datetime(nxt.year, nxt.month, nxt.day, tzinfo=TZ).astimezone(UTC)
    return int((end - start) / QUARTER)


def read_spot_file(path: Path, separator: str = "auto", decimal: str = "auto",
                   timestamp_position: str = "start") -> tuple[pd.DataFrame, dict]:
    """Read a Nord Pool spot CSV with Europe/Helsinki local timestamps without time zone.

    Returns a frame with timestamp_utc (quarter-hour start) and value, and an info dict.
    Stops with SpotFileError if a day does not have the expected number of
    quarter-hours (100 on 25.10.2026, otherwise 96), the UTC series has a gap or a
    duplicate, or the rows of the repeated hour are not in chronological order.
    """
    if not path.is_file():
        raise SpotFileError(f"spot file not found: {path}")
    try:
        text = path.read_text(encoding="utf-8-sig")  # a UTF-8 byte order mark is removed
    except UnicodeDecodeError as exc:
        raise SpotFileError(f"spot file is not UTF-8 ({exc.reason} at byte {exc.start}). "
                            "Save it in Excel with Save As -> CSV UTF-8 (Comma delimited) and run again.") from None
    lines = text.splitlines()
    if not lines:
        raise SpotFileError("spot file is empty.")
    if separator == "auto":
        separator = ";" if lines[0].count(";") > 0 else ","
    reader = csv.reader(io.StringIO(text), delimiter=separator)
    rows = list(reader)
    header, body = rows[0], rows[1:]
    if len(header) < 2:
        raise SpotFileError(f"spot file header has {len(header)} column(s) with separator '{separator}'; "
                            "expected a timestamp column and a price column.")
    body_numbered = [(i + 2, r) for i, r in enumerate(body) if any(c.strip() for c in r)]
    if not body_numbered:
        raise SpotFileError("spot file has no data rows.")
    if decimal == "auto":
        sample = [r[1].strip() for _, r in body_numbered if len(r) > 1]
        decimal = "," if any(re.fullmatch(r"-?\d+,\d+", s) for s in sample) else "."
    if decimal == "," and separator == ",":
        raise SpotFileError("separator and decimal separator are both ','.")

    naive, values, line_numbers = [], [], []
    for line_no, row in body_numbered:
        if len(row) < 2:
            raise SpotFileError(f"spot file line {line_no}: expected at least 2 columns.")
        ts_text, value_text = row[0], row[1].strip()
        if re.search(r"(Z|[+-]\d{2}:?\d{2})\s*$", ts_text.strip()):
            raise SpotFileError(f"spot file line {line_no}: timestamp '{ts_text.strip()}' has a time zone; "
                                "local time without time zone is expected.")
        naive.append(_parse_local_naive(ts_text, line_no))
        if not value_text:
            raise SpotFileError(f"spot file line {line_no}: empty price.")
        normalised = value_text.replace(" ", "")
        if decimal == ",":
            normalised = normalised.replace(".", "").replace(",", ".") if "," in normalised else normalised
        try:
            values.append(float(normalised))
        except ValueError:
            raise SpotFileError(f"spot file line {line_no}: price '{value_text}' is not a number "
                                f"(decimal separator '{decimal}').") from None
        line_numbers.append(line_no)

    utc, ambiguous = local_to_utc_by_order(naive, line_numbers)
    if timestamp_position == "end":
        utc = [t - QUARTER for t in utc]

    # 1. Quarter-hours per local day (delivery start)
    counts: dict[date, int] = {}
    for t in utc:
        d = t.astimezone(TZ).date()
        counts[d] = counts.get(d, 0) + 1
    for d in sorted(counts):
        expected = _expected_quarters(d)
        if counts[d] != expected:
            hint = " (the repeated hour 03:00-03:45 must appear twice)" if expected == 100 else ""
            raise SpotFileError(f"spot file: {d:%d.%m.%Y} has {counts[d]} quarter-hours, expected {expected}{hint}.")

    # 2. Row order, duplicates and gaps in the UTC series
    for i in range(1, len(utc)):
        delta = utc[i] - utc[i - 1]
        if delta == QUARTER:
            continue
        where = f"lines {line_numbers[i - 1]}-{line_numbers[i]}"
        if ambiguous[i] or ambiguous[i - 1]:
            # the day count is already correct here, so a jump inside the repeated hour is an ordering problem
            raise SpotFileError(f"spot file {where}: the rows of the repeated hour (25.10. 03:00-03:45) are not "
                                "in chronological order (first all four summer time quarters, then the four "
                                "winter time quarters).")
        if delta <= timedelta(0):
            if delta == timedelta(0):
                raise SpotFileError(f"spot file {where}: duplicate quarter-hour {fmt_utc(utc[i])}.")
            raise SpotFileError(f"spot file {where}: rows are not in chronological order.")
        raise SpotFileError(f"spot file {where}: gap between {fmt_utc(utc[i - 1])} and {fmt_utc(utc[i])}.")

    frame = pd.DataFrame({"timestamp_utc": pd.to_datetime(utc, utc=True), "value": values})
    info = {"separator": separator, "decimal": decimal, "timestamp_position": timestamp_position,
            "rows": len(frame), "days": len(counts), "repeated_hour_rows": sum(ambiguous)}
    return frame, info


# --------------------------------------------------------------------------------------
# Assembly, quality report and output
# --------------------------------------------------------------------------------------

def frame_to_rows(frame: SeriesFrame, window_index: pd.DatetimeIndex, rules: list[AvailabilityRule]
                  ) -> tuple[pd.DataFrame, dict]:
    """Deduplicate, clip to the window and compute first_usable_origin_utc."""
    if frame.rows:
        df = pd.DataFrame(frame.rows, columns=["timestamp_utc", "value", "period_end"])
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        df["period_end"] = pd.to_datetime(df["period_end"], utc=True)
    else:
        df = pd.DataFrame({"timestamp_utc": pd.Series(dtype="datetime64[ns, UTC]"),
                           "value": pd.Series(dtype=float),
                           "period_end": pd.Series(dtype="datetime64[ns, UTC]")})
    df = df[df["timestamp_utc"].isin(window_index)]
    # Duplicates (same timestamp_utc): identical rows (same value and same period_end) are removed and
    # counted; rows with different values or period ends stop the run, because the correct one is unknown.
    dup_mask = df.duplicated("timestamp_utc", keep="last")
    duplicates_identical = 0
    if dup_mask.any():
        dup_all = df[df.duplicated("timestamp_utc", keep=False)]
        groups = dup_all.groupby("timestamp_utc")
        value_conflict = groups["value"].apply(lambda v: v.isna().nunique() > 1 or v.dropna().nunique() > 1)
        end_conflict = groups["period_end"].nunique(dropna=False).gt(1)
        conflict = value_conflict | end_conflict
        if conflict.any():
            first = conflict[conflict].index.min()
            values = dup_all.loc[dup_all["timestamp_utc"] == first, ["value", "period_end"]]
            shown = "; ".join(f"value={v} period_end={fmt_utc(e)}" for v, e in values.itertuples(index=False))
            raise BuildError(f"{frame.series_id}: {int(conflict.sum())} quarter-hour(s) have duplicate rows with "
                             f"different values or period ends. First: {fmt_utc(first)} ({shown}).")
        duplicates_identical = int(dup_mask.sum())
        df = df[~dup_mask]
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    rule = find_rule(rules, *frame.rule_key)
    df["first_usable_origin_utc"] = compute_first_usable(df["timestamp_utc"], df["period_end"], rule)
    df["series_id"] = frame.series_id
    df["source"] = frame.source
    df["dataset_id"] = frame.dataset_id
    df["unit"] = frame.unit
    return df[OUTPUT_COLUMNS], {"duplicates_removed_identical": duplicates_identical}


def quality_row(series_id: str, source: str, dataset_id: str, df: pd.DataFrame, window_index: pd.DatetimeIndex,
                duplicates_removed_identical: int, note: str, info: dict | None = None) -> dict:
    present = pd.DatetimeIndex(df["timestamp_utc"]) if len(df) else pd.DatetimeIndex([], tz="UTC")
    missing = window_index.difference(present)
    runs, longest = 0, 0
    if len(missing):
        positions = window_index.get_indexer(missing)
        breaks = (pd.Series(positions).diff().fillna(2) != 1)
        runs = int(breaks.sum())
        run_ids = breaks.cumsum()
        longest = int(run_ids.value_counts().max())
    row = {
        "series_id": series_id, "source": source, "dataset_id": dataset_id,
        "rows": len(df), "expected_rows": len(window_index),
        "missing_rows": len(missing), "gap_runs": runs, "longest_gap_quarters": longest,
        "first_missing_utc": fmt_utc(missing[0]) if len(missing) else "",
        "duplicates_removed_identical": duplicates_removed_identical,
        "first_timestamp_utc": fmt_utc(present.min()) if len(present) else "",
        "last_timestamp_utc": fmt_utc(present.max()) if len(present) else "",
        "first_usable_min_utc": fmt_utc(df["first_usable_origin_utc"].min()) if len(df) else "",
        "first_usable_max_utc": fmt_utc(df["first_usable_origin_utc"].max()) if len(df) else "",
        "note": note,
    }
    if info:
        row["source_info"] = json.dumps(info, default=str, sort_keys=True)
    return row


def build_actuals(period: tuple[datetime, datetime], window_index: pd.DatetimeIndex,
                  imbalance: pd.DataFrame | None, spot: pd.DataFrame | None, direction: pd.DataFrame | None
                  ) -> pd.DataFrame:
    period_index = quarter_index(period[0], period[1] + QUARTER)
    index = period_index.intersection(window_index)
    out = pd.DataFrame({"delivery_start_utc": index})
    usable_cols = []
    for name, src in (("imbalance_price", imbalance), ("spot_price", spot), ("direction", direction)):
        if src is None or src.empty:
            out[name] = pd.NA
            continue
        part = src[["timestamp_utc", "value", "first_usable_origin_utc"]].rename(
            columns={"timestamp_utc": "delivery_start_utc", "value": name, "first_usable_origin_utc": f"fu_{name}"})
        out = out.merge(part, on="delivery_start_utc", how="left")
        usable_cols.append(f"fu_{name}")
    if direction is not None and not direction.empty:
        unknown = sorted(set(out["direction"].dropna().unique()) - set(DIRECTION_MAP))
        if unknown:
            raise BuildError(f"Fingrid 369: unexpected direction value(s) {unknown}")
        out["direction"] = out["direction"].map(lambda v: DIRECTION_MAP.get(int(v)) if pd.notna(v) else pd.NA)
    # Specification 7.2: the latest first_usable_origin_utc of the row's sources
    out["first_usable_origin_utc"] = out[usable_cols].max(axis=1) if usable_cols else pd.NaT
    return out[ACTUALS_COLUMNS]


def _format_long(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp_utc"] = out["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out["first_usable_origin_utc"] = out["first_usable_origin_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return out


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, float_format="%.10g", lineterminator="\n", encoding="utf-8")
    os.replace(tmp, path)


def write_actuals(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    for col in ("delivery_start_utc", "first_usable_origin_utc"):
        out[col] = pd.to_datetime(out[col], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    _write_csv(out, path)


def write_parquet(df: pd.DataFrame, path: Path) -> bool:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        warn("pyarrow is not installed: inputs.parquet was not written (inputs.csv has the same data).")
        return False
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return True


def write_quality_report(rows: list[dict], out_dir: Path, run_info: dict) -> None:
    report = pd.DataFrame(rows)
    _write_csv(report, out_dir / "quality_report.csv")  # source_info = JSON: source periods, resolutions, curve types
    lines = [
        "# Data quality report",
        "",
        f"- build_dataset.py version {SCRIPT_VERSION}, schema {SCHEMA_VERSION}",
        f"- Run (UTC): {run_info['run_utc']}",
        f"- Window: {run_info['window_start']} .. {run_info['window_last']} (quarter-hour starts, "
        f"{run_info['window_quarters']} quarter-hours)",
        f"- Spot file: {run_info['spot']}",
        f"- API calls: {run_info['api_calls']}; cache hits {run_info['cache_hits']}, new responses {run_info['cache_misses']}",
        "",
        "| series_id | source | dataset_id | rows | expected | missing | gap runs | longest gap (q) | duplicates removed (identical) | first | last | note |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['series_id']} | {r['source']} | {r['dataset_id']} | {r['rows']} | {r['expected_rows']} | "
                     f"{r['missing_rows']} | {r['gap_runs']} | {r['longest_gap_quarters']} | {r['duplicates_removed_identical']} | "
                     f"{r['first_timestamp_utc']} | {r['last_timestamp_utc']} | {r['note']} |")
    if WARNINGS:
        lines += ["", "## Warnings", ""] + [f"- {w}" for w in WARNINGS]
    path = out_dir / "quality_report.md"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def parse_args(argv: list[str] | None) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Fetch Fingrid and ENTSO-E data, read the Nord Pool spot file and build the starter "
                    "package data files with first_usable_origin_utc.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("Dates (YYYY-MM-DD) are local Europe/Helsinki dates and the end date is included. "
                "UTC timestamps (YYYY-MM-DDTHH:MM:SSZ) are quarter-hour starts; --end is the last one.\n"
                "Keys: environment variables FINGRID_API_KEY and ENTSOE_API_TOKEN, or --env-file."))
    parser.add_argument("--start", default=fmt_utc(DATA_PERIOD_START), help="window start (default: %(default)s)")
    parser.add_argument("--end", default=fmt_utc(DATA_PERIOD_LAST), help="window end, included (default: %(default)s)")
    parser.add_argument("--env-file", type=Path, help="file with KEY=VALUE lines for the API keys (no default)")
    parser.add_argument("--spot-file", type=Path, help="Nord Pool spot price CSV (local time without time zone)")
    parser.add_argument("--spot-separator", choices=["auto", ";", ","], default="auto",
                        help="spot file column separator (default: %(default)s)")
    parser.add_argument("--spot-decimal", choices=["auto", ",", "."], default="auto",
                        help="spot file decimal separator (default: %(default)s)")
    parser.add_argument("--spot-timestamp-position", choices=["start", "end"], default="start",
                        help="spot file timestamp is the start or the end of the quarter-hour (default: %(default)s)")
    parser.add_argument("--entsoe-capacity-publish-time", metavar="HH:MM",
                        help="publication time (D-1, Europe/Helsinki) of the ENTSO-E FI-EE capacities; "
                             "default: the value in availability.csv (14:00, D17)")
    parser.add_argument("--availability", type=Path, default=script_dir / "availability.csv",
                        help="availability file (default: availability.csv next to this script)")
    parser.add_argument("--out-dir", type=Path, default=script_dir / "data",
                        help="output folder for the team files (default: data/ next to this script)")
    parser.add_argument("--organizer-dir", type=Path, default=script_dir.parent / "organizer_only",
                        help="output folder for actuals_test.csv (default: ../organizer_only)")
    parser.add_argument("--cache-dir", type=Path, default=script_dir.parent / "data_cache",
                        help="raw API response cache (default: ../data_cache)")
    parser.add_argument("--sources", default="fingrid,entsoe,calendar",
                        help="comma separated sources to include (default: %(default)s)")
    parser.add_argument("--refresh", action="store_true", help="ignore the cache and fetch everything again")
    parser.add_argument("--version", action="version", version=f"%(prog)s {SCRIPT_VERSION} (schema {SCHEMA_VERSION})")
    args = parser.parse_args(argv)
    if args.entsoe_capacity_publish_time and not re.fullmatch(r"\d{2}:\d{2}", args.entsoe_capacity_publish_time):
        parser.error("--entsoe-capacity-publish-time must be HH:MM")
    return args


def run(args: argparse.Namespace) -> int:
    run_utc = datetime.now(UTC)
    start = parse_window_arg(args.start, is_end=False)
    end_excl = parse_window_arg(args.end, is_end=True)
    if end_excl <= start:
        raise BuildError("--end must be after --start.")
    if start < DATA_PERIOD_START or end_excl - QUARTER > DATA_PERIOD_LAST:
        warn("the window extends outside the data period of the specification (2026-05-31T21:00Z..2026-11-02T06:45Z).")
    sources = {s.strip() for s in args.sources.split(",") if s.strip()}
    unknown = sources - {"fingrid", "entsoe", "calendar"}
    if unknown:
        raise BuildError(f"unknown source(s) in --sources: {', '.join(sorted(unknown))}")

    window_index = quarter_index(start, end_excl)
    chunks = local_month_chunks(start, end_excl)
    log(f"build_dataset.py {SCRIPT_VERSION}: window {fmt_utc(start)} .. {fmt_utc(end_excl - QUARTER)} "
        f"({len(window_index)} quarter-hours, {len(chunks)} chunk(s))")

    keys = KeyStore(args.env_file)
    rules = load_availability(args.availability, args.entsoe_capacity_publish_time)
    cache = RawCache(args.cache_dir, args.refresh)
    clients: dict[str, RateLimitedClient] = {}

    def client_factory(name: str) -> RateLimitedClient:
        if name not in clients:
            interval = FINGRID_MIN_INTERVAL_S if name == "fingrid" else ENTSOE_MIN_INTERVAL_S
            clients[name] = RateLimitedClient(name, interval)
        return clients[name]

    long_parts: list[pd.DataFrame] = []
    quality: list[dict] = []
    by_series: dict[str, pd.DataFrame] = {}

    if "fingrid" in sources:
        log("Fingrid:")
        for series in FINGRID_SERIES:
            raw_rows = fetch_fingrid(series, chunks, client_factory, keys, cache)
            frame = expand_fingrid_rows(series, raw_rows)
            df, dup = frame_to_rows(frame, window_index, rules)
            long_parts.append(df)
            by_series[series.series_id] = df
            quality.append(quality_row(series.series_id, "fingrid", str(series.dataset_id), df, window_index,
                                       dup["duplicates_removed_identical"], series.note, frame.info))
            durations = frame.info["source_durations_min"]
            if set(durations) - {15}:
                warn(f"fingrid {series.dataset_id}: source periods (endTime - startTime) {durations} min; "
                     "rows longer than 15 min were repeated per quarter-hour.")

    if "entsoe" in sources:
        log("ENTSO-E:")
        for series in ENTSOE_SERIES:
            for frame in fetch_entsoe(series, chunks, client_factory, keys, cache):
                df, dup = frame_to_rows(frame, window_index, rules)
                long_parts.append(df)
                by_series[frame.series_id] = df
                quality.append(quality_row(frame.series_id, "entsoe", frame.dataset_id, df, window_index,
                                           dup["duplicates_removed_identical"], frame.note, frame.info))

    if "calendar" in sources:
        cal = build_calendar(window_index)
        cal["first_usable_origin_utc"] = pd.Timestamp(DATA_PERIOD_START)  # calendar is known in advance (D17)
        cal = cal[OUTPUT_COLUMNS]
        long_parts.append(cal)
        for series_id in CALENDAR_SERIES:
            part = cal[cal["series_id"] == series_id]
            quality.append(quality_row(series_id, "calendar", "", part, window_index, 0,
                                       "known in advance: first_usable_origin_utc = data period start (D17)"))

    # Spot (Nord Pool, D3): separate file
    spot_df = None
    spot_text = "not given"
    if args.spot_file:
        spot_raw, spot_info = read_spot_file(args.spot_file, args.spot_separator, args.spot_decimal,
                                             args.spot_timestamp_position)
        spot_raw = spot_raw[spot_raw["timestamp_utc"].isin(window_index)].reset_index(drop=True)
        rule = find_rule(rules, "nordpool", None)
        spot_raw["first_usable_origin_utc"] = compute_first_usable(spot_raw["timestamp_utc"], spot_raw["timestamp_utc"] + QUARTER, rule)
        spot_raw["series_id"] = "fi_spot_price"
        spot_raw["source"] = "nordpool"
        spot_raw["dataset_id"] = ""
        spot_raw["unit"] = "EUR/MWh"
        spot_df = spot_raw[OUTPUT_COLUMNS]
        spot_text = f"{args.spot_file.name} ({json.dumps(spot_info)})"
        quality.append(quality_row("fi_spot_price", "nordpool", "", spot_df, window_index, 0,
                                   "Nord Pool, Turku Energia licence (D3)", spot_info))
    else:
        warn("--spot-file not given: spot_nordpool.csv is not written and spot_price in the actuals files is "
             "empty. Exceptional price metrics cannot be scored without it (D5).")

    # Output
    args.out_dir.mkdir(parents=True, exist_ok=True)
    inputs = pd.concat(long_parts, ignore_index=True) if long_parts else pd.DataFrame(columns=OUTPUT_COLUMNS)
    inputs = inputs.sort_values(["series_id", "timestamp_utc"]).reset_index(drop=True)
    write_parquet(inputs, args.out_dir / "inputs.parquet")
    _write_csv(_format_long(inputs), args.out_dir / "inputs.csv")
    if spot_df is not None:
        _write_csv(_format_long(spot_df), args.out_dir / "spot_nordpool.csv")

    imbalance = by_series.get("fi_imbalance_price")
    direction = by_series.get("fi_mfrr_direction")
    actuals_val = build_actuals(VALIDATION_PERIOD, window_index, imbalance, spot_df, direction)
    actuals_test = build_actuals(TEST_PERIOD, window_index, imbalance, spot_df, direction)
    if actuals_val.empty:
        warn("the window does not overlap the validation period: actuals_validation.csv has only the header row.")
    write_actuals(actuals_val, args.out_dir / "actuals_validation.csv")
    write_actuals(actuals_test, args.organizer_dir / "actuals_test.csv")
    for name, df in (("actuals_validation.csv", actuals_val), ("actuals_test.csv", actuals_test)):
        quality.append({
            "series_id": name, "source": "actuals", "dataset_id": "319;369;nordpool",
            "rows": len(df), "expected_rows": len(df),
            "missing_rows": int(df["imbalance_price"].isna().sum()), "gap_runs": "", "longest_gap_quarters": "",
            "first_missing_utc": "", "duplicates_removed_identical": int(df["delivery_start_utc"].duplicated().sum()),
            "first_timestamp_utc": fmt_utc(df["delivery_start_utc"].min()) if len(df) else "",
            "last_timestamp_utc": fmt_utc(df["delivery_start_utc"].max()) if len(df) else "",
            "first_usable_min_utc": "", "first_usable_max_utc": "",
            "note": (f"missing = rows without imbalance_price; empty spot_price {int(df['spot_price'].isna().sum())}, "
                     f"empty direction {int(df['direction'].isna().sum())}"),
        })

    run_info = {
        "run_utc": fmt_utc(run_utc), "window_start": fmt_utc(start), "window_last": fmt_utc(end_excl - QUARTER),
        "window_quarters": len(window_index), "spot": spot_text,
        "api_calls": sum(c.calls for c in clients.values()), "cache_hits": cache.hits, "cache_misses": cache.misses,
    }
    write_quality_report(quality, args.out_dir, run_info)
    log(f"Done: {len(inputs)} input rows in {inputs['series_id'].nunique() if len(inputs) else 0} series; "
        f"actuals_validation {len(actuals_val)} rows, actuals_test {len(actuals_test)} rows; "
        f"{len(WARNINGS)} warning(s). API calls {run_info['api_calls']}, cache hits {cache.hits}.")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except BuildError as exc:
        print(f"ERROR: {MASKER.mask(exc)}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted. Cached responses are kept; run again to continue.", file=sys.stderr)
        return 130
    except Exception:  # unexpected: print a masked traceback without the exception chain
        print("UNEXPECTED ERROR:\n" + MASKER.mask(traceback.format_exc(chain=False)), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
