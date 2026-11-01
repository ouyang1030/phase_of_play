"""Match metadata: meta.json, exported by notebooks/preprocessing.ipynb from the databallpy Game.

  {"match_id": "2561107",
   "home_team": {"name": "Ajax", "id": 215}, "away_team": {"name": "PSV", "id": 204},
   "period_start_frame": {"1": 0, "2": 95369},     # kick-off, 0-based row of the fused 25 Hz CSV
   "period_end_frame": {"1": 69023, "2": 168828},  # optional, last frame of each period
   "pitch": {"length": 105.0, "width": 68.0},     # m
   "frame_rate": 25.0,                             # Hz of the fused CSV
   "prefix": "ajax_psv_full",                      # optional, default "<home>_<away>_full" (lower case)
   "players": [{"column": "home_home_26", "team": "home", "shirt_num": 26, "name": "...",
                "position": "goalkeeper", "is_gk": true}, ...]}   # optional: roster for phase.prep

Tracking team 0 (home, the labels' reference team) is meta home_team; its id is the event-data team_id.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REQUIRED = ("match_id", "home_team", "away_team", "period_start_frame", "pitch", "frame_rate")


def load(path: str | Path) -> dict:
    m = json.loads(Path(path).read_text())
    missing = [k for k in REQUIRED if k not in m]
    if missing:
        raise ValueError(f"{path}: missing keys {missing} (re-export it with notebooks/preprocessing.ipynb)")
    for side in ("home_team", "away_team"):
        if not {"name", "id"} <= set(m[side]):
            raise ValueError(f"{path}: {side} needs name and id")
    for k in ("period_start_frame", "period_end_frame"):
        if k in m:
            m[k] = {int(p): int(f) for p, f in m[k].items()}
    return m


def teams(meta: dict) -> tuple[str, str]:
    """(home name, away name); home = tracking team 0."""
    return meta["home_team"]["name"], meta["away_team"]["name"]


def slug(name: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", str(name).lower()).strip("_")


def prefix(meta: dict) -> str:
    """File prefix of the label npz files: meta 'prefix', else '<home>_<away>_full' ('ajax_psv_full')."""
    return meta.get("prefix") or f"{slug(meta['home_team']['name'])}_{slug(meta['away_team']['name'])}_full"


def periods(meta: dict) -> dict | None:
    """{period: (kick-off frame, last frame)} (0-based rows of the fused CSV, inclusive) for phase.prep, or None if
    meta has no period_end_frame."""
    if not meta.get("period_end_frame"):
        return None
    start, end = meta["period_start_frame"], meta["period_end_frame"]
    missing = sorted(set(start) ^ set(end))
    if missing:
        raise ValueError(f"periods {missing} have only a start or only an end frame in meta.json")
    return {p: (start[p], end[p]) for p in sorted(start)}


def roster(meta: dict) -> dict:
    """Player column of the fused CSV ('home_home_26': the number is the shirt number) -> is goalkeeper."""
    if not meta.get("players"):
        raise ValueError("meta has no players: re-export it with notebooks/preprocessing.ipynb")
    return {p["column"]: bool(p["is_gk"]) for p in meta["players"]}
