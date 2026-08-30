"""Persistence: a jsonl archive per run, plus a sqlite index over them.

Two stores, because they answer different questions:

* ``runs/<run_id>.jsonl`` is the **archive** and the source of truth. Line 0 is
  the manifest and variant; every later line is one replicate. Append-only, plain
  text, greppable, and readable by anything - a recorded trajectory outlives the
  code that produced it, which is the whole point of recording it.
* ``runs/index.db`` is a **cache** for listing and lookup. Every column in it is
  recomputable from the archive, and ``reindex()`` does exactly that. Nothing that
  takes judgement - flake bands, verdicts, noise floors - is stored anywhere:
  those are recomputed on read, so improving the analysis never means re-running
  the agent.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

from .runner import RunResult
from .types import SCHEMA_VERSION

DEFAULT_ROOT = "runs"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id         TEXT PRIMARY KEY,
    created_at     REAL,
    variant_id     TEXT,
    variant_digest TEXT,
    suite_digest   TEXT,
    agent          TEXT,
    model          TEXT,
    effort         TEXT,
    replicates     INTEGER,
    n_tasks        INTEGER,
    passes         INTEGER,
    cells          INTEGER,
    cost_usd       REAL,
    notes          TEXT,
    path           TEXT
);
CREATE TABLE IF NOT EXISTS task_stats (
    run_id  TEXT,
    task_id TEXT,
    passes  INTEGER,
    n       INTEGER,
    PRIMARY KEY (run_id, task_id)
);
CREATE INDEX IF NOT EXISTS runs_variant ON runs (variant_id, created_at DESC);
"""

def write_jsonl(path, rows: Iterable[dict]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return p


def append_jsonl(path, row: dict) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return p


def read_jsonl(path) -> Iterator[dict]:
    """Skips blank lines, and names the line number when one will not parse -
    a half-written archive should tell you which line to look at."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{p}:{lineno}: {exc}") from exc


@dataclass
class RunRow:
    """One row of the index - enough to list and choose, not to analyse."""

    run_id: str
    created_at: float
    variant_id: str
    variant_digest: str
    suite_digest: str
    agent: str
    model: str
    effort: str
    replicates: int
    n_tasks: int
    passes: int
    cells: int
    cost_usd: float
    notes: str
    path: str

    @property
    def pass_rate(self) -> float:
        return self.passes / self.cells if self.cells else 0.0

    def when(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(self.created_at))

    def line(self) -> str:
        return (f"{self.run_id:<34} {self.when()}  {self.variant_id:<10} "
                f"{self.n_tasks:>3}t x{self.replicates:<3} "
                f"{self.passes}/{self.cells} ({self.pass_rate:.0%})"
                + (f"  ${self.cost_usd:.2f}" if self.cost_usd else ""))

class Store:
    def __init__(self, root: str | Path = DEFAULT_ROOT) -> None:
        self.root = Path(root)
        self.db_path = self.root / "index.db"

    # -- archive ----------------------------------------------------------
    def path_for(self, run_id: str) -> Path:
        return self.root / f"{run_id}.jsonl"

    def save(self, run: RunResult) -> Path:
        path = self.path_for(run.manifest.run_id)
        header = {"kind": "run", "schema": SCHEMA_VERSION,
                  "manifest": run.manifest.to_dict(),
                  "variant": run.variant.to_dict()}
        rows = [header] + [{"kind": "result", **r.to_dict()} for r in run.results]
        write_jsonl(path, rows)
        self.index(run, path)
        return path

    def load_path(self, path) -> RunResult:
        rows = list(read_jsonl(path))
        if not rows or rows[0].get("kind") != "run":
            raise ValueError(f"{path}: first line is not a run header")
        head = rows[0]
        if head.get("schema") != SCHEMA_VERSION:
            raise ValueError(
                f"{path}: recorded with schema {head.get('schema')}, this build "
                f"reads {SCHEMA_VERSION}")
        return RunResult.from_dict({
            "manifest": head["manifest"], "variant": head["variant"],
            "results": [r for r in rows[1:] if r.get("kind") == "result"]})

    def load(self, run_id: str) -> RunResult:
        path = self.path_for(run_id)
        if not path.exists():
            raise FileNotFoundError(f"no run {run_id!r} under {self.root}")
        return self.load_path(path)

    # -- index ------------------------------------------------------------
    def connect(self) -> sqlite3.Connection:
        self.root.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.executescript(SCHEMA)
        return conn

    def index(self, run: RunResult, path: Path) -> None:
        counts = run.pass_counts()
        passes = sum(p for p, _ in counts.values())
        cells = sum(n for _, n in counts.values())
        with self.connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (run.manifest.run_id, run.manifest.created_at,
                 run.manifest.variant_id, run.manifest.variant_digest,
                 run.manifest.suite_digest, run.manifest.agent,
                 run.manifest.model, run.manifest.effort,
                 run.manifest.replicates, len(counts), passes, cells,
                 run.total_cost_usd(), run.manifest.notes, str(path)))
            conn.execute("DELETE FROM task_stats WHERE run_id = ?",
                         (run.manifest.run_id,))
            conn.executemany(
                "INSERT INTO task_stats VALUES (?,?,?,?)",
                [(run.manifest.run_id, t, p, n) for t, (p, n) in counts.items()])

    def runs(self, variant_id: Optional[str] = None,
             limit: Optional[int] = None) -> list[RunRow]:
        sql = "SELECT * FROM runs"
        args: list = []
        if variant_id:
            sql += " WHERE variant_id = ?"
            args.append(variant_id)
        sql += " ORDER BY created_at DESC"
        if limit:
            sql += " LIMIT ?"
            args.append(limit)
        with self.connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, args).fetchall()
        return [RunRow(**dict(r)) for r in rows]

    def latest(self, variant_id: Optional[str] = None) -> Optional[RunRow]:
        rows = self.runs(variant_id, limit=1)
        return rows[0] if rows else None

    def resolve(self, ref: str) -> RunResult:
        """Accepts a run id, a variant id (meaning "its most recent run"), or a
        path. Ambiguity resolves toward the run id, since that is the exact one."""
        if not ref:
            raise ValueError("empty run reference")
        path = Path(ref)
        if path.exists() and path.is_file():
            return self.load_path(path)
        if self.path_for(ref).exists():
            return self.load(ref)
        row = self.latest(ref)
        if row is None:
            raise FileNotFoundError(
                f"{ref!r} is not a run id, a variant with a recorded run, or a "
                f"file under {self.root}")
        return self.load_path(row.path)

    def reindex(self) -> int:
        """Rebuild the index from the archive. The archive wins on disagreement."""
        with self.connect() as conn:
            conn.execute("DELETE FROM runs")
            conn.execute("DELETE FROM task_stats")
        n = 0
        for path in sorted(self.root.glob("*.jsonl")):
            try:
                run = self.load_path(path)
            except Exception:  # a corrupt archive should not stop the rebuild
                continue
            self.index(run, path)
            n += 1
        return n

    def pass_counts(self, run_id: str) -> dict[str, tuple[int, int]]:
        """Per-task counts straight from the index, without loading trajectories."""
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT task_id, passes, n FROM task_stats WHERE run_id = ? "
                "ORDER BY task_id", (run_id,)).fetchall()
        return {t: (p, n) for t, p, n in rows}
