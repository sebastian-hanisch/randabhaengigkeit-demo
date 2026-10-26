"""Rechnet das Raster der Messreihen vor (E2 bis E4) und schreibt data/tail_grid.json.

    python tools/make_grid.py            # volles Raster, mehrere Prozesse
    python tools/make_grid.py --quick    # kleines Probe-Raster für Tests (nicht in data/tail_grid.json)

Das Raster ist deterministisch (SplitMix64, feste Seeds); tests/test_grid.py rechnet eine feste Probe nach und vergleicht sie mit der Datei.
Die exakten Kurven der bedingten Wahrscheinlichkeit (E1, E5) stehen nicht im Raster: sie werden in der App per Quadratur gerechnet."""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tdp_constants as C  # noqa: E402
import tdp_evaluation as ev  # noqa: E402

JOBS = {"sum": ev.sum_cell, "rank": ev.rank_cell, "estimate": ev.estimate_cell}


def _run(task):
    kind, args = task
    return kind, JOBS[kind](*args)


def main() -> None:
    quick = "--quick" in sys.argv
    tasks = [("estimate", (n, share)) for n in sorted(C.EST_NS, reverse=True) for share in C.EST_SHARES]     # die teuren Zellen zuerst
    tasks += [("sum", (rho, nu)) for rho in C.SUM_RHOS for nu in C.SUM_NUS]
    tasks += [("rank", (rho, nu)) for rho in C.RANK_RHOS for nu in C.RANK_NUS]
    if quick:
        tasks = [("sum", (0.5, 3, 20_000, (7,))), ("rank", (0.5, 3, 5_000, 1_000, (7,))), ("estimate", (120, 0.05, C.EST_RHO, C.EST_NU, 20, (7,)))]
    t0 = time.time()
    out = {k: [] for k in JOBS}
    with ProcessPoolExecutor(max_workers=10) as pool:
        for kind, row in pool.map(_run, tasks, chunksize=1):
            out[kind].append(row)
    grid = {"meta": {"seeds": list(C.GRID_SEEDS), "seconds": round(time.time() - t0)}, **out, "probe": ev.grid_probe()}
    dest = ROOT / "data" / ("tail_grid_quick.json" if quick else "tail_grid.json")
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(grid, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{dest} geschrieben: " + ", ".join(f"{len(v)} {k}" for k, v in out.items()) + f", {grid['meta']['seconds']} s")


if __name__ == "__main__":
    main()
