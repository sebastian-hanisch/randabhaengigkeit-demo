"""Konstanten: Regler-Grenzen, Kopulas, Raster der Messreihen, Farben und Presets.

Zufallserzeuger, Pearson und Spearman kommen als wörtliche Kopien aus pearson-demo (prs_rng.py, prs_constants.py, prs_scenario.py, prs_algorithm.py).
Rechenbudget am Regler-Maximum (DEMO-PLAYBOOK §1c): ein Datensatz mit n = 5 000 Paaren für beide Kopulas (t-Verteilungsfunktion als endliche Summe, Normal-Quantil als Näherung mit
einem Newton-Schritt) braucht Sekundenbruchteile, Kendalls τ für n = 5 000 etwa 0,1 Sekunden (n Schleifen über Felder der Länge n); die exakte Kurve der bedingten Wahrscheinlichkeit
kostet je Punkt Millisekunden. Die Wiederholungsexperimente stehen vorgerechnet im Raster (data/tail_grid.json)."""

from prs_constants import SEED_MAX  # noqa: F401  (Wiederverwendung der Kopie)

RHO_MIN, RHO_MAX, DEFAULT_RHO = 0.0, 0.95, 0.5
NU_MIN, NU_MAX, DEFAULT_NU = 3, 30, 3
N_MIN, N_MAX, DEFAULT_N = 50, 5000, 500
LEVELS = (0.9, 0.95, 0.99, 0.999, 0.9999, 0.99999)
DEFAULT_LEVEL = 0.99
SHARE_MIN, SHARE_MAX, DEFAULT_SHARE = 1, 20, 5          # Anteil k/n der Wochen am Rand in Prozent
DEFAULT_SEED = 35
SUM_LEVELS = (0.99, 0.999, 0.9999)
SUM_BOUND = 6.0
LIVE_SUM_PAIRS = 100_000                # Paare der Live-Rechnung für die Summenquantile (das Raster rechnet mit 4 · 1 000 000)

# Kopulas: Schlüssel -> Anzeigename
COPULAS = {"gauss": "Gauß-Kopula", "t": "t-Kopula"}

# Quadratur der bedingten Wahrscheinlichkeit (zusammengesetzte Gauß-Legendre-Regel auf der Substitution x = c/(1 − w))
QUAD_NODES = 32
QUAD_EDGES = (0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 0.98, 0.995, 1.0)

# Raster der Messreihen (tools/make_grid.py)
GRID_SEEDS = (101, 102, 103, 104)
COND_RHOS, COND_NUS = (0.3, 0.5, 0.7, 0.9), (3, 5, 10, 30)
SUM_RHOS, SUM_NUS, SUM_PAIRS = (0.0, 0.3, 0.5, 0.7, 0.9), (3, 5, 10), 1_000_000        # Paare je Seed
RANK_RHOS, RANK_NUS, RANK_PAIRS, RANK_TAU_PAIRS = (0.5, 0.7), (3, 5, 10), 200_000, 20_000
EST_RHO, EST_NU, EST_NS, EST_SHARES, EST_RUNS = 0.5, 3, (260, 520, 1040, 5000), (0.02, 0.05, 0.10), 500   # Läufe je Seed

COLOR_GAUSS = "#4c78a8"
COLOR_T = "#e45756"
COLOR_EDGE = "#222222"
COPULA_COLORS = {"gauss": COLOR_GAUSS, "t": COLOR_T}
NU_COLORS = {3: "#e45756", 5: "#f58518", 10: "#54a24b", 30: "#9d9d9d"}

PRESETS = {
    "Standardfall (Voreinstellung)": {"rho": 0.5, "nu": 3, "n": 500, "level": 0.99, "share": 5, "seed": DEFAULT_SEED},
    "Gleiche Zahl, anderer Rand": {"rho": 0.5, "nu": 3, "n": 2000, "level": 0.999, "share": 5, "seed": DEFAULT_SEED},
    "Gauß ohne Rand?": {"rho": 0.9, "nu": 3, "n": 2000, "level": 0.9999, "share": 5, "seed": DEFAULT_SEED},
    "Poolen gegen Spitzen": {"rho": 0.3, "nu": 3, "n": 1000, "level": 0.99, "share": 5, "seed": DEFAULT_SEED},
    "Wie viele Jahre Daten?": {"rho": 0.5, "nu": 3, "n": 260, "level": 0.95, "share": 5, "seed": DEFAULT_SEED},
    "Fast Gauß (ν = 30)": {"rho": 0.5, "nu": 30, "n": 500, "level": 0.999, "share": 5, "seed": DEFAULT_SEED},
}
# Die Hilfetexte nennen Zahlen des vorgerechneten Rasters (data/tail_grid.json) und exakte Werte; tests/test_claims.py prüft jede davon.
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "Zwei Standorte mit der Korrelationszahl 0,5, 500 Wochen, Niveau 99 %: Die Wahrscheinlichkeit, dass der zweite Standort ebenfalls oben steht, wenn der erste oben steht, ist exakt 0,129 bei der Gauß-Kopula und 0,330 bei der t-Kopula mit ν = 3.",
    "Gleiche Zahl, anderer Rand": "Gleiche Korrelationszahl 0,5, Niveau 99,9 %, 2 000 Wochen: exakt 0,054 (Gauß-Kopula) gegen 0,316 (t-Kopula, ν = 3), also das 5,8-Fache. In den 2 000 Wochen stehen nur wenige Wochen oben: die exakte Kurve zeigt, was die Zählung nicht kann.",
    "Gauß ohne Rand?": "Korrelationszahl 0,9: Der Randabhängigkeitskoeffizient der Gauß-Kopula ist 0, doch die bedingte Wahrscheinlichkeit fällt von 0,689 (Niveau 90 %) nur auf 0,305 (Niveau 99,999 %); die t-Kopula (ν = 3) bleibt bei 0,670 (λ = 0,670).",
    "Poolen gegen Spitzen": "Korrelationszahl 0,3: Der Pooling-Gewinn beim Quantil der Summe zum Niveau 99,9 % ist bei der Gauß-Kopula 19,4 %, bei der t-Kopula mit ν = 3 nur etwa 9 % (Raster, 4 · 1 000 000 Paare).",
    "Wie viele Jahre Daten?": "Korrelationszahl 0,5, 260 Wochen (fünf Jahre), 5 % der Wochen am Rand: Der Rangschätzer erkennt bei bekanntem ρ und ν die richtige der zwei Kopulas in 70,6 % der Fälle (Raten wäre 50 %); exakt beim Niveau 95 % sind 0,244 und 0,366.",
    "Fast Gauß (ν = 30)": "t-Kopula mit ν = 30: λ ist nur 0,0030, doch beim Niveau 99,9 % liegt die bedingte Wahrscheinlichkeit mit 0,081 noch um das 1,5-Fache über der Gauß-Kopula (0,054).",
}
