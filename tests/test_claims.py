"""Jede Zahl der README steht hier und wird aus dem vorgerechneten Raster (data/tail_grid.json) oder über dieselben Funktionen wie in der App nachgerechnet (DEMO-PLAYBOOK §4).

Zufall nur aus SplitMix64 mit festen Seeds. Die Tabellen der README werden Zeile für Zeile in der Rundung der App (`_c`, `_p`) gegen die README geprüft, sonst gilt eine Toleranz von einem
halben Tausendstel bzw. Hundertstel. Die Zeilen-Bauer (`cond_rows` usw.) dienen auch dazu, die README-Tabellen zu erzeugen."""

from pathlib import Path

import pytest

import tdp_algorithm as K
import tdp_constants as C
import tdp_evaluation as ev

README_PATH = Path(__file__).resolve().parent.parent / "README.md"
README = README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else ""


def _c(v, d):
    return f"{v:.{d}f}".replace(".", ",")


def _p(v):
    return f"{100 * v:.1f}".replace(".", ",")


def _lv(q):
    return f"{100 * q:.3f}".rstrip("0").rstrip(".").replace(".", ",") + " %"


@pytest.fixture(scope="module")
def grid():
    g = ev.load_grid()
    assert g is not None
    return g


# --- Zeilen-Bauer (Rundung der App) ------------------------------------------------------------------------------------------------------------

def cond_rows() -> list[str]:
    rows = []
    for rho in C.COND_RHOS:
        rows.append(f"| {_c(rho, 1)} | Gauß | " + " | ".join(_c(K.cond_exceed_gauss(q, rho), 3) for q in C.LEVELS) + " | 0 |")
        for nu in C.COND_NUS:
            rows.append(f"| {_c(rho, 1)} | t, ν = {nu} | " + " | ".join(_c(K.cond_exceed_t(q, rho, nu), 3) for q in C.LEVELS) + f" | {_c(K.lambda_t(rho, nu), 4)} |")
    return rows


def sum_rows(grid) -> list[str]:
    rows = []
    for rho in C.SUM_RHOS:
        g = ev.find(grid["sum"], rho=rho, nu=C.SUM_NUS[0])["gauss"]
        rows.append(f"| {_c(rho, 1)} | Gauß | " + " | ".join(_c(v, 3) for v in g["quantiles"]) + f" | {_c(g['exceed'], 5)} | {_p(g['gain'])} % |")
        for nu in C.SUM_NUS:
            t = ev.find(grid["sum"], rho=rho, nu=nu)["t"]
            rows.append(f"| {_c(rho, 1)} | t, ν = {nu} | " + " | ".join(_c(v, 3) for v in t["quantiles"]) + f" | {_c(t['exceed'], 5)} | {_p(t['gain'])} % |")
    return rows


def rank_rows(grid) -> list[str]:
    rows = []
    for rho in C.RANK_RHOS:
        for nu in C.RANK_NUS:
            r = ev.find(grid["rank"], rho=rho, nu=nu)
            rows.append(f"| {_c(rho, 1)} | {nu} | " + " / ".join(_c(r["gauss"][k], 3) for k in ("pearson", "spearman", "kendall")) + " | " + " / ".join(_c(r["t"][k], 3) for k in ("pearson", "spearman", "kendall"))
                        + f" | {_c(r['tau_formula'], 3)} |")
    return rows


def estimate_rows(grid) -> list[str]:
    rows = []
    for n in C.EST_NS:
        for share in C.EST_SHARES:
            r = ev.find(grid["estimate"], n=n, share=share)
            rows.append(f"| {n} | {n / 52:.0f} | {_c(100 * share, 0)} % ({r['k']}) | {_c(r['exact']['gauss'], 3)} / {_c(r['exact']['t'], 3)} | {_c(r['mean']['gauss'], 3)} ± {_c(r['sd']['gauss'], 3)} / "
                        f"{_c(r['mean']['t'], 3)} ± {_c(r['sd']['t'], 3)} | {_p(r['accuracy'])} % |")
    return rows


# --- Tabellen der README --------------------------------------------------------------------------------------------------------------------

def test_claim_the_exact_table_is_the_app_function_in_the_apps_rounding():
    """README: Tabelle „Bedingte Wahrscheinlichkeit P(B oben | A oben), exakt“ (20 Zeilen)."""
    for row in cond_rows():
        assert row in README, row


def test_claim_the_sum_table_is_the_grid_in_the_apps_rounding(grid):
    """README: Tabelle „Summe beider Standorte“ (20 Zeilen)."""
    for row in sum_rows(grid):
        assert row in README, row


def test_claim_the_rank_table_is_the_grid_in_the_apps_rounding(grid):
    """README: Tabelle „Rangmaße“ (6 Zeilen)."""
    for row in rank_rows(grid):
        assert row in README, row


def test_claim_the_estimate_table_is_the_grid_in_the_apps_rounding(grid):
    """README: Tabelle „Rangschätzer des Randes“ (12 Zeilen)."""
    for row in estimate_rows(grid):
        assert row in README, row


# --- Aussagen -----------------------------------------------------------------------------------------------------------------------------

def test_claim_the_headline_numbers():
    """README: bei ρ = 0,5 und Niveau 99,9 % Gauß 0,054 gegen t(3) 0,316, das 5,8-Fache; bei 99 %: 0,129 gegen 0,330 (das 2,5-Fache)."""
    g, t = K.cond_exceed_gauss(0.999, 0.5), K.cond_exceed_t(0.999, 0.5, 3)
    assert (g, t) == pytest.approx((0.054, 0.316), abs=0.0006) and round(t / g, 1) == 5.8
    assert K.cond_exceed_gauss(0.99, 0.5) == pytest.approx(0.129, abs=0.0006) and K.cond_exceed_t(0.99, 0.5, 3) == pytest.approx(0.330, abs=0.0006)
    assert round(K.cond_exceed_t(0.99, 0.5, 3) / K.cond_exceed_gauss(0.99, 0.5), 1) == 2.5


def test_claim_asymptotic_independence_is_slow():
    """README: Gauß, ρ = 0,9: 0,689 (Niveau 90 %) auf 0,305 (99,999 %); λ der Gauß-Kopula 0; t(3): λ = 0,670; ρ = 0,5: Gauß 0,324 → 0,010."""
    assert (K.cond_exceed_gauss(0.9, 0.9), K.cond_exceed_gauss(0.99999, 0.9)) == pytest.approx((0.689, 0.305), abs=0.0006)
    assert K.lambda_t(0.9, 3) == pytest.approx(0.670, abs=0.0006)
    assert (K.cond_exceed_gauss(0.9, 0.5), K.cond_exceed_gauss(0.99999, 0.5)) == pytest.approx((0.324, 0.010), abs=0.0006)


def test_claim_small_lambda_is_not_small_risk():
    """README: t(30), ρ = 0,5: λ = 0,0030, bei Niveau 99,9 % 0,081 gegen Gauß 0,054 (das 1,5-Fache); bei 99 %: 0,152 gegen 0,129."""
    assert K.lambda_t(0.5, 30) == pytest.approx(0.0030, abs=0.00005)
    a, b = K.cond_exceed_t(0.999, 0.5, 30), K.cond_exceed_gauss(0.999, 0.5)
    assert (a, b) == pytest.approx((0.081, 0.054), abs=0.0006) and round(a / b, 1) == 1.5
    assert K.cond_exceed_t(0.99, 0.5, 30) == pytest.approx(0.152, abs=0.0006)


def test_claim_lambda_values():
    """README: λ bei ρ = 0,5: ν = 3: 0,3125, ν = 5: 0,2070, ν = 10: 0,0819, ν = 30: 0,0030; bei ρ = 0,9, ν = 3: 0,6702."""
    assert [K.lambda_t(0.5, nu) for nu in (3, 5, 10, 30)] == pytest.approx([0.3125, 0.2070, 0.0819, 0.0030], abs=0.00006)
    assert K.lambda_t(0.9, 3) == pytest.approx(0.6702, abs=0.00006)


def test_claim_the_sum_statements(grid):
    """README: ρ = 0,5, ν = 3: P(Summe > 6) 0,00072 gegen 0,00027 (das 2,7-Fache), Pooling-Gewinn 6,3 % gegen 13,4 %; ρ = 0: 15,0 % gegen 29,3 %; ρ = 0,9: 1,2 % gegen 2,5 %; ν = 10, ρ = 0,5: 10,6 %."""
    c = ev.find(grid["sum"], rho=0.5, nu=3)
    assert (c["t"]["exceed"], c["gauss"]["exceed"]) == pytest.approx((0.00072, 0.00027), abs=0.000006) and round(c["t"]["exceed"] / c["gauss"]["exceed"], 1) == 2.7
    assert (c["t"]["gain"], c["gauss"]["gain"]) == pytest.approx((0.063, 0.134), abs=0.0006)
    zero, nine = ev.find(grid["sum"], rho=0.0, nu=3), ev.find(grid["sum"], rho=0.9, nu=3)
    assert (zero["t"]["gain"], zero["gauss"]["gain"], nine["t"]["gain"], nine["gauss"]["gain"]) == pytest.approx((0.150, 0.293, 0.012, 0.025), abs=0.0006)
    assert ev.find(grid["sum"], rho=0.5, nu=10)["t"]["gain"] == pytest.approx(0.106, abs=0.0006)


def test_claim_the_sum_percentiles_depend_on_the_level(grid):
    """README: bei ρ = 0,5: Quantil der Summe zum Niveau 99 % 4,029 gegen 4,206, zum Niveau 99,99 % 6,442 gegen 7,093."""
    c = ev.find(grid["sum"], rho=0.5, nu=3)
    assert (c["gauss"]["quantiles"][0], c["t"]["quantiles"][0], c["gauss"]["quantiles"][2], c["t"]["quantiles"][2]) == pytest.approx((4.029, 4.206, 6.442, 7.093), abs=0.0006)
    assert (c["t"]["quantiles"][0] - c["gauss"]["quantiles"][0], c["t"]["quantiles"][2] - c["gauss"]["quantiles"][2]) == pytest.approx((0.177, 0.651), abs=0.0011)


def test_claim_rank_statements(grid):
    """README: Kendall bei beiden Kopulas 0,332 und 0,331 gegen die Formel 0,333 (ρ = 0,5); Pearson 0,500 gegen 0,487 und Spearman 0,482 gegen 0,464 bei ν = 3."""
    r = ev.find(grid["rank"], rho=0.5, nu=3)
    assert (r["gauss"]["kendall"], r["t"]["kendall"], r["tau_formula"]) == pytest.approx((0.332, 0.331, 0.333), abs=0.0006)
    assert (r["gauss"]["pearson"], r["t"]["pearson"], r["gauss"]["spearman"], r["t"]["spearman"]) == pytest.approx((0.500, 0.487, 0.482, 0.464), abs=0.0006)
    assert max(abs(x["t"]["pearson"] - x["gauss"]["pearson"]) for x in grid["rank"]) == pytest.approx(0.014, abs=0.0006)


def test_claim_the_estimator_statements(grid):
    """README: richtig erkannt (Anteil am Rand 5 %): 70,6 % bei 260 Wochen, 78,1 % bei 520, 86,0 % bei 1 040, 99,1 % bei 5 000; Mittel bis 0,017 neben dem exakten Wert (n = 260, 2 %, t-Kopula);
    Streuung der t-Daten bei 260 Wochen und 5 %: 0,120, bei 5 000: 0,027."""
    acc = [ev.find(grid["estimate"], n=n, share=0.05)["accuracy"] for n in C.EST_NS]
    assert acc == pytest.approx([0.706, 0.781, 0.860, 0.991], abs=0.0006)
    dev = max(abs(r["mean"][c] - r["exact"][c]) for r in grid["estimate"] for c in C.COPULAS)
    assert dev == pytest.approx(0.017, abs=0.0006)
    assert (ev.find(grid["estimate"], n=260, share=0.05)["sd"]["t"], ev.find(grid["estimate"], n=5000, share=0.05)["sd"]["t"]) == pytest.approx((0.120, 0.027), abs=0.0006)


def test_claim_the_estimator_exact_values_at_the_five_percent_level():
    """README: beim Niveau 95 % sind die exakten Werte 0,244 (Gauß) und 0,366 (t(3))."""
    assert (K.cond_exceed_gauss(0.95, 0.5), K.cond_exceed_t(0.95, 0.5, 3)) == pytest.approx((0.244, 0.366), abs=0.0006)


def test_claim_hand_values():
    """README: Cauchy: F(1) = 3/4; ν = 2: F(√2) = 1/2 + √2/4; Kendalls τ von x = (1, 2, 3, 4), y = (2, 1, 4, 3) ist 1/3; bei ρ = 0,5 ist 2/π · arcsin ρ = 1/3; Rangschätzer von (1, …, 5) gegen (1, 2, 5, 3, 4) bei k = 2: 1/2."""
    assert float(K.t_cdf(1.0, 1)) == pytest.approx(0.75, abs=1e-14) and float(K.t_cdf(2 ** 0.5, 2)) == pytest.approx(0.5 + 2 ** 0.5 / 4, abs=1e-14)
    assert K.kendall_tau([1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 4.0, 3.0]) == pytest.approx(1 / 3, abs=1e-15) and K.tau_from_rho(0.5) == pytest.approx(1 / 3, abs=1e-15)
    assert K.tail_rate([1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 5.0, 3.0, 4.0], 2) == pytest.approx(0.5)
    assert K.cond_exceed_gauss(0.99, 0.0) == pytest.approx(0.01, rel=1e-9)


def test_claim_the_grid_time(grid):
    """README: das Raster dauert 41 Sekunden auf 10 Prozessen (Wert aus der Datei; ändert sich beim Neuerzeugen)."""
    assert grid["meta"]["seconds"] == 41


# --- Texte der Presets ---------------------------------------------------------------------------------------------------------------------

def test_claim_the_preset_help_texts_quote_exact_and_grid_numbers(grid):
    h = C.PRESET_HELP
    g99, t99 = K.cond_exceed_gauss(0.99, 0.5), K.cond_exceed_t(0.99, 0.5, 3)
    assert f"exakt {_c(g99, 3)} bei der Gauß-Kopula und {_c(t99, 3)} bei der t-Kopula" in h["Standardfall (Voreinstellung)"]
    g999, t999 = K.cond_exceed_gauss(0.999, 0.5), K.cond_exceed_t(0.999, 0.5, 3)
    assert f"exakt {_c(g999, 3)} (Gauß-Kopula) gegen {_c(t999, 3)} (t-Kopula, ν = 3), also das {_c(t999 / g999, 1)}-Fache" in h["Gleiche Zahl, anderer Rand"]
    text = h["Gauß ohne Rand?"]
    assert f"von {_c(K.cond_exceed_gauss(0.9, 0.9), 3)} (Niveau 90 %) nur auf {_c(K.cond_exceed_gauss(0.99999, 0.9), 3)} (Niveau 99,999 %)" in text
    assert f"bleibt bei {_c(K.cond_exceed_t(0.99999, 0.9, 3), 3)} (λ = {_c(K.lambda_t(0.9, 3), 3)})" in text
    cell = ev.find(grid["sum"], rho=0.3, nu=3)
    assert f"bei der Gauß-Kopula {_p(cell['gauss']['gain'])} %, bei der t-Kopula mit ν = 3 nur etwa {round(100 * cell['t']['gain'])} %" in h["Poolen gegen Spitzen"]
    est = ev.find(grid["estimate"], n=260, share=0.05)
    assert f"in {_p(est['accuracy'])} % der Fälle" in h["Wie viele Jahre Daten?"]
    assert f"exakt beim Niveau 95 % sind {_c(K.cond_exceed_gauss(0.95, 0.5), 3)} und {_c(K.cond_exceed_t(0.95, 0.5, 3), 3)}" in h["Wie viele Jahre Daten?"]
    text = h["Fast Gauß (ν = 30)"]
    assert f"λ ist nur {_c(K.lambda_t(0.5, 30), 4)}" in text and f"mit {_c(K.cond_exceed_t(0.999, 0.5, 30), 3)} noch um das {_c(K.cond_exceed_t(0.999, 0.5, 30) / g999, 1)}-Fache" in text and f"Gauß-Kopula ({_c(g999, 3)})" in text
