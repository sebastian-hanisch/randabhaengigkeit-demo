"""Das vorgerechnete Raster (data/tail_grid.json): Aufbau, Plausibilität und Übereinstimmung mit dem Erzeuger (Probe)."""

import pytest

import tdp_algorithm as K
import tdp_constants as C
import tdp_evaluation as ev


@pytest.fixture(scope="module")
def grid():
    g = ev.load_grid()
    assert g is not None, "data/tail_grid.json fehlt: python tools/make_grid.py"
    return g


def _assert_close(a, b, path="probe"):
    """Verschachtelte Wörter und Listen: Zahlen mit relativer Toleranz (Summationsreihenfolge je Plattform, DEMO-PLAYBOOK §4), alles andere exakt."""
    if isinstance(a, dict):
        assert set(a) == set(b), path
        for k in a:
            _assert_close(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_close(x, y, f"{path}[{i}]")
    elif isinstance(a, float):
        assert a == pytest.approx(b, rel=1e-8, abs=1e-12), path
    else:
        assert a == b, path


def test_probe_still_matches_the_generator(grid):
    """Ändert sich Erzeuger, Statistik oder Seed-Ableitung, passt die gespeicherte Probe nicht mehr: dann tools/make_grid.py neu laufen lassen."""
    probe = ev.grid_probe()
    _assert_close(probe, grid["probe"])
    assert probe["quadrant"] == grid["probe"]["quadrant"] and probe["k"] == grid["probe"]["k"]


def test_grid_has_every_cell_exactly_once(grid):
    def keys(rows, *names):
        return sorted(tuple(r[n] for n in names) for r in rows)

    assert keys(grid["sum"], "rho", "nu") == sorted((r, n) for r in C.SUM_RHOS for n in C.SUM_NUS)
    assert keys(grid["rank"], "rho", "nu") == sorted((r, n) for r in C.RANK_RHOS for n in C.RANK_NUS)
    assert keys(grid["estimate"], "n", "share") == sorted((n, s) for n in C.EST_NS for s in C.EST_SHARES)


def test_grid_meta_and_run_counts(grid):
    assert grid["meta"]["seeds"] == list(C.GRID_SEEDS)
    assert all(r["pairs"] == len(C.GRID_SEEDS) * C.SUM_PAIRS for r in grid["sum"]) and all(r["pairs"] == len(C.GRID_SEEDS) * C.RANK_PAIRS for r in grid["rank"])
    assert all(r["runs"] == len(C.GRID_SEEDS) * C.EST_RUNS for r in grid["estimate"])


def test_the_gauss_numbers_in_the_sum_cells_are_exact(grid):
    for r in grid["sum"]:
        assert r["gauss"] == pytest.approx(ev.gauss_sums(r["rho"]), rel=1e-12)


def test_the_t_copula_always_has_the_smaller_pooling_gain_and_the_larger_exceedance(grid):
    for r in grid["sum"]:
        assert r["t"]["gain"] < r["gauss"]["gain"], (r["rho"], r["nu"])
        assert r["t"]["exceed"] > r["gauss"]["exceed"], (r["rho"], r["nu"])
        assert r["t"]["quantiles"][2] > r["gauss"]["quantiles"][2]


def test_fewer_degrees_of_freedom_mean_a_smaller_gain(grid):
    for rho in C.SUM_RHOS:
        gains = [ev.find(grid["sum"], rho=rho, nu=nu)["t"]["gain"] for nu in sorted(C.SUM_NUS)]
        assert gains == sorted(gains), rho                      # ν = 3 hat den kleinsten Gewinn


def test_the_simulated_t_exceedance_agrees_with_an_independent_sum_formula_for_rho_zero(grid):
    """Bei ρ = 0 ist die Gauß-Kopula die Unabhängigkeit: P(Summe > 6) = P(N(0, 2) > 6) = 1,1·10⁻⁵ (Plan: 0,00001)."""
    cell = ev.find(grid["sum"], rho=0.0, nu=3)
    assert cell["gauss"]["exceed"] == pytest.approx(K.gauss_sum_exceed(0.0, 6.0), rel=1e-12) and 0.9e-5 < cell["gauss"]["exceed"] < 1.3e-5


def test_the_rank_cells_follow_the_theory(grid):
    for r in grid["rank"]:
        assert r["gauss"]["pearson"] == pytest.approx(r["rho"], abs=0.005)
        assert r["gauss"]["kendall"] == pytest.approx(r["tau_formula"], abs=0.01) and r["t"]["kendall"] == pytest.approx(r["tau_formula"], abs=0.01)
        assert r["t"]["pearson"] < r["gauss"]["pearson"] and r["t"]["spearman"] < r["gauss"]["spearman"]


def test_the_estimator_hits_the_exact_conditional_probability(grid):
    for r in grid["estimate"]:
        for copula in C.COPULAS:
            assert r["mean"][copula] == pytest.approx(r["exact"][copula], abs=0.03), (r["n"], r["share"], copula)
        assert r["exact"]["gauss"] == pytest.approx(K.cond_exceed_gauss(r["level"], C.EST_RHO), rel=1e-12)
        assert r["mid"] == pytest.approx(0.5 * (r["exact"]["gauss"] + r["exact"]["t"]), rel=1e-12)


def test_the_accuracy_rises_with_the_length_of_the_series(grid):
    for share in C.EST_SHARES:
        acc = [ev.find(grid["estimate"], n=n, share=share)["accuracy"] for n in C.EST_NS]
        assert acc == sorted(acc) and acc[0] > 0.5 and acc[-1] > 0.95, share


def test_the_spread_falls_with_the_length_of_the_series(grid):
    for share in C.EST_SHARES:
        sd = [ev.find(grid["estimate"], n=n, share=share)["sd"]["t"] for n in C.EST_NS]
        assert sd == sorted(sd, reverse=True), share
