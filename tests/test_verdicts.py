"""Verdikte (reine Funktionen in tdp_evaluation.py) an handgesetzten Werten: Art und jede Zahl im Text."""

import numpy as np

import tdp_constants as C
import tdp_evaluation as ev


def _analysis(**over):
    s = ev.Settings(0.5, 3, 500, 0.99, 5, 1)
    kw = dict(settings=s, xg=np.zeros(1), yg=np.zeros(1), xt=np.zeros(1), yt=np.zeros(1), c=2.33, quadrant={"gauss": (5, 1), "t": (6, 3)}, exact={"gauss": 0.129, "t": 0.33}, lam=0.3125,
              measures={"gauss": {"pearson": 0.5, "spearman": 0.48, "kendall": 0.33}, "t": {"pearson": 0.49, "spearman": 0.47, "kendall": 0.33}}, k=25, tail_level=0.95,
              tail_est={"gauss": 0.24, "t": 0.36}, tail_exact={"gauss": 0.244, "t": 0.366},
              sums={"gauss": {"quantiles": [4.0, 5.35, 6.44], "exceed": 0.00027, "gain": 0.134}, "t": {"quantiles": [4.2, 5.8, 7.1], "exceed": 0.00072, "gain": 0.063}})
    kw.update(over)
    return ev.Analysis(**kw)


def test_verdict_main_names_every_number_and_warns_when_the_t_copula_is_clearly_higher():
    kind, text = ev.verdict_main(_analysis())
    assert kind == "warning"
    for needle in ("99 %", "2,33", "0,129", "0,330", "ν = 3", "das 2,6-Fache", "500 Wochen", "5 bzw. 6", "1 bzw. 3"):
        assert needle in text, needle
    assert "nur grob" in text                                                    # unter 10 Wochen oben: Hinweis auf die Zählung


def test_verdict_main_is_info_when_the_copulas_barely_differ_and_has_no_hint_with_many_exceedances():
    kind, text = ev.verdict_main(_analysis(exact={"gauss": 0.30, "t": 0.31}, quadrant={"gauss": (50, 15), "t": (48, 15)}))
    assert kind == "info" and "nur grob" not in text


def test_verdict_main_threshold_for_the_warning_is_one_and_a_half():
    assert ev.verdict_main(_analysis(exact={"gauss": 0.2, "t": 0.3}))[0] == "info"       # genau das 1,5-Fache: keine Warnung
    assert ev.verdict_main(_analysis(exact={"gauss": 0.2, "t": 0.3001}))[0] == "warning"


def test_verdict_measures_names_the_six_numbers_and_tau():
    kind, text = ev.verdict_measures(_analysis())
    assert kind == "info"
    for needle in ("0,500", "0,480", "0,330", "0,490", "0,470", "0,333", "keine der drei Zahlen zeigt den Rand"):
        assert needle in text, needle


def test_verdict_sums_names_exceedance_quantile_and_gain():
    kind, text = ev.verdict_sums(_analysis())
    assert kind == "warning"
    for needle in ("Schranke 6", "0,00027", "0,00072", "das 2,7-Fache", "99,9 %", "5,35", "5,80", "13,4 %", "6,3 %", "100 000"):
        assert needle in text, needle


def test_verdict_tail_explains_that_the_estimator_is_not_lambda():
    kind, text = ev.verdict_tail(_analysis())
    assert kind == "info"
    for needle in ("k = 25", "95 %", "0,24", "0,36", "0,24 bzw. 0,37", "0,312", "nicht λ"):
        assert needle in text, needle


def _curves():
    levels = ev.curve_levels()
    return {"levels": levels, "gauss": [0.32 - 0.02 * i for i in range(len(levels))], "t": {3: [0.40 - 0.004 * i for i in range(len(levels))]}, "lambda": {3: 0.3125}}


def test_verdict_curves_reads_the_level_nearest_to_99_9_percent():
    kind, text = ev.verdict_curves(_curves(), 3)
    assert kind == "warning"
    for needle in ("Bei Niveau 99,9 %", "Gauß **0,160**", "t(ν = 3) **0,368**", "das 2,3-Fache", "von 0,320 (Niveau 90 %)", "auf 0,000 (Niveau 99,999 %)", "von 0,400 auf 0,336", "λ = **0,312**"):
        assert needle in text, needle


def _sum_rows():
    cell = {"rho": 0.5, "nu": 3, "gauss": {"quantiles": [4.0, 5.35, 6.44], "exceed": 0.00027, "gain": 0.134}, "t": {"quantiles": [4.2, 5.8, 7.1], "exceed": 0.00072, "gain": 0.063}}
    return [cell]


def test_verdict_sum_grid_reads_one_cell():
    kind, text = ev.verdict_sum_grid(_sum_rows(), 0.5, 3)
    assert kind == "warning"
    for needle in ("ρ = 0,5", "ν = 3", "0,00072", "0,00027", "das 2,7-Fache", "6,3 %", "13,4 %", "7,10", "6,44", "99,99 %"):
        assert needle in text, needle


def test_verdict_rank_grid_uses_the_smallest_nu_and_says_which_copula_has_the_smaller_number():
    rows = [{"rho": 0.5, "nu": 10, "tau_formula": 1 / 3, "gauss": {"pearson": 0.5, "spearman": 0.48, "kendall": 0.333}, "t": {"pearson": 0.499, "spearman": 0.478, "kendall": 0.333}},
            {"rho": 0.5, "nu": 3, "tau_formula": 1 / 3, "gauss": {"pearson": 0.5, "spearman": 0.482, "kendall": 0.332}, "t": {"pearson": 0.487, "spearman": 0.464, "kendall": 0.331}}]
    kind, text = ev.verdict_rank_grid(rows, 0.5)
    assert kind == "info" and "ν = 3" in text and "0,487 gegen 0,500" in text and "0,464 gegen 0,482" in text and "0,331 gegen 0,332" in text and "Formel 0,333" in text and "kleinere Zahl" in text


def _est_rows(acc):
    return [{"n": 260, "share": 0.05, "k": 13, "exact": {"gauss": 0.244, "t": 0.366}, "mean": {"gauss": 0.24, "t": 0.36}, "sd": {"gauss": 0.107, "t": 0.12}, "accuracy": acc}]


def test_verdict_estimate_warns_below_eighty_percent_and_reports_all_numbers():
    kind, text = ev.verdict_estimate(_est_rows(0.706), 260, 0.05)
    assert kind == "warning"
    for needle in ("n = 260 Wochen (5 Jahre)", "k = 13", "**0,24** ± 0,11", "**0,36** ± 0,12", "0,24 und 0,37", "70,6 %", "50 % wäre Raten"):
        assert needle in text, needle
    assert ev.verdict_estimate(_est_rows(0.8), 260, 0.05)[0] == "success" and ev.verdict_estimate(_est_rows(0.7999), 260, 0.05)[0] == "warning"


def test_verdict_nu_reads_the_last_row_and_the_middle_level():
    table = {"levels": [0.99, 0.999, 0.9999], "gauss": [0.129, 0.054, 0.023], "rows": [{"nu": 3, "lambda": 0.3125, "cond": [0.33, 0.316, 0.313]}, {"nu": 30, "lambda": 0.003, "cond": [0.152, 0.081, 0.048]}]}
    kind, text = ev.verdict_nu(table)
    assert kind == "warning" and "ν = 30" in text and "0,0030" in text and "99,9 %" in text and "0,081" in text and "0,054" in text and "das 1,5-Fache" in text


def test_helper_formats():
    assert ev._pct(0.1234) == "12,3 %" and ev._num(0.5, 2) == "0,50" and ev._level(0.99999) == "99,999 %" and ev._level(0.9) == "90 %" and ev._int(100000) == "100 000"
    assert ev._ratio(1.0, 0.0) == "unbestimmt" and ev._ratio(0.3, 0.1) == "das 3,0-Fache"


def test_every_preset_text_fits_the_constants():
    assert all(len(v) > 40 for v in C.PRESET_HELP.values())
