"""Randabhängigkeit: gleiche Korrelation, anderes Risiko - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Neuntes Stück der Korrelations-Reihe. Die Nachfrage zweier Standorte hat in zwei Welten dieselben Ränder und dieselbe Korrelationszahl, aber eine andere Kopula (Gauß oder t): exakte bedingte
Wahrscheinlichkeit gleichzeitiger Spitzen, Summe und Pooling-Gewinn, Rangmaße und ein Rangschätzer des Randes aus n Wochen.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import tdp_algorithm as K
import tdp_constants as C
import tdp_evaluation as ev
from tdp_layout import dunkles_design, kopf_css
from tdp_pdf_export import build_pdf
from tdp_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from tdp_visualization import (
    build_accuracy,
    build_cond_curves,
    build_estimate,
    build_exceed,
    build_gain,
    build_nu_curves,
    build_rank_bars,
    build_scatter,
)

st.set_page_config(page_title="Randabhängigkeit – Sebastian Hanisch", layout="wide")


def de(v: float, digits: int = 3) -> str:
    """Zahl mit Dezimalkomma."""
    return f"{v:.{digits}f}".replace(".", ",")


def thousands(v: int) -> str:
    """Ganze Zahl mit Leerzeichen als Tausendertrenner (nur die Zahl wird ersetzt, nie ein ganzer Satz)."""
    return f"{v:,}".replace(",", " ")


def level_text(q: float) -> str:
    return f"{100 * q:.3f}".rstrip("0").rstrip(".").replace(".", ",") + " %"


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return ev.analyse(settings)


@st.cache_data(show_spinner=False)
def _grid():
    return ev.load_grid()


@st.cache_data(show_spinner=False)
def _curves(rho):
    return ev.cond_curves(rho)


@st.cache_data(show_spinner=False)
def _nu_table(rho):
    return ev.nu_table(rho)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in tdp_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("📈 Randabhängigkeit: gleiche Korrelation, anderes Risiko")
st.markdown(
    """
**Neuntes Stück der Korrelations-Reihe** und **die erste Demo, die Kopulas als Gegenstand hat**. Die wöchentliche Nachfrage zweier Standorte hat in zwei Welten **dieselben Ränder** (jeder Standort für sich
normalverteilt) und **dieselbe Korrelationszahl**, aber eine andere **Kopula**, das heißt eine andere Art, wie die beiden zusammenhängen. Die Frage der Stücke 1 bis 8 war, was eine Zahl über die **Mitte**
sagt. Hier geht es um den **Rand**: Wie wahrscheinlich ist es, dass **beide** Standorte in derselben Woche eine Spitze haben, und was folgt daraus für den gemeinsamen Sicherheitsbestand?

Weiter unten: fünf Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**. Den Pooling-Gewinn als Funktion der Korrelation zeigt `standort-bestand-demo`, den
Exponenten schwerer Ränder **einer** Größe `potenzgesetz-test-demo`, Abhängigkeitsmaße für beliebige Formen `abhaengigkeitsmasse-demo` (Stück 3).
"""
)

with st.expander("So funktioniert es", expanded=True):
    st.markdown(
        r"""
| Begriff | Was er hier bedeutet |
|---|---|
| **Kopula** | Der Teil des Zusammenhangs zweier Größen, der übrig bleibt, wenn man jede für sich auf ihre Rangordnung bringt. Zwei Welten können gleiche Ränder und gleiche Korrelationszahl haben und trotzdem eine verschiedene Kopula. |
| **Gauß-Kopula** | Der Zusammenhang, den eine gemeinsame Normalverteilung mit Parameter ρ hat. Spitzen treffen selten zusammen. |
| **t-Kopula** | Dieselbe Idee mit einem gemeinsamen, zufälligen Schwankungsfaktor der Stärke ν (wenige Freiheitsgrade = starke Schwankung): Spitzen treten eher gemeinsam auf. |
| **P(B oben \| A oben)** | Die Wahrscheinlichkeit, dass der zweite Standort ebenfalls über seiner Schwelle liegt, wenn es der erste tut. Die Schwelle ist das Niveau q (z. B. 99 % der Wochen liegen darunter). Hier **exakt** gerechnet, nicht gezählt. |
| **Randabhängigkeitskoeffizient λ** | Der Grenzwert dieser Wahrscheinlichkeit, wenn die Schwelle gegen das Äußerste wandert. Bei der Gauß-Kopula ist er 0 (**asymptotische Unabhängigkeit**), bei der t-Kopula größer als 0. |
| **Rangschätzer** | Aus n Wochen: Anteil der k größten Wochen von A, in denen auch B unter den k größten ist. Er schätzt die Wahrscheinlichkeit beim Niveau 1 − k/n, nicht λ selbst. |
| **Pooling-Gewinn** | Um wie viel Prozent der gemeinsame Sicherheitsbestand (Quantil der Summe) kleiner ist als zwei getrennte Bestände. |
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Daten**")
    rho = st.slider("Korrelationsparameter ρ", *bounds("rho_slider"), step=0.05, key="rho_slider", format="%.2f",
                    help="Der Parameter beider Kopulas. Er legt auch Kendalls τ fest (τ = 2/π · arcsin ρ).")
    nu = st.slider("Freiheitsgrade ν der t-Kopula", *bounds("nu_slider"), key="nu_slider",
                   help="Wenige Freiheitsgrade = starke gemeinsame Schwankung = Spitzen treten eher gemeinsam auf. Ab ν = 30 sieht die t-Kopula in der Mitte fast wie die Gauß-Kopula aus.")
    n = st.slider("Wochen n", *bounds("n_slider"), key="n_slider", help="Zahl der simulierten Wochen (Paare).")
    level = st.select_slider("Niveau q der Schwelle", options=C.LEVELS, key="level_select", format_func=level_text,
                             help="Eine Woche zählt als Spitze, wenn der Standort über seinem q-Quantil liegt (bei 99 % in 1 von 100 Wochen).")
    st.markdown("**Verfahren**")
    share = st.slider("Anteil der Wochen am Rand für den Rangschätzer (%)", *bounds("share_slider"), key="share_slider",
                      help="k = dieser Anteil von n (mindestens 3). Kleines k: wenig verzerrt, aber streuend. Großes k: stabil, aber ein anderes Niveau.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neuen Datensatz ziehen", width="stretch", on_click=randomize_seed)

sync_query_params({"rho_slider": float(rho), "nu_slider": int(nu), "n_slider": int(n), "level_select": float(level), "share_slider": int(share), "seed_input": int(seed)})

settings = ev.Settings(float(rho), int(nu), int(n), float(level), int(share), int(seed))
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
cols = st.columns(3)
cols[0].metric("Gauß: B oben, wenn A oben", de(a.exact["gauss"]), help=f"Exakte Wahrscheinlichkeit beim Niveau {level_text(settings.level)} (Quadratur).")
cols[1].metric("t-Kopula: B oben, wenn A oben", de(a.exact["t"]), help=f"Exakte Wahrscheinlichkeit beim Niveau {level_text(settings.level)} (Quadratur).")
cols[2].metric("λ der t-Kopula", de(a.lam), help="Grenzwert der bedingten Wahrscheinlichkeit für q → 1. Bei der Gauß-Kopula ist er 0.")
cols = st.columns(3)
cols[0].metric("Pooling-Gewinn, Gauß", f"{100 * a.sums['gauss']['gain']:.1f} %".replace(".", ","), help="Quantil der Summe zum Niveau 99,9 % gegen zwei getrennte Bestände (exakt).")
cols[1].metric("Pooling-Gewinn, t-Kopula", f"{100 * a.sums['t']['gain']:.1f} %".replace(".", ","), help=f"Wie links, simuliert aus {thousands(C.LIVE_SUM_PAIRS)} Paaren.")
cols[2].metric("Kendalls τ (beide Kopulas)", de(K.tau_from_rho(settings.rho)), help="τ = 2/π · arcsin ρ gilt für jede elliptische Kopula; die Zahlen aus den Wochen dieses Laufs stehen unter dem Diagramm.")
left, right = st.columns(2)
with left:
    st.plotly_chart(build_scatter(a.xg, a.yg, a.c, "#4c78a8", "Gauß-Kopula"), width="stretch", key="main_gauss")
with right:
    st.plotly_chart(build_scatter(a.xt, a.yt, a.c, "#e45756", f"t-Kopula, ν = {settings.nu}"), width="stretch", key="main_t")
st.caption("Jeder Punkt ist eine Woche (Nachfrage der Standorte A und B, normiert). Gestrichelt: die Schwelle zum gewählten Niveau. Rot hervorgehoben: Wochen, in denen beide Standorte darüber liegen. "
           "Beide Bilder haben dieselben Ränder und dieselbe Korrelationszahl; sie nutzen dieselben Zufallszahlen.")
kind, text = ev.verdict_main(a)
getattr(st, kind)(text)
curves = _curves(settings.rho)
st.plotly_chart(build_cond_curves(curves, (settings.nu,), mark_level=settings.level), width="stretch", key="main_curve")
st.caption("Exakte bedingte Wahrscheinlichkeit gegen das Niveau (Quadratur, keine Simulation). Senkrecht gestrichelt: das gewählte Niveau. Gepunktet: λ der t-Kopula.")
kind, text = ev.verdict_measures(a)
getattr(st, kind)(text)
kind, text = ev.verdict_sums(a)
getattr(st, kind)(text)
kind, text = ev.verdict_tail(a)
getattr(st, kind)(text)
if st.button("📄 PDF erstellen", key="pdf_make"):
    st.session_state["pdf_bytes"] = (settings, build_pdf(a))
pdf = st.session_state.get("pdf_bytes")
if pdf and pdf[0] == settings:
    st.download_button("⬇️ PDF herunterladen", pdf[1], file_name="randabhaengigkeit-demo.pdf", mime="application/pdf", key="pdf_download")

st.markdown("## ❓ Kernfrage: Reicht die Korrelationszahl, wenn es auf gleichzeitige Spitzen ankommt?")
st.markdown("Nein. Die Zahl ρ und die Ränder lassen offen, **wie oft beide zugleich** extrem sind; diese Wahrscheinlichkeit und mit ihr der Pooling-Gewinn hängen an der Kopula (Ergebnis oben und Experimente unten).")

st.markdown("---")

grid = _grid()
if grid is None:
    st.warning("Das vorgerechnete Raster (data/tail_grid.json) fehlt; es entsteht mit `python tools/make_grid.py`. Die Experimente 2 bis 4 zeigen es.")

# --- E1 Das Niveau macht den Unterschied ---------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Das Niveau macht den Unterschied")
st.caption("Exakte bedingte Wahrscheinlichkeit P(B oben | A oben) gegen das Niveau der Schwelle, für die Gauß-Kopula und t-Kopulas mit ν = 3, 5, 10 und 30. Gepunktet: λ je t-Kopula.")
e1_rho = st.selectbox("Korrelationsparameter ρ", list(C.COND_RHOS), index=1, key="e1_rho", format_func=lambda v: de(v, 1))
e1_curves = _curves(e1_rho)
st.plotly_chart(build_cond_curves(e1_curves, C.COND_NUS), width="stretch", key="cond_chart")
kind, text = ev.verdict_curves(e1_curves, C.COND_NUS[0])
getattr(st, kind)(text)
st.caption("Die Gauß-Kurve sinkt mit dem Niveau, ohne sich schnell der 0 zu nähern: der Koeffizient λ = 0 sagt nichts über die Wahrscheinlichkeiten bei den Niveaus, die Daten überhaupt erreichen. "
           "Die t-Kurven nähern sich ihrem λ von oben bei 90 % liegen die t-Kurven mit kleinem ν höher als die Gauß-Kurve (bis zum 1,4-Fachen), die mit ν = 30 kaum.")

st.markdown("---")

# --- E2 Gleiches ρ, andere Summe ---------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Gleiches ρ, andere Summe")
st.caption(f"Summe der Nachfrage beider Standorte (Einheitsvarianz je Standort). Pooling-Gewinn = 1 − Quantil der Summe zum Niveau {level_text(C.SUM_LEVELS[1])} / (2 · Quantil eines Standorts). "
           f"Vorgerechnet: {thousands(C.SUM_PAIRS)} Paare je Seed, 4 Seeds; die Gauß-Zahlen sind exakt.")
if grid is not None:
    e2c1, e2c2 = st.columns(2)
    e2_nu = e2c1.selectbox("Freiheitsgrade ν der t-Kopula", list(C.SUM_NUS), index=0, key="e2_nu")
    e2_rho = e2c2.selectbox("ρ für das Verdikt", list(C.SUM_RHOS), index=2, key="e2_rho", format_func=lambda v: de(v, 1))
    s1, s2 = st.columns(2)
    with s1:
        st.plotly_chart(build_gain(grid["sum"], e2_nu), width="stretch", key="gain_chart")
    with s2:
        st.plotly_chart(build_exceed(grid["sum"], e2_nu), width="stretch", key="exceed_chart")
    kind, text = ev.verdict_sum_grid(grid["sum"], e2_rho, e2_nu)
    getattr(st, kind)(text)
    st.caption("Der Pooling-Gewinn ist bei der t-Kopula überall kleiner: Spitzen kommen gemeinsam, das Poolen puffert weniger. Der Unterschied ist in Prozentpunkten bei kleinem ρ am größten, "
               "bei großem ρ ist schon die Gauß-Kopula kaum besser als getrennte Bestände.")

st.markdown("---")

# --- E3 Welche Zahl sieht den Rand? ------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Welche Zahl sieht den Rand?")
st.caption(f"Pearson, Spearman und Kendall beider Kopulas bei gleichem ρ. Vorgerechnet: {thousands(C.RANK_PAIRS)} Paare je Seed, 4 Seeds (Kendall aus den ersten {thousands(C.RANK_TAU_PAIRS)} Paaren je Seed).")
if grid is not None:
    e3c1, e3c2 = st.columns(2)
    e3_rho = e3c1.selectbox("ρ", list(C.RANK_RHOS), index=0, key="e3_rho", format_func=lambda v: de(v, 1))
    e3_nu = e3c2.selectbox("Freiheitsgrade ν", list(C.RANK_NUS), index=0, key="e3_nu")
    cell = ev.find(grid["rank"], rho=e3_rho, nu=e3_nu)
    st.plotly_chart(build_rank_bars(cell), width="stretch", key="rank_chart")
    kind, text = ev.verdict_rank_grid(grid["rank"], e3_rho)
    getattr(st, kind)(text)
    st.caption("Keine der drei Zahlen zeigt den Rand. Pearson und Spearman liegen bei der t-Kopula mit kleinem ν leicht niedriger, also in der falschen Richtung für die Frage, wer das größere Randrisiko hat; "
               "Kendalls τ ist bei beiden gleich, weil es nur von ρ abhängt.")

st.markdown("---")

# --- E4 Wie viele Wochen braucht man? ---------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Wie viele Wochen braucht man, um den Rand zu sehen?")
st.caption(f"Rangschätzer des Randes aus n Wochen, für Daten der Gauß-Kopula und der t-Kopula (ρ = {de(C.EST_RHO, 1)}, ν = {C.EST_NU}). Rauten: exakter Wert beim Niveau 1 − k/n. "
           f"Vorgerechnet: {thousands(4 * C.EST_RUNS)} Datensätze je Zelle. „Richtig erkannt“: die Schwelle liegt in der Mitte der beiden exakten Werte, Raten wäre 50 %.")
if grid is not None:
    e4c1, e4c2 = st.columns(2)
    e4_share = e4c1.selectbox("Anteil am Rand", list(C.EST_SHARES), index=1, key="e4_share", format_func=lambda v: f"{100 * v:g} %")
    e4_n = e4c2.selectbox("Wochen n für das Verdikt", list(C.EST_NS), index=0, key="e4_n", format_func=lambda v: f"{v} ({v / 52:.0f} Jahre)")
    r1, r2 = st.columns(2)
    with r1:
        st.plotly_chart(build_estimate(grid["estimate"], e4_share), width="stretch", key="estimate_chart")
    with r2:
        st.plotly_chart(build_accuracy(grid["estimate"]), width="stretch", key="accuracy_chart")
    kind, text = ev.verdict_estimate(grid["estimate"], e4_n, e4_share)
    getattr(st, kind)(text)
    st.caption("Der Schätzer trifft im Mittel den exakten Wert beim gewählten Niveau, nicht λ. Fünf Jahre Wochendaten reichen nicht, um die beiden Welten sicher zu unterscheiden; "
               "kleines k verzerrt wenig, streut aber stark, großes k ist stabil, misst aber die Wahrscheinlichkeit bei einem anderen, weniger extremen Niveau.")

st.markdown("---")

# --- E5 Wann ist ν groß genug? ---------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 5: Wann ist ν groß genug?")
st.caption("Exakte bedingte Wahrscheinlichkeit der t-Kopula bei drei Niveaus für wachsende Freiheitsgrade, gestrichelt die Gauß-Kopula je Niveau, gepunktet λ.")
e5_rho = st.selectbox("Korrelationsparameter ρ", list(C.COND_RHOS), index=1, key="e5_rho", format_func=lambda v: de(v, 1))
table = _nu_table(e5_rho)
st.plotly_chart(build_nu_curves(table), width="stretch", key="nu_chart")
rows_md = ["| ν | λ | " + " | ".join(f"Niveau {level_text(q)}" for q in table["levels"]) + " |", "|---|---|" + "---|" * len(table["levels"])]
rows_md.append("| Gauß | 0 | " + " | ".join(de(v) for v in table["gauss"]) + " |")
rows_md += [f"| {r['nu']} | {de(r['lambda'], 4)} | " + " | ".join(de(v) for v in r["cond"]) + " |" for r in table["rows"]]
st.markdown("\n".join(rows_md))
kind, text = ev.verdict_nu(table)
getattr(st, kind)(text)
st.caption("λ geht mit wachsendem ν gegen 0, die bedingten Wahrscheinlichkeiten bei den erreichbaren Niveaus aber viel langsamer: eine t-Kopula mit ν = 30 hat ein kaum sichtbares λ und liegt bei 99,9 % und ρ = 0,5 doch um das 1,5-Fache über der Gauß-Kopula, bei ρ = 0,9 nur noch um etwa 8 %.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    r"""
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Es gibt nur zwei Kopulas** | echte Abhängigkeit kann schief sein (Spitzen oben häufiger als unten, etwa die Kopulas nach Clayton oder Gumbel); die Demo zeigt nur die beiden symmetrischen | – |
| **Die Ränder sind normalverteilt** | echte Nachfrage hat andere Ränder; die Kopula ist unabhängig davon, die Summe aber nicht (Experiment 2) | Ränder getrennt anpassen |
| **ρ ist für beide Welten gleich** | das ist die Fragestellung (gleiche Zahl, anderer Rand); in der Praxis kennt man ρ nicht und schätzt es mit Fehler (Stück 1) | Stück 1 |
| **Der Rangschätzer ist das Schätzverfahren** | er zeigt, was Daten hergeben, nicht das beste Verfahren; Maximum-Likelihood-Anpassung der Kopula und Extremwerttheorie mit Tail-Index liefern mehr, brauchen aber Modellannahmen | – |
| **Die Wochen sind unabhängig und gleich verteilt** | Autokorrelation und Trends ändern alle Zahlen (Stück 4) | Stück 4 |
| **Die Simulation der Summenquantile ist genau** | die t-Zahlen der Live-Ansicht stammen aus 100 000 Paaren und rauschen beim Pooling-Gewinn um etwa ± 1 Prozentpunkt (bei großem ρ und ν kann die Reihenfolge von Gauß und t kippen); das Raster nutzt 4 · 1 000 000 | – |
| **Der Pooling-Gewinn bestimmt den Bestand** | gemessen wird nur das Quantil der Summe; Lieferzeiten, Kosten und Servicegrade fehlen | `standort-bestand-demo` |
| **Aussagen über echte Märkte** | keine: die Demo enthält weder Finanzprodukte noch Versicherungspreise | – |
"""
)
st.caption("Alle Daten sind simuliert; Gauß- und t-Kopula sind Modellwahlen, kein Befund über echte Standorte.")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** $X, Y$ sind die normierten Nachfragen zweier Standorte mit Standardnormal-Rändern; $\Phi$, $\varphi$ sind Verteilungsfunktion und Dichte der Standardnormalverteilung, $t_\nu$ und $f_\nu$ Verteilungsfunktion und Dichte der
Student-Verteilung mit $\nu$ Freiheitsgraden; $q$ ist das Niveau, $c = \Phi^{-1}(q)$ die Schwelle.

| Symbol | Bedeutung |
|---|---|
| $\rho$ | Parameter beider Kopulas |
| $\nu$ | Freiheitsgrade der t-Kopula |
| $\lambda$ | Randabhängigkeitskoeffizient |
| $k$ | Zahl der Wochen am Rand beim Rangschätzer |

**Erzeugung.** Gauß-Kopula: $(X, Y) = (Z_1,\ \rho Z_1 + \sqrt{1-\rho^2}\,Z_2)$ mit unabhängigen Standardnormalen $Z_i$. t-Kopula: mit $W \sim \chi^2_\nu$ unabhängig von den $Z_i$ wird $(T_1, T_2) = (Z_1, \rho Z_1 + \sqrt{1-\rho^2}\,Z_2)\,\sqrt{\nu/W}$ gebildet und jeder Rand auf eine Standardnormale zurückgeführt:

$$ X = \Phi^{-1}\big(t_\nu(T_1)\big), \qquad Y = \Phi^{-1}\big(t_\nu(T_2)\big) $$

**Bedingte Wahrscheinlichkeit, Gauß-Kopula.** Mit $s = \sqrt{1-\rho^2}$ und $\bar\Phi = 1 - \Phi$:

$$ P(Y > c \mid X > c) = \frac{1}{1-q}\int_c^\infty \varphi(x)\,\bar\Phi\!\left(\frac{c - \rho x}{s}\right)\mathrm{d}x $$

**Bedingte Wahrscheinlichkeit, t-Kopula.** Mit der t-Dichte $f_\nu$, der Schwelle $c = t_\nu^{-1}(q)$ (auf der Skala der Kopula, dieselbe Quantilstufe $q$) und $\bar t_{\nu+1} = 1 - t_{\nu+1}$:

$$ P(Y > c \mid X > c) = \frac{1}{1-q}\int_c^\infty f_\nu(x)\;\bar t_{\nu+1}\!\left(\frac{c - \rho x}{\sqrt{(1-\rho^2)/(\nu+1)}\,\sqrt{\nu + x^2}}\right)\mathrm{d}x $$

**Randabhängigkeitskoeffizient.** Der Grenzwert für $q \to 1$; bei der Gauß-Kopula ist er 0, bei der t-Kopula gilt

$$ \lambda_t = 2\, t_{\nu+1}\!\left(-\sqrt{\frac{(\nu+1)(1-\rho)}{1+\rho}}\right) $$

**Rangmaß.** Kendalls $\tau$ ist bei jeder elliptischen Kopula dieselbe Funktion von $\rho$:

$$ \tau = \frac{2}{\pi}\arcsin\rho $$

**Summe und Pooling.** Bei der Gauß-Kopula ist $X + Y \sim N(0,\,2+2\rho)$, das Quantil zur Stufe $p$ also $z_p\sqrt{2+2\rho}$; bei der t-Kopula wird es simuliert. Der Pooling-Gewinn gegen zwei getrennte Bestände ist

$$ G_p = 1 - \frac{Q_p(X+Y)}{2\,z_p}, \qquad z_p = \Phi^{-1}(p) $$

**Rangschätzer.** Mit den Rängen $R_i$ der $X$- und $S_i$ der $Y$-Werte ($1,\dots,n$):

$$ \hat\lambda(k) = \frac{1}{k}\sum_{i=1}^{n}\mathbf 1\{R_i > n-k,\ S_i > n-k\} $$

Er schätzt $P(Y \text{ oben} \mid X \text{ oben})$ beim Niveau $1 - k/n$, nicht $\lambda$ selbst.

**Quellen.** In dieser Fassung sind keine Literaturangaben ausgewiesen; die Formeln sind durch unabhängige Rechenwege (Tests gegen scipy) und die Messungen der Experimente gestützt.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
