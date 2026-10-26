# Randabhängigkeit: gleiche Korrelation, anderes Risiko – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-randabhaengigkeit-demo.streamlit.app/)** -->

Neuntes Stück der Korrelations-Reihe in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, Nachfolger von [pearson-demo](https://github.com/sebastian-hanisch/pearson-demo) (Kante `fix`: eine Korrelationszahl beschreibt die Mitte, nicht den Rand), neben [rangkorrelation-demo](https://github.com/sebastian-hanisch/rangkorrelation-demo), [abhaengigkeitsmasse-demo](https://github.com/sebastian-hanisch/abhaengigkeitsmasse-demo), [scheinkorrelation-demo](https://github.com/sebastian-hanisch/scheinkorrelation-demo) und [korrelationsmatrix-demo](https://github.com/sebastian-hanisch/korrelationsmatrix-demo). Es ist die **erste Demo, die Kopulas als Gegenstand hat**: die wöchentliche Nachfrage **zweier Standorte** hat in zwei Welten **dieselben Ränder** (normalverteilt) und **dieselbe Korrelationszahl**, aber eine **Gauß-Kopula** oder eine **t-Kopula**. Die Fragen: **Wie wahrscheinlich ist es, dass beide Standorte in derselben Woche eine Spitze haben? Was folgt daraus für den gemeinsamen Sicherheitsbestand? Welche Zahl sieht den Rand, und wie viele Wochen braucht man, um ihn aus Daten zu schätzen?**

## Warum dieses Problem

Die Korrelationszahl beschreibt die Mitte der gemeinsamen Verteilung, nicht ihren Rand. Befunde: bei Korrelationsparameter ρ = 0,5 steht der zweite Standort, wenn der erste beim Niveau 99,9 % oben steht, mit Wahrscheinlichkeit **0,054** ebenfalls oben, wenn die Welt eine Gauß-Kopula hat, und mit **0,316** bei einer t-Kopula mit drei Freiheitsgraden: das **5,8-Fache** bei derselben Zahl und denselben Rändern. Die Schranke 6 für die Summe beider Standorte wird mit 0,00072 gegen 0,00027 überschritten (das 2,7-Fache), und der Pooling-Gewinn des gemeinsamen Bestands halbiert sich von **13,4 %** auf **6,3 %**. **Die Gauß-Kopula ist „asymptotisch unabhängig“ (Randabhängigkeitskoeffizient 0), doch das ist keine Entwarnung:** bei ρ = 0,9 fällt die bedingte Wahrscheinlichkeit über vier Zehnerpotenzen der Überschreitungswahrscheinlichkeit (von 10 % auf 0,001 %) nur von 0,689 auf 0,305. Aus **fünf Jahren Wochendaten** (260 Wochen) erkennt ein Rangschätzer bei bekanntem ρ = 0,5 und ν = 3, mit 5 % der Wochen am Rand und einer Schwelle zwischen den beiden exakten Werten, die richtige der zwei Kopulas in **70,6 %** der Fälle (Raten wäre 50 %), aus 96 Jahren in 99,1 %.

## Befunde und Korrekturen gegenüber dem Plan

Der Einzelplan (`korrelation-planung/STUECK-9-randabhaengigkeit.md`, nicht in diesem Repo) stützte sich auf Wegwerfskripte mit gleichem Kern, anderer Ziehungsreihenfolge und eigenem Rechenweg für die t-Verteilung. Gebaut ist alles mit SplitMix64 und eigenem Kern; die Zahlen weichen leicht ab, die Befunde bleiben, mit diesen Ausnahmen:

- **Summe bei ρ = 0,5, t(3):** Schranke 6 wird mit 0,00072 überschritten (Plan: 0,00073), Pooling-Gewinn 6,3 % (Plan: 6,2 %); bei ρ = 0: 15,0 % statt 14,8 %.
- **Rangmaße bei ρ = 0,5, t(3):** Pearson 0,487 (Plan: 0,485), Spearman 0,464 (0,463); Kendall unverändert 0,331 gegen 0,332.
- **Rangschätzer:** der Plan maß die Trennschärfe grob als d′ ≈ 1 bei 260 Wochen; gebaut ist die Aussage **„richtige Kopula erkannt“** mit einer Schwelle in der Mitte der beiden exakten Werte: 70,6 % bei 260, 78,1 % bei 520, 86,0 % bei 1 040 und 99,1 % bei 5 000 Wochen (Anteil am Rand 5 %). Das Mittel liegt höchstens 0,017 neben dem exakten Wert beim Niveau 1 − k/n (Plan: 0,02), bei kleinem k leicht zu niedrig.
- **Kein Regler für die Kopula:** beide Kopulas stehen nebeneinander und nutzen dieselben Zufallszahlen; die Freiheitsgrade ν sind **ganzzahlig** (3 bis 30), weil die t-Verteilungsfunktion für ganze Freiheitsgrade als endliche Summe vorliegt.
- **Niveau:** statt eines stetigen Reglers sechs Stufen (90 % bis 99,999 %).
- **Ohne Gegenstück im Plan:** die Hilfetexte der Presets nennen exakte Werte und Zahlen des Rasters.

## Modell

Zwei Standorte A und B, normierte wöchentliche Nachfrage (Mittel 0, Streuung 1). **Gauß-Kopula:** (A, B) = (Z₁, ρ Z₁ + √(1 − ρ²) Z₂) mit unabhängigen Standardnormalen. **t-Kopula:** dieselben Normalen werden durch √(W/ν) geteilt, W ~ χ²ᵥ (ein Wert je Woche, gemeinsam für beide Standorte), und jeder Rand wird über die t-Verteilungsfunktion auf eine Standardnormale zurückgeführt. Erzeuger `tdp_algorithm.py`, Zufall aus SplitMix64 in wörtlichen Kopien von `pearson-demo`: `prs_rng.py`, `prs_constants.py`, `prs_scenario.py`, `prs_algorithm.py`. Beide Welten haben also Standardnormal-Ränder und denselben Parameter ρ; sie unterscheiden sich nur im gemeinsamen Schwankungsfaktor.

## Methodik

- **Bedingte Überschreitungswahrscheinlichkeit** P(B oben | A oben) beim Niveau q (Schwelle c = Normal-Quantil zu q): **exakt per Quadratur** (zusammengesetzte Gauß-Legendre-Regel auf der Substitution x = c/(1 − w), `tdp_algorithm.py`, ohne scipy zur Laufzeit), für die Gauß-Kopula und für die t-Kopula (Integral über die t-Dichte mit der t-Verteilung zu ν + 1 Freiheitsgraden). Der **Randabhängigkeitskoeffizient** der t-Kopula ist 2 · t_{ν+1}(−√((ν + 1)(1 − ρ)/(1 + ρ))); der der Gauß-Kopula ist 0.
- **t-Verteilung ohne scipy:** Verteilungsfunktion für ganzzahlige Freiheitsgrade als endliche Summe in θ = arctan(x/√ν) (Abramowitz/Stegun 26.7.3 und 26.7.4), vektorisiert; Quantil durch Bisektion; Normal-Quantil nach Acklam mit einem Newton-Schritt.
- **Summe und Pooling:** Summe der Nachfrage beider Standorte; Quantil zu den Niveaus 99 %, 99,9 % und 99,99 %, Wahrscheinlichkeit, die Schranke 6 zu überschreiten, **Pooling-Gewinn** = 1 − Quantil der Summe zum Niveau 99,9 % / (2 · Quantil eines Standorts). Gauß-Kopula exakt (Summe ~ N(0, 2 + 2ρ)), t-Kopula per Simulation (Live: 100 000 Paare, Raster: 4 · 1 000 000).
- **Rangmaße:** Pearson, Spearman (Kopien aus `pearson-demo`) und Kendalls τ (eigener Kern); die Theorie sagt: τ = 2/π · arcsin ρ bei jeder elliptischen Kopula.
- **Rangschätzer des Randes:** Anteil der k größten Wochen von A, in denen auch B unter den k größten liegt, mit k = Anteil am Rand · n (mindestens 3). Er schätzt die bedingte Wahrscheinlichkeit beim Niveau 1 − k/n, nicht λ.
- **Experimente:** (1) das Niveau macht den Unterschied (exakte Kurven), (2) gleiches ρ, andere Summe, (3) welche Zahl sieht den Rand, (4) wie viele Wochen braucht man, (5) wann ist ν groß genug.
- **Vorgerechnetes Raster** (`tools/make_grid.py`, `data/tail_grid.json`, 41 Sekunden auf 10 Prozessen): Summe (15 Zellen: ρ = 0, 0,3, 0,5, 0,7, 0,9 · ν = 3, 5, 10, je 4 · 1 000 000 Paare), Rangmaße (6 Zellen: ρ = 0,5, 0,7 · ν = 3, 5, 10, je 4 · 200 000 Paare, Kendall aus den ersten 20 000 je Seed), Rangschätzer (12 Zellen: n = 260, 520, 1 040, 5 000 Wochen · 2 %, 5 %, 10 % am Rand, ρ = 0,5, ν = 3, je 4 · 500 Datensätze). Vier Seeds je Zelle (101 bis 104). **Live:** ein Datensatz mit beiden Kopulas (bis 5 000 Wochen), die exakten Kurven, die Summenquantile aus 100 000 Paaren.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py` (Raster-Werte gegen die JSON-Datei, die exakten Zahlen über dieselben Funktionen wie in der App, die Tabellen Zeile für Zeile in der Rundung der App).

**Bedingte Wahrscheinlichkeit P(B oben | A oben), exakt** (Experiment 1; Niveau 90 % / 95 % / 99 % / 99,9 % / 99,99 % / 99,999 %, rechts der Randabhängigkeitskoeffizient λ):

| ρ | Kopula | 90 % | 95 % | 99 % | 99,9 % | 99,99 % | 99,999 % | λ |
|---|---|---|---|---|---|---|---|---|
| 0,3 | Gauß | 0,216 | 0,143 | 0,056 | 0,015 | 0,004 | 0,001 | 0 |
| 0,3 | t, ν = 3 | 0,301 | 0,265 | 0,231 | 0,219 | 0,217 | 0,216 | 0,2161 |
| 0,3 | t, ν = 5 | 0,269 | 0,220 | 0,165 | 0,137 | 0,128 | 0,125 | 0,1224 |
| 0,3 | t, ν = 10 | 0,243 | 0,182 | 0,109 | 0,069 | 0,052 | 0,044 | 0,0332 |
| 0,3 | t, ν = 30 | 0,225 | 0,156 | 0,073 | 0,029 | 0,014 | 0,008 | 0,0003 |
| 0,5 | Gauß | 0,324 | 0,244 | 0,129 | 0,054 | 0,023 | 0,010 | 0 |
| 0,5 | t, ν = 3 | 0,402 | 0,366 | 0,330 | 0,316 | 0,313 | 0,313 | 0,3125 |
| 0,5 | t, ν = 5 | 0,373 | 0,321 | 0,259 | 0,226 | 0,214 | 0,210 | 0,2070 |
| 0,5 | t, ν = 10 | 0,349 | 0,284 | 0,197 | 0,140 | 0,114 | 0,101 | 0,0819 |
| 0,5 | t, ν = 30 | 0,332 | 0,257 | 0,152 | 0,081 | 0,048 | 0,032 | 0,0030 |
| 0,7 | Gauß | 0,468 | 0,392 | 0,267 | 0,160 | 0,098 | 0,061 | 0 |
| 0,7 | t, ν = 3 | 0,533 | 0,499 | 0,465 | 0,452 | 0,449 | 0,448 | 0,4481 |
| 0,7 | t, ν = 5 | 0,508 | 0,461 | 0,400 | 0,364 | 0,351 | 0,346 | 0,3432 |
| 0,7 | t, ν = 10 | 0,488 | 0,428 | 0,338 | 0,272 | 0,238 | 0,219 | 0,1911 |
| 0,7 | t, ν = 30 | 0,475 | 0,404 | 0,291 | 0,198 | 0,145 | 0,111 | 0,0260 |
| 0,9 | Gauß | 0,689 | 0,637 | 0,542 | 0,441 | 0,364 | 0,305 | 0 |
| 0,9 | t, ν = 3 | 0,728 | 0,706 | 0,682 | 0,673 | 0,671 | 0,670 | 0,6702 |
| 0,9 | t, ν = 5 | 0,713 | 0,681 | 0,638 | 0,611 | 0,601 | 0,597 | 0,5945 |
| 0,9 | t, ν = 10 | 0,701 | 0,660 | 0,595 | 0,540 | 0,510 | 0,492 | 0,4627 |
| 0,9 | t, ν = 30 | 0,693 | 0,645 | 0,561 | 0,477 | 0,419 | 0,377 | 0,2110 |

**Summe beider Standorte** (Experiment 2; Quantil der Summe zum Niveau 99 % / 99,9 % / 99,99 %, Wahrscheinlichkeit P(Summe > 6), Pooling-Gewinn beim Niveau 99,9 %; Gauß exakt, t aus 4 · 1 000 000 Paaren; die t-Pooling-Gewinne schwanken mit anderen Zufallszahlen um etwa ± 0,4 Prozentpunkte):

| ρ | Kopula | Q 99 % | Q 99,9 % | Q 99,99 % | P(Summe > 6) | Pooling-Gewinn |
|---|---|---|---|---|---|---|
| 0,0 | Gauß | 3,290 | 4,370 | 5,259 | 0,00001 | 29,3 % |
| 0,0 | t, ν = 3 | 3,610 | 5,254 | 6,642 | 0,00030 | 15,0 % |
| 0,0 | t, ν = 5 | 3,489 | 4,994 | 6,315 | 0,00017 | 19,2 % |
| 0,0 | t, ν = 10 | 3,387 | 4,711 | 5,945 | 0,00009 | 23,8 % |
| 0,3 | Gauß | 3,751 | 4,983 | 5,997 | 0,00010 | 19,4 % |
| 0,3 | t, ν = 3 | 3,993 | 5,628 | 6,962 | 0,00054 | 8,9 % |
| 0,3 | t, ν = 5 | 3,903 | 5,433 | 6,754 | 0,00038 | 12,1 % |
| 0,3 | t, ν = 10 | 3,830 | 5,232 | 6,473 | 0,00024 | 15,3 % |
| 0,5 | Gauß | 4,029 | 5,352 | 6,442 | 0,00027 | 13,4 % |
| 0,5 | t, ν = 3 | 4,206 | 5,793 | 7,093 | 0,00072 | 6,3 % |
| 0,5 | t, ν = 5 | 4,143 | 5,683 | 6,982 | 0,00058 | 8,1 % |
| 0,5 | t, ν = 10 | 4,083 | 5,524 | 6,773 | 0,00043 | 10,6 % |
| 0,7 | Gauß | 4,290 | 5,698 | 6,858 | 0,00057 | 7,8 % |
| 0,7 | t, ν = 3 | 4,395 | 5,958 | 7,247 | 0,00093 | 3,6 % |
| 0,7 | t, ν = 5 | 4,357 | 5,912 | 7,187 | 0,00086 | 4,3 % |
| 0,7 | t, ν = 10 | 4,329 | 5,823 | 7,067 | 0,00073 | 5,8 % |
| 0,9 | Gauß | 4,535 | 6,024 | 7,250 | 0,00104 | 2,5 % |
| 0,9 | t, ν = 3 | 4,561 | 6,109 | 7,342 | 0,00118 | 1,2 % |
| 0,9 | t, ν = 5 | 4,557 | 6,092 | 7,318 | 0,00116 | 1,4 % |
| 0,9 | t, ν = 10 | 4,546 | 6,070 | 7,321 | 0,00112 | 1,8 % |

**Rangmaße** (Experiment 3; Pearson / Spearman / Kendall je Kopula, 4 · 200 000 Paare, Kendall aus 4 · 20 000):

| ρ | ν | Gauß-Kopula | t-Kopula | 2/π · arcsin ρ |
|---|---|---|---|---|
| 0,5 | 3 | 0,500 / 0,482 / 0,332 | 0,487 / 0,464 / 0,331 | 0,333 |
| 0,5 | 5 | 0,500 / 0,482 / 0,334 | 0,494 / 0,472 / 0,334 | 0,333 |
| 0,5 | 10 | 0,500 / 0,483 / 0,333 | 0,499 / 0,478 / 0,333 | 0,333 |
| 0,7 | 3 | 0,700 / 0,683 / 0,495 | 0,686 / 0,662 / 0,495 | 0,494 |
| 0,7 | 5 | 0,700 / 0,682 / 0,491 | 0,695 / 0,671 / 0,492 | 0,494 |
| 0,7 | 10 | 0,700 / 0,682 / 0,491 | 0,698 / 0,676 / 0,491 | 0,494 |

**Rangschätzer des Randes aus n Wochen** (Experiment 4; ρ = 0,5, ν = 3; Anteil am Rand (k); exakter Wert beim Niveau 1 − k/n für Gauß- / t-Daten; Mittel ± Streuung des Schätzers über 2 000 Datensätze; Anteil richtiger Zuordnungen mit der Schwelle in der Mitte der beiden exakten Werte):

| Wochen n | Jahre | Anteil (k) | exakt Gauß / t | Schätzer Gauß-Daten / t-Daten | richtig erkannt |
|---|---|---|---|---|---|
| 260 | 5 | 2 % (5) | 0,167 / 0,339 | 0,155 ± 0,154 / 0,322 ± 0,188 | 68,5 % |
| 260 | 5 | 5 % (13) | 0,244 / 0,366 | 0,239 ± 0,107 / 0,359 ± 0,120 | 70,6 % |
| 260 | 5 | 10 % (26) | 0,324 / 0,402 | 0,322 ± 0,082 / 0,400 ± 0,083 | 68,1 % |
| 520 | 10 | 2 % (10) | 0,167 / 0,339 | 0,157 ± 0,110 / 0,333 ± 0,137 | 76,5 % |
| 520 | 10 | 5 % (26) | 0,244 / 0,366 | 0,241 ± 0,076 / 0,364 ± 0,083 | 78,1 % |
| 520 | 10 | 10 % (52) | 0,324 / 0,402 | 0,322 ± 0,057 / 0,398 ± 0,059 | 74,1 % |
| 1040 | 20 | 2 % (21) | 0,170 / 0,340 | 0,167 ± 0,077 / 0,333 ± 0,094 | 83,5 % |
| 1040 | 20 | 5 % (52) | 0,244 / 0,366 | 0,243 ± 0,054 / 0,365 ± 0,059 | 86,0 % |
| 1040 | 20 | 10 % (104) | 0,324 / 0,402 | 0,323 ± 0,040 / 0,403 ± 0,042 | 84,0 % |
| 5000 | 96 | 2 % (100) | 0,169 / 0,340 | 0,170 ± 0,034 / 0,339 ± 0,042 | 98,7 % |
| 5000 | 96 | 5 % (250) | 0,244 / 0,366 | 0,243 ± 0,024 / 0,365 ± 0,027 | 99,1 % |
| 5000 | 96 | 10 % (500) | 0,324 / 0,402 | 0,324 ± 0,018 / 0,402 ± 0,019 | 98,3 % |

| Frage | Befund |
|---|---|
| Wie wahrscheinlich sind gleichzeitige Spitzen? (Hauptansicht, Experiment 1) | Bei ρ = 0,5 und Niveau 99,9 % **0,054** (Gauß) gegen **0,316** (t(3)), das **5,8-Fache**; bei 99 % **0,129** gegen **0,330** (das 2,5-Fache). Die Zahlen sind exakt (Quadratur); die Tests prüfen sie gegen `scipy`, `scipy.integrate.quad`, `scipy.stats.multivariate_normal` und `multivariate_t` sowie gegen Simulationen mit 400 000 Paaren. |
| Ist „asymptotisch unabhängig“ eine Entwarnung? (Experiment 1) | **Nein:** die Gauß-Kopula hat λ = 0, aber bei ρ = 0,9 fällt die bedingte Wahrscheinlichkeit von **0,689** (Niveau 90 %) nur auf **0,305** (Niveau 99,999 %); bei ρ = 0,5 von 0,324 auf 0,010. Die t-Kopula (ν = 3) fällt bei ρ = 0,9 von 0,728 auf 0,670 (λ = 0,6702) und nähert sich ihrem λ von oben. |
| Heißt ein kleines λ ein kleines Risiko? (Experiment 5) | **Nein, nicht bei den Niveaus, die Daten erreichen:** t(30) hat bei ρ = 0,5 λ = **0,0030**, liegt aber beim Niveau 99,9 % mit **0,081** noch um das **1,5-Fache** über der Gauß-Kopula (0,054) und bei 99 % mit 0,152 gegen 0,129. λ fällt bei ρ = 0,5 von 0,3125 (ν = 3) über 0,2070 (5) und 0,0819 (10) auf 0,0030 (30). |
| Was ändert sich bei der Summe? (Experiment 2) | Bei ρ = 0,5 wird die Schranke 6 mit **0,00072** (t(3)) gegen **0,00027** (Gauß) überschritten, das **2,7-Fache**; der Pooling-Gewinn halbiert sich von **13,4 %** auf **6,3 %** (t(5): 8,1 %, t(10): 10,6 %). Bei ρ = 0 sind es 29,3 % gegen 15,0 %, bei ρ = 0,9 2,5 % gegen 1,2 %: der Unterschied ist bei kleinem ρ in Prozentpunkten am größten. |
| Hängt „das Risiko“ am gewählten Niveau? | **Ja:** bei ρ = 0,5 liegt das Quantil der Summe zum Niveau 99 % bei 4,029 (Gauß) gegen 4,206 (t(3)), zum Niveau 99,99 % bei 6,442 gegen 7,093: die Lücke wächst von 0,177 auf 0,651. |
| Welche Zahl sieht den Rand? (Experiment 3) | **Keine.** Kendalls τ ist bei beiden Kopulas dieselbe Funktion von ρ (0,332 und 0,331 gegen die Formel 0,333); Pearson und Spearman sind bei t(3) kleiner (0,487 gegen 0,500 und 0,464 gegen 0,482, bei ρ = 0,7 und ν = 3 liegt Pearson 0,014 darunter): **bei der Kopula mit dem größeren Randrisiko sind Pearson und Spearman kleiner, Kendall ist gleich.** |
| Wie viele Wochen braucht man, um den Rand zu sehen? (Experiment 4) | Der Rangschätzer trifft im Mittel die bedingte Wahrscheinlichkeit beim gewählten Niveau (höchstens 0,017 daneben), nicht λ: beim Niveau 95 % sind das **0,244** (Gauß) und **0,366** (t(3)), nicht 0 und 0,3125. Mit **260 Wochen** erkennt er die richtige Kopula in **70,6 %** der Fälle (Streuung der t-Daten 0,120), mit 520 in **78,1 %**, mit 1 040 in **86,0 %** und mit **5 000 Wochen** (96 Jahre) in **99,1 %** (Streuung 0,027); kleines k verzerrt wenig und streut stark, großes k streut wenig und misst bei einem weniger extremen Niveau. |
| Handwerte | Cauchy-Verteilung (1 Freiheitsgrad): F(1) = 3/4; 2 Freiheitsgrade: F(√2) = 1/2 + √2/4; bei ρ = 0 ist die Gauß-Kopula die Unabhängigkeit: P(B oben \| A oben) = 1 − Niveau (0,01 bei 99 %); Kendalls τ von x = (1, 2, 3, 4), y = (2, 1, 4, 3) ist 1/3, ebenso 2/π · arcsin 0,5; Rangschätzer von (1, …, 5) gegen (1, 2, 5, 3, 4) bei k = 2: 1/2. |

## Ehrliche Grenzen

- **Es gibt nur zwei Kopulas.** Echte Abhängigkeit kann schief sein (Spitzen oben häufiger als unten, etwa nach Clayton oder Gumbel); die Demo zeigt nur die beiden symmetrischen, deren Unterschied sich mit einer Zahl (ν) einstellen lässt.
- **Die Ränder sind normalverteilt.** Echte Nachfrage hat andere Ränder; die Kopula ist davon unabhängig, die Summe und der Pooling-Gewinn aber nicht.
- **ρ ist für beide Welten gleich.** Das ist die Fragestellung (gleiche Zahl, anderer Rand); in der Praxis kennt man ρ nicht und schätzt es mit Fehler (Stück 1).
- **Der Rangschätzer zeigt, was Daten hergeben, nicht das beste Verfahren.** Maximum-Likelihood-Anpassung der Kopula und Extremwerttheorie liefern mehr, brauchen aber Modellannahmen; „richtig erkannt“ gilt für diese zwei Welten bei bekanntem ρ = 0,5 und ν = 3 und schwankt mit anderen Zufallszahlen um etwa ± 1 bis 1,5 Prozentpunkte.
- **Die Wochen sind unabhängig und gleich verteilt.** Autokorrelation und Trends ändern alle Zahlen (Stück 4).
- **Die t-Zahlen der Live-Summe stammen aus 100 000 Paaren** und rauschen beim Pooling-Gewinn um etwa ± 1 Prozentpunkt (bei großem ρ und ν kann die Reihenfolge von Gauß und t kippen); das Raster nutzt 4 · 1 000 000.
- **Der Pooling-Gewinn bestimmt keinen Bestand:** gemessen wird nur das Quantil der Summe, Lieferzeiten, Kosten und Servicegrade fehlen (`standort-bestand-demo`).
- **Keine Aussage über echte Märkte:** die Demo enthält weder Finanzprodukte noch Versicherungspreise.
- **Quellen:** keine Literaturangaben; die Formeln sind durch unabhängige Rechenwege (unten) und die Messungen gestützt.

## Verifikation

- **Orakel** (`tests/test_oracle_randabhaengigkeit.py`): t-Verteilungsfunktion, -Dichte und -Quantil gegen `scipy.stats.t`, Normal-Quantil gegen `scipy.stats.norm`; die **bedingte Wahrscheinlichkeit der Gauß-Kopula gegen `scipy.stats.multivariate_normal.cdf`** (Inklusion–Exklusion) und `scipy.integrate.quad`, die der **t-Kopula gegen `quad` und `scipy.stats.multivariate_t.cdf`** und gegen eine **Brute-Force-Zählung des Quadranten** an 400 000 simulierten Paaren; λ als Grenzwert der bedingten Wahrscheinlichkeit; Kendall und Spearman gegen scipy, der Rangschätzer gegen eine Schleifenfassung, die Gauß-Summenquantile gegen die Formel und eine Simulation.
- **Kern** (`tests/test_algorithm.py`, `tests/test_edge_cases.py`): Handrechnungen (Mini-Instanzen in `conftest.py`: Cauchy, 2 Freiheitsgrade, ρ = 0, Kendall, Rangschätzer), Reglergrenzen, kleinste und größte Reihe, die Teilung der Quadratur.
- **Eigenschaften** (`tests/test_eigenschaften_randabhaengigkeit.py`): Symmetrie in A und B, Invarianz der Rangmaße gegen monotone Umformungen und Nicht-Invarianz von Pearson, Monotonie in ρ und im Niveau, Ordnung in ν, Normal-Ränder in beiden Welten.
- **Raster und Aussagen** (`tests/test_grid.py`, `tests/test_claims.py`): Probe des Erzeugers gegen die gespeicherte, jede Zahl dieser README gegen das Raster oder die Live-Funktion, die Tabellen in der Rundung der App.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler an seinen Grenzen), alle Presets, Adresszeile, jedes Auswahlfeld der Experimente, Layout, Footer, Laufzeit-Importe.
- **Mutationsprüfung:** `tools/mutate_demo.py` des Website-Repos mutiert den Kern `tdp_algorithm` (die wörtlichen Kopien `prs_*` sind ausgenommen, sie werden in `pearson-demo` geprüft; Auswertung, Verdikte und Formen prüfen die übrigen Tests): 172 Stellen, alle gesichert; 147 davon durch eigene Tests gefangen, 15 durch wortgleiche Zeilen in Nachbar-Demos und **10 als gleichwertig festgehalten** (Vergleiche an exakten Gleichständen, etwa `>` gegen `>=` bei p = 0,5 in der Normal-Quantilfunktion und bei den Grenzen der Bisektion, das Niveau genau 1 in der Prüfung, die den Fehler dann in `norm_ppf` auslöst, sowie Abbruchbedingung und Schrittzahl der Bisektion, die das Ergebnis nicht ändern). Der erste volle Lauf fand 16 Überlebende; 6 davon waren echte Lücken oder überflüssiger Code und führten zu einem Newton-Schritt statt zweier Halley-Schritte in der Normal-Quantilfunktion, einer einfacheren Schleife in der t-Verteilungsfunktion und einem Test für Gleichstände an der Schwelle (`quadrant_count`).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `data/tail_grid.json` | vorgerechnetes Raster (erzeugt von `tools/make_grid.py`) |
| `prs_*.py` | wörtliche Kopien aus `pearson-demo` (SplitMix64, Szenario, Pearson-Kern, Konstanten) |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy für die Orakel |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tdp_algorithm.py` | Kern: t-Verteilung, Erzeuger beider Kopulas, Quadratur, λ, Summe und Pooling, Rangmaße, Rangschätzer |
| `tdp_constants.py` | Regler-Grenzen, Kopulas, Raster der Messreihen, Farben, Presets |
| `tdp_evaluation.py` | Einzeldatensatz, exakte Kurven, Zellen des Rasters, Verdikte |
| `tdp_layout.py` | Fester Kopf (Stylesheet) |
| `tdp_pdf_export.py` | PDF-Export der Hauptansicht |
| `tdp_presets.py` | Permalink, Presets (Schnellstart mit Hervorhebung), Zufalls-Seed-Button |
| `tdp_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `tests/` | Orakel, Kern, Eigenschaften, Raster, Aussagen, Verdikte, Oberfläche, Layout, Footer, Laufzeit-Importe |
| `tools/make_grid.py` | rechnet das Raster vor |

## Bewusst nicht umgesetzt

- **Weitere Kopulas** (Clayton, Gumbel, Archimedische Familien) und **Maximum-Likelihood-Anpassung der Kopula an Daten:** die Aussage „der Rand hängt an der Kopula“ trägt mit zwei Welten, mehr würde sie verwässern.
- **Finanzprodukte** (Kreditportfolios, Value at Risk im Handelsbuch) und **Extremwerttheorie in voller Breite** (Tail-Index, Block-Maxima): nur als Hinweis; der Hill-Schätzer für **eine** Größe steht in `potenzgesetz-test-demo`.
- **Zeitveränderliche Kopula und Abhängigkeit über die Wochen:** Stück 4.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Das Raster entsteht neu mit `python tools/make_grid.py` (41 Sekunden auf zehn Prozessen); `tests/test_grid.py` rechnet eine feste Probe nach und schlägt an, wenn Erzeuger oder Statistik das Raster überholt haben.

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
