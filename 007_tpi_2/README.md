# Text Mining y Análisis Discursivo Comparado
### TPI 2 · Cenital vs. Anfibia — Cobertura de Inteligencia Artificial

---

## ¿De qué trata?

Trabajo práctico integrador de Text Mining que compara cómo dos medios digitales argentinos — **Cenital** y **Anfibia** — cubren el tema de la inteligencia artificial. El análisis combina herramientas de procesamiento de lenguaje natural con lectura crítica de fragmentos para identificar diferencias discursivas entre los dos grupos.

---

## Estructura del repositorio

```
├── TPI_2_Text_Mining_y_Analisis_Discursivo_Comparado.ipynb     # Notebook principal con todo el análisis
├── TPI_2_Consigna_y_Rubrica.md                                 # Consigna y rubrica del trabajo
├── ScraperCorpus_TPI2.py                                       # Script de scraping para armar el corpus
├── corpus_tpi2.csv                                             # Corpus: 10 artículos, 5 por grupo
└── README.md
```

---

## Corpus

| Grupo | Medio | Artículos | Palabras |
|---|---|---|---|
| Anfibia | Revista Anfibia | 5 | 14.077 |
| Cenital | Cenital | 5 | 10.408 |

Todos los artículos son de acceso libre y fueron publicados entre 2023 y 2026. El scraping se realizó con `trafilatura`.

---

## Herramientas utilizadas

- **spaCy** — tokenización, lematización y reconocimiento de entidades (NER)
- **scikit-learn** — Bag of Words y TF-IDF
- **pandas / matplotlib / seaborn** — análisis y visualización
- **trafilatura** — extracción de texto desde URLs

---

## Principales hallazgos

- Los dos grupos presentan vocabularios casi sin solapamiento, algo que el heatmap de TF-IDF hace visible de forma inmediata.
- **Cenital** usa la IA como lente para explicar fenómenos de actualidad (política, desinformación, economía). Su gesto discursivo es pedagógico: define términos, educa al lector.
- **Anfibia** trata la IA como objeto crítico, con marco geopolítico (Sur, LatamGPT) y referencias académicas explícitas.
- En este corpus, BoW y TF-IDF convergen porque la separación léxica es tan marcada que no hay términos compartidos que penalizar.

---

## Nota sobre el uso de IA

Este trabajo registra explícitamente el uso de herramientas de IA generativa en la tabla de la sección inicial del notebook, incluyendo qué se conservó y qué se descartó de cada respuesta.
