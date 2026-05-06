"""
Para correr:
    pip install streamlit pandas scikit-learn plotly spacy
    streamlit run streamlit_app.py

Requiere es_core_news_md ya instalado en el entorno:
    python -m spacy download es_core_news_md
"""

import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# ---------------------------------------------------------------------------
# Configuración de la página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TP3 PLN — Infobae vs. La Nación",
    page_icon="📰",
    layout="wide",
)

RUTA_CORPUS       = Path("corpus_tp3.csv")
COLUMNA_TEXTO     = "texto"
COLUMNA_GRUPO     = "grupo_comparacion"
MODELO_SPACY      = "es_core_news_md"
TOP_N             = 12
VENTANA_FRAGMENTO = 200

COLORES = {
    "infobae":  "#e63946",
    "lanacion": "#1d3557",
}

# Mismas listas que en el notebook ----------------------------------------
STOPWORDS_EXTRA = {
    "él", "el", "ella", "ellos", "ellas", "lo", "la", "los", "las", "le", "les", "se",
    "aunque", "afirmó", "afirmo", "agregó","artículo", "dijo",
    "acá", "ahí", "algo", "algunas", "aseguró", "aseguro", "así", "aun", "bajar",
    "destaco", "entonces", "ese", "esos", "este", "esto", "fue", "fueron", "hubo",
    "explicó", "explico", "quizás", "probablemente", "porque",
    "indicó", "indico", "informó", "manifestó", "medio", "nota",
    "podría", "precisó", "remarcó", "remarco", "señaló", "senalo"," sostuvo",
    "tal", "también", "tanto", "tener", "todavía", "todo", "todos", "unos", "vale",
    "volver", "semana",
    "infobae",
}

STOPWORDS_EXCLUIR = ["no", "más", "sin", "pero", "bajó", "mayor", "entre"]

CORRECCIONES_LEMAS = {
    "indec":  "INDEC",
    "ipc":    "IPC",
    "caputo": "Caputo",
    "milei":  "Milei",
    "rem":    "REM",
    "fmi":    "FMI",
    "bcra":   "BCRA",
    # Correcciones de voseo
    "sos":    "ser",
    "tenés":  "tener",
    "sabés":  "saber",
    "querés": "querer",
    "venís":  "venir",
    "podés":  "poder",
    "avisá":  "avisar",
    "revisá": "revisar",
    "analizá": "analizar",

}

PATRONES_MATCHER = {
    "INDICE_PRECIOS_CONSUMIDOR": [[{"LOWER": "índice"}, {"LOWER": "de"}, {"LOWER": "precios"}, {"LOWER": "al"}, {"LOWER": "consumidor"}]],
    "BANCO_CENTRAL": [[{"LOWER": "banco"}, {"LOWER": "central"}]],
    "FONDO_MONETARIO_INTERNACIONAL": [[{"LOWER": "fondo"}, {"LOWER": "monetario"}, {"LOWER": "internacional"}]],
    "GRAN_BUENOS_AIRES": [[{"LOWER": "gran"}, {"LOWER": "buenos"}, {"LOWER": "aires"}]],
    "RELEVAMIENTO_EXPECTATIVAS_MERCADO": [[{"LOWER": "relevamiento"}, {"LOWER": "de"}, {"LOWER": "expectativas"}, {"LOWER": {"IN": ["de", "del"]}}, {"LOWER": "mercado"}]],
}

PATRONES_ENTIDADES = [

    # ORG
    {"label": "ORG", "pattern": "INDEC"},
    {"label": "ORG", "pattern": "Instituto Nacional de Estadística y Censos"},
    {"label": "ORG", "pattern": "Banco Central"},
    {"label": "ORG", "pattern": "BCRA"},
    {"label": "ORG", "pattern": "Ministerio de Economía"},
    {"label": "ORG", "pattern": "FMI"},
    {"label": "ORG", "pattern": "Fondo Monetario Internacional"},
    {"label": "ORG", "pattern": "Fundación Libertad y Progreso"},
    {"label": "ORG", "pattern": "Eco Go"},
    {"label": "ORG", "pattern": "LCG"},
    {"label": "ORG", "pattern": "Econviews"},
    {"label": "ORG", "pattern": "Equilibra"},
    {"label": "ORG", "pattern": "Analytica"},
    {"label": "ORG", "pattern": "Invecq"},
    {"label": "ORG", "pattern": "ATE Indec"},
    {"label": "ORG", "pattern": "Gobierno"},
    {"label": "ORG", "pattern": "El Gobierno"},
    {"label": "ORG", "pattern": "gobierno"},
    {"label": "ORG", "pattern": "Libertad y Progreso"},
    {"label": "ORG", "pattern": "Dirección General de Estadística y Censos"},

    # PER
    {"label": "PER", "pattern": "Caputo"},
    {"label": "PER", "pattern": "Javier Milei"},
    {"label": "PER", "pattern": "Milei"},
    {"label": "PER", "pattern": "Marco"},
    {"label": "PER", "pattern": "Lorenzo Sigaut Gravina"},
    {"label": "PER", "pattern": "Miguel Kiguel"},
    {"label": "PER", "pattern": "Iván Cachanosky"},
    {"label": "PER", "pattern": "Camilo Tiscornia"},
    {"label": "PER", "pattern": "Claudio Caprarulo"},
    {"label": "PER", "pattern": "Ricardo Delgado"},
    {"label": "PER", "pattern": "Adorni"},

    # LOC
    {"label": "LOC", "pattern": "Argentina"},
    {"label": "LOC", "pattern": "República Argentina"},
    {"label": "LOC", "pattern": "CABA"},
    {"label": "LOC", "pattern": "Ciudad Autónoma de Buenos Aires"},
    {"label": "LOC", "pattern": "Gran Buenos Aires"},
    {"label": "LOC", "pattern": "Medio Oriente"},
    {"label": "LOC", "pattern": "AMBA"},

    # MISC: indices/relevamientos/instrumentos, no personas ni instituciones.
    {"label": "MISC", "pattern": "IPC"},
    {"label": "MISC", "pattern": "Índice de Precios al Consumidor"},
    {"label": "MISC", "pattern": "REM"},
    {"label": "MISC", "pattern": "Relevamiento de Expectativas de Mercado"},
    {"label": "MISC", "pattern": "Relevamiento de Expectativas del Mercado"},
    {"label": "MISC", "pattern": "ENGHo"},
    {"label": "MISC", "pattern": "Encuesta Nacional de Gastos de los Hogares"},

    # Ajustes adicionales. Correcciones de categorías económicas mal clasificadas por spaCy
    {"label": "MISC", "pattern": "Carnes"},
    {"label": "MISC", "pattern": "carnes"},
    {"label": "MISC", "pattern": "Prendas"},
    {"label": "MISC", "pattern": "prendas"},
    {"label": "MISC", "pattern": "Alimentos"},
    {"label": "MISC", "pattern": "alimentos"},
    {"label": "MISC", "pattern": "Restaurantes"},
    {"label": "MISC", "pattern": "restaurantes"},
    {"label": "MISC", "pattern": "Regulados"},
    {"label": "MISC", "pattern": "regulados"},
    {"label": "MISC", "pattern": "los Estacionales"},
    {"label": "MISC", "pattern": "Estacionales"},
    {"label": "MISC", "pattern": "estacionales"},
    {"label": "MISC", "pattern": "Prendas"},
    {"label": "MISC", "pattern": "prendas"},
    {"label": "MISC", "pattern": "Proyectan"},
    {"label": "MISC", "pattern": "proyectan"},
    {"label": "MISC", "pattern": "Economía"},
    {"label": "MISC", "pattern": "economía"},

    # DATE  
    {"label": "DATE", "pattern": "enero"},
    {"label": "DATE", "pattern": "Enero"},
    {"label": "DATE", "pattern": "febrero"},
    {"label": "DATE", "pattern": "Febrero"},
    {"label": "DATE", "pattern": "marzo"},
    {"label": "DATE", "pattern": "Marzo"},
    {"label": "DATE", "pattern": "abril"},  
    {"label": "DATE", "pattern": "Abril"},
]

# ---------------------------------------------------------------------------
# spaCy — cargado una sola vez con cache_resource
# ---------------------------------------------------------------------------

@st.cache_resource
def cargar_nlp():
    import spacy

    nlp = spacy.load(MODELO_SPACY)
    nlp.vocab["él"].is_stop   = True
    nlp.vocab["ella"].is_stop = True

    for palabra in STOPWORDS_EXTRA:
        nlp.vocab[palabra].is_stop = True
    for palabra in STOPWORDS_EXCLUIR:
        nlp.vocab[palabra].is_stop = False

    ruler = nlp.add_pipe("entity_ruler", config={"overwrite_ents": True}, before="ner")
    ruler.add_patterns(PATRONES_ENTIDADES)

    return nlp


# Funciones de lematización — rehusadas del notebook
def normalizar_lemma(token):
    lema = token.lemma_.strip().lower()
    if not lema or lema == "-pron-":
        lema = token.text.lower()
    return lema

def lema_ajustado(token):
    forma = token.text.lower()
    if forma in CORRECCIONES_LEMAS:
        return CORRECCIONES_LEMAS[forma]
    return normalizar_lemma(token)


# ---------------------------------------------------------------------------
# Preprocesamiento con spaCy
# ---------------------------------------------------------------------------

@st.cache_data
def cargar_corpus(ruta: Path) -> pd.DataFrame:
    df = pd.read_csv(ruta)
    for col in ["id", "medio", "autor", "titulo", "url", COLUMNA_TEXTO, COLUMNA_GRUPO]:
        df[col] = df[col].fillna("").astype(str).str.strip()
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    patron = r"\b[^\W\d_]{2,}\b"
    df["num_palabras"] = df[COLUMNA_TEXTO].str.findall(patron).str.len()
    return df


@st.cache_data
def preprocesar_con_spacy(textos: list, _nlp) -> list:
    """Mismo pipeline que arma texto_modelo en el notebook."""
    resultado = []
    for doc in _nlp.pipe(textos):
        tokens = []
        for token in doc:
            if not token.is_alpha:
                continue
            forma = token.text.lower()
            lema  = lema_ajustado(token)
            if _nlp.vocab[forma].is_stop:
                continue
            if _nlp.vocab[lema].is_stop:
                continue
            tokens.append(lema)
        resultado.append(" ".join(tokens))
    return resultado


@st.cache_data
def calcular_bow_tfidf(textos_limpios: list, grupos: list, top_n: int = 12):
    vect_bow   = CountVectorizer(min_df=1, max_df=0.95)
    vect_tfidf = TfidfVectorizer(min_df=1, max_df=0.95)

    vect_bow.fit(textos_limpios)
    vect_tfidf.fit(textos_limpios)

    mat_bow   = vect_bow.transform(textos_limpios)
    mat_tfidf = vect_tfidf.transform(textos_limpios)

    df_bow   = pd.DataFrame(mat_bow.toarray(),   columns=vect_bow.get_feature_names_out())
    df_tfidf = pd.DataFrame(mat_tfidf.toarray(), columns=vect_tfidf.get_feature_names_out())
    df_bow["grupo"]   = grupos
    df_tfidf["grupo"] = grupos

    filas_bow, filas_tfidf = [], []
    for grupo in set(grupos):
        bow_g   = df_bow[df_bow["grupo"]     == grupo].drop(columns="grupo").mean()
        tfidf_g = df_tfidf[df_tfidf["grupo"] == grupo].drop(columns="grupo").mean()
        for t, v in bow_g.nlargest(top_n).items():
            filas_bow.append({"grupo": grupo, "termino": t, "frecuencia_media": round(v, 4)})
        for t, v in tfidf_g.nlargest(top_n).items():
            filas_tfidf.append({"grupo": grupo, "termino": t, "tfidf_medio": round(v, 4)})

    tfidf_grupos = df_tfidf.groupby("grupo").mean()
    return pd.DataFrame(filas_bow), pd.DataFrame(filas_tfidf), tfidf_grupos


@st.cache_data
def calcular_bigramas(textos_limpios: list, grupos: list, top_n: int = 12):
    vect = CountVectorizer(ngram_range=(2, 2), min_df=1)
    vect.fit(textos_limpios)
    mat   = vect.transform(textos_limpios)
    vocab = vect.get_feature_names_out()

    df_bg = pd.DataFrame(mat.toarray(), columns=vocab)
    df_bg["grupo"] = grupos

    filas = []
    for grupo in set(grupos):
        serie = df_bg[df_bg["grupo"] == grupo].drop(columns="grupo").sum()
        for bigrama, freq in serie.nlargest(top_n).items():
            filas.append({"grupo": grupo, "bigrama": bigrama, "frecuencia": int(freq)})
    return pd.DataFrame(filas)


@st.cache_data
def calcular_entidades(textos: list, grupos: list, _nlp):
    filas = []
    for texto, grupo in zip(textos, grupos):
        doc = _nlp(texto)
        for ent in doc.ents:
            filas.append({"grupo": grupo, "texto": ent.text, "label": ent.label_})
    return pd.DataFrame(filas)


# Buscador de fragmentos — réplica de extraer_fragmentos del notebook

def extraer_fragmentos_spacy(nlp, df, grupo, termino, ventana=200, max_fragmentos=6):
    resultados = []
    subdf = df if grupo == "todos" else df[df[COLUMNA_GRUPO] == grupo]

    for fila in subdf.itertuples(index=False):
        texto_original = getattr(fila, COLUMNA_TEXTO)
        doc = nlp(texto_original)

        for token in doc:
            if lema_ajustado(token) == termino or token.text.lower() == termino:
                inicio    = max(0, token.idx - ventana)
                fin       = min(len(texto_original), token.idx + len(token.text) + ventana)
                fragmento = texto_original[inicio:fin].strip()
                if inicio > 0:
                    fragmento = "…" + fragmento
                if fin < len(texto_original):
                    fragmento = fragmento + "…"
                resultados.append({
                    "grupo":    getattr(fila, COLUMNA_GRUPO),
                    "titulo":   fila.titulo,
                    "fragmento": fragmento,
                })
                break  # un fragmento por texto

        if len(resultados) >= max_fragmentos:
            break

    return resultados


# ---------------------------------------------------------------------------
# Inicialización principal
# ---------------------------------------------------------------------------
if not RUTA_CORPUS.exists():
    st.error(
        f"No se encontró `{RUTA_CORPUS}`. "
        "Copiá el corpus en la misma carpeta que este script."
    )
    st.stop()

with st.spinner("Cargando modelo spaCy..."):
    nlp = cargar_nlp()

df            = cargar_corpus(RUTA_CORPUS)
grupos        = df[COLUMNA_GRUPO].tolist()
grupos_unicos = sorted(df[COLUMNA_GRUPO].unique().tolist())

with st.spinner("Preprocesando corpus con spaCy..."):
    textos_limpios = preprocesar_con_spacy(df[COLUMNA_TEXTO].tolist(), nlp)

top_bow, top_tfidf, tfidf_grupos = calcular_bow_tfidf(textos_limpios, grupos, TOP_N)
top_bigramas                     = calcular_bigramas(textos_limpios, grupos, TOP_N)
df_entidades                     = calcular_entidades(df[COLUMNA_TEXTO].tolist(), grupos, nlp)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("📰 TP3 PLN")
st.sidebar.markdown("**Infobae vs. La Nación**  \nInflación 2026")
st.sidebar.divider()
seccion = st.sidebar.radio(
    "Sección",
    [
        "📋 Auditoría del corpus",
        "📊 BoW vs TF-IDF",
        "🔗 Bigramas",
        "🌡️ Heatmap",
        "🏷️ Entidades (NER)",
        "🔍 Buscador de fragmentos",
    ],
)

# ---------------------------------------------------------------------------
# SECCIÓN 1 — Auditoría del corpus
# ---------------------------------------------------------------------------
if seccion == "📋 Auditoría del corpus":
    st.title("Auditoría del corpus")

    col1, col2, col3 = st.columns(3)
    col1.metric("Textos totales", len(df))
    col2.metric("Palabras totales (aprox.)", int(df["num_palabras"].sum()))
    col3.metric("Promedio por texto", f"{df['num_palabras'].mean():.0f}")

    st.divider()

    resumen = (
        df.groupby(COLUMNA_GRUPO)
        .agg(
            documentos=("id", "count"),
            palabras_totales=("num_palabras", "sum"),
            promedio=("num_palabras", "mean"),
        )
        .reset_index()
        .round(1)
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Textos por grupo")
        fig = px.bar(
            resumen, x=COLUMNA_GRUPO, y="documentos",
            color=COLUMNA_GRUPO, color_discrete_map=COLORES,
            text="documentos",
            labels={COLUMNA_GRUPO: "Grupo", "documentos": "Cantidad"},
        )
        fig.update_layout(showlegend=False, yaxis_range=[0, 8])
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Palabras por grupo")
        fig = px.bar(
            resumen, x=COLUMNA_GRUPO, y="palabras_totales",
            color=COLUMNA_GRUPO, color_discrete_map=COLORES,
            text="palabras_totales",
            labels={COLUMNA_GRUPO: "Grupo", "palabras_totales": "Palabras"},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Longitud por artículo")
    fig = px.bar(
        df.sort_values(COLUMNA_GRUPO), x="titulo", y="num_palabras",
        color=COLUMNA_GRUPO, color_discrete_map=COLORES,
        labels={"titulo": "Artículo", "num_palabras": "Palabras", COLUMNA_GRUPO: "Grupo"},
    )
    fig.update_layout(xaxis_tickangle=-40)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Tabla completa")
    st.dataframe(
        df[["id", "fecha", "medio", "autor", "titulo", COLUMNA_GRUPO, "num_palabras"]],
        use_container_width=True,
    )

# ---------------------------------------------------------------------------
# SECCIÓN 2 — BoW vs TF-IDF
# ---------------------------------------------------------------------------
elif seccion == "📊 BoW vs TF-IDF":
    st.title("Bag of Words vs. TF-IDF")

    grupo_sel = st.selectbox("Seleccioná un grupo", grupos_unicos)

    bow_g   = top_bow[top_bow["grupo"]     == grupo_sel].sort_values("frecuencia_media", ascending=True)
    tfidf_g = top_tfidf[top_tfidf["grupo"] == grupo_sel].sort_values("tfidf_medio",      ascending=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"BoW — {grupo_sel}")
        fig = px.bar(
            bow_g, x="frecuencia_media", y="termino", orientation="h",
            color_discrete_sequence=[COLORES.get(grupo_sel, "#888")],
            labels={"frecuencia_media": "Frecuencia media", "termino": ""},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader(f"TF-IDF — {grupo_sel}")
        fig = px.bar(
            tfidf_g, x="tfidf_medio", y="termino", orientation="h",
            color_discrete_sequence=[COLORES.get(grupo_sel, "#888")],
            labels={"tfidf_medio": "TF-IDF medio", "termino": ""},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Comparación directa entre grupos (TF-IDF)")
    fig = px.bar(
        top_tfidf.sort_values("tfidf_medio", ascending=True),
        x="tfidf_medio", y="termino", color="grupo",
        barmode="group", orientation="h", color_discrete_map=COLORES,
        labels={"tfidf_medio": "TF-IDF medio", "termino": ""},
        height=600,
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# SECCIÓN 3 — Bigramas
# ---------------------------------------------------------------------------
elif seccion == "🔗 Bigramas":
    st.title("Bigramas por grupo")

    top_n_bg = st.slider("Top N bigramas", 5, TOP_N, 10)

    for grupo in grupos_unicos:
        bg_g = (
            top_bigramas[top_bigramas["grupo"] == grupo]
            .nlargest(top_n_bg, "frecuencia")
            .sort_values("frecuencia", ascending=True)
        )
        st.subheader(grupo)
        fig = px.bar(
            bg_g, x="frecuencia", y="bigrama", orientation="h",
            color_discrete_sequence=[COLORES.get(grupo, "#888")],
            labels={"frecuencia": "Frecuencia", "bigrama": ""},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Comparación entre grupos")
    fig = px.bar(
        top_bigramas.sort_values("frecuencia", ascending=True),
        x="frecuencia", y="bigrama", color="grupo",
        barmode="group", orientation="h", color_discrete_map=COLORES,
        labels={"frecuencia": "Frecuencia", "bigrama": ""},
        height=700,
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# SECCIÓN 4 — Heatmap
# ---------------------------------------------------------------------------
elif seccion == "🌡️ Heatmap":
    st.title("Heatmap de términos distintivos (TF-IDF)")

    top_n_heat = st.slider("Términos por grupo", 5, 15, 8)

    terminos_heat = []
    for grupo in grupos_unicos:
        top_g = top_tfidf[top_tfidf["grupo"] == grupo].nlargest(top_n_heat, "tfidf_medio")
        for t in top_g["termino"].tolist():
            if t not in terminos_heat:
                terminos_heat.append(t)

    cols_disponibles = [t for t in terminos_heat if t in tfidf_grupos.columns]
    matriz_heat = tfidf_grupos[cols_disponibles]

    fig = go.Figure(data=go.Heatmap(
        z=matriz_heat.values,
        x=cols_disponibles,
        y=matriz_heat.index.tolist(),
        colorscale="Blues",
        text=[[f"{v:.3f}" for v in row] for row in matriz_heat.values],
        texttemplate="%{text}",
    ))
    fig.update_layout(
        title="Peso TF-IDF por término y grupo",
        xaxis_tickangle=-45,
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Cuanto más oscura la celda, más distintivo es ese término para ese grupo. "
        "Las celdas claras en ambos grupos indican vocabulario compartido."
    )

# ---------------------------------------------------------------------------
# SECCIÓN 5 — Entidades NER
# ---------------------------------------------------------------------------
elif seccion == "🏷️ Entidades (NER)":
    st.title("Entidades reconocidas por spaCy")
    st.markdown(
        "Entidades extraídas con el pipeline ajustado (`EntityRuler` + `ner`). "
        "Permite ver qué actores, instituciones e índices menciona cada medio."
    )

    labels_disponibles = sorted(df_entidades["label"].unique().tolist())
    labels_sel = st.multiselect(
        "Filtrar por tipo de entidad",
        labels_disponibles,
        default=[l for l in ["ORG", "PER", "LOC", "MISC"] if l in labels_disponibles],
    )

    df_ent_filtrado = df_entidades[df_entidades["label"].isin(labels_sel)]

    col1, col2 = st.columns(2)
    for i, grupo in enumerate(grupos_unicos):
        df_g   = df_ent_filtrado[df_ent_filtrado["grupo"] == grupo]
        conteo = df_g["texto"].value_counts().head(15).reset_index()
        conteo.columns = ["entidad", "frecuencia"]

        with [col1, col2][i]:
            st.subheader(grupo)
            fig = px.bar(
                conteo.sort_values("frecuencia", ascending=True),
                x="frecuencia", y="entidad", orientation="h",
                color_discrete_sequence=[COLORES.get(grupo, "#888")],
                labels={"frecuencia": "Menciones", "entidad": ""},
            )
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Distribución por tipo de entidad")
    dist = (
        df_ent_filtrado
        .groupby(["grupo", "label"])
        .size()
        .reset_index(name="frecuencia")
    )
    fig = px.bar(
        dist, x="label", y="frecuencia", color="grupo",
        barmode="group", color_discrete_map=COLORES,
        labels={"label": "Tipo", "frecuencia": "Menciones", "grupo": "Grupo"},
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# SECCIÓN 6 — Buscador de fragmentos con spaCy
# ---------------------------------------------------------------------------
elif seccion == "🔍 Buscador de fragmentos":
    st.title("Buscador de fragmentos")
    st.markdown(
        "Buscá un término por **forma** o por **lema** usando el mismo pipeline ajustado "
        "del notebook. Permite volver de la lectura distante a la lectura cercana."
    )
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        termino_buscar = st.text_input(
            "Término a buscar",
            placeholder="ej: inflación, dato, Caputo, IPC",
        )
    with col2:
        grupo_filtro = st.selectbox("Grupo", ["todos"] + grupos_unicos)
    with col3:
        ventana = st.slider("Ventana (chars)", 100, 400, VENTANA_FRAGMENTO)
    if termino_buscar:
        termino_lower = termino_buscar.strip().lower()
        with st.spinner("Buscando con spaCy..."):
            resultados = extraer_fragmentos_spacy(
                nlp, df, grupo_filtro, termino_lower,
                ventana=ventana, max_fragmentos=6,
            )
        if resultados:
            st.success(f"{len(resultados)} fragmento(s) encontrado(s) para **'{termino_buscar}'**.")
            for r in resultados:
                color = COLORES.get(r["grupo"], "#888")
                fragmento_html = re.sub(
                    rf"(?i)({re.escape(termino_buscar)})",
                    r"<mark>\1</mark>",
                    r["fragmento"],
                )
                st.markdown(
                    f"""
                    <div style="border-left:4px solid {color}; padding:8px 14px;
                                margin-bottom:14px; border-radius:4px; background:#f9f9f9;
                                color:#111111;">
                        <small><strong style="color:{color}">{r['grupo'].upper()}</strong>
                        — <span style="color:#111111">{r['titulo']}</span></small><br><br>
                        <span style="line-height:1.6; color:#111111">{fragmento_html}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.warning(
                f"No se encontró **'{termino_buscar}'** (ni por forma ni por lema) "
                "en el grupo seleccionado."
            )
    else:
        st.info("Ingresá un término para buscar sus ocurrencias en el corpus.")
        st.subheader("Términos sugeridos (top TF-IDF)")
        st.dataframe(
            top_tfidf.groupby("grupo").head(5)[["grupo", "termino", "tfidf_medio"]],
            use_container_width=True,
        )