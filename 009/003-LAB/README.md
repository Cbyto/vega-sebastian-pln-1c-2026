# 📰 TP3 PLN — Infobae vs. La Nación (Inflación 2026)

Este proyecto es una aplicación interactiva desarrollada con **Streamlit** para analizar y comparar el tratamiento periodístico de la inflación durante el año 2026 en dos de los principales medios digitales de Argentina: **Infobae** y **La Nación**.

La herramienta utiliza técnicas de **Procesamiento de Lenguaje Natural (PLN)** para extraer insights, comparar vocabularios y visualizar tendencias en el discurso mediático.

## 🚀 Demo en Vivo
Podés ver la aplicación funcionando en Streamlit Cloud aquí:
👉 **[https://vega-sebastian-pln-1c-2026-hvplemyqelhcnkwvaqmvug.streamlit.app/]**

---

## 🛠️ Tecnologías utilizadas
* **Python 3.11.9+**
* **Streamlit**: Framework de la interfaz de usuario.
* **spaCy**: Procesamiento de texto, lematización y Reconocimiento de Entidades Nombradas (NER) usando el modelo `es_core_news_md`.
* **Scikit-Learn**: Vectorización de texto (BoW y TF-IDF).
* **Plotly**: Visualizaciones interactivas.
* **Pandas**: Manipulación y análisis de datos.

⚠️ Nota importante: El archivo requirements.txt incluye la descarga directa del modelo es_core_news_md de spaCy. Si en el futuro necesitás actualizar la versión del modelo (por ejemplo, a una versión 3.9.x), asegurate de actualizar también la versión de la librería spacy en ese mismo archivo para que coincidan y evitar errores de compatibilidad.

---

## 📊 Funcionalidades
1.  **Auditoría del corpus**: Estadísticas generales sobre la cantidad de documentos y palabras por medio.
2.  **BoW vs. TF-IDF**: Análisis de los términos más frecuentes frente a los más distintivos de cada grupo.
3.  **Bigramas**: Identificación de frases de dos palabras que suelen aparecer juntas.
4.  **Heatmap**: Mapa de calor para visualizar el peso relativo de términos clave.
5.  **Entidades (NER)**: Extracción automática de Personas (PER), Organizaciones (ORG), Lugares (LOC) e Índices (MISC).
6.  **Buscador de fragmentos**: Herramienta para realizar "lectura cercana" buscando términos por su forma o lema.

---