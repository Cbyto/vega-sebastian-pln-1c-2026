"""
Script de scraping para el TPI2: Text Mining y análisis discursivo comparado.
Corpus: Cenital vs. Anfibia — cobertura de Inteligencia Artificial.

Requisitos:
    pip install trafilatura

Uso:
    python ScraperCorpus_TPI2.py

Genera: corpus_tpi2.csv (listo para usar en el notebook del TPI2)
"""

import csv
import time
from datetime import date

import trafilatura

# ---------------------------------------------------------------------------
# URLs seleccionadas — 5 de Cenital y 5 de Anfibia sobre IA
# ---------------------------------------------------------------------------

ARTICULOS = [
    # --- CENITAL ---
    {
        "id": "c01",
        "fecha": "2025-10-16",
        "medio": "Cenital",
        "autor": "Cenital",
        "titulo": "La burbuja de la inteligencia artificial",
        "url": "https://cenital.com/la-burbuja-de-la-inteligencia-artificial/",
        "grupo_comparacion": "Cenital",
    },
    {
        "id": "c02",
        "fecha": "2026-04-23",
        "medio": "Cenital",
        "autor": "Cenital",
        "titulo": "La inteligencia artificial no muestra el pasado, lo reescribe",
        "url": "https://cenital.com/la-inteligencia-artificial-no-muestra-el-pasado-lo-reescribe/",
        "grupo_comparacion": "Cenital",
    },
    {
        "id": "c03",
        "fecha": "2025-05-13",
        "medio": "Cenital",
        "autor": "Cenital",
        "titulo": "Espectros, vampiros y fantasmas: ¿para esto usamos la inteligencia artificial?",
        "url": "https://cenital.com/espectros-vampiros-y-fantasmas-para-esto-usamos-la-inteligencia-artificial/",
        "grupo_comparacion": "Cenital",
    },
    {
        "id": "c04",
        "fecha": "2025-05-13",
        "medio": "Cenital",
        "autor": "Cenital",
        "titulo": "Una nueva vida no biológica: ¿qué tan gobernados estamos por la inteligencia artificial?",
        "url": "https://cenital.com/una-nueva-vida-no-biologica-que-tan-gobernados-estamos-por-la-inteligencia-artificial/",
        "grupo_comparacion": "Cenital",
    },
    {
        "id": "c05",
        "fecha": "2024-12-13",
        "medio": "Cenital",
        "autor": "Cenital",
        "titulo": "La inteligencia artificial no previene la estupidez natural",
        "url": "https://cenital.com/la-inteligencia-artificial-no-previene-la-estupidez-natural/",
        "grupo_comparacion": "Cenital",
    },

    # --- ANFIBIA ---
    {
        "id": "a01",
        "fecha": "2024-09-16",
        "medio": "Anfibia",
        "autor": "Anfibia",
        "titulo": "Es lo que AI",
        "url": "https://www.revistaanfibia.com/inteligencia-artificial-es-lo-que-ai/",
        "grupo_comparacion": "Anfibia",
    },
    {
        "id": "a02",
        "fecha": "2023-09-22",
        "medio": "Anfibia",
        "autor": "Sofía Trejo",
        "titulo": "Inteligencia artificial, pero a qué costo",
        "url": "https://www.revistaanfibia.com/inteligencia-artificial-pero-a-que-costo/",
        "grupo_comparacion": "Anfibia",
    },
    {
        "id": "a03",
        "fecha": "2023-04-13",
        "medio": "Anfibia",
        "autor": "Anfibia",
        "titulo": "Inteligencia Artificial (dossier)",
        "url": "https://www.revistaanfibia.com/inteligencia-artificial/",
        "grupo_comparacion": "Anfibia",
    },
    {
        "id": "a04",
        "fecha": "2023-04-25",
        "medio": "Anfibia",
        "autor": "Anfibia",
        "titulo": "¿Hacia dónde nos lleva la inteligencia artificial?",
        "url": "https://www.revistaanfibia.com/hacia-donde-nos-lleva-la-inteligencia-artificial/",
        "grupo_comparacion": "Anfibia",
    },
    {
        "id": "a05",
        "fecha": "2025-08-27",
        "medio": "Anfibia",
        "autor": "Anfibia",
        "titulo": "Una IA latinoamericana es posible",
        "url": "https://www.revistaanfibia.com/latamgpt-una-inteligencia-artificial-latinoamericana-es-posible/",
        "grupo_comparacion": "Anfibia",
    },
]

# ---------------------------------------------------------------------------
# Función de scraping con trafilatura
# ---------------------------------------------------------------------------

def scrapear(url: str, pausa: float = 2.0) -> str:
    """
    Descarga la URL y extrae el texto con trafilatura.
    Devuelve el texto o una cadena vacía si falla.
    """
    try:
        html = trafilatura.fetch_url(url)
        if html is None:
            print(f"  [!] No se pudo descargar: {url}")
            return ""
        texto = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
            favor_precision=True,
        )
        time.sleep(pausa)  # pausa cortés entre requests
        return texto.strip() if texto else ""
    except Exception as e:
        print(f"  [!] Error en {url}: {e}")
        return ""


# ---------------------------------------------------------------------------
# Loop principal y escritura del CSV
# ---------------------------------------------------------------------------

SALIDA = "corpus_tpi2.csv"
COLUMNAS = ["id", "fecha", "medio", "autor", "titulo", "texto", "grupo_comparacion"]

print(f"Iniciando scraping de {len(ARTICULOS)} artículos...\n")

filas = []
for art in ARTICULOS:
    print(f"→ [{art['id']}] {art['titulo'][:60]}...")
    texto = scrapear(art["url"])
    palabras = len(texto.split()) if texto else 0
    print(f"   ✓ {palabras} palabras extraídas")

    filas.append({
        "id": art["id"],
        "fecha": art["fecha"],
        "medio": art["medio"],
        "autor": art["autor"],
        "titulo": art["titulo"],
        "texto": texto,
        "grupo_comparacion": art["grupo_comparacion"],
    })

# Verificación rápida antes de guardar
vacios = [f["id"] for f in filas if not f["texto"]]
if vacios:
    print(f"\n⚠️  Artículos sin texto extraído: {vacios}")
    print("   Revisá esas URLs manualmente y pegá el texto a mano si es necesario.")

with open(SALIDA, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=COLUMNAS)
    writer.writeheader()
    writer.writerows(filas)

print(f"\n✅ Corpus guardado en: {SALIDA}")
print(f"   Total de artículos: {len(filas)}")
print(f"   Grupos: Cenital ({sum(1 for r in filas if r['grupo_comparacion']=='Cenital')}) | "
      f"Anfibia ({sum(1 for r in filas if r['grupo_comparacion']=='Anfibia')})")
print(f"   Columnas: {', '.join(COLUMNAS)}")
print("\nPodés abrir corpus_tpi2.csv y moverlo a la misma carpeta que el notebook.")