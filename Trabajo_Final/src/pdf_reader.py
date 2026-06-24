import sqlite3
import fitz  # PyMuPDF

from config import DB_PATH


# ---------------------------------------------------------------------------
# Validación de texto extraído
# ---------------------------------------------------------------------------
def es_texto_valido(texto: str, umbral_basura: float = 0.30) -> bool:
    """
    Devuelve True si el texto parece legible.

    Criterios:
    - Mínimo 20 caracteres de contenido.
    - Menos del `umbral_basura` (30%) de caracteres no imprimibles
      (excluye saltos de línea y tabs, que son normales).
    """
    if not texto or len(texto.strip()) < 20:
        return False

    no_imprimibles = sum(
        1 for c in texto
        if ord(c) < 32 and c not in ('\n', '\t', '\r')
    )

    return (no_imprimibles / len(texto)) < umbral_basura


def pdf_tiene_texto_valido(ruta_pdf: str, paginas_a_revisar: int = 3) -> bool:
    """
    Revisa las primeras N páginas del PDF para determinar si tiene texto
    nativo legible. Devuelve True si al menos la mitad tiene texto válido.
    """
    doc = fitz.open(ruta_pdf)
    total = min(paginas_a_revisar, len(doc))
    validas = 0

    for i in range(total):
        texto = doc[i].get_text("text")
        if es_texto_valido(texto):
            validas += 1

    doc.close()
    return validas >= (total / 2)


# ---------------------------------------------------------------------------
# Extracción nativa (PyMuPDF)
# ---------------------------------------------------------------------------
def extraer_texto_pdf(ruta_pdf: str) -> list[dict]:
    """
    Extrae texto nativo página por página con PyMuPDF.

    Devuelve lista de {"pagina": int, "texto": str} solo para páginas
    con texto válido.

    Si el PDF no tiene texto legible, devuelve lista vacía — en ese caso
    app.py debe marcar el documento como 'pendiente_ocr' para que
    claude_vision.py lo procese en el siguiente batch.
    """
    paginas = []
    doc = fitz.open(ruta_pdf)

    for i, page in enumerate(doc, start=1):
        texto = page.get_text("text")

        if texto and es_texto_valido(texto):
            paginas.append({
                "pagina": i,
                "texto": texto.strip()
            })

    doc.close()
    return paginas


# ---------------------------------------------------------------------------
# Helper para marcar un documento como pendiente_ocr en SQLite
# ---------------------------------------------------------------------------
def marcar_pendiente_ocr(doc_id: int):
    """
    Actualiza el estado del documento a 'pendiente_ocr' en la base de datos.
    Se llama desde app.py cuando extraer_texto_pdf() devuelve lista vacía.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "UPDATE documentos SET estado = 'pendiente_ocr', metodo_extraccion = 'pendiente' WHERE id = ?",
        (doc_id,)
    )
    conn.commit()
    conn.close()
