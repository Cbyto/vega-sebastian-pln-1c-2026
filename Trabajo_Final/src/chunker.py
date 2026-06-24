import re


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------
def dividir_en_chunks(texto, max_chars=1200, overlap=150):
    """
    Divide texto en chunks respetando la estructura natural del documento.

    Estrategia por prioridad:
      1. Detecta bloques de tabla y los mantiene unidos.
      2. Nunca corta en el medio de una oración.
      3. Prefiere cortar entre párrafos (línea en blanco).
      4. Si un párrafo es muy largo, corta entre oraciones.
      5. Aplica overlap entre chunks para no perder contexto en los bordes.
    """
    texto = texto.strip()

    if not texto:
        return []

    if len(texto) <= max_chars:
        return [texto]

    # Paso 1: segmentar el texto en bloques (párrafos normales o tablas)
    bloques = _segmentar_en_bloques(texto)

    # Paso 2: agrupar bloques en chunks respetando max_chars
    chunks = []
    chunk_actual = ""
    overlap_buffer = ""

    for bloque in bloques:
        es_tabla = bloque["tipo"] == "tabla"
        contenido = bloque["texto"]

        # Si el bloque de tabla solo ya supera max_chars, lo dividimos
        # en sub-bloques por filas pero sin mezclar con otro contenido
        if es_tabla and len(contenido) > max_chars:
            # Vaciar chunk actual antes de procesar la tabla grande
            if chunk_actual.strip():
                chunks.append(chunk_actual.strip())
                overlap_buffer = _extraer_overlap(chunk_actual, overlap)
                chunk_actual = overlap_buffer

            sub_chunks = _dividir_tabla_grande(contenido, max_chars, overlap)
            chunks.extend(sub_chunks)
            overlap_buffer = _extraer_overlap(sub_chunks[-1], overlap) if sub_chunks else ""
            chunk_actual = overlap_buffer
            continue

        # Intentar agregar el bloque al chunk actual
        candidato = (chunk_actual + "\n\n" + contenido).strip()

        if len(candidato) <= max_chars:
            chunk_actual = candidato
        else:
            # No entra: guardar chunk actual y empezar uno nuevo
            if chunk_actual.strip():
                chunks.append(chunk_actual.strip())
                overlap_buffer = _extraer_overlap(chunk_actual, overlap)

            if es_tabla:
                # Las tablas no se mezclan con overlap de texto anterior
                chunk_actual = contenido
            else:
                # Párrafo normal: aplicar overlap y agregar oraciones
                chunk_actual = _agregar_parrafo_con_overlap(
                    overlap_buffer, contenido, max_chars, overlap, chunks
                )

    if chunk_actual.strip():
        chunks.append(chunk_actual.strip())

    return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Segmentación en bloques (párrafos vs tablas)
# ---------------------------------------------------------------------------
def _segmentar_en_bloques(texto):
    """
    Divide el texto en bloques tipados: {"tipo": "parrafo"|"tabla", "texto": str}

    Heurística de detección de tabla:
    - 4 o más líneas consecutivas (sin línea en blanco entre ellas)
    - Donde al menos el 60% de las líneas son "cortas" (< 80 chars)
    - Indica una estructura tabular: celdas, ítems de lista densa, etc.
    """
    # Separar en grupos de líneas continuas (bloques separados por línea en blanco)
    grupos_raw = re.split(r'\n\s*\n', texto)
    bloques = []

    for grupo in grupos_raw:
        grupo = grupo.strip()
        if not grupo:
            continue

        lineas = [l for l in grupo.split('\n') if l.strip()]

        if _es_tabla(lineas):
            bloques.append({"tipo": "tabla", "texto": grupo})
        else:
            bloques.append({"tipo": "parrafo", "texto": grupo})

    return bloques


def _es_tabla(lineas):
    """
    Devuelve True si el grupo de líneas parece una tabla o lista densa.

    Heurística basada en cómo PyMuPDF extrae el texto:
    - Los párrafos normales llegan como pocas líneas muy largas (>200 chars).
    - Las tablas llegan como muchas líneas medianas (40-150 chars), una por celda/fila.

    Criterios:
    - Mínimo 4 líneas consecutivas.
    - Promedio de longitud menor a 150 chars/línea (distingue de párrafos largos).
    - Al menos 50% de líneas entre 20 y 150 chars (el rango típico de una celda).
    """
    if len(lineas) < 4:
        return False

    promedio = sum(len(l.strip()) for l in lineas) / len(lineas)
    if promedio >= 150:
        return False

    rango_celda = sum(1 for l in lineas if 20 <= len(l.strip()) <= 150)
    return (rango_celda / len(lineas)) >= 0.5


def _dividir_tabla_grande(texto_tabla, max_chars, overlap):
    """
    Para tablas que superan max_chars solas: divide por filas
    manteniendo siempre filas completas juntas (nunca corta una fila al medio).
    """
    lineas = texto_tabla.split('\n')
    sub_chunks = []
    chunk_actual = ""

    for linea in lineas:
        candidato = (chunk_actual + "\n" + linea).strip()

        if len(candidato) <= max_chars:
            chunk_actual = candidato
        else:
            if chunk_actual.strip():
                sub_chunks.append(chunk_actual.strip())
                # Overlap: últimas N líneas del chunk anterior
                lineas_overlap = chunk_actual.split('\n')[-3:]
                chunk_actual = "\n".join(lineas_overlap) + "\n" + linea
            else:
                # Línea sola que supera max_chars (caso extremo)
                sub_chunks.append(linea[:max_chars])
                chunk_actual = linea[max_chars - overlap:]

    if chunk_actual.strip():
        sub_chunks.append(chunk_actual.strip())

    return sub_chunks


# ---------------------------------------------------------------------------
# Manejo de párrafos normales con oraciones
# ---------------------------------------------------------------------------
def _agregar_parrafo_con_overlap(overlap_buffer, parrafo, max_chars, overlap, chunks):
    """
    Intenta agregar un párrafo normal al chunk actual (con overlap previo).
    Si el párrafo es largo, lo parte por oraciones.
    Devuelve el nuevo chunk_actual.
    """
    oraciones = _partir_en_oraciones(parrafo)
    chunk_actual = overlap_buffer

    for oracion in oraciones:
        candidato = (chunk_actual + " " + oracion).strip()

        if len(candidato) <= max_chars:
            chunk_actual = candidato
        else:
            if chunk_actual.strip():
                chunks.append(chunk_actual.strip())
                overlap_buffer = _extraer_overlap(chunk_actual, overlap)
                chunk_actual = (overlap_buffer + " " + oracion).strip()
            else:
                # Oración sola que supera max_chars
                chunks.append(oracion[:max_chars].strip())
                chunk_actual = oracion[max_chars - overlap:].strip()

    return chunk_actual


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _partir_en_oraciones(texto):
    """
    Divide un bloque de texto en oraciones.
    Respeta abreviaturas comunes en el contexto médico/administrativo.
    """
    # Importante: los lookbehind en Python requieren longitud FIJA.
    # "O.S" cubre tanto "O.S." como "O.S " (obra social abreviada)
    abreviaturas = (
        r'(?<!Dr)'
        r'(?<!Dra)'
        r'(?<!Sr)'
        r'(?<!Sra)'
        r'(?<!Art)'
        r'(?<!Inc)'
        r'(?<!Ej)'
        r'(?<!O\.S)'
    )

    patron = abreviaturas + r'(?<=[.!?])\s+'
    partes = re.split(patron, texto)
    return [p.strip() for p in partes if p.strip()]


def _extraer_overlap(texto, overlap_chars):
    """
    Devuelve los últimos `overlap_chars` caracteres del texto
    sin cortar en el medio de una palabra.
    """
    if len(texto) <= overlap_chars:
        return texto

    fragmento = texto[-overlap_chars:]

    idx = fragmento.find(" ")
    if idx != -1:
        fragmento = fragmento[idx:].strip()

    return fragmento
