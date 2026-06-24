from dotenv import load_dotenv
load_dotenv()

import anthropic
from openai import OpenAI

from config import AI_PROVIDER, CHAT_MODEL, ANTHROPIC_API_KEY, OPENAI_API_KEY
from src.vector_store import buscar_chunks

# ---------------------------------------------------------------------------
# Clientes (solo se instancia el que corresponde)
# ---------------------------------------------------------------------------
_openai_client    = None
_anthropic_client = None

if AI_PROVIDER == "anthropic":
    _anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
else:
    _openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------------------------------------------------------
# Configuración de historial
# ---------------------------------------------------------------------------
HISTORIAL_MAX_MENSAJES = 6   # últimos N mensajes (user + assistant intercalados)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
Sos un asistente especializado en normas operativas de obras sociales y prepagas.

Reglas obligatorias:
- Respondé únicamente usando el CONTEXTO provisto.
- Si la respuesta no está en el contexto, decí que no encontraste información suficiente.
- No inventes requisitos, fechas, códigos, coberturas ni autorizaciones.
- Sé claro y concreto.
- Al final de cada respuesta, agregá una sección "Fuentes" con archivo y página.
- Podés usar el historial de la conversación para entender referencias como
  "y en esa misma obra social", "¿y para ese trámite?", "¿qué pasa si...?", etc.
  pero el CONTEXTO siempre tiene prioridad sobre el historial para los datos concretos.
"""


def armar_contexto(chunks):
    partes = []

    for i, item in enumerate(chunks, start=1):
        meta = item["metadata"]
        partes.append(
            f"[Fuente {i}]\n"
            f"Entidad: {meta.get('entidad')}\n"
            f"Archivo: {meta.get('archivo')}\n"
            f"Página: {meta.get('pagina')}\n"
            f"Texto:\n{item['texto']}"
        )

    return "\n\n".join(partes)


def preparar_historial(historial: list[dict], max_mensajes: int) -> list[dict]:
    """
    Toma el historial completo de la sesión (lista de {"role", "content"})
    y devuelve los últimos `max_mensajes` mensajes, asegurándose de que:
    - Siempre empiece con un mensaje de "user" (requisito de ambas APIs).
    - No incluya el mensaje actual (se agrega aparte con el contexto RAG).
    """
    if not historial:
        return []

    # Tomamos los últimos N mensajes del historial
    recientes = historial[-max_mensajes:]

    # Si el primero es "assistant", lo descartamos para cumplir el requisito
    # de que los mensajes siempre arranquen con "user"
    if recientes and recientes[0]["role"] == "assistant":
        recientes = recientes[1:]

    return recientes


# ---------------------------------------------------------------------------
# Llamadas a la API — ahora reciben historial
# ---------------------------------------------------------------------------
def _llamar_openai(contexto: str, pregunta: str, historial: list[dict]) -> str:
    """
    Arma los mensajes intercalando:
      [system] → [historial previo] → [user con contexto RAG + pregunta actual]
    """
    mensajes = [{"role": "system", "content": SYSTEM_PROMPT}]
    mensajes.extend(historial)
    mensajes.append({
        "role": "user",
        "content": f"CONTEXTO DE NORMAS RELEVANTE:\n{contexto}\n\nPREGUNTA:\n{pregunta}"
    })

    response = _openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=mensajes
    )
    return response.choices[0].message.content


def _llamar_anthropic(contexto: str, pregunta: str, historial: list[dict]) -> str:
    """
    Anthropic maneja el system prompt por separado.
    Los mensajes intercalan historial + pregunta actual con contexto RAG.
    """
    mensajes = []
    mensajes.extend(historial)
    mensajes.append({
        "role": "user",
        "content": f"CONTEXTO DE NORMAS RELEVANTE:\n{contexto}\n\nPREGUNTA:\n{pregunta}"
    })

    response = _anthropic_client.messages.create(
        model=CHAT_MODEL,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=mensajes
    )
    return response.content[0].text


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------
def responder_pregunta(pregunta: str, historial: list[dict] | None = None) -> dict:
    """
    Parámetros:
      pregunta  — texto del usuario en el turno actual.
      historial — lista de mensajes anteriores de la sesión
                  [{"role": "user"|"assistant", "content": "..."}]
                  No debe incluir el mensaje actual.
    """
    historial = historial or []

    # Buscar chunks relevantes para la pregunta actual
    chunks = buscar_chunks(pregunta)

    if not chunks:
        return {
            "respuesta": "No encontré información suficiente en las normas cargadas.",
            "fuentes": [],
            "proveedor": AI_PROVIDER,
            "modelo": CHAT_MODEL
        }

    contexto = armar_contexto(chunks)
    historial_recortado = preparar_historial(historial, HISTORIAL_MAX_MENSAJES)

    if AI_PROVIDER == "anthropic":
        respuesta = _llamar_anthropic(contexto, pregunta, historial_recortado)
    else:
        respuesta = _llamar_openai(contexto, pregunta, historial_recortado)

    return {
        "respuesta": respuesta,
        "fuentes":   [item["metadata"] for item in chunks],
        "proveedor": AI_PROVIDER,
        "modelo":    CHAT_MODEL
    }
