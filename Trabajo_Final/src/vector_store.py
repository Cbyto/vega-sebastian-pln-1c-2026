from dotenv import load_dotenv
load_dotenv()

import chromadb
from openai import OpenAI, RateLimitError, APIError, AuthenticationError

from config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, TOP_K, OPENAI_API_KEY

client_openai = OpenAI(api_key=OPENAI_API_KEY)

client_chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))

collection = client_chroma.get_or_create_collection(
    name=COLLECTION_NAME
)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
def crear_embedding(texto):
    try:
        response = client_openai.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texto
        )
        return response.data[0].embedding

    except RateLimitError as e:
        raise RuntimeError(
            "No se pudo generar el embedding: la cuenta de OpenAI no tiene cuota disponible. "
            "Revisá Billing/Usage en la plataforma de OpenAI."
        ) from e

    except AuthenticationError as e:
        raise RuntimeError(
            "No se pudo autenticar con OpenAI. "
            "Revisá que OPENAI_API_KEY esté bien configurada en el archivo .env."
        ) from e

    except APIError as e:
        raise RuntimeError(
            f"Error de API de OpenAI al generar embedding: {str(e)}"
        ) from e


# ---------------------------------------------------------------------------
# Chequeo de duplicados
# ---------------------------------------------------------------------------
def chunk_existe(chunk_id):
    """
    Devuelve True si el chunk_id ya está en la colección.
    """
    try:
        resultado = collection.get(ids=[chunk_id])
        return len(resultado["ids"]) > 0
    except Exception:
        return False


def filtrar_duplicados(chunks):
    """
    Recibe la lista completa de chunks y devuelve solo los que no existen aún.
    También informa cuántos se saltaron.
    """
    nuevos = []
    duplicados = 0

    for item in chunks:
        if chunk_existe(item["id"]):
            duplicados += 1
        else:
            nuevos.append(item)

    return nuevos, duplicados


# ---------------------------------------------------------------------------
# Inserción
# ---------------------------------------------------------------------------
def agregar_chunks(chunks):
    """
    Inserta chunks en ChromaDB ignorando los que ya existen.
    Devuelve un dict con cantidad insertada y cantidad de duplicados salteados.
    """
    nuevos, duplicados = filtrar_duplicados(chunks)

    if not nuevos:
        return {"insertados": 0, "duplicados_salteados": duplicados}

    ids        = []
    documents  = []
    metadatas  = []
    embeddings = []

    for item in nuevos:
        ids.append(item["id"])
        documents.append(item["texto"])
        metadatas.append(item["metadata"])
        embeddings.append(crear_embedding(item["texto"]))

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )

    return {"insertados": len(nuevos), "duplicados_salteados": duplicados}


# ---------------------------------------------------------------------------
# Búsqueda
# ---------------------------------------------------------------------------
def buscar_chunks(pregunta, top_k=TOP_K):
    embedding = crear_embedding(pregunta)

    result = collection.query(
        query_embeddings=[embedding],
        n_results=top_k
    )

    documentos = result.get("documents", [[]])[0]
    metadatas  = result.get("metadatas",  [[]])[0]
    distances  = result.get("distances",  [[]])[0]

    encontrados = []

    for texto, metadata, distance in zip(documentos, metadatas, distances):
        encontrados.append({
            "texto":    texto,
            "metadata": metadata,
            "distance": distance
        })

    return encontrados
