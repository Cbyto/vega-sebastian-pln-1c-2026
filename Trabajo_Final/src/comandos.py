from src.db import listar_documentos
from src.vector_store import buscar_chunks


def es_comando(texto):
    return texto.strip().startswith("--")


def procesar_comando(texto):
    partes = texto.strip().split(maxsplit=1)
    comando = partes[0].lower()
    argumento = partes[1].strip() if len(partes) > 1 else ""

    if comando == "--help":
        return ayuda()

    if comando == "--info":
        return info_documentos()

    if comando == "--docs":
        return docs_por_entidad(argumento)

    if comando == "--buscar":
        return busqueda_directa(argumento)
    
    return "Comando no reconocido. Escribí `--help` para ver los comandos disponibles."


def ayuda():
    return """
### Comandos disponibles

`--help`  
Muestra esta ayuda.

`--info`  
Lista las normas cargadas.

`--docs <entidad>`  
Muestra documentos de una obra social/prepaga.  
Ejemplo: `--docs OSDE`

`--buscar <texto>`  
Busca fragmentos relevantes sin generar una respuesta larga.  
Ejemplo: `--buscar autorización resonancia` 
"""

def info_documentos():
    docs = listar_documentos()

    if not docs:
        return "Todavía no hay normas cargadas."

    lineas = ["### Normas cargadas"]

    for doc in docs:
        doc_id, entidad, tipo, archivo, fecha, estado = doc
        lineas.append(
            f"- **{entidad}** | {tipo or 'Sin tipo'} | `{archivo}` | {fecha} | {estado}"
        )

    return "\n".join(lineas)


def docs_por_entidad(entidad):
    if not entidad:
        return "Indicá una entidad. Ejemplo: `--docs OSDE`"

    docs = listar_documentos()
    filtrados = [
        d for d in docs
        if entidad.lower() in d[1].lower()
    ]

    if not filtrados:
        return f"No encontré documentos cargados para `{entidad}`."

    lineas = [f"### Documentos para {entidad}"]

    for doc in filtrados:
        doc_id, entidad, tipo, archivo, fecha, estado = doc
        lineas.append(
            f"- **{entidad}** | {tipo or 'Sin tipo'} | `{archivo}` | {fecha} | {estado}"
        )

    return "\n".join(lineas)


def busqueda_directa(texto):
    if not texto:
        return "Indicá qué querés buscar. Ejemplo: `--buscar autorización resonancia`"

    resultados = buscar_chunks(texto, top_k=5)

    if not resultados:
        return "No encontré fragmentos relevantes."

    lineas = [f"### Resultados para: `{texto}`"]

    for i, item in enumerate(resultados, start=1):
        meta = item["metadata"]
        fragmento = item["texto"][:700].replace("\n", " ")

        lineas.append(
            f"""
            **Resultado {i}**  
            Archivo: `{meta.get("archivo")}`  
            Entidad: **{meta.get("entidad")}**  
            Página: **{meta.get("pagina")}**  
            
            > {fragmento}...
            """
        )

    return "\n".join(lineas)