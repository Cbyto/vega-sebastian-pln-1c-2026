import os
import shutil

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from config import DOCUMENTOS_DIR
from src.db import (
    init_db, insertar_documento, listar_documentos,
    insertar_solicitud, listar_solicitudes, actualizar_estado_solicitud,
    insertar_reporte, listar_reportes, actualizar_estado_reporte,
    crear_sesion, guardar_mensaje, cargar_historial
)
from src.pdf_reader import extraer_texto_pdf, marcar_pendiente_ocr
from src.chunker import dividir_en_chunks
from src.vector_store import agregar_chunks
from src.rag import responder_pregunta
from src.comandos import es_comando, procesar_comando


init_db()

st.set_page_config(
    page_title="Chat Normas Operativas",
    page_icon="📄",
    layout="wide"
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "admin_logueado" not in st.session_state:
    st.session_state.admin_logueado = False

if "ultima_respuesta" not in st.session_state:
    st.session_state.ultima_respuesta = None

if "reporte_enviado" not in st.session_state:
    st.session_state.reporte_enviado = False

if "solicitud_enviada" not in st.session_state:
    st.session_state.solicitud_enviada = False

if "mostrar_form_reporte" not in st.session_state:
    st.session_state.mostrar_form_reporte = False

# Sesion persistida: se crea una vez por pestaña abierta
if "sesion_id" not in st.session_state:
    st.session_state.sesion_id = crear_sesion()
    # Cargar historial previo de esta sesión desde SQLite
    st.session_state.messages = cargar_historial(st.session_state.sesion_id)


# ---------------------------------------------------------------------------
# Login admin
# ---------------------------------------------------------------------------
def login_admin():
    st.sidebar.subheader("Administrador")

    if st.session_state.admin_logueado:
        st.sidebar.success("Administrador conectado")

        if st.sidebar.button("Cerrar sesión"):
            st.session_state.admin_logueado = False
            st.rerun()

        return

    usuario = st.sidebar.text_input("Usuario")
    password = st.sidebar.text_input("Contraseña", type="password")

    if st.sidebar.button("Ingresar"):
        admin_user = os.getenv("ADMIN_USER")
        admin_pass = os.getenv("ADMIN_PASS")

        if usuario == admin_user and password == admin_pass:
            st.session_state.admin_logueado = True
            st.sidebar.success("Administrador conectado")
            st.rerun()
        else:
            st.sidebar.error("Usuario o contraseña incorrectos")


# ---------------------------------------------------------------------------
# Panel admin — carga de PDFs
# ---------------------------------------------------------------------------
def procesar_pdf(uploaded_file, entidad, tipo_documento):
    destino = DOCUMENTOS_DIR / uploaded_file.name

    with open(destino, "wb") as f:
        shutil.copyfileobj(uploaded_file, f)

    doc_id = insertar_documento(
        entidad=entidad,
        tipo_documento=tipo_documento,
        nombre_archivo=uploaded_file.name,
        ruta_archivo=str(destino)
    )

    paginas = extraer_texto_pdf(destino)

    if not paginas:
        marcar_pendiente_ocr(doc_id)
        return 0, "⚠️ No se pudo extraer texto nativo. El documento quedó registrado como pendiente de OCR. Corré: python claude_vision.py"

    chunks_para_guardar = []

    for pagina in paginas:
        nro_pagina = pagina["pagina"]
        texto = pagina["texto"]
        chunks = dividir_en_chunks(texto)

        for idx, chunk in enumerate(chunks):
            chunk_id = f"doc_{doc_id}_pag_{nro_pagina}_chunk_{idx}"

            chunks_para_guardar.append({
                "id": chunk_id,
                "texto": chunk,
                "metadata": {
                    "documento_id": doc_id,
                    "entidad": entidad,
                    "tipo_documento": tipo_documento,
                    "archivo": uploaded_file.name,
                    "pagina": nro_pagina,
                    "chunk": idx
                }
            })

    resultado = agregar_chunks(chunks_para_guardar)
    insertados = resultado["insertados"]
    duplicados = resultado["duplicados_salteados"]

    if insertados == 0 and duplicados > 0:
        return 0, f"El documento ya estaba cargado ({duplicados} fragmentos duplicados salteados)."

    mensaje = "PDF procesado correctamente."
    if duplicados > 0:
        mensaje += f" ({duplicados} fragmentos duplicados salteados)"

    return insertados, mensaje


def panel_admin():
    st.sidebar.divider()
    st.sidebar.header("📥 Cargar norma")

    entidad = st.sidebar.text_input(
        "Obra social / prepaga",
        placeholder="Ej: OSDE, Galeno, Swiss Medical"
    )
    tipo_documento = st.sidebar.text_input(
        "Tipo de documento",
        value="Norma operativa"
    )
    uploaded_file = st.sidebar.file_uploader("Cargar PDF", type=["pdf"])

    if st.sidebar.button("Procesar PDF"):
        if not uploaded_file:
            st.sidebar.error("Primero cargá un PDF.")
            return
        if not entidad.strip():
            st.sidebar.error("Indicá la obra social o prepaga.")
            return

        with st.status("Procesando PDF...", expanded=True):
            try:
                cantidad, mensaje = procesar_pdf(
                    uploaded_file,
                    entidad.strip(),
                    tipo_documento.strip()
                )
                if cantidad > 0:
                    st.success(f"{mensaje} Fragmentos creados: {cantidad}")
                else:
                    st.warning(mensaje)
            except RuntimeError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Error inesperado: {e}")

    # --- Solicitudes pendientes ---
    st.sidebar.divider()
    st.sidebar.header("📋 Solicitudes pendientes")

    solicitudes = listar_solicitudes(solo_pendientes=True)

    if not solicitudes:
        st.sidebar.caption("No hay solicitudes pendientes.")
    else:
        for sol in solicitudes:
            sol_id, ent, tipo, desc, solicitante, fecha, estado = sol
            with st.sidebar.expander(f"{ent} — {fecha[:10]}"):
                st.write(f"**Tipo:** {tipo or 'No especificado'}")
                st.write(f"**Solicitante:** {solicitante or 'Anónimo'}")
                st.write(f"**Descripción:** {desc or '—'}")
                col1, col2 = st.columns(2)
                if col1.button("✅ Resolver", key=f"sol_ok_{sol_id}"):
                    actualizar_estado_solicitud(sol_id, "resuelta")
                    st.rerun()
                if col2.button("❌ Descartar", key=f"sol_no_{sol_id}"):
                    actualizar_estado_solicitud(sol_id, "descartada")
                    st.rerun()

    # --- Reportes pendientes ---
    st.sidebar.divider()
    st.sidebar.header("🐛 Reportes pendientes")

    reportes = listar_reportes(solo_pendientes=True)

    if not reportes:
        st.sidebar.caption("No hay reportes pendientes.")
    else:
        for rep in reportes:
            rep_id, pregunta, respuesta, comentario, fecha, estado = rep
            with st.sidebar.expander(f"Reporte {rep_id} — {fecha[:10]}"):
                st.write(f"**Pregunta:** {pregunta[:120]}...")
                st.write(f"**Comentario:** {comentario or '—'}")
                if st.button("✅ Revisado", key=f"rep_ok_{rep_id}"):
                    actualizar_estado_reporte(rep_id, "revisado")
                    st.rerun()


# ---------------------------------------------------------------------------
# Panel de documentos (visible para todos)
# ---------------------------------------------------------------------------
def panel_documentos():
    st.sidebar.divider()
    st.sidebar.header("📄 Normas cargadas")

    docs = listar_documentos()

    if not docs:
        st.sidebar.caption("Todavía no hay normas cargadas.")
        return

    for doc in docs[:10]:
        doc_id, ent, tipo, archivo, fecha, estado = doc
        st.sidebar.caption(f"**{ent}** | {tipo or '—'} | {fecha[:10]}")


# ---------------------------------------------------------------------------
# Panel de solicitud de normas (usuarios)
# ---------------------------------------------------------------------------
def panel_solicitud():
    st.sidebar.divider()

    with st.sidebar.expander("📬 Solicitar una norma"):
        entidad_sol = st.text_input(
            "Obra social / prepaga *",
            placeholder="Ej: Swiss Medical",
            key="sol_entidad"
        )
        tipo_sol = st.text_input(
            "Tipo de documento",
            placeholder="Ej: Norma operativa, Anexo",
            key="sol_tipo"
        )
        desc_sol = st.text_area(
            "Descripción o motivo (opcional)",
            placeholder="Ej: Necesito la norma de ambulatorios.",
            key="sol_desc",
            height=80
        )
        nombre_sol = st.text_input(
            "Tu nombre (opcional)",
            key="sol_nombre"
        )

        if st.button("Enviar solicitud", key="btn_solicitud"):
            if not entidad_sol.strip():
                st.error("Indicá la obra social o prepaga.")
            else:
                insertar_solicitud(
                    entidad=entidad_sol.strip(),
                    tipo_documento=tipo_sol.strip() or None,
                    descripcion=desc_sol.strip() or None,
                    solicitante=nombre_sol.strip() or None
                )
                st.session_state.solicitud_enviada = True
                st.rerun()

        if st.session_state.solicitud_enviada:
            st.success("✅ Solicitud enviada. El administrador la revisará a la brevedad.")
            st.session_state.solicitud_enviada = False


# ---------------------------------------------------------------------------
# Widget de feedback inline (debajo del chat)
# ---------------------------------------------------------------------------
def widget_feedback():
    resultado = st.session_state.ultima_respuesta
    if not resultado:
        return

    st.divider()
    st.caption("¿Esta respuesta fue útil?")

    col1, col2 = st.columns([1, 4])

    with col1:
        if st.button("👍", key="fb_ok", help="Respuesta correcta"):
            st.session_state.ultima_respuesta = None
            st.session_state.mostrar_form_reporte = False
            st.toast("¡Gracias por tu feedback!")

    with col2:
        if st.button("👎 Reportar error", key="fb_error"):
            st.session_state.mostrar_form_reporte = True

    if st.session_state.mostrar_form_reporte:
        with st.form("form_reporte", clear_on_submit=True):
            comentario = st.text_area(
                "¿Qué estuvo mal? (opcional)",
                placeholder="Ej: La respuesta no corresponde a la norma correcta.",
                height=80
            )
            enviado = st.form_submit_button("Enviar reporte")

            if enviado:
                insertar_reporte(
                    pregunta=resultado.get("pregunta", ""),
                    respuesta=resultado.get("respuesta", ""),
                    comentario=comentario.strip() or None
                )
                st.session_state.mostrar_form_reporte = False
                st.session_state.ultima_respuesta = None
                st.session_state.reporte_enviado = True
                st.rerun()

    if st.session_state.reporte_enviado:
        st.success("Reporte enviado. ¡Gracias por ayudarnos a mejorar!")
        st.session_state.reporte_enviado = False


# ---------------------------------------------------------------------------
# Layout principal
# ---------------------------------------------------------------------------
st.title("📄 Chat Normas Operativas")

with st.sidebar:
    login_admin()

    if st.session_state.admin_logueado:
        panel_admin()

    panel_documentos()
    panel_solicitud()


st.subheader("Chat")
st.caption("Podés hacer una pregunta o escribir `--help` para ver comandos disponibles.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Preguntá algo o escribí --help")

if prompt:
    # Persistir mensaje del usuario
    guardar_mensaje(st.session_state.sesion_id, "user", prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if es_comando(prompt):
            respuesta = procesar_comando(prompt)
            st.markdown(respuesta)
            guardar_mensaje(st.session_state.sesion_id, "assistant", respuesta)
            st.session_state.messages.append({"role": "assistant", "content": respuesta})

        else:
            with st.spinner("Buscando en las normas cargadas..."):
                # Pasamos el historial SIN el mensaje actual
                historial_previo = st.session_state.messages[:-1]
                resultado = responder_pregunta(prompt, historial=historial_previo)
                respuesta = resultado["respuesta"]

            st.markdown(respuesta)

            # Persistir respuesta del asistente
            guardar_mensaje(st.session_state.sesion_id, "assistant", respuesta)
            resultado["pregunta"] = prompt
            st.session_state.ultima_respuesta = resultado
            st.session_state.messages.append({"role": "assistant", "content": respuesta})

# Widget de feedback debajo del chat
if st.session_state.ultima_respuesta:
    widget_feedback()
