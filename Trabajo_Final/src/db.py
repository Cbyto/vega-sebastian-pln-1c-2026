"""
db.py
=====
Capa de acceso a datos usando SQLAlchemy Core.

Ventaja sobre sqlite3 directo: para migrar a PostgreSQL basta con cambiar
la variable DATABASE_URL en config.py — el resto del código no se toca.

    SQLite (hoy):     sqlite:///ruta/al/archivo.db
    PostgreSQL luego: postgresql://user:pass@host:5432/dbname
"""

from datetime import datetime, timezone
import uuid

from sqlalchemy import (
    create_engine, text,
    Table, Column, MetaData,
    Integer, String, Text, DateTime,
    ForeignKey
)

from config import DATABASE_URL

# ---------------------------------------------------------------------------
# Engine — punto central de conexión
# ---------------------------------------------------------------------------
# connect_args solo aplica a SQLite; en PostgreSQL se ignora sin errores
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False, "timeout": 10},
    echo=False
)

metadata = MetaData()

# ---------------------------------------------------------------------------
# Definición de tablas
# ---------------------------------------------------------------------------
t_documentos = Table("documentos", metadata,
    Column("id",                Integer, primary_key=True, autoincrement=True),
    Column("entidad",           String,  nullable=False),
    Column("tipo_documento",    String),
    Column("nombre_archivo",    String,  nullable=False),
    Column("ruta_archivo",      String,  nullable=False),
    Column("fecha_carga",       DateTime, default=datetime.now),
    Column("estado",            String,  default="activo"),
    Column("metodo_extraccion", String,  default="texto_nativo"),
)

t_solicitudes = Table("solicitudes", metadata,
    Column("id",            Integer, primary_key=True, autoincrement=True),
    Column("entidad",       String,  nullable=False),
    Column("tipo_documento",String),
    Column("descripcion",   Text),
    Column("solicitante",   String),
    Column("fecha",         DateTime, default=datetime.now),
    Column("estado",        String,  default="pendiente"),
)

t_reportes = Table("reportes", metadata,
    Column("id",        Integer, primary_key=True, autoincrement=True),
    Column("pregunta",  Text,   nullable=False),
    Column("respuesta", Text,   nullable=False),
    Column("comentario",Text),
    Column("fecha",     DateTime, default=datetime.now),
    Column("estado",    String,  default="pendiente"),
)

t_sesiones = Table("sesiones", metadata,
    Column("id",         String,  primary_key=True),   # UUID
    Column("inicio",     DateTime, default=datetime.now),
    Column("ultimo_uso", DateTime, default=datetime.now),
)

t_historial = Table("historial", metadata,
    Column("id",        Integer, primary_key=True, autoincrement=True),
    Column("sesion_id", String,  ForeignKey("sesiones.id"), nullable=False),
    Column("role",      String,  nullable=False),   # 'user' | 'assistant'
    Column("content",   Text,    nullable=False),
    Column("fecha",     DateTime, default=datetime.now),
)


def init_db():
    """Crea todas las tablas si no existen. Seguro de llamar múltiples veces."""
    metadata.create_all(engine)


# ---------------------------------------------------------------------------
# Helper interno
# ---------------------------------------------------------------------------
def _now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# Documentos
# ---------------------------------------------------------------------------
def insertar_documento(entidad, tipo_documento, nombre_archivo, ruta_archivo,
                       metodo_extraccion="texto_nativo"):
    with engine.begin() as conn:
        result = conn.execute(
            t_documentos.insert().values(
                entidad=entidad,
                tipo_documento=tipo_documento,
                nombre_archivo=nombre_archivo,
                ruta_archivo=ruta_archivo,
                fecha_carga=_now(),
                estado="activo",
                metodo_extraccion=metodo_extraccion,
            )
        )
        return result.inserted_primary_key[0]


def listar_documentos():
    with engine.connect() as conn:
        rows = conn.execute(
            t_documentos.select().order_by(t_documentos.c.fecha_carga.desc())
        ).fetchall()
    return [
        (r.id, r.entidad, r.tipo_documento, r.nombre_archivo,
         str(r.fecha_carga)[:19], r.estado)
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Solicitudes
# ---------------------------------------------------------------------------
def insertar_solicitud(entidad, tipo_documento, descripcion, solicitante):
    with engine.begin() as conn:
        conn.execute(
            t_solicitudes.insert().values(
                entidad=entidad,
                tipo_documento=tipo_documento,
                descripcion=descripcion,
                solicitante=solicitante,
                fecha=_now(),
                estado="pendiente",
            )
        )


def listar_solicitudes(solo_pendientes=False):
    with engine.connect() as conn:
        q = t_solicitudes.select().order_by(t_solicitudes.c.fecha.desc())
        if solo_pendientes:
            q = q.where(t_solicitudes.c.estado == "pendiente")
        rows = conn.execute(q).fetchall()
    return [
        (r.id, r.entidad, r.tipo_documento, r.descripcion,
         r.solicitante, str(r.fecha)[:19], r.estado)
        for r in rows
    ]


def actualizar_estado_solicitud(solicitud_id, nuevo_estado):
    with engine.begin() as conn:
        conn.execute(
            t_solicitudes.update()
            .where(t_solicitudes.c.id == solicitud_id)
            .values(estado=nuevo_estado)
        )


# ---------------------------------------------------------------------------
# Reportes
# ---------------------------------------------------------------------------
def insertar_reporte(pregunta, respuesta, comentario):
    with engine.begin() as conn:
        conn.execute(
            t_reportes.insert().values(
                pregunta=pregunta,
                respuesta=respuesta,
                comentario=comentario,
                fecha=_now(),
                estado="pendiente",
            )
        )


def listar_reportes(solo_pendientes=False):
    with engine.connect() as conn:
        q = t_reportes.select().order_by(t_reportes.c.fecha.desc())
        if solo_pendientes:
            q = q.where(t_reportes.c.estado == "pendiente")
        rows = conn.execute(q).fetchall()
    return [
        (r.id, r.pregunta, r.respuesta, r.comentario,
         str(r.fecha)[:19], r.estado)
        for r in rows
    ]


def actualizar_estado_reporte(reporte_id, nuevo_estado):
    with engine.begin() as conn:
        conn.execute(
            t_reportes.update()
            .where(t_reportes.c.id == reporte_id)
            .values(estado=nuevo_estado)
        )


# ---------------------------------------------------------------------------
# Sesiones
# ---------------------------------------------------------------------------
def crear_sesion() -> str:
    """Genera un UUID, lo registra en la tabla sesiones y lo devuelve."""
    sesion_id = str(uuid.uuid4())
    with engine.begin() as conn:
        conn.execute(
            t_sesiones.insert().values(
                id=sesion_id,
                inicio=_now(),
                ultimo_uso=_now(),
            )
        )
    return sesion_id


def actualizar_ultimo_uso(sesion_id: str):
    with engine.begin() as conn:
        conn.execute(
            t_sesiones.update()
            .where(t_sesiones.c.id == sesion_id)
            .values(ultimo_uso=_now())
        )


# ---------------------------------------------------------------------------
# Historial
# ---------------------------------------------------------------------------
def guardar_mensaje(sesion_id: str, role: str, content: str):
    """Persiste un mensaje (user o assistant) asociado a una sesión."""
    with engine.begin() as conn:
        conn.execute(
            t_historial.insert().values(
                sesion_id=sesion_id,
                role=role,
                content=content,
                fecha=_now(),
            )
        )
    actualizar_ultimo_uso(sesion_id)


def cargar_historial(sesion_id: str) -> list[dict]:
    """
    Devuelve el historial completo de una sesión ordenado cronológicamente,
    en el formato que esperan las APIs: [{"role": "user"|"assistant", "content": "..."}]
    """
    with engine.connect() as conn:
        rows = conn.execute(
            t_historial.select()
            .where(t_historial.c.sesion_id == sesion_id)
            .order_by(t_historial.c.fecha.asc())
        ).fetchall()
    return [{"role": r.role, "content": r.content} for r in rows]
