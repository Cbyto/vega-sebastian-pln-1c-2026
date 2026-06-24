import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
DOCUMENTOS_DIR = DATA_DIR / "documentos"
CHROMA_DIR = DATA_DIR / "chroma"
DB_PATH = DATA_DIR / "normas.db"

COLLECTION_NAME = "normas_operativas"

# ---------------------------------------------------------------------------
# Base de datos
# ---------------------------------------------------------------------------
# SQLite por defecto. Para migrar a PostgreSQL, cambiar en .env:
#   DATABASE_URL=postgresql://user:pass@host:5432/dbname
# El resto del código (db.py) no necesita ningún cambio.
# ---------------------------------------------------------------------------
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{DATA_DIR / 'normas.db'}"
)

# ---------------------------------------------------------------------------
# Proveedor de IA
# ---------------------------------------------------------------------------
# El sistema detecta automáticamente qué proveedor usar según las claves
# disponibles en el .env. Si están las dos, ANTHROPIC tiene prioridad.
#
# Para OpenAI:
#   OPENAI_API_KEY=...
#   EMBEDDING_MODEL=text-embedding-3-small
#   CHAT_MODEL=gpt-4.1-mini
#
# Para Anthropic:
#   ANTHROPIC_API_KEY=...
#   CHAT_MODEL=claude-sonnet-4-20250514
#   (los embeddings siempre se generan con OpenAI, Anthropic no los ofrece)
#
# Forzar un proveedor con: AI_PROVIDER=openai | anthropic
# ---------------------------------------------------------------------------

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")

_forzado = os.getenv("AI_PROVIDER", "").lower()

if _forzado in ("openai", "anthropic"):
    AI_PROVIDER = _forzado
elif ANTHROPIC_API_KEY:
    AI_PROVIDER = "anthropic"
elif OPENAI_API_KEY:
    AI_PROVIDER = "openai"
else:
    raise EnvironmentError(
        "No se encontró ninguna API key. "
        "Definí OPENAI_API_KEY o ANTHROPIC_API_KEY en el archivo .env"
    )

# Embedding: siempre OpenAI (Anthropic no tiene endpoint de embeddings)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Modelo de chat: default según proveedor
_default_chat = (
    "claude-sonnet-4-20250514" if AI_PROVIDER == "anthropic"
    else "gpt-4.1-mini"
)
CHAT_MODEL = os.getenv("CHAT_MODEL", _default_chat)

TOP_K = int(os.getenv("TOP_K", "5"))

for folder in [DATA_DIR, DOCUMENTOS_DIR, CHROMA_DIR]:
    folder.mkdir(parents=True, exist_ok=True)
