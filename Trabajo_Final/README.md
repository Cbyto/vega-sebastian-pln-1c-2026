# Chat Normas Operativas

Chatbot RAG (Retrieval-Augmented Generation) para consultar normas operativas de obras sociales y prepagas argentinas. Permite cargar PDFs, indexarlos semánticamente y hacer preguntas en lenguaje natural obteniendo respuestas fundamentadas en los documentos cargados.

---

## Arquitectura general

```
Usuario
  │
  ▼
app.py  (Streamlit — interfaz web)
  │
  ├── src/comandos.py     → comandos especiales (--help, --buscar, etc.)
  │
  └── src/rag.py          → pipeline principal de pregunta → respuesta
        │
        ├── src/vector_store.py  → búsqueda semántica en ChromaDB
        │     └── OpenAI Embeddings API
        │
        └── OpenAI o Anthropic    → genera la respuesta final

  Panel admin (sidebar):
    ├── src/pdf_reader.py   → extrae texto de PDFs (PyMuPDF)
    ├── src/chunker.py      → divide el texto en fragmentos
    ├── src/vector_store.py → genera embeddings y guarda en ChromaDB
    └── src/db.py           → registra metadatos en SQLite
```

---

## Stack tecnológico

| Componente       | Tecnología                          |
|-----------------|--------------------------------------|
| Interfaz web    | Streamlit                            |
| Embeddings      | OpenAI `text-embedding-3-small`      |
| Vector store    | ChromaDB (local, persistente en disco) |
| Base de datos   | SQLite (metadatos de documentos)     |
| Lectura de PDF  | PyMuPDF (`fitz`)                     |
| Modelo de chat  | OpenAI `gpt-4.1-mini` o Anthropic `claude-sonnet-4-20250514` |

---

## Estructura del proyecto

```
proyecto/
│
├── app.py                  # Aplicación principal Streamlit
├── config.py               # Configuración central (rutas, modelos, proveedor de IA)
├── .env                    # Claves de API y configuración (no commitear)
├── requirements.txt        # Dependencias
├── .gitignore              # Archivos ignorados por Git
│
├── claude_vision.py        # Script batch OCR via Claude Vision (correr desde CLI)
│
├── src/
│   ├── pdf_reader.py       # Extracción de texto de PDFs (nativo + detección de texto inválido)
│   ├── chunker.py          # División de texto en fragmentos
│   ├── vector_store.py     # Embeddings + ChromaDB
│   ├── rag.py              # Pipeline RAG (pregunta → respuesta)
│   ├── db.py               # Registro de documentos, solicitudes y reportes en SQLite
│   └── comandos.py         # Comandos especiales del chat
│
└── data/                   # Generado automáticamente
    ├── documentos/         # PDFs cargados
    ├── chroma/             # Base vectorial persistente
    └── normas.db           # Base de datos SQLite
```

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone <url-del-repo>
cd proyecto

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno
cp .env.example .env
# Editar .env con tus claves
```

---

## Configuración (.env)

```env
# --- Claves de API ---

# OpenAI (obligatorio para embeddings, siempre requerido)
OPENAI_API_KEY=sk-...

# Anthropic (opcional — si está presente, se usa para el chat)
ANTHROPIC_API_KEY=sk-ant-...

# --- Modelos ---
# Si usás OpenAI para el chat:
CHAT_MODEL=gpt-4.1-mini

# Si usás Anthropic para el chat:
# CHAT_MODEL=claude-sonnet-4-20250514

EMBEDDING_MODEL=text-embedding-3-small

# --- Parámetros RAG ---
TOP_K=5          # Cantidad de fragmentos que se recuperan por pregunta

# --- Forzar proveedor (opcional) ---
# AI_PROVIDER=openai
# AI_PROVIDER=anthropic

# --- Admin ---
ADMIN_USER=admin
ADMIN_PASS=tu_password_seguro
```

### Lógica de selección de proveedor

El sistema detecta automáticamente qué API usar:

1. Si `AI_PROVIDER` está definido en `.env` → usa ese.
2. Si no, y existe `ANTHROPIC_API_KEY` → usa Anthropic.
3. Si no, y existe `OPENAI_API_KEY` → usa OpenAI.
4. Si no hay ninguna clave → error al iniciar.

> **Nota:** Los embeddings **siempre se generan con OpenAI**, ya que Anthropic no ofrece endpoint de embeddings. Por eso `OPENAI_API_KEY` es siempre obligatorio.

---

## Uso

### Iniciar la aplicación

```bash
streamlit run app.py
```

### Cargar un documento (panel admin)

1. Hacer clic en **Administrador** en el sidebar.
2. Ingresar usuario y contraseña.
3. Completar **Obra social / prepaga** (ej: `OSDE`) y **Tipo de documento** (ej: `Norma operativa`).
4. Subir el PDF y hacer clic en **Procesar PDF**.

El sistema extrae el texto, lo divide en fragmentos, genera embeddings y los guarda en ChromaDB. Si el documento ya fue cargado anteriormente, los fragmentos duplicados se saltean automáticamente.

### Hacer preguntas

Escribir directamente en el chat en lenguaje natural:

```
¿Qué documentación necesito para autorizar una resonancia magnética en OSDE?
¿Cuál es el plazo para responder una solicitud de autorización?
```

### Comandos especiales

| Comando | Descripción |
|---------|-------------|
| `--help` | Lista todos los comandos disponibles |
| `--info` | Muestra todos los documentos cargados |
| `--docs OSDE` | Filtra documentos por obra social |
| `--buscar autorización resonancia` | Búsqueda directa de fragmentos sin generar respuesta |

---

## Módulos principales

### `pdf_reader.py`

Extrae texto página por página usando PyMuPDF. Devuelve una lista de dicts con número de página y texto.

> Limitación actual: solo procesa PDFs con texto nativo. PDFs escaneados (imágenes) devuelven texto vacío.

### `chunker.py`

Divide el texto respetando la estructura del documento:

1. Separa por párrafos (líneas en blanco).
2. Dentro de cada párrafo largo, separa por oraciones.
3. Respeta abreviaturas médico-administrativas para no cortar donde no corresponde: `Dr.`, `Dra.`, `Sr.`, `Sra.`, `Art.`, `O.S.`, etc.
4. Aplica overlap entre chunks para no perder contexto en los bordes.

Parámetros: `max_chars=1200`, `overlap=150`.

### `vector_store.py`

- Genera embeddings con `text-embedding-3-small` de OpenAI.
- Persiste los vectores en ChromaDB (carpeta `data/chroma/`).
- Antes de insertar, verifica si el chunk ya existe (evita duplicados).
- La búsqueda devuelve los `TOP_K` fragmentos más similares con su metadata.

### `rag.py`

Pipeline principal:

1. Recibe la pregunta del usuario.
2. Busca los fragmentos más relevantes en ChromaDB.
3. Arma el contexto con metadata (entidad, archivo, página).
4. Llama al modelo de chat (OpenAI o Anthropic) con el contexto y la pregunta.
5. Devuelve la respuesta y las fuentes usadas.

El prompt del sistema instruye al modelo a responder **solo con el contexto provisto** y a citar fuentes al final.

### `db.py`

SQLite para registrar metadatos de los documentos cargados: entidad, tipo, nombre de archivo, ruta, fecha de carga y estado.

### `comandos.py`

Interpreta mensajes que empiezan con `--` y ejecuta funciones específicas sin pasar por el pipeline RAG.

---

## Consideraciones de costos

Cada pregunta genera:
- **1 embedding** de la pregunta (muy barato, fracción de centavo).
- **1 llamada al modelo de chat** con `TOP_K` fragmentos como contexto (~1.000–3.000 tokens de entrada).

La carga de PDFs genera embeddings por cada chunk. Un PDF de 50 páginas genera aproximadamente 100–200 chunks.

Para reducir costos:
- Bajar `TOP_K` a 3 en `.env`.
- Usar `gpt-4.1-mini` o `claude-haiku` en lugar de modelos más grandes.

---

## Roadmap

- [x] Soporte OCR para PDFs con encoding roto (Claude Vision batch)
- [x] Solicitudes de normas por parte de usuarios
- [x] Reporte de errores inline con contexto de pregunta/respuesta
- [ ] Soporte de imágenes sueltas (fotos de recetas, autorizaciones)
- [ ] Historial de conversación con contexto multi-turno
- [ ] Panel de administración para eliminar/reemplazar documentos
- [ ] Tests unitarios para chunker y pipeline RAG
- [ ] Logging de consultas para auditoría

---

## Solicitudes y reportes de errores

### Solicitar una norma (usuarios)

En el sidebar hay un panel colapsado **📬 Solicitar una norma**. El usuario completa la obra social, el tipo de documento y un comentario opcional. La solicitud queda registrada en SQLite con estado `pendiente`.

El administrador ve las solicitudes pendientes en su panel y puede marcarlas como **Resuelta** o **Descartada**.

### Reportar un error en una respuesta

Debajo de cada respuesta del chat aparecen los botones **👍** y **👎 Reportar error**. Al reportar, el usuario puede agregar un comentario opcional. El sistema guarda automáticamente la pregunta y la respuesta involucrada para que el admin tenga el contexto completo.

---

## PDFs con texto no legible — OCR via Claude Vision

Algunos PDFs (generados con Adobe Illustrator, InDesign u otras herramientas gráficas) usan fuentes con encoding personalizado que PyMuPDF no puede decodificar. En esos casos el texto extraído es ilegible.

### Detección automática

`pdf_reader.py` evalúa si el texto extraído es legible analizando la proporción de caracteres no imprimibles. Si el PDF no supera el umbral, el documento queda registrado en SQLite con estado `pendiente_ocr` y `app.py` muestra un aviso al administrador.

### Procesamiento batch con Claude Vision

El módulo `claude_vision.py` procesa todos los documentos pendientes enviando cada página como imagen a Claude Vision, que devuelve el texto estructurado.

**Procesar todos los pendientes:**
```bash
python claude_vision.py
```

**Procesar un documento específico por ID:**
```bash
python claude_vision.py --doc-id 42
```

**Ver qué se procesaría sin hacer nada (dry-run):**
```bash
python claude_vision.py --dry-run
```

### Flujo completo de ingesta

```
Admin sube PDF
      ↓
pdf_reader.py intenta extracción nativa (PyMuPDF)
      ↓
¿Texto válido? ──SI──→ chunker → vector_store → doc estado: activo
      ↓ NO
doc estado: pendiente_ocr  (aviso al admin en la UI)
      ↓
(cuando el admin quiera, fuera del horario de uso)
python claude_vision.py
      ↓
Por cada página → rasterizar a JPEG → Claude Vision → texto
      ↓
chunker → vector_store → doc estado: activo | metodo: vision_claude
```

### Estados de un documento en SQLite

| Estado | Significado |
|--------|-------------|
| `activo` | Indexado correctamente, disponible para búsquedas |
| `pendiente_ocr` | Texto nativo no legible, esperando procesamiento Vision |
| `error_ocr` | Falló el procesamiento Vision (ver logs) |

### Consideraciones de costo (Claude Vision)

Cada página procesada con Vision consume aproximadamente 1.500–2.500 tokens de entrada (imagen) + ~500 tokens de salida. Para un PDF de 10 páginas el costo es de aproximadamente $0.05–0.10 USD. La ingesta es un proceso que se realiza **una sola vez por documento**.

---

## Nota sobre migración de base de datos

Si ya tenés la base de datos creada con una versión anterior (sin las columnas `metodo_extraccion`, `solicitudes` o `reportes`), ejecutá esto una sola vez:

```sql
-- Agregar columna a tabla existente
ALTER TABLE documentos ADD COLUMN metodo_extraccion TEXT DEFAULT 'texto_nativo';

-- Las tablas nuevas se crean automáticamente con init_db() al iniciar la app
```
