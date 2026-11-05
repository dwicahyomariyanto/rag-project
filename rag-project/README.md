# MiniRAG — RAG REST API Siap Uji di VS Code

Project **Retrieval-Augmented Generation (RAG)** production-style dalam satu folder: unggah PDF → dokumen di-chunk & di-embedding → disimpan di **ChromaDB** → pertanyaan dijawab LLM dengan **sitasi halaman**, lengkap dengan **hybrid search**, **reranking cross-encoder**, **streaming SSE**, **halaman demo web**, **test suite offline**, dan **evaluasi kualitas RAGAS**.

> Diujikan end-to-end pada Python 3.12 (semua 34 test pytest hijau + smoke test server live: ingest → chat → streaming).

---

## Daftar Isi

1. [Fitur](#fitur)
2. [Arsitektur](#arsitektur)
3. [Struktur Folder](#struktur-folder)
4. [Prasyarat](#prasyarat)
5. [Instalasi](#instalasi)
6. [Konfigurasi (.env)](#konfigurasi-env)
7. [Cara Menjalankan](#cara-menjalankan)
8. [Cara Menguji (5 jalur)](#cara-menguji)
9. [Referensi Endpoint](#referensi-endpoint)
10. [Streaming SSE](#streaming-sse)
11. [Evaluasi Kualitas dengan RAGAS](#evaluasi-kualitas-dengan-ragas)
12. [Pemecahan Masalah](#pemecahan-masalah)
13. [Ide Pengembangan Lanjutan](#ide-pengembangan-lanjutan)

---

## Fitur

| Fitur | Implementasi |
|---|---|
| Ingest PDF | `pymupdf` (PyMuPDF), ekstraksi per halaman + peringatan halaman scan |
| Chunking | packing antar-halaman, hard-split per kalimat dengan **overlap**, metadata halaman tetap akurat |
| Vector store | **ChromaDB** persisten (cosine space) di `storage/chroma/` |
| Keyword search | **BM25** (`rank-bm25`) dengan tokenizer + stopwords Indonesia/Inggris |
| Hybrid search | **Reciprocal Rank Fusion (RRF, k=60)** atas kedua ranking |
| Reranking | **Cross-encoder** `ms-marco-MiniLM-L-6-v2` (lazy-load, fallback otomatis jika model gagal dimuat) |
| Sitasi | Jawaban menyertakan `[1]`, `[2]` → dipetakan ke nama file + halaman PDF |
| Streaming | **Server-Sent Events** di `/api/chat/stream` (event `sources`, `token`, `done`, `error`) |
| Multi-provider | Endpoint **OpenAI-compatible apa pun** (OpenAI, Groq, DeepSeek, Ollama, vLLM…) via `.env` |
| Mode offline | `LLM_BACKEND=fake` + `EMBEDDING_BACKEND=hash` → seluruh pipeline dapat diuji **tanpa API key** |
| Evaluasi | **RAGAS**: faithfulness, response relevancy, context precision & recall |
| Observability | Statistik retrieval per request (kandidat, latensi ms), endpoint debug `/api/retrieve` |
| Docs otomatis | Swagger UI di `/docs` (FastAPI) |

## Arsitektur

```
                         ┌──────────────────────────────────────────────────┐
   PDF ──/ingest────────►│  loader (pymupdf) → chunker → embedder → ChromaDB│
                         │                        │                          │
                         │                 BM25Index (rebuild otomatis)      │
                         └──────────────────────────────────────────────────┘
                                            │
 Pertanyaan ──/chat──────┐                  ▼
                         │   ┌────────────────────────────────┐
                         ├──►│ retrieve:                      │
                         │   │  semantic (ChromaDB, top-20)   │
                         │   │  + BM25 (top-20)               │
                         │   │  → RRF fusion                  │
                         │   │  → cross-encoder rerank        │
                         │   │  → top-5 passage konteks       │
                         │   └──────────────┬─────────────────┘
                         │                  ▼
                         │   ┌────────────────────────────────┐
                         └──►│ generate: prompt + konteks [1..5] │
                             │ → LLM (OpenAI-compatible)     │
                             │ → jawaban dengan sitasi [n]   │
                             └────────────────────────────────┘
```

## Struktur Folder

```
rag-project/
├── app/
│   ├── main.py               # FastAPI app factory + entrypoint uvicorn
│   ├── config.py             # Settings (.env) tervalidasi pydantic-settings
│   ├── service.py            # Service layer: rakit komponen + logika ingest
│   ├── api/
│   │   ├── deps.py           # Dependency injection service
│   │   ├── routes_chat.py    # /api/chat, /api/chat/stream, /api/retrieve
│   │   ├── routes_documents.py # /api/documents (ingest, list, delete)
│   │   └── routes_health.py  # /api/health
│   ├── core/
│   │   ├── llm.py            # OpenAIChatLLM + FakeLLM (offline)
│   │   └── embeddings.py     # OpenAIEmbedder + HashEmbedder (offline)
│   ├── rag/
│   │   ├── loader.py         # Ekstraksi teks PDF per halaman
│   │   ├── chunker.py        # Chunking + overlap + pemetaan halaman
│   │   ├── vectorstore.py    # Wrapper ChromaDB
│   │   ├── bm25.py           # Index BM25 + stopwords
│   │   ├── hybrid.py         # Reciprocal Rank Fusion
│   │   ├── reranker.py       # CrossEncoder / Lexical / Null
│   │   ├── prompts.py        # System prompt + template user
│   │   ├── pipeline.py       # Orkestrasi retrieve → generate → sitasi
│   │   └── models.py         # Dataclass Chunk / RetrievedChunk
│   └── schemas/models.py     # Model request/response Pydantic
├── scripts/
│   ├── ingest.py             # CLI ingest file/folder PDF
│   ├── query.py              # CLI tanya jawab cepat (tanpa server)
│   └── evaluate.py           # Runner evaluasi RAGAS
├── tests/                    # 34 tes offline (pytest) + 1 tes RAGAS
├── static/index.html         # Halaman demo chat (dilayani di "/")
├── examples/stream_client.py # Contoh klien Python untuk SSE
├── data/
│   ├── sample/panduan_karyawan_pt_nusantara.pdf   # dokumen contoh
│   └── eval/eval_dataset.jsonl                    # dataset evaluasi (10 Q&A)
├── .env.example              # Template konfigurasi (salin jadi .env)
├── requirements.txt          # Dependensi (versi teruji)
├── pyproject.toml            # Konfigurasi pytest (marker ragas)
└── .vscode/launch.json       # Profil F5: server, pytest, ingest
```

## Prasyarat

- **Python 3.10 – 3.12** (diujikan pada 3.12). Cek: `python --version`
- **VS Code** + extension *Python* (ms-python.python)
- Koneksi internet saat `pip install` dan (sekali pertama) saat mengunduh model cross-encoder (~80 MB)
- API key salah satu provider LLM — **atau tidak sama sekali** untuk mode offline (lihat bagian Konfigurasi)

## Instalasi

Buka folder project di VS Code → **Terminal → New Terminal**, lalu:

```bash
# 1) Buat & aktifkan virtual environment
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 2) Install dependensi (Linux tanpa GPU, opsional hemat disk:)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 3) Salin konfigurasi
cp .env.example .env        # Windows: copy .env.example .env
```

> **VS Code**: setelah `.venv` dibuat, tekan `Ctrl+Shift+P` → *Python: Select Interpreter* → pilih `.venv`. Profil debug **F5** sudah disiapkan di `.vscode/launch.json`.

## Konfigurasi (.env)

Semua variabel (lihat penjelasan lengkap di `.env.example`):

| Variabel | Default | Keterangan |
|---|---|---|
| `LLM_BACKEND` | `openai` | `openai` (endpoint OpenAI-compatible) atau `fake` (offline) |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | Ganti untuk Groq/DeepSeek/Ollama (contoh di `.env.example`) |
| `LLM_API_KEY` | — | API key provider |
| `LLM_MODEL` | `gpt-4o-mini` | Nama model chat |
| `LLM_TEMPERATURE` / `LLM_MAX_TOKENS` | `0.2` / `1024` | Parameter generasi |
| `EMBEDDING_BACKEND` | `api` | `api` atau `hash` (offline testing) |
| `EMBEDDING_BASE_URL` / `EMBEDDING_API_KEY` / `EMBEDDING_MODEL` | OpenAI | Endpoint `/v1/embeddings` (Ollama: `nomic-embed-text`) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` / `MIN_CHUNK_CHARS` | `900`/`150`/`80` | Karakter per chunk, overlap, ambang merge |
| `RETRIEVAL_K_SEMANTIC` / `RETRIEVAL_K_BM25` | `20`/`20` | Jumlah kandidat per kanal |
| `RRF_K` | `60` | Konstanta Reciprocal Rank Fusion |
| `CONTEXT_TOP_K` | `5` | Passage final yang dikirim ke LLM |
| `RERANK_BACKEND` | `cross_encoder` | `cross_encoder` / `lexical` / `none` |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Model reranker |
| `STORAGE_DIR` | `storage` | Lokasi chroma DB, upload, hasil evaluasi |

**Mulai tanpa API key (disarankan untuk uji pertama):**

```env
LLM_BACKEND=fake
EMBEDDING_BACKEND=hash
RERANK_BACKEND=lexical
```

Dengan 3 baris ini server, pipeline, sitasi, streaming, dan seluruh tes berjalan penuh di laptop Anda — tanpa internet dan tanpa biaya. Setelah siap, kembalikan ke `openai`/`api`/`cross_encoder` dan isi API key.

## Cara Menjalankan

```bash
# dari root project, dengan .venv aktif
uvicorn app.main:app --reload --port 8000
```

atau di VS Code: tekan **F5** → pilih **"MiniRAG API (uvicorn)"**.

Lalu buka:

| URL | Fungsi |
|---|---|
| `http://localhost:8000/` | **Halaman demo chat** (unggah PDF + tanya jawab streaming) |
| `http://localhost:8000/docs` | **Swagger UI** — uji semua endpoint interaktif |

## Cara Menguji

### 1. Test suite otomatis (tanpa API key — harus hijau 34/34)

```bash
pytest -v
```

Mencakup: chunker, RRF/BM25, reranker, embedder, vectorstore (persistensi & delete), pipeline + sitasi, streaming, dan seluruh alur API (ingest → list → chat → stream → delete).

### 2. Halaman demo (paling cepat terlihat)

Buka `http://localhost:8000/` → klik panel **Dokumen** → unggah `data/sample/panduan_karyawan_pt_nusantara.pdf` → tanya: *"Berapa hari cuti tahunan?"* → jawaban streaming muncul dengan chip sitasi `[1] panduan… · hal. 3` (arahkan kursor ke chip untuk melihat potongan sumber).

### 3. Swagger UI

Buka `http://localhost:8000/docs` → coba `POST /api/chat` dengan body:

```json
{ "question": "Berapa plafon reimbursement medis per tahun?" }
```

### 4. curl

```bash
# ingest
curl -X POST http://localhost:8000/api/documents/ingest \
  -F "file=@data/sample/panduan_karyawan_pt_nusantara.pdf"

# chat
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Kapan gaji dibayarkan?"}'

# streaming
curl -N -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "Berapa hari cuti tahunan?"}'
```

### 5. CLI tanpa server

```bash
python scripts/ingest.py data/sample/panduan_karyawan_pt_nusantara.pdf
python scripts/query.py "Bagaimana prosedur pengajuan cuti tahunan?"
```

## Referensi Endpoint

| Method & Path | Fungsi |
|---|---|
| `GET /api/health` | Status app, backend aktif, jumlah chunk & dokumen |
| `POST /api/documents/ingest` | Upload PDF (multipart `file`) → ekstrak, chunk, embed, index |
| `GET /api/documents` | Daftar dokumen (id, nama, jumlah chunk, rentang halaman) |
| `DELETE /api/documents/{source_id}` | Hapus dokumen + semua chunk-nya |
| `POST /api/chat` | Tanya jawab lengkap: `answer`, `citations[]`, `stats` |
| `POST /api/chat/stream` | Versi **SSE streaming** dari `/api/chat` |
| `POST /api/retrieve` | Debug retrieval: passage terpilih + skor tiap tahap, tanpa LLM |

Contoh respons `/api/chat` (dipangkas):

```json
{
  "answer": "Cuti tahunan karyawan tetap adalah 12 hari kerja per tahun [1] ...",
  "citations": [
    { "id": 1, "source_name": "panduan_karyawan_pt_nusantara.pdf",
      "page_start": 3, "page_end": 3, "snippet": "3. Cuti Tahunan Karyawan tetap..." }
  ],
  "stats": { "semantic_candidates": 5, "bm25_candidates": 5,
             "fused_candidates": 5, "context_used": 3,
             "retrieval_ms": 2.7, "generation_ms": 810.4, "total_ms": 813.1 }
}
```

## Streaming SSE

`/api/chat/stream` mengirim event berurutan:

```
event: sources        <- seluruh passage konteks + sitasi (sekali, di awal)
data: {"citations": [...]}

event: token          <- potongan jawaban, berulang
data: {"delta": "Cuti "}

event: done           <- statistik akhir
data: {"stats": {...}}
```

Konsumsi siap pakai: `examples/stream_client.py` (Python + httpx) atau tombol *streaming* di halaman demo. Catatan: `EventSource` browser hanya mendukung GET, sehingga klien memakai `fetch()` + pembacaan `ReadableStream` (sudah diimplementasikan di `static/index.html`).

## Evaluasi Kualitas dengan RAGAS

1. Isi `.env` dengan **API key asli** (judge RAGAS memakai LLM sungguhan).
2. Dataset contoh tersedia di `data/eval/eval_dataset.jsonl` (10 pasang pertanyaan–jawaban referensi atas PDF contoh). Format baris:

   ```json
   {"user_input": "pertanyaan", "reference": "jawaban ideal"}
   ```

3. Jalankan:

   ```bash
   python scripts/evaluate.py                # seluruh dataset
   python scripts/evaluate.py --limit 3      # 3 pertanyaan pertama
   ```

4. Hasil: skor **faithfulness**, **response relevancy**, **context precision**, **context recall** + laporan di `storage/eval_results/ragas_report.csv`.

Untuk uji bergaya CI: `pytest -m ragas` (skip otomatis bila `LLM_API_KEY` belum diisi).

| Metrik | Arti |
|---|---|
| Faithfulness | Seberapa besar jawaban didukung konteks (anti-halusinasi) |
| Response relevancy | Seberapa relevan jawaban terhadap pertanyaan |
| Context precision | Apakah passage relevan berada di peringkat atas |
| Context recall | Apakah konteks menutup seluruh isi jawaban referensi |

## Pemecahan Masalah

| Gejala | Solusi |
|---|---|
| `EMBEDDING_API_KEY kosong` / `LLM_API_KEY kosong` saat start | Isi `.env`, atau gunakan mode offline (`fake`/`hash`) |
| Jawaban selalu "tidak ditemukan di dokumen" | Pastikan PDF sudah di-ingest (`GET /api/documents`), dan `EMBEDDING_BACKEND` saat **query sama** dengan saat **ingest** (ganti backend embedding membatalkan kompatibilitas vektor lama — hapus folder `storage/chroma` lalu ingest ulang) |
| Cross-encoder gagal dimuat / unduh model terhenti | Cek koneksi ke huggingface.co; atau set `RERANK_BACKEND=lexical` (fallback otomatis juga aktif) |
| `ModuleNotFoundError` saat `pytest` | Pastikan menjalankan dari root project dengan interpreter `.venv` (Ctrl+Shift+P → Python: Select Interpreter) |
| Error konflik versi langchain saat instalasi | Gunakan `requirements.txt` apa adanya — versi langchain* di-pin karena kompatibilitas ragas |
| PDF hasil scan tidak menghasilkan chunk | Ekstraktor hanya membaca teks; PDF image-only butuh OCR (belum termasuk) |
| Port 8000 terpakai | `uvicorn app.main:app --port 8001` |
| ChromaDB terkunci setelah crash | Hapus file `storage/chroma/chroma.sqlite3-wal` (jika ada), atau reset folder `storage/chroma` |

## Ide Pengembangan Lanjutan

- **Riwayat percakapan multi-turn** (query condensing) di `/api/chat`
- **OCR** (Tesseract) untuk PDF scan
- **Dockerfile + docker-compose** untuk deployment
- **Autentikasi API key** pada header (mis. `X-API-Key`) untuk produksi
- **Dokumen DOCX/TXT/URL** — cukup tambahkan loader baru di `app/rag/loader.py`
- **pgvector/Qdrant** — ganti `vectorstore.py` (interface sudah terisolasi)
