# RAG Pipeline: PDF Ingestion & Semantic Retrieval

A modular **Retrieval-Augmented Generation (RAG)** pipeline built step by step in a Jupyter notebook. It loads PDFs, splits them into chunks, converts the chunks into vector embeddings, stores them in a persistent **ChromaDB** vector store, and retrieves the most relevant chunks for a natural-language query.

This repository covers the **ingestion and retrieval** half of RAG, which is the foundation that any LLM answer-generation step sits on top of.

---

## Table of Contents

- [Pipeline Overview](#pipeline-overview)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Components](#components)
- [Configuration](#configuration)
- [Example](#example)
- [Design Notes](#design-notes)
- [Limitations & Next Steps](#limitations--next-steps)

---

## Pipeline Overview

```
PDFs  ->  Documents  ->  Chunks  ->  Embeddings  ->  Vector Store  ->  Retriever
(data/pdfs)  (PyPDFLoader)  (RecursiveCharacterTextSplitter)  (all-MiniLM-L6-v2)  (ChromaDB)  (top-k search)
```

| Stage | What happens | Why it matters |
|-------|--------------|----------------|
| **1. Load** | Every `.pdf` in `data/pdfs` is read page by page into LangChain `Document` objects (text + metadata such as source and page number). | A uniform `Document` format lets the rest of the pipeline stay loader-agnostic. |
| **2. Chunk** | Pages are split into ~500-character pieces with a 50-character overlap. | Embedding models and LLM context windows work best on small, focused passages; overlap prevents a sentence from being cut off from its context. |
| **3. Embed** | Each chunk becomes a dense vector using `all-MiniLM-L6-v2` (384 dimensions). | Semantically similar text ends up close together in vector space, so search works by meaning rather than keywords. |
| **4. Store** | Vectors, raw text, and metadata are saved in a persistent ChromaDB collection. | Persistence means you embed once and reuse the index across sessions. |
| **5. Retrieve** | A query is embedded with the same model and the nearest chunks are returned with scores and ranks. | Using the *same* embedding model for documents and queries is what makes the comparison valid. |

---

## Tech Stack

- **Python 3.9+**
- **LangChain** (`langchain`, `langchain-core`, `langchain-community`, `langchain-text-splitters`): document abstraction, PDF loading, text splitting
- **pypdf**: PDF parsing used by `PyPDFLoader`
- **sentence-transformers**: embedding model (`all-MiniLM-L6-v2`)
- **ChromaDB**: persistent vector database
- **scikit-learn**: imported for similarity utilities
- **Jupyter**: interactive development

---

## Project Structure

```
.
├── RAG_pipeline.ipynb      # Full pipeline: load -> chunk -> embed -> store -> retrieve
├── data/
│   ├── pdfs/               # Put your source PDF files here
│   └── vector_store/       # Auto-created ChromaDB persistence directory
└── README.md
```

> `data/pdfs/` must exist and contain at least one PDF before you run the loader. `data/vector_store/` is created automatically.

---

## Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd <your-repo-folder>

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install langchain langchain-core langchain-community langchain-text-splitters \
            pypdf pymupdf sentence-transformers chromadb scikit-learn jupyter
```

The first run downloads the `all-MiniLM-L6-v2` model (roughly 90 MB), so an internet connection is needed once.

---

## Usage

1. Place your PDFs in `data/pdfs/`.
2. Launch the notebook:
   ```bash
   jupyter notebook RAG_pipeline.ipynb
   ```
3. Run the cells top to bottom. The ingestion flow is:

```python
# Load PDFs into Document objects
all_pdf = pdf_loader()

# Split into chunks
chunks = split_doc(all_pdf)

# Embed the chunk texts
emb_mng = EmbeddingManager()
texts = [doc.page_content for doc in chunks]
embeddings = emb_mng.generate_embedding(texts)

# Store in ChromaDB
vector_store = VectorStoreManager()
vector_store.add_documents(chunks, embeddings)

# Retrieve
rag_retriever = RAGRetriever(emb_mng, vector_store)
results = rag_retriever.retrieve("What is encoder decoder", top_k=5)
```

Run the embed-and-store step **once** per set of documents. Running it again on the same PDFs adds duplicate entries to the collection.

---

## Components

### `pdf_loader()`
Walks `data/pdfs`, loads every file ending in `.pdf` (case-insensitive) with `PyPDFLoader`, and returns one flat list of page-level `Document` objects. It prints the number of PDFs and total pages found.

### `split_doc(documents, chunk_size=500, chunk_overlap=50)`
Wraps `RecursiveCharacterTextSplitter`. It tries to split on paragraph breaks first, then lines, then words, so chunks stay as semantically intact as possible. Metadata (source, page) is carried over to every chunk.

### `EmbeddingManager`
Loads a `SentenceTransformer` model once at construction and exposes `generate_embedding(text)`, which accepts a string or a list of strings and returns a NumPy array. The default model is `all-MiniLM-L6-v2`, which is small and fast and produces 384-dimensional vectors.

### `VectorStoreManager`
Manages a persistent ChromaDB collection.

- Creates the persistence directory if it does not exist.
- Uses `get_or_create_collection`, so reopening the store reuses existing data.
- `add_documents(documents, embeddings)` validates that both lists have equal length, assigns each chunk a unique `doc_<uuid>` ID, and stores the vector, text, and metadata. It also records `doc_index` and `content_length` in each chunk's metadata.

### `RAGRetriever`
Takes an `EmbeddingManager` and a `VectorStoreManager`.

`retrieve(query, top_k=5, score_threshold=0.0)`:
1. Embeds the query.
2. Runs a nearest-neighbour search in the collection.
3. Converts each distance to `similarity_score = 1 - distance`.
4. Filters out results below `score_threshold`.
5. Returns a list of dicts with `id`, `document`, `metadata`, `distance`, `similarity_score`, and `rank`.

---

## Configuration

| Parameter | Where | Default | Effect |
|-----------|-------|---------|--------|
| `chunk_size` | `split_doc` | `500` | Max characters per chunk. Larger gives more context per chunk but less precise matches. |
| `chunk_overlap` | `split_doc` | `50` | Characters shared between neighbouring chunks. |
| `model_name` | `EmbeddingManager` | `all-MiniLM-L6-v2` | Any sentence-transformers model. If you change it, rebuild the index, since vectors from different models are not comparable. |
| `persist_directory` | `VectorStoreManager` | `data/vector_store` | Where ChromaDB writes its files. |
| `collection_name` | `VectorStoreManager` | `pdf_documents` | Logical grouping of vectors, similar to a table in SQL. |
| `top_k` | `retrieve` | `5` | Number of chunks returned. |
| `score_threshold` | `retrieve` | `0.0` | Minimum similarity to keep a result. |

---

## Example

```python
results = rag_retriever.retrieve("What is encoder decoder", top_k=3)

for r in results:
    print(f"Rank {r['rank']}  |  score = {r['similarity_score']:.3f}")
    print(f"Source: {r['metadata'].get('source')}, page {r['metadata'].get('page')}")
    print(r["document"][:200], "...\n")
```

---

## Design Notes

- **Same model for indexing and querying.** Query and chunk vectors must come from one embedding model, otherwise distances are meaningless.
- **Distance metric.** ChromaDB uses squared L2 distance by default. Because `all-MiniLM-L6-v2` outputs are normalised, L2 and cosine rankings agree, so result *ordering* is correct. However, `1 - distance` is not a true cosine similarity under the default metric. For exact cosine scores, create the collection with `metadata={"hnsw:space": "cosine"}`, in which case `1 - distance` is the cosine similarity.
- **Persistent client.** `chromadb.PersistentClient` writes to disk, so the index survives kernel restarts.
- **Chunk size trade-off.** Small chunks give sharper retrieval; large chunks keep more surrounding context. 500/50 is a reasonable starting point to tune against your documents.

---

## Limitations & Next Steps

- Only **PDF** ingestion is wired into the pipeline (a plain-text loader is imported but not used yet).
- There is **no generation step**: results are retrieved chunks, not synthesised answers. The natural next step is passing the top-k chunks plus the question to an LLM.
- No **deduplication**: re-ingesting the same file creates duplicate chunks.
- Scanned (image-only) PDFs contain no extractable text and need OCR first.
- Ideas for extension: hybrid (keyword + vector) search, re-ranking, metadata filtering, semantic caching, and an evaluation set to measure retrieval quality.

---