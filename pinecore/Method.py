import os
import time
import pickle
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

import faiss
import numpy as np

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# =====================================================
# CONFIG
# =====================================================
RESUME_FOLDER = r"Education"
CACHE_FILE = "resume_vectors.pkl"

BATCH_SIZE = 64
CHUNK_SIZE = 250
RUNS = 3


# =====================================================
# 1. LOAD MODEL
# =====================================================
print("Loading model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

_ = model.encode("warmup")

print("Model Ready ✅")

DIM = model.get_sentence_embedding_dimension()


# =====================================================
# 2. PDF EXTRACTION
# =====================================================
def extract_pdf(path):

    reader = PdfReader(path)

    text = ""

    for p in reader.pages:

        t = p.extract_text()

        if t:
            text += t + " "

    return text.strip()


print("\nExtracting PDFs...")


pdf_files = [
    os.path.join(RESUME_FOLDER, f)
    for f in os.listdir(RESUME_FOLDER)
    if f.endswith(".pdf")
]


texts = []

with ThreadPoolExecutor(max_workers=8) as exe:

    for txt in tqdm(exe.map(extract_pdf, pdf_files),
                    total=len(pdf_files)):

        if txt:
            texts.append(txt)


# =====================================================
# 3. CHUNKING
# =====================================================
def chunk_text(text, size=CHUNK_SIZE):

    words = text.split()

    return [
        " ".join(words[i:i+size])
        for i in range(0, len(words), size)
    ]


chunks = []

for t in texts:

    chunks.extend(chunk_text(t))


print("Total chunks:", len(chunks))


# =====================================================
# 4. EMBEDDINGS (CACHE FIXED)
# =====================================================
if os.path.exists(CACHE_FILE):

    print("\nLoading cache...")

    with open(CACHE_FILE, "rb") as f:
        data = pickle.load(f)

    # ✅ Handle old and new cache formats
    if isinstance(data, dict) and "embeddings" in data:

        embeddings = np.array(
            data["embeddings"],
            dtype="float32"
        )

    elif isinstance(data, list):

        embeddings = np.array(
            [v["values"] for v in data],
            dtype="float32"
        )

    else:
        raise ValueError("Unknown cache format!")

else:

    print("\nGenerating embeddings...")

    embeddings = model.encode(
        chunks,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True
    ).astype("float32")

    # Save in new format
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(
            {"embeddings": embeddings},
            f
        )

    print("Cache saved ✅")


# =====================================================
# 5. NORMALIZE
# =====================================================
faiss.normalize_L2(embeddings)

N = embeddings.shape[0]


# =====================================================
# 6. BUILD FAISS INDEXES
# =====================================================
print("\nBuilding FAISS indexes...")


# ---------- FLAT ----------
flat_index = faiss.IndexFlatIP(DIM)
flat_index.add(embeddings)


# ---------- HNSW ----------
hnsw_index = faiss.IndexHNSWFlat(DIM, 32)
hnsw_index.add(embeddings)


# ---------- IVF ----------
nlist = int(np.sqrt(N))

quantizer = faiss.IndexFlatIP(DIM)

ivf_index = faiss.IndexIVFFlat(
    quantizer,
    DIM,
    nlist,
    faiss.METRIC_INNER_PRODUCT
)

ivf_index.train(embeddings)
ivf_index.add(embeddings)


# =====================================================
# 7. BENCHMARK
# =====================================================
QUERY = "Azure Data Engineer with Databricks and ADF"

methods = {
    "faiss_flat": flat_index,
    "faiss_hnsw": hnsw_index,
    "faiss_ivf": ivf_index
}

results = []


for run in range(1, RUNS + 1):

    print(f"\nRun {run}")

    for name, index in methods.items():

        # -------- Embedding Time --------
        t1 = time.time()

        q_vec = model.encode(
            QUERY,
            normalize_embeddings=True
        ).astype("float32")

        t2 = time.time()


        # -------- Search Time --------
        t3 = time.time()

        D, I = index.search(
            q_vec.reshape(1, -1),
            5
        )

        t4 = time.time()


        embed_time = t2 - t1
        search_time = t4 - t3
        total_time = t4 - t1


        results.append({
            "Method": name,
            "Embedding": round(embed_time, 4),
            "Search": round(search_time, 4),
            "Total": round(total_time, 4),
            "Run": run
        })


# =====================================================
# 8. PRINT TABLE
# =====================================================
print("\n================ BENCHMARK RESULTS ================\n")

print(
    f"{'Method':<12} | {'Embedding':<10} | "
    f"{'Search':<10} | {'Total':<10} | {'Run'}"
)

print("-" * 60)


for r in results:

    print(
        f"{r['Method']:<12} | "
        f"{r['Embedding']:<10} | "
        f"{r['Search']:<10} | "
        f"{r['Total']:<10} | "
        f"{r['Run']}"
    )


print("\n===============================================\n")
