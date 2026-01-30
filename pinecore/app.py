import os
import time
import pickle
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from pypdf import PdfReader
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer


# =====================================================
# CONFIG
# =====================================================
RESUME_FOLDER = r"Education"
CACHE_FILE = "resume_vectors.pkl"

# ✅ Correct way to load key
PINECONE_API_KEY = os.getenv("pcsk_49sMfN_SGT8y2Hxfz5M5Pvf4vCZiGbdfWz2TgjrKAG3SE1arBMxutSzG6EweWy9HLBfJJ4")

INDEX_NAME = "resume-search-index"

CHUNK_SIZE = 250
BATCH_SIZE = 64
UPSERT_BATCH = 200


# =====================================================
# 1. LOAD MODEL (Warm-Up)
# =====================================================
print("Loading model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

# Warm-up
_ = model.encode("warmup query")

print("Model Ready ✅")

embedding_dim = model.get_sentence_embedding_dimension()


# =====================================================
# 2. CONNECT TO PINECONE
# =====================================================
pc = Pinecone(api_key="pcsk_49sMfN_SGT8y2Hxfz5M5Pvf4vCZiGbdfWz2TgjrKAG3SE1arBMxutSzG6EweWy9HLBfJJ4")

if INDEX_NAME not in pc.list_indexes().names():

    print("Creating index...")

    pc.create_index(
        name=INDEX_NAME,
        dimension=embedding_dim,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)

print("Pinecone Index Ready ✅")


# =====================================================
# 3. TEXT CHUNKING
# =====================================================
def chunk_text(text, chunk_size=CHUNK_SIZE):

    words = text.split()

    return [
        " ".join(words[i:i+chunk_size])
        for i in range(0, len(words), chunk_size)
    ]


# =====================================================
# 4. FAST PDF EXTRACTION
# =====================================================
def extract_pdf(file_path):

    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + " "

    return text.strip()


print("\nExtracting PDFs...")


pdf_files = [
    os.path.join(RESUME_FOLDER, f)
    for f in os.listdir(RESUME_FOLDER)
    if f.endswith(".pdf")
]


texts = []

start = time.time()

with ThreadPoolExecutor(max_workers=8) as executor:

    for txt in tqdm(executor.map(extract_pdf, pdf_files),
                    total=len(pdf_files)):

        if txt:
            texts.append(txt)


print("PDF Extraction Done ✅",
      round(time.time() - start, 2), "sec")


# =====================================================
# 5. BUILD CHUNKS
# =====================================================
print("\nChunking resumes...")

all_chunks = []
metadata = []


for file_name, resume_text in zip(pdf_files, texts):

    chunks = chunk_text(resume_text)

    for chunk in chunks:

        all_chunks.append(chunk)

        metadata.append({
            "file": os.path.basename(file_name)
        })


print("Total chunks:", len(all_chunks))


# =====================================================
# 6. EMBEDDING CACHE
# =====================================================
if os.path.exists(CACHE_FILE):

    print("\nLoading cached embeddings...")

    with open(CACHE_FILE, "rb") as f:
        vectors = pickle.load(f)

else:

    print("\nGenerating embeddings (1-time)...")

    start = time.time()

    embeddings = model.encode(
        all_chunks,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    vectors = []

    for i, emb in enumerate(embeddings):

        vectors.append({
            "id": f"chunk-{i}",
            "values": emb.tolist(),
            "metadata": {
                "file": metadata[i]["file"],
                "text": all_chunks[i][:300]
            }
        })

    print("Embedding done ✅",
          round(time.time() - start, 2), "sec")

    # Save cache
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(vectors, f)

    print("Embeddings cached locally ✅")


# =====================================================
# 7. UPSERT
# =====================================================
print("\nUploading to Pinecone...")

for i in tqdm(range(0, len(vectors), UPSERT_BATCH)):

    index.upsert(vectors=vectors[i:i+UPSERT_BATCH])


print("Upload completed ✅")


# =====================================================
# 8. SEARCH WITH TABLE OUTPUT
# =====================================================
def search_resume(query, top_k=5):

    print("\nSearching for:", query)

    # -------------------------------
    # Timing
    # -------------------------------
    start = time.time()

    query_vector = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    result = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    end = time.time()

    latency = round(end - start, 4)


    # -------------------------------
    # Build Table
    # -------------------------------
    rows = []

    for i, match in enumerate(result["matches"], 1):

        rows.append({
            "Rank": i,
            "Score": round(match["score"], 4),
            "Resume": match["metadata"]["file"],
            "Snippet": match["metadata"]["text"][:80] + "...",
            "Latency (s)": latency
        })


    # -------------------------------
    # Print Table
    # -------------------------------
    print("\n================ SEARCH RESULTS ================\n")

    print(
        f"{'Rank':<5} | {'Score':<8} | {'Resume':<25} | "
        f"{'Snippet':<40} | {'Latency(s)'}"
    )

    print("-" * 95)

    for row in rows:

        print(
            f"{row['Rank']:<5} | "
            f"{row['Score']:<8} | "
            f"{row['Resume']:<25} | "
            f"{row['Snippet']:<40} | "
            f"{row['Latency (s)']}"
        )

    print("\n==============================================\n")


# =====================================================
# RUN SEARCH
# =====================================================
search_resume("Azure Data Engineer with Databricks and ADF")
