import os
import json
import uuid
from typing import List, Dict, Any
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv

# Load variables from .env file into the environment
load_dotenv()

# Now retrieve them
api_key = os.getenv("QDRANT_API_KEY")
url = os.getenv("QDRANT_URL")

# --- CONFIGURATION ---
INPUT_DIR = "scraped_data"
COLLECTION_NAME = "university_knowledge_base"
BATCH_SIZE = 100

# 1. Select Free Open-Source Model & Match its Dimensions
# bge-small-en-v1.5 is an industry favorite for fast, accurate local RAG
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384  # Changed from 1536! Required for small open-source models

print(f"🧠 Loading local AI model '{EMBEDDING_MODEL_NAME}' into memory...")
# This automatically downloads the weights (~130MB) on first run and caches them locally
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Updated Client Initialization
if url and api_key:
    # Connect to Qdrant Cloud/Server
    qdrant_client = QdrantClient(url=url, api_key=api_key)
    print("🌐 Connected to Qdrant Cloud/Server.")
else:
    # Fallback to local persistent storage
    qdrant_client = QdrantClient(path="./qdrant_local_data")
    print("💾 Using local Qdrant storage.")


def setup_qdrant_collection():
    """Creates the vector collection matching our 384-dimension local model."""
    collections = qdrant_client.get_collections().collections
    exists = any(col.name == COLLECTION_NAME for col in collections)

    if not exists:
        print(f"📦 Creating 384-dimensional Qdrant collection: '{COLLECTION_NAME}'...")
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=rest.VectorParams(
                size=EMBEDDING_DIM, distance=rest.Distance.COSINE
            ),
        )
    else:
        print(f"📦 Collection '{COLLECTION_NAME}' already exists. Updating payloads...")


def generate_deterministic_id(url: str, chunk_index: int) -> str:
    unique_string = f"{url}_chunk_{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, unique_string))


def main():
    if not os.path.exists(INPUT_DIR):
        raise FileNotFoundError(f"❌ Input directory '{INPUT_DIR}' not found.")

    setup_qdrant_collection()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
        keep_separator=False,
    )

    all_chunks: List[Dict[str, Any]] = []
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]

    print(f"📑 Reading and chunking {len(files)} JSON files...")
    for filename in tqdm(files, desc="Chunking files"):
        filepath = os.path.join(INPUT_DIR, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            content = data.get("content_markdown", "")
            metadata = data.get("metadata", {})
            url = metadata.get("url", "unknown_url")

            if not content.strip():
                continue

            text_chunks = splitter.split_text(content)

            for idx, text_segment in enumerate(text_chunks):
                all_chunks.append(
                    {
                        "id": generate_deterministic_id(url, idx),
                        "text": text_segment,
                        "payload": {
                            "text": text_segment,
                            "url": url,
                            "title": metadata.get("title", "Untitled"),
                            "doc_id": data.get("doc_id", filename),
                            "chunk_index": idx,
                            "total_chunks": len(text_chunks),
                        },
                    }
                )
        except Exception as e:
            print(f"\n⚠️ Error processing {filename}: {str(e)}")

    print(
        f"🚀 Total chunks: {len(all_chunks)}. Starting local vector generation & ingestion..."
    )

    # 3. Batch Embed Locally and Upload to Qdrant
    for i in tqdm(range(0, len(all_chunks), BATCH_SIZE), desc="Ingesting to Qdrant"):
        batch = all_chunks[i : i + BATCH_SIZE]
        texts_to_embed = [item["text"] for item in batch]

        # Run inference locally on your CPU/GPU using SentenceTransformer (.encode returns numpy array/list)
        embeddings = embedder.encode(
            texts_to_embed, show_progress_bar=False, normalize_embeddings=True
        ).tolist()

        points = [
            rest.PointStruct(id=item["id"], vector=emb, payload=item["payload"])
            for item, emb in zip(batch, embeddings)
        ]

        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)

    collection_info = qdrant_client.get_collection(collection_name=COLLECTION_NAME)
    print(
        f"\n🏁 Zero-Cost Ingestion Complete! Total 384-dim vectors in Qdrant: {collection_info.points_count}"
    )


if __name__ == "__main__":
    main()
