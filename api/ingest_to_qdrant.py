import os
import json
import uuid
import hashlib
from functools import lru_cache
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
import logging

# Configure the logging format and level
logger = logging.getLogger(__name__)

# Qdrant's client is chatty on every upsert; quiet it once here instead of
# toggling the level inside the ingestion loop.
logging.getLogger("httpx").setLevel(logging.WARNING)

# Load variables from .env file into the environment
load_dotenv()

# Now retrieve them
api_key = os.getenv("QDRANT_API_KEY")
url = os.getenv("QDRANT_URL")

# --- CONFIGURATION ---
INPUT_DIR = "scraped_data"
COLLECTION_NAME = "university_knowledge_base"
BATCH_SIZE = 100
ENCODE_BATCH_SIZE = 64  # Internal batch used by SentenceTransformer.encode

# Safety guard for the reconcile/purge step: if this run scraped fewer than
# this fraction of the documents already in Qdrant, we assume the scrape
# largely failed and REFUSE to delete stale docs (which would otherwise wipe
# most of the knowledge base). Purely additive/update work still proceeds.
MIN_RECONCILE_RATIO = 0.5

# 1. Select Free Open-Source Model & Match its Dimensions
# bge-small-en-v1.5 is an industry favorite for fast, accurate local RAG
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384  # Changed from 1536! Required for small open-source models


@lru_cache(maxsize=1)
def get_embedder():
    """Lazily load the embedding model on first use (not at import time).

    The `sentence_transformers`/`torch` import lives *inside* this function on
    purpose: importing them pulls ~300-500MB of torch runtime into memory. On a
    1GB box that must NOT be resident while the scraper's headless Chromium is
    running, or the two peaks overlap and the container gets OOM-killed. By
    importing here, torch only loads during ingest() — after the browser (and
    its memory) is gone.
    """
    from sentence_transformers import SentenceTransformer

    logger.info(f"🧠 Loading local AI model '{EMBEDDING_MODEL_NAME}' into memory...")
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def get_qdrant() -> QdrantClient:
    """Lazily create the Qdrant client on first use (not at import time)."""
    if url and api_key:
        logger.info("🌐 Connected to Qdrant Cloud/Server.")
        return QdrantClient(url=url, api_key=api_key)
    logger.info("💾 Using local Qdrant storage.")
    return QdrantClient(path="./qdrant_local_data")


def setup_qdrant_collection(client: QdrantClient):
    """Creates the vector collection matching our 384-dimension local model."""
    collections = client.get_collections().collections
    exists = any(col.name == COLLECTION_NAME for col in collections)

    if not exists:
        logger.info(
            f"📦 Creating 384-dimensional Qdrant collection: '{COLLECTION_NAME}'..."
        )
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=rest.VectorParams(
                size=EMBEDDING_DIM, distance=rest.Distance.COSINE
            ),
        )
    else:
        logger.info(
            f"📦 Collection '{COLLECTION_NAME}' already exists. Updating payloads..."
        )


def get_ingested_hashes(client: QdrantClient) -> Dict[str, Optional[str]]:
    """Return {doc_id: content_hash} for every document already in Qdrant.

    The content_hash lets us tell an *unchanged* document (skip it) from a
    *changed* one (re-ingest it). Documents ingested before content_hash
    existed map to None, so they are treated as changed and get backfilled.
    All chunks of a doc share the same doc_id/content_hash, so the first wins.
    """
    seen: Dict[str, Optional[str]] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,
            offset=offset,
            with_payload=["doc_id", "content_hash"],
            with_vectors=False,
        )
        for p in points:
            payload = p.payload or {}
            doc_id = payload.get("doc_id")
            if doc_id and doc_id not in seen:
                seen[doc_id] = payload.get("content_hash")
        if offset is None:
            break
    return seen


def delete_doc_points(client: QdrantClient, doc_id: str):
    """Delete every point belonging to a document.

    Called before re-ingesting a changed doc so that stale chunks — especially
    orphaned high-index chunks when the new version splits into fewer pieces —
    don't linger. wait=True so the delete finishes before we upsert the new
    (deterministically-IDed) chunks; otherwise an async delete could race and
    wipe the freshly written points.
    """
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=rest.FilterSelector(
            filter=rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="doc_id", match=rest.MatchValue(value=doc_id)
                    )
                ]
            )
        ),
        wait=True,
    )


def reconcile_deletions(
    client: QdrantClient,
    existing: Dict[str, Optional[str]],
    seen_doc_ids: set,
):
    """Delete docs that are in Qdrant but were absent from this run's scrape.

    Keeps the collection a true mirror of the live site (handles removals, not
    just adds/updates). Guarded by MIN_RECONCILE_RATIO: if the scrape returned
    far fewer docs than Qdrant already holds, we assume it largely failed and
    refuse to purge — otherwise a broken crawl would empty the knowledge base.
    """
    stale = set(existing) - seen_doc_ids
    if not stale:
        return

    # Safety guard against a failed/partial scrape wiping everything.
    if existing and len(seen_doc_ids) < MIN_RECONCILE_RATIO * len(existing):
        logger.warning(
            f"⚠️ Skipping reconcile: only {len(seen_doc_ids)} docs scraped vs "
            f"{len(existing)} in Qdrant (< {MIN_RECONCILE_RATIO:.0%}). "
            f"Assuming a failed scrape; {len(stale)} stale docs left untouched."
        )
        return

    for doc_id in stale:
        delete_doc_points(client, doc_id)

    logger.info(f"🗑️ Reconcile: removed {len(stale)} docs no longer on the source.")


def clear_qdrant_collection() -> Dict[str, Any]:
    """Empty the Qdrant collection by dropping and recreating it.

    Recreating (rather than deleting points one-by-one) is the cheapest way to
    guarantee an empty collection while keeping the 384-dim schema intact, so a
    subsequent ingest() can upsert straight away without a separate setup step.
    """
    client = get_qdrant()

    collections = client.get_collections().collections
    existed = any(col.name == COLLECTION_NAME for col in collections)

    if existed:
        client.delete_collection(collection_name=COLLECTION_NAME)
        logger.info(f"🧨 Dropped Qdrant collection '{COLLECTION_NAME}'.")

    # Recreate the empty collection so the API is ready for the next ingest.
    setup_qdrant_collection(client)

    info = client.get_collection(collection_name=COLLECTION_NAME)
    logger.info(
        f"✅ Qdrant collection '{COLLECTION_NAME}' emptied. "
        f"Points now: {info.points_count}."
    )
    return {"collection": COLLECTION_NAME, "existed": existed, "points_count": info.points_count}


def generate_deterministic_id(url: str, chunk_index: int) -> str:
    unique_string = f"{url}_chunk_{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, unique_string))


def ingest():
    if not os.path.exists(INPUT_DIR):
        raise FileNotFoundError(f"❌ Input directory '{INPUT_DIR}' not found.")

    client = get_qdrant()
    embedder = get_embedder()

    setup_qdrant_collection(client)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
        keep_separator=False,
    )

    files = [
        entry.name
        for entry in os.scandir(INPUT_DIR)
        if entry.is_file() and entry.name.endswith(".json")
    ]

    # Map of doc_id -> stored content hash, so we can detect changes.
    existing = get_ingested_hashes(client)
    if existing:
        logger.info(f"🔎 {len(existing)} documents already in the collection.")

    total_ingested = 0  # chunks embedded+upserted this run
    new_docs = 0
    changed_docs = 0
    skipped = 0
    seen_doc_ids: set = set()  # docs present in this run's scrape (for reconcile)
    buffer: List[Dict[str, Any]] = []

    def flush(wait: bool = False):
        """Embed and upsert the current buffer, then clear it."""
        nonlocal total_ingested
        if not buffer:
            return
        texts = [item["text"] for item in buffer]
        embeddings = embedder.encode(
            texts,
            batch_size=ENCODE_BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).tolist()
        points = [
            rest.PointStruct(id=item["id"], vector=emb, payload=item["payload"])
            for item, emb in zip(buffer, embeddings)
        ]
        # wait=False lets uploads pipeline instead of blocking on indexing;
        # the final flush waits so the reported count is accurate.
        client.upsert(collection_name=COLLECTION_NAME, points=points, wait=wait)
        total_ingested += len(points)
        buffer.clear()

    logger.info(f"📑 Streaming, chunking, and ingesting {len(files)} JSON files...")
    for filename in tqdm(files, desc="Ingesting"):
        filepath = os.path.join(INPUT_DIR, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            doc_id = data.get("doc_id", filename)
            content = data.get("content_markdown", "")
            metadata = data.get("metadata", {})
            url_value = metadata.get("url", "unknown_url")

            # A file exists for this doc, so it's still live on the site. Mark it
            # seen even if content is momentarily empty, so a transient blank
            # scrape doesn't get it purged in the reconcile step below.
            seen_doc_ids.add(doc_id)

            if not content.strip():
                continue

            # Fingerprint the document's content to detect changes cheaply.
            content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()

            if doc_id in existing:
                if existing[doc_id] == content_hash:
                    # Unchanged since last run -> nothing to do.
                    skipped += 1
                    continue
                # Content changed -> remove old chunks before writing new ones.
                delete_doc_points(client, doc_id)
                changed_docs += 1
            else:
                new_docs += 1

            text_chunks = splitter.split_text(content)

            for idx, text_segment in enumerate(text_chunks):
                buffer.append(
                    {
                        "id": generate_deterministic_id(url_value, idx),
                        "text": text_segment,
                        "payload": {
                            "text": text_segment,
                            "url": url_value,
                            "title": metadata.get("title", "Untitled"),
                            "doc_id": doc_id,
                            "content_hash": content_hash,
                            "chunk_index": idx,
                            "total_chunks": len(text_chunks),
                        },
                    }
                )

                # Flush in fixed-size batches so peak memory stays at one batch,
                # not the entire corpus.
                if len(buffer) >= BATCH_SIZE:
                    flush()
        except Exception as e:
            logger.error(f"\n⚠️ Error processing {filename}: {str(e)}")

    # Final flush (wait=True so the count below reflects everything).
    flush(wait=True)

    logger.info(
        f"📊 Documents: {new_docs} new, {changed_docs} changed, {skipped} unchanged (skipped)."
    )

    # Reconcile: purge docs that vanished from the source (in Qdrant but not in
    # this run's scrape). Guarded so a mostly-failed scrape can't wipe the KB.
    reconcile_deletions(client, existing, seen_doc_ids)

    collection_info = client.get_collection(collection_name=COLLECTION_NAME)
    logger.info(
        f"\n🏁 Zero-Cost Ingestion Complete! Ingested {total_ingested} chunks this run. "
        f"Total 384-dim vectors in Qdrant: {collection_info.points_count}"
    )
