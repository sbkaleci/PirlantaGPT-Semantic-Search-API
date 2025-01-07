"""
Script: generate_faiss_incremental.py

1) Reads chunks from SQLite database.
2) Checks for an existing FAISS index and adds only the chunks that are not yet embedded.
3) Generates embeddings using OpenAI's `text-embedding-ada-002`.
4) Stores embeddings in the FAISS index, continuing from where it left off.
"""

import sqlite3
import faiss
import openai
import numpy as np
import json
from pathlib import Path

# SQLite database location
DB_PATH = "books.db"

# FAISS index file
FAISS_INDEX_PATH = "faiss_index.index"

# Mapping file for chunk_id to FAISS vector IDs
CHUNK_MAPPING_PATH = "id_to_chunk_mapping.json"


def get_chunks_from_db(db_path):
    """
    Reads chunks from the SQLite database.
    Returns a list of (chunk_id, text).
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, text FROM chunks")
    chunks = cursor.fetchall()  # List of (id, text)
    conn.close()
    return chunks


def generate_embedding(text):
    """
    Generates an embedding for a given text using OpenAI's `text-embedding-ada-002`.
    Returns a 1536-dimensional vector.
    """
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return np.array(response["data"][0]["embedding"])


def create_faiss_index(dimension, index_path=None):
    """
    Creates a FAISS index.
    If `index_path` exists, loads the existing index.
    Otherwise, creates a new one.
    """
    if index_path and Path(index_path).exists():
        print(f"Loading existing FAISS index from {index_path}")
        index = faiss.read_index(index_path)
    else:
        print("Creating a new FAISS index")
        index = faiss.IndexFlatL2(dimension)  # L2 distance
    return index


def load_chunk_mapping(mapping_path):
    """
    Loads the chunk-to-FAISS mapping from a JSON file.
    Returns an empty dict if the file doesn't exist.
    """
    if Path(mapping_path).exists():
        with open(mapping_path, "r") as f:
            return json.load(f)
    return {}


def save_chunk_mapping(mapping, mapping_path):
    """
    Saves the chunk-to-FAISS mapping to a JSON file.
    """
    with open(mapping_path, "w") as f:
        json.dump(mapping, f)
    print(f"Chunk-to-FAISS mapping saved to {mapping_path}")


def save_faiss_index(index, index_path):
    """
    Saves the FAISS index to a file.
    """
    faiss.write_index(index, index_path)
    print(f"FAISS index saved to {index_path}")


def main():
    # 1. Load chunks from SQLite database
    print("Loading chunks from SQLite database...")
    chunks = get_chunks_from_db(DB_PATH)

    # 2. Initialize FAISS index
    dimension = 1536  # Embedding size of text-embedding-ada-002
    index = create_faiss_index(dimension, FAISS_INDEX_PATH)

    # 3. Load chunk-to-FAISS mapping
    chunk_mapping = load_chunk_mapping(CHUNK_MAPPING_PATH)

    # 4. Identify unembedded chunks
    embedded_chunk_ids = set(chunk_mapping.values())
    unembedded_chunks = [(chunk_id, text) for chunk_id, text in chunks if str(chunk_id) not in embedded_chunk_ids]
    print(f"{len(unembedded_chunks)} chunks need embeddings (out of {len(chunks)} total).")

    # 5. Embed chunks and add to FAISS index
    for chunk_id, text in unembedded_chunks:
        try:
            embedding = generate_embedding(text)
            index.add(np.array([embedding], dtype=np.float32))  # Add to FAISS
            chunk_mapping[str(index.ntotal - 1)] = chunk_id  # Map FAISS vector ID to chunk_id
        except Exception as e:
            print(f"Error embedding chunk {chunk_id}: {e}")
            continue

    # 6. Save updated FAISS index and mapping
    save_faiss_index(index, FAISS_INDEX_PATH)
    save_chunk_mapping(chunk_mapping, CHUNK_MAPPING_PATH)

    print("All embeddings processed and stored in FAISS index.")


if __name__ == "__main__":
    main()
