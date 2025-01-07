"""
Script: app.py

1) Exposes a Flask API for query handling.
2) Performs FAISS-based semantic search to retrieve relevant chunks and metadata.
3) Returns the chunks and metadata as JSON for use by the custom GPT.
"""

from flask import Flask, request, jsonify
import sqlite3
import faiss
import openai
import numpy as np
import json

# File paths
DB_PATH = "books.db"
FAISS_INDEX_PATH = "faiss_index.index"
CHUNK_MAPPING_PATH = "id_to_chunk_mapping.json"

# Flask app setup
app = Flask(__name__)

def load_faiss_index(index_path):
    """
    Loads the FAISS index from the file.
    """
    print(f"Loading FAISS index from {index_path}...")
    index = faiss.read_index(index_path)
    return index


def load_chunk_mapping(mapping_path):
    """
    Loads the FAISS vector ID to chunk ID mapping.
    """
    with open(mapping_path, "r") as f:
        return json.load(f)


def embed_query(query):
    """
    Embeds the user query using OpenAI's `text-embedding-ada-002`.
    """
    response = openai.Embedding.create(
        input=query,
        model="text-embedding-ada-002"
    )
    return np.array(response["data"][0]["embedding"], dtype=np.float32)


def search_faiss(index, query_vector, top_k=5):
    """
    Performs a vector search in FAISS.
    Returns the FAISS vector IDs and distances of the top-k results.
    """
    query_vector = np.expand_dims(query_vector, axis=0)  # Reshape for FAISS
    distances, vector_ids = index.search(query_vector, top_k)
    return vector_ids[0], distances[0]  # Return the first (and only) query results


def get_chunks_from_db(db_path, chunk_ids):
    """
    Retrieves the chunk text and metadata for given chunk IDs from SQLite.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    query = f"""
        SELECT id, book_title, chapter_title, local_index, text
        FROM chunks
        WHERE id IN ({','.join(['?'] * len(chunk_ids))})
    """
    cursor.execute(query, chunk_ids)
    results = cursor.fetchall()
    conn.close()
    return results


@app.route("/query", methods=["POST"])
def handle_query():
    """
    Handles user queries sent to the Flask API.
    """
    data = request.json
    if "query" not in data:
        return jsonify({"error": "Query parameter is missing"}), 400

    user_query = data["query"]
    print(f"Received query: {user_query}")

    # Step 1: Load FAISS index and chunk mapping
    index = load_faiss_index(FAISS_INDEX_PATH)
    chunk_mapping = load_chunk_mapping(CHUNK_MAPPING_PATH)

    # Step 2: Embed the user query
    print("Embedding the query...")
    query_vector = embed_query(user_query)

    # Step 3: Perform FAISS search
    print("Searching FAISS index...")
    top_k = 5  # Number of results to retrieve
    vector_ids, distances = search_faiss(index, query_vector, top_k)

    # Step 4: Map FAISS vector IDs to chunk IDs
    chunk_ids = [chunk_mapping[str(vector_id)] for vector_id in vector_ids]

    # Step 5: Retrieve chunks and their metadata
    print("Fetching chunks and metadata...")
    matched_chunks = get_chunks_from_db(DB_PATH, chunk_ids)

    # Format results for JSON response
    results = []
    for chunk_id, book_title, chapter_title, local_index, text in matched_chunks:
        results.append({
            "chunk_id": chunk_id,
            "book_title": book_title,
            "chapter_title": chapter_title,
            "local_index": local_index,
            "text": text
        })

    # Step 6: Return results as JSON
    return jsonify({"query": user_query, "results": results})


if __name__ == "__main__":
    app.run(debug=True)
