"""
Script: search_faiss.py

1) Takes a user input query.
2) Embeds the query using OpenAI's `text-embedding-ada-002`.
3) Searches for the most relevant chunks in the FAISS index.
4) Retrieves chunk text and metadata from SQLite based on the results.
"""

import sqlite3
import faiss
import openai
import numpy as np
import json

# File paths
DB_PATH = "books.db"
FAISS_INDEX_PATH = "faiss_index.index"
CHUNK_MAPPING_PATH = "id_to_chunk_mapping.json"


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


def get_all_chunks_for_chapter(db_path, book_title, chapter_title):
    """
    Retrieves all chunks for the specified book and chapter.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    query = """
        SELECT id, book_title, chapter_title, local_index, text
        FROM chunks
        WHERE book_title = ? AND chapter_title = ?
        ORDER BY local_index
    """
    cursor.execute(query, (book_title, chapter_title))
    results = cursor.fetchall()
    conn.close()
    return results


def merge_chunks_by_chapter(chunks):
    """
    Merges chunks belonging to the same chapter into a single cohesive text.
    Returns a dictionary where keys are (book_title, chapter_title) and values are merged texts.
    """
    chapter_map = {}
    for chunk_id, book_title, chapter_title, local_index, text in chunks:
        key = (book_title, chapter_title)
        if key not in chapter_map:
            chapter_map[key] = []
        chapter_map[key].append((local_index, text))

    # Sort chunks by local_index and merge the text for each chapter
    merged_chapters = []
    for (book_title, chapter_title), chunk_list in chapter_map.items():
        sorted_chunks = sorted(chunk_list, key=lambda x: x[0])  # Sort by local_index
        merged_text = " ".join(text for _, text in sorted_chunks)
        merged_chapters.append({
            "book_title": book_title,
            "chapter_title": chapter_title,
            "text": merged_text
        })

    return merged_chapters


def main():
    # 1. Load FAISS index and mapping
    index = load_faiss_index(FAISS_INDEX_PATH)
    chunk_mapping = load_chunk_mapping(CHUNK_MAPPING_PATH)

    # 2. Take user input
    query = input("Enter your query: ")

    # 3. Embed the user query
    print("Embedding the query...")
    query_vector = embed_query(query)

    # 4. Perform FAISS search
    print("Searching FAISS index...")
    top_k = 5  # Number of results to retrieve
    vector_ids, distances = search_faiss(index, query_vector, top_k)

    # 5. Map FAISS vector IDs to chunk IDs
    chunk_ids = [chunk_mapping[str(vector_id)] for vector_id in vector_ids]

    # 6. Retrieve chunks and their metadata
    print("Fetching chunks and metadata...")
    matched_chunks = get_chunks_from_db(DB_PATH, chunk_ids)

    # 7. Retrieve all chunks from the chapters of the matched chunks
    all_relevant_chunks = []
    seen_chapters = set()  # To avoid duplicate processing
    for _, book_title, chapter_title, _, _ in matched_chunks:
        chapter_key = (book_title, chapter_title)
        if chapter_key not in seen_chapters:
            chapter_chunks = get_all_chunks_for_chapter(DB_PATH, book_title, chapter_title)
            all_relevant_chunks.extend(chapter_chunks)
            seen_chapters.add(chapter_key)

    # 8. Merge chunks by chapter
    print("Merging chunks by chapter...")
    merged_chapters = merge_chunks_by_chapter(all_relevant_chunks)

    # 9. Output results as JSON
    print("\nResults (JSON):")
    results_json = json.dumps(merged_chapters, ensure_ascii=False, indent=4)
    with open("results.json", "w", encoding="utf-8") as f:
        f.write(results_json)
    print(results_json)


if __name__ == "__main__":
    main()