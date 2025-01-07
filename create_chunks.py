"""
Script: toc_chunk_epubs_to_sqlite.py

1) Reads all EPUB files from 'books/' folder.
2) Retrieves the Table of Contents (TOC) so we can get real chapter titles (entry.title).
3) For each TOC entry, we parse the matching HTML file, extract text, chunk it, and store in SQLite.

Requirements:
    pip install ebooklib beautifulsoup4 sqlalchemy
"""

import os
import re
from pathlib import Path

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import sessionmaker, declarative_base

Base = declarative_base()

class Chunk(Base):
    __tablename__ = 'chunks'

    id = Column(Integer, primary_key=True, autoincrement=True)  # global PK
    book_title = Column(String, nullable=False)
    chapter_title = Column(String, nullable=True)
    local_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)

def get_db_session(db_name="books.db"):
    engine = create_engine(f"sqlite:///{db_name}", echo=False)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()

def split_into_chunks(chapter_text, max_single=1000, chunk_size=800):
    """
    Splits chapter_text:
      - If under max_single words, return as one chunk.
      - Else, break into ~chunk_size word segments.
    """
    words = chapter_text.split()
    num_words = len(words)

    if num_words <= max_single:
        return [chapter_text]

    chunks = []
    start = 0
    while start < num_words:
        end = start + chunk_size
        chunk_text = " ".join(words[start:end])
        chunks.append(chunk_text)
        start = end

    return chunks

def extract_text_from_href(book, href):
    """
    Given an epub.EpubBook 'book' and an href (e.g. 'OEBPS/chapter1.html'),
    find the matching item, parse the HTML, and return the cleaned text.
    """
    item = book.get_item_with_href(href)
    if not item:
        return ""  # no matching item found

    if item.get_type() != ebooklib.ITEM_DOCUMENT:
        return ""  # it's not an HTML/XHTML document

    html_content = item.get_content()
    soup = BeautifulSoup(html_content, 'html.parser')

    # Extract text from body, removing extra whitespace
    text = soup.get_text(separator=' ', strip=True)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_toc_entries(book, entries, book_title, session):
    """
    Recursively process a list of TOC entries (which can be Link objects or nested lists).
    For each Link, we:
      1) Get entry.title (the real chapter name)
      2) Get entry.href to find the relevant HTML
      3) Extract text and chunk it
      4) Store in the DB
    """
    for entry in entries:
        if isinstance(entry, epub.Link):
            # It's a single TOC entry
            chapter_title = entry.title
            href = entry.href

            chapter_text = extract_text_from_href(book, href)
            if not chapter_text:
                continue

            # Split into chunks
            chunk_texts = split_into_chunks(chapter_text, max_single=1000, chunk_size=800)
            for local_idx, chunk_str in enumerate(chunk_texts):
                db_chunk = Chunk(
                    book_title=book_title,
                    chapter_title=chapter_title[:200],  # truncate for safety
                    local_index=local_idx,
                    text=chunk_str
                )
                session.add(db_chunk)

        elif isinstance(entry, list):
            # It's a nested TOC (sub-chapters). Recursively process.
            process_toc_entries(book, entry, book_title, session)

        else:
            # Some EPUBs embed tuples like (href, title, subitems) 
            # Instead of the Link class. Let's handle that:
            if isinstance(entry, tuple) and len(entry) >= 2:
                href = entry[0]
                chapter_title = entry[1]
                if len(entry) == 3 and isinstance(entry[2], list):
                    # sub-chapters in entry[2]
                    subitems = entry[2]
                else:
                    subitems = []

                # Extract text for the main entry
                chapter_text = extract_text_from_href(book, href)
                if chapter_text:
                    chunk_texts = split_into_chunks(chapter_text, 1000, 800)
                    for local_idx, chunk_str in enumerate(chunk_texts):
                        db_chunk = Chunk(
                            book_title=book_title,
                            chapter_title=str(chapter_title)[:200],
                            local_index=local_idx,
                            text=chunk_str
                        )
                        session.add(db_chunk)

                # Recursively process subitems
                if subitems:
                    process_toc_entries(book, subitems, book_title, session)
            # else: ignore other odd structures

def main():
    session = get_db_session("books.db")
    books_folder = Path("books")
    epub_files = list(books_folder.glob("*.epub"))

    if not epub_files:
        print("No EPUB files found in 'books' folder.")
        return

    for epub_file in epub_files:
        print(f"Processing: {epub_file.name}")
        book_title = epub_file.stem

        # Read the EPUB
        book = epub.read_epub(str(epub_file))

        # For older ebooklib versions, you don't have get_toc(), so use book.toc:
        toc = book.toc  

        # Recursively process TOC entries (same logic as before)
        process_toc_entries(book, toc, book_title, session)
        session.commit()

    print("All EPUBs processed with real TOC chapter titles stored in books.db.")


if __name__ == "__main__":
    main()
