import os
import re
import statistics
from ebooklib import epub
from bs4 import BeautifulSoup

def get_text_from_epub_fallback(epub_path):
    """
    Extract text from every item in the EPUB that is 'application/xhtml+xml',
    ignoring the spine. This helps handle non-standard EPUBs.
    Returns a list of (section_title, section_text).
    """
    book = epub.read_epub(epub_path)
    sections = []
    section_index = 0

    for item in book.get_items():
        if item.media_type == 'application/xhtml+xml':
            html_content = item.get_content()
            soup = BeautifulSoup(html_content, 'html.parser')

            # Attempt to extract a title from an <h1>, <h2>, or <h3>
            title_tag = soup.find(['h1', 'h2', 'h3'])
            section_title = title_tag.get_text().strip() if title_tag else f"Section_{section_index}"

            # Remove scripts and styles
            for tag in soup(["script", "style"]):
                tag.decompose()

            # Extract visible text
            text = soup.get_text(separator=' ')
            # Collapse extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()

            sections.append((section_title, text))
            section_index += 1

    return sections

def analyze_epub(epub_path):
    """
    Analyzes an EPUB file: extracts sections, counts words, prints stats.
    Returns a string containing the analysis results (per-chapter & summary).
    """
    sections = get_text_from_epub_fallback(epub_path)
    if not sections:
        return f"No valid text extracted from '{epub_path}'\n"

    word_counts = []
    lines = []
    lines.append(f"=== Analyzing '{epub_path}' ===")

    for i, (title, text) in enumerate(sections, start=1):
        words = text.split()
        word_count = len(words)
        word_counts.append(word_count)
        lines.append(f"  Section {i}: '{title[:50]}...' => Word Count: {word_count}")

    if not word_counts:
        lines.append("No word counts found (possibly empty sections).")
        return "\n".join(lines) + "\n"

    min_words = min(word_counts)
    max_words = max(word_counts)
    avg_words = sum(word_counts) / len(word_counts)
    median_words = statistics.median(word_counts)

    lines.append("\n--- Summary ---")
    lines.append(f" Total Sections: {len(word_counts)}")
    lines.append(f" Min Word Count: {min_words}")
    lines.append(f" Max Word Count: {max_words}")
    lines.append(f" Average Words: {avg_words:.2f}")
    lines.append(f" Median Words: {median_words:.2f}")
    lines.append("")

    return "\n".join(lines) + "\n"

def main():
    folder_path = "books"
    output_file = "analysis_output.txt"

    # Collect all EPUB files in the 'books' folder
    epub_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".epub")]

    # If there are no EPUB files, let the user know
    if not epub_files:
        print(f"No EPUB files found in '{folder_path}'.")
        return

    # Open an output file to write all results
    with open(output_file, "w", encoding="utf-8") as out:
        for epub_name in epub_files:
            epub_path = os.path.join(folder_path, epub_name)
            try:
                result = analyze_epub(epub_path)
                out.write(result + "\n")
                print(f"Analysis for '{epub_name}' written to {output_file}")
            except Exception as e:
                error_msg = f"Error analyzing '{epub_name}': {e}\n"
                out.write(error_msg)
                print(error_msg)

    print(f"\nAll analysis complete! Results saved to '{output_file}'.")

if __name__ == "__main__":
    main()
