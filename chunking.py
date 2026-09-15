import json
from langchain_text_splitters import RecursiveCharacterTextSplitter

INPUT_FILE = "document_pages.json"
FIXED_OUTPUT = "fixed_chunks.json"
PARAGRAPH_OUTPUT = "paragraph_chunks.json"


# -----------------------------------
# Load extracted pages
# -----------------------------------

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    pages = json.load(file)


# ===================================
# STRATEGY 1: FIXED-SIZE CHUNKING
# ===================================

fixed_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""]
)

fixed_chunks = []

for page in pages:
    text = page["text"]

    if not text.strip():
        continue

    chunks = fixed_splitter.split_text(text)

    for i, chunk in enumerate(chunks):
        fixed_chunks.append({
            "chunk_id": f"fixed_{page['page']}_{i}",
            "method": "fixed_size",
            "page": page["page"],
            "text": chunk.strip()
        })


# ===================================
# STRATEGY 2: PARAGRAPH-BASED CHUNKING
# ===================================

paragraph_chunks = []

MAX_CHUNK_SIZE = 1800

for page in pages:
    text = page["text"].strip()

    if not text:
        continue

    # Try to identify paragraphs
    paragraphs = [
        p.strip()
        for p in text.replace("\r", "").split("\n\n")
        if p.strip()
    ]

    # If the PDF does not contain blank lines,
    # use non-empty lines as paragraph units.
    if len(paragraphs) <= 1:
        paragraphs = [
            line.strip()
            for line in text.split("\n")
            if line.strip()
        ]

    current = ""
    chunk_number = 0

    for paragraph in paragraphs:

        if len(paragraph) > MAX_CHUNK_SIZE:
            if current:
                paragraph_chunks.append({
                    "chunk_id": f"paragraph_{page['page']}_{chunk_number}",
                    "method": "paragraph_based",
                    "page": page["page"],
                    "text": current.strip()
                })
                chunk_number += 1
                current = ""

            paragraph_chunks.append({
                "chunk_id": f"paragraph_{page['page']}_{chunk_number}",
                "method": "paragraph_based",
                "page": page["page"],
                "text": paragraph[:MAX_CHUNK_SIZE].strip()
            })

            chunk_number += 1
            continue

        if not current:
            current = paragraph

        elif len(current) + len(paragraph) + 2 <= MAX_CHUNK_SIZE:
            current += "\n\n" + paragraph

        else:
            paragraph_chunks.append({
                "chunk_id": f"paragraph_{page['page']}_{chunk_number}",
                "method": "paragraph_based",
                "page": page["page"],
                "text": current.strip()
            })

            chunk_number += 1
            current = paragraph

    if current:
        paragraph_chunks.append({
            "chunk_id": f"paragraph_{page['page']}_{chunk_number}",
            "method": "paragraph_based",
            "page": page["page"],
            "text": current.strip()
        })


# -----------------------------------
# Save results
# -----------------------------------

with open(FIXED_OUTPUT, "w", encoding="utf-8") as file:
    json.dump(fixed_chunks, file, ensure_ascii=False, indent=2)

with open(PARAGRAPH_OUTPUT, "w", encoding="utf-8") as file:
    json.dump(paragraph_chunks, file, ensure_ascii=False, indent=2)


# -----------------------------------
# Display comparison
# -----------------------------------

print("======================================")
print("CHUNKING COMPLETED")
print("======================================")

print("Pages:", len(pages))

print("\nStrategy 1: Fixed-size chunking")
print("Chunks:", len(fixed_chunks))
print(
    "Average characters:",
    round(
        sum(len(c["text"]) for c in fixed_chunks) / len(fixed_chunks),
        2
    )
)

print("\nStrategy 2: Paragraph-based chunking")
print("Chunks:", len(paragraph_chunks))
print(
    "Average characters:",
    round(
        sum(len(c["text"]) for c in paragraph_chunks) / len(paragraph_chunks),
        2
    )
)

print("\nFiles created:")
print("-", FIXED_OUTPUT)
print("-", PARAGRAPH_OUTPUT)

print("\n--- Fixed-size example ---")
print(fixed_chunks[0]["text"][:500])

print("\n--- Paragraph-based example ---")
print(paragraph_chunks[0]["text"][:500])