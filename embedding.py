import json
from sentence_transformers import SentenceTransformer

FIXED_FILE = "fixed_chunks.json"
PARAGRAPH_FILE = "paragraph_chunks.json"

FIXED_OUTPUT = "fixed_embeddings.json"
PARAGRAPH_OUTPUT = "paragraph_embeddings.json"

print("======================================")
print("LOADING EMBEDDING MODEL")
print("======================================")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model loaded successfully.")


def create_embeddings(input_file, output_file, strategy_name):

    with open(input_file, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    print(f"\nCreating embeddings for {strategy_name}...")
    print("Chunks:", len(chunks))

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    output = []

    for chunk, embedding in zip(chunks, embeddings):

        output.append({
            "chunk_id": chunk["chunk_id"],
            "method": chunk["method"],
            "page": chunk["page"],
            "text": chunk["text"],
            "embedding": embedding.tolist()
        })

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(
            output,
            file,
            ensure_ascii=False
        )

    print("Saved:", output_file)


create_embeddings(
    FIXED_FILE,
    FIXED_OUTPUT,
    "Fixed-size"
)

create_embeddings(
    PARAGRAPH_FILE,
    PARAGRAPH_OUTPUT,
    "Paragraph-based"
)


print("\n======================================")
print("EMBEDDINGS COMPLETED")
print("======================================")
print("Created:")
print("-", FIXED_OUTPUT)
print("-", PARAGRAPH_OUTPUT)