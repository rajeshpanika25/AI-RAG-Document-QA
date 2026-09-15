import json
import chromadb

FIXED_FILE = "fixed_embeddings.json"
PARAGRAPH_FILE = "paragraph_embeddings.json"

DB_PATH = "chroma_db"


print("======================================")
print("BUILDING / UPDATING CHROMADB")
print("======================================")


# Create persistent ChromaDB
client = chromadb.PersistentClient(path=DB_PATH)


# --------------------------------------
# Delete old collections if they exist
# --------------------------------------

for collection_name in ["fixed_chunks", "paragraph_chunks"]:
    try:
        client.delete_collection(name=collection_name)
        print(f"Deleted old collection: {collection_name}")
    except Exception:
        pass


# --------------------------------------
# Create fresh collections
# --------------------------------------

fixed_collection = client.create_collection(
    name="fixed_chunks"
)

paragraph_collection = client.create_collection(
    name="paragraph_chunks"
)


# --------------------------------------
# Load embeddings
# --------------------------------------

with open(FIXED_FILE, "r", encoding="utf-8") as file:
    fixed_chunks = json.load(file)

with open(PARAGRAPH_FILE, "r", encoding="utf-8") as file:
    paragraph_chunks = json.load(file)


# --------------------------------------
# Add fixed-size chunks
# --------------------------------------

print("\nAdding fixed-size chunks...")

fixed_collection.add(
    ids=[chunk["chunk_id"] for chunk in fixed_chunks],
    embeddings=[chunk["embedding"] for chunk in fixed_chunks],
    documents=[chunk["text"] for chunk in fixed_chunks],
    metadatas=[
        {
            "page": chunk["page"],
            "method": chunk["method"]
        }
        for chunk in fixed_chunks
    ]
)

print("Fixed chunks added:", fixed_collection.count())


# --------------------------------------
# Add paragraph-based chunks
# --------------------------------------

print("\nAdding paragraph-based chunks...")

paragraph_collection.add(
    ids=[chunk["chunk_id"] for chunk in paragraph_chunks],
    embeddings=[chunk["embedding"] for chunk in paragraph_chunks],
    documents=[chunk["text"] for chunk in paragraph_chunks],
    metadatas=[
        {
            "page": chunk["page"],
            "method": chunk["method"]
        }
        for chunk in paragraph_chunks
    ]
)

print("Paragraph chunks added:", paragraph_collection.count())


# --------------------------------------
# Final verification
# --------------------------------------

print("\n======================================")
print("CHROMADB UPDATED SUCCESSFULLY")
print("======================================")

print("Database:", DB_PATH)
print("Fixed collection:", fixed_collection.count())
print("Paragraph collection:", paragraph_collection.count())