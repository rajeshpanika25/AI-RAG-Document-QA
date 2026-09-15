import json
import csv
import chromadb
from sentence_transformers import SentenceTransformer

QUESTIONS_FILE = "evaluation_questions.json"
DB_PATH = "chroma_db"
TOP_K = 1

print("======================================")
print("RAG RETRIEVAL EVALUATION")
print("======================================")

print("\nLoading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded.")

with open(QUESTIONS_FILE, "r", encoding="utf-8") as file:
    questions = json.load(file)

print("Evaluation questions:", len(questions))

chroma_client = chromadb.PersistentClient(path=DB_PATH)

fixed_db = chroma_client.get_collection("fixed_chunks")
paragraph_db = chroma_client.get_collection("paragraph_chunks")

print("Fixed-size chunks:", fixed_db.count())
print("Paragraph-based chunks:", paragraph_db.count())


def retrieve(collection, question):
    query_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K
    )

    return {
        "text": result["documents"][0][0],
        "page": result["metadatas"][0][0]["page"],
        "distance": result["distances"][0][0]
    }


def retrieval_correct(result, source_pages):
    return result["page"] in source_pages


results = []

print("\nRunning retrieval evaluation...")

for item in questions:
    question_id = item["id"]
    question = item["question"]
    source_pages = item.get("source_pages", [])
    answerable = item.get("answerable", True)

    fixed = retrieve(fixed_db, question)
    paragraph = retrieve(paragraph_db, question)

    fixed_correct = False
    paragraph_correct = False

    if answerable:
        fixed_correct = retrieval_correct(
            fixed,
            source_pages
        )

        paragraph_correct = retrieval_correct(
            paragraph,
            source_pages
        )

    results.append({
        "id": question_id,
        "question": question,
        "answerable": answerable,

        "gold_pages": ",".join(
            str(page) for page in source_pages
        ),

        "fixed_page": fixed["page"],
        "fixed_distance": round(fixed["distance"], 4),
        "fixed_retrieval_correct": fixed_correct,

        "paragraph_page": paragraph["page"],
        "paragraph_distance": round(
            paragraph["distance"], 4
        ),
        "paragraph_retrieval_correct": paragraph_correct
    })

    print(
        f"Q{question_id}: "
        f"Fixed={fixed['page']} "
        f"{'✓' if fixed_correct else '✗'} | "
        f"Paragraph={paragraph['page']} "
        f"{'✓' if paragraph_correct else '✗'}"
    )


with open(
    "retrieval_evaluation.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    fieldnames = results[0].keys()

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(results)


answerable_results = [
    r for r in results
    if r["answerable"]
]

fixed_correct_count = sum(
    r["fixed_retrieval_correct"]
    for r in answerable_results
)

paragraph_correct_count = sum(
    r["paragraph_retrieval_correct"]
    for r in answerable_results
)

total_answerable = len(answerable_results)

fixed_accuracy = (
    fixed_correct_count / total_answerable * 100
)

paragraph_accuracy = (
    paragraph_correct_count / total_answerable * 100
)


print("\n======================================")
print("RETRIEVAL RESULTS")
print("======================================")

print(
    f"Fixed-size: "
    f"{fixed_correct_count}/{total_answerable} "
    f"({fixed_accuracy:.1f}%)"
)

print(
    f"Paragraph-based: "
    f"{paragraph_correct_count}/{total_answerable} "
    f"({paragraph_accuracy:.1f}%)"
)

if fixed_accuracy > paragraph_accuracy:
    print("\nRetrieval winner: Fixed-size")
elif paragraph_accuracy > fixed_accuracy:
    print("\nRetrieval winner: Paragraph-based")
else:
    print("\nRetrieval winner: Tie")

print("\nSaved to: retrieval_evaluation.csv")