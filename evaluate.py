import json
import csv
import chromadb
from sentence_transformers import SentenceTransformer


# ======================================
# SETUP
# ======================================

print("Loading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# ======================================
# LOAD QUESTIONS
# ======================================

with open(
    "evaluation_questions.json",
    "r",
    encoding="utf-8"
) as file:
    questions = json.load(file)


# ======================================
# CONNECT TO CHROMADB
# ======================================

client = chromadb.PersistentClient(
    path="chroma_db"
)

fixed_db = client.get_collection(
    "fixed_chunks"
)

paragraph_db = client.get_collection(
    "paragraph_chunks"
)


# ======================================
# RETRIEVAL FUNCTION
# ======================================

def retrieve(collection, question):

    embedding = model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    result = collection.query(
        query_embeddings=[embedding],
        n_results=1
    )

    document = result["documents"][0][0]
    metadata = result["metadatas"][0][0]
    distance = result["distances"][0][0]

    return {
        "text": document,
        "page": metadata["page"],
        "distance": distance
    }


# ======================================
# EVALUATE BOTH STRATEGIES
# ======================================

results = []


print("\n======================================")
print("CHUNKING STRATEGY EVALUATION")
print("======================================")


for index, item in enumerate(
    questions,
    start=1
):

    question = item["question"]

    print(
        f"\nQuestion {index}/{len(questions)}:"
    )

    print(question)


    # ----------------------------------
    # Fixed-size
    # ----------------------------------

    fixed = retrieve(
        fixed_db,
        question
    )


    # ----------------------------------
    # Paragraph-based
    # ----------------------------------

    paragraph = retrieve(
        paragraph_db,
        question
    )


    # ----------------------------------
    # Compare distances
    # ----------------------------------

    if fixed["distance"] < paragraph["distance"]:

        winner = "Fixed-size"

    elif paragraph["distance"] < fixed["distance"]:

        winner = "Paragraph-based"

    else:

        winner = "Tie"


    print(
        f"Fixed-size distance: "
        f"{fixed['distance']:.4f}"
    )

    print(
        f"Paragraph distance: "
        f"{paragraph['distance']:.4f}"
    )

    print(
        f"Winner: {winner}"
    )


    results.append({

        "question": question,

        "fixed_distance":
            fixed["distance"],

        "fixed_page":
            fixed["page"],

        "paragraph_distance":
            paragraph["distance"],

        "paragraph_page":
            paragraph["page"],

        "winner":
            winner
    })


# ======================================
# SAVE RESULTS
# ======================================

with open(
    "chunking_evaluation.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "question",
            "fixed_distance",
            "fixed_page",
            "paragraph_distance",
            "paragraph_page",
            "winner"
        ]
    )

    writer.writeheader()

    writer.writerows(results)


# ======================================
# SUMMARY
# ======================================

fixed_wins = sum(
    1
    for result in results
    if result["winner"] == "Fixed-size"
)

paragraph_wins = sum(
    1
    for result in results
    if result["winner"] == "Paragraph-based"
)

ties = sum(
    1
    for result in results
    if result["winner"] == "Tie"
)


total = len(results)


print("\n======================================")
print("FINAL RESULTS")
print("======================================")

print(
    f"Total questions: {total}"
)

print(
    f"Fixed-size wins: "
    f"{fixed_wins}"
)

print(
    f"Paragraph-based wins: "
    f"{paragraph_wins}"
)

print(
    f"Ties: {ties}"
)


if fixed_wins > paragraph_wins:

    print("\nOverall winner: Fixed-size")

elif paragraph_wins > fixed_wins:

    print("\nOverall winner: Paragraph-based")

else:

    print("\nOverall winner: Tie")


print("\nResults saved to:")
print("chunking_evaluation.csv")