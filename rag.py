import os
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai


# ======================================
# SETUP
# ======================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found")

MODEL_NAME = "gemini-3.5-flash-lite"

TOP_K = 1
THRESHOLD = 0.75


# ======================================
# LOAD EMBEDDING MODEL
# ======================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# ======================================
# CONNECT TO CHROMADB
# ======================================

chroma_client = chromadb.PersistentClient(
    path="chroma_db"
)

fixed_db = chroma_client.get_collection(
    "fixed_chunks"
)

paragraph_db = chroma_client.get_collection(
    "paragraph_chunks"
)


# ======================================
# CONNECT TO GEMINI
# ======================================

gemini = genai.Client(
    api_key=api_key
)


# ======================================
# RETRIEVAL
# ======================================

def retrieve(collection, question):

    query_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K
    )

    passages = []

    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):

        if distance <= THRESHOLD:

            passages.append({
                "text": document,
                "page": metadata["page"],
                "distance": distance
            })

    return passages


# ======================================
# GENERATE ANSWER
# ======================================

def generate_answer(question, passages):

    context = ""

    for i, passage in enumerate(passages, 1):

        context += f"""
PASSAGE {i}
PAGE: {passage['page']}

{passage['text']}

-------------------------
"""

    prompt = f"""
You are a document question-answering assistant.

Answer the question using ONLY the information provided in the passages.

IMPORTANT RULES:
1. Do not use outside knowledge.
2. Do not guess.
3. If the answer is clearly present in a passage, give that answer.
4. If the answer is not present or cannot be determined from the passages, reply exactly:
I don't know based on the provided document.
5. Keep the answer concise.
6. When giving an answer, include the source page number.

QUESTION:
{question}

PASSAGES:
{context}

ANSWER:
"""

    response = gemini.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text.strip()


# ======================================
# DOCUMENT Q&A SYSTEM
# ======================================

print("\n======================================")
print("DOCUMENT Q&A SYSTEM")
print("======================================")

print("\n1. Fixed-size")
print("2. Paragraph-based")

choice = input("\nChoose strategy: ").strip()


if choice == "1":

    collection = fixed_db
    strategy = "Fixed-size"

elif choice == "2":

    collection = paragraph_db
    strategy = "Paragraph-based"

else:

    print("Invalid choice.")
    exit()


print("\nUsing:", strategy)
print("Type 'exit' to stop.")


# ======================================
# QUESTION LOOP
# ======================================

while True:

    question = input("\nQuestion: ").strip()

    if question.lower() in ["exit", "quit", "stop"]:

        print("Goodbye!")
        break

    if not question:
        continue


    # ----------------------------------
    # RETRIEVE
    # ----------------------------------

    passages = retrieve(
        collection,
        question
    )


    # ----------------------------------
    # NO RELEVANT INFORMATION
    # ----------------------------------

    if not passages:

        print("\n======================================")
        print("ANSWER")
        print("======================================")

        print(
            "I don't know based on the provided document."
        )

        print("\n======================================")
        print("SOURCES USED")
        print("======================================")

        print("No relevant passages found.")

        continue


    # ----------------------------------
    # GENERATE ANSWER
    # ----------------------------------

    try:

        answer = generate_answer(
            question,
            passages
        )

    except Exception as e:

        print("\nGemini error:")
        print(e)

        continue


    # ----------------------------------
    # DISPLAY ANSWER
    # ----------------------------------

    print("\n======================================")
    print("ANSWER")
    print("======================================")

    print(answer)


    # ----------------------------------
    # DISPLAY SOURCES
    # ----------------------------------

    print("\n======================================")
    print("SOURCES USED")
    print("======================================")

    for i, passage in enumerate(
        passages,
        1
    ):

        print(
            f"\nPassage {i} | "
            f"Page {passage['page']} | "
            f"Distance {passage['distance']:.4f}"
        )

        print("--------------------------------------")

        print(
            passage["text"]
        )