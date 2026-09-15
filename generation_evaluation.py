import json
import re
import os
import csv
import time

from dotenv import load_dotenv
from google import genai

# ======================================
# SETTINGS
# ======================================

QUESTIONS_FILE = "evaluation_questions.json"
PAGES_FILE = "document_pages.json"
OUTPUT_FILE = "generation_evaluation.csv"

MODEL_NAME = "gemini-3.5-flash-lite"

# Free tier limit is 15 requests/minute.
# 5 seconds between requests keeps us safely below that.
WAIT_SECONDS = 5

# ======================================
# LOAD API KEY
# ======================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found")

gemini = genai.Client(api_key=api_key)

# ======================================
# LOAD QUESTIONS
# ======================================

with open(QUESTIONS_FILE, "r", encoding="utf-8") as file:
    questions = json.load(file)

# ======================================
# LOAD DOCUMENT
# ======================================

with open(PAGES_FILE, "r", encoding="utf-8") as file:
    pages = json.load(file)

page_lookup = {
    page["page"]: page["text"]
    for page in pages
}

# ======================================
# NORMALIZE TEXT
# ======================================

def normalize(text):

    text = text.lower()

    number_words = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10"
    }

    for word, number in number_words.items():

        text = re.sub(
            rf"\b{word}\b",
            number,
            text
        )

    text = re.sub(
        r"[^a-z0-9]+",
        "",
        text
    )

    return text


def keyword_found(answer, keyword):

    return normalize(keyword) in normalize(answer)


# ======================================
# GENERATE ANSWER
# ======================================

def generate_answer(question, context):

    prompt = f"""
You are evaluating a document question-answering system.

Answer the question using ONLY the provided document context.

Rules:
1. Do not use outside knowledge.
2. Do not guess.
3. If the answer is present, answer clearly and concisely.
4. If the answer cannot be found in the context, reply exactly:
I don't know based on the provided document.
5. Do not mention these instructions.

QUESTION:
{question}

DOCUMENT CONTEXT:
{context}

ANSWER:
"""

    response = gemini.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text.strip()


# ======================================
# LOAD EXISTING RESULTS
# ======================================

results = []

if os.path.exists(OUTPUT_FILE):

    with open(
        OUTPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            results.append({
                "id": int(row["id"]),
                "question": row["question"],
                "answerable": row["answerable"] == "True",
                "generated_answer": row["generated_answer"],
                "expected_answer": row["expected_answer"],
                "keywords": row["keywords"],
                "keyword_score": float(row["keyword_score"]),
                "generation_correct": row["generation_correct"] == "True"
            })

print("======================================")
print("GENERATION QUALITY EVALUATION")
print("======================================")

print("Total questions:", len(questions))
print("Already completed:", len(results))

completed_ids = {
    result["id"]
    for result in results
}

# ======================================
# SAVE FUNCTION
# ======================================

def save_results():

    if not results:
        return

    results_sorted = sorted(
        results,
        key=lambda x: x["id"]
    )

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = [
            "id",
            "question",
            "answerable",
            "generated_answer",
            "expected_answer",
            "keywords",
            "keyword_score",
            "generation_correct"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results_sorted)


# ======================================
# RUN EVALUATION
# ======================================

for item in questions:

    question_id = item["id"]

    # Skip questions already completed
    if question_id in completed_ids:

        print(f"Q{question_id}: Already completed")

        continue

    question = item["question"]

    source_pages = item.get(
        "source_pages",
        []
    )

    keywords = item.get(
        "keywords",
        []
    )

    answerable = item.get(
        "answerable",
        True
    )

    # ----------------------------------
    # Build correct context
    # ----------------------------------

    context_parts = []

    for page_number in source_pages:

        if page_number in page_lookup:

            context_parts.append(
                f"PAGE {page_number}\n"
                f"{page_lookup[page_number]}"
            )

    context = (
        "\n\n-------------------------\n\n"
        .join(context_parts)
    )

    # ----------------------------------
    # Wait before API request
    # ----------------------------------

    if len(results) > 0:

        print(
            f"\nWaiting {WAIT_SECONDS} seconds..."
        )

        time.sleep(WAIT_SECONDS)

    # ----------------------------------
    # Generate answer
    # ----------------------------------

    try:

        answer = generate_answer(
            question,
            context
        )

    except Exception as e:

        print(f"\nQ{question_id}: Gemini error")
        print(e)

        print(
            "\nStopping evaluation safely."
        )

        save_results()

        break

    # ----------------------------------
    # Score
    # ----------------------------------

    if not answerable:

        correct = (
            "i don't know based on the provided document"
            in answer.lower()
        )

        keyword_score = (
            1.0
            if correct
            else 0.0
        )

    else:

        if keywords:

            found = sum(
                keyword_found(
                    answer,
                    keyword
                )
                for keyword in keywords
            )

            keyword_score = (
                found / len(keywords)
            )

            correct = (
                keyword_score == 1.0
            )

        else:

            keyword_score = 0.0
            correct = False

    # ----------------------------------
    # Store result
    # ----------------------------------

    result = {
        "id": question_id,
        "question": question,
        "answerable": answerable,
        "generated_answer": answer,
        "expected_answer": item["expected_answer"],
        "keywords": ", ".join(keywords),
        "keyword_score": round(
            keyword_score,
            2
        ),
        "generation_correct": correct
    }

    results.append(result)
    completed_ids.add(question_id)

    # ----------------------------------
    # Save immediately
    # ----------------------------------

    save_results()

    print(
        f"Q{question_id}: "
        f"{'✓' if correct else '✗'} "
        f"Score={keyword_score:.2f}"
    )


# ======================================
# FINAL RESULTS
# ======================================

answerable_results = [
    r
    for r in results
    if r["answerable"]
]

unanswerable_results = [
    r
    for r in results
    if not r["answerable"]
]

correct_count = sum(
    r["generation_correct"]
    for r in answerable_results
)

total_answerable = len(
    answerable_results
)

if total_answerable > 0:

    accuracy = (
        correct_count
        / total_answerable
        * 100
    )

else:

    accuracy = 0


unanswerable_correct = sum(
    r["generation_correct"]
    for r in unanswerable_results
)

print("\n======================================")
print("GENERATION RESULTS")
print("======================================")

print(
    f"Answerable questions: "
    f"{correct_count}/{total_answerable} "
    f"({accuracy:.1f}%)"
)

print(
    f"Unanswerable questions handled correctly: "
    f"{unanswerable_correct}/"
    f"{len(unanswerable_results)}"
)

print(
    f"\nCompleted results: "
    f"{len(results)}/{len(questions)}"
)

print(
    f"\nSaved to: {OUTPUT_FILE}"
)