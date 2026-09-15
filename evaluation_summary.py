import csv

# ======================================
# LOAD RETRIEVAL RESULTS
# ======================================

with open(
    "retrieval_evaluation.csv",
    "r",
    encoding="utf-8"
) as file:

    retrieval = list(
        csv.DictReader(file)
    )


# ======================================
# LOAD GENERATION RESULTS
# ======================================

with open(
    "generation_evaluation.csv",
    "r",
    encoding="utf-8"
) as file:

    generation = list(
        csv.DictReader(file)
    )


# ======================================
# RETRIEVAL RESULTS
# ======================================

answerable_retrieval = [
    row for row in retrieval
    if row["answerable"] == "True"
]

fixed_correct = sum(
    row["fixed_retrieval_correct"] == "True"
    for row in answerable_retrieval
)

paragraph_correct = sum(
    row["paragraph_retrieval_correct"] == "True"
    for row in answerable_retrieval
)

total_retrieval = len(
    answerable_retrieval
)

fixed_accuracy = (
    fixed_correct / total_retrieval * 100
)

paragraph_accuracy = (
    paragraph_correct / total_retrieval * 100
)


# ======================================
# GENERATION RESULTS
# ======================================

answerable_generation = [
    row for row in generation
    if row["answerable"] == "True"
]

generation_correct = sum(
    row["generation_correct"] == "True"
    for row in answerable_generation
)

total_generation = len(
    answerable_generation
)

generation_accuracy = (
    generation_correct
    / total_generation
    * 100
)


# ======================================
# UNANSWERABLE TEST
# ======================================

unanswerable = [
    row for row in generation
    if row["answerable"] == "False"
]

unanswerable_correct = sum(
    row["generation_correct"] == "True"
    for row in unanswerable
)


# ======================================
# DISPLAY FINAL RESULTS
# ======================================

print("======================================")
print("FINAL RAG EVALUATION SUMMARY")
print("======================================")

print("\nRETRIEVAL QUALITY")
print("--------------------------------------")

print(
    f"Fixed-size: "
    f"{fixed_correct}/{total_retrieval} "
    f"({fixed_accuracy:.1f}%)"
)

print(
    f"Paragraph-based: "
    f"{paragraph_correct}/{total_retrieval} "
    f"({paragraph_accuracy:.1f}%)"
)

if fixed_accuracy > paragraph_accuracy:

    print("Winner: Fixed-size")

elif paragraph_accuracy > fixed_accuracy:

    print("Winner: Paragraph-based")

else:

    print("Winner: Tie")


print("\nGENERATION QUALITY")
print("--------------------------------------")

print(
    f"Generation accuracy: "
    f"{generation_correct}/{total_generation} "
    f"({generation_accuracy:.1f}%)"
)

print("\nUNANSWERABLE QUESTION")
print("--------------------------------------")

print(
    f"Correctly handled: "
    f"{unanswerable_correct}/"
    f"{len(unanswerable)}"
)

print("\n======================================")
print("INTERPRETATION")
print("======================================")

print(
    f"\nFixed-size chunking achieved "
    f"{fixed_accuracy:.1f}% retrieval accuracy."
)

print(
    f"Paragraph-based chunking achieved "
    f"{paragraph_accuracy:.1f}% retrieval accuracy."
)

print(
    f"\nFixed-size performed "
    f"{fixed_accuracy - paragraph_accuracy:.1f} "
    f"percentage points better."
)

print(
    f"\nGeneration accuracy was "
    f"{generation_accuracy:.1f}% "
    f"when the correct document context "
    f"was provided."
)

print(
    f"\nThe system correctly handled "
    f"{unanswerable_correct}/{len(unanswerable)} "
    f"deliberately unanswerable question(s)."
)

print("\n======================================")
print("SUMMARY COMPLETED")
print("======================================")