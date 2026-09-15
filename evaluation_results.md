# RAG Evaluation Results

## 1. Evaluation Overview

The RAG system was evaluated using a question set containing 51 questions.

- 50 answerable questions
- 1 deliberately unanswerable question
- Two chunking strategies were evaluated:
  - Fixed-size chunking
  - Paragraph-based chunking

Retrieval quality and generation quality were evaluated separately.

---

## 2. Evaluation Methodology

### Retrieval Evaluation

For each answerable question, the system retrieved the top-ranked passage from the vector database.

Retrieval was considered correct when the retrieved passage came from one of the documented source pages associated with the question.

The evaluation compared the two chunking strategies using the same embedding model and retrieval setup.

### Generation Evaluation

Generation was evaluated separately by providing the correct document context to the language model.

The generated answer was compared with the expected answer using predefined keywords.

This separates generation quality from retrieval quality.

### Unanswerable Question

One deliberately unanswerable question was included:

> What was the population of Mars in 2024?

The expected behavior was:

> I don't know based on the provided document.

---

## 3. Retrieval Results

| Chunking Strategy | Correct | Total | Accuracy |
|---|---:|---:|---:|
| Fixed-size | 31 | 50 | 62.0% |
| Paragraph-based | 25 | 50 | 50.0% |

### Retrieval Interpretation

Fixed-size chunking achieved 62.0% retrieval accuracy, while paragraph-based chunking achieved 50.0%.

Fixed-size chunking therefore performed 12 percentage points better on the evaluation question set.

The results indicate that fixed-size chunking provided more effective retrieval for this document and evaluation set.

---

## 4. Generation Results

| Metric | Correct | Total | Accuracy |
|---|---:|---:|---:|
| Generation on answerable questions | 31 | 50 | 62.0% |
| Unanswerable question handling | 1 | 1 | 100% |

### Generation Interpretation

When the correct document context was provided, the system produced answers matching the expected evaluation keywords for 31 of the 50 answerable questions.

This resulted in a generation accuracy of 62.0%.

The generation evaluation is separate from retrieval evaluation, allowing retrieval failures and generation failures to be considered independently.

---

## 5. Unanswerable Question

The evaluation included one deliberately unanswerable question.

| Question | Expected Behavior | Result |
|---|---|---|
| What was the population of Mars in 2024? | I don't know based on the provided document. | Correct |

The system correctly identified that the provided document did not contain the requested information.

Unanswerable-question handling: **1/1 (100%)**.

---

## 6. Chunking Strategy Comparison

| Metric | Fixed-size | Paragraph-based |
|---|---:|---:|
| Retrieval Accuracy | 62.0% | 50.0% |
| Difference | +12 percentage points | — |

### Which Strategy Performed Better?

Fixed-size chunking performed better on the evaluation set.

It achieved 62.0% retrieval accuracy compared with 50.0% for paragraph-based chunking.

The fixed-size strategy was therefore selected as the retrieval strategy for the final RAG system.

---

## 7. Final Evaluation Summary

The evaluation demonstrates that the RAG system can retrieve relevant information from the document and generate answers using document context.

The evaluation used 51 questions, including a deliberately unanswerable question.

The main results were:

- Fixed-size retrieval accuracy: **62.0%**
- Paragraph-based retrieval accuracy: **50.0%**
- Generation accuracy with correct context: **62.0%**
- Unanswerable question handling: **100%**

The comparison provides quantitative evidence for the chunking-strategy decision rather than selecting a strategy without evaluation.

---

## 8. Files Used for Evaluation

The evaluation results are stored in:

- `evaluation_questions.json`
- `retrieval_evaluation.csv`
- `generation_evaluation.csv`
- `rag_evaluation.py`
- `generation_evaluation.py`
- `evaluation_summary.py`

These files allow the evaluation to be reproduced and inspected.