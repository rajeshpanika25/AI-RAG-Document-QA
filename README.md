# Document Q&A RAG System

A Retrieval-Augmented Generation (RAG) based Document Question-Answering system.

The system allows users to ask questions about a 50-page AI research document and receive answers based only on the information retrieved from the document.

## Project Overview

This project implements a complete RAG pipeline:

1. PDF document processing
2. Text extraction
3. Text chunking
4. Text embedding
5. Vector database indexing
6. Relevant passage retrieval
7. Answer generation using Gemini
8. Source/page citation
9. Retrieval evaluation
10. Generation evaluation

## Dataset

The system uses a 50-page subset of the AI Index 2025 report.

The document is:

`AI_Index_2025_50_Page_Clean.pdf`

The corpus contains 50 pages.

## RAG Pipeline

The system follows this workflow:

PDF
↓
Text Extraction
↓
Chunking
↓
Embeddings
↓
ChromaDB
↓
Question
↓
Relevant Passage Retrieval
↓
Gemini
↓
Answer + Source Page

## Chunking Strategies

Two different chunking strategies were implemented and evaluated.

### 1. Fixed-Size Chunking

The document was divided into chunks using a recursive character text splitter.

Configuration:

- Chunk size: 1000 characters
- Chunk overlap: 150 characters

Result:

- 116 chunks

### 2. Paragraph-Based Chunking

The document was divided primarily using paragraph boundaries while preserving page information.

Result:

- 68 chunks

## Chunking Evaluation

The two strategies were evaluated using the same embedding model and retrieval setup.

| Strategy | Correct Retrievals | Total | Accuracy |
|---|---:|---:|---:|
| Fixed-size | 31 | 50 | 62.0% |
| Paragraph-based | 25 | 50 | 50.0% |

Fixed-size chunking achieved 12 percentage points higher retrieval accuracy on the evaluation question set.

Therefore, fixed-size chunking was selected as the final retrieval strategy.

## Embedding Model

The project uses:

`all-MiniLM-L6-v2`

The embeddings are normalized before being stored in the vector database.

## Vector Database

ChromaDB is used to store and retrieve document embeddings.

Two collections were created:

- `fixed_chunks`
- `paragraph_chunks`

This allows both chunking strategies to be evaluated independently.

## Language Model

Gemini is used for answer generation.

The language model receives:

- User question
- Retrieved document passage
- Source page information

The model is instructed to answer using only the retrieved document context.

## Source Citations

Every generated answer includes the source page number.

The system also displays the retrieved passage under:

`SOURCES USED`

This allows the user to verify where the answer came from.

## Handling Unanswerable Questions

The system is designed not to guess when relevant information cannot be retrieved.

For example:

Question:

"What was the population of Mars in 2024?"

Expected response:

`I don't know based on the provided document.`

The evaluation included one deliberately unanswerable question.

Result:

- Correctly handled: 1/1
- Accuracy: 100%

## Evaluation

The evaluation dataset contains:

- 51 questions
- 50 answerable questions
- 1 deliberately unanswerable question

Retrieval and generation were evaluated separately.

### Retrieval Quality

| Strategy | Accuracy |
|---|---:|
| Fixed-size | 62.0% |
| Paragraph-based | 50.0% |

### Generation Quality

Generation was evaluated using the correct document context.

Result:

`31/50 = 62.0%`

### Unanswerable Question

Result:

`1/1 = 100%`

## Evaluation Files

The evaluation can be reproduced using:

- `evaluation_questions.json`
- `rag_evaluation.py`
- `retrieval_evaluation.csv`
- `generation_evaluation.py`
- `generation_evaluation.csv`
- `evaluation_summary.py`
- `evaluation_results.md`

## Project Files

### Main RAG files

- `extract_text.py` — extracts text from the PDF
- `chunking.py` — creates the two chunking strategies
- `embedding.py` — generates embeddings
- `vector_db.py` — creates ChromaDB collections
- `rag.py` — runs the Document Q&A system

### Evaluation files

- `evaluation_questions.json` — question and answer dataset
- `rag_evaluation.py` — retrieval evaluation
- `generation_evaluation.py` — generation evaluation
- `retrieval_evaluation.csv` — retrieval results
- `generation_evaluation.csv` — generation results
- `evaluation_summary.py` — final evaluation summary
- `evaluation_results.md` — written evaluation report

## Technologies Used

- Python
- PyPDF
- LangChain Text Splitters
- Sentence Transformers
- ChromaDB
- Google Gemini
- JSON
- CSV

## How to Run

First activate the virtual environment.

Then run the pipeline in this order:

```text
python extract_text.py
python chunking.py
python embedding.py
python vector_db.py
python rag.py



## For Evaluation

```text
python rag_evaluation.py
python generation_evaluation.py
python evaluation_summary.py






### Important


```text
python extract_text.py
python chunking.py
python embedding.py
python vector_db.py
python rag.py