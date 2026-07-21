# RAG Pipeline for Domain-Specific Question Answering

## Team Members

| S. No. | Name | Student ID |
| ------ | ---- | ---------- |
| 1 | Garv Oberoi | 2024AC05738 |
| 2 | Hari Sharma | 2024AD05357 |
| 3 | Mrityunjay Kumar Dubey | 2024AD05369 |
| 4 | Sreedhar Gundavarapu | 2024AC05410 | 

## Assignment Overview

This repository implements a Retrieval-Augmented Generation (RAG) pipeline for domain-specific document understanding and question answering. The project is developed for Assignment 2B: Retrieval-Augmented Generation (RAG) Pipeline, covering chunking, dense and sparse retrieval, reranking, and tabular RAG.

### Assignment Details
- Assignment Title: Assignment 2B — Retrieval-Augmented Generation (RAG) Pipeline
- Focus Areas:
  - Chunking
  - Dense and Sparse Retrieval
  - Reranking
  - Tabular RAG

### Prerequisite
Complete Assignment 1A or 1B before proceeding. This project reuses the cleaned domain corpus from Assignment 1 Part A as the RAG document store, and the fine-tuned adapter from Assignment 1 Part B is used as the generator component.

## Project Objective

The goal of this project is to build a practical RAG system that can:
- ingest and preprocess domain-specific documents,
- split documents into meaningful chunks,
- retrieve relevant content using multiple retrieval strategies,
- rerank candidate passages for improved relevance,
- generate answers grounded in retrieved evidence.

## Key Components

- Document loading and preprocessing
- Chunking strategies including fixed-size, sliding-window, and semantic chunking
- Embedding and tokenizer utilities
- Dense retrieval support
- Modular project structure for experimentation and testing

## Project Structure

```text
config/              # Configuration and logging setup
scripts/             # Utility scripts for downloading models and resources
src/                 # Core source code for loaders, chunking, retrieval, and models
tests/               # Test scripts for validating components
models/              # Downloaded embedding and tokenizer artifacts
data/                # Corpus and domain documents
```

## Technologies Used

- Python
- Sentence Transformers
- Hugging Face Transformers
- NumPy and Pandas
- NLTK
- PyTorch

## Setup Instructions

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Download the required language resources and models:

```bash
python scripts/download_nltk_resources.py
python scripts/download_embeddings.py
python scripts/download_tokenizer.py
```

## Running the Project

The project includes several test and demonstration scripts under the tests directory. These can be used to validate the core pipeline components.

Example:

```bash
python -m tests.test_chunker_performance
```

Output:
```text
21-07-2026 07:57:21 | INFO | RAGPipeline | Loading HR Policy documents...
Loading weights: 100%|##########| 103/103 [00:00<00:00, 7098.94it/s]
21-07-2026 07:57:21 | INFO | RAGPipeline | Found 6 text files.
Token indices sequence length is longer than the specified maximum sequence length for this model (2089 > 512). Running this sequence through the model will result in indexing errors
21-07-2026 07:57:21 | INFO | RAGPipeline | Loaded CHSQF060-Workplace-Health-and-Safety-Policy-and-Procedures-v1.2.txt (1661 words)
21-07-2026 07:57:21 | INFO | RAGPipeline | Loaded ENVIROFLY-HR-PolicyCode-of-Conduct.txt (5981 words)
21-07-2026 07:57:21 | INFO | RAGPipeline | Loaded hr-policy-remote_work_policy.txt (3616 words)
21-07-2026 07:57:21 | INFO | RAGPipeline | Loaded HR-Policy.txt                  (4466 words)
21-07-2026 07:57:21 | INFO | RAGPipeline | Loaded Leave-and-Holiday-Policy.txt   (1316 words)
21-07-2026 07:57:21 | INFO | RAGPipeline | Loaded travel-expenses-policy-nov-251.txt (9097 words)
21-07-2026 07:57:21 | INFO | RAGPipeline | Successfully loaded 6 documents.
Document: travel-expenses-policy-nov-251.txt, Words: 9097, Characters: 64132
21-07-2026 07:57:21 | INFO | RAGPipeline | Starting Fixed Size Chunking...
21-07-2026 07:57:21 | INFO | RAGPipeline | Processing document: travel-expenses-policy-nov-251.txt
21-07-2026 07:57:21 | INFO | RAGPipeline | Fixed chunking completed. Generated 79 chunks.
21-07-2026 07:57:21 | INFO | RAGPipeline | Starting Sliding Window Chunking...
21-07-2026 07:57:21 | INFO | RAGPipeline | Sliding Window Chunking completed. Generated 88 chunks.
Loading weights: 100%|##########| 103/103 [00:00<00:00, 7327.10it/s]
Embedding generated successfully! Vector shape: (403, 384)
      Strategy  Total Chunks  Avg Chunk Length (Chars)  Std Dev Chunk Length  Broken Sentences (%)
    Fixed-Size            79                124.164557             75.888898             69.620253
Sliding Window            88                123.363636             76.416202             67.045455
      Semantic           282                 32.258865             38.092278              0.000000

```

## Notes

This project is designed as a modular and extensible RAG pipeline that can be further improved with:
- advanced reranking methods,
- hybrid retrieval strategies,
- richer evaluation metrics,
- integration with a full question-answering interface.

## Summary

This repository provides a strong foundation for implementing and evaluating a professional RAG system using a reusable document corpus and a fine-tuned generation adapter.
