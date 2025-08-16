# RAG_Project
This is a demo for a RAG AI for uploading PDF files
# PDF RAG Question Answering

## Description
This project is a **Naive Retrieval-Augmented Generation (RAG) system** that answers user questions based on custom PDFs.  
It demonstrates a complete pipeline from **document ingestion → chunking → embedding → vector store → retrieval → answer generation**.

**Tech Stack:**
- **LangChain + LangChain Community**: Document loaders, text splitting, vector store integration
- **FAISS**: Vector database for embeddings and retrieval
- **HuggingFace Embeddings**: `all-MiniLM-L6-v2` for dense text embeddings
- **FLAN-T5 (small)**: Local LLM for answer generation
- **Gradio**: Interactive web app interface
- **Google Colab**: Runs fully offline; no API keys required

---

## Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/pdf-rag-project.git
cd pdf-rag-project
pip install -r requirements.txt
Open the Colab notebook

File: rag_pdf_colab.ipynb

Upload your PDF

Replace the pdf_path variable in the notebook with your PDF file path.

Run all cells

This will load the PDF, split into chunks, create embeddings, build FAISS index, and start Gradio.

Use the Gradio interface

Type your questions about the PDF and get grounded answers.

Example Queries and Answers
Question	Example Answer
What are the three main types of Spanish verbs?	Regular, irregular, and stem-changing verbs.
How do you conjugate the verb "comer" in past tense?	Example: yo comí, tú comiste, él/ella comió...
What is the rule for stem-changing verbs in the present tense?	The vowel in the stem changes in all forms except nosotros and vosotros.
Limitations / Known Issues

Small LLM: FLAN-T5 small is lightweight and may give incomplete or inaccurate answers for complex queries.

Context limitation: Only the top chunks returned by FAISS are considered — some relevant information may be missed in very large PDFs.

PDF-only support: Currently supports PDFs only; other file types are not implemented.

FAISS rebuild: Index is rebuilt each time the notebook runs; saving/loading the vector store for faster startup is not yet implemented.

Temporary URLs in Colab: Gradio URLs generated in Colab are temporary; they expire when the session ends.
