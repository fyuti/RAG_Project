# ====== INSTALL REQUIRED PACKAGES ======
!pip install -q langchain langchain-community sentence-transformers faiss-cpu transformers accelerate

# ====== IMPORTS ======
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ====== STEP 1: LOAD PDF ======
pdf_path = "Practice Makes Perfect Spanish Verb Tenses, Second Edition.pdf"  # change if needed
loader = PyPDFLoader(pdf_path)
docs = loader.load()
print(f"Loaded {len(docs)} pages from PDF.")

# ====== STEP 2: SPLIT INTO CHUNKS ======
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"Total chunks created: {len(chunks)}")

# ====== STEP 3: EMBEDDINGS + FAISS VECTOR STORE ======
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # return top 3 chunks

# ====== STEP 4: LOAD LOCAL LLM (FLAN-T5) ======
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
llm_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=256)

# ====== STEP 5: SAFE RAG QUERY FUNCTION ======
def rag_query(question, k=3, input_budget=480, max_new_tokens=128):
    docs = retriever.get_relevant_documents(question)
    if not docs:
        return "No relevant context found in the PDF."
Run via Google Colab

Open your notebook in Colab.

Run all cells.

When you run the Gradio iface.launch() cell, Colab will give you a temporary public URL
    header = "Use the context to answer concisely. If unknown, say you don't know.\n\nContext:\n"
    tail = f"\n\nQuestion: {question}\nAnswer:"
    
    # pack chunks safely under token budget
    parts = []
    for d in docs[:k]:
        cand = header + "\n\n".join(parts + [d.page_content]) + tail
        if len(tokenizer.encode(cand)) > input_budget:
            break
        parts.append(d.page_content)
        
    prompt = header + "\n\n".join(parts) + tail
    return llm_pipeline(prompt, truncation=True, max_new_tokens=max_new_tokens)[0]["generated_text"]

# ====== EXAMPLE USAGE ======
question = "What are the three main types of Spanish verbs?"
answer = rag_query(question)
print("Q:", question)
print("A:", answer)
# ====== INSTALL GRADIO ======
!pip install -q gradio

# ====== IMPORT ======
import gradio as gr

# ====== WRAP RAG FUNCTION FOR GRADIO ======
def answer_pdf(question):
    return rag_query(question)

# ====== CREATE GRADIO INTERFACE ======
iface = gr.Interface(
    fn=answer_pdf,
    inputs=gr.Textbox(lines=2, placeholder="Ask a question about your PDF..."),
    outputs=gr.Textbox(label="Answer"),
    title="PDF Question Answering",
    description="Ask questions about your uploaded PDF. Fully offline using local embeddings and FLAN-T5."
)

# ====== LAUNCH INTERFACE ======
iface.launch()

# =========Limitations / Known Issues==========

Small LLM: FLAN-T5 small is lightweight and may give incomplete or inaccurate answers for complex queries.

Context limitation: Only the top chunks returned by FAISS are considered — some relevant information may be missed in very large PDFs.

PDF-only support: Currently supports PDFs only; other file types are not implemented.

FAISS rebuild: Index is rebuilt each time the notebook runs; saving/loading the vector store for faster startup is not yet implemented.

Temporary URLs in Colab: Gradio URLs generated in Colab are temporary; they expire when the session ends.
