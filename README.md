# ====== INSTALL REQUIRED PACKAGES ======
!pip install -q langchain langchain-community sentence-transformers faiss-cpu transformers accelerate gradio pypdf

# ====== IMPORTS ======
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import gradio as gr

# ====== STEP 1: EMBEDDINGS + MODEL ======
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
llm_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=256)

# ====== GLOBAL RETRIEVER ======
retriever = None

# ====== STEP 2: PROCESS PDF FUNCTION ======
def process_pdf(pdf_path):
    global retriever
    if pdf_path is None:
        return "⚠️ Please upload a PDF file first."

    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        return f"✅ Processed {len(chunks)} chunks from {pdf_path}. Now you can ask questions!"
    except Exception as e:
        return f"❌ Error while processing PDF: {str(e)}"

# ====== STEP 3: RAG QUERY FUNCTION ======
def rag_query(question, k=3, input_budget=480, max_new_tokens=128):
    global retriever
    if retriever is None:
        return "⚠️ Please upload and process a PDF first."

    docs = retriever.get_relevant_documents(question)
    if not docs:
        return "No relevant context found in the PDF."

    header = "Use the context to answer concisely. If unknown, say you don't know.\n\nContext:\n"
    tail = f"\n\nQuestion: {question}\nAnswer:"

    # Pack chunks safely under token budget
    parts = []
    for d in docs[:k]:
        cand = header + "\n\n".join(parts + [d.page_content]) + tail
        if len(tokenizer.encode(cand)) > input_budget:
            break
        parts.append(d.page_content)

    prompt = header + "\n\n".join(parts) + tail
    return llm_pipeline(prompt, truncation=True, max_new_tokens=max_new_tokens)[0]["generated_text"]

# ====== STEP 4: GRADIO INTERFACE ======
with gr.Blocks() as demo:
    gr.Markdown("# 📘 PDF Question Answering (Naive RAG)\nUpload a PDF and ask questions based on its content.")

    # Upload PDF and process button
    with gr.Row():
        pdf_input = gr.File(label="Upload PDF", type="filepath")  # must use filepath
        process_button = gr.Button("Process PDF")
    status_output = gr.Textbox(label="Processing Status", interactive=False)

    # Question input and answer output
    question_input = gr.Textbox(lines=2, placeholder="Ask a question about your PDF...")
    answer_output = gr.Textbox(label="Answer")
    get_answer_btn = gr.Button("Get Answer")  # ✅ new button to trigger query

    # ====== EVENT BINDINGS ======
    process_button.click(process_pdf, inputs=pdf_input, outputs=status_output)
    get_answer_btn.click(rag_query, inputs=question_input, outputs=answer_output)

demo.launch()
