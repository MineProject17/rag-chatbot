import os, glob
from pathlib import Path
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DATA_DIR, VECTOR_STORE_PATH = "data", "vector_store"

def ingest_documents():
    pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {DATA_DIR}/")
        return False
    
    documents = []
    for pdf_file in pdf_files:
        documents.extend(PyPDFLoader(pdf_file).load())
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)
    print(f"✅ Ingestion complete! {len(chunks)} chunks stored.")
    return True

if __name__ == "__main__":
    ingest_documents()
