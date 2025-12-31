import os, sys
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found. Add it to .env")
    sys.exit(1)

from langchain_google_genai import GoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🤖 RAG Chatbot")
st.markdown("Ask questions about your documents.")

@st.cache_resource
def load_rag_system():
    if not os.path.exists("vector_store"):
        st.error("Run `python ingest.py` first")
        st.stop()
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vector_store = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
    llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY, temperature=0.7)
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever(search_kwargs={"k": 3}), return_source_documents=True)

qa_chain = load_rag_system()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_query = st.chat_input("Ask a question...")
if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = qa_chain.invoke({"query": user_query})
            answer = result["result"]
            st.markdown(answer)
            if result.get("source_documents"):
                st.markdown("**Sources:**")
                for doc in result["source_documents"]:
                    st.markdown(f"- {doc.metadata.get('source', 'Unknown')}")
    st.session_state.messages.append({"role": "assistant", "content": answer})
