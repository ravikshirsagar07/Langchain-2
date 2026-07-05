import os
import shutil
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

# Import backend modules
from rag_backend import ingest_documents, get_rag_chain

# Load default environment variables
load_dotenv()

# App configuration
DATA_DIR = "data"
FAISS_DB_PATH = "faiss_index"

# Create directories if they do not exist
os.makedirs(DATA_DIR, exist_ok=True)

# ----------------- Custom UI Styling -----------------
st.set_page_config(
    page_title="Knowledge QA Assistant",
    page_icon="🤖",
    layout="wide"
)

# Premium CSS customization
st.markdown("""
<style>
/* Custom Fonts */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* Gradient Title */
.header-container {
    padding: 1.5rem 0;
    margin-bottom: 2rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}
.header-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00C6FF 0%, #0072FF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.header-subtitle {
    font-size: 1.1rem;
    color: #8b949e;
}

/* Sidebar Custom Look */
[data-testid="stSidebar"] {
    background-color: #0b0f17;
    border-right: 1px solid #1f2a3d;
}

/* Source Box Styling */
.source-card {
    background-color: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 0.8rem;
}
.source-title {
    font-weight: 600;
    color: #58a6ff;
    font-size: 0.9rem;
    margin-bottom: 0.3rem;
}
.source-text {
    font-size: 0.85rem;
    color: #c9d1d9;
    line-height: 1.4;
}
</style>
""", unsafe_allow_html=True)

# ----------------- App State Management -----------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []
if "api_key_configured" not in st.session_state:
    st.session_state.api_key_configured = bool(os.getenv("OPENAI_API_KEY"))

# Refresh RAG chain if vector db exists and API key is present
def init_rag_chain(model_name):
    if st.session_state.api_key_configured and os.path.exists(os.path.join(FAISS_DB_PATH, "index.faiss")):
        try:
            st.session_state.rag_chain = get_rag_chain(
                faiss_db_path=FAISS_DB_PATH, 
                model_name=model_name
            )
        except Exception as e:
            st.warning(f"Error loading vector index: {e}")
            st.session_state.rag_chain = None

# Helper to format chat history for LangChain
def get_langchain_history():
    history = []
    # Take last 10 messages to manage context window length
    for m in st.session_state.messages[-10:]:
        if m["role"] == "user":
            history.append(HumanMessage(content=m["content"]))
        else:
            history.append(AIMessage(content=m["content"]))
    return history

# ----------------- Sidebar Layout -----------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=80)
    st.title("Settings & Ingestion")
    
    # 1. API Key Check
    api_key_input = os.getenv("OPENAI_API_KEY", "")
    if not st.session_state.api_key_configured:
        st.warning("⚠️ OpenAI API Key is missing.")
        api_key_input = st.text_input("Enter OpenAI API Key:", type="password")
        if api_key_input:
            os.environ["OPENAI_API_KEY"] = api_key_input
            st.session_state.api_key_configured = True
            st.success("API Key set successfully!")
            st.rerun()
    else:
        st.success("🔒 OpenAI API Key Active")

    st.markdown("---")

    # 2. RAG Model Configuration (gpt-4o-mini as default based on "gpt 4.1 mini")
    model_options = {
        "gpt-4o-mini (GPT 4.1 Mini)": "gpt-4o-mini",
        "gpt-4o (Full GPT-4o)": "gpt-4o",
        "gpt-3.5-turbo": "gpt-3.5-turbo"
    }
    selected_model_label = st.selectbox(
        "Select LLM Model:",
        options=list(model_options.keys()),
        index=0
    )
    selected_model = model_options[selected_model_label]
    
    # Initialize RAG Chain if not already done
    if st.session_state.rag_chain is None:
        init_rag_chain(selected_model)

    st.markdown("---")

    # 3. Document Ingestion Section
    st.subheader("📚 Knowledge Base Ingestion")
    uploaded_files = st.file_uploader(
        "Upload Documents:",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True
    )
    
    ingest_clicked = st.button("🚀 Ingest & Index Documents", use_container_width=True)
    
    if ingest_clicked:
        if not st.session_state.api_key_configured:
            st.error("Please configure your OpenAI API Key first!")
        elif not uploaded_files:
            st.error("Please select at least one document to ingest.")
        else:
            with st.spinner("Processing documents... (Loading, Chunking [Size: 1000, Overlap: 200], Embedding, and Indexing)"):
                saved_paths = []
                for file in uploaded_files:
                    path = os.path.join(DATA_DIR, file.name)
                    with open(path, "wb") as f:
                        f.write(file.getvalue())
                    saved_paths.append(path)
                
                success = ingest_documents(saved_paths, FAISS_DB_PATH)
                if success:
                    st.success("Documents successfully indexed!")
                    init_rag_chain(selected_model)
                    st.rerun()
                else:
                    st.error("Document ingestion failed.")

    # 4. View / Reset Section
    st.markdown("---")
    st.subheader("Database Status")
    
    # List ingested documents
    files_in_store = os.listdir(DATA_DIR) if os.path.exists(DATA_DIR) else []
    if files_in_store:
        st.write(f"📁 **Ingested Files ({len(files_in_store)}):**")
        for f in files_in_store:
            st.caption(f"• {f}")
    else:
        st.caption("No files ingested yet.")

    # Reset Button
    if st.button("🗑️ Reset Database", type="primary", use_container_width=True):
        # Clear directories
        if os.path.exists(DATA_DIR):
            shutil.rmtree(DATA_DIR)
            os.makedirs(DATA_DIR, exist_ok=True)
        if os.path.exists(FAISS_DB_PATH):
            shutil.rmtree(FAISS_DB_PATH)
        
        # Clear session state
        st.session_state.rag_chain = None
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.success("Database reset complete!")
        st.rerun()

# ----------------- Main Chat UI -----------------
# Header
st.markdown("""
<div class="header-container">
    <h1 class="header-title">Knowledge QA Assistant</h1>
</div>
""", unsafe_allow_html=True)

# Check if vector index exists
if st.session_state.rag_chain is None:
    st.info("👋 Welcome! To begin, please enter your OpenAI API Key and ingest documents in the sidebar on the left.")
else:
    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat Input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate LLM response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            with st.spinner("Retrieving facts and formulating response..."):
                try:
                    chain = st.session_state.rag_chain
                    history = get_langchain_history()
                    
                    response = chain.invoke({
                        "input": prompt,
                        "chat_history": history
                    })
                    
                    answer = response.get("answer", "")
                    sources = response.get("context", [])
                    
                    response_placeholder.write(answer)
                    
                    # Store message and sources in session state
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.session_state.last_sources = [
                        {
                            "name": os.path.basename(doc.metadata.get("source", "Unknown")),
                            "page": doc.metadata.get("page", None),
                            "content": doc.page_content
                        }
                        for doc in sources
                    ]
                    
                    # Trigger rerun to show sources sidebar/accordion
                    st.rerun()
                    
                except Exception as e:
                    response_placeholder.write(f"⚠️ Error executing query: {e}")

# ----------------- Collapsible Source Document Panel -----------------
if st.session_state.last_sources:
    st.markdown("---")
    with st.expander("🔍 Reference Chunks & Source Documents", expanded=False):
        st.markdown("Here are the retrieved text chunks utilized by the model to generate the response:")
        
        cols = st.columns(len(st.session_state.last_sources))
        for idx, src in enumerate(st.session_state.last_sources):
            with cols[idx]:
                page_info = f" (Page {src['page'] + 1})" if src['page'] is not None else ""
                st.markdown(f"""
                <div class="source-card">
                    <div class="source-title">📄 {src['name']}{page_info}</div>
                    <div class="source-text">"{src['content'][:250]}..."</div>
                </div>
                """, unsafe_allow_html=True)
