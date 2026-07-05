import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Load environment variables
load_dotenv()

def get_document_loader(file_path: str):
    """
    Returns the appropriate LangChain document loader based on file extension.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(file_path)
    elif ext == ".docx":
        return Docx2txtLoader(file_path)
    elif ext == ".txt":
        return TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def ingest_documents(file_paths: list[str], faiss_db_path: str = "faiss_index", openai_api_key: str = None) -> bool:
    """
    Loads files, splits them into chunks (size 1000, overlap 200),
    generates OpenAI embeddings, and indexes them in a local FAISS store.
    """
    if not file_paths:
        return False
    
    documents = []
    for fp in file_paths:
        if not os.path.exists(fp):
            continue
        try:
            loader = get_document_loader(fp)
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading file {fp}: {e}")
            continue

    if not documents:
        return False

    # Text Splitting - Configured to chunk_size=1000 and chunk_overlap=200
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    
    if not chunks:
        return False

    # Embedding and Vector Store Ingestion
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    
    # If FAISS index already exists, load and merge/add; otherwise create new
    if os.path.exists(os.path.join(faiss_db_path, "index.faiss")):
        try:
            db = FAISS.load_local(faiss_db_path, embeddings, allow_dangerous_deserialization=True)
            db.add_documents(chunks)
        except Exception as e:
            print(f"Failed to load existing FAISS index: {e}. Recreating index...")
            db = FAISS.from_documents(chunks, embeddings)
    else:
        db = FAISS.from_documents(chunks, embeddings)
        
    db.save_local(faiss_db_path)
    return True

def get_rag_chain(faiss_db_path: str = "faiss_index", model_name: str = "gpt-4o-mini", openai_api_key: str = None):
    """
    Creates and returns a conversational RAG retrieval chain using LangChain.
    """
    if not os.path.exists(os.path.join(faiss_db_path, "index.faiss")):
        return None

    # Load FAISS index
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = FAISS.load_local(faiss_db_path, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # Initialize LLM
    llm = ChatOpenAI(model=model_name, temperature=0.1, openai_api_key=openai_api_key)

    # 1. Contextualize query
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 2. Answer question using retrieved context
    qa_system_prompt = (
        "You are an expert AI assistant. Answer the user's question using the provided context. "
        "If you do not know the answer or if the context does not contain the answer, "
        "say that you do not know or that the answer is not present in the documents. "
        "Keep your answer structured, clear, and refer to source document names when appropriate.\n\n"
        "Context:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # 3. Create final RAG retrieval chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain
