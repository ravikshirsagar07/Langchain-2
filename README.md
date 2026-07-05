# Knowledge QA Assistant - Streamlit RAG Application

A modern, high-performance Retrieval-Augmented Generation (RAG) web application built using **Streamlit** (frontend), **LangChain** (orchestrator), and **FAISS** (local vector store).

---

## 1. System Architecture

The application separates concerns between the user interface, backend RAG pipeline orchestration, local databases, and external cognitive APIs:

```mermaid
graph TD
    User([User / Browser])
    
    subgraph Streamlit_App [Streamlit Application (Frontend)]
        UI[User Interface]
        Upload[Upload Documents]
        Chat[Chat Interface]
        History[Chat History State]
        Sources[Source Documents Viewer]
    end

    subgraph LangChain_Backend [LangChain Backend (Orchestration)]
        RC[Retrieval Chain]
        HA_Retriever[History-Aware Retriever]
        Prompt[Prompt Template Builder]
        QA_Chain[QA Stuff Documents Chain]
    end

    subgraph Data_Stores [Data Stores (Local)]
        DS[(Document Store /data)]
        VS[(Vector Store FAISS)]
    end

    subgraph External_APIs [External APIs]
        OpenAI[OpenAI API / LLM]
    end

    User -->|Upload / Chat| UI
    UI --> Upload
    UI --> Chat
    UI --> History
    UI --> Sources

    Upload -->|Save Raw Files| DS
    Upload -->|Send File Chunks| RC
    Chat -->|Send Query & History| RC

    RC --> HA_Retriever
    HA_Retriever -->|Similarity Query| VS
    VS -->|Relevant Chunks| HA_Retriever
    HA_Retriever --> Prompt
    Prompt --> QA_Chain
    QA_Chain -->|Prompt with Context| OpenAI
    OpenAI -->|Answer & Embeddings| QA_Chain
    QA_Chain -->|Answer + Source Citations| UI
```

---

## 2. Core RAG Parameters

The system is configured with the following parameters:
* **LLM Engine**: `gpt-4o-mini` (representing the requested `gpt-4.1-mini` configuration) for fast, cost-efficient, and accurate reasoning.
* **Embeddings**: `text-embedding-3-small` / standard OpenAI Embeddings.
* **Text Chunking**: `RecursiveCharacterTextSplitter` with:
  * **Chunk Size**: `1000` characters.
  * **Chunk Overlap**: `200` characters.
* **Vector Index**: **FAISS** (Facebook AI Similarity Search) running locally for sub-millisecond similarity search.

---

## 3. Dataflows

The system executes two primary workflows: **Document Ingestion** and **Query Generation**.

### Ingestion Flow (Write Path)
1. The user uploads `.pdf`, `.txt`, or `.docx` files through the Streamlit sidebar.
2. The backend stores copies of the raw documents in the local `data/` folder (Document Store).
3. The backend splits the texts into chunks (Size: 1000, Overlap: 200).
4. The chunks are converted into dense vector embeddings using the OpenAI API.
5. The embeddings are stored and saved locally in the `faiss_index/` directory.

### Query Flow (Read Path)
1. The user inputs a message in the chat bar.
2. The user's query and the last 10 turns of chat history are packaged together.
3. LangChain's **History-Aware Retriever** reformulates the query into a standalone question and performs a similarity search against the local FAISS index.
4. The top $K$ relevant text chunks are retrieved and loaded as context.
5. The chunks, standalone question, and chat history are merged into the system prompt template.
6. The prompt is sent to `gpt-4o-mini` to generate a contextually accurate, grounded response.
7. The Streamlit UI displays the response to the user along with a collapsible drawer showing the exact source references.

---

## 4. Setup & Running Instructions

### Prerequisites
* Python 3.9 or higher
* OpenAI API Key

### Installation

1. Clone or copy the project files to your directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables. Create a `.env` file in the root folder (or use the sidebar in the UI to input it manually):
   ```env
   OPENAI_API_KEY=your-actual-api-key-here
   ```

### Running the Web App

Start the Streamlit application:
```bash
streamlit run app.py
```
The app will open automatically in your default browser at `http://localhost:8501`.
