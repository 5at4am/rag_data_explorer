# 📊 Dataset Chat (RAG + Analytics Hybrid Engine)

An advanced, feature-rich conversational data assistant that lets you upload datasets and ask questions in plain natural language. The system dynamically combines **Semantic Vector Search (RAG)**, **Precise Pandas Querying**, and a **Dynamic Visualization Engine** to give you highly accurate statistical answers, visual trends, and semantic insights.

This project showcases three distinct implementation tiers:
1. **🛡️ Advanced Hybrid Engine (Streamlit)**: High-fidelity, modular Python implementation leveraging LangChain, vector store embeddings, Pandas routing, and automated visualization plotting.
2. **⚡ Simple LLM Chat (Streamlit)**: Lightweight, direct Google Gemini SDK integration for flat-context CSV chatting.
3. **🌐 Zero-Backend Browser Client (HTML/JS)**: A beautifully styled, single-page offline application with rule-based client-side parsing and analysis.

---

## 🌟 Key Features

### 1. Multi-Format Loading & Validation (`app/loaders/`)
- Supports **CSV**, **Excel (`.xlsx`)**, **JSON (Standard & Line-delimited)**, and **Parquet** files.
- Robust encoding fallback (tries UTF-8, Latin-1, CP1252, etc.) for resilient CSV reading.
- Validates file sizes, non-empty files, and system schemas dynamically before ingestion.

### 2. Deep Data Profiling (`app/profiling/`)
- Automatically generates comprehensive dataset stats on import.
- Infers granular semantic column types (`numeric`, `categorical`, `datetime`, `text`, `boolean`).
- Calculates min, max, mean, median, standard deviation, top frequent categories, null percentages, duplicate rates, and memory usage.
- Computes mathematical correlation matrices for numeric variables.

### 3. Smart Hybrid Routing Engine (`app/services/hybrid_engine.py`)
- **Pandas Mode**: Automatically detects statistical/quantitative keywords (e.g., *average*, *sum*, *total*, *how many*, *greater than*, *lowest*) and runs lightning-fast local Pandas calculations for **100% numerical precision** (avoiding LLM math hallucinations).
- **RAG Mode**: For semantic or complex text queries, falls back to a Vector-Store-augmented Retrieval-QA pipeline.

### 4. Semantic RAG Pipeline (`app/rag/`)
- **Chunking**: Custom-tuned `RecursiveCharacterTextSplitter` chunking strategy.
- **Embeddings**: Plug-and-play support for HuggingFace Transformers, OpenAI, and Ollama.
- **Vector Stores**: Supports local high-performance **FAISS** index or **ChromaDB** with persistence capability.

### 5. Automated Visualization Engine (`app/services/viz_service.py`)
- Generates beautiful Seaborn and Matplotlib charts purely from natural language queries (e.g., *"Show a histogram of prices"*, *"Plot age vs income"*).
- Supports:
  - **Histograms & Distributions**
  - **Categorical Bar Charts**
  - **Distribution Pie Charts**
  - **Scatter Plots**
  - **Time-series Line Charts**
  - **Box Plots** (with IQR & Outlier markers)
  - **Correlation Heatmaps**
- Encodes output plots directly to base64 PNGs and renders them smoothly in Streamlit.

### 6. Beautiful Client-Side Only App (`index.html`)
- Completely standalone browser application featuring modern Glassmorphism aesthetics with a full Light/Dark mode.
- Custom local CSV parser, stats engine, and rule-based SQL-like querying system executing entirely within your browser sandboxed environment—no server, internet, or API keys required!

---

## 📂 Directory Structure

```text
RAG_project/
├── app.py                # Simple CSV chat entry point (direct Google GenAI SDK)
├── utils.py              # Core utilities for the simple chat app
├── index.html            # Standalone zero-backend client-side CSV browser app
├── requirements.txt      # Comprehensive python package dependencies
├── run.bat               # Windows batch script to install and launch the app
├── .env.example          # Environment variables configuration template
├── uploads/              # Transient storage for uploaded dataset files
├── data/                 # Local directory for vector stores and persisted cache
├── app/                  # Main Advanced Modular Application
│   ├── main.py           # Streamlit Web App entry point
│   ├── chat/             # Memory and chat history orchestrator
│   ├── config/           # Pydantic Settings and environment configuration
│   ├── loaders/          # File loaders, decoders, and validation services
│   ├── memory/           # Conversation memory & context formatting
│   ├── profiling/        # Statistical and semantic column profiler
│   ├── prompts/          # LangChain Chat prompt templates & hallucination guards
│   ├── rag/              # Document creator, chunker, embeddings, and vector stores
│   └── services/         # Routing Engine, Summary Service, and Visualization Service
└── tests/                # Testing suite directory
```

---

## 🛠️ Installation & Setup

### Prerequisites
- **Python 3.10+**
- (Optional) **Google Gemini API Key** (Recommended), **OpenAI API Key**, or a running local **Ollama** server.

### Quick Start (Windows)
1. Double-click the **`run.bat`** script. This will automatically install dependencies from `requirements.txt` and boot up the advanced application.

### Manual Setup
1. **Clone the repository** and navigate to the project directory:
   ```bash
   cd RAG_project
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   
   # Windows PowerShell
   .venv\Scripts\Activate.ps1
   # Windows CMD
   .venv\Scripts\activate.bat
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   Copy `.env.example` to `.env` and configure your API keys:
   ```bash
   cp .env.example .env
   ```
   Edit `.env`:
   ```env
   # API Keys
   GOOGLE_API_KEY=your-gemini-api-key-here
   OPENAI_API_KEY=your-openai-api-key-here

   # Provider Settings
   LLM_PROVIDER=gemini
   LLM_MODEL=gemini-2.0-flash-lite

   # Embedding & Chunking Configuration
   EMBEDDING_PROVIDER=huggingface
   EMBEDDING_MODEL=all-MiniLM-L6-v2
   VECTOR_STORE=faiss
   CHUNK_SIZE=512
   CHUNK_OVERLAP=64
   ```

---

## 🚀 Running the Apps

### Tier 1: Advanced Hybrid RAG Engine (Streamlit)
This is the full modular application incorporating the Hybrid Pandas Router, Document Embeddings, and the Visualizer:
```bash
streamlit run app/main.py
```

### Tier 2: Simple CSV Chat (Streamlit)
For a straightforward, single-file chat experience with light memory footprint:
```bash
streamlit run app.py
```

### Tier 3: Static Browser App (HTML/JS)
To run the lightweight visual dashboard without Python:
- Simply double-click **`index.html`** to open it in any browser!

---

## 💬 Example Queries to Try

Once your dataset is uploaded, try asking:
- **Numerical queries (Routed to Pandas)**:
  - *"How many rows are in the dataset?"*
  - *"What is the average of [Column Name]?"*
  - *"What is the maximum [Column Name] where [Other Column] is greater than 10?"*
- **Visual queries (Routed to Visualization Engine)**:
  - *"Plot a histogram of [Numeric Column]"*
  - *"Show a correlation heatmap"*
  - *"Draw a boxplot of [Numeric Column]"*
  - *"Generate a pie chart for [Categorical Column]"*
- **Semantic queries (Routed to RAG Engine)**:
  - *"Summarize the descriptions in the [Text Column]"*
  - *"Can you extract the main themes from the comments?"*
