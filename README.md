```markdown
# DocChat 🐥

DocChat is an advanced, multi-agent Retrieval-Augmented Generation (RAG) system powered by LangGraph, Docling, and Gradio. It allows users to upload documents (PDFs, Word docs, Markdown, text files) or select from pre-loaded technical examples to query and receive fact-checked, verified answers.

## 🚀 Key Features

* **Multi-Agent Pipeline (LangGraph):** Built using an orchestrated workflow involving specialized agents:
  * **Relevance Checker:** Validates whether the user's question can actually be answered by the retrieved document chunks before generation.
  * **Research Agent:** Synthesizes precise answers from the retrieved context.
  * **Verification Agent:** Cross-checks the generated answer against the source documents to verify factual support and eliminate hallucinations.
* **Robust Document Processing:** Leverages Docling and RapidOCR for clean extraction and hybrid vector retrieval.
* **Interactive UI (Gradio):** A modern, responsive dark-mode interface complete with custom styling, document upload handlers, and example quick-loaders.
* **Automated Housekeeping:** Automatically cleans up vector databases (`chroma_db/`) and document caches upon server shutdown.

---

## 🛠️ Project Structure

```text
DocChat-multi-agent/
├── agents/                 # LangGraph workflow and agent definitions (relevance, research, verification)
├── config/                 # Configuration and constants
├── document_processor/     # PDF/Doc handlers and OCR integration
├── examples/               # Sample PDF technical reports for quick testing
├── retriever/              # Hybrid retriever builder
├── utils/                  # Logging and helper utilities
├── app.py                  # Main Gradio application entry point
├── pyproject.toml          # Project dependencies
└── uv.lock                 # Dependency lockfile

```

---

## 📦 Prerequisites & Installation

This project uses **`uv`** for lightning-fast Python package management.

1. **Clone the repository:**
```bash
git clone [https://github.com/YOUR-USERNAME/DocChat-multi-agent.git](https://github.com/YOUR-USERNAME/DocChat-multi-agent.git)
cd DocChat-multi-agent

```


2. **Install dependencies:**
```bash
uv sync

```


3. **Set up your Environment Variables:**
Create a `.env` file in the root directory and add your necessary API keys (e.g., Hugging Face token or LLM provider keys if required by your pipeline).

---

## 🚀 Running the Application

Start the local Gradio development server using `uv`:

```bash
uv run app.py

```

Open your browser and navigate to: **`http://127.0.0.1:5000`**

---

## 📜 License

This project is open-source and available under the [MIT License](https://www.google.com/search?q=LICENSE).

```

```