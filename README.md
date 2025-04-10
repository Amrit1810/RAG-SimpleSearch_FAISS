# RAG-SimpleSearch_FAISS
🚀 Python RAG toolkit: Index local files (PDF, DOCX, DOC, XLSX, XLS, CSV, TXT) using LangChain & FAISS. Enables semantic search & enhances LLM prompts with relevant context. Simple setup.




---

## 🛠️ Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Amrit1810/RAG-SimpleSearch_FAISS.git
    cd YourProjectName
    ```

2.  **Create Virtual Environment (Recommended):**
    ```bash
    # Linux/macOS
    python -m venv venv
    source venv/bin/activate

    # Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **⚠️ Install System Dependencies (IMPORTANT!):**
    The `unstructured` library needs system packages for handling certain file types reliably (especially `.doc`, complex PDFs, etc.).
    *   **Debian/Ubuntu:**
        ```bash
        sudo apt-get update && sudo apt-get install -y libreoffice tesseract-ocr poppler-utils
        ```
    *   **MacOS:**
        ```bash
        brew install libreoffice tesseract poppler
        ```
    *   **Other OS:** Consult the [unstructured documentation]([https://unstructured-io.github.io/unstructured/installation/full_installation.html](https://docs.unstructured.io/welcome)).
    *   *Failure to install these may result in errors when processing certain files.*

5.  **GPU Support (Optional):**
    *   Requires a working NVIDIA driver and CUDA toolkit.
    *   For GPU-accelerated FAISS search, install `faiss-gpu`:
        ```bash
        pip uninstall faiss-cpu
        pip install faiss-gpu
        ```
    *   Embeddings will automatically try using the GPU if `torch` detects CUDA.

6.  **Add Your Documents:**
    *   Place all your files (PDF, DOCX, DOC, XLSX, XLS, CSV, TXT) directly into the `Documents/` folder.

---

## ▶️ How to Use

1.  **Index Your Documents:**
    Run this script to process files in the `Documents` folder and build/update the search index.
    ```bash
    python create_update_faiss.py
    ```
    *   Creates `faiss_index/` if it doesn't exist.
    *   Loads an existing index and adds *all* documents currently in the `Documents` folder (it doesn't track individual file changes - delete `faiss_index/` to rebuild completely).

2.  **Enhance a Prompt for an LLM:**
    Use this to generate a prompt containing relevant context for a given question.
    ```bash
    python enhance_prompt.py
    ```
    *   *Modify the `original_prompt` variable inside the script's `if __name__ == "__main__":` block to test your questions.*
    *   The output is the full prompt ready to be sent to an LLM.

3.  **Perform Direct Search (Optional Testing):**
    Directly query the index without prompt formatting.
    ```bash
    python search_faiss_index.py
    ```
    *   *Modify the `test_query` variable inside the script's `if __name__ == "__main__":` block to test.*

---

## ⚙️ Configuration

Adjust key parameters in `utils.py`:

*   `DOCS_DIR`: Path to your documents folder.
*   `FAISS_INDEX_DIR`, `FAISS_INDEX_NAME`: Index storage location and name.
*   `EMBEDDING_MODEL_NAME`: Sentence Transformer model (e.g., `all-MiniLM-L6-v2`).
*   `CHUNK_SIZE`, `CHUNK_OVERLAP`: Document splitting parameters.
*   `DEFAULT_SEARCH_K` (in `search_faiss_index.py`): Default number of context chunks to retrieve.

---

## 🧩 Technology Stack

*   **LangChain:** Core framework for RAG components.
*   **Sentence Transformers:** For generating text embeddings.
*   **FAISS:** For efficient vector similarity search.
*   **Unstructured:** For parsing diverse document formats.
*   **PyPDF, python-docx, openpyxl, pandas:** Specific file handling.
*   **Torch:** Deep learning framework (backend for embeddings).

---

## 📖 How It Works (RAG Simplified)

1.  **Indexing:** Files in `Documents/` are loaded, split into text chunks, converted to numerical vectors (embeddings), and stored in a FAISS index along with metadata (like the source filename).
2.  **Retrieval:** Your query is embedded. FAISS finds the text chunks in the index whose embeddings are most similar to your query's embedding.
3.  **Generation Prep:** The text of these retrieved chunks is combined with your original query into a structured prompt, instructing an LLM to answer based on the provided context. (This tool prepares the prompt; sending it to an LLM is the next step).

---

## 💡 Future Ideas

*   Implement smarter index updates (only process new/changed files).
*   Add metadata filtering during search (e.g., by file type, date).
*   Integrate directly with an LLM API (OpenAI, Hugging Face Hub, etc.).
*   Build a simple web UI (Streamlit, Flask).
*   Experiment with different embedding models or vector stores (ChromaDB, Weaviate).

---

## 📜 License

[Specify Your License Here - e.g., MIT License]
