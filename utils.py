import os
import torch
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    CSVLoader,
    TextLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import logging

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Single documents folder
DOCS_DIR = os.path.join(BASE_DIR, "Documents") # Renamed and simplified
FAISS_INDEX_DIR = os.path.join(BASE_DIR, "faiss_index") # Directory for index
FAISS_INDEX_NAME = "docs_index" # Generic index name

# Embedding model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Text Splitting parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Helper Functions ---

def get_device():
    """Checks for CUDA availability and returns the appropriate device."""
    if torch.cuda.is_available():
        logger.info("CUDA (GPU) is available. Using GPU.")
        return "cuda"
    else:
        logger.info("CUDA (GPU) not available. Using CPU.")
        return "cpu"

DEVICE = get_device() # Determine device once

def get_embedding_model():
    """Initializes and returns the HuggingFace embedding model."""
    logger.info(f"Initializing embedding model '{EMBEDDING_MODEL_NAME}' on device '{DEVICE}'...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': DEVICE},
        encode_kwargs={'normalize_embeddings': False}
    )
    logger.info("Embedding model initialized.")
    return embeddings

def _load_single_document(file_path):
    """Loads a single document based on its extension."""
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext in [".docx", ".doc"]:
            loader = UnstructuredWordDocumentLoader(file_path, mode="single", strategy="fast")
        elif ext in [".xlsx", ".xls"]:
            loader = UnstructuredExcelLoader(file_path, mode="elements")
        elif ext == ".csv":
            loader = CSVLoader(file_path, autodetect_encoding=True)
        elif ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            logger.warning(f"Unsupported file type: {ext} for file {file_path}. Skipping.")
            return None
        return loader.load()
    except Exception as e:
        logger.error(f"Error loading document {file_path}: {e}", exc_info=True)
        return None


def load_documents(folder_path): # Removed source_tag parameter
    """Loads all supported documents from the specified folder."""
    documents = []
    logger.info(f"Loading documents from: {folder_path}")
    if not os.path.isdir(folder_path):
        logger.warning(f"Directory not found: {folder_path}. Skipping.")
        return []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            logger.debug(f"Attempting to load: {filename}")
            loaded_docs = _load_single_document(file_path)
            if loaded_docs:
                # Add metadata about the source file
                for doc in loaded_docs:
                    doc.metadata["source_file"] = filename # Store the source filename
                documents.extend(loaded_docs)
                logger.debug(f"Successfully loaded and added metadata for: {filename}")
        else:
             logger.debug(f"Skipping non-file item: {filename}")

    logger.info(f"Loaded {len(documents)} document sections from {folder_path}.")
    return documents

def split_documents(documents):
    """Splits loaded documents into chunks."""
    logger.info(f"Splitting {len(documents)} document sections into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    split_docs = text_splitter.split_documents(documents)
    logger.info(f"Split into {len(split_docs)} chunks.")
    return split_docs