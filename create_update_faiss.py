import os
import time
from langchain_community.vectorstores import FAISS
from utils import (
    load_documents,
    split_documents,
    get_embedding_model,
    logger,
    DOCS_DIR,         # Use the single directory variable
    FAISS_INDEX_DIR,
    FAISS_INDEX_NAME,
)

def create_or_update_faiss_index():
    """
    Loads documents from the 'Documents' folder, creates embeddings,
    and creates/updates a FAISS index.
    """
    start_time = time.time()

    # --- 1. Load Documents ---
    logger.info("Starting document loading phase...")
    # Load documents from the single source directory
    all_docs = load_documents(DOCS_DIR)

    if not all_docs:
        logger.warning(f"No documents found in '{DOCS_DIR}'. Exiting.")
        return

    logger.info(f"Total document sections loaded: {len(all_docs)}")

    # --- 2. Split Documents ---
    logger.info("Starting document splitting phase...")
    split_docs = split_documents(all_docs)

    if not split_docs:
        logger.warning("No text chunks generated after splitting. Exiting.")
        return

    logger.info(f"Total text chunks created: {len(split_docs)}")

    # --- 3. Initialize Embeddings ---
    embeddings = get_embedding_model()

    # --- 4. Create or Update FAISS Index ---
    index_path = os.path.join(FAISS_INDEX_DIR, FAISS_INDEX_NAME)
    faiss_file = index_path + ".faiss"
    pkl_file = index_path + ".pkl"

    # Create the index directory if it doesn't exist
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

    if os.path.exists(faiss_file) and os.path.exists(pkl_file):
        logger.info(f"Existing FAISS index found at '{index_path}'. Loading and adding new documents...")
        try:
            vector_store = FAISS.load_local(
                FAISS_INDEX_DIR,
                embeddings,
                index_name=FAISS_INDEX_NAME,
                allow_dangerous_deserialization=True
            )
            logger.info("Existing index loaded successfully.")

            # Simple approach: Add all currently found documents.
            logger.info(f"Adding {len(split_docs)} chunks to the existing index...")
            vector_store.add_documents(split_docs)
            logger.info("Chunks added to the index.")

            vector_store.save_local(FAISS_INDEX_DIR, index_name=FAISS_INDEX_NAME)
            logger.info(f"FAISS index updated and saved successfully at '{index_path}'.")

        except Exception as e:
            logger.error(f"Error loading or updating existing FAISS index: {e}", exc_info=True)
            logger.info("Attempting to create a new index from scratch...")
            try:
                vector_store = FAISS.from_documents(split_docs, embeddings)
                vector_store.save_local(FAISS_INDEX_DIR, index_name=FAISS_INDEX_NAME)
                logger.info(f"New FAISS index created and saved successfully at '{index_path}'.")
            except Exception as e_create:
                 logger.error(f"Failed to create new FAISS index after error: {e_create}", exc_info=True)

    else:
        logger.info(f"No existing FAISS index found at '{index_path}'. Creating a new one...")
        try:
            vector_store = FAISS.from_documents(split_docs, embeddings)
            vector_store.save_local(FAISS_INDEX_DIR, index_name=FAISS_INDEX_NAME)
            logger.info(f"New FAISS index created and saved successfully at '{index_path}'.")
        except Exception as e:
            logger.error(f"Error creating new FAISS index: {e}", exc_info=True)

    end_time = time.time()
    logger.info(f"Index creation/update process finished in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    create_or_update_faiss_index()