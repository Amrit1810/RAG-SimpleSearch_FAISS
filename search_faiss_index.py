import os
from langchain_community.vectorstores import FAISS
from utils import (
    get_embedding_model,
    logger,
    FAISS_INDEX_DIR,
    FAISS_INDEX_NAME,
)

# Default number of results (k)
DEFAULT_SEARCH_K = 10 # Adjust as needed

def search_index(query: str, k: int = DEFAULT_SEARCH_K):
    """
    Searches the FAISS index for documents similar to the query.

    Args:
        query (str): The user's search query.
        k (int): The number of top results to retrieve. Defaults to DEFAULT_SEARCH_K.

    Returns:
        list: A list of LangChain Document objects containing the search results,
              or an empty list if the index doesn't exist or an error occurs.
    """
    index_path = os.path.join(FAISS_INDEX_DIR, FAISS_INDEX_NAME)
    faiss_file = index_path + ".faiss"
    pkl_file = index_path + ".pkl"

    if not (os.path.exists(faiss_file) and os.path.exists(pkl_file)):
        logger.error(f"FAISS index '{FAISS_INDEX_NAME}' not found at '{FAISS_INDEX_DIR}'.")
        logger.error("Please run 'create_update_faiss.py' first.")
        return []

    try:
        logger.info(f"Loading FAISS index from '{index_path}' for search...")
        embeddings = get_embedding_model()
        vector_store = FAISS.load_local(
            FAISS_INDEX_DIR,
            embeddings,
            index_name=FAISS_INDEX_NAME,
            allow_dangerous_deserialization=True
        )
        logger.info("Index loaded successfully.")

        logger.info(f"Performing similarity search for query: '{query}' with k={k}")
        results = vector_store.similarity_search(query, k=k)
        logger.info(f"Found {len(results)} relevant document chunks.")
        return results

    except Exception as e:
        logger.error(f"Error during FAISS index search: {e}", exc_info=True)
        return []

# Example Usage:
if __name__ == "__main__":
    test_query = "What information exists about project methodologies?"
    num_results_to_fetch = 8

    logger.info(f"\n--- Example Search ---")
    logger.info(f"Searching for '{test_query}' and requesting k={num_results_to_fetch} results.")

    search_results = search_index(test_query, k=num_results_to_fetch)

    if search_results:
        logger.info(f"\nSearch Results (Top {len(search_results)}):")
        for i, doc in enumerate(search_results):
            print(f"\n--- Result {i+1} ---")
            # Updated print statement for simplified metadata
            source_file = doc.metadata.get('source_file', 'N/A')
            print(f"Source File: {source_file}") # Removed tag info
            content_preview = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
            print(f"Content Preview:\n{content_preview}")
            # print(f"Metadata: {doc.metadata}")
    else:
        logger.info("No search results found or an error occurred.")