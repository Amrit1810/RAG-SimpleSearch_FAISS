from search_faiss_index import search_index, DEFAULT_SEARCH_K
from utils import logger

def enhance_prompt_with_context(user_prompt: str, k: int = DEFAULT_SEARCH_K) -> str:
    """
    Enhances the user prompt by prepending relevant context retrieved
    from the FAISS index.

    Args:
        user_prompt (str): The original prompt from the user.
        k (int): The number of relevant documents to retrieve for context.
                 Defaults to DEFAULT_SEARCH_K.

    Returns:
        str: The enhanced prompt including the retrieved context, or a message
             indicating no context was found.
    """
    logger.info(f"Enhancing prompt for: '{user_prompt}' using top {k} results.")

    retrieved_docs = search_index(user_prompt, k=k)

    if not retrieved_docs:
        logger.warning("Could not retrieve any context for the prompt.")
        return f"No relevant context was found in the indexed documents for the question.\n\nUser Question: {user_prompt}"

    # Format the retrieved context
    context_parts = []
    logger.info(f"Formatting context from {len(retrieved_docs)} retrieved chunks...")
    for i, doc in enumerate(retrieved_docs):
        # Updated context formatting for simplified metadata
        source_file = doc.metadata.get('source_file', 'Unknown File')
        context_parts.append(f"--- Context {i+1} (Source File: {source_file}) ---\n{doc.page_content}") # Removed tag info

    context_string = "\n\n".join(context_parts)

    # Construct the enhanced prompt
    enhanced_prompt = f"""Please answer the following question based *only* on the provided context below. If the context does not contain the information needed to answer the question, state that clearly.

--- Start of Context ---

{context_string}

--- End of Context ---

User Question: {user_prompt}

Answer:
"""
    logger.info("Prompt successfully enhanced with retrieved context.")
    return enhanced_prompt

# Example Usage:
if __name__ == "__main__":
    original_prompt = "Give me a summary of Phase 2 from the successful report"
    num_results_for_prompt = 7

    logger.info(f"\n--- Example Prompt Enhancement ---")
    logger.info(f"Original Prompt: {original_prompt}")
    logger.info(f"Requesting k={num_results_for_prompt} results for context.")

    final_prompt = enhance_prompt_with_context(original_prompt, k=num_results_for_prompt)

    print("\n--- Enhanced Prompt ---")
    print(final_prompt)