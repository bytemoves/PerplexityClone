import google.generativeai as genai
import os
import pprint # For pretty printing the output

# --- Configuration ---
# Best practice: Store your API key as an environment variable
# (e.g., export GOOGLE_API_KEY="YOUR_API_KEY")
# Or replace "YOUR_API_KEY" directly below (less secure)
try:
    # Attempt to get the API key from an environment variable
    api_key = os.environ["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
except KeyError:
    print("Error: GEMINI_API_KEY environment variable not set.")
    print("Please set the environment variable or replace os.environ[...] with your actual key.")
    # You could hardcode it here for testing, but it's not recommended for production:
    # genai.configure(api_key="YOUR_API_KEY")
    exit() # Exit if the key isn't configured

# --- Choose the Embedding Model ---
# 'models/embedding-001' is the standard model as of early 2024.
# Check the Google AI documentation for the latest available models.
model_name = "models/embedding-001"

# --- Text(s) to Embed ---
text_to_embed_single = "What is the meaning of life?"
texts_to_embed_batch = [
    "How does a car engine work?",
    "Explain the theory of relativity.",
    "Recipe for chocolate chip cookies."
]

# --- Generate Embedding for a Single Text ---
print(f"--- Embedding for single text: '{text_to_embed_single}' ---")
try:
    # Task type can be specified for potentially better results depending on the use case.
    # Common task types:
    # RETRIEVAL_QUERY: The text is a query for retrieval.
    # RETRIEVAL_DOCUMENT: The text is a document to be retrieved.
    # SEMANTIC_SIMILARITY: The text will be used for semantic similarity comparison.
    # CLASSIFICATION: The text will be used for classification.
    # CLUSTERING: The text will be used for clustering.
    # If unsure, you can omit task_type or use a general one like SEMANTIC_SIMILARITY.

    result_single = genai.embed_content(
        model=model_name,
        content=text_to_embed_single,
        task_type="RETRIEVAL_QUERY" # Example task type
    )

    # The embedding is a list of floating-point numbers (vector)
    embedding_vector_single = result_single['embedding']
    print(f"Embedding Vector (first 10 values): {embedding_vector_single[:10]}...")
    print(f"Embedding Dimension: {len(embedding_vector_single)}")

except Exception as e:
    print(f"An error occurred: {e}")

print("\n" + "="*50 + "\n") # Separator

# --- Generate Embeddings for a Batch of Texts ---
print(f"--- Embedding for batch of texts ---")
try:
    # When embedding documents for retrieval, use RETRIEVAL_DOCUMENT
    result_batch = genai.embed_content(
        model=model_name,
        content=texts_to_embed_batch,
        task_type="RETRIEVAL_DOCUMENT" # Example task type for documents
    )

    # The result['embedding'] will be a list of embedding vectors, one for each input text
    embedding_vectors_batch = result_batch['embedding']

    print(f"Number of embeddings generated: {len(embedding_vectors_batch)}")
    for i, (text, vector) in enumerate(zip(texts_to_embed_batch, embedding_vectors_batch)):
        print(f"\nText {i+1}: '{text}'")
        print(f"Embedding Vector (first 10 values): {vector[:10]}...")
        print(f"Embedding Dimension: {len(vector)}")

except Exception as e:
    print(f"An error occurred: {e}")