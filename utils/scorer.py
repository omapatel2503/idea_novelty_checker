from sentence_transformers import SentenceTransformer, util

# Load a pre-trained sentence transformer model. 
# 'all-MiniLM-L6-v2' is a great, fast model for semantic similarity.
# The model will be downloaded automatically on the first run.
print("[Module Scorer] Loading Sentence Transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("[Module Scorer] Model loaded.")

def calculate_novelty_score_for_concept(input_abstract: str, similar_abstracts: list[str]) -> float:
    """
    Calculates a novelty score for a single concept by comparing an input abstract
    against a list of similar abstracts.

    Args:
        input_abstract: The user's original abstract.
        similar_abstracts: A list of abstracts from papers found by the searcher.

    Returns:
        A novelty score between 0.0 and 1.0.
    """
    if not similar_abstracts:
        # If no similar papers are found, the concept is considered completely novel.
        return 1.0

    # Encode all abstracts into dense vector embeddings.
    input_embedding = model.encode(input_abstract, convert_to_tensor=True)
    similar_embeddings = model.encode(similar_abstracts, convert_to_tensor=True)

    # Calculate the cosine similarity between the input abstract and all similar abstracts.
    cosine_scores = util.cos_sim(input_embedding, similar_embeddings)

    # The highest similarity score represents the "closest" prior work.
    # We take the max value from the resulting tensor.
    max_similarity = cosine_scores.max().item()

    # Novelty is the inverse of the highest similarity.
    # A max similarity of 0.9 (very similar) results in a novelty of 0.1.
    # A max similarity of 0.2 (very different) results in a novelty of 0.8.
    novelty_score = 1 - max_similarity

    return round(novelty_score, 3) # Return a rounded score
