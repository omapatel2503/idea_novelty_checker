import os
import asyncio
import json
from dotenv import load_dotenv
import google.generativeai as genai

# Import modular functions
from utils.extractor import extract_concepts_from_abstract
from utils.searcher import search_papers_s2orc
from utils.scorer import calculate_novelty_score_with_gemini

def setup():
    """Loads environment variables and configures API clients."""
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    s2_api_key = os.getenv("S2_API_KEY")
    
    if not google_api_key or not s2_api_key:
        print("❌ One or more API keys not found in .env file.")
        return None, None
        
    genai.configure(api_key=google_api_key)
    print("✅ API keys loaded and configured successfully.")
    return google_api_key, s2_api_key

async def main():
    """Orchestrates the full workflow from abstract to final JSON output."""
    google_api_key, s2_api_key = setup()
    if not s2_api_key: return

    # --- INPUTS ---
    K_PAPERS_PER_CONCEPT = 3
    sample_abstract = """
    Addressing the critical gap in early-warning systems for agricultural pandemics, this paper introduces a novel Graph Convolutional Network (GCN) architecture, named CropGuardNet, for predicting crop disease outbreaks. Traditional methods rely on inconsistent manual reporting, leading to significant delays. Our approach leverages a newly compiled, large-scale dataset of satellite imagery and meteorological data from under-studied regions in Southeast Asia. CropGuardNet's unique spatio-temporal attention mechanism allows it to model disease spread with unprecedented accuracy. We demonstrate through extensive experiments that our model achieves a 94% prediction accuracy, outperforming existing models by over 15%. This framework provides a cost-effective and scalable solution for proactive farm management, marking a significant step towards global food security.
    """

    # Step 1: Extract concepts and keywords
    keyword_analysis = extract_concepts_from_abstract(sample_abstract)
    if not keyword_analysis: return

    # Step 2: Find similar papers for those concepts
    similar_papers_map = await search_papers_s2orc(keyword_analysis, K_PAPERS_PER_CONCEPT, s2_api_key)

    # Step 3: Asynchronously calculate novelty scores using Gemini
    print("\n[Module 3] Calculating novelty scores using Gemini...")
    scoring_tasks = []
    for concept, keywords in keyword_analysis.items():
        papers_found = similar_papers_map.get(concept, [])
        similar_abstracts = [p['abstract'] for p in papers_found if p.get('abstract')]
        task = asyncio.create_task(calculate_novelty_score_with_gemini(concept, sample_abstract, similar_abstracts))
        scoring_tasks.append(task)
    
    score_results = await asyncio.gather(*scoring_tasks)
    print("✅ [Module 3] Scoring complete.")

    # Step 4: Compile the final, structured JSON output
    print("\n[Module 4] Compiling final JSON output...")
    final_output = []
    # Create a mapping from concept to its score result for easy lookup
    concept_to_score = {res['concept']: res for res in score_results_with_concept_name(keyword_analysis.keys(), score_results)}
    
    for concept, keywords in keyword_analysis.items():
        score_result = concept_to_score.get(concept, {"score": 0.0, "reasoning": "Scoring failed."})
        papers_found = similar_papers_map.get(concept, [])
        
        print(f"   -> Final score for '{concept}': {score_result['score']} (Reason: {score_result['reasoning']})")
        
        final_output.append({
            "concept": concept,
            "extracted_keywords": keywords,
            "novelty_score": score_result['score'],
            "novelty_reasoning": score_result['reasoning'],
            "similar_papers": [{"title": p.get('title'), "url": p.get('url'), "abstract": p.get('abstract')} for p in papers_found]
        })

    print("\n\n" + "="*30)
    print("✅✅✅ FINAL JSON OUTPUT ✅✅✅")
    print("="*30)
    print(json.dumps(final_output, indent=2))

def score_results_with_concept_name(concepts, results):
    """Helper to re-associate concepts with their score results after asyncio.gather."""
    return [{"concept": concept, **result} for concept, result in zip(concepts, results)]

if __name__ == "__main__":
    asyncio.run(main())















# from sentence_transformers import SentenceTransformer, util

# # Load a pre-trained sentence transformer model. 
# # 'all-MiniLM-L6-v2' is a great, fast model for semantic similarity.
# # The model will be downloaded automatically on the first run.
# print("[Module Scorer] Loading Sentence Transformer model...")
# model = SentenceTransformer('all-MiniLM-L6-v2')
# print("[Module Scorer] Model loaded.")

# def calculate_novelty_score_for_concept(input_abstract: str, similar_abstracts: list[str]) -> float:
#     """
#     Calculates a novelty score for a single concept by comparing an input abstract
#     against a list of similar abstracts.

#     Args:
#         input_abstract: The user's original abstract.
#         similar_abstracts: A list of abstracts from papers found by the searcher.

#     Returns:
#         A novelty score between 0.0 and 1.0.
#     """
#     if not similar_abstracts:
#         # If no similar papers are found, the concept is considered completely novel.
#         return 1.0

#     # Encode all abstracts into dense vector embeddings.
#     input_embedding = model.encode(input_abstract, convert_to_tensor=True)
#     similar_embeddings = model.encode(similar_abstracts, convert_to_tensor=True)

#     # Calculate the cosine similarity between the input abstract and all similar abstracts.
#     cosine_scores = util.cos_sim(input_embedding, similar_embeddings)

#     # The highest similarity score represents the "closest" prior work.
#     # We take the max value from the resulting tensor.
#     max_similarity = cosine_scores.max().item()

#     # Novelty is the inverse of the highest similarity.
#     # A max similarity of 0.9 (very similar) results in a novelty of 0.1.
#     # A max similarity of 0.2 (very different) results in a novelty of 0.8.
#     novelty_score = 1 - max_similarity

#     return round(novelty_score, 3) # Return a rounded score
