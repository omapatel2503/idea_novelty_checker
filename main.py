import asyncio
import json
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Import our modular functions
from utils.extractor import extract_concepts_from_abstract
from utils.searcher import search_papers_s2orc
from utils.scorer import calculate_novelty_score_for_concept # <-- IMPORT NEW MODULE

# --- CONFIGURATION ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
S2_API_KEY = os.getenv("S2_API_KEY")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    print(" GOOGLE_API_KEY not found in .env file.")

if not S2_API_KEY:
    print("S2_API_KEY not found in .env file.")


# --- MAIN ORCHESTRATOR ---
async def main():
    """
    Orchestrates the full workflow from abstract to final JSON output with novelty scores.
    """
    if not S2_API_KEY or not GOOGLE_API_KEY:
        print("Aborting pipeline due to missing API keys.")
        return

    # --- INPUTS ---
    K_PAPERS_PER_CONCEPT = 3
    sample_abstract = """
    Addressing the critical gap in early-warning systems for agricultural pandemics, this paper introduces a novel Graph Convolutional Network (GCN) architecture, named CropGuardNet, for predicting crop disease outbreaks. Traditional methods rely on inconsistent manual reporting, leading to significant delays. Our approach leverages a newly compiled, large-scale dataset of satellite imagery and meteorological data from under-studied regions in Southeast Asia. CropGuardNet's unique spatio-temporal attention mechanism allows it to model disease spread with unprecedented accuracy. We demonstrate through extensive experiments that our model achieves a 94% prediction accuracy, outperforming existing models by over 15%. This framework provides a cost-effective and scalable solution for proactive farm management, marking a significant step towards global food security.
    """

    # Step 1: Extract concepts and keywords
    keyword_analysis = extract_concepts_from_abstract(sample_abstract)
    if not keyword_analysis: return

    # Step 2: Find similar papers for those concepts
    similar_papers_map = await search_papers_s2orc(keyword_analysis, K_PAPERS_PER_CONCEPT, S2_API_KEY)

    # Step 3: Calculate novelty scores for each concept
    print("\n[Module 3] Calculating novelty scores...")
    final_output = []
    for concept, keywords in keyword_analysis.items():
        papers_found = similar_papers_map.get(concept, [])
        
        # We only need the abstracts from the found papers for scoring
        similar_abstracts = [
            paper['abstract'] for paper in papers_found if paper.get('abstract')
        ]
        
        # Calculate the score by comparing the original abstract to the found ones
        novelty_score = calculate_novelty_score_for_concept(sample_abstract, similar_abstracts)
        print(f"   -> Novelty score for '{concept}': {novelty_score}")

        # Append all data for this concept to our final output list
        final_output.append({
            "concept": concept,
            "extracted_keywords": keywords,
            "novelty_score": novelty_score, # <-- ADD THE SCORE
            "similar_papers": [
                {
                    "title": paper.get('title', 'N/A'),
                    "url": paper.get('url', 'N/A'),
                    "abstract": paper.get('abstract', 'N/A')
                } for paper in papers_found
            ]
        })
    print("[Module 3] Scoring complete.")
    
    # Step 4: Display the final compiled JSON
    print("\n\n" + "="*30)
    print("FINAL JSON OUTPUT ")
    print("="*30)
    print(json.dumps(final_output, indent=2))
    with open("data.json", "w") as f:
      json.dump(final_output, f, indent=4)



# --- EXECUTION ---
if __name__ == "__main__":
    asyncio.run(main())

