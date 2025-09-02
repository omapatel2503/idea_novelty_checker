import json
import google.generativeai as genai

async def calculate_novelty_score_with_gemini(concept: str, input_abstract: str, similar_abstracts: list[str]) -> dict:
    """
    Calculates a novelty score by using Gemini to compare an input abstract 
    against a list of similar abstracts for a specific concept.
    """
    if not similar_abstracts:
        return {"score": 1.0, "reasoning": "No similar papers were found for this concept, indicating high novelty."}

    formatted_evidence = ""
    for i, abstract in enumerate(similar_abstracts, 1):
        formatted_evidence += f"Evidence Abstract {i}:\n{abstract}\n\n"

    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        
        prompt = f"""
        You are an expert academic reviewer. Your task is to assess the novelty of a "Proposed Abstract" specifically concerning the concept of '{concept}'. Compare it against a list of "Existing Abstracts".

        **Instructions:**
        1.  Read the Proposed Abstract to understand its contribution related to '{concept}'.
        2.  Compare this specific contribution to the abstracts of existing papers.
        3.  Assign a novelty score on a continuous float scale from 0.0 to 1.0 using the rubric below.
        4.  Provide a brief, one-sentence justification for your score.

        **Scoring Rubric:**
        - **0.0 - 0.2 (Derivative):** The idea is virtually identical to existing work.
        - **0.3 - 0.5 (Incremental):** A minor improvement or variation on existing methods.
        - **0.6 - 0.8 (Substantial):** A significant improvement or application to a new domain.
        - **0.9 - 1.0 (Highly Novel):** A fundamentally new method, theory, or approach.

        Respond ONLY with a valid JSON object in the format:
        {{"novelty_score": <float>, "reasoning": "<string>"}}

        --- PROPOSED ABSTRACT ---
        {input_abstract}
        --- END PROPOSED ABSTRACT ---

        --- EXISTING ABSTRACTS (EVIDENCE) ---
        {formatted_evidence}
        --- END EXISTING ABSTRACTS ---
        """
        
        response = await model.generate_content_async(prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        result = json.loads(cleaned_response)
        return {"score": result.get("novelty_score", 0.0), "reasoning": result.get("reasoning", "No reasoning provided.")}
    except Exception as e:
        print(f"❌ [Module 3] An error occurred during Gemini scoring for '{concept}': {e}")
        return {"score": 0.0, "reasoning": "Error during analysis."}