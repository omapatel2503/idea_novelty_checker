import json
import google.generativeai as genai

def extract_concepts_from_abstract(abstract_text: str) -> dict | None:
    """
    Analyzes an abstract using the Gemini API to extract concepts and keywords.
    
    Args:
        abstract_text: The string of the abstract to analyze.

    Returns:
        A dictionary of {concept: [keywords]} or None if an error occurs.
    """
    print("\n[Module 1] Analyzing abstract with Gemini...")
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        
        prompt_template = """
        You are a highly skilled research analyst... [Same detailed prompt as before]
        
        --- ABSTRACT ---
        {abstract}
        --- END ABSTRACT ---
        """
        
        full_prompt = prompt_template.format(abstract=abstract_text)
        response = model.generate_content(full_prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        print(" [Module 1] Analysis complete.")
        return json.loads(cleaned_response)

    except Exception as e:
        print(f"[Module 1] An error occurred during Gemini analysis: {e}")
        return None

