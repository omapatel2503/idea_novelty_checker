import google.generativeai as genai
from google.colab import userdata
import json
import os

try:
    api_key = userdata.get('GOOGLE_API_KEY')
    genai.configure(api_key=api_key)
except userdata.SecretNotFoundError as e:
    print(f"Secret not found: {e}")
    print("Please make sure you have stored your API key in Colab Secrets with the name 'GOOGLE_API_KEY'.")


def analyze_abstract_with_gemini(abstract_text: str) -> dict:
    """
    Analyzes an abstract using the Gemini LLM with a detailed prompt to extract high-quality keywords.
    """
    model = genai.GenerativeModel('gemini-1.5-flash-latest')


    prompt_template = """
    You are a highly skilled research analyst. Your task is to meticulously analyze the provided abstract and extract specific, high-quality keywords for each of the following research novelty concepts. The keywords should be suitable for finding similar academic papers in databases like Google Scholar or IEEE Xplore.

    **Concept Definitions:**
    - **Research Gap**: The core problem, limitation, or missing knowledge the paper explicitly addresses.
    - **Methodological Novelty**: New procedures or techniques. Look for: new algorithms, unique statistical methods, novel experimental designs, new data collection tools (surveys, protocols), or cross-disciplinary adaptations of methods.
    - **Empirical Novelty**: New data-driven findings or evidence. Look for: studies on new populations or contexts, analysis of a newly created or unique dataset, or the discovery of new facts or relationships that contradict previous work.
    - **Dataset**: Specific details about the dataset used or created. Look for: its name, source (e.g., satellite imagery, clinical trials), type (e.g., time-series, text corpus), and unique characteristics (e.g., large-scale, longitudinal).
    - **Applicational Novelty**: New use-cases for existing knowledge. Look for: applying a known theory/method to a new problem or field, or solving a practical problem for the first time.
    - **Theoretical Novelty**: New ideas or conceptual frameworks. Look for: proposals of new theories, critiques of existing theories, or a synthesis of multiple theories into a new one.

    **Instructions**:
    - For each concept, extract 3-4 of the most specific and descriptive keywords or short phrases.
    - If a concept is not clearly present in the abstract, return an empty list `[]`.
    - Your output MUST be a single, valid JSON object and nothing else.

    **Example Output Format**:
    {{
      "Research Gap": ["keyword1", "keyword2"],
      "Methodological Novelty": ["keyword3", "keyword4", "keyword5"],
      "Empirical Novelty": ["keyword6", "keyword7"],
      "Dataset": ["dataset name", "data source", "data type"],
      "Applicational Novelty": ["keyword8", "keyword9"],
      "Theoretical Novelty": []
    }}

    --- ABSTRACT ---
    {abstract}
    --- END ABSTRACT ---
    """

    full_prompt = prompt_template.format(abstract=abstract_text)

    try:
        response = model.generate_content(full_prompt)
        # Clean up the response to ensure it's valid JSON
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        result_dict = json.loads(cleaned_response)
        return result_dict
    except Exception as e:
        print(f"An error occurred during API call or JSON parsing: {e}")
        print("--- Raw Model Response ---")
        print(response.text if 'response' in locals() and hasattr(response, 'text') else "No response received or response object is invalid.")
        return None


# === 3. RUN THE ANALYSIS ===
sample_abstract = """We explore and enhance the ability of neural language models to generate novel scientific directions grounded in literature. Work
on literature-based hypothesis generation has
traditionally focused on binary link prediction—
severely limiting the expressivity of hypotheses. This line of work also does not focus on
optimizing novelty. We take a dramatic departure with a novel setting in which models use
as input background contexts (e.g., problems,
experimental settings, goals), and output natural language ideas grounded in literature. We
present SCIMON, a modeling framework that
uses retrieval of “inspirations” from past scientific papers, and explicitly optimizes for novelty
by iteratively comparing to prior papers and updating idea suggestions until sufficient novelty
is achieved. Comprehensive evaluations reveal
that GPT-4 tends to generate ideas with overall low technical depth and novelty, while our
methods partially mitigate this issue. Our work
represents a first step toward evaluating and
developing language models that generate new
ideas derived from the scientific literature1
"""

print("Analyzing Abstract with Enriched Gemini Prompt in Colab...")
analysis_results = analyze_abstract_with_gemini(sample_abstract)

if analysis_results:
    print("\n--- High-Quality Keyword Analysis Results ---")
    # Pretty print the JSON output
    print(json.dumps(analysis_results, indent=2))
    print("---------------------------------------------")
