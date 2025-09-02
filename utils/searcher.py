import asyncio
import aiohttp

S2_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

async def _fetch_single_concept(session, concept_name, keywords, k, s2_api_key):
    """Helper function to fetch papers for one concept."""
    if not keywords: return concept_name, []
    query = "+".join(keywords)
    params = {'query': query, 'limit': k, 'fields': 'title,abstract,url'}
    headers = {'x-api-key': s2_api_key}
    
    try:
        async with session.get(S2_API_URL, params=params, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                return concept_name, data.get('data', [])
            else:
                print(f" [Module 2] Error for '{concept_name}': HTTP {response.status} - Details: {await response.text()}")
                return concept_name, []
    except Exception as e:
        print(f"[Module 2] An exception occurred for '{concept_name}': {e}")
        return concept_name, []

async def search_papers_s2orc(keyword_analysis: dict, k_papers_per_concept: int, s2_api_key: str) -> dict:
    """
    Asynchronously searches for papers for all concepts using the S2ORC API.
    """
    print("\n[Module 2] Searching for similar papers on Semantic Scholar...")
    async with aiohttp.ClientSession() as session:
        tasks = []
        for concept, keywords in keyword_analysis.items():
            if keywords:
                print(f"   -> Launching S2 search for: '{concept}'")
                task = asyncio.create_task(_fetch_single_concept(session, concept, keywords, k_papers_per_concept, s2_api_key))
                tasks.append(task)
                await asyncio.sleep(1.1)
        
        search_results = await asyncio.gather(*tasks)
    
    print("[Module 2] All searches complete.")
    return {concept: papers for concept, papers in search_results}

