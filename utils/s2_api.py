import asyncio
import os
from typing import Any, Optional

import aiohttp

_DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=60)  # seconds

def _build_headers() -> dict:
    key = os.getenv("S2_API_KEY")
    return {"x-api-key": key} if key else {}

async def make_request_with_retries(
    url: str,
    headers: Optional[dict] = None,
    params: Optional[dict] = None,
    input_json: Optional[dict] = None,
    request_type: str = "get",
    url_string: str = "",
    retries: int = 3,
    delay: float = 5.0,
    session: Optional[aiohttp.ClientSession] = None,
    timeout: aiohttp.ClientTimeout = _DEFAULT_TIMEOUT,
) -> Optional[Any]:
    """
    Returns parsed JSON on success, None on total failure.
    """
    _headers = {}
    _headers.update(headers or {})
    # avoid sending None values in headers
    _headers = {k: v for k, v in _headers.items() if v is not None}

    owns_session = session is None
    if owns_session:
        session = aiohttp.ClientSession(timeout=timeout)

    try:
        last_error = None
        for attempt in range(1, retries + 1):
            try:
                if request_type.lower() == "get":
                    resp_ctx = session.get(url, headers=_headers, params=params)
                else:
                    resp_ctx = session.post(
                        url, headers=_headers, params=params, json=input_json
                    )

                async with resp_ctx as response:
                    # Retry on 5xx; return/raise on 4xx as appropriate
                    if 200 <= response.status < 300:
                        # If no body, return empty dict
                        if response.content_length == 0:
                            return {}
                        # Tolerate wrong content-type from upstream
                        return await response.json(content_type=None)
                    elif 500 <= response.status < 600:
                        last_error = f"{response.status} {await response.text()}"
                    else:
                        # 4xx – likely caller error; include details
                        text = await response.text()
                        raise RuntimeError(
                            f"{url_string or 'request'} failed with {response.status}: {text}"
                        )
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_error = str(e)

            if attempt < retries:
                await asyncio.sleep(delay)

        # All attempts failed
        if last_error:
            # surface the reason; caller can decide what to do
            return None
        return None
    finally:
        if owns_session:
            await session.close()

# async def get_paper_data(id_: str, id_type: str = "corpus_id", batch_wise: bool = False):
#     attributes = "corpusId,paperId,url,title,year,abstract,authors.name,fieldsOfStudy,citationCount,venue"
#     params = {"fields": attributes}
#     headers = _build_headers()

#     if not batch_wise:
#         if id_type == "paper_id":
#             url = f"https://api.semanticscholar.org/graph/v1/paper/{id_}"
#         else:
#             url = f"https://api.semanticscholar.org/graph/v1/paper/CorpusId:{id_}"
#         return await make_request_with_retries(
#             url, headers=headers, params=params, url_string="get paper data"
#         )
#     else:
#         url = "https://api.semanticscholar.org/graph/v1/paper/batch"
#         # no stray prints; just return the payload
#         return await make_request_with_retries(
#             url,
#             headers=headers,
#             input_json={"ids": id_},
#             request_type="post",
#             url_string="get paper data (batch)",
#             params=params,
#         )

async def papers_from_search_api(
    query: str = "",
    start_year: str = "",
    end_year: str = "",
    search_type: str = "keyword",
    limit: int = 5,
):
    attributes = (
        "corpusId,paperId,url,title,year,abstract,authors.name,openAccessPdf,"
        "fieldsOfStudy,s2FieldsOfStudy,citationCount,venue"
    )
    if search_type == "keyword":
        url = "https://api.semanticscholar.org/graph/v1/paper/search/"
    elif search_type == "snippet":
        url = "https://api.semanticscholar.org/graph/v1/snippet/search/"
    else:
        raise ValueError("search_type must be 'keyword' or 'snippet'")

    params = {"query": query, "fields": attributes, "limit": limit}
    if start_year and end_year:
        params["year"] = f"{start_year}-{end_year}"

    return await make_request_with_retries(
        url, headers=_build_headers(), params=params, url_string="search_api"
    )

# async def papers_from_recommendation_api_allCs(corpus_id: int, limit: int = 100):
#     url = (
#         f"https://api.semanticscholar.org/recommendations/v1/papers/forpaper/"
#         f"CorpusId:{corpus_id}?fields=corpusId,paperId,title,abstract,paperId,url,"
#         f"venue,publicationDate,fieldsOfStudy,authors&limit={limit}&from=all-cs"
#     )
#     return await make_request_with_retries(
#         url, headers=_build_headers(), url_string="recommendations api - allCS"
#     )

# async def papers_from_recommendation_api_recent(corpus_id: int, limit: int = 100):
#     url = (
#         f"https://api.semanticscholar.org/recommendations/v1/papers/forpaper/"
#         f"CorpusId:{corpus_id}?fields=corpusId,paperId,title,abstract,paperId,url,"
#         f"venue,publicationDate,fieldsOfStudy,authors&limit={limit}&from=recent"
#     )
#     return await make_request_with_retries(
#         url, headers=_build_headers(), url_string="recommendations api - recent"
#     )

# async def getSpecterEmbedding_paperIDs(paperIDs):
#     url = "https://api.semanticscholar.org/graph/v1/paper/batch"
#     return await make_request_with_retries(
#         url,
#         headers=_build_headers(),
#         params={"fields": "corpusId,embedding"},
#         input_json={"ids": paperIDs},
#         request_type="post",
#         url_string="Specter Embeddings",
#     )



# data= await papers_from_search_api('Design an experimental framework to assess action tracking capabilities in ReAct agents navigating text-based game environments. This framework involves a comparative analysis of three agent models: a random baseline, a conventional ReAct agent, and an enhanced ReAct agent with integrated action history tracking. Utilizing the CookingWorld environment from TextWorldExpress, the agents are evaluated over 50 episodes, focusing on metrics such as task completion rates and average scores. The findings indicate that while both ReAct models significantly surpass the random baseline, incorporating action tracking does not yield statistically significant enhancements over the standard ReAct model.')  # Example usage
# print("Data=",data)
# exit (0)
