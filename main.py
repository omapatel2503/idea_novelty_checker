import asyncio
import warnings
from utils.s2_api import *
import asyncio

warnings.filterwarnings("ignore")
# import your functions here...

if __name__ == "__main__":
    result = asyncio.run(
        papers_from_search_api(query="machine learning", start_year="2020", end_year="2024",limit=5)
    )
    print(result)
