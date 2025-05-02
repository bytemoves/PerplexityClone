from typing import List
from dotenv import load_dotenv
import google.generativeai as genai
load_dotenv()


class SortSourceService:
    def __init__(self):
        self.model_name =  "models/embedding-001"
        
        
    def sort_sources(self, query: str, search_result: list):
        result = genai.embed_content(
            model=self.model_name,
            content=query,
            task_type="RETRIEVAL_QUERY"
        )
        query_embedding = result['embedding']
        
