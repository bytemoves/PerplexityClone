from typing import List
import os
from dotenv import load_dotenv
import numpy as np
import cohere

load_dotenv()

class SortSourceService:
    def __init__(self):
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY environment variable is not set")
        self.co = cohere.Client(api_key=api_key)
        
    def _get_embedding(self, text: str, input_type: str) -> List[float]:
        response = self.co.embed(
            model="embed-english-v3.0",
            texts=[text],
            input_type=input_type
        )
        return response.embeddings[0]
        
    def sort_sources(self, query: str, search_result: list):
        query_embedding = self._get_embedding(query, input_type="search_query")
        
        # Calculate cosine similarity for each result
        for res in search_result:
            res_embedding = self._get_embedding(res['content'], input_type="search_document")
            similarity = np.dot(query_embedding, res_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(res_embedding))
            print(similarity)
