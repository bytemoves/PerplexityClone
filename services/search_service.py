from tavily import TavilyClient
import trafilatura
from config import Settings
settings = Settings()
tavily_client = TavilyClient(api_key=settings.TAVILY_API_KEY)

class SearchService:
    def web_search(self,query:str):
        try:
            results = []
            response = tavily_client.search(query,max_results=10)
            search_result = response.get("results",[])
            
            for result in search_result:
                downloaded = trafilatura.fetch_url(result.get("url"))
                content = trafilatura.extract(downloaded,include_comments=False)
            
            
            results.append({
                "title": result.get("title",""),
                "url": result.get("url"),
                "content": content,
            })
            return results 
        except Exception as e:
            print(e)

