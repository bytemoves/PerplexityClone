import google.generativeai as genai

from config import Settings

settings = Settings()

class LLMService:
    
    def __init__(self):
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GemerativeModel("gemini-2.0-flash")
    def generate_response(self,query: str,search_results: list[dict]):
        
        #firt source url
        #second soircr url
        #query:
        #provide detailed response bsed on this
        
        context_text = "\n\n".join([\
            # url : content for every source
            f"source {i+1} {result['url']}: \n{result['content']}" # 
            for i , result in enumerate (search_results)
        ])
        
        full_prompt = f"""
        
        Context from web search:
        {context_text}

        Query: {query}

        Please provide a comprehensive, detailed, well-cited accurate response using the above context do not exceed 100 words.
        Think and reason deeply. Ensure it answers the query the user is asking. Do not use your knowledge until it is absolutely necessary
        
        """
        
        response = self.model.generate.content(full_prompt,stream=True)
        
        for chunk in response:
            yield chunk.text