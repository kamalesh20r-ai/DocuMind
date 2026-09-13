from dotenv import load_dotenv
import os
from google import genai
load_dotenv()
api_key=os.getenv("GEMINI_API_KEY")
client=genai.Client(api_key=api_key)
response=client.models.generate_content(model="gemini-3.7-flash",contents="Explain RAG in two simple sentences.")
print(response.text)