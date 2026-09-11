from dotenv import load_dotenv
import os
import numpy as np
from google import genai
from google.genai import types
from pypdf import PdfReader
load_dotenv()
api_key=os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
reader = PdfReader("DATA/DBMS Lab Manual.pdf")
text = ""
for page in reader.pages :
    page_text = page.extract_text()
    if page_text :
        text += page_text + "\n"
chunk_size = 1000
overlap = 200
chunks = []
for i in range(0,len(text),chunk_size-overlap) :
    chunk = text[i:i+chunk_size]
    chunks.append(chunk)
print("Total Chunks :",len(chunks))
contents = []
for chunk in chunks :
    content = types.Content(parts = [types.Part.from_text(text = chunk)])
    contents.append(content)
result = client.models.embed_content(model = "gemini-embedding-2",contents = contents)
embeddings = []
for embedding in result.embeddings :
    embeddings.append(embedding.values)
question = "What is INNER JOIN ?"
question_result = client.models.embed_content(model="gemini-embedding-2",contents=question)
question_embedding = question_result.embeddings[0].values
similarities = []
for embedding in embeddings :
    similarity = np.dot(question_embedding,embedding)/((np.linalg.norm(question_embedding))*(np.linalg.norm(embeddings)))
    similarities.append(similarity)
top_k = 3
top_indices = np.argsort(similarities)[-top_k:][::-1]
context = ""
for index in top_indices:
    context += chunks[index] + "\n\n"
prompt = f"""
Answer the question using only the context below.
Context:{context}
Question:{question}
If the answer is not present in the context, say:
"The answer is not available in the uploaded document."
"""
response = client.models.generate_content(model="gemini-3.5-flash",contents=prompt)
print("\nFinal Answer:\n")
print(response.text)