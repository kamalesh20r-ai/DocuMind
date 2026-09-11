from dotenv import load_dotenv
import os
import json
import numpy as np
from google import genai
from google.genai import types
from pypdf import PdfReader
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
reader = PdfReader("DATA/DBMS Lab Manual.pdf")
text = ""
for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        text += page_text + "\n"
chunk_size = 1000
overlap = 200
chunks = []
for i in range(0, len(text), chunk_size - overlap):
    chunk = text[i:i + chunk_size]
    chunks.append(chunk)
contents = []
for chunk in chunks:
    content = types.Content(parts=[types.Part.from_text(text=chunk)])
    contents.append(content)
result = client.models.embed_content(model="gemini-embedding-2",contents=contents)
embeddings = []
for embedding in result.embeddings:
    embeddings.append(embedding.values)
with open("chunks.json", "w", encoding="utf-8") as file:json.dump(chunks, file, ensure_ascii=False, indent=2)
np.save("embeddings.npy", np.array(embeddings))
print("Index built successfully")
print("Total chunks:", len(chunks))
print("Total embeddings:", len(embeddings))