from dotenv import load_dotenv
import os
import json
import numpy as np
from google import genai
import time
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
with open("chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)
embeddings = np.load("embeddings.npy")
question = input("Ask a question: ")
start = time.perf_counter()
question_result = client.models.embed_content(model="gemini-embedding-2",contents=question)
question_embedding = question_result.embeddings[0].values
embedding_end = time.perf_counter()
similarities = []
for embedding in embeddings:
    similarity = np.dot(question_embedding, embedding) / (np.linalg.norm(question_embedding) * np.linalg.norm(embedding))
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
If the answer is not available in the context, say:
"The answer is not available in the uploaded document."
"""
generation_start = time.perf_counter()
response = client.models.generate_content(model="gemini-3.5-flash-lite",contents=prompt)
generation_end = time.perf_counter()
print("\nFinal Answer:\n")
print(response.text)
print("\n--- Speed Test ---")
print("Question Embedding Time:",round(embedding_end - start, 2), "seconds")
print("Answer Generation Time:",round(generation_end - generation_start, 2), "seconds")