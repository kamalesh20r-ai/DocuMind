import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
import os
from google import genai
from google.genai import types
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
st.title("DocuMind")
st.write("AI-Powered Document Question Answering Assistant")
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "processed_file_name" not in st.session_state:
    st.session_state.processed_file_name = None
uploaded_file = st.file_uploader("Upload Your PDF",type=["pdf"],accept_multiple_files=False)
if uploaded_file is not None:
    st.success("PDF Uploaded Successfully")
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    st.write("Total Characters Extracted:",len(text))
    st.subheader("Preview")
    st.text(text[:1000])
    chunk_size = 1000
    overlap = 200
    chunks = []
    for i in range(0,len(text),chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
    st.write("Total Chunks Created:",len(chunks))
    if (st.session_state.processed_file_name!= uploaded_file.name):
        contents = []
        for chunk in chunks:
            content = types.Content(parts=[types.Part.from_text(text=chunk)])
            contents.append(content)
        with st.spinner("Creating document embeddings..."):
            result = client.models.embed_content(model="gemini-embedding-2",contents=contents)
        embeddings = []
        for embedding in result.embeddings:
            embeddings.append(embedding.values)
        st.session_state.chunks = chunks
        st.session_state.embeddings = embeddings
        st.session_state.processed_file_name = (uploaded_file.name)
        st.success("Document processed successfully!")
        st.write("Total Embeddings Created:",len(embeddings))
    else:
        st.info("This document is already processed.")
        st.write("Total Embeddings Stored:",len(st.session_state.embeddings))