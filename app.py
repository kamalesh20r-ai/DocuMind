import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
import os
from google import genai
from google.genai import types
import numpy as np
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    api_key = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=api_key)
st.set_page_config(page_title="DocuMind",page_icon="📄",layout="wide")
st.title("📄 DocuMind")
st.caption("AI-Powered Document Question Answering & Study Assistant")
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "processed_file_name" not in st.session_state:
    st.session_state.processed_file_name = None
if "history" not in st.session_state:
    st.session_state.history = []
with st.sidebar:
    st.header("📄 DocuMind")
    st.write("Upload a PDF and ask questions based on its content.")
    st.divider()
    if st.session_state.processed_file_name is not None:
        st.subheader("Document Info")
        st.write("File:",st.session_state.processed_file_name)
        if st.session_state.chunks is not None:
            st.write("Chunks:",len(st.session_state.chunks))
        if st.session_state.embeddings is not None:
            st.write("Embeddings:",len(st.session_state.embeddings))
    else:
        st.info("No document processed yet.")
uploaded_file = st.file_uploader("Upload Your PDF",type=["pdf"],accept_multiple_files=False)
if uploaded_file is not None:
    st.success("PDF Uploaded Successfully")
    reader = PdfReader(uploaded_file)
    text = ""
    low_text_pages = 0
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if len(page_text.strip()) < 50:
            low_text_pages += 1
        text += page_text + "\n"
    total_pages = len(reader.pages)
    image_like_ratio = low_text_pages / total_pages
    if image_like_ratio > 0.30:
        st.warning(f"This PDF contains many scanned/image-based pages "f"({low_text_pages} out of {total_pages} pages have very little extractable text). ""Some content may be missing. OCR support is required for full extraction.")
    if len(text.strip()) < 1000:
        st.warning("Very little text could be extracted from this PDF. " "The document may be scanned or image-based. " "OCR support will be added in a future version.")
        st.stop()
    chunk_size = 1000
    overlap = 200
    chunks = []
    for i in range(0,len(text),chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Characters Extracted",len(text))
    with col2:
        st.metric("Chunks Created",len(chunks))
    with st.expander("Preview extracted text"):
        st.text(text[:1000])
    if (st.session_state.processed_file_name!= uploaded_file.name):
        st.session_state.history = []
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
        st.rerun()
        st.success(f"Document processed successfully! " f"{len(st.session_state.chunks)} chunks indexed.")
    else:
        st.info("This document is already processed.")
        st.write("Total Embeddings Stored:",len(st.session_state.embeddings))
if st.session_state.chunks is not None and st.session_state.embeddings is not None:
    st.divider()
    st.subheader("Ask a Question")
    if st.button("Clear Conversation"):
        st.session_state.history = []
        st.rerun()
    st.caption("Ask questions based only on the uploaded document.")
    question = st.text_input("Enter your question about the uploaded document")
    if st.button("Ask"):
        if question.strip() == "":
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching the document..."):
                question_result = client.models.embed_content(model="gemini-embedding-2",contents=question)
                question_embedding = question_result.embeddings[0].values
                similarities = []
                for embedding in st.session_state.embeddings:
                    similarity = np.dot(question_embedding,embedding) / (np.linalg.norm(question_embedding) * np.linalg.norm(embedding))
                    similarities.append(similarity)
                top_k = 3
                top_indices = np.argsort(similarities)[-top_k:][::-1]
                context = ""
                for index in top_indices:
                    context += (st.session_state.chunks[index] + "\n\n")
            prompt = f"""
                    Answer the question using only the context below.
                    Context:{context}
                    Question:{question}
                    If the answer is not available in the context, say:"The answer is not available in the uploaded document."
                    """
            try:
                with st.spinner("Generating answer..."):
                    response = (client.models.generate_content(model="gemini-3.5-flash-lite",contents=prompt))
                    answer_text = response.text.strip()
                retrieved_chunks = []
                not_available_message = ("The answer is not available in the uploaded document.")
                if not_available_message.lower() not in answer_text.lower():
                    for index in top_indices:
                        retrieved_chunks.append({"chunk": st.session_state.chunks[index],"score": similarities[index]})
                st.session_state.history.append({"question": question,"answer": answer_text,"retrieved_chunks": retrieved_chunks})
            except Exception as error:
                st.error("Unable to generate the answer right now. " "Please try again.")
                print(error)
    if st.session_state.history:
        st.subheader("Conversation")
        for item in st.session_state.history:
            st.markdown(f"**Question:** {item['question']}")
            st.markdown(f"**Answer:** {item['answer']}")
            if item["retrieved_chunks"]:
                with st.expander("View Retrieved Context"):
                    for rank, source in enumerate(item["retrieved_chunks"],start=1):
                        st.markdown(f"### Retrieved Chunk {rank}")
                        st.write(source["chunk"])
                        st.caption(f"Similarity Score: {source['score']:.4f}")
            st.divider()