import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
import os
from google import genai
from google.genai import types
import numpy as np
import fitz
from PIL import Image
import io
from difflib import SequenceMatcher
import pytesseract
import time
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    api_key = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=api_key)
def remove_duplicate_text(native_text, ocr_text, threshold=0.85):
    native_lines = [line.strip() for line in native_text.splitlines() if line.strip()]
    ocr_lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]
    unique_ocr_lines = []
    for ocr_line in ocr_lines:
        is_duplicate = False
        for native_line in native_lines:
            similarity = SequenceMatcher(None,ocr_line.lower(),native_line.lower()).ratio()
            if similarity >= threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_ocr_lines.append(ocr_line)
    combined_lines = native_lines + unique_ocr_lines
    return "\n".join(combined_lines)
def extract_hybrid_text(uploaded_file):
    uploaded_file.seek(0)
    pdf_bytes = uploaded_file.read()
    pdf = fitz.open(stream=pdf_bytes,filetype="pdf")
    full_text = ""
    ocr_cache = {}
    for page in pdf:
        native_text = page.get_text("text")
        page_ocr_text = ""
        images = page.get_images(full=True)
        for image_info in images:
            xref = image_info[0]
            if xref in ocr_cache:
                page_ocr_text += ocr_cache[xref] + "\n"
                continue
            image_data = pdf.extract_image(xref)
            image_bytes = image_data["image"]
            image = Image.open(io.BytesIO(image_bytes))
            width, height = image.size
            if width < 200 or height < 100:
                continue
            image.thumbnail((1800, 1800))
            ocr_text = pytesseract.image_to_string(image,config="--psm 6")
            ocr_cache[xref] = ocr_text
            page_ocr_text += ocr_text + "\n"
        combined_page_text = remove_duplicate_text(native_text,page_ocr_text)
        full_text += combined_page_text + "\n"
    return full_text
def create_embeddings_in_batches(client, chunks, batch_size=100):
    all_embeddings = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        result = client.models.embed_content(model="gemini-embedding-2",contents=batch)
        batch_embeddings = [embedding.values for embedding in result.embeddings]
        all_embeddings.extend(batch_embeddings)
    return all_embeddings
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
if "extracted_characters" not in st.session_state:
    st.session_state.extracted_characters = 0
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
    if st.session_state.processed_file_name != uploaded_file.name:
        st.session_state.history = []
        with st.spinner("Extracting text and images..."):
            uploaded_file.seek(0)
            text = extract_hybrid_text(uploaded_file)
        if len(text.strip()) < 100:
            st.warning("Very little readable text could be extracted from this document.")
            st.stop()
        chunk_size = 1000
        overlap = 200
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            chunks.append(chunk)
        if len(chunks) > 1000:
            st.warning(f"This document produced {len(chunks)} chunks. " "Too much OCR text was extracted.")
            st.stop()
        st.session_state.extracted_characters = len(text)
    if (st.session_state.processed_file_name!= uploaded_file.name):
        st.session_state.history = []
        contents = []
        for chunk in chunks:
            content = types.Content(parts=[types.Part.from_text(text=chunk)])
            contents.append(content)
        embeddings = []
        with st.spinner("Creating document embeddings..."):
            batch_size = 100
            for start in range(0, len(contents), batch_size):
                batch = contents[start:start + batch_size]
                while True:
                    try:
                        result = client.models.embed_content(model="gemini-embedding-2",contents=batch)
                        break
                    except Exception as e:
                        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                            st.warning("Embedding rate limit reached. " "Waiting 60 seconds before retrying...")
                            time.sleep(60)
                        else:
                            raise e
                for embedding in result.embeddings:
                    embeddings.append(embedding.values)
        st.session_state.chunks = chunks
        st.session_state.embeddings = embeddings
        st.session_state.processed_file_name = uploaded_file.name
        st.rerun()
    else:
        st.success(f"Document processed successfully! "f"{len(st.session_state.chunks)} chunks indexed.")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Characters Extracted",st.session_state.extracted_characters)
        with col2:
            st.metric("Chunks Created",len(st.session_state.chunks))
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