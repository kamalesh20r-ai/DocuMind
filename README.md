# DocuMind

DocuMind is an AI-powered document question answering and study assistant built using Retrieval-Augmented Generation (RAG).

Users can upload a PDF and ask questions based on its content. DocuMind retrieves the most relevant chunks using semantic similarity and generates a grounded answer using the Gemini API.

## Features

- Single PDF upload
- PDF text extraction
- Overlapping text chunking
- Gemini embeddings
- Semantic similarity search
- Top-K relevant chunk retrieval
- Grounded question answering
- Conversation history
- Retrieved context display
- Similarity scores
- Clear conversation option
- Automatic reset when a new PDF is uploaded
- Detection of scanned/image-heavy PDFs
- API error handling

## Tech Stack

- Python
- Streamlit
- Google Gemini API
- Gemini Embeddings
- NumPy
- PyPDF
- python-dotenv
- Git and GitHub

## RAG Workflow

PDF Upload  
↓  
Text Extraction  
↓  
Chunking with Overlap  
↓  
Embedding Generation  
↓  
User Question  
↓  
Question Embedding  
↓  
Cosine Similarity Search  
↓  
Top Relevant Chunks  
↓  
Context + Question  
↓  
Gemini  
↓  
Grounded Answer

## Current Limitations

- Supports one PDF at a time
- Scanned PDFs require OCR
- Typographical errors may affect retrieval
- Tanglish and advanced multilingual queries are not fully optimized

## Future Enhancements

- Multi-PDF support
- OCR for scanned PDFs
- Tamil OCR support
- Tanglish question support
- Typo correction and fuzzy matching
- Query normalization
- Source tracking across documents
- Preferred-language answers
- Improved multilingual support

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt