# 📄 DocuMind

**AI-Powered Document Question Answering & Study Assistant using Retrieval-Augmented Generation (RAG)**

DocuMind is a Streamlit-based AI application that allows users to upload PDF documents and ask questions based on their content.

The system combines PDF text extraction, OCR, semantic embeddings, similarity-based retrieval, and Google Gemini to generate document-grounded answers.

---

## 🎯 Project Objective

The goal of DocuMind is to create a practical AI-powered study assistant that helps users interact with their PDF documents through natural-language questions.

Instead of sending the entire document directly to an LLM, DocuMind retrieves only the most relevant sections of the document and sends those sections as context to the language model.

This follows the **Retrieval-Augmented Generation (RAG)** approach.

---

## ✨ Current Features

- Upload a PDF document
- Extract normal/selectable text from PDFs
- Detect embedded images inside PDF pages
- Perform OCR on image-based content
- Combine native text and OCR-extracted text
- Support mixed PDFs containing both text and images
- Resize large images before OCR for better performance
- Cache repeated image OCR results
- Remove duplicate OCR/native text
- Use fuzzy similarity to detect near-duplicate text
- Split extracted text into overlapping chunks
- Generate semantic embeddings using Gemini
- Process embedding requests in batches
- Handle Gemini embedding rate-limit errors
- Retrieve the Top-3 most relevant chunks
- Generate answers using document context only
- Store conversation history
- Display retrieved chunks and similarity scores
- Clear conversation history
- Avoid unnecessary document reprocessing using Streamlit Session State
- Display document information, chunk count, and embedding count

---

# 🧠 How DocuMind Works

The overall DocuMind pipeline is:

```text
PDF Upload
     ↓
Hybrid Text Extraction
     ↓
Native PDF Text + Image OCR
     ↓
Duplicate Removal
     ↓
Clean Text
     ↓
Chunking
     ↓
Embedding Generation
     ↓
User Question
     ↓
Question Embedding
     ↓
Cosine Similarity
     ↓
Top-3 Relevant Chunks
     ↓
Gemini LLM
     ↓
Document-Grounded Answer
```

---

# 📄 Hybrid PDF Text Extraction

Traditional PDF extraction libraries mainly extract selectable text.

However, many PDFs also contain:

- Scanned pages
- Screenshots
- Images containing text
- Image-based notes
- Mixed text and image content

DocuMind therefore uses a hybrid extraction system.

```text
PDF Page
   ↓
   ├── Native Text Extraction
   │
   └── Embedded Image Extraction
              ↓
          Tesseract OCR
              ↓
Native Text + OCR Text
              ↓
Duplicate Removal
              ↓
Combined Page Text
```

All page content is finally combined into a single text representation for further processing.

---

# 🔍 Native Text Extraction

DocuMind uses **PyMuPDF (`fitz`)** to extract selectable text directly from PDF pages.

Example:

```python
native_text = page.get_text("text")
```

This is useful for normal digital PDFs where text can be selected and copied.

---

# 🖼 OCR Support

DocuMind includes OCR support using:

- **Tesseract OCR**
- **pytesseract**
- **Pillow**
- **PyMuPDF**

Images embedded inside PDF pages are extracted and passed through the OCR engine.

```text
PDF Image
   ↓
Extract Image Bytes
   ↓
Pillow Image
   ↓
Tesseract OCR
   ↓
Extracted Text
```

---

## OCR Optimizations

Large images are resized before OCR processing:

```python
image.thumbnail((1800, 1800))
```

This helps reduce OCR processing time.

DocuMind also caches OCR results using the image reference ID (`xref`).

If the same embedded image appears again, the previously extracted OCR text can be reused instead of processing the image again.

---

# 🧹 Duplicate Text Removal

Some PDF files contain the same content as:

- Selectable PDF text
- Text inside an image

Simply combining both may create duplicate information.

Example:

```text
Native Text:
INNER JOIN combines matching rows.

OCR Text:
INNER J0IN combines matching rows.
```

The OCR output contains a small recognition error:

```text
JOIN → J0IN
```

A simple exact comparison would fail to detect the duplicate.

DocuMind therefore uses Python's:

```python
difflib.SequenceMatcher
```

to calculate similarity between native-text lines and OCR-text lines.

The current similarity threshold is:

```text
0.85
```

If two lines are highly similar, the OCR line is treated as duplicate information and skipped.

---

# ✂️ Text Chunking

After extraction and cleaning, the text is divided into overlapping chunks.

Current configuration:

```text
Chunk Size : 1000 characters
Overlap    : 200 characters
```

Example:

```text
Chunk 1
Characters 0 - 999

Chunk 2
Characters 800 - 1799

Chunk 3
Characters 1600 - 2599
```

The overlap helps preserve context between neighbouring chunks.

---

# ⚠️ Large Document Protection

OCR-based PDFs can sometimes generate extremely large amounts of text.

To avoid excessive embedding requests, DocuMind includes a chunk-count safety limit.

If the extracted document produces too many chunks, processing can be stopped before sending thousands of embedding requests.

This helps prevent:

- Long processing times
- Excessive API usage
- Gemini quota errors
- Unnecessary resource usage

---

# 🧠 Embedding Generation

DocuMind currently uses:

**Google Gemini `gemini-embedding-2`**

Each document chunk is converted into a semantic vector.

These vectors represent the semantic meaning of each chunk.

```text
Document Chunk
      ↓
Gemini Embedding Model
      ↓
Embedding Vector
```

---

# 📦 Embedding Batching

The Gemini embedding API limits the number of contents that can be sent in a single request.

Therefore DocuMind processes chunks in batches.

Example:

```text
253 chunks

Batch 1 → 100 chunks
Batch 2 → 100 chunks
Batch 3 → 53 chunks
```

Each batch is processed separately and all generated embeddings are combined.

---

# ⏳ API Rate-Limit Handling

Gemini free-tier embedding requests may return:

```text
429 RESOURCE_EXHAUSTED
```

when the request quota is exceeded.

DocuMind currently detects this condition and waits before retrying the request.

Example:

```text
Embedding Rate Limit Reached
        ↓
Wait
        ↓
Retry Request
```

This prevents the entire application from immediately failing because of temporary embedding rate limits.

---

# 🔎 Semantic Retrieval

When the user asks a question, DocuMind first creates an embedding for the question.

```text
User Question
      ↓
Question Embedding
```

The question embedding is compared with all document chunk embeddings.

DocuMind currently uses **cosine similarity** with NumPy.

```python
similarity = np.dot(question_embedding, embedding) / (
    np.linalg.norm(question_embedding)
    * np.linalg.norm(embedding)
)
```

Chunks with the highest similarity values are considered the most relevant.

---

# 🎯 Top-K Retrieval

DocuMind currently retrieves:

```text
Top-K = 3
```

So only the three most relevant chunks are sent to Gemini.

```text
Question
   ↓
Similarity Search
   ↓
Top-3 Chunks
   ↓
Context
```

This reduces the amount of irrelevant document content sent to the language model.

---

# 🤖 Answer Generation

DocuMind currently uses:

**Gemini Flash**

for final answer generation.

The model receives:

```text
Retrieved Context
+
User Question
```

The prompt instructs Gemini to answer only from the retrieved document context.

If the answer cannot be found, DocuMind instructs the model to respond:

> The answer is not available in the uploaded document.

---

# 💬 Conversation History

DocuMind maintains conversation history using:

```python
st.session_state
```

For every question, the application stores:

- Question
- Generated answer
- Retrieved chunks
- Similarity scores

Users can view earlier questions and answers during the current session.

---

# 🔎 Retrieved Context Viewer

For valid answers, DocuMind provides an expandable section:

```text
View Retrieved Context
```

Users can inspect:

- Retrieved chunk text
- Similarity score
- Retrieval ranking

This makes the RAG process easier to understand and verify.

---

# 🧹 Clear Conversation

Users can clear the current question-answer history using:

```text
Clear Conversation
```

This resets the conversation while keeping the processed document available.

---

# ⚡ Streamlit Session State Optimization

Streamlit reruns the Python script whenever the user interacts with the application.

Without optimization, this could cause:

```text
PDF
↓
OCR
↓
Chunking
↓
Embedding
```

to run repeatedly.

DocuMind uses Streamlit Session State to store:

- Processed filename
- Document chunks
- Embeddings
- Conversation history
- Extracted character count

This helps prevent unnecessary processing of the same document during the active session.

---

# 📊 Document Information

DocuMind displays useful document statistics including:

- Uploaded filename
- Characters extracted
- Number of chunks created
- Number of embeddings stored
- Extracted text preview

---

# 🛠 Technology Stack

## Programming Language

- Python

## User Interface

- Streamlit

## PDF Processing

- PyMuPDF (`fitz`)
- PyPDF
- Pillow

## OCR

- Tesseract OCR
- pytesseract

## AI / RAG

- Google Gemini API
- Gemini Embedding Model
- Gemini Flash Model

## Retrieval

- NumPy
- Cosine Similarity

## Text Processing

- Python `difflib.SequenceMatcher`

## Environment Management

- Python Virtual Environment
- python-dotenv

## Version Control

- Git
- GitHub

---

# 📁 Project Structure

```text
DocuMind/
│
├── app.py
├── build_index.py
├── ask_question.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── Testing_Files/
    ├── test_api.py
    ├── test_ocr.py
    └── test_hybrid.py
```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/kamalesh20r-ai/DocuMind.git
```

Move into the project directory:

```bash
cd DocuMind
```

---

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

---

## 3. Activate the Virtual Environment

### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

Successful activation should look similar to:

```text
(.venv) PS C:\Projects\DocuMind>
```

---

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

OCR-related packages include:

```bash
pip install pymupdf pillow pytesseract
```

---

# 🔑 Gemini API Configuration

Create a file named:

```text
.env
```

inside the project root.

Example:

```text
DocuMind/
├── .env
├── app.py
├── README.md
└── ...
```

Add:

```text
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
```

Do not upload the API key to GitHub.

Make sure `.gitignore` contains:

```text
.env
.venv/
__pycache__/
```

---

# 🔤 Tesseract OCR Installation

The Python package:

```text
pytesseract
```

is only a Python wrapper.

The actual **Tesseract OCR Engine** must also be installed separately.

After installation on Windows, verify it using:

```powershell
& "C:\Program Files\Tesseract-OCR\tesseract.exe" --version
```

The current local development configuration uses:

```python
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)
```

The Tesseract path may be different on another computer.

---

# ▶️ Running DocuMind

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then run:

```bash
python -m streamlit run app.py
```

Streamlit will display a local URL in the terminal.

Open the URL in a browser to use DocuMind.

---

# 🧪 Testing

The project contains separate test files inside:

```text
Testing_Files/
```

Examples include:

```text
test_api.py
test_ocr.py
test_hybrid.py
```

These files can be used to test individual components before integrating them into the main application.

---

# ⚠️ Current Limitations

The current DocuMind version has the following limitations:

### Single PDF Support

Only one PDF can currently be processed at a time.

### OCR Processing Time

Large scanned PDFs can take significant time to process.

### Handwritten Text

Tesseract works best with clear printed text.

Handwritten notes may produce lower OCR accuracy.

### Tamil OCR

Tamil OCR is not fully implemented in the current version.

### Very Large Documents

Very large scanned PDFs can generate excessive OCR text and thousands of chunks.

### API Rate Limits

Gemini embedding requests are subject to API quotas and rate limits.

### Character-Based Chunking

The current version uses character-based chunking instead of semantic or paragraph-aware chunking.

### No Page-Level Citations

Answers currently show retrieved chunks but do not yet display exact page-number citations.

### Temporary Session Storage

Processed documents and conversation history are stored temporarily using Streamlit Session State.

They are not permanently stored.

---

# 🔮 Planned Improvements

Future versions of DocuMind may include:

## 🚀 Performance

- Local embedding models
- Faster large-document processing
- Reduced dependency on embedding API quotas
- Better OCR caching
- Processing progress indicators
- Persistent document indexes

## 📄 Document Processing

- Multi-PDF support
- Better OCR preprocessing
- Semantic chunking
- Paragraph-based chunking
- Better OCR noise removal

## 🌐 Multilingual Support

- Tamil OCR
- Better multilingual document support
- Tanglish question support
- Tanglish-to-English query normalization
- Same-language answer generation

## 🔎 Retrieval Improvements

- Retrieval reranking
- Hybrid search
- Better chunk selection
- Query rewriting
- RAG evaluation

## 📚 Source Tracking

- Page-level source tracking
- PDF filename tracking
- Source citations in generated answers

## ✍️ Query Handling

- Typo correction
- Fuzzy query handling
- Query normalization

## 🖼 Advanced OCR

- Better handwritten-text recognition
- Improved scanned-document preprocessing
- Language-aware OCR

---

# 🚀 Planned Architecture Improvement

One major future optimization is replacing document embedding API calls with a local embedding model.

Current:

```text
Chunks
   ↓
Gemini Embedding API
   ↓
Embedding Vectors
```

Future:

```text
Chunks
   ↓
Local Embedding Model
   ↓
Embedding Vectors
   ↓
Local Similarity Search
   ↓
Top Relevant Chunks
   ↓
Gemini Only for Final Answer
```

This can reduce:

- API calls
- Embedding quota issues
- Rate-limit errors
- Processing delays for large documents

---

# 📌 Current Project Status

```text
PDF Upload                     ✅
Native Text Extraction         ✅
Image Extraction               ✅
OCR V1                         ✅
Hybrid Text Extraction         ✅
Duplicate Detection            ✅
Fuzzy Duplicate Removal        ✅
Text Chunking                  ✅
Gemini Embeddings              ✅
Embedding Batching             ✅
Rate-Limit Handling            ✅
Cosine Similarity Retrieval    ✅
Top-K Retrieval                ✅
Gemini Answer Generation       ✅
Conversation History           ✅
Retrieved Context Viewer       ✅
Streamlit Session State        ✅
GitHub Integration             ✅
Streamlit Deployment           ✅

Tamil OCR                      ⏳ Planned
Handwriting OCR Improvement    ⏳ Planned
Multi-PDF Support              ⏳ Planned
Page-Level Citations           ⏳ Planned
Local Embeddings               ⏳ Planned
Tanglish Support               ⏳ Planned
Semantic Chunking              ⏳ Planned
```

---

# 👨‍💻 Developed By

**Kamalesh R**

AI & Data Science Student

Project:

**DocuMind — AI-Powered Document Question Answering & Study Assistant using Retrieval-Augmented Generation (RAG)**

---

## ⭐ Project Summary

DocuMind demonstrates how modern AI systems can combine:

```text
Document Processing
+
OCR
+
Embeddings
+
Semantic Search
+
Retrieval-Augmented Generation
+
Large Language Models
```

to transform static PDF documents into interactive AI-powered knowledge sources.