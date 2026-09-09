from pypdf import PdfReader
reader = PdfReader("DATA/DBMS Lab Manual.pdf")
text =""
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
print("Total Characters :",len(text))
print("Total Chunks :",len(chunks))
print("\n First Chunk \n")
print(chunks[0])