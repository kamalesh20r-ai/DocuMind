from pypdf import PdfReader
reader = PdfReader("DATA/DBMS Lab Manual.pdf")
text = ""
for page in reader.pages :
    page_text = page.extract_text()
    if page_text :
        text += page_text + "\n"
print(text)