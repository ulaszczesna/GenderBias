import fitz
from pdf2image import convert_from_path
import pytesseract

class PDFExtractor:
    def __init__(self, path):
        self.path = path

    def extract_text(self):
        texts = []
        with fitz.open(self.path) as doc:
            for page in doc:
                page_text = page.get_text()
                texts.append(page_text)

        full_text = "".join(texts).strip()
        if len(full_text) == 0:
            pages = convert_from_path(self.path)
            ocr_texts = []
            for i, page in enumerate(pages):
                page_text = pytesseract.image_to_string(page)
                ocr_texts.append(page_text)
            full_text = "\n".join(ocr_texts)

        return full_text
    
    def chunk_text(self, text):
        paragraphs = text.split('\n')
        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            if len(current_chunk) + len(paragraph) + 1 <= self.chunk_size:
                current_chunk += (paragraph + " ")
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
            
                current_chunk = paragraph + " "

        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks
    
    def extract(self, chunk_size=1000, overlap=200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        text = self.extract_text()
        return self.chunk_text(text)
    
    
    