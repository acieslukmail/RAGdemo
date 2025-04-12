from PyPDF2 import PdfReader
import uuid

def parse_pdfs(files):
    """
    Parse PDF files and extract text, splitting into chunks.
    
    Args:
        files: List of uploaded PDF files
        
    Returns:
        List of chunks, each with unique ID and text content
    """
    chunks = []
    for file in files:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
                
        # Split text into paragraphs
        paragraphs = text.split("\n\n")
        for para in paragraphs:
            para = para.strip()
            if len(para) > 30:  # Skip very short fragments
                chunks.append({"id": str(uuid.uuid4()), "text": para})
    return chunks