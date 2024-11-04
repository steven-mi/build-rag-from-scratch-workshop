import PyPDF2

from embedding_db import EmbeddingDatabase
from embedding_model import EmbeddingModel


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    text = ""
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""  # Handle cases where text extraction might fail
            page_text = page_text.replace("\t", " ")
            text += page_text
    return text


def create_text_chunks(text: str, chunk_size: int, overlap_size: int) -> list[str]:
    """Create overlapping text chunks from the extracted text."""
    chunks = []
    for i in range(0, len(text) - chunk_size + 1, chunk_size - overlap_size):
        chunks.append(text[i : i + chunk_size])
    return chunks


# Extract text from PDF and create chunks
text = extract_text_from_pdf("../data/food_lab_green_chapter.pdf")
text_chunks = create_text_chunks(text, chunk_size=1000, overlap_size=200)

# Initialize embedding model and database
model = EmbeddingModel()
db = EmbeddingDatabase(model)

# Add text chunks to the database and save the state
db.add_documents(text_chunks)
db.save_state()