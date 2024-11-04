import PyPDF2

from embedding_db import EmbeddingDatabase
from embedding_model import EmbeddingModel


def extract_text_from_pdf(pdf_path: str) -> str:
    # TODO: Extract text from a PDF file
    raise NotImplementedError()

def create_text_chunks(text: str, chunk_size: int, overlap_size: int) -> list[str]:
    # TODO: Create overlapping text chunks from the extracted text
    raise NotImplementedError()


text = extract_text_from_pdf("../data/food_lab_green_chapter.pdf")
text_chunks = create_text_chunks(text, chunk_size=1000, overlap_size=200)

# Initialize embedding model and database
model = EmbeddingModel()
db = EmbeddingDatabase(model)

# Add text chunks to the database and save the state
db.add_documents(text_chunks)
db.save_state()