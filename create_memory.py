from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Set data path
DATA_PATH = "Data/"

# Load PDF files from the directory
def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob="**/*.pdf",  # Recursively matches all PDFs
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

documents = load_pdf_files(data=DATA_PATH)
print("Loaded PDF pages:", len(documents))

# Split documents into text chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = create_chunks(extracted_data=documents)
print("Number of text chunks:", len(text_chunks))

# Create embedding model
def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model = get_embedding_model()

# Create Chroma DB and persist it locally
DB_CHROMA_PATH = "vectorstore/db_chroma"

db = Chroma.from_documents(
    documents=text_chunks,
    embedding=embedding_model,
    persist_directory=DB_CHROMA_PATH
)

db.persist()  # Saves the DB to disk
print(f"Chroma vectorstore saved at {DB_CHROMA_PATH}")

