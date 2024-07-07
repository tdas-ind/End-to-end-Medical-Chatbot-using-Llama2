from src.helper import load_pdf, text_split, download_embedding
from langchain_pinecone import PineconeVectorStore
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)
index_name="medicalchatbot"
extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embedding = download_embedding()


docsearch = PineconeVectorStore.from_documents(text_chunks, embedding, index_name=index_name)