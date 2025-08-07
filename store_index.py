from src.helper import load_pdf, text_split, download_hugging_face_embeddings, filter_to_minimal_docs
from dotenv import load_dotenv
import os

from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME")

# Initialize Pinecone client
pc = PineconeClient(api_key=pinecone_api_key)

# Create the index if it doesn't exist
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,  # match your embedding dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Load and process data
extracted_data = load_pdf("data/")
filtered_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filtered_data)
embeddings = download_hugging_face_embeddings()

# Store documents in Pinecone via LangChain wrapper
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name,
    namespace="default",  # Optional: set if you use namespaces
    pinecone_api_key=pinecone_api_key  # required for from_documents to connect
)
