from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
load_dotenv()
import os

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")


extracted_data =load_pdf_file(data='Data/')
text_chunks=text_split(extracted_data)
embeddings=download_hugging_face_embeddings()

pc = Pinecone(api_key="pcsk_2VxGKZ_Q1fXmNN66sdzST55PwFrAtCSicvedzB5YkUYcn3wjtR2hCfoS2PdsTdU3sD44F5")

index_name="medicalbot"


pc.create_index(
    name=index_name,
    dimension=384, 
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)

docsearch= PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name
)

