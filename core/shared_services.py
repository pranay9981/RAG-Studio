import os
from typing import List, Any
import chromadb
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SharedServices:
    def __init__(self):
        # 1. Initialize Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite",
            temperature=0.2,
            max_tokens=1024
        )
        
        # 2. Initialize Embeddings (Switched to HuggingFace to avoid Google API 404s)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # Reusing the same model for multilingual to prevent 'No space left on disk' errors during large model downloads
        self.multilingual_embeddings = self.embeddings
        
        # 3. Initialize ChromaDB Client
        self.chroma_client = chromadb.PersistentClient(path="./data/chroma_db")
        
        # 4. Text Splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
    def load_pdf(self, file_path: str) -> List[Document]:
        """Loads a PDF and returns chunks as LangChain Documents."""
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Create LangChain Documents
        documents = [Document(page_content=chunk, metadata={"source": file_path, "type": "pdf"}) for chunk in chunks]
        return documents

    def extract_response_text(self, response: Any) -> str:
        """Reliably extracts text from LangChain's AIMessage content."""
        content = response.content
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            # Sometimes Gemini returns a list of dicts: [{'type': 'text', 'text': '...'}]
            text_parts = []
            for part in content:
                if isinstance(part, dict) and 'text' in part:
                    text_parts.append(part['text'])
                elif isinstance(part, str):
                    text_parts.append(part)
            return " ".join(text_parts) if text_parts else str(content)
        return str(content)

# Singleton instance for the app to use
services = SharedServices()
