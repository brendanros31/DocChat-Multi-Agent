from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

import logging
from config.settings import settings


logger = logging.getLogger(__name__)

class RetrieverBuilder:
    def __init__(self):
        """Initialize retriever builder with embeddings"""
        # self.embeddings = GoogleGenerativeAIEmbeddings(
        #     model="gemini-embedding-2",
        #     api_key=settings.GEMINI_API_KEY
        # )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
    
    def build_hybrid_retriever(self, docs):
        """Build a hybrid retriever using BM25 and vector-based retrieval"""
        try:
            # Create chroma verctor DB
            vector_store = Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings,
                persist_directory=settings.CHROMA_DB_PATH,
            )
            logger.info("Vector store created successfully.")

            # Create BM25 retriever
            bm25 = BM25Retriever.from_documents(docs)
            logger.info("BM25 retriever created successfully.")

            # Create vector based retriever
            vector_retriever = vector_store.as_retriever(search_kwargs={"k": settings.VECTOR_SEARCH_K})
            logger.info("Vector retriever created successfully.")

            # Combine retrievers to build hybrid
            hybrid_retriever = EnsembleRetriever(
                retrievers=[bm25, vector_retriever],
                weights=settings.HYBRID_RETRIEVER_WEIGHTS,
            )
            logger.info("Hybrid retrievers created successfully.")

            return hybrid_retriever
        
        except Exception as e:
            logger.error(f"Failed to build hybrid retriever: {e}")
            raise