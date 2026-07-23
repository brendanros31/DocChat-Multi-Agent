"""
Responsible for generating an initial draft answer using retrieved documents. 
It interacts with an LLM to synthesize responses based on relevant content. 
This step is crucial in the RAG pipeline, ensuring that AI-generated answers are grounded in the provided data.

Key functions of the research agent:
    - Context-aware answer generation: Produces fact-based responses using retrieved documents
    - Structured prompting: Ensures that the AI model adheres to precise instructions for accurate outputs
    - Response sanitization: Cleans and formats LLM responses for better readability
"""

from config.settings import settings
from utils.logging import logger

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_ollama import ChatOllama


class ResearchAgent:
    def __init__(self):
        self.model = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite",
            api_key=settings.GEMINI_API_KEY,
            max_tokens=1024,
            temperature=0.3,
        )

        # # OLLAMA TESTER
        # self.model = ChatOllama(
        #     model="qwen3:8b",
        #     temperature=0.3,
        #     num_ctx=1024
        # )
        logger.info("ModelInference initialized successfully.")

    def sanitize_response(self, response_text: str) -> str:
        """Sanitize response by clearing whitespaces"""
        return response_text.strip()
    
    def generate_prompt(self, question: str, context: str) -> str:
        """Generate structured prompt to create precise and factual answer"""
        prompt = f""" 
        You are an AI assistant designed to provide precise and factual answers based on the given context.
        Instructions:
            - Answer the following question using only the provided context.
            - Be clear, concise, and factual.
            - Return as much information as you can get from the context.        
        Question: {question}
        Context: 
            {context}        
        Provide your answer below:
        """
        return prompt
    
    def generate(self, question: str, documents: list[Document]) -> dict:
        """Generate an intial answer using provided documents"""
        logger.debug(f"ResearchAgent.generate called with question='{question}' and {len(documents)} documents.")
        
        # Failsafe: Don't waste an API call if there are no documents
        if not documents:
            logger.debug("No documents provided. Returning default fallback answer.")
            return {
                "draft_answer": "I cannot answer this question based on the provided documents.",
                "context_used": ""
            }
        
        # Combine the top document contents into one string
        context = "\n\n".join([doc.page_content for doc in documents])
        logger.debug(f"Combined context length: {len(context)} characters.")     

        # Create a prompt for the LLM
        prompt = self.generate_prompt(question, context)
        logger.debug("Prompt created for the LLM.")

        # Call the LLM to generate the answer
        try:
            logger.debug("Sending prompt to model...")
            response = self.model.invoke(prompt)
            logger.debug("LLM response recieved.")
        except Exception as e:
            logger.debug(f"Error during model inference: {e}")
            raise RuntimeError("Failed to generate answer due to a model error.") from e 
        
        # Extract and process LLM's response
        try:
            raw_content = response.content
            
            # Unpack list structure if returned by LangChain
            if isinstance(raw_content, list):
                if raw_content and isinstance(raw_content[0], dict):
                    llm_response = raw_content[0].get("text", "")
                elif raw_content:
                    llm_response = str(raw_content[0])
                else:
                    llm_response = ""
            else:
                llm_response = str(raw_content)
                
            llm_response = llm_response.strip()
            logger.debug(f"LLM response length: {len(llm_response)} characters")
            
        except Exception as e:
            logger.error(f"Unexpected response structure: {e}")
            llm_response = "I cannot answer this question based on the provided documents."

        # Sanitize the response
        draft_answer = self.sanitize_response(llm_response) if llm_response else "I cannot answer this question based on the provided documents."
        logger.info(f"Generated answer: {draft_answer}")

        return {
            "draft_answer": draft_answer,
            "context_used": context,
        }

