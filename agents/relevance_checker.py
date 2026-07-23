"""
Responsible for determining whether retrieved documents contain relevant 
information to answer a given question. 

It uses an ensemble retriever to fetch document chunks and then leverages 
IBM watsonX AI for classification. 

The goal is to categorize relevance into three possible labels:
    - "CAN_ANSWER" – The documents provide sufficient information for a full answer.
    - "PARTIAL" – The documents mention the topic but lack complete details.  
    - "NO_MATCH" – The documents do not discuss the question at all.
"""

import os
import re
from config.settings import settings
from utils.logging import logger

from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_ollama import ChatOllama


class RelevanceChecker:
    def __init__(self):
        self.model = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite",
            api_key=settings.GEMINI_API_KEY,
            max_tokens=300,
            temperature=0.3,
        )

        # # OLLAMA TESTER
        # self.model = ChatOllama(
        #     model="qwen3:8b",
        #     temperature=0.3,
        #     num_ctx=1024
        # )
        logger.info("ModelInference initialized successfully.")

    
    def check(self, question: str, retriever, k=3) -> str:
        """
        1. Retrieve the top-k document chunks from the global retriever.
        2. Combine them into a single text string.
        3. Pass that text + question to the LLM for classification.        
        
        Returns: "CAN_ANSWER", "PARTIAL", or "NO_MATCH".
        """

        # Retrieve docs from Ensemble Retriever
        logger.debug(f"RelevanceChecker.check called with question= '{question}' and k = {k}")
        top_docs = retriever.invoke(question)
        
        # NO document - failsafe
        if not top_docs:
            logger.info("No documents returned from retriever.invoke(). Classifying as NO_MATCH")
            return "NO_MATCH"
        
        document_content = "\n\n".join(doc.page_content for doc in top_docs[:k])

        prompt = f"""
        You are an AI relevance checker between a user's question and provided document content.        
        Instructions:
            - Classify how well the document content addresses the user's question.
            - Respond with only one of the following labels: CAN_ANSWER, PARTIAL, NO_MATCH.
            - Do not include any additional text or explanation.        
        Labels:
            1) "CAN_ANSWER": The passages contain enough explicit information to fully answer the question.
            2) "PARTIAL": The passages mention or discuss the question's topic but do not provide all the details needed for a complete answer.
            3) "NO_MATCH": The passages do not discuss or mention the question's topic at all.        
        Important: If the passages mention or reference the topic or timeframe of the question in any way, even if incomplete, respond with "PARTIAL" instead of "NO_MATCH".        
        Question: {question}
        Passages: {document_content}        
        Respond ONLY with one of the following labels: CAN_ANSWER, PARTIAL, NO_MATCH
        """

        # Call LLM
        try:
            response = self.model.invoke(prompt)
        except Exception as e:
            logger.error(f"Error during model interference: {e}")
            return "NO_MATCH"
        
        # Extract content form response
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

            llm_response = llm_response.strip().upper()
            logger.debug(f"LLM response: {llm_response}")

        except (AttributeError, IndexError) as e:
            logger.error(f"Unexpected response structure: {e}")
            return "NO_MATCH"
        
        # Valid llm response
        print(f"Checker response: {llm_response}")
        valid_labels = {"CAN_ANSWER", "PARTIAL", "NO_MATCH"}

        # Check LLM response
        if llm_response not in valid_labels:
            logger.debug("LLM did not respond with a valid label. Forcing 'NO_MATCH'.")
            classification = "NO_MATCH"
        else:
            logger.debug(f"Classification recognized as '{llm_response}'.")
            classification = llm_response

        return classification
