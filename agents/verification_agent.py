"""
Responsible for fact-checking and validating generated answers 
using the retrieved documents. 

This agent ensures that the AI-generated response is:
    - Supported by factual evidence from the documents
    - Free from contradictions or misinformation
    - Relevant to the original question
"""


from config.settings import settings
from utils.logging import logger

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama


class VerificationAgent:
    def __init__(self):
        self.model = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite",
            api_key=settings.GEMINI_API_KEY,
            max_tokens=1024,
            temperature=0.0,
        )

        # # OLLAMA TESTER
        # self.model = ChatOllama(
        #     model="qwen3:8b",
        #     temperature=0.3,
        #     num_ctx=1024
        # )
        logger.info("ModelInference initialized successfully.")

    def sanitize_response(self, response_text: str) -> str:
        """Sanitize the LLM's response by stripping unnecessary whitespace."""
        return response_text.strip()
    
    def generate_prompt(self, answer: str, context: str) -> str:
        """Generate a structured prompt for the LLM to verify the answer against the context."""
        prompt = f"""
        You are an AI assistant designed to verify the accuracy and relevance of answers based on the provided context.
        Instructions:
            - Verify the following answer against the provided context.
            - Check for:
                1. Direct/indirect factual support (YES/NO)
                2. Unsupported claims (list any if present)
                3. Contradictions (list any if present)
                4. Relevance to the question (YES/NO)
            - Provide additional details or explanations where relevant.
            - Respond in the exact format specified below without adding any unrelated information.        
        Format:
        Supported: YES/NO
        Unsupported Claims: [item1, item2, ...]
        Contradictions: [item1, item2, ...]
        Relevant: YES/NO
        Additional Details: [Any extra information or explanations]  

        Answer: {answer}
        Context:
        {context}  
              
        Respond ONLY with the above format.
        """
        return prompt

    def parse_verification_response(self, response_text: str) -> dict:
        """Parse the LLM's verification response into a structured dictionary."""
        try:
            lines = response_text.split('\n')
            verification = {}

            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().title()
                    value = value.strip()

                    # Handle List Fields
                    if key in {"Unsupported Claims", "Contradictions"}:
                        if value.startswith('[') and value.endswith(']'):
                            items = value[1:-1].split(',')
                            # Safely remove quotes, spaces, and discard completely empty strings
                            cleaned_items = [item.strip().strip('"').strip("'") for item in items if item.strip()]
                            verification[key] = cleaned_items
                        else:
                            # Fallback in case LLM outputs "None" instead of "[]"
                            verification[key] = []

                    # Handle YES/NO/PARTIAL String Fields
                    elif key in {"Supported", "Relevant"}:
                        verification[key] = value.upper()

                    # 3. Handle Paragraph String Fields
                    elif key == "Additional Details":
                        verification[key] = value

            # Ensure all expected keys are strictly present in final dictionary
            default_keys = {
                "Supported": "NO",
                "Unsupported Claims": [],
                "Contradictions": [],
                "Relevant": "NO",
                "Additional Details": "None"
            }

            for req_key, default_val in default_keys.items():
                if req_key not in verification:
                    verification[req_key] = default_val

            return verification
        
        except Exception as e:
            logger.error(f"Error parsing verification response: {e}")
            return None
        
    def format_verification_report(self, verification: dict) -> str:
        """Format the verification report dictionary into a readable paragraph."""
        supported = verification.get("Supported", "NO")
        unsupported_claims = verification.get("Unsupported Claims", [])
        contradictions = verification.get("Contradictions", [])
        relevant = verification.get("Relevant", "NO")
        additional_details = verification.get("Additional Detials", "")
        report = f"Supported: {supported}\n"

        if unsupported_claims:
            report += f"Unsupported Claims: {', '.join(unsupported_claims)}\n"
        else:
            report += f"Unsupported Claims: None\n"

        if contradictions:
            report += f"Contradictions: {', '.join(contradictions)}\n"
        else:
            report += f"Contradictions: None\n"
        
        report += f"Relevant: {relevant}\n"

        if additional_details:
            report += f"Additional Details: {additional_details}\n"
        else:
            report += f"Additional Details: None\n"

        return report
    
    def check(self, answer: str, documents: list[Document]) -> dict:
        """Verify the answer against the provided documents"""
         # Combine all document contents into one string without truncation
        logger.debug(f"VerificationAgent.check called with answer='{answer}' and {len(documents)} documents.")
        context = "\n\n".join([doc.page_content for doc in documents])
        logger.debug(f"Combined context length: {len(context)} characters.")

        # Create prompt for LLM to verify answer
        prompt = self.generate_prompt(answer, context)
        logger.debug("Prompt created for the LLM.")

        # Call the LLM to generate the verification report
        try:
            logger.debug("Sending prompt to the model...")
            response = self.model.invoke(prompt)
            logger.debug("LLM response recieved.")
        except Exception as e:
            print(f"Error during model inference: {e}")
            raise RuntimeError("Failed to verify answer due to a model error.") from e
        
        # Extract LLM response
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
            logger.debug(f"Raw LLM response:\n{llm_response}")
            
        except Exception as e: # Catching broader exception to handle list mismatches
            logger.error(f"Unexpected response structure: {e}")
            verification_report = {
                "Supported": "NO",
                "Unsupported Claims": [],
                "Contradictions": [],
                "Relevant": "NO",
                "Additional Details": "Invalid response structure from the model."
            }
            verification_report_formatted = self.format_verification_report(verification_report)
            logger.debug(f"Verification report:\n{verification_report_formatted}")
            logger.debug(f"Context used: {context}")

            return {
                "verification_report": verification_report_formatted,
                "context_used": context,
            }
        
        # Sanitize response
        sanitized_response = self.sanitize_response(llm_response) if llm_response else ""
        if not sanitized_response:
            logger.info("LLM returned an empty response.")
            verification_report = {
                "Supported": "NO",
                "Unsupported Claims": [],
                "Contradictions": [],
                "Relevant": "NO",
                "Additional Details": "Empty response from the model.",
            }
        else:
            # Parse response into expected format
            verification_report = self.parse_verification_response(sanitized_response)
            if verification_report is None:
                logger.info("LLM did not respond with the expected format. Using default verification report.")
                verification_report = {
                    "Supported": "NO",
                    "Unsupported Claims": [],
                    "Contradictions": [],
                    "Relevant": "NO",
                    "Additional Details": "Failed to parse the model's response."
                }

        # Format the verification report into a paragraph
        verification_report_formatted = self.format_verification_report(verification_report)
        logger.debug(f"Verification report:\n{verification_report_formatted}")
        logger.debug(f"Context used: {context}")

        return {
            "verification_report": verification_report_formatted,
            "context_used": context
        }