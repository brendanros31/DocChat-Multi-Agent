from langgraph.graph import StateGraph, END
from typing import TypedDict
import logging

from .research_agent import ResearchAgent
from .verification_agent import VerificationAgent
from .relevance_checker import RelevanceChecker

from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    question: str
    documents: list[Document]
    draft_answer: str
    verification_report: str
    is_relevant: bool
    retriever: EnsembleRetriever

class AgentWorkFlow:
    def __init__(self):
        self.researcher = ResearchAgent()
        self.verifier = VerificationAgent()
        self.relevance_checker = RelevanceChecker()
        self.compiled_workflow = self.build_workflow()

    # Compile once at initialization
    def build_workflow(self):
        """
        Constructs the state machine (nodes and edges) for the agent pipeline.
        
        Returns:
            CompiledGraph: The executable LangGraph application.
        """
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("check_relevance", self._check_relevance_step)
        workflow.add_node("research", self._research_step)
        workflow.add_node("verify", self._verification_step)

        # Set entry point
        workflow.set_entry_point("check_relevance")

        # Add edges
        workflow.add_conditional_edges(
            "check_relevance",
            self._decide_after_relevance_check,
            {
                "relevant": "research",
                "irrelevant": END,
            }
        )
        workflow.add_edge("research", "verify")
        workflow.add_conditional_edges(
            "verify",
            self._decide_next_step,
            {
                "re_research": "research",
                "end": END,
            }
        )

        # Compile and return
        return workflow.compile()
        
    def _check_relevance_step(self, state: AgentState) -> dict:
        """
        Node 1: Evaluates if the user's question can actually be answered by our database.

        Returns:
            A state update dict that LangGraph merges into the global AgentState.
        """
        retriever = state["retriever"]

        # Ask the RelevanceChecker if question is valid
        classification = self.relevance_checker.check(
            question=state["question"],
            retriever=retriever,
            # k=20,
            k=10,
        )

        # Translate classification into boolean
        if classification == "CAN_ANSWER":
            return {"is_relevant": True}
        elif classification == "PARTIAL":
            return {"is_relevant": True}
        else:
            return{
                "is_relevant": False,
                "draft_answer": "This question isn't related (or there is no data) for your query. Please ask another question."
            }
        
    def _decide_after_relevance_check(self, state: AgentState) -> str:
        """
        Router: Reads the 'is_relevant' flag and dictates the next edge.
        """
        decision = "relevant" if state["is_relevant"] else "irrelevant"
        logger.info(f"[DEBUG] _decide_after_relevance_check -> {decision}")
        return decision
    
    def full_pipeline(self, question: str, retriever: EnsembleRetriever):
        """
        The main public execution method. Takes a raw question, retrieves initial context, 
        initializes the state, and runs it through the compiled LangGraph workflow.
        """
        try:
            logger.info(f"[DEBUG] Starting full_pipeline with question='{question}'")

            # Pre-fetch initial context to seed agent state
    # The [:4] slices the list to only keep the top 5 most relevant chunks
            documents = retriever.invoke(question)[:15]
            logger.info(f"Retrieved {len(documents)} relevant documents (from .invoke)") 

            # Build starting state container
            initial_state = AgentState(
                question=question,
                documents=documents,
                draft_answer="",
                verification_report="",
                is_relevant=False,
                retriever=retriever,
            )

            # Invoke graph
            final_state = self.compiled_workflow.invoke(initial_state)

            return{
                "draft_answer": final_state["draft_answer"],
                "verification_report": final_state["verification_report"],
            }
        
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            raise e

    def _research_step(self, state: AgentState) -> dict:
        """
        Node 2: Research
        Takes the user's question and the retrieved context, and asks the Researcher agent to draft an answer.
        """
        logger.info(f"[DEBUG] Entered research step for question: '{state['question']}'")
        result = self.researcher.generate(state["question"], state["documents"])
        logger.info("[DEBUG] Researcher successfully generated a draft answer.")     

        # Overwrite 'draft_answer' with return value
        return {"draft_answer": result["draft_answer"]}

    def _verification_step(self, state: AgentState) -> dict:
        """
        Node 3: Verification
        Reviews the drafted answer against the original source documents to ensure the model didn't hallucinate.
        """
        logger.info("[DEBUG] Entered _verification_step. Verifying the draft answer...")
        result = self.verifier.check(state["draft_answer"], state["documents"])
        logger.info("[DEBUG] VerificationAgent returned a verification report.")

        # Updates global state so router can read final report
        return {"verification_report": result["verification_report"]}
    
    def _decide_next_step(self, state: AgentState) -> str:
        """
        Router 2: Evaluates the verification report.
        If the answer is hallucinated or irrelevant, it forces the workflow to loop back and try again.
        """
        verification_report = state["verification_report"]
        logger.info(f"[DEBUG] _decide_next_step with verification_report='{verification_report}'")

        # Look for specific failure flags in the verifier's text output
        if "Supported: NO" in verification_report or "Relevant: NO" in verification_report:
            logger.info("[DEBUG] Verification indicates re-research needed.")
            return "re_research"
        else:
            logger.info("[DEBUG] Verification successful, ending workflow.")
            return "end"

