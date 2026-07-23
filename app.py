import gradio as gr
import hashlib
import os

from document_processor.file_handler import DocumentProcessor
from retriever.builder import RetrieverBuilder
from agents.workflow import AgentWorkFlow
from config import settings, constants
from utils.logging import logger

import atexit
import shutil
from pathlib import Path


def cleanup_cache():
    """Automatically clears vector DB and document cache on shutdown."""
    folders_to_clear = ["chroma_db", "document_cache"]
    
    for folder in folders_to_clear:
        folder_path = Path(folder)
        if folder_path.exists() and folder_path.is_dir():
            try:
                shutil.rmtree(folder_path)
                print(f"[Cleanup] Successfully deleted cache folder: {folder}")
            except Exception as e:
                print(f"[Cleanup] Error deleting {folder}: {e}")

# Register the function to run automatically upon exit
atexit.register(cleanup_cache)


# Define example data
# (i.e., question + paths to documents relevant to that question)
EXAMPLES = {
    "Google 2024 Environmental Report": {
        "question": "Retrieve the data center PUE efficiency values in Singapore 2nd facility in 2019 and 2022. Also retrieve regional average CFE in Asia pacific in 2023",
        "file_paths": ["examples/google-2024-environmental-report.pdf"]    },
    "DeepSeek-R1 Technical Report": {
        "question": "Summarize DeepSeek-R1 model's performance evaluation on all coding tasks against OpenAI o1-mini model",
        "file_paths": ["examples/DeepSeek Technical Report.pdf"]
    }
}


def main():
    processor = DocumentProcessor()
    retriever_builder = RetrieverBuilder()
    workflow = AgentWorkFlow()

    # Custom styling
    css = """
    .title {
        font-size: 1.5em !important; 
        text-align: center !important;
        color: #3b82f6; 
    }
    .subtitle {
        font-size: 1em !important; 
        text-align: center !important;
        color: #3b82f6; 
    }
    .text {
        text-align: center;
    }
    /* Force Gradio primary buttons to blue */
    button.primary, .gr-button-primary {
        background-color: #3b82f6 !important;
        border-color: #3b82f6 !important;
    }
    """

    js = """
    function createGradioAnimation() {
        var container = document.createElement('div');
        container.id = 'gradio-animation';
        container.style.fontSize = '2em';
        container.style.fontWeight = 'bold';
        container.style.textAlign = 'center';
        container.style.marginBottom = '20px';
        container.style.color = '#3b82f6';        
        var text = 'Welcome to DocChat 🐥!';

        for (var i = 0; i < text.length; i++) {
            (function(i){
                setTimeout(function(){
                    var letter = document.createElement('span');
                    letter.style.opacity = '0';
                    letter.style.transition = 'opacity 0.1s';
                    letter.innerText = text[i];                    
                    container.appendChild(letter);                    
                    setTimeout(function() {
                        letter.style.opacity = '0.9';
                    }, 50);
                }, i * 250);
            })(i);
        }        
        var gradioContainer = document.querySelector('.gradio-container');
        gradioContainer.insertBefore(container, gradioContainer.firstChild);        
        return 'Animation created';
    }
    """
    with gr.Blocks(title="DocChat 🍁") as demo:
        gr.Markdown("## DocChat 🍁 : powered by Docling 🐥 and LangGraph")
        gr.Markdown("# How it works ✨:", elem_classes="title")
        gr.Markdown("📤 Upload your document(s), enter your query then press Submit 📝", elem_classes="text")
        gr.Markdown("Or you can select one of the examples from the drop-down menu, select Load Example then press Submit 📝", elem_classes="text")
        gr.Markdown("⚠️ Note: DocChat only accepts documents in these formats: '.pdf', '.docx', '.txt', '.md'", elem_classes="text")        # 2) Maintain the session state for retrieving doc changes
        
        # Maintain session state for retrieving doc changes
        session_state = gr.State({
            "file_hashes": frozenset(),
            "retriever": None,
        })

        # Layout
        with gr.Row():
            with gr.Column():
                # Section for examples
                gr.Markdown("### Example 📂")
                example_dropdown = gr.Dropdown(
                    label="Select an example: ",
                    choices=list(EXAMPLES.keys()),
                    value=None, # Initially unselected
                )
                load_example_btn = gr.Button("Load Example 🛠️")

                # Standard input components
                files = gr.Files(label="📄 Upload Documents", file_types=constants.ALLOWED_TYPES)
                question = gr.Textbox(label="❓ Question", lines=3)
                submit_btn = gr.Button("Submit 🚀")            
            
            with gr.Column():
                answer_output = gr.Textbox(label="🐥 Answer", interactive=False)
                verification_output = gr.Textbox(label="✅ Verification Report")

        def load_example(example_key: str):
            """
            Given a key like 'Example 1', 
            read the relevant docs from disk and return
            them as file-like objects, plus the example question.
            """
            if not example_key or example_key not in EXAMPLES:
                return []
            
            ex_data = EXAMPLES[example_key]
            question = ex_data["question"]
            file_paths = ex_data["file_paths"]

            loaded_files = []

            for path in file_paths:
                if os.path.exists(path):
                    loaded_files.append(path)
                else:
                    logger.warning(f"File not found: {path}")
            

            return loaded_files, question
        
        load_example_btn.click(
            fn=load_example,
            inputs=[example_dropdown],
            outputs=[files, question],
        )

        # Standard flow for submission
        def process_question(question_text: str, uploaded_files: list, state: dict):
            """Handle questions with document caching."""
            try:
                if not question_text.strip():
                    raise ValueError("❌ Question cannot be empty")
                if not uploaded_files:
                    raise ValueError("❌ No documents uploaded")
                
                current_hashes = _get_file_hashes(uploaded_files)

                # Rebuild retriever ONLY if documents are new or changed
                if state["retriever"] is None or current_hashes != state["file_hashes"]:
                    logger.info("Processing new/changed documents...")
                    chunks = processor.process(uploaded_files)
                    retriever = retriever_builder.build_hybrid_retriever(chunks)

                    state.update({
                        "file_hashes": current_hashes,
                            "retriever": retriever,
                    })

                result = workflow.full_pipeline(
                    question=question_text,
                    retriever=state["retriever"],
                )
                return result["draft_answer"], result["verification_report"], state 
                
            except Exception as e:
                return f"❌ Error: {str(e)}", "", state
            
        submit_btn.click(
            fn=process_question,
            inputs=[question, files, session_state],
            outputs=[answer_output, verification_output, session_state]
        )

    def _get_file_hashes(uploaded_files: list) -> frozenset:
        hashes = set()
        for file in uploaded_files:
            file_path = file if isinstance(file, str) else getattr(file, 'name', str(file)) 
            with open (file_path, "rb") as f:
                hashes.add(hashlib.sha3_256(f.read()).hexdigest())
        return frozenset(hashes)


    # Launch server
    demo.launch(
        server_name="127.0.0.1", 
        server_port=5000, 
        share=False,    # Public host
        theme=gr.themes.Soft(primary_hue="blue"),
        css=css,
        js=js
    )
    

if __name__ == "__main__":
    main()