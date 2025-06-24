import gradio as gr
import os

from modules.ingestion.pdf_parser import extract_text_from_pdf
from modules.ingestion.text_chunker import chunk_text
from modules.embedding.embedder import Embedder
from modules.vector_store.faiss_indexer import FaissVectorStore
from modules.qa.qa_engine import QAEngine

# Global store and embedder
store = FaissVectorStore(dim=384)
store.load()
embedder = Embedder()
qa_engine = None
chat_history = []

def set_api_key(api_key):
    global qa_engine
    if not api_key.strip():
        return "Please enter a valid API key."
    qa_engine = QAEngine(embedder, store, api_key)
    return "API key loaded successfully!"

def upload_pdfs(files):
    if not files:
        return "No files selected."
    
    uploaded = []
    total_chunks = 0
    
    for file in files:
        text = extract_text_from_pdf(file.name)
        if not text.strip():
            uploaded.append(f"{os.path.basename(file.name)}: No text found")
            continue
        
        chunks = chunk_text(text)
        embeddings = embedder.embed(chunks)
        store.add(embeddings, chunks)
        total_chunks += len(chunks)
        uploaded.append(f"{os.path.basename(file.name)}: {len(chunks)} chunks added")
    
    store.save()
    uploaded.append(f"Total: {total_chunks} chunks added to knowledge base")
    return "\n".join(uploaded)

def chat(user_input, history):
    if not user_input.strip():
        return history, ""
    
    if qa_engine is None:
        history.append([user_input, "Please set your API key first."])
        return history, ""
    
    try:
        answer = qa_engine.answer_question(user_input, top_k=10)
        history.append([user_input, answer])
    except Exception as e:
        history.append([user_input, f"Error: {str(e)}"])
    
    return history, ""

def clear_chat():
    return []

def reset_memory():
    global store
    store.reset()
    store.save()
    return "Knowledge base cleared successfully!"

# UI layout
with gr.Blocks(title="Build Your Brain - PDF Chat") as demo:
    gr.Markdown("# 🧠 Build Your Brain")
    gr.Markdown("Upload PDF documents and chat with your knowledge base using AI")
    
    # API Key Section
    with gr.Group():
        gr.Markdown("### 🔑 API Configuration")
        with gr.Row():
            api_key_box = gr.Textbox(
                label="Gemini API Key", 
                type="password", 
                placeholder="Enter your Gemini API key",
                scale=3
            )
            api_btn = gr.Button("Set API Key", scale=1, variant="primary")
        api_status = gr.Textbox(label="Status", interactive=False)
    
    # Document Upload Section
    with gr.Group():
        gr.Markdown("### 📄 Document Upload")
        upload = gr.File(
            file_types=[".pdf"], 
            file_count="multiple", 
            label="Upload PDF Documents"
        )
        upload_output = gr.Textbox(
            label="Upload Status", 
            lines=5, 
            interactive=False
        )
    
    # Chat Section
    with gr.Group():
        gr.Markdown("### 💬 Chat with Your Documents")
        chatbot = gr.Chatbot(
            label="Conversation", 
            height=400,
            show_copy_button=True
        )
        
        with gr.Row():
            user_input = gr.Textbox(
                label="Your Question", 
                placeholder="Ask anything about your uploaded documents...",
                scale=4
            )
            send_btn = gr.Button("Send", scale=1, variant="primary")
    
    # Control Buttons
    with gr.Group():
        gr.Markdown("### ⚙️ Controls")
        with gr.Row():
            clear_btn = gr.Button("Clear Chat", variant="secondary")
            reset_btn = gr.Button("Reset Knowledge Base", variant="stop")
    
    # Event bindings
    api_btn.click(
        fn=set_api_key,
        inputs=[api_key_box],
        outputs=[api_status]
    )
    
    upload.change(
        fn=upload_pdfs,
        inputs=[upload],
        outputs=[upload_output]
    )
    
    send_btn.click(
        fn=chat,
        inputs=[user_input, chatbot],
        outputs=[chatbot, user_input]
    )
    
    user_input.submit(
        fn=chat,
        inputs=[user_input, chatbot],
        outputs=[chatbot, user_input]
    )
    
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot]
    )
    
    reset_btn.click(
        fn=reset_memory,
        outputs=[upload_output]
    )

# Launch
if __name__ == "__main__":
    demo.launch()