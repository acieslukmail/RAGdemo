import streamlit as st
from dotenv import load_dotenv
import os
import uuid
from PyPDF2 import PdfReader
import numpy as np
import sys
import asyncio

# Naprawienie problemu z event loop dla PyTorch
if sys.platform == 'darwin' and 'torch' in sys.modules:
    # Na macOS, napraw problem z PyTorch i asyncio
    try:
        # Sprawdź czy pętla już istnieje
        asyncio.get_running_loop()
    except RuntimeError:
        # Jeśli nie, utwórz nową pętlę i ustaw ją jako domyślną
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

# Import functions from utilis
from utilis import check_api_keys, display_error, display_success, format_context

# Import functions from modules.models.llm
from modules.models.llm import generate_answer

# Import functions from embeddings/engine.py
from embeddings.engine import load_embedding_model, build_faiss_index

# Konfiguracja dla MPS na macOS
if sys.platform == 'darwin':
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Load environment variables
load_dotenv()

# Initialize session state
if 'processing_done' not in st.session_state:
    st.session_state.processing_done = False
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'index' not in st.session_state:
    st.session_state.index = None
if 'history' not in st.session_state:
    st.session_state.history = []

# Function to parse PDFs
def parse_pdfs(files):
    """
    Parse PDF files and extract text, splitting into chunks.
    
    Args:
        files: List of uploaded PDF files
        
    Returns:
        List of chunks, each with unique ID and text content
    """
    chunks = []
    for file in files:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
                
        # Split text into paragraphs
        paragraphs = text.split("\n\n")
        for para in paragraphs:
            para = para.strip()
            if len(para) > 30:  # Skip very short fragments
                chunks.append({"id": str(uuid.uuid4()), "text": para})
    return chunks

def main():
    st.title("🧠 Mini RAG - Enhanced Version")
    st.markdown("You can ask multiple questions and choose a different model for each question.")

    # Check API keys
    api_keys = check_api_keys()
    st.write("OpenAI API key status:", "✅ Loaded" if api_keys["openai_status"] else "❌ Missing")
    st.write("DeepSeek API key status:", "✅ Loaded" if api_keys["deepseek_status"] else "❌ Missing")

    # Store keys in variables for easier access
    openai_key = api_keys["openai_key"]
    deepseek_key = api_keys["deepseek_key"]

    # Load embedding model
    with st.spinner("Loading embedding model..."):
        model = load_embedding_model()
        display_success("Embedding model loaded.")

    # PDF uploader
    uploaded_files = st.file_uploader("📎 Upload PDF files", type=["pdf"], accept_multiple_files=True)

    # Process PDFs button
    if uploaded_files:
        st.write(f"Loaded {len(uploaded_files)} files")
        
        if st.button("🔍 Process PDFs"):
            with st.spinner("Processing files and creating embeddings..."):
                chunks = parse_pdfs(uploaded_files)
                
                if chunks:
                    index, chunks = build_faiss_index(chunks, model)
                    st.session_state.index = index
                    st.session_state.chunks = chunks
                    st.session_state.processing_done = True
                    display_success(f"Successfully processed {len(chunks)} text fragments.")
                else:
                    display_error("No suitable text fragments found in the documents.")

    # Question answering section - appears only after processing files
    if st.session_state.processing_done:
        # Model selection
        model_provider = st.radio(
            "Select model provider:",
            ["OpenAI", "DeepSeek"]
        )
        
        if model_provider == "OpenAI":
            if not api_keys["openai_status"]:
                display_error("OpenAI API key missing. Add the key to the .env file.")
            model_name = st.selectbox(
                "OpenAI model:",
                ["gpt-3.5-turbo", "gpt-4"]
            )
        else:  # DeepSeek
            if not api_keys["deepseek_status"]:
                display_error("DeepSeek API key missing. Add the key to the .env file.")
            model_name = st.selectbox(
                "DeepSeek model:",
                ["deepseek-chat", "deepseek-coder"]
            )
        
        # Text generation parameters
        temperature = st.slider("Temperature (creativity)", 0.0, 1.0, 0.2, 0.1)
        
        query = st.text_input("❓ Your question")
        
        if query and st.button("🤖 Answer"):
            if (model_provider == "OpenAI" and not api_keys["openai_status"]) or \
               (model_provider == "DeepSeek" and not api_keys["deepseek_status"]):
                display_error("API key missing for the selected provider.")
            else:
                with st.spinner(f"Generating answer using {model_provider}..."):
                    # Create embedding for query
                    query_vec = model.encode(query).astype("float32")
                    
                    # Search for similar fragments
                    D, I = st.session_state.index.search(np.array([query_vec]), 5)
                    
                    # Get text
                    ctx = [st.session_state.chunks[i]["text"] for i in I[0]]
                    context_text = "\n\n".join(ctx)
                    
                    # Build prompt
                    prompt = f"""Based on the following information, answer the user's question.
If you don't have sufficient data, honestly say 'I don't know'.

### CONTEXT:
{context_text}

### QUESTION:
{query}

### ANSWER:
"""
                    
                    # Generate answer
                    try:
                        answer = generate_answer(prompt, model_provider, model_name, openai_key, deepseek_key, temperature)
                        
                        # Add to history
                        st.session_state.history.append({
                            "query": query,
                            "answer": answer,
                            "context": ctx,
                            "model": f"{model_provider} ({model_name})",
                            "temperature": temperature
                        })
                        
                        # Display answer
                        st.markdown(f"### ✅ Answer ({model_provider} - {model_name}):")
                        display_success(answer)
                        
                        # Display context fragments
                        with st.expander("📎 Context fragments"):
                            for i, frag in enumerate(ctx, 1):
                                st.markdown(f"**{i}.** {frag}")
                    except Exception as e:
                        display_error(f"An error occurred: {str(e)}")

    # Display question and answer history
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 📚 Question History")
        
        for i, item in enumerate(reversed(st.session_state.history), 1):
            with st.expander(f"{i}. {item['query']} ({item['model']}, temp: {item.get('temperature', 0.2)})"):
                st.markdown("**Answer:**")
                st.write(item['answer'])
                
                st.markdown("**Used fragments:**")
                formatted_contexts = format_context(item['context'])
                for ctx_html in formatted_contexts:
                    st.markdown(ctx_html)

if __name__ == "__main__":
    main()