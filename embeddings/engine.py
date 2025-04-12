import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
import logging
import os
import sys

# Skonfiguruj logowanie FAISS, żeby uniknąć ostrzeżeń GPU
logging.getLogger('faiss').setLevel(logging.ERROR)
logging.getLogger('faiss.loader').setLevel(logging.ERROR)

# Logger dla tego modułu
logger = logging.getLogger(__name__)

@st.cache_resource
def load_embedding_model():
    """
    Load and cache the sentence transformer model for embeddings.
    
    Returns:
        SentenceTransformer model
    """
    # Wycisz komunikaty SentenceTransformer podczas ładowania modelu
    temp_level = logging.getLogger('sentence_transformers').level
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    
    logger.info("Loading embedding model...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Przywróć poprzedni poziom logowania
    logging.getLogger('sentence_transformers').setLevel(temp_level)
    
    return model

def build_faiss_index(chunks, model):
    """
    Create embeddings for text chunks and build a FAISS index.
    
    Args:
        chunks: List of text chunks
        model: SentenceTransformer model for creating embeddings
        
    Returns:
        Tuple of (FAISS index, updated chunks with embeddings)
    """
    logger.info(f"Building FAISS index for {len(chunks)} chunks...")
    
    # Create embeddings for each chunk
    for c in chunks:
        c["embedding"] = model.encode(c["text"]).tolist()
    
    # Make sure chunks is not empty
    if not chunks:
        logger.warning("No chunks to index!")
        return None, chunks
    
    # Create FAISS index - jawnie wybierz CPU zamiast próbować GPU
    dim = len(chunks[0]["embedding"])
    
    # Użyj StandardGpuResources tylko gdy GPU jest dostępne
    # Inaczej użyj indeksu CPU
    use_gpu = False
    
    if not use_gpu:
        index = faiss.IndexFlatL2(dim)
    else:
        try:
            # Próba inicjalizacji indeksu GPU tylko jeśli dostępny
            res = faiss.StandardGpuResources()
            index = faiss.GpuIndexFlatL2(res, dim)
            logger.info("Using GPU FAISS index")
        except Exception as e:
            logger.warning(f"Could not initialize GPU FAISS index: {str(e)}")
            logger.info("Falling back to CPU FAISS index")
            index = faiss.IndexFlatL2(dim)
    
    # Add embeddings to the index
    embs = np.array([c["embedding"] for c in chunks]).astype("float32")
    index.add(embs)
    
    logger.info(f"FAISS index built with dimension {dim}")
    return index, chunks