# Mini RAG - Retrieval-Augmented Generation System

## 📝 Project Description

Mini RAG is an application that enables PDF document analysis using Retrieval-Augmented Generation (RAG) technique. The application allows:

- Loading PDF documents
- Processing text into chunks
- Creating embeddings using a SentenceTransformer model
- Indexing text with FAISS for fast semantic search
- Generating answers to questions using OpenAI or DeepSeek models
- Adjusting text generation parameters (e.g., temperature)
- Reviewing query history

## 🛠️ Project Structure

```
.
├── app.py                 # Main Streamlit application file
├── app1.py                # Alternative modular version 
├── app_simple.py          # Simpler version of the application (all in one file)
├── utilis.py              # Helper functions (logging, error handling)
├── embeddings/            # Module for handling embeddings
│   ├── __init__.py
│   └── engine.py          # Functions for creating embeddings and FAISS index
├── modules/               # Main modules directory
│   ├── __init__.py
│   ├── models/            # Module for language models
│   │   ├── __init__.py
│   │   └── llm.py         # Functions for generating responses via models
├── data/                  # Directory with example data
│   └── constitution.pdf   # Example document
└── requirements.txt       # Required dependencies
```

## 🔧 Requirements and Installation

1. Install Python (version 3.9+ recommended)
2. Clone this repository
3. Install required dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project's root directory and add your API keys:

```
OPENAI_API_KEY=your_openai_key_here
DEEPSEEK_API_KEY=your_deepseek_key_here
```

## 🚀 Running the Application

To run the application, execute:

```bash
streamlit run app.py
```

For alternative versions:

```bash
streamlit run app1.py
```

or

```bash
streamlit run app_simple.py
```

The application will be available at http://localhost:8501

## 💡 How to Use

1. **Load PDF documents** - click on the "Upload PDFs" area and select PDF files
2. **Process documents** - click the "Process PDFs" button
3. **Ask a question** - type your question in the text field
4. **Select a model** - choose the model provider (OpenAI or DeepSeek) and specific model
5. **Adjust parameters** - set the temperature to control response creativity
6. **Generate answer** - click the "Answer" button
7. **Review results** - read the answer and expand the "Context Fragments" section to see where the information comes from
8. **Query history** - browse previous questions and answers in the "Question History" section

## 🧩 Project Modularity

The project uses a modular architecture:

- **utilis.py** - helper functions for messages and error handling
- **embeddings/engine.py** - handling embeddings and text indexing
- **modules/models/llm.py** - generating responses through language models

## 📝 Notes

- The application requires an active internet connection to use OpenAI or DeepSeek models
- Processing large PDF documents may require significant RAM
- Only PDF files are supported - other document formats are not currently supported

## 🔮 Possible Extensions

- Adding support for other document formats (DOCX, TXT, HTML)
- Implementing more advanced chunking methods (e.g., with overlapping fragments)
- Adding support for more language models
- Implementing a response caching mechanism
- Adding visualization of text fragment similarity

## 📄 License

This project is released under the MIT license.