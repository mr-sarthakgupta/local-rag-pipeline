# PDF Chatbot Documentation

## Overview
The PDF Chatbot is an advanced Python application that transforms PDF documents into interactive, queryable knowledge bases using natural language processing techniques.

## Core Architecture: PDFChatbot Class

### Key Components
- Vector store for document embeddings
- Language model for understanding and generating responses
- Text processing and cleaning mechanisms
- Semantic search capabilities

### Core Methods

#### Model Initialization
```python
def __init__(self):
    self.vector_store = None     # Embedding storage
    self.embeddings = None       # Embedding generation
    self.llm = self._load_llama_model()  # Language model
    self.chat_history = []       # Conversation context
    self.context_docs = []       # Relevant document contexts
```

#### Text Processing
```python
def clean_text(self, text):
    # Normalize text by removing extra whitespaces and markers
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'^[\*\-•]\s*', '', text)
    return text.strip()

def load_pdf(self, pdf_path='input.pdf'):
    # Extract and clean text from PDF
    reader = PdfReader(pdf_path)
    full_text = "".join(page.extract_text() or "" for page in reader.pages)
    return self.clean_text(full_text)

def chunk_text(self, text, chunk_size=200, overlap=25):
    # Break text into semantic chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    return text_splitter.split_text(text)
```

#### Embedding and Retrieval
```python
def create_or_load_embeddings(self, chunks):
    # Create vector representations with caching
    embedding_path = 'pdf_embeddings.pkl'
    
    if os.path.exists(embedding_path):
        with open(embedding_path, 'rb') as f:
            self.vector_store = pickle.load(f)
        return
    
    self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    self.vector_store = FAISS.from_texts(chunks, self.embeddings)
    
    with open(embedding_path, 'wb') as f:
        pickle.dump(self.vector_store, f)

def query_document(self, query):
    # Retrieve and answer queries contextually
    retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
    self.context_docs = retriever.get_relevant_documents(query)
    
    context = "\n".join([doc.page_content for doc in self.context_docs])
    response = self._generate_response(query, context)
    
    return response
```

## Dependencies
Simply use `pip install -r requirements.txt`


## Usage
1. Install required libraries
2. Set up Hugging Face token
3. Prepare input PDF
4. Run with Streamlit: `streamlit run pdf_chatbot.py`

## Key Features
- Semantic document search
- Context-aware responses
- Embedding caching
- Conversation history tracking

## Limitations
- CPU-based processing
- Dependent on Llama-3 3B model
- Performance varies with document complexity

## Potential Improvements
- Multi-PDF support
- Advanced embedding techniques
- Robust error handling
- Configurable model parameters

## Use Cases
- Academic research
- Technical manual analysis
- Contract review
- Training material comprehension

The PDF Chatbot bridges the gap between traditional document reading and intelligent information retrieval, offering a powerful tool for extracting insights from static documents.
