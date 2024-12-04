import os
import re
import pickle
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class PDFChatbot:
    def __init__(self):
        self.vector_store = None
        self.embeddings = None
        self.llm = self._load_llama_model()
        self.chat_history = []
        self.context_docs = []
       
    def _load_llama_model(self):
        """Load Llama-3 model optimized for CPU"""
        huggingface_token = os.getenv("HF_TOKEN", "")
       
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-3B-Instruct",
            token=huggingface_token,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.2-3B-Instruct",
            token=huggingface_token
        )
        return model, tokenizer
   
    def clean_text(self, text):
        """Clean and normalize text"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'^[\*\-•]\s*', '', text, flags=re.MULTILINE)
        return text.strip()
   
    def load_pdf(self, pdf_path='input.pdf'):
        """Extract text from PDF"""
        reader = PdfReader(pdf_path)
        full_text = "".join(page.extract_text() or "" for page in reader.pages)
        return self.clean_text(full_text)
   
    def chunk_text(self, text, chunk_size=200, overlap=25):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )
        return text_splitter.split_text(text)
   
    def create_or_load_embeddings(self, chunks):
        """Create or load precomputed embeddings"""
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
        """Retrieve and answer query with context"""
        if not self.vector_store:
            return "Please load a PDF first."
       
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        self.context_docs = retriever.get_relevant_documents(query)
       
        context_details = [
            f"{self.clean_text(doc.page_content)}" 
            for doc in self.context_docs
        ]
        context = "\n".join(context_details)
       
        history_context = "\n".join([f"Previous Query: {h[0]}" for h in self.chat_history[-2:]])
       
        model, tokenizer = self.llm
        prompt = f"""Use ONLY these contexts to answer the question
        
        Previous Conversation Context:
        {history_context}

        Retrieved Document Contexts:
        {context}

        Question: {query}
       
        If the answer is not in the contexts, respond with:
        "Sorry, I didn’t understand your question. Do you want to connect with a live agent?”"
       
        Helpful Answer:"""
       
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=500, num_return_sequences=1)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
       
        self.chat_history.append((query))
       
        return response

def main():
    st.title("PDF Chatbot")
   
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = PDFChatbot()
   
    text = st.session_state.chatbot.load_pdf()
    chunks = st.session_state.chatbot.chunk_text(text)
    st.session_state.chatbot.create_or_load_embeddings(chunks)
    st.success("PDF Loaded Successfully!")
   
    if 'messages' not in st.session_state:
        st.session_state.messages = []
   
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
   
    if prompt := st.chat_input("Ask a question about your PDF"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
   
        with st.chat_message("assistant"):
            response = st.session_state.chatbot.query_document(prompt)
            helpful_answer = response.split("Helpful Answer:")[1].strip()
            if "Sorry, I didn’t understand your question. Do you want to connect with a live agent?".lower() in helpful_answer.lower():
                st.markdown("Sorry, I didn’t understand your question. Do you want to connect with a live agent?")
            else:
                st.markdown(helpful_answer)
   
            with st.expander("View Contexts"):
                for i, doc in enumerate(st.session_state.chatbot.context_docs, 1):
                    st.text_area(f"Context {i} ", value=doc.page_content, height=100)
   
        st.session_state.messages.append({"role": "assistant", "content": helpful_answer})

if __name__ == "__main__":
    main()