# 🦠 Ebola Virus Guidance Assistant

An AI-powered **Ebola Virus Guidance Chatbot** built with **Streamlit**, **LangChain**, **Google Gemini**, and **Pinecone**.  
The system delivers **brief, accurate, and reliable Ebola-related health guidance** using a Retrieval-Augmented Generation (RAG) pipeline over a curated medical knowledge base.

---

## 🚀 Features

- 🧠 Retrieval-Augmented Generation (RAG)
- 📄 PDF-based medical knowledge ingestion
- 🤖 Google Gemini 2.0 Flash for fast responses
- 🔍 Semantic search with Pinecone
- 🧵 Conversational memory with LangChain
- ⚡ Concurrent embedding and batched vector upserts
- 🖥️ Interactive Streamlit chat interface

---

## 🏗️ Project Structure

---
.
├── main.py # Streamlit app (chat UI + inference)
├── pinecone.py # PDF ingestion & vector indexing
├── combined_ebola_pdf.pdf # Ebola knowledge base
├── .env # API keys
├── requirements.txt
└── README.md
---
---

## 🧠 System Architecture (RAG Pipeline)

1. **Document Ingestion**
   - Ebola PDFs are loaded and split into overlapping chunks
   - Chunks are embedded using Google Generative AI Embeddings
   - Embeddings are stored in a Pinecone vector index

2. **Question Answering**
   - User submits a question via Streamlit
   - Question is embedded and queried against Pinecone
   - Top relevant document chunks are retrieved
   - Retrieved context + system prompt is sent to Gemini 2.0 Flash
   - Model returns a concise, accurate response

---

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
PINECONE_API_KEY=your_pinecone_api_key
GOOGLE_API_KEY=your_google_api_key


## 📦 Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/ebola-virus-guidance-assistant.git
cd ebola-virus-guidance-assistant

## Dependencies installation

pip install -r requirements.txt

## Before run, indexing the document into pinecone

python pinecone.py

## Run the application

streamlit run main.py


## Example Prompt

- What are the early symptoms of Ebola?

- How is Ebola transmitted?

- What preventive measures reduce Ebola infection risk?

- Is Ebola curable?

## ⚠️ Disclaimer

This project is intended **for educational and informational purposes only**.  
It **does not replace professional medical advice, diagnosis, or treatment**.

---

## 🛠️ Technologies Used

- Python  
- Streamlit  
- LangChain  
- Google Gemini 2.0 Flash  
- Google Generative AI Embeddings  
- Pinecone Vector Database  
- PDF Processing  
- Concurrent & Batched Vector Operations  

---

## 👤 Author

**Haruna Adegoke**  
Applied AI / Machine Learning Engineer

---

## 📄 License

This project is licensed under the **MIT License**.




