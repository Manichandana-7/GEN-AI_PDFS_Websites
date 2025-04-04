import os
import json
import uuid
import PyPDF2
import requests
import numpy as np
import streamlit as st
import google.generativeai as genai
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from urllib.parse import urljoin, urlparse
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
 
load_dotenv()
 
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")
BUCKET_NAME =  os.getenv("SUPABASE_BUCKET")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_API_KEY)
 
genai_api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=genai_api_key)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")
 
st.set_page_config(layout="wide")
st.sidebar.markdown("<h2 style='text-align: center; color: #4B0082;'>InfoAI</h2>", unsafe_allow_html=True)
st.sidebar.write("Our AI-powered knowledge retrieval system helps you extract insights from documents and websites effortlessly. Simply upload PDFs or enter a website domain, When you ask a question, it retrieves the most relevant information and uses Gemini LLM to generate accurate responses. You can also review and verify sources through clickable links. Start by uploading a file or entering a website domain!")
 
if "messages" not in st.session_state:
    st.session_state.messages = []
if "file_data" not in st.session_state:
    st.session_state.file_data = {}
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
 
# Function to fetch recent chat history from Supabase
def fetch_chat_history():
    response = supabase.table('history').select('question', 'answer').order('created_at', desc=True).limit(5).execute()
    return response.data if response.data else []
 
# Display recent questions in the sidebar
st.sidebar.markdown("<h2 style='text-align: center; color: #4B0082;'>Recent Questions</h2>", unsafe_allow_html=True)
recent_questions = fetch_chat_history()
 
if recent_questions:
    for item in recent_questions:
        with st.sidebar.expander(f"Q: {item['question']}"):
            st.markdown(f"**Answer:** {item['answer']}")
else:
    st.sidebar.write("No recent questions yet.")
 
# Function to upload PDF to Supabase Storage
def upload_to_supabase(file_name, file_data):
    unique_id = uuid.uuid4().hex[:8]  
    file_path = f"{unique_id}_{file_name}" 
    supabase.storage.from_(BUCKET_NAME).upload(file_path, file_data, {"content-type": "application/pdf"})
    public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(file_path)
    return public_url
 
# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text.strip()
 
# Function to generate embeddings
def generate_embeddings(text):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return model.encode([text]).tolist()[0]
 
# Function to scrape data from a website with depth up to 5 internal links
def scrape_website(url, depth=5, visited=None):
    if visited is None:
        visited = set()
    if depth == 0 or url in visited:
        return ""   
    visited.add(url)
    text_content = ""
   
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            text_content += " ".join([p.get_text() for p in paragraphs])
            for link in soup.find_all('a', href=True):
                full_url = urljoin(url, link['href'])
                if urlparse(full_url).netloc == urlparse(url).netloc:
                    text_content += scrape_website(full_url, depth-1, visited)
    except Exception as e:
        return ""
   
    return text_content.strip()
 
# Function to retrieve relevant chunks from Supabase
def get_relevant_chunks(question_embedding):
    response = supabase.rpc(
        "match_documents",
        {"query_embedding": question_embedding, "match_count": 10}
    ).execute()
 
    if not response.data:
        return [], {}
    extracted_data = [(item["content"], item["file_name"]) for item in response.data]
   
    file_content_map = {file_name: content for content, file_name in extracted_data}
   
    return extracted_data, file_content_map
 
# Function to get AI response using Gemini
def get_ai_response(question, context):
    response = gemini_model.generate_content(f"Answer the following question based on the document:\n\nContext: {context}\n\nQuestion: {question}")
    return response.text
 
st.markdown("<h2 style='text-align: center; color: #4B0082;'>Get Information from the PDFs or Websites</h2>", unsafe_allow_html=True)
 
# os.makedirs("temp", exist_ok=True)

input_method = st.radio("Select Input Method:", ["Upload PDFs", "Enter Website Domain"])
uploaded_files = None
domain_input = ""
 
if input_method == "Upload PDFs":
    uploaded_files = st.file_uploader("Upload Your PDFs", type="pdf", accept_multiple_files=True, label_visibility="collapsed")   
    upload_button = st.button("Upload Files",help="Click to upload selected files")

elif input_method == "Enter Website Domain":
    domain_input = st.text_input("Enter a website domain (e.g., https://example.com):")
    upload_button = st.button("Scrape Website",help="Click to scrape the data")
if upload_button: 
    if uploaded_files:
        supabase.table("filesdata").delete().neq("id", 0).execute()
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            file_bytes = uploaded_file.read()
            pdf_url = upload_to_supabase(file_name, file_bytes)
            pdf_text = extract_text_from_pdf(uploaded_file)
            embedding = generate_embeddings(pdf_text)
    
            supabase.table("filesdata").insert({
                "file_name": file_name,
                "content": pdf_text,
                "embedding": json.dumps(embedding),
                "pdf_url": pdf_url
            }).execute()
    
            st.session_state.file_data[file_name] = pdf_url        
            st.success(f"PDF '{file_name}' Uploaded and Processed Successfully!")
            st.session_state.data_loaded = True
    
    elif domain_input:
        scraped_text = scrape_website(domain_input)
        if scraped_text:
            embedding = generate_embeddings(scraped_text)
        
            supabase.table("filesdata").delete().neq("id", 0).execute()
        
            supabase.table("filesdata").insert({
                "file_name": domain_input,
                "content": scraped_text,
                "embedding": json.dumps(embedding)  
            }).execute()
        
            st.success(f"Data from '{domain_input}' scraped and stored successfully!")
            st.session_state.data_loaded = True
        else:
            st.error("Failed to scrape data from the provided website.")
 
def get_pdf_link(file_name, file_url):
    """Generates a link to view the PDF online."""
    return f'<a href="{file_url}" target="_blank" style="color: #4B0082; text-decoration: underline;"> {file_name}</a>'
 
if st.session_state.data_loaded:
    # Accept user input first
    user_input = st.chat_input("Enter your question and press Enter:")
   
    if user_input:
        question_embedding = generate_embeddings(user_input)
        relevant_chunks, file_content_map = get_relevant_chunks(question_embedding)
        context = "\n".join([chunk for chunk, _ in relevant_chunks])
        response = get_ai_response(user_input, context)
        sources = [file_name for _, file_name in relevant_chunks]
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": response, "sources": sources})
        supabase.table('history').insert({'question': user_input, 'answer': response, 'created_at': 'now()'}).execute()
   
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):          
            if "sources" in message and message["sources"]:
                st.write("**Sources:**")
                for source in message["sources"]:
                    if source.startswith("http"):
                        st.markdown("""
                            <style>
                                a {
                                    color: #4B0082 !important;
                                    text-decoration: none;
                                }
                            </style>
                        """, unsafe_allow_html=True)
                        st.markdown(f" [ {source}]({source})")
                    else:
                        if source in st.session_state.file_data:
                            pdf_link = get_pdf_link(source, st.session_state.file_data[source])
                            st.markdown(pdf_link, unsafe_allow_html=True)
                    break
                   
            st.markdown(message["content"])