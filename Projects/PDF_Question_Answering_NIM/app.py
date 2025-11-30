import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain



# Streamlit Framework Starts

st.set_page_config(page_title="PDF Question Answering NIM", page_icon="📖")
st.title("PDF Question Answering NIM")

groq_api_key = st.sidebar.text_input("Enter your groq api key")

if not groq_api_key:
    st.warning("Enter groq api key to begin")
    st.stop()

model = ChatGroq(api_key=groq_api_key, model="llama-3.1-8b-instant") 
if model:
    st.success("Model initialized successfully")

template='''
answer questions of user based only on the context information given to you in step by step approach.
if question is not related to context, say it clearly that you don't know answer, don't answer on your own.
Context:
{context}

Question:
{input}
'''   
prompt = PromptTemplate(input_variables=["context", "input"],
    template=template
    
) 

#to get pdf documents from user
uploaded_documents= st.file_uploader("Upload PDF files", type="pdf",  accept_multiple_files=True)
with st.spinner("Getting Retriever Ready.."):
    if uploaded_documents:
        docs=[]
        for uploaded_file in uploaded_documents:
            with open("./temp", mode="wb") as f:
                f.write(uploaded_file.getvalue())
                loader = PyPDFLoader(os.path.join(os.getcwd(), "temp"))
                docs.extend(loader.load())
          
        chunks=(RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs))
        embeddings= OllamaEmbeddings(model="all-minilm:latest")
        vectorstore=FAISS.from_documents(chunks, embeddings)
        retriever=vectorstore.as_retriever()
        st.success("Retriever is ready")

        stuff_documents_chain=create_stuff_documents_chain(model, prompt)
        retriever_chain = create_retrieval_chain(retriever, stuff_documents_chain)

        question=st.text_input("Please ask your question here..")
        if question:
            response=retriever_chain.invoke({"input":question})
            st.write(response['answer'])




