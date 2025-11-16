import os
import streamlit as st
from dotenv import load_dotenv
from langsmith import uuid7
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
id = uuid7()

#Langsmith Tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] = "APP Using OLLAMA"

#Prompt Template

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant, please respond to question asked."), 
    ("user", "Question:{question}")
])

#Streamlit framework
st.title("App Using Ollama")
input_text= st.text_input("Help me with your question please..")

#Ollama model
model = OllamaLLM(model="llama2:latest")

parser = StrOutputParser()

chain = prompt| model| parser

if input_text:
    st.write(chain.invoke({"question": input_text}))

