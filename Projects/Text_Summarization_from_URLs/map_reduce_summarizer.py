import validators
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_core.prompts import PromptTemplate


st.title("youtube url loader")

model=None
groq_api_key=st.sidebar.text_input("enter groq key", type="password")
if not groq_api_key:
    st.warning("groq key is missing")
    st.stop()

model = ChatGroq(model="llama-3.1-8b-instant",api_key=groq_api_key)

map_prompt = PromptTemplate(
    template="Summarize the following chunk:\n\n{text}",
    input_variables=["text"]
)

combine_prompt = PromptTemplate(
    template="Combine these summaries into a concise overall summary:\n\n{text}",
    input_variables=["text"]
)

url=st.chat_input("please enter your url here..")

if url and validators.url(url):
   with st.spinner("loading..patience please.."):
        if "youtube.com" in url:
            loader= YoutubeLoader.from_youtube_url(url, add_video_info=False)
        else:
            loader = UnstructuredURLLoader(urls=[url], ssl_verify=False, headers={"User-agent":"Mozilla/5.0"})   
            
        documents=loader.load()
        splits=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(documents)
        summarize_chain = load_summarize_chain(llm=model,
                                           chain_type="map_reduce",
                                           verbose=True,
                                           map_prompt=map_prompt,
                                           combine_prompt=combine_prompt 
                                           ) 
        output_summary=summarize_chain.invoke({"input_documents": splits})   

        st.success(output_summary) 

else:
    st.warning("url not provided..")            



   





