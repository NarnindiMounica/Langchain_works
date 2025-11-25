import validators
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains.summarize import load_summarize_chain
#from langchain_yt_dlp.youtube_loader import YoutubeLoaderDL
from langchain_community.document_loaders import (
    UnstructuredURLLoader,
    YoutubeLoader,
)

#Streamlit Settings
st.set_page_config(page_title="LangChain Summarization Bot Architecture", page_icon="🦜🔗")
st.title("🦜🔗 LangChain Summarization Bot Architecture")
st.subheader("Summarize URL")

#Get Groq API Key and URL (Youtube or Website) to be Summarized

groq_api_key=st.sidebar.text_input("GROQ API KEY", type="password")

model = None
if groq_api_key:
    try:
        model = ChatGroq(model="llama-3.1-8b-instant",api_key=groq_api_key)
        st.sidebar.success("Groq Model Initialized Successfully! 🎉")
    except Exception as e:
        st.sidebar.error(f"Model Initialization Failed: {e}")
else:
    st.sidebar.warning("Enter Groq API Key to initialize model") 

#prompt template
template='''Provide the summary of following content in 300 words: {text}'''
prompt = PromptTemplate(template=template)

#validating passed URL
generic_url=st.text_input("URL", label_visibility="collapsed")  

if generic_url and not validators.url(generic_url):
    st.error("Invalid URL format")

url=generic_url.split("&")[0]

if st.button("Summarize the content from Youtube or Website URL"):
    try:
        with st.spinner("Loading URL.."):
            #loading the youtube or website url to summarize
            if "youtube.com" in url or "youtu.be" in url:
                try:
                    loader=YoutubeLoader.from_youtube_url(url, add_video_info=True, language='en')
                except Exception:
                    st.warning("Transcript not available — switching to audio summarization")
                    loader=YoutubeLoader.from_youtube_url(url, add_video_info=True, language='en', download_audio=True)
            else:
                loader=UnstructuredURLLoader(urls=[url],
                                             ssl_verify=False,
                                             headers={"User-Agent": "Mozilla/5.0"}) 

            documents=loader.load() 

            #chain for summarization
            summarize_chain=load_summarize_chain(llm=model,
                                                chain_type="stuff",
                                                verbose=True, 
                                                prompt=prompt, 
                                                ) 

            output_summary=summarize_chain.run(documents)   

            st.success(output_summary)
    except Exception as e:
        st.exception(f"Exception:{e}")







