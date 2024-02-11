from langchain_community.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

import streamlit as st 
from wxai_langchain.llm import LangChainInterface
from wxai_langchain.credentials import Credentials
creds =Credentials(
    api_key= 'Efh1IgL4tblBQ2GQ5u5y7YX_VsPoiB-Q4oVGT5Jgvvtg',
    api_endpoint=  'https://us-south.ml.cloud.ibm.com',
    project_id = 'e2066d3e-4c83-453a-8e5a-ed09b23a2982'
)

llm = LangChainInterface(
    credentials=creds,
    model = 'ibm/granite-13b-chat-v1',
    params={
        'decoding_tokens' : 'sample',
        'max_new_tokens' :200,
        'temperature' : 0.5,
    })

@st.cache_resource
def load_pdf(file):
    loaders = [PyPDFLoader(file)]
    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name = 'all-mpnet-base-v2'),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=150)
    ).from_loaders(loaders)
    return index

# File upload
pdf_file = st.file_uploader("Upload PDF file", type=["pdf"])

if pdf_file is not None:
    # Save the PDF file temporarily
    temp_pdf_path = "temp_pdf.pdf"
    with open(temp_pdf_path, "wb") as temp_file:
        temp_file.write(pdf_file.read())

    # Load PDF and create index
    index = load_pdf(temp_pdf_path)

    # Create a Q&A chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=index.vectorstore.as_retriever(),
        input_key='question'
    )

    # Remove the temporary PDF file
    os.remove(temp_pdf_path)

st.title('SQLScribe')
st.write("Translating English into SQL Statements Seamlessly")

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])


prompt = st.chat_input("Your prompt here")

if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role':'user' , 'content': prompt})

    prompt_with_user_input = f"You are an expert in converting English questions to SQL queries. I have explained the database schema in the attached PDF ({pdf_file}). For a given English {prompt}, write an SQL query. Don't write any explaination for the query. The output should contain SQL query only."
    response = chain.run(prompt_with_user_input)  # Pass the prompt along with user input

    response = chain.run(prompt_with_user_input)  

    st.chat_message('assistant').markdown(response)
    st.session_state.messages.append({'role':'assistant' , 'content': response})