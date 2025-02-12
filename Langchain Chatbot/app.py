import torch
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import streamlit as st
from accelerate import dispatch_model

# Load documents from URLs
urls = [
    "https://python.langchain.com/docs/tutorials/llm_chain/",
    "https://python.langchain.com/docs/tutorials/classification/",
    "https://python.langchain.com/docs/tutorials/chatbot/",
    "https://python.langchain.com/docs/tutorials/agents/",
    "https://python.langchain.com/docs/tutorials/rag/",
    "https://python.langchain.com/docs/tutorials/qa_chat_history/"
]

data = WebBaseLoader(urls)
documents = data.load()

# Split documents into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=100)
split_docs = text_splitter.split_documents(documents)

# Create embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v1")
db = Chroma.from_documents(documents=split_docs, embedding=embeddings)

model_id = "openai-community/gpt2-xl"

# Load model with disk offload
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu",token="")
tokenizer = AutoTokenizer.from_pretrained(model_id,token="")
model.config.pad_token_id = model.config.eos_token_id

# Create a text generation pipeline
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, temperature=0.1)
llm = HuggingFacePipeline(pipeline=pipe)

# Define the prompt template
template = """
You are a highly intelligent and knowledgeable assistant. When responding to a question, make sure to break down your thought process into clear and logical steps. Provide detailed, structured reasoning before presenting the final answer.

Do not include any context, explanation, or reasoning in your final answer. Only present the answer itself, directly addressing the question asked.

Context: {context}

Question: {input}

Answer:
"""

prompt = PromptTemplate(template=template, input_variables=['context', 'input'])

# Create the document chain and retrieval chain
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = db.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Streamlit UI
st.title("Chatbot")
user_input = st.text_input("You:", "")

if user_input:
    response = retrieval_chain.invoke({"input": user_input})
    # Extract the response text from the output
    st.text_area("Chatbot:", response["answer"].split("\n\n\n")[-1], height=100)  # Adjust based on your output structure
else:
    st.warning("Please enter a message.")