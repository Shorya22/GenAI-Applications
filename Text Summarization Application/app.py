# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,pipeline
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain import LLMChain
from langchain_huggingface import HuggingFacePipeline
import warnings
from huggingface_hub import login
warnings.filterwarnings("ignore")
import streamlit as st

# Retrieve the token from the secrets file
token = st.secrets["hf_token"]

# Login to Hugging Face using the token
import huggingface_hub
huggingface_hub.login(token=token)


model= AutoModelForSeq2SeqLM.from_pretrained("Shorya22/BART-Large-Fine_Tunned")
tokenizer= AutoTokenizer.from_pretrained("Shorya22/BART-Large-Fine_Tunned")

# Define the prompt template
prompt = PromptTemplate(template="Summarize the following text\n\n{input_text}\n\nSummary:\n\n", input_variables=['input_text'])

pipe= pipeline(task='text2text-generation',model=model,tokenizer=tokenizer,max_new_tokens=256)
llm= HuggingFacePipeline(pipeline=pipe)

chain= prompt | llm

input= st.text_area("Enter text to summarize")

# Create the Streamlit button
if st.button("Summarize"):
    output = chain.invoke({'input_text':input})
    st.write(output)