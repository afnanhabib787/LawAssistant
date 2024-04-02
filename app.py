import streamlit as st
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import StorageContext, load_index_from_storage
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# OpenAI API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llama_parse_api_key = os.getenv("LLAMA_CLOUD_API_KEY")

# Load index at startup
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)
retriever = index.as_retriever()

# Streamlit interface
st.title("Pakistan Law Assistant")
user_input = st.text_area("Enter your query:")

if st.button("Submit"):
    with st.spinner("Processing the request..."):
        nodes = retriever.retrieve(user_input)

        context = ""
        for node in nodes:
            context += node.text + "\n"

        SYSTEM_MESSAGE = "You are a helpful Law assistant specifically for Pakistan Law."
        INSTRUCTIONS = """
        Instruction:
        1- You will be given a "USER INPUT QUERY", you have to generate a detailed response based on context below.
        2- The response should solve the problem in "USER INPUT QUERY".
        3- The response should be in points.
        4- If possible, Refer to the 'act' from context with each point.

        Answer the question based on the context below, and if the question can't be answered based on the context, say "I don't know" 
        Context: {context}
        """

        instruction_message = INSTRUCTIONS
        instruction_message = instruction_message.replace("{context}", context)

        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "name": "system",
                    "content": SYSTEM_MESSAGE.strip(),
                },
                {
                    "role": "system",
                    "name": "instruction",
                    "content": instruction_message.strip(),
                },
                {
                    "role": "user",
                    "name": "user_input",
                    "content": f"USER INPUT QUERY: {user_input.strip()}",
                }
            ]
        )

        try:
            message_content = response.choices[0].message.content
        except IndexError:
            message_content = "I don't know"

        st.write(message_content)
