import streamlit as st
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import StorageContext, load_index_from_storage
from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()

# OpenAI API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llama_parse_api_key = os.getenv("LLAMA_CLOUD_API_KEY")
debug_flag = False
client = OpenAI()

with open('article.json', 'r') as file:
    loaded_dict = json.load(file)

articles_dict = {value.split(' ')[1]: value for key, value in loaded_dict.items()}

articles = [article for article in loaded_dict.values()]

# Load index at startup
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)
retriever = index.as_retriever()


# Streamlit interface
st.title("Pakistan Law Assistant")
user_input = st.text_area("Enter your query:")

if st.button("Submit"):
    with st.spinner("Processing the request..."):
        response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "name": "system",
                "content": "Your role is to extract Article Number from input query.".strip(),
            },
            {
                "role": "system",
                "name": "instruction",
                "content": "Given an input query, extract Article Number if mentioned, if no article is mentioned, reponse 'None'".strip(),
            },
            {
                "role": "user",
                "name": "user_input",
                "content": f"USER INPUT QUERY: {user_input.strip()}",
            }
            ]
            )
        article=''
        possible_article = response.choices[0].message.content
        if "None" not in possible_article:
            article_number=possible_article.split(":")[-1].strip().upper()
            if article_number+':' in articles_dict.keys():
                article = articles_dict[article_number+':']
                context = article
        # ----first call ends

        if article == "":
            nodes = retriever.retrieve(user_input)
            # context = nodes[0].text
            context = ""
            for node in nodes:
                context += node.text + "\n"

            article = context
            print("Context: ", context)

            SYSTEM_MESSAGE = "You are a helpful Law assistant specifically for Pakistan Law."
            INSTRUCTIONS = """
            Instruction:
            1- You will be given a "USER INPUT QUERY", you have to generate a detailed response based on context below.
            2- The response should solve the problem in "USER INPUT QUERY".

            Answer the question based on the context below, and if the question can't be answered based on the context, say "I don't know" 
            Context: {context}
            """
            # Answer the question based on the context below, and if the question can't be answered based on the context, say "I don't know" 
            instruction_message = INSTRUCTIONS
            # instruction_message = instruction_message.replace("{OUTPUT_FORMAT}", OUTPUT_FORMAT)
            instruction_message = instruction_message.replace("{context}", context)

            
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
            resp = ""
            resp = f"{response.choices[0].message.content}"
            if article != "":
                resp += f"\n\nReference:\n{article}"

            response_dict = {"response_content": "I don't know"}
            if debug_flag:
                response_dict["context"] = context
            try:
                response_dict["response_content"] = resp
            except IndexError:
                pass

            try:
                message_content = response_dict
            except IndexError:
                message_content = "I don't know"

        st.write(message_content)
