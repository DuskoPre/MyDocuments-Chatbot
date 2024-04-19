#  Copyright (c) 18.04.2024 [D. P.] aka duskop; after the call a day after from a IPO-agency from Japan, i'm adding my patreon ID: https://www.patreon.com/florkz_com
#  All rights reserved.

import os
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader  # Import DirectoryLoader
from langchain.chains.question_answering import load_qa_chain
import markdown

# Function to load documents from a directory
def load_docs(directory):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents

# Set the directory path containing your documents
directory = "/home/duskop/AGiXT/docs"

# Load text documents from the specified directory
documents = load_docs(directory)

# Define the LLamaCpp model with increased context length
llm = LlamaCpp(
    model_path="/media/duskop/df3369fd-02a2-4ec3-8914-95333332845a1/LLM/LLAMA2-7Bs/llama-2-7b-chat.Q2_K.gguf",
    temperature=0.75,
    max_tokens=2000,
    top_p=1,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=True,
    n_ctx=2048  # Increase context length
)


def split_docs(documents, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    docs = text_splitter.split_documents(documents)
    return docs

docs = split_docs(documents)

# Initialize HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create a vector store from the documents using Chroma
db = Chroma.from_documents(documents=documents, embedding=embeddings)

# Create a QA chain
chain = load_qa_chain(llm, chain_type="stuff")

def get_answer(query, documents):
    similar_docs = db.similarity_search(query, k=2)  # Get two closest chunks
    # Use LLamaCpp to generate the answer
    # Construct input for the QA chain
    input_data = {
        'input_documents': similar_docs,
        'question': query
    }
    answer = chain.run(input_data)
    return answer, similar_docs


print("Private Q&A chatbot")

while True:
    prompt = input("Enter your query here (or type 'exit' to quit): ")
    if prompt.lower() == "exit":
        break
    else:
        answer, similar_docs = get_answer(prompt, documents)
        print(f"Answer: {answer}")
