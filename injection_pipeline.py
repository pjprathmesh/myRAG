import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
import ollama
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def get_embedding_vector(input):
    vector = ollama.embed(model='embeddinggemma',input=input)
    return vector

def main():
    #1. Loading the files
    #2. Chunking the files
    #3. Embedding and storing in Chroma Vector DB
    pass

if __name__=="__main__":
    main()
