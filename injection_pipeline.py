import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


def load_documents(docs_path):
    '''
    Load documents content and metadata in a list and return it
    
    :param docs_path: path to training docs
    '''
    print(f"Loading docs from {docs_path}")
    loader = DirectoryLoader(path=docs_path,
                             glob="*.txt",
                             loader_cls=TextLoader,
                             loader_kwargs={"encoding": "utf-8"}
                             )
    documents = loader.load()

    #print first doc info
    print(f"Total docs = {len(documents)}")
    print("***************************************docs preview and info********************")
    print(f"Source: {documents[0].metadata["source"]}")
    print(f"Content length: {len(documents[0].page_content)}")
    print(f"Content Preview: {documents[0].page_content[:100]}")
    print(f"Metadata: {documents[0].metadata}")
    print("***************************************END of docs preview and info********************")

    return documents

def split_documents(documents, chunk_size=800, chunk_overlap=0):
    '''
    Split documents into smaller chunks without overap
    
    :param documents: input list of docs return from load_documents()
    :param chunk_size: size of chunk
    :param chunk_overlap: Description
    '''
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)

    print(f"Total chunks = {len(chunks)}")
    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    '''
    create and persist ChromaDB vector store
    
    :param chunks: chunks list that has all the docs merge and broken into chunks
    :param persist_directory: directory to store ChromaDB
    '''
    print("Creating embeeding and storing in ChromaDB")
    embedding_model = OllamaEmbeddings(model="embeddinggemma")
    print("--- Creating Vector store ---")
    vectorstore = Chroma.from_documents(documents=chunks,
                                        embedding=embedding_model,
                                        persist_directory=persist_directory,
                                        collection_metadata={"hnsw:space":"cosine"})
    print("--- Finished creating vector store ---")
    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore

def main():
    #1. Loading the files
    documents = load_documents(docs_path="training_docs")

    #2. Chunking the files
    chunks = split_documents(documents=documents,)

    #3. Embedding and storing in Chroma Vector DB
    vectorstore = create_vector_store(chunks=chunks)
    

if __name__=="__main__":
    main()
