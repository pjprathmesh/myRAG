import os
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def retrive_top_k_docs(query, 
                       k=10, 
                       persist_directory = "db/chroma_db", 
                       score_threshold=0.3, 
                       model="embeddinggemma"
                       ):
    '''
    retrive top k docs by cosine similarity
    
    :param query: query to retrive top_k_docs
    :param k: top_k_docs
    :param persist_directory: directory where vectorDB exist
    :param score_threshold: cosine similarity score threshold
    :param model: model to use for embedding generation
    '''
    embedding_model = OllamaEmbeddings(model=model)

    vectorstore = Chroma(embedding_function=embedding_model,
                        persist_directory=persist_directory,
                        collection_metadata={"hnsw:space":"cosine"})


    retriever = vectorstore.as_retriever(search_type = "similarity_score_threshold",
                                    search_kwargs = {"k": k, "score_threshold": score_threshold})

    relevant_docs = retriever.invoke(query)

    return relevant_docs

query = "how to build a conda package?"
print(f"User query = {query}")
relevant_docs = retrive_top_k_docs(query, k=10)

if __name__=="__main__":
    query = "how to build a conda package?"
    print(f"User query = {query}")
    relevant_docs = retrive_top_k_docs(query, k=10)

    # print("--- context ---")
    # for i,doc in enumerate(relevant_docs):
    #     print(f"doc {i}")
    #     print(doc.page_content)
