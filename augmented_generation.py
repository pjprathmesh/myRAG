from retrieval_pipeline import retrive_top_k_docs
import ollama

def augment_pipeline(top_k_docs,query):
    input_to_llm = f''' Based on the following docs seperated by "---context---" answer this query
    {query}
    Please provide clear and helpful answer using only the info from these docs. If you can't find the answer in the documents say "I don't have enough info to answer your question based on the provided documents"
    Docs:
    '''
    for i in top_k_docs:
        input_to_llm = input_to_llm + "---context---\n" + i.page_content + "\n\n"
    print(input_to_llm)
    return input_to_llm

if __name__=="__main__":
    query = "how to build a conda package?"
    print(f"User query = {query}")
    relevant_docs = retrive_top_k_docs(query, k=10)
    augment_pipeline(relevant_docs,query)