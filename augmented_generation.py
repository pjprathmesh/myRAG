from retrieval_pipeline import retrive_top_k_docs
import ollama

def augment_pipeline(top_k_docs,query):
    '''
    return Augmented prompt by merging the top_k_docs and query 
    
    :param top_k_docs: retrived top k docs based on the query passed to vectorDB
    :param query: query passed to retrive docs from vsctorDB
    '''

    input_to_llm = f''' Based on the following docs seperated by "---context---" answer this query
    {query}
    Please provide clear and helpful answer using only the info from these docs. If you can't find the answer in the documents say "I don't have enough info to answer your question based on the provided documents"
    Docs:
    '''
    for i in top_k_docs:
        input_to_llm = input_to_llm + "---context---\n" + i.page_content + "\n\n"
    return input_to_llm

def generate_pipeline(prompt,model="gemma3:1b"):
    '''
    return LLM response using the prompt generated from augment_pipeline()
    
    :param prompt: prompt created by augment_pipeline()
    :param model: ollama model to use to generate response
    '''
    response = ollama.generate(model=model,prompt=prompt)
    return response

if __name__=="__main__":
    query = "how to build a conda package? with complete step wit example"
    print(f"User query = {query}")
    relevant_docs = retrive_top_k_docs(query, k=10)
    prompt = augment_pipeline(relevant_docs,query)
    response = generate_pipeline(prompt)
    print(response["response"])