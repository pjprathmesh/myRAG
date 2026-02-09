from augmented_generation import augment_pipeline,generate_pipeline
from retrieval_pipeline import retrive_top_k_docs
import ollama

def update_history(query,history,model="gemma3:1b"):
    '''
    rewrite the original query to include past context 
    
    :param query: New query
    :param history: context
    :param model: model to use to rewrite the new query
    '''
    if len(history)>0:
        prompt = f'''Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question, no need of explanation. 
        
        Here's the new question = {query} and history = {history} '''
        
        response = ollama.generate(model=model,prompt=prompt)
        return response["response"]
    else:
        return query

def chatbot():
    '''
    chatbot that runs infinitely till user type "quit"
    '''
    history = ""
    count = 0
    print("Ask me questions related to Anaconda! Type 'quit' to exit")
    while True:
        query = input("\nYour question: ")
        if query.lower()=="quit":
            break
        if query=="":
            continue
        updated_query = update_history(query, history)
        #print(f"--- updated query using history = '{updated_query}' ---")
        #print(f"--- history is = '{history}' --- ")
        relevant_docs = retrive_top_k_docs(updated_query, k=10)  #Retrive  (R)
        prompt = augment_pipeline(relevant_docs,updated_query)   #Augment  (A)
        response = generate_pipeline(prompt)                     #Generate (G)
        print(response["response"])
        history = f"{count}. " + history + updated_query + "\n"
        


if __name__=="__main__":
    chatbot()