# Benim hesap

import pinecone
import gradio as gr
import pinecone
import openai



def ask(Pinecone_api,OpenAI_key,query):


    index_name = 'gpt-4-langchain-docs'

# initialize connection to pinecone
    pinecone.init(
    api_key=Pinecone_api,  # app.pinecone.io (console)
    environment="us-west4-gcp"  # next to API key in console
)

# check if index already exists (it shouldn't if this is first time)
    if index_name not in pinecone.list_indexes():
    # if does not exist, create index
        pinecone.create_index(
        index_name,
        dimension=len(res['data'][0]['embedding']),
        metric='dotproduct'
    )
# connect to index
    index = pinecone.GRPCIndex(index_name)
    
    
    openai.api_key = OpenAI_key  #platform.openai.com

    embed_model = "text-embedding-ada-002"

    res = openai.Embedding.create(
    input=[
        "Sample document text goes here",
        "there will be several phrases in each batch"
    ], engine=embed_model
)
    
    index_name = 'gpt-4-langchain-docs'

    
    
  
    
    
    res = openai.Embedding.create(
    input=[query],
    engine=embed_model
)

# retrieve from Pinecone
    xq = res['data'][0]['embedding']

# get relevant contexts (including the questions)
    res = index.query(xq, top_k=5, include_metadata=True)

    contexts = [item['metadata']['text'] for item in res['matches']]

    augmented_query = "\n\n---\n\n".join(contexts)+"\n\n-----\n\n"+query

    primer = f"""You are Q&A bot. A highly intelligent system that answers
user questions based on the information provided by the user above
each question. If the information can not be found in the information
provided by the user you truthfully say "I don't know".
"""

    res = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": primer},
        {"role": "user", "content": augmented_query}
    ]
)
    from IPython.display import Markdown

    response = (res['choices'][0]['message']['content'])
  
    
    return response
     
demo = gr.Interface(title = 'ShipsGo AI Assistant' , fn=ask, inputs=["text","text","text"] , outputs="text")
demo.launch() 