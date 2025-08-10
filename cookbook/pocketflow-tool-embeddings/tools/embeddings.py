from utils.call_llm import client

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small-ada-002",
        input=text
    )
    return response.data[0].embedding