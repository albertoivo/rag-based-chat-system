import chromadb
client = chromadb.PersistentClient(path="./chroma_db_openai")
try:
    collection = client.get_collection("nasa_space_missions_text")
    print(f"Count: {collection.count()}")
    print(f"Peek: {collection.peek(1)}")
except Exception as e:
    print(f"Error: {e}")
