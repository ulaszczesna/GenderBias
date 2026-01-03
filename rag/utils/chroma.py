import chromadb

class ChromaDatabase:
    def __init__(self, collection_name: str = "gender_equality_docs", path: str = "rag/chroma_db"):
        self.client = chromadb.PersistentClient(path)
        self.collection = self.client.get_or_create_collection(name=collection_name)


    def add_embeddings(self, texts, embeddings, prefix,  metadatas= None):
        ids = [f"{prefix}_{i}" for i in range(len(texts))]
        if metadatas is None:
            metadatas = [{}] * len(texts)
        self.collection.add(documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)

    def query(self, query_embedding, n_results=10):
        results = self.collection.query(query_embeddings=[query_embedding], n_results=n_results)
        return results["documents"][0]