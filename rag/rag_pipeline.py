import sys, os
from pprint import pprint
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.pdf_extractor import PDFExtractor
from utils.embeddings import EmbeddingGenerator
from utils.chroma import ChromaDatabase

class RAGPipeline:

    def __init__(self, pdf_paths, embedding_model_name="thenlper/gte-large", chroma_collection_name="gender_equality_docs", chroma_path="./chroma_db"):
        self.pdf_paths = pdf_paths if isinstance(pdf_paths, list) else [pdf_paths]
        self.embedding_model_name = embedding_model_name
        self.chroma_collection_name = chroma_collection_name
        self.chroma_path = chroma_path

        self.generator = EmbeddingGenerator(model_name=self.embedding_model_name)
        self.chroma = ChromaDatabase(collection_name=self.chroma_collection_name, path=self.chroma_path)

    def process_pdfs(self, chunk_size=600, overlap=100):
        all_chunks = []
        for path in self.pdf_paths:
            extractor = PDFExtractor(path)
            chunks = extractor.extract(chunk_size=chunk_size, overlap=overlap)
            embeddings = self.generator.generate_embeddings(chunks)
            self.chroma.add_embeddings(
                texts=chunks,
                embeddings=embeddings,
                prefix=path,
                metadatas=[{"source": path}] * len(chunks)
            )
            all_chunks.extend(chunks)
        return all_chunks

    def retrieve_chunks(self, query_text, top_n=5):
        """Zwraca top-N najbardziej podobnych chunków z ChromaDB dla zapytania"""
        query_emb = self.generator.generate_embeddings([query_text])[0]
        results = self.chroma.query(query_emb, n_results=top_n)
        return results


if __name__ == "__main__":
    pdf_files = ["data/inclusivity1.pdf", "data/inclusivity2.pdf", "data/inclusivity3.pdf"]
    rag = RAGPipeline(pdf_files)

    rag.process_pdfs(chunk_size=500, overlap=50)

    query_text = "Best practices to ensure that job roles are gender-neutral, promoting inclusion of male, female, and non-binary candidates, using correct pronouns and avoiding selection based on stereotypes."
    top_chunks = rag.retrieve_chunks(query_text, top_n=20)

    with open("chunks.txt", "w", encoding="utf-8") as f:
        for i, chunk in enumerate(top_chunks, 1):
            output = f"Chunk {i}:\n{chunk}\n{'-'*40}\n"
            print(output)      # nadal wypisuje w konsoli
            f.write(output)

