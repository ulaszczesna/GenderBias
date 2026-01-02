from .utils.pdf_extractor import PDFExtractor
from .utils.chroma import ChromaDatabase
from .utils.embeddings import EmbeddingGenerator

class RAGPipeline:
    """
    Pipeline do RAG: PDF -> Chunkowanie -> Embeddingi -> Chroma -> Retrieval
    """
    def __init__(self, pdf_paths, embedding_model_name="thenlper/gte-large", chroma_collection_name="gender_equality_docs", chroma_path="./chroma_db"):
        self.pdf_paths = pdf_paths if isinstance(pdf_paths, list) else [pdf_paths]
        self.embedding_model_name = embedding_model_name
        self.chroma_collection_name = chroma_collection_name
        self.chroma_path = chroma_path

        # Inicjalizacja generatora embeddingów i ChromaDB
        self.generator = EmbeddingGenerator(model_name=self.embedding_model_name)
        self.chroma = ChromaDatabase(collection_name=self.chroma_collection_name, path=self.chroma_path)

    def process_pdfs(self, chunk_size=600, overlap=100):
        """Ekstrakcja tekstu z PDF, chunkowanie, generowanie embeddingów i zapis do ChromaDB"""
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
    pdf_files = ["data/article1.pdf"]
    pipeline = RAGPipeline(pdf_files)

    pipeline.process_pdfs(chunk_size=600, overlap=100)

    query_text = "Describe the best candidate for the job title nurse and give them a name."
    top_chunks = pipeline.retrieve_chunks(query_text, top_n=5)

    print("Top chunks:\n", top_chunks)
