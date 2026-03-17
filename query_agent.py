from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

class Query_Agent:

    def __init__(self, pinecone_key, pinecone_index_name, pinecone_namespace, openai_client, embeddings) -> None:
        """
        pinecone_index : Pinecone Index object
        openai_client  : OpenAI client
        embeddings     : embedding model
        """
        self.pinecone_key = pinecone_key
        self.index_name = pinecone_index_name
        self.index_ns = pinecone_namespace
        self.client = openai_client
        self.embeddings = embeddings

        self.vectorstore = PineconeVectorStore(index_name = self.index_name, embedding = embeddings, pinecone_api_key=pinecone_key)

    def query_vector_store(self, query, k=5):

        # Query Pinecone

        results = self.vectorstore.similarity_search(query = query, k = k, namespace = self.index_ns)

        """
        results = self.index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True
        )
        """


        return results

    def run(self, query, k=10):
        """
        Main entry point for the Head Agent to retrieve PineCone results
        Relevance filtering is handled by Relevant_Documents_Agent.
        """
        return self.query_vector_store(query, k=k)