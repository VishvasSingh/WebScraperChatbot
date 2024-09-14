import logging
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from qdrant_client.http.models.models import VectorParams, Distance
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import SentenceTransformerEmbeddings
from data_helper import split_text

qdrant_client = QdrantClient("localhost", port=6333)

embed_model = SentenceTransformer('all-MiniLm-L6-v2')

embeddings = SentenceTransformerEmbeddings(model_name='sentence-transformers/all-MiniLm-L6-v2')


class QdrantHelper:
    def __init__(self, client: QdrantClient = qdrant_client, embedding_model=embed_model,
                 collection_name: str = "global", embeddings=embeddings):
        self.embedding_model = embedding_model
        self.client = client
        self.collection_name = collection_name
        self.embeddings = embeddings

    def ingest_data(self, collection_name: str, data: str):

        if not self.client.collection_exists(collection_name=collection_name):
            self.client.recreate_collection(collection_name=collection_name, vectors_config=self._get_vector_config())

        chunks = self._get_chunks(data)
        embeddings = self._generate_embeddings(chunks)

        points = [PointStruct(id=i, vector=embedding, payload={'page_content': chunks[i], 'metadata': {}})
                  for i, embedding in enumerate(embeddings)]

        self.client.upsert(collection_name=collection_name, points=points)

        logging.info("DATA INGESTION COMPLETED SUCCESSFULLY")

    def get_relevant_chunks(self, query: str, collection_name: str, limit: int = 5, **kwargs):
        query_embedding = self._generate_embeddings(query)
        search_results = self.client.search(collection_name=collection_name, query_vector=query_embedding[0],
                                            limit=limit)

        return [hit.payload.get("text") for hit in search_results]

    def _get_vector_config(self) -> VectorParams:
        vector_config = VectorParams(size=self.embedding_model.get_sentence_embedding_dimension(),
                                     distance=Distance.COSINE)

        return vector_config

    def get_vector_store(self, collection_name: str):
        collection_name = collection_name or self.collection_name
        qdrant = Qdrant(client=self.client,
                        embeddings=embeddings,
                        collection_name=collection_name)

        return qdrant

    def ingest_data_with_langchain_qdrant_client(self, data: str, collection_name: str):
        if not self.client.collection_exists(collection_name=collection_name):
            self.client.recreate_collection(collection_name=collection_name, vectors_config=self._get_vector_config())

        documents_list = [Document(page_content=chunk) for chunk in split_text(data)]
        vector_store = self.get_vector_store(collection_name)
        vector_store.add_documents(documents_list)

        logging.info("DATA INGESTION COMPLETED SUCCESSFULLY")

    def embedding_func(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        return self.embedding_model.encode(texts).tolist()

    @staticmethod
    def _get_chunks(data: str) -> list:
        """
        Data cleaning logic
        :param data: str
        :return: chunks
        """
        chunks = data.split('.')

        return chunks

    def _generate_embeddings(self, chunks):
        embeddings = self.embedding_model.encode(chunks)

        return embeddings


