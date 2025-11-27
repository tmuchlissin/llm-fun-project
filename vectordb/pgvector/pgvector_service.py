from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores.pgvector import (
    PGVector,
    _get_embedding_collection_store,
)
from langchain_core.documents import Document
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from dotenv import load_dotenv
import logging


EmbeddingStore = _get_embedding_collection_store()[0]


class PgvectorService:
    def __init__(self, connection_string):
        load_dotenv()
        self.embeddings = OllamaEmbeddings(model="bge-m3:latest")
        self.cnx = connection_string
        self.collections = []
        self.engine = create_engine(self.cnx)
        self.EmbeddingStore = EmbeddingStore

    def get_vector(self, text):
        return self.embeddings.embed_query(text)

    def custom_similarity_search_with_scores(self, query, k=3):
        query_vector = self.get_vector(query)

        with Session(self.engine) as session:
            cosine_distance = self.EmbeddingStore.embedding.cosine_distance(
                query_vector
            ).label("distance")

            results = (
                session.query(
                    self.EmbeddingStore.document,
                    self.EmbeddingStore.custom_id,
                    cosine_distance,
                )
                .order_by(cosine_distance.asc())
                .limit(k)
                .all()
            )

        docs = [(Document(page_content=result[0]), 1 - result[2]) for result in results]

        return docs

    def update_pgvector_collection(
        self, docs, collection_name, overwrite=False
    ) -> None:
        """
        Create a new collection from documents. Set overwrite to True to delete the collection if it already exists.
        """
        logging.info(f"Creating new collection: {collection_name}")
        with self.engine.connect() as connection:
            pgvector = PGVector.from_documents(
                embeddings=self.embeddings,
                documents=docs,
                collection_name=collection_name,
                connection_string=self.cnx,
                connection=connection,
                pre_delete_collection=overwrite,
            )

    def get_collections(self) -> list:
        with self.engine.connect() as connection:
            try:
                query = text("SELECT * FROM public.langchain_pg_collection")
                result = connection.execute(query)
                collections = [row[0] for row in result]
            except:
                collections = []
        return collections

    def update_collection(self, docs, collection_name):
        """Updates a collection with data from a given blob URL."""
        logging.info(f"Updating collection: {collection_name}")
        collections = self.get_collections()

        if docs is not None:
            overwrite = collection_name in collections
            self.update_pgvector_collection(docs, collection_name, overwrite)

    def delete_collection(self, collection_name: str):
        """
        Deletes a collection completely from both tables:
        - langchain_pg_embedding
        - langchain_pg_collection
        """
        logging.info(f"Deleting collection: {collection_name}")
        with self.engine.begin() as connection:
            query = text("""
                SELECT uuid FROM public.langchain_pg_collection WHERE name = :name
            """)
            result = connection.execute(query, {"name": collection_name}).fetchone()

            if not result:
                logging.warning(f"⚠️ Collection '{collection_name}' not found.")
                return

            collection_id = result[0]

        delete_embeddings = text("""
            DELETE FROM public.langchain_pg_embedding WHERE collection_id = :cid
        """)
        connection.execute(delete_embeddings, {"cid": collection_id})

        delete_collection = text("""
            DELETE FROM public.langchain_pg_collection WHERE uuid = :cid
        """)
        connection.execute(delete_collection, {"cid": collection_id})

        logging.info(f"✅ Collection '{collection_name}' deleted successfully.")

