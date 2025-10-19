import pandas as pd
import ollama
import chromadb
from pyprojroot import here
import logging


logging.basicConfig(
    filename="ingestion.log", 
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def ingest_documents():
    """
    Ingest Excel file into Chroma collection with Ollama embeddings.
    """

    file_path = here("data/docs/properti_rental.xlsx")

    chroma_client = chromadb.PersistentClient(path="data/chroma")

    try:
        collection = chroma_client.create_collection(name="property_rental")
        logging.info("Created new collection 'property_rental'")
    except:
        collection = chroma_client.get_collection(name="property_rental")
        logging.info("Loaded existing collection 'property_rental'")

    df = pd.read_excel(file_path)

    docs, metadatas, ids, embeddings = [], [], [], []

    for index, row in df.iterrows():
        output_str = "\n".join([f"{col}: {row[col]}" for col in df.columns])

        response = ollama.embeddings(
            model="bge-m3:latest",
            prompt=output_str
        )

        docs.append(output_str)
        embeddings.append(response["embedding"])
        metadatas.append({"source": str(file_path)})
        ids.append(f"id{index}")

        logging.info(f"Added document ID={ids[-1]} | Metadata={metadatas[-1]}")

    collection.add(
        documents=docs,
        metadatas=metadatas,
        embeddings=embeddings,
        ids=ids
    )

    logging.info(f"✅ {len(docs)} documents ingested into collection 'property_rental'")
    print(f"✅ {len(docs)} documents ingested into collection 'property_rental'")


if __name__ == "__main__":
    ingest_documents()
