import sys
import os
import re
import warnings
import pandas as pd
import camelot

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document
from pyprojroot import here 

BASE_ROOT = here()
sys.path.append(str(BASE_ROOT))

CHROMA_COLLECTION_NAME = "documents"
CHROMA_PERSIST_DIR = here("chroma_db")
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

LOG_DIR = here("logs")
os.makedirs(LOG_DIR, exist_ok=True)

warnings.filterwarnings("ignore", message="Could get FontBBox.*")


def save_chunks_to_log(filename, chunks):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"--- TOTAL CHUNKS: {len(chunks)} ---\n\n")
        for i, chunk in enumerate(chunks):
            f.write(f"--- CHUNK {i} ---\n")
            f.write(f"Source: {chunk.metadata.get('source', 'N/A')}\n")
            f.write(f"Type: {chunk.metadata.get('type', 'N/A')}\n")
            f.write(f"Row/Page: {chunk.metadata.get('row_index', chunk.metadata.get('page', 'N/A'))}\n")
            f.write(f"Length (chars): {len(chunk.page_content)}\n")
            f.write(f"Content:\n{chunk.page_content}\n")
            f.write("-" * 80 + "\n\n")


def load_excel(file_path: str):
    try:
        df = pd.read_excel(file_path)
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        df = df.fillna('')
    except Exception as e:
        print(f"❌ Error reading Excel file {file_path}: {e}")
        return []

    documents = []
    headers = [str(h) for h in df.columns]

    # Header doc
    documents.append(Document(
        page_content="This table has the following columns: " + ", ".join(headers),
        metadata={"source": file_path, "type": "excel_header"}
    ))

    # Full table doc
    # markdown_table = df.to_markdown(index=False)
    # documents.append(Document(
    #     page_content=markdown_table,
    #     metadata={"source": file_path, "type": "excel_full", "columns": ", ".join(headers)}
    # ))

    # Split jika tabel besar
    # MAX_ROWS = 10
    # if len(df) > MAX_ROWS:
    #     for i in range(0, len(df), MAX_ROWS):
    #         chunk_df = df.iloc[i:i+MAX_ROWS]
    #         documents.append(Document(
    #             page_content=chunk_df.to_markdown(index=False),
    #             metadata={
    #                 "source": file_path,
    #                 "type": "excel_full_chunk",
    #                 "chunk_index": i // MAX_ROWS,
    #                 "columns": ", ".join(headers)
    #             }
    #         ))

    # Row-level docs
    for idx, row in df.iterrows():
        row_content = " | ".join([f"{col}: {row[col]}" for col in headers if str(row[col])])
        documents.append(Document(
            page_content=row_content,
            metadata={
                "source": file_path,
                "type": "excel_row",
                "row_index": int(idx),
                "columns": ", ".join(headers)
            }
        ))

    return documents


def load_pdf_tables(file_path: str):
    documents = []
    try:
        tables = camelot.read_pdf(file_path, pages="all", flavor="lattice")
        if len(tables) == 0:
            tables = camelot.read_pdf(file_path, pages="all", flavor="stream")

        for table in tables:
            df = table.df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
            headers = df.iloc[0].tolist()

            # Header doc
            documents.append(Document(
                page_content="This table has the following columns: " + ", ".join(headers),
                metadata={"source": file_path, "type": "pdf_table_header", "page": table.page}
            ))

            # Full table
            markdown_table = df.to_markdown(index=False)
            documents.append(Document(
                page_content=markdown_table,
                metadata={"source": file_path, "type": "pdf_table_full", "page": table.page}
            ))

            # Split jika tabel besar
            MAX_ROWS = 10
            if len(df) > MAX_ROWS:
                for j in range(0, len(df), MAX_ROWS):
                    chunk_df = df.iloc[j:j+MAX_ROWS]
                    documents.append(Document(
                        page_content=chunk_df.to_markdown(index=False),
                        metadata={
                            "source": file_path,
                            "type": "pdf_table_full_chunk",
                            "page": table.page,
                            "chunk_index": j // MAX_ROWS
                        }
                    ))

            # Row-level docs
            for row_idx, row in df.iloc[1:].iterrows():
                row_dict = dict(zip(headers, row))
                row_text = " | ".join([f"{k}: {v}" for k, v in row_dict.items()])
                documents.append(Document(
                    page_content=row_text,
                    metadata={
                        "source": file_path,
                        "type": "pdf_table_row",
                        "page": table.page,
                        "row_index": int(row_idx)
                    }
                ))

    except Exception as e:
        print(f"⚠️ Could not process tables in {file_path}: {e}")
    return documents



def load_and_process_text_docs(documents_path: str):
    print("📄 Processing text documents...")
    pdf_loader = DirectoryLoader(
        documents_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    raw_text_docs = pdf_loader.load()

    processed_docs = []
    for doc in raw_text_docs:
        enriched_content = ""
        lines = doc.page_content.split('\n')
        potential_header = ""

        for line in lines:
            stripped_line = line.strip()
            if not stripped_line:
                continue

            if len(stripped_line) < 80 and not stripped_line.endswith(('.', ':', '●')) and potential_header == "":
                potential_header = stripped_line
            else:
                if potential_header:
                    enriched_content += f"{potential_header}\n\n{stripped_line}\n"
                    potential_header = ""
                else:
                    enriched_content += stripped_line + "\n"

        if potential_header:
            enriched_content += potential_header

        final_content = re.sub(r'\s+', ' ', enriched_content).strip()

        processed_docs.append(Document(
            page_content=final_content,
            metadata={**doc.metadata, "type": "pdf_text"}
        ))

    print("✅ Text documents cleaned and enriched.")
    return processed_docs



def ingest_documents():
    print("🚀 Starting document ingestion process...")

    documents_path = os.path.join(BASE_ROOT, 'data', 'docs')
    if not os.path.exists(documents_path) or not os.listdir(documents_path):
        print(f"❌ Error: Directory '{documents_path}' empty / not exist.")
        return

    # Load PDF text
    text_docs = load_and_process_text_docs(documents_path)

    # Load tables (PDF + Excel)
    table_docs = []
    for file in os.listdir(documents_path):
        if file.endswith(".pdf"):
            print(f"📊 Extracting tables from {file}...")
            table_docs.extend(load_pdf_tables(os.path.join(documents_path, file)))
        elif file.endswith((".xlsx", ".xls")):
            print(f"📈 Loading data from {file}...")
            table_docs.extend(load_excel(os.path.join(documents_path, file)))

    print("✂️ Splitting text documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=200
    )
    text_chunks = text_splitter.split_documents(text_docs)

    print(f"👍 Text documents split into {len(text_chunks)} chunks.")
    save_chunks_to_log(os.path.join(LOG_DIR, "text_chunks.log"), text_chunks)
    save_chunks_to_log(os.path.join(LOG_DIR, "table_chunks.log"), table_docs)

    final_chunks = text_chunks + table_docs
    if not final_chunks:
        print("❌ No chunks were generated. Aborting.")
        return

    print(f"📦 Total chunks: {len(final_chunks)}")

    print(f"🔎 Embedding chunks with '{os.getenv('OLLAMA_EMBEDDING')}'model...")
    embedding_function = OllamaEmbeddings(model=os.getenv('OLLAMA_EMBEDDING'))

    
    Chroma.from_documents(
        final_chunks,
        embedding=embedding_function,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR
    )
    
    print(f"✅ Stored {len(final_chunks)} chunks in Chroma DB at '{CHROMA_PERSIST_DIR}'")
    save_chunks_to_log(os.path.join(LOG_DIR, "chroma_final_chunks.log"), final_chunks)

    print("🎉 Ingestion process completed successfully!")

if __name__ == '__main__':
    ingest_documents()
