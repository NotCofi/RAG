import os

from langchain_classic.storage import LocalFileStore, EncoderBackedStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.stores import InMemoryStore
from langchain_chroma import Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pathlib import Path


def get_storage_paths():
    base_dir = Path(__file__).resolve().parent.parent.parent / "db_storage"

    vector_dir = base_dir / "chroma"
    kv_dir = base_dir / "kv_store"

    vector_dir.mkdir(parents=True, exist_ok=True)
    kv_dir.mkdir(parents=True, exist_ok=True)
    return str(vector_dir), str(kv_dir)

async def build_pipeline(pdf_path: str):
    """
     Builds a parent-child hierarchical retriever matching
      """
    vector_dir, kv_dir = get_storage_paths()

    loader = PyPDFLoader(pdf_path)
    documents = await loader.aload()

    embeddings = OpenAIEmbeddings(model="openai/text-embedding-3-small",
                                  api_key=os.getenv("OPENROUTER_API_KEY"),
                                  openai_api_base="https://openrouter.ai/api/v1"
                                  )

    vectorstore = Chroma(
        collection_name="production_rag_collection",
        embedding_function=embeddings,
        persist_directory=vector_dir
    )

    raw_disk_storage = LocalFileStore(kv_dir)

    def key_encoder(key: str) -> bytes: return key.encode("utf-8")
    def value_decoder(value: bytes) -> str: return value.decode("utf-8")

    docstore = EncoderBackedStore(
        store=raw_disk_storage,
        key_encoder=key_encoder,
        value_decoder=value_decoder,
        value_encoder=key_encoder,
        key_decoder=value_decoder

    )

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=75)

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )

    compressor = FlashrankRerank(top_n=10)

    compiled_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever
    )

    retriever.add_documents(documents)
    return compiled_retriever
