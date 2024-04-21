import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone as PineconeLangChain

separators = ["\n\n", "\n", " ", ""]


def ingest_docs(embeddings, index_name):
    pc = PineconeVectorStore(index_name=index_name, embedding=embeddings)

    loader = ReadTheDocsLoader(
        "langchain-docs/api.python.langchain.com/en/latest/chains"
    )
    loader.encoding = "latin-1"

    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=50,
        separators=separators
    )
    documents = text_splitter.split_documents(raw_documents)

    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to add {len(documents)} to Pinecone")
    PineconeLangChain.from_documents(documents, embeddings, index_name=index_name)
    print("****Loading to vectorstore done ***")


if __name__ == "__main__":
    load_dotenv()

    INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    ingest_docs(
        embeddings=embeddings,
        index_name=INDEX_NAME
    )
