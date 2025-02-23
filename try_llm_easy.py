from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from try_text_splitter import load_and_process_documents

# 2. 创建向量数据库
def create_vector_store(docs, persist_directory="./chroma_db"):
    # embeddings = OpenAIEmbeddings()  # 也可以使用HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings()
    
    # 创建并持久化向量存储
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vector_store.persist()
    return vector_store

if __name__ == "__main__":
    split_docs = load_and_process_documents("text.txt")
    
    # 创建/加载向量数据库
    vector_db = create_vector_store(split_docs)
