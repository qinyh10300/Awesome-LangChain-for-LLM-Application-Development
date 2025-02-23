from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from try_text_splitter import load_and_process_documents

# 2. 创建向量数据库
def create_vector_store(docs, persist_directory="./chroma_db"):
    # embeddings = OpenAIEmbeddings()  # 也可以使用HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/sentence-t5-large")
    
    # 创建并持久化向量存储
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        # persist_directory=persist_directory,
        # # persist_directory参数用于指定向量存储的持久化目录。这个目录用于存储向量数据库，以便在程序重新启动或系统重启后，可以从该目录中加载向量数据，而不需要重新计算嵌入向量。
        collection_name="my_collection"
    )
    
    # # 会自动进行persist
    # vector_store.persist()

    # # 打印每个文档的嵌入向量
    # for i, doc in enumerate(docs):
    #     embedding_vector = embeddings.embed_documents([doc.page_content])
    #     print(f"Document {i} embedding: {embedding_vector}")

    return vector_store

if __name__ == "__main__":
    split_docs = load_and_process_documents("text.txt")
    
    # # 打印文档内容
    # for i, doc in enumerate(split_docs):
    #     print(f"Document {i}: {doc.page_content}")

    # 创建/加载向量数据库
    vector_db = create_vector_store(split_docs)
    # print(vector_db)

    query = "RAG是什么?"
    result = vector_db.similarity_search(query, k=4)

    # 打印相似性搜索结果
    for i, doc in enumerate(result):
        print(f"Result {i}: {doc.page_content}")
