# 安装必要库（确保版本兼容）
# !pip install langchain openai chromadb tiktoken

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. 文档加载与处理
def load_and_process_documents(file_path):
    # 使用文本加载器（支持多种格式：PDF、CSV等）
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()
    
    # 文本分割（自定义参数根据需求调整）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False
    )
    return text_splitter.split_documents(documents)

# 2. 创建向量数据库
def create_vector_store(docs, persist_directory="./chroma_db"):
    embeddings = OpenAIEmbeddings()  # 也可以使用HuggingFaceEmbeddings
    
    # 创建并持久化向量存储
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vector_store.persist()
    return vector_store

# 3. 构建RAG链
def create_rag_chain(vector_store):
    # 定义检索器（可调整搜索参数）
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # 定义提示模板（可自定义修改）
    template = """基于以下上下文信息回答问题：
    {context}
    
    问题：{question}
    
    请用中文给出详细、专业的回答。如果不知道答案，请说明。"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # 初始化大语言模型（可更换为其他模型）
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
    
    # 构建处理链
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

# 使用示例
if __name__ == "__main__":
    
    # 处理文档（替换为你的文件路径）
    split_docs = load_and_process_documents("text.txt")
    
    # 创建/加载向量数据库
    vector_db = create_vector_store(split_docs)
    
    # 构建RAG链
    rag_chain = create_rag_chain(vector_db)
    
    # 进行问答
    while True:
        question = input("\n请输入问题（输入q退出）: ")
        if question.lower() == 'q':
            break
        print("\n生成回答：")
        for chunk in rag_chain.stream(question):
            print(chunk, end="", flush=True)
        print("\n" + "-"*50)