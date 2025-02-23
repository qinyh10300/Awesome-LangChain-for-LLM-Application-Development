from langchain_core.prompts import ChatPromptTemplate
from SiliconFlow import SiliconFlow
from Embedding import load_and_process_documents, create_vector_store

# 创建 RAG 链的函数
def ask_rag_chain(file_path, query):
    split_docs = load_and_process_documents(file_path)
    
    vector_db = create_vector_store(split_docs)

    template = """基于以下上下文信息回答问题：
    {context}
    
    问题：{query}
    
    请用中文给出详细、专业的回答。如果不知道答案，请说明。"""

    result = vector_db.similarity_search(query, k=4)
    
    context = ""
    for i, doc in enumerate(result):
        context += doc.page_content
        # print(f"Result {i}: {doc.page_content}")

    # print(context)
    # prompt = ChatPromptTemplate.from_template(formatted_prompt)
    # print(prompt, type(prompt))

    # 将 prompt 中的 {context} 和 {query} 替换为实际值
    formatted_prompt = template.format(context=context, query=query)
    print(formatted_prompt)

    llm = SiliconFlow()
    messages = [SiliconFlow.HumanMessage(content=formatted_prompt)]
    response = llm.call(messages=messages, model="deepseek-ai/DeepSeek-V2.5", temperature=0.9)
    
    return response

if __name__ == "__main__":
    question = "RAG是什么?"

    response = ask_rag_chain(file_path = "./RAG_Implement/text.txt", query = question)

    print(response["content"])
    # print("\n" + "-"*50)