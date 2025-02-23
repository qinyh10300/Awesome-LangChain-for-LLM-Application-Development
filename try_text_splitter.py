from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# 1. 文档加载与处理
def load_and_process_documents(file_path):
    # 使用文本加载器（支持多种格式：PDF、CSV等）
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()
    
    # 文本分割（自定义参数根据需求调整）
    text_splitter = RecursiveCharacterTextSplitter(
        separators=[
            "\n\n",
            "\n",
            " ",
            ".",
            ",",
            "\u200B",  # Zero-width space
            "\uff0c",  # Fullwidth comma
            "\u3001",  # Ideographic comma
            "\uff0e",  # Fullwidth full stop
            "\u3002",  # Ideographic full stop
            "",
        ],
        chunk_size=200,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False
    )
    return text_splitter.split_documents(documents)

if __name__ == "__main__":
    texts = load_and_process_documents("text.txt")
    print(texts)
    print(texts[0])
    print(texts[1])