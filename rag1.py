import os
from langchain_community.chat_models import ChatTongyi #阿里通义前问的聊天模型
from langchain_community.vectorstores import FAISS # FASII向量数据库
from langchain_community.embeddings import DashScopeEmbeddings # 阿里的词嵌入
from langchain_text_splitters import CharacterTextSplitter # 文本切分工具
from langchain_community.document_loaders import TextLoader #读取本地txt文件
from langchain.chains import RetrievalQA

YOU_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
api_key = YOU_API_KEY

# 1.初始化通义千问模型--
llm = ChatTongyi(model_name = "qwen-turbo",dashscope_api_key = api_key)

# 2.读取数据
file_path = 'kb.txt'
loader = TextLoader(file_path,encoding="utf-8")
docs = loader.load()

# 3.切分知识库文本：每300个字符切一块，块之间重叠20字符
# 这是为了保证每块文本既不太大（方便检索），又有上下文（避免断句）
text_splitter = CharacterTextSplitter(chunk_size = 300,chunk_overlap = 20)
documents = text_splitter.split_documents(docs)

# 4.初始化向量化工具：使用阿里的DashScope API,把文本块转化为向量
embedding = DashScopeEmbeddings(dashscope_api_key = api_key)

# 5. 用FAISS创建一个向量数据库，把所有文本块存进去
# FAISS 负责做"相似度检索"，快速查找与用户提问最接近的内容块
db = FAISS.from_documents(documents,embedding)

# 6. 创建 RAG 问答链
# RetrievalQA 就是 Langchain 内置的 RAG 实现
# 先用FAISS检索最相关的文本块，然后喂给大模型(qwen-turbo)生成最终回答
qa = RetrievalQA.from_chain_type(
    llm = llm, #大语言模型
    chain_type = "stuff",# 简单的RAG模式，把检索到的文档直接拼接起来
    retriever = db.as_retriever()# 检索器，负责查找相似知识块
)

# 7. 用户提问
query = "公司员工请假的流程是怎么样的？"

# 8. 调用QA链：先检索，再生成
result = qa.invoke({"query":query})

# 9. 打印回答
print(result["result"])