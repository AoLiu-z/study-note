from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models import ChatTongyi
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS # FASII向量数据库
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains import RetrievalQA
import os

api_key = os.environ.get("DASHSCOPE_APIP_KEY")

llm = ChatTongyi(
    model="qwen-turbo",
    api_key=api_key
)

# 读取文档
loader = TextLoader("F:\python\Learning\LLMlearning\Embedding\qa.txt",encoding = "utf-8")
docs = loader.load()

# 数据切分,调用文档分割器，分块chunk 
text_splitter = CharacterTextSplitter()
documents = text_splitter.split_documents(docs)

# 创建embedding
embedding = DashScopeEmbeddings(
    dashscope_api_key= api_key
)

# 建立向量数据库存储
vecter = FAISS.from_documents(documents,embedding)

# 查询检索
qa = RetrievalQA.from_chain_type(
    llm = llm,
    chain_type = "stuff",
    retriever = vecter.as_retriever()
)

# 创建prompt
query = "你是谁"
response = qa.invoke({"query":query})
print(response["result"])
