# 学习笔记
## 1. RAG_1
- Langchain+通义千问完成一个简单的知识库问答系统
- 流程如下：
  1. 导包，ChatTongyi、FAISS向量数据库、DashScopeEmbeddings阿里的词嵌入、CharacterTextSplitter文本切分工具、TextLoader读取本地文件
  2. 将qwen api_key放置在环境变量中，通过os.environ.get()获得api_key
  3. 初始化通义千问模型ChatTongyi(model_name = "qwen-turbo",dashscope_api_key = api_key)
  4. 读取本地文件,文件地址，实例化，load方法，path、TextLoader(path,encoding="")、loader.load()
  5. 切分知识库文本，CharacterTextSplitter(chunk_size=a,chunk_overlap=b),a--每块大小、b--每块之前重叠大小（防止断句，减少语义缺失）,split_documents()
  6. 初始化向量化工具，DashScopeEmbeddings(dashscope_api_key = api_key)，得到词嵌入
  7. FAISS创建向量数据库，把所有文本块存进去
  8. 创建RAG问答链，输入问题进行相似度检索，从向量数据库获得最相关的文本块，同问题一起喂给大模型生成答案
  9. 用户提问
  10. 调用问答链，调用invoke()方法
  11. 输出答案
