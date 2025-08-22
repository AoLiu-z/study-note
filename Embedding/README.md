# Embedding_document 词嵌入
1. 导包 from langchain_community.embeddings import DashScopeEmbeddings
2. 获得api_key
3. 初始化embeddings模型，DashScopeEmbeddings(api_key=,mode="")这里默认 text-embedding-v1   向量维度 1536
4. 单文本嵌入  embedding.embed_query(query_text)
5. 多文档嵌入  embedding.embed_documents(documents)

# Embedding_search 问答助手--匹配最相似文档
1. 导包 from langchain_community.embeddings import DashScopeEmbeddings
2. 获得api_key，初始化embeddings模型
3. 构造词嵌入方法get_embedding()
4. 构造相似度计算方法,使用numpy.dot()计算点积,tensor.norm(Z)计算Z的模,这里的Z是张量形式~通过torch.tensor()的方法，将list转化为张量
5. 构造搜索方法，输入query和documents，首先分别获得query和documents的向量化表示
6. 然后通过相似度计算方法获得相似度，选择最大的相似度
7. 通过.index()方法获得对应的指标，然后输出最匹配文档和相似度值
8. 创建实例计算

# Embedding_intention 意图匹配--通过外部知识库获取新知识
1. 导包，词嵌入、模型、向量数据库、文档读取、问题链、分块器
2. api_key 并初始化大模型
3. 读取文档 load
4. 文档切块
5. 初始化embedding
6. 通过FAISS，连接文档和词嵌入，将词向量存储到向量数据库
7. 通过问答链连接大模型和向量数据库，通过as_retriever()方法进行相似性检索
8. 创建prompt，通过invoke()获得回答

