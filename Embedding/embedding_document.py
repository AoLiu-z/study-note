from langchain_community.embeddings import DashScopeEmbeddings
import os
YOU_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
api_key = YOU_API_KEY

# 初始化Embeddings模型
embedding = DashScopeEmbeddings(
    dashscope_api_key = api_key,
    model="text-embedding-v1"#可选指定模型，默认text-embedding-v1
    )
# 1. 单句文本嵌入(适合查询场景,即将prompt嵌入进行相似度匹配)
query_text = "什么是大模型开发？"
query_embedding = embedding.embed_query(query_text)
score = query_embedding.similarity_search_with_score(query_embedding)
print(f"单句嵌入结果（向量长度）：{len(query_embedding)}")  # 通常为768或1536维
print(f"向量前5位：{query_embedding[:5]}")  # 打印向量前5个值
print(f"相似度得分是：{score}")

# 2.多文档嵌入（适合批量处理文档）
documents = [
    "什好呀大模型",
    "你是谁？",
    "Langchain是一个用于构建大模型应用的框架",
    "Embeddings技术可以将文本转化为数值向量存储到向量数据库中"
]
documents_embedding = embedding.embed_documents(documents)
print(f"\n第一句嵌入的维度：{len(documents_embedding[0])}")
print(f"多文档嵌入数量：{len(documents_embedding)}")  # 应与输入文档数量一致
print(f"第一个文档向量前5位：{documents_embedding[0][:5]}")
