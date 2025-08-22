# 计算prompt与数据库文档之间的相似度
from langchain_community.embeddings import DashScopeEmbeddings
import os 
import numpy
import torch
api_key = os.environ.get("DASHSCOPE_API_KEY")
embeddings = DashScopeEmbeddings(
    dashscope_api_key = api_key,
    model="text-embedding-v1"
)

# 获取词向量
def get_embedding(text):
    response = embeddings.embed_query(text)
    return response

# 计算余弦相似度
def cosine_similarity(vec1,vec2):
    tensor1 = torch.tensor(vec1)
    tensor2 = torch.tensor(vec2)
    return numpy.dot(vec1,vec2)/(torch.norm(tensor1)*torch.norm(tensor2))

# 搜索最相似文档
def search_documents(query,documents):
    # 查询向量化
    query_text_embedding = get_embedding(query)
    # 文档向量化
    document_embeddings = [get_embedding(doc) for doc in documents]
    # 计算相似度得分
    similarties = []
    for doc_embedding in document_embeddings:
        similarty = cosine_similarity(query_text_embedding,doc_embedding)
        similarties.append(similarty)
    
    # 选择出相似度得分最高的
    most_similar_index = similarties.index(max(similarties))
    # 返回最相似文档和相似度得分
    return documents[most_similar_index],max(similarties)

if __name__ == "__main__":
    documents = [
        "openai的chatgpt是一个强大的大语言模型",
        "天空是蓝色的，今天心情很好",
        "日期:今天是2025年8月22日",
        "人工智能会成为未来的风口",
        "AI Agent是大模型的一个应用，它能记忆决策以及行动"
    ]
    query = "天空是蓝色的"
    doc,score = search_documents(query,documents)
    print(f"最匹配的文档内容是：{doc}")
    print(f"相似度得分是：{score}")