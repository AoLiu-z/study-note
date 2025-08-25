# 基于Streamlit构建的简易web版 RAG+Agent
**应用描述**：
1. 可以上传文件，目前仅支持txt文本文件的上传（原因是只有txt版本的loader，后续可增加pdf版、html版等加载器）
2. 可以检索上传文件，结合文件回答问题---RAG主要功能，检索增强生成（生成体现比较弱）
3. 通过调用检索工具进行检索，Agent的雏形~（目前只有一个检索工具，后续可再工具序列中增加工具，使得Agent更加完善智能）
4. 可以记忆上下文，但记忆长度有限，存在幻觉现象

**构建流程**：
1. 根据Streamlit构建一个简易web，仅包含一个上传文件按钮、一个对话框、一个聊天输入框
2. 定义文件处理方法，上传文件后，需要最终返回向量数据库的检索对象
    2.1 文件处理方法--读取上传文件，利用TextLoader加载文本文件，通过loader.load()方法获得对应对象，紧接着对文档进行分割，使用RecursiveCharacterSplitter()方法和.split_documents()
    2.2 获取api_key，使用通义千问的Embedding模型生成文档的向量表示，调用DashScopeEmbeddings()方法
    2.3 通过FAISS建立向量数据库，as_retriever()建立文本检索器，然后返回检索对象
3. 配置检索器，即实例化上述方法
4. 创建检索工具，create_retriever_tool()方法
5. 创建聊天历史记录以及对话缓冲区内存
6. 建立指令模板--通过提示词让大模型与工具进行协同作用，React（分析加行动）
7. 获取api，建立千问大模型
8. 创建 agent对象以及agent_executor agent执行器
9. 判断用户是否输入查询，然后执行agent并获取响应，得到回复
