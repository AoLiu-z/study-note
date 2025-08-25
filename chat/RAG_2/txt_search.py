# 首先构建一个web
# 启动--streamlit run 文件名,结束--Ctrl+c
import tempfile
from langchain_text_splitters import RecursiveCharacterTextSplitter #文本切割
from langchain_community.embeddings import DashScopeEmbeddings  #词嵌入技术
from langchain_community.document_loaders import TextLoader  #文本读取
from langchain_community.vectorstores import FAISS    #向量数据库
from langchain_community.chat_message_histories import StreamlitChatMessageHistory  # 历史记录
from langchain_community.chat_models import ChatTongyi  #通义大模型
from langchain.memory import ConversationBufferMemory  # 记忆
from langchain_core.prompts import PromptTemplate   # 提示词
from langchain.agents import create_react_agent,AgentExecutor   # agent相关组件
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
import os
import streamlit as st

# 设置streamlit 应用的页面标题和布局
st.set_page_config(page_title="RAG Agent",layout="wide")
# 设置应用的标题
st.title("RAG Agent")

# 上传txt文件,允许上传多个文件
uploaded_files = st.sidebar.file_uploader(
    label = "上传txt文件",type = ["txt"],accept_multiple_files=True
)

# 如果没有上传文件，提示用户上传文件并停止运行
if not uploaded_files:
    st.info("请先上传TXT文档")
    st.stop()

# 创建聊天输入框
user_query = st.chat_input(placeholder="请开始提问吧")
@st.cache_resource(ttl="1h")

# 拿到一个文件对象后，最终返回向量数据库的检索对象
def configure_retriever(uploaded_files):
    # 读取上传的文档，并写入一个临时目录
    docs = []
    # 将文档写入到D盘
    temp_dir = tempfile.TemporaryDirectory(dir=r"D:\\")
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name,file.name)
        with open(temp_filepath,"wb") as f:
            f.write(file.getvalue())
        # 使用TextLoader加载文本文件
        loader = TextLoader(temp_filepath,encoding = "utf-8")
        docs.extend(loader.load())
    
    # 进行文档分割
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 300,chunk_overlap = 50)
    splits = text_splitter.split_documents(docs)
    
    # 获取api_key
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    
    # 使用通义千问的Embedding模型生成文档的向量表示
    embedding = DashScopeEmbeddings(
        dashscope_api_key= api_key,
        model="text-embedding-v1"
    )
    vectordb = FAISS.from_documents(splits,embedding)
    
    # 创建文案检索器,RAG中的“R”，检索
    retriever = vectordb.as_retriever()
    
    return retriever

# 配置检索器
retriever = configure_retriever(uploaded_files)

# 如果sessio_state 中没有消息记录或用户点击了清空聊天记录按钮，则初始化消息记录
if "messages" not in st.session_state or st.sidebar.button("清空聊天记录"):
    st.session_state["messages"] = [{"role":"assistant","content":"你好，我是私人专属AI助手，我可以查询文档"}]

# 加载聊天记录
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
    

# 创建检索工具
from langchain.tools.retriever import create_retriever_tool

# 创建用于文档检索的工具,封装一个tools--因为用了Agent，需要工具调用
tool = create_retriever_tool(
    retriever,
    name="文档检索",
    description="用于检索用户提出的问题，并基于检索到的文档内容进行回复.",
)

tools = [tool]

# 创建聊天消息历史记录
msgs = StreamlitChatMessageHistory()
# 创建对话缓冲区内存
memory = ConversationBufferMemory(
    chat_memory=msgs,return_messages=True,
    memory_key="chat_history",output_key="output"    
)

# 指令模板,通过提示词让大模型与工具进行协调作用--React
# 分析加行动
instructions = """您是一个设计用于查询文档来回答问题的代理，
您可以使用文档检索工具，并基于检索内容来回答问题。
您可能不查询文档就知道答案，但是您仍然应该查询文档来获得答案。

"""

# 定制agent提示模板
base_prompt_template = """
{instructions}

TOOLS:
-------
You have access to the following tools:

{tools}

To use a tool,please use the following format:


Thought: do i need to use a tool? yes
Action: the action to take, should be one of {tool_names}
Action Input:{input}
Observation: the result of the action


when you have a response to say to the Human, or if you do not need to use a tool, you must use the format:


Thought: do i need to use a tool? no
Final Answer: {{your response here}}

Begin!

Previous conversation history:
{chat_history}

New input:{input}
{agent_scratchpad}

"""

# 创建基础提示词模板
base_prompt = PromptTemplate.from_template(base_prompt_template)

# 创建部分填充的提示模板
prompt = base_prompt.partial(instructions = instructions,
                             tools = str(tools),
                             tool_names = tool)

 # 获取api_key
api_key = os.environ.get("DASHSCOPE_API_KEY")
    
# 创建 llm
llm = ChatTongyi(model_name = "qwen-turbo",dashscope_api_key = api_key)

# 创建 react Agent
agent = create_react_agent(llm,tools,prompt)

# 创建Agent执行器
agent_executor = AgentExecutor(agent=agent,
                               tools = tools,
                               memory=memory,
                               verbose=True,
                               handle_parsing_errors=True,
                               max_iterations=5,#添加迭代限制防止无限循环
                               early_stopping_method="generate"
                               )



# 如果有用户输入的查询
if user_query:
    # 添加用户消息到session_state
    st.session_state.messages.append({"role":"user","content":user_query})
    # 显示用户信息
    st.chat_message("user").write(user_query)
    
    with st.chat_message("assistant"):
        # 创建Streamlit问题处理器
        st_cb = StreamlitCallbackHandler(st.container())
        # agent执行过程日志问题显示在Streamlit Container （如思考、选择工具、执行查询、观察结果等）
        config = {"callbacks":[st_cb]}
        try:
            # 执行agent并获取响应
            response = agent_executor.invoke({"input": user_query}, config=config)
            # 添加助手消息到session_state
            st.session_state.messages.append({"role": "assistant", "content": response["output"]})
            # 显示助手响应
            st.write(response["output"])
        except Exception as e:
            st.error(f"执行过程中出现错误: {str(e)}")
            st.session_state.messages.append({"role": "assistant", "content": "抱歉，处理您的请求时出现了问题。"})
        




