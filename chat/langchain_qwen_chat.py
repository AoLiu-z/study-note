from langchain_community.chat_models import ChatTongyi
import os
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

api_key = os.environ.get("DashScope_API_KEY")
llm = ChatTongyi(model="qwen-turbo",api_key=api_key)

# 创建prompt模板
prompt = PromptTemplate(
    input_variables = ["question"],
    template = "请回答以下问题：{question}"
)
while True:
    # 创建问答链
    qa = LLMChain(llm = llm,prompt = prompt)
    print("欢迎进入智能问答系统，退出系统请按:1")
    # 进行问答
    question = input()
    if question == "1":
        print("欢迎下次继续使用，再见！")
        break
    response =qa.run(question)

    print(f"问答：{question}")
    print(f"回答：{response}")
    print("*"*50)