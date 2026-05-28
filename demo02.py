from __future__ import annotations

from dotenv import load_dotenv

"""LangChain + DeepSeek 简单对话 Demo

运行后会进入交互式对话模式，用户可以在终端中持续输入问题，输入 `exit` / `quit` / `q`
退出。

依赖：
- langchain
- langchain-openai

环境变量：
- DEEPSEEK_API_KEY：DeepSeek API Key
- DEEPSEEK_BASE_URL：可选，默认值为 `https://api.deepseek.com`

安装示例：
```bash
pip install langchain langchain-openai
```
"""

import os

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()
def build_llm() -> ChatOpenAI:
    """构建 DeepSeek Chat 模型。"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL")

    if not api_key:
        raise RuntimeError(
            "请先设置环境变量 DEEPSEEK_API_KEY，"
            "例如：setx DEEPSEEK_API_KEY your_api_key"
        )

    return ChatOpenAI(
        model="deepseek-v4-flash",
        api_key=api_key,
        base_url=base_url,
        temperature=0.7,
    )


def main() -> None:
    llm = build_llm()

    messages = [
        SystemMessage(content="你是一个简洁、友好的客服助手，请用中文回答用户问题。"),
    ]

    print("DeepSeek 对话已启动，输入 exit / quit / q 结束。")

    while True:
        user_input = input("\n你：").strip()
        if user_input.lower() in {"exit", "quit", "q"}:
            print("已退出对话。")
            break
        if not user_input:
            continue

        messages.append(HumanMessage(content=user_input))
        response = llm.invoke(messages)

        if isinstance(response, AIMessage):
            assistant_text = response.content
        else:
            assistant_text = str(response.content)

        print(f"助手：{assistant_text}")
        messages.append(AIMessage(content=assistant_text))


if __name__ == "__main__":
    main()
