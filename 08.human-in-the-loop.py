#!/usr/bin/env python
# coding: utf-8

# # LangGraph HITL (Human in the Loop)
# 
# - **동적 인터럽트 (Dynamic Interrupts)**: 특정 노드 내에서 코드 실행 중 조건에 따라 `interrupt()` 함수를 호출하여 그래프를 일시 중지
# - **정적 인터럽트 (Static Interrupts)**: 그래프를 컴파일할 때 `interrupt_before` 또는 `interrupt_after` 인자를 사용하여 특정 노드의 실행 전후에 항상 일시 중지되도록 설정

# In[ ]:


import uuid
import gradio as gr
from typing import Annotated, TypedDict

from langchain.schema import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_teddynote.graphs import visualize_graph
from langchain_teddynote.tools import GoogleNews

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, BaseMessage
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from dotenv import load_dotenv


load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ## 도구 및 상태 구성

# In[ ]:


# 구글 뉴스 도구
google_news= GoogleNews()

# 키워드 뉴스 검색 도구 생성
@tool
def search_keyword(query: str) -> list[dict[str, str]]:
    """Look up news by keyword"""
    print("✅ 뉴스 검색 도구 실행")
    return google_news.search_by_keyword(query, k=5)

tools = [search_keyword]

# 상태 저장소 정의
class State(TypedDict):
    messages: Annotated[list[BaseMessage], "메시지 목록", add_messages]
    query: Annotated[str, "뉴스 검색 키워드"]

# 인메모리 체크포인터
checkpointer = InMemorySaver()


# ## 그래프 노드 정의

# In[ ]:


def query_extractor(state: State):
    """사용자의 마지막 메시지에서 뉴스 검색어를 추출합니다."""
    last_message = state.get("messages", [])[-1]
    print("✅ 뉴스 검색어 추출 중...")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "다음 문장에서 뉴스 검색에 가장 적합한 키워드나 질문을 추출합니다. 다른 말은 하지 말고 키워드만 정확히 반환해줍니다."),
        ("human", "{question}")
    ])
    prompt_message = prompt.format_messages(question=last_message.content)
    response = llm.invoke(prompt_message)
    extracted_query = response.content.strip()
    print(f"✅ 추출된 검색어: {extracted_query}")
    return {"messages": [AIMessage(content=extracted_query)]}

def news_summarizer(state: State):
    """뉴스 검색 결과를 받아 AI가 요약하고, 동적 인터럽트를 결정합니다."""
    tool_output = state.get("messages", [])[-1].content
    print("✅ 뉴스 검색 결과 요약 중...")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "다음 뉴스 검색 결과를 바탕으로 핵심 내용을 한국어로 간결하게 요약합니다."),
        ("human", "{tool_output}")
    ])
    prompt_message = prompt.format_messages(tool_output=tool_output)
    summary = llm.invoke(prompt_message).content
    print(f"✅ 뉴스 요약:\n{summary}")

    # 동적 인터럽트: 특정 키워드가 포함되면 사용자의 확인을 받기 위해 실행을 멈춤
    sensitive_keywords = ["논란", "민감", "갈등", "우려", "비판"]
    if any(keyword in summary for keyword in sensitive_keywords):
        print("🔴 동적 인터럽트 발생: 민감 키워드 감지")
        interrupt()

    return {"messages": [AIMessage(content=summary)]}

def report_generator(state: State):
    """요약된 내용을 바탕으로 최종 보고서를 생성합니다."""
    summary = state.get("messages", [])[-1].content
    # 최초 요청을 찾기 위해 HumanMessage 필터링
    user_request = next((msg.content for msg in state.get("messages", []) if isinstance(msg, HumanMessage)), "")
    print("✅ 최종 보고서 생성 중...")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "사용자의 최초 요청사항인 '{user_request}'을 참고하여, 다음 뉴스 요약 내용을 바탕으로 간결한 보고서를 작성해줘."),
        ("human", "{summary}")
    ])
    prompt_message = prompt.format_messages(user_request=user_request, summary=summary)
    report = llm.invoke(prompt_message).content
    print(f"✅ 생성된 보고서:\n{report}")
    return {"messages": [AIMessage(content=report)]}

# tool 노드 생성
tool_node = ToolNode(tools)


# ## 그래프 구성 및 컴파일

# In[ ]:


# --- 그래프 구성 및 컴파일 ---
builder = StateGraph(State)

builder.add_node("query_extractor", query_extractor)
builder.add_node("news_search", tool_node)
builder.add_node("news_summarizer", news_summarizer)
builder.add_node("report_generator", report_generator)

builder.add_edge(START, "query_extractor")
builder.add_edge("query_extractor", "news_search")
builder.add_edge("news_search", "news_summarizer")
builder.add_edge("news_summarizer", "report_generator")
builder.add_edge("report_generator", END)

# 정적/동적 인터럽트 설정과 함께 그래프 컴파일
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["news_search"]
)
visualize_graph(graph)


# ## Gradio UI 및 실행 로직

# In[ ]:


def convert_messages_to_ui(messages: list[HumanMessage | AIMessage]) -> list[dict[str, str]]:
    """LangChain 메시지 형식을 Gradio 챗봇 형식으로 변환합니다."""
    chat_history = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            chat_history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            chat_history.append({"role": "assistant", "content": msg.content})
    return chat_history

def agent_process(user_input: str, thread_id: str):
    """사용자 입력에 따라 LangGraph 에이전트를 실행하고 상호작용합니다."""
    config = {"configurable": {"thread_id": thread_id}}

    # HumanMessage 추가
    message = HumanMessage(content=user_input)

    # 그래프 실행
    events = graph.stream(
        {"messages": [message]},
        config,
        stream_mode="values",
    )

    for event in events:
        snapshot = graph.get_state(config)
        ui_messages = convert_messages_to_ui(snapshot.values["messages"])
        yield ui_messages

def create_chatbot_response(message, history, thread_id_state):
    """Gradio 챗봇의 메인 콜백 함수"""
    thread_id = thread_id_state.value

    for chunk in agent_process(message, thread_id):
        yield chunk

def on_load():
    """UI가 로드될 때 고유한 thread_id를 생성합니다."""
    return gr.State(str(uuid.uuid4()))


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# LangGraph 뉴스 검색 에이전트 (InMemorySaver 사용)")

    thread_id_state = gr.State()

    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        height=300,
        type="messages",
        placeholder="뉴스 검색어를 입력하세요. 예: '오늘의 주요 경제 뉴스'"
    )

    chat_input = gr.Textbox(
        show_label=False, 
        placeholder="여기에 메시지를 입력하고 Enter를 누르거나 '보내기' 버튼을 클릭하세요.", 
        container=False
    )

    chat_input.submit(
        create_chatbot_response,
        [chat_input, chatbot, thread_id_state],
        [chatbot],
    )

    demo.load(on_load, inputs=[], outputs=[thread_id_state])


demo.launch(debug=True)


# In[ ]:


demo.close()


# -----
# ** End of Documents **
