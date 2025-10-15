#!/usr/bin/env python
# coding: utf-8
import uuid
import gradio as gr
from typing import Annotated, TypedDict, List

from langchain.schema import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, BaseMessage

from langchain_teddynote.tools import GoogleNews
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# -------- 도구 & 상태 --------
google_news = GoogleNews()

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    query: str

checkpointer = InMemorySaver()

# -------- 노드 구현 --------
def query_extractor(state: State):
    last_message = state.get("messages", [])[-1]
    prompt = ChatPromptTemplate.from_messages([
        ("system", "다음 문장에서 뉴스 검색에 가장 적합한 키워드나 질문을 추출합니다. 다른 말은 하지 말고 키워드만 정확히 반환합니다."),
        ("human", "{question}")
    ])
    extracted_query = llm.invoke(
        prompt.format_messages(question=last_message.content)
    ).content.strip()

    return {
        "query": extracted_query,
        "messages": [AIMessage(content=f"검색 키워드: {extracted_query}")]
    }

def news_search(state: State):
    q = state["query"]
    results = google_news.search_by_keyword(q, k=5)
    # 간단 문자열로 변환 (요약 입력용)
    text = "\n".join([f"- {r.get('content','(내용없음)')} | {r.get('url','')}" for r in results]) or "검색 결과 없음"
    return {"messages": [AIMessage(content=f"[검색결과]\n{text}")]}

def news_summarizer(state: State):
    tool_output = state.get("messages", [])[-1].content
    prompt = ChatPromptTemplate.from_messages([
        ("system", "다음 뉴스 검색 결과를 바탕으로 핵심 내용을 한국어로 간결하게 요약합니다."),
        ("human", "{tool_output}")
    ])
    summary = llm.invoke(
        prompt.format_messages(tool_output=tool_output)
    ).content.strip()

    # 동적 '인터럽트'는 실제 interrupt() 호출 대신 사용자 확인을 요구하는 메시지로 처리
    sensitive_keywords = ["논란", "민감", "갈등", "우려", "비판", "주식"]
    if any(k in summary for k in sensitive_keywords):
        caution = (
            "⚠️ 뉴스 요약에 민감 키워드가 포함되었습니다. 계속해서 최종 보고서를 생성할까요? "
            "‘예’ 또는 ‘아니오’라고 입력해 주세요."
        )
        return {"messages": [AIMessage(content=f"{summary}\n\n{caution}")]}
    return {"messages": [AIMessage(content=summary)]}

def report_generator(state: State):
    messages = state.get("messages", [])
    last = messages[-1].content if messages else ""
    # 사용자가 "예/아니오"로 응답했는지 확인
    user_said_no = any(
        isinstance(m, HumanMessage) and m.content.strip().lower() in ("아니오", "no", "n")
        for m in messages[-3:]  # 최근 몇 개만 체크
    )
    user_said_yes = any(
        isinstance(m, HumanMessage) and m.content.strip().lower() in ("예", "yes", "y")
        for m in messages[-3:]
    )

    if "최종 보고서를 생성할까요" in last and not (user_said_yes or user_said_no):
        # 아직 사용자의 의사 확인 전이면 여기서 멈추고 안내만 남김
        return {"messages": [AIMessage(content="사용자 확인 대기 중입니다. ‘예’ 또는 ‘아니오’를 입력해 주세요.")]}

    if user_said_no:
        return {"messages": [AIMessage(content="요청에 따라 보고서 생성을 중단했습니다. 다른 키워드를 시도해 보시겠어요?")]}

    # 보고서 생성
    summary = next((m.content for m in reversed(messages) if isinstance(m, AIMessage)), "")
    user_request = next((m.content for m in messages if isinstance(m, HumanMessage)), "")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "사용자의 최초 요청 '{user_request}'을 참고하여 다음 요약을 바탕으로 간결한 보고서를 작성해줘."),
        ("human", "{summary}")
    ])
    report = llm.invoke(
        prompt.format_messages(user_request=user_request, summary=summary)
    ).content.strip()
    return {"messages": [AIMessage(content=report)]}

# -------- 그래프 구성 --------
builder = StateGraph(State)
builder.add_node("query_extractor", query_extractor)
builder.add_node("news_search", news_search)
builder.add_node("news_summarizer", news_summarizer)
builder.add_node("report_generator", report_generator)

builder.add_edge(START, "query_extractor")
builder.add_edge("query_extractor", "news_search")
builder.add_edge("news_search", "news_summarizer")
builder.add_edge("news_summarizer", "report_generator")
builder.add_edge("report_generator", END)

# 정적 인터럽트 제거(핵심)
graph = builder.compile(checkpointer=checkpointer)

# -------- Gradio UI --------
def to_ui(messages: List[HumanMessage | AIMessage]):
    chat = []
    for m in messages:
        if isinstance(m, HumanMessage):
            chat.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            chat.append({"role": "assistant", "content": m.content})
    return chat

def agent_process(user_input: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    events = graph.stream({"messages": [HumanMessage(content=user_input)]},
                          config, stream_mode="values")
    for _ in events:
        snapshot = graph.get_state(config)
        yield to_ui(snapshot.values["messages"])

def create_chatbot_response(message, history, thread_id_state):
    thread_id = thread_id_state.value
    for chunk in agent_process(message, thread_id):
        yield chunk

def on_load():
    return gr.State(str(uuid.uuid4()))

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# LangGraph 뉴스 검색 에이전트 (InMemorySaver 사용)")

    thread_id_state = gr.State()
    chatbot = gr.Chatbot([], elem_id="chatbot", height=300, type="messages",
                         placeholder="뉴스 검색어를 입력하세요. 예: '오늘의 주요 경제 뉴스'")
    chat_input = gr.Textbox(show_label=False,
                            placeholder="메시지를 입력하고 Enter(또는 보내기)",
                            container=False)

    chat_input.submit(create_chatbot_response, [chat_input, chatbot, thread_id_state], [chatbot])
    demo.load(on_load, inputs=[], outputs=[thread_id_state])

demo.launch(debug=True)
demo.close()
