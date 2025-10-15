#!/usr/bin/env python
# coding: utf-8
import uuid
import gradio as gr
from typing import Annotated, TypedDict, List, Optional

from langchain.schema import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, BaseMessage
from langgraph.types import interrupt  # ✅ 실제 일시정지

from langchain_teddynote.tools import GoogleNews
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
google_news = GoogleNews()

# -----------------------------
# 상태 정의
# -----------------------------
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    query: Optional[str]
    decision: Optional[str]   # "yes" | "no" | None

checkpointer = InMemorySaver()

# -----------------------------
# 노드들
# -----------------------------
def query_extractor(state: State):
    """사용자 입력에서 검색 키워드 추출"""
    last = state["messages"][-1]
    prompt = ChatPromptTemplate.from_messages([
        ("system", "다음 문장에서 뉴스 검색에 가장 적합한 키워드만 반환하세요."),
        ("human", "{q}")
    ])
    extracted = llm.invoke(
        prompt.format_messages(q=last.content)
    ).content.strip()

    return {
        "query": extracted,
        "messages": [AIMessage(content=f"검색 키워드: {extracted}")]
    }

def approval_gate(state: State):
    """
    진짜 HITL: 여기서 멈추고 사용자 승인 받기.
    - 1회차 실행: 승인 질문 메시지를 남기고 interrupt()
    - 2회차(재개) 실행: 최근 사용자 메시지로 '예/아니오' 판정 → decision 세팅
    """
    # 최근 사용자 답변에 '예/아니오'가 존재하는지 검사
    recent_user = None
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            recent_user = m.content.strip().lower()
            break

    # 아직 승인/거절 의사 표시가 없으면 질문 띄우고 정지
    if recent_user not in ("예", "yes", "y", "아니오", "no", "n"):
        ask = (
            "🔒 뉴스 검색을 진행하기 위해 승인이 필요합니다.\n"
            "계속하시겠습니까? ‘예’ 또는 ‘아니오’로 답해주세요."
        )
        # 메시지를 남기고 실제로 중지
        interrupt({"messages": [AIMessage(content=ask)]})
        # note: interrupt 후 반환 코드는 실행되지 않지만, 타입 충족을 위해 값 리턴
        return {"messages": [AIMessage(content=ask)]}

    # 사용자가 응답했다면 decision 기록
    decision = "yes" if recent_user in ("예", "yes", "y") else "no"
    return {"decision": decision}

def news_search(state: State):
    """실제 뉴스 검색"""
    q = state.get("query") or ""
    results = google_news.search_by_keyword(q, k=5)
    text = "\n".join([f"- {r.get('title','(제목없음)')} | {r.get('link','')}" for r in results]) or "검색 결과 없음"
    return {"messages": [AIMessage(content=f"[검색결과]\n{text}")]}

def news_summarizer(state: State):
    """검색 결과 요약"""
    tool_output = state["messages"][-1].content
    prompt = ChatPromptTemplate.from_messages([
        ("system", "다음 뉴스 검색 결과를 바탕으로 한국어로 간결하게 요약하세요."),
        ("human", "{t}")
    ])
    summary = llm.invoke(
        prompt.format_messages(t=tool_output)
    ).content.strip()
    return {"messages": [AIMessage(content=summary)]}

def report_generator(state: State):
    """최종 보고서 작성"""
    messages = state["messages"]
    summary = next((m.content for m in reversed(messages) if isinstance(m, AIMessage)), "")
    user_request = next((m.content for m in messages if isinstance(m, HumanMessage)), "")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "사용자의 최초 요청 '{user_request}'을 참고하여 다음 요약으로 간결한 보고서를 작성하세요."),
        ("human", "{summary}")
    ])
    report = llm.invoke(
        prompt.format_messages(user_request=user_request, summary=summary)
    ).content.strip()
    return {"messages": [AIMessage(content=report)]}

def aborted(state: State):
    """사용자가 거절했을 때 종료 메시지"""
    return {"messages": [AIMessage(content="요청에 따라 검색을 진행하지 않았습니다. 다른 키워드를 시도해 보세요.")]}

# -----------------------------
# 조건 라우팅
# -----------------------------
def route_after_approval(state: State) -> str:
    if state.get("decision") == "yes":
        return "news_search"
    else:
        return "aborted"

# -----------------------------
# 그래프 구성
# -----------------------------
builder = StateGraph(State)

builder.add_node("query_extractor", query_extractor)
builder.add_node("approval_gate", approval_gate)
builder.add_node("news_search", news_search)
builder.add_node("news_summarizer", news_summarizer)
builder.add_node("report_generator", report_generator)
builder.add_node("aborted", aborted)

builder.add_edge(START, "query_extractor")
builder.add_edge("query_extractor", "approval_gate")
builder.add_conditional_edges("approval_gate", route_after_approval,
                              {"news_search": "news_search", "aborted": "aborted"})
builder.add_edge("news_search", "news_summarizer")
builder.add_edge("news_summarizer", "report_generator")
builder.add_edge("report_generator", END)
builder.add_edge("aborted", END)

graph = builder.compile(checkpointer=checkpointer)

# -----------------------------
# Gradio UI (일시정지 → 승인 → 재개)
# -----------------------------
def to_ui(messages: List[HumanMessage | AIMessage]):
    chat = []
    for m in messages:
        if isinstance(m, HumanMessage):
            chat.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            chat.append({"role": "assistant", "content": m.content})
    return chat

def agent_process(user_input: str, thread_id: str):
    """
    - 매 호출마다 같은 thread_id를 사용하므로
      interrupt()로 멈춘 지점에서 자동으로 이어서 진행됩니다.
    """
    config = {"configurable": {"thread_id": thread_id}}

    # 새 사용자 메시지를 추가하며 실행/재개
    events = graph.stream({"messages": [HumanMessage(content=user_input)]},
                          config,
                          stream_mode="values")

    for _ in events:
        snap = graph.get_state(config)
        yield to_ui(snap.values["messages"])

def create_chatbot_response(message, history, thread_id_state):
    thread_id = thread_id_state.value
    for chunk in agent_process(message, thread_id):
        yield chunk

def on_load():
    return gr.State(str(uuid.uuid4()))

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# LangGraph 뉴스 검색 에이전트 • HITL(일시정지→승인→재개) 데모")

    thread_id_state = gr.State()
    chatbot = gr.Chatbot([], elem_id="chatbot", height=360, type="messages",
                         placeholder="예: '오늘의 주요 경제 뉴스'")
    chat_input = gr.Textbox(show_label=False,
                            placeholder="메시지를 입력하고 Enter",
                            container=False)

    chat_input.submit(create_chatbot_response, [chat_input, chatbot, thread_id_state], [chatbot])
    demo.load(on_load, inputs=[], outputs=[thread_id_state])

demo.launch(debug=True)
# demo.close()  # 노트북/서버 환경에 맞게 필요 시 사용
