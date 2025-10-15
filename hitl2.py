import uuid
from typing import Annotated, Optional
from typing_extensions import TypedDict

import gradio as gr
from dotenv import load_dotenv

# ✅ 최신 경로로 교체
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_teddynote.tools import GoogleNews

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

google_news = GoogleNews()

@tool
def news_search(query: str) -> str:
    """Search news articles by keyword."""
    print(f"[Tool] news_search 호출: {query}")
    results = google_news.search_by_keyword(query, k=5)
    text = "\n".join(
        f"- {r.get('content','(내용없음)')} | {r.get('url','')}" for r in results
    ) or "검색 결과 없음"
    return text

@tool
def human_assistance(query: str) -> str:
    """Request assistance from human."""
    human_response = interrupt({"query": query})
    return human_response["data"]

tools = [news_search, human_assistance]
llm_with_tools = llm.bind_tools(tools)

# ✅ LangGraph의 메시지 누적을 위해 add_messages 사용
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    thread_id: Annotated[Optional[str], "대화 스레드 ID"]
    interrupted: Annotated[bool, "HITL로 인해 중단되었는지 여부"]

def chatbot(state: State):
    """LLM 호출 노드 (툴 콜 포함)."""
    messages = state.get("messages", [])
    if not messages:
        return {"messages": []}
    response = llm_with_tools.invoke(messages)
    # 다중 툴콜을 허용하려면 이 assert 제거 가능
    if hasattr(response, "tool_calls"):
        assert len(response.tool_calls) <= 1, "여러 개의 툴콜은 이 데모에서 지원하지 않습니다."
    return {"messages": [response]}

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)

def ensure_thread_id(thread_id: Optional[str]) -> str:
    """thread_id가 없으면 새로 생성."""
    return thread_id or str(uuid.uuid4())

def last_ai_text_from_state_output(state_output: dict) -> str:
    msgs = state_output.get("messages", [])
    last_ai_content = None
    for m in reversed(msgs):
        if isinstance(m, AIMessage) or getattr(m, "type", "") == "ai":
            last_ai_content = getattr(m, "content", None)
            break
        if isinstance(m, dict) and m.get("type") == "ai":
            last_ai_content = m.get("content")
            break
    return last_ai_content or ""

def play_chat(user_text: str, state: dict):
    state["thread_id"] = ensure_thread_id(state.get("thread_id"))
    messages = state.get("messages", [])
    chat_history = []

    if not messages:
        persona = (
            "당신은 사용자의 요청에 따라 최신 뉴스를 검색하고, "
            "사람에게 '전체 목록 요약', '특정 기사 요약' 등의 추가 작업을 요청(human_assistance)합니다."
        )
        messages.append(SystemMessage(content=persona))
    else:
        for m in messages:
            role = "user" if isinstance(m, HumanMessage) else "assistant"
            chat_history.append({"role": role, "content": getattr(m, "content", "")})

    # ✅ 사용자 입력만 추가
    messages.append(HumanMessage(content=user_text))
    chat_history.append({"role": "user", "content": user_text})

    config = {"configurable": {"thread_id": state["thread_id"]}}
    out = graph.invoke({"messages": messages}, config=config)
    state["messages"] = out.get("messages", messages)

    ai_text = last_ai_text_from_state_output(out)

    # ✅ 중단 여부(툴콜 존재)로 패널 토글
    last_msg = state["messages"][-1] if state["messages"] else None
    has_tool_call = bool(getattr(last_msg, "tool_calls", None))

    if ai_text and not has_tool_call:
        chat_history.append({"role": "assistant", "content": ai_text})
        return chat_history, state, gr.update(visible=False), gr.update(visible=True)
    else:
        # 중단되었거나(툴콜 대기) ai_text가 없으면 HITL 패널 오픈
        if ai_text:
            chat_history.append({"role": "assistant", "content": ai_text})
        else:
            chat_history.append({"role": "assistant", "content": "(HITL 응답 대기 중)"})
        return chat_history, state, gr.update(visible=True), gr.update(visible=False)

def resume_chat(user_text: str, state: dict):
    """재개(Command.resume): 사람의 입력을 resume payload로 전달."""
    state["thread_id"] = ensure_thread_id(state.get("thread_id"))

    # ✅ 기존 messages를 건드리지 않는다 (HumanMessage 추가 금지)
    messages = state.get("messages", [])
    chat_history = []
    for m in messages:
        role = "user" if isinstance(m, HumanMessage) else "assistant"
        chat_history.append({"role": role, "content": getattr(m, "content", "")})

    # UX용 로그만 추가(채팅창에만 보임), 실제 messages에는 추가하지 않음
    chat_history.append({"role": "user", "content": f"[HITL 승인] {user_text}"})

    config = {"configurable": {"thread_id": state["thread_id"]}}

    # ✅ input=None, command=resume 로만 호출
    resume_command = Command(resume={"data": user_text})
    out = graph.invoke(
        None,
        config=config,
        command=resume_command,
    )

    # ✅ 재개 후 state 갱신
    state["messages"] = out.get("messages", messages)

    ai_text = last_ai_text_from_state_output(out) or "(재개 후 응답이 없습니다)"
    chat_history.append({"role": "assistant", "content": ai_text})

    # 재개 완료 → HITL 패널 닫기
    return chat_history, state, gr.update(visible=False), gr.update(visible=True)

def init_state():
    thread_id = str(uuid.uuid4())
    return {"messages": [], "interrupted": False, "thread_id": thread_id}

with gr.Blocks(title="LangGraph HITL Demo") as demo:
    gr.Markdown("## LangGraph Human-in-the-Loop")
    gr.Markdown(
        "- **메시지 전송**으로 에이전트와 대화합니다.\n"
        "- 에이전트가 사람 도움(`human_assistance`)을 요청하면 **일시중지**되고, "
        "**승인 입력** 후 **재개(Resume)** 버튼으로 진행됩니다."
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot_ui = gr.Chatbot(label="대화", height=420, type="messages")
            user_input = gr.Textbox(label="메시지 입력", placeholder="예) 최신 AI 뉴스 검색 해줘")
        with gr.Column(scale=2):
            hitl_panel = gr.Group(visible=False)
            with hitl_panel:
                gr.Markdown("### 🔔 승인 필요 (HITL)")
                hitl_input = gr.Textbox(
                    label="사람의 답변(승인/정정/추가정보)",
                    placeholder="에이전트에게 전달할 내용을 입력"
                )
                resume_btn = gr.Button("재개 (Resume)", variant="primary")
            done_panel = gr.Group(visible=True)
            with done_panel:
                gr.Markdown("### ✅ 일반 대화 진행중")

    state = gr.State(init_state())
    user_input.submit(
        fn=play_chat,
        inputs=[user_input, state],
        outputs=[chatbot_ui, state, hitl_panel, done_panel],
    )
    user_input.submit(lambda: "", None, user_input)

    resume_btn.click(
        fn=resume_chat,
        inputs=[hitl_input, state],
        outputs=[chatbot_ui, state, hitl_panel, done_panel],
    )

demo.launch(debug=True)
