import uuid
from typing import Annotated, Optional
from typing_extensions import TypedDict

import gradio as gr
from dotenv import load_dotenv

from langchain.schema import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_teddynote.tools import GoogleNews

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import InMemorySaver

# =============================================================================
# 1) 준비
# =============================================================================
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
    """Request assistance from a human (HITL)."""
    print(f"[Tool] human_assistance 호출: {query}")
    human_response = interrupt({"query": query})
    print(f"[Tool] human_assistance 응답 수신: {human_response}")
    return human_response["data"]

tools = [news_search, human_assistance]
llm_with_tools = llm.bind_tools(tools)

# =============================================================================
# 2) LangGraph 정의
# =============================================================================
class State(TypedDict):
    messages: Annotated[list, "대화 메시지 목록", add_messages]
    interrupted: Annotated[bool, "HITL로 인해 중단되었는지 여부"] = False

def chatbot(state: State):
    """LLM 호출 노드 (툴 콜 포함)."""
    messages = state.get("messages", [])
    if not messages:
        return {"messages": []}
    response = llm_with_tools.invoke(messages)
    if hasattr(response, "tool_calls"):
        assert len(response.tool_calls) <= 1
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
    """반환(dict)에서 마지막 AI 메시지의 텍스트를 추출."""
    msgs = state_output.get("messages", [])
    last_ai_content = None
    for m in reversed(msgs):
        print(f"[Debug] 메시지: {m}")
        if isinstance(m, AIMessage) or getattr(m, "type", "") == "ai":
            last_ai_content = m.content
            break
    return last_ai_content or ""

SYSTEM_PROMPT = SystemMessage(
    content=(
        "당신은 사용자의 요청에 따라 최신 뉴스를 검색하고, "
        "사람에게 '전체 목록 요약', '특정 기사 요약' 등의 추가 작업을 요청(human_assistance)합니다."
    )
)

def chat_fn(user_text: str, thread_id: Optional[str], history: Optional[list[dict]]):
    """한 번 실행: 사용자 메시지를 받아 graph를 한 번 실행."""
    thread_id = ensure_thread_id(thread_id)
    history = history or []
    history.append({"role": "user", "content": user_text})

    inputs = {"messages": [SYSTEM_PROMPT, HumanMessage(content=user_text)]}
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # 정상 경로
        out = graph.invoke(inputs, config=config)
        ai_text = last_ai_text_from_state_output(out) or "(응답이 비어 있습니다)"
        history.append({"role": "assistant", "content": ai_text})
        return history, thread_id, None, gr.update(visible=False), gr.update(visible=True)

    except Exception as e:
        print(f"[HITL] 예외 발생: {type(e).__name__}: {e}")
        hitl_notice = (
            "🔔 **승인 필요(HITL)**\n\n"
            "에이전트가 사람의 도움이 필요합니다.\n"
            "오른쪽 패널에 답변을 입력하고 **재개(Resume)** 버튼을 눌러주세요."
        )
        history.append({"role": "assistant", "content": hitl_notice})
        # 대기 상태 진입(pending_query는 None으로 둠)
        return history, thread_id, None, gr.update(visible=True), gr.update(visible=False)

def resume_with_human_input(human_text: str, thread_id: str, pending_query: Optional[str], history: Optional[list[dict]]):
    """재개(Command.resume): 사람의 입력을 resume payload로 전달."""
    print(f"[HITL] 재개 호출: thread_id={thread_id}, pending_query={pending_query}, human_text={human_text}")

    history = history or []
    history.append({"role": "user", "content": f"[HITL 승인] {human_text}"})

    config = {"configurable": {"thread_id": thread_id}}

    try:
        resume_command = Command(resume={"data": human_text})
        out = graph.invoke(
            None,
            config=config,
            command=resume_command,
        )
        ai_text = last_ai_text_from_state_output(out) or "(재개 후 응답이 없습니다)"
        history.append({"role": "assistant", "content": ai_text})
    except Exception as e:
        history.append({"role": "assistant", "content": f"(재개 중 오류) {type(e).__name__}: {e}"})

    # 재개 완료 → HITL 패널 닫기
    return history, thread_id, None, gr.update(visible=False), gr.update(visible=True)

def clear_all():
    """대화/상태 초기화"""
    return [], None, None, gr.update(visible=False), gr.update(visible=True)

# =============================================================================
# 5) Gradio UI
# =============================================================================
with gr.Blocks(title="LangGraph HITL Demo") as demo:
    gr.Markdown("## LangGraph Human-in-the-Loop")
    gr.Markdown(
        "- **메시지 전송**으로 에이전트와 대화합니다.\n"
        "- 에이전트가 사람 도움(`human_assistance`)을 요청하면 **일시중지**되고, "
        "**승인 입력** 후 **재개(Resume)** 버튼으로 진행됩니다.\n"
        "- **대화 초기화**로 상태를 초기화할 수 있습니다.")

    with gr.Row():
        with gr.Column(scale=3):
            chat = gr.Chatbot(label="대화", height=420, type="messages")
            user_input = gr.Textbox(label="메시지 입력", placeholder="예) 최신 AI 뉴스 검색 해줘")
            with gr.Row():
                send_btn = gr.Button("메시지 전송", variant="primary")
                clear_btn = gr.Button("대화 초기화")
        with gr.Column(scale=2):
            hitl_panel = gr.Group(visible=False)
            with hitl_panel:
                gr.Markdown("### 🔔 승인 필요 (HITL)")
                hitl_input = gr.Textbox(
                    label="사람의 답변(승인/정정/추가정보)",
                    placeholder="에이전트에게 전달할 내용을 입력"
                )
                resume_btn = gr.Button("재개 (Resume)", variant="primary")

            # 상태 변수
            thread_id_state = gr.State(value=None)  # str | None
            pending_query_state = gr.State(value=None)  # str | None

            done_panel = gr.Group(visible=True)
            with done_panel:
                gr.Markdown("✅ **대기 중이 아닙니다.** 일반 대화를 계속해도 됩니다.")

    # 바인딩
    user_input.submit(
        fn=chat_fn,
        inputs=[user_input, thread_id_state, chat],
        outputs=[chat, thread_id_state, pending_query_state, hitl_panel, done_panel],
    )
    user_input.submit(lambda: "", None, user_input)

    send_btn.click(
        fn=chat_fn,
        inputs=[user_input, thread_id_state, chat],
        outputs=[chat, thread_id_state, pending_query_state, hitl_panel, done_panel],
    )
    send_btn.click(lambda: "", None, user_input)

    resume_btn.click(
        fn=resume_with_human_input,
        inputs=[hitl_input, thread_id_state, pending_query_state, chat],
        outputs=[chat, thread_id_state, pending_query_state, hitl_panel, done_panel],
    )

    clear_btn.click(
        fn=clear_all,
        inputs=None,
        outputs=[chat, thread_id_state, pending_query_state, hitl_panel, done_panel],
    )

demo.launch(debug=True)
