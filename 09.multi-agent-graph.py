#!/usr/bin/env python
# coding: utf-8
"""
LangGraph 멀티 에이전트: 문제은행 기반 퀴즈 출제/채점 & 반별 리포트
- Applicant Agent: 응시자 정보 추출/검증
- Quiz Agent: 출제, 응답 수집, LLM 채점, SQLite 저장
- Report Agent: 교수 리포트(오답률, 성적 순위)

기반 코드: 단일 퀴즈 플로우(agent.py)를 멀티 에이전트 구조로 확장
데이터: data/quizzes.json, data/applicants.json, sqlite: data/quiz_results.db
UI: Gradio 챗봇 (단일 입력창에서 학생/교수 명령 모두 처리)
"""

import os
import re
import json
import random
import sqlite3
from datetime import datetime
from typing import List, Literal, TypedDict, Optional

import gradio as gr
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# ----------------------
# 경로/상수
# ----------------------
DATA_DIR = os.path.join("data")
QUIZ_FILE = os.path.join(DATA_DIR, "quizzes.json")
APPLICANTS_FILE = os.path.join(DATA_DIR, "applicants.json")
DB_PATH = os.path.join(DATA_DIR, "quiz_results.db")
QUIZ_COUNT = 10  # 프로그램 설명 기준 10문항
QUIZ_COMMANDS = ["퀴즈", "퀴즈 시작"]

# ----------------------
# 환경/LLM
# ----------------------
load_dotenv()
llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"), temperature=0)

# ----------------------
# 데이터 로딩 & DB 초기화
# ----------------------

def ensure_db():
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    # 저장 스키마: 프로그램 설명의 의도 반영
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS quiz_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            taken_at TEXT NOT NULL,
            class TEXT,
            name TEXT,
            student_id TEXT,
            phone TEXT,
            total_score INTEGER,
            total_count INTEGER,
            details_json TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def load_quizzes() -> List[dict]:
    with open(QUIZ_FILE, "r", encoding="utf-8") as f:
        all_q = json.load(f)
    if len(all_q) < QUIZ_COUNT:
        return random.sample(all_q, len(all_q))
    return random.sample(all_q, QUIZ_COUNT)


def load_applicants() -> List[dict]:
    with open(APPLICANTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------
# Pydantic 모델
# ----------------------

class ApplicantInfo(BaseModel):
    klass: str = Field(alias="class", description="반 (예: '1반')")
    name: str
    student_id: str
    phone: str


class GradingResult(BaseModel):
    question_id: str
    question: str
    correct_answer: str
    user_answer: str
    is_correct: bool
    explanation: str


class FinalReport(BaseModel):
    results: List[GradingResult]
    total_score: int
    total_count: int


class RoleRoute(BaseModel):
    role: Literal["student", "professor", "unknown"]


class ReportRequest(BaseModel):
    date: Optional[str] = Field(None, description="YYYY-MM-DD 또는 YYYY/MM/DD")
    klass: Optional[str] = Field(None, description="반 (예: '2반')")
    report_type: Literal["오답", "성적", "전체"] = "전체"


llm_report = llm
llm_struct_final = llm.with_structured_output(FinalReport)
llm_struct_route = llm.with_structured_output(RoleRoute)

# ----------------------
# 상태 정의
# ----------------------

class AppState(TypedDict):
    user_input: str
    chat_history: List[tuple]
    role: Literal["student", "professor", "unknown"]

    # 응시자 흐름
    applicant: Optional[ApplicantInfo]
    questions: List[dict]
    quiz_index: int
    user_answers: List[str]
    grading_input_str: Optional[str]
    final_report: Optional[FinalReport]

    # 교수 리포트 흐름
    report_req: Optional[ReportRequest]
    report_output_md: Optional[str]


# ----------------------
# 유틸
# ----------------------

ROLE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
당신은 입력 문장을 읽고, 이 사용자가 학생인지 교수인지 분류합니다.
- 학생: 본인 반/이름/학번/전화번호를 주거나, 퀴즈 시작/응시/문제 풀이 관련 요청을 함
- 교수: 특정 날짜/반에 대한 리포트/오답/성적 등 분석/출력 요청을 함
명확하지 않으면 unknown.
JSON {"role": "student|professor|unknown"} 한 값만 주세요.
            """.strip(),
        ),
        ("human", "입력: {text}"),
    ]
)


APPLICANT_PARSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
아래 문장에서 반(class), 이름(name), 학번(student_id), 전화번호(phone)을 추출하세요.
가능한 경우만 채워주세요. 값이 없으면 추정하지 말고 빈 문자열로 두세요.
JSON 키는 class,name,student_id,phone입니다.
            """.strip(),
        ),
        ("human", "문장: {text}"),
    ]
)


def classify_role(text: str) -> Literal["student", "professor", "unknown"]:
    try:
        route = (ROLE_PROMPT | llm_struct_route).invoke({"text": text})
        return route.role
    except Exception:
        # 간단한 휴리스틱 백업
        if "리포트" in text or "오답" in text or "성적" in text:
            return "professor"
        return "student" if re.search(r"S\d+", text) else "unknown"


def parse_applicant(text: str) -> ApplicantInfo | None:
    try:
        out = (APPLICANT_PARSE_PROMPT | llm.with_structured_output(ApplicantInfo)).invoke({"text": text})
        # 빈 값 필터링
        if not out.name or not out.student_id:
            return None
        return out
    except Exception:
        return None


# ----------------------
# Applicant Agent
# ----------------------

def applicant_validate_and_gate(state: AppState) -> AppState:
    """입력에서 응시자 정보를 추출하고 사전 등록자 및 응시여부 확인"""
    
    text = state["user_input"].strip()

    # "퀴즈 시작" 트리거도 허용
    if text in QUIZ_COMMANDS and not state.get("applicant"):
        state["chat_history"].append(("assistant", "응시자 정보를 먼저 입력해 주세요. 예) 1반 김영희 S25B002 010-0000-0000"))
        return state

    # 정보 추출
    applicant = parse_applicant(text)
    if not applicant:
        state["chat_history"].append(("assistant", "응시자 정보를 인식하지 못했습니다. 예) 1반 김영희 S25B002 010-0000-0000"))
        return state

    # 등록자 확인
    try:
        roster = load_applicants()
    except FileNotFoundError:
        roster = []

    exists = next((r for r in roster if r.get("student_id") == applicant.student_id), None)
    if not exists:
        state["chat_history"].append(("assistant", f"등록된 응시자를 찾지 못했습니다: {applicant.student_id}"))
        return state

    # 이미 응시했는지 확인
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT taken_at,total_score FROM quiz_results WHERE student_id=? ORDER BY id DESC LIMIT 1", (applicant.student_id,))
    row = cur.fetchone()
    conn.close()
    if row:
        taken_at, total_score = row
        state["chat_history"].append(("assistant", f"이미 응시 기록이 있습니다. 응시일자: {taken_at}, 점수: {total_score}"))
        return state

    # 통과
    state["applicant"] = applicant
    state["chat_history"].append(("assistant", f"{applicant.klass} {applicant.name}님, 퀴즈를 시작하려면 '퀴즈 시작'이라고 입력하세요."))
    return state


# ----------------------
# Quiz Agent
# ----------------------

GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
당신은 퀴즈 채점관입니다. 각 문항에 대해 정오 판정과 친절한 해설을 주세요.
반드시 JSON 스키마(FinalReport)에 맞추세요. total_score는 정답 개수, total_count는 전체 문항수입니다.
            """.strip(),
        ),
        ("human", "{grading_data}"),
    ]
)


def quiz_maybe_start(state: AppState) -> AppState:
    text = state["user_input"].strip().lower()
    if text in [c.lower() for c in QUIZ_COMMANDS]:
        if not state.get("applicant"):
            state["chat_history"].append(("assistant", "응시자 확인 후 진행됩니다. 먼저 본인 정보를 입력해 주세요."))
            return state
        qs = load_quizzes()
        state["questions"] = qs
        state["quiz_index"] = 0
        state["user_answers"] = []
        state["final_report"] = None
        state["chat_history"].append(("assistant", "퀴즈를 시작합니다. 총 10문항입니다."))
    return state


def quiz_ask_if_needed(state: AppState) -> AppState:
    if not state.get("questions"):
        return state
    qi = state.get("quiz_index", 0)
    if qi < len(state["questions"]):
        q = state["questions"][qi]
        text = f"문제 {qi+1}: {q['question']}"
        if q["type"] == "multiple_choice":
            choices = "\n".join([f"{i+1}. {c}" for i, c in enumerate(q["choices"])])
            text += "\n" + choices
        state["chat_history"].append(("assistant", text))
    return state


def quiz_collect_answer(state: AppState) -> AppState:
    if not state.get("questions"):
        return state
    qi = state.get("quiz_index", 0)
    if qi >= len(state["questions"]):
        return state

    q = state["questions"][qi]
    ans_raw = state["user_input"].strip()
    if not ans_raw:
        state["chat_history"].append(("assistant", "답변을 입력해 주세요."))
        return state

    processed = ans_raw
    if q["type"] == "multiple_choice":
        try:
            idx = int(ans_raw) - 1
            if 0 <= idx < len(q["choices"]):
                processed = q["choices"][idx]
        except Exception:
            pass
    state["user_answers"].append(processed)
    state["quiz_index"] = qi + 1
    return state


def quiz_should_continue(state: AppState) -> str:
    if not state.get("questions"):
        return "no_quiz"
    return "continue" if state["quiz_index"] < len(state["questions"]) else "grade"


def quiz_prepare_grading(state: AppState) -> AppState:
    parts = ["채점 대상 데이터"]
    for q, a in zip(state["questions"], state["user_answers"]):
        parts.append(f"\n---\nid: {q['id']}\n문제: {q['question']}")
        if q["type"] == "multiple_choice":
            parts.append(f"선택지: {', '.join(q['choices'])}")
        parts.append(f"정답: {q['answer']}")
        parts.append(f"사용자 답변: {a}")
    state["grading_input_str"] = "\n".join(parts)
    state["chat_history"].append(("assistant", "채점을 진행합니다..."))
    return state


def quiz_grade_and_store(state: AppState) -> AppState:
    try:
        chain = GRADE_PROMPT | llm_struct_final
        report: FinalReport = chain.invoke({"grading_data": state["grading_input_str"]})
        state["final_report"] = report
    except Exception as e:
        state["final_report"] = FinalReport(results=[], total_score=0, total_count=len(state.get("questions", [])))
        state["chat_history"].append(("assistant", f"채점 오류: {e}"))
        return state

    # DB 저장
    applicant = state.get("applicant")
    if applicant:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        details = [r.model_dump() for r in state["final_report"].results]
        cur.execute(
            """
            INSERT INTO quiz_results (taken_at,class,name,student_id,phone,total_score,total_count,details_json)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                applicant.klass,
                applicant.name,
                applicant.student_id,
                applicant.phone,
                state["final_report"].total_score,
                state["final_report"].total_count,
                json.dumps(details, ensure_ascii=False),
            ),
        )
        conn.commit()
        conn.close()

    return state


def quiz_format_final(state: AppState) -> AppState:
    rep = state.get("final_report")
    if not rep or not rep.results:
        state["chat_history"].append(("assistant", "채점 결과를 생성하지 못했습니다."))
        return state

    lines = ["채점이 완료되었습니다! 🎉\n"]
    for i, r in enumerate(rep.results, 1):
        ok = "✅ 정답" if r.is_correct else "❌ 오답"
        lines.append(f"--- 문제 {i} ({r.question_id}) ---")
        lines.append(f"문제: {r.question}")
        lines.append(f"정답: {r.correct_answer}")
        lines.append(f"제출: {r.user_answer}")
        lines.append(f"결과: {ok}")
        lines.append(f"해설: {r.explanation}\n")
    lines.append(f"**총점: {rep.total_score}/{rep.total_count}**")
    lines.append("\n리포트가 필요하면 'YYYY-MM-DD X반 리포트 출력'이라고 입력해보세요.")
    state["chat_history"].append(("assistant", "\n".join(lines)))
    return state


# ----------------------
# Report Agent
# ----------------------

REPORT_PARSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
교수의 리포트 요청을 파싱합니다. 다음 JSON 키를 사용하세요: {date, klass, report_type}
- report_type은 [오답, 성적, 전체] 중 하나. 입력에 '오답'이 있으면 오답, '성적'이 있으면 성적, 둘 다 없으면 전체
- date 없으면 빈 문자열
- klass 없으면 빈 문자열
            """.strip(),
        ),
        ("human", "요청: {text}"),
    ]
)


def report_parse_request(state: AppState) -> AppState:
    text = state["user_input"].strip()
    try:
        req = (REPORT_PARSE_PROMPT | llm.with_structured_output(ReportRequest)).invoke({"text": text})
    except Exception:
        # 간단 파싱 대체
        m = re.search(r"(\d{4}[-/]\d{2}[-/]\d{2})", text)
        date = m.group(1) if m else ""
        km = re.search(r"(\d+반)", text)
        klass = km.group(1) if km else ""
        rtype = "오답" if "오답" in text else ("성적" if "성적" in text else "전체")
        req = ReportRequest(date=date, klass=klass, report_type=rtype)
    state["report_req"] = req
    return state


def report_generate(state: AppState) -> AppState:
    req = state.get("report_req")
    if not req:
        state["chat_history"].append(("assistant", "리포트 요청을 파싱하지 못했습니다."))
        return state

    date_prefix = (req.date or "").replace("/", "-")
    klass = req.klass or ""

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    sql = "SELECT name,student_id,class,total_score,total_count,details_json,taken_at FROM quiz_results WHERE 1=1"
    params = []
    if date_prefix:
        sql += " AND taken_at LIKE ?"
        params.append(f"{date_prefix}%")
    if klass:
        sql += " AND class=?"
        params.append(klass)

    cur.execute(sql + " ORDER BY total_score DESC, taken_at ASC", params)
    rows = cur.fetchall()
    conn.close()

    if not rows:
        state["chat_history"].append(("assistant", "해당 조건의 응시 기록이 없습니다."))
        return state

    # 성적 순위 테이블
    rank_lines = ["### 성적 순위 (높은 점수 우선)\n", "이름 | 학번 | 반 | 점수 | 일시", "---|---|---|---|---"]
    # 오답률 집계: {quiz_id: [incorrect_count, total]}
    agg: dict[str, list[int]] = {}

    for name, sid, klass, score, total, details_json, taken_at in rows:
        rank_lines.append(f"{name} | {sid} | {klass} | {score}/{total} | {taken_at}")
        try:
            details = json.loads(details_json)
            for d in details:
                qid = d.get("question_id") or "?"
                is_correct = d.get("is_correct", False)
                if qid not in agg:
                    agg[qid] = [0, 0]
                # total
                agg[qid][1] += 1
                # incorrect
                if not is_correct:
                    agg[qid][0] += 1
        except Exception:
            continue

    # 오답률 정렬
    wrong_table = ["\n### 오답률 상위 문항", "문항ID | 오답수/응시수 | 오답률(%)", "---|---|---"]
    items = []
    for qid, (wrong, tot) in agg.items():
        rate = (wrong / tot * 100) if tot else 0.0
        items.append((qid, wrong, tot, rate))
    items.sort(key=lambda x: x[3], reverse=True)
    for qid, wrong, tot, rate in items[:20]:
        wrong_table.append(f"{qid} | {wrong}/{tot} | {rate:.1f}")

    out_parts = []
    if req.report_type in ("성적", "전체"):
        out_parts.append("\n".join(rank_lines))
    if req.report_type in ("오답", "전체"):
        out_parts.append("\n".join(wrong_table))

    state["report_output_md"] = "\n\n".join(out_parts)
    state["chat_history"].append(("assistant", state["report_output_md"]))
    return state


# ----------------------
# Role Router & 진입점
# ----------------------

def route_entry(state: AppState) -> str:
    role = classify_role(state["user_input"]) if not state.get("role") else state["role"]
    state["role"] = role
    if role == "student":
        # 학생: 응시자 정보가 없으면 ApplicantAgent → 있으면 QuizAgent 단계로 라우팅
        return "student_entry"
    elif role == "professor":
        return "prof_entry"
    else:
        return "unknown_entry"


# ----------------------
# 그래프 구성
# ----------------------

def build_graph():
    g = StateGraph(AppState)

    # 노드 등록
    g.add_node("applicant_gate", applicant_validate_and_gate)
    g.add_node("quiz_maybe_start", quiz_maybe_start)
    g.add_node("quiz_ask", quiz_ask_if_needed)
    g.add_node("quiz_collect", quiz_collect_answer)
    g.add_node("quiz_prepare", quiz_prepare_grading)
    g.add_node("quiz_grade_store", quiz_grade_and_store)
    g.add_node("quiz_format", quiz_format_final)

    g.add_node("report_parse", report_parse_request)
    g.add_node("report_generate", report_generate)

    # 진입점(조건부)
    g.set_conditional_entry_point(
        route_entry,
        {
            "student_entry": "student_switch",
            "prof_entry": "report_parse",
            "unknown_entry": "unknown_help",
        },
    )

    # 학생용 스위치 노드 (익명 람다를 노드로 등록)
    def student_switch(state: AppState) -> AppState:
        # 응시자 확인 안 됨 → 게이트로
        if not state.get("applicant"):
            return applicant_validate_and_gate(state)
        # 응시자 확인됨 → 퀴즈 흐름
        return quiz_maybe_start(state)

    g.add_node("student_switch", student_switch)

    def unknown_help(state: AppState) -> AppState:
        state["chat_history"].append((
            "assistant",
            "학생은 '1반 김영희 S25B002 010-0000-0000' 처럼 본인 정보를 입력하세요.\n"
            "교수는 '2025-07-07 2반 리포트 출력'처럼 날짜와 반을 포함해 입력하세요."
        ))
        return state

    g.add_node("unknown_help", unknown_help)

    # 학생 플로우 엣지
    g.add_edge("student_switch", "quiz_ask")  # 퀴즈 시작 명령이면 내부에서 세팅됨

    # 사용자의 응답을 받을 때는 다음 조건부 엣지 사용
    def quiz_cond(state: AppState) -> str:
        return quiz_should_continue(state)

    g.add_edge("quiz_ask", END)  # UI가 다음 사용자 입력을 받을 때까지 정지
    g.add_node("quiz_cond", lambda s: s)  # dummy

    g.add_edge("quiz_collect", "quiz_cond")
    g.add_conditional_edges(
        "quiz_cond",
        quiz_cond,
        {
            "no_quiz": "quiz_maybe_start",
            "continue": "quiz_ask",
            "grade": "quiz_prepare",
        },
    )
    g.add_edge("quiz_prepare", "quiz_grade_store")
    g.add_edge("quiz_grade_store", "quiz_format")
    g.add_edge("quiz_format", END)

    # 교수 플로우
    g.add_edge("report_parse", "report_generate")
    g.add_edge("report_generate", END)

    return g.compile()


# ----------------------
# UI 함수
# ----------------------

def init_state() -> dict:
    return {
        "app_state": {
            "chat_history": [],
            "role": "unknown",
            "questions": [],
            "quiz_index": 0,
            "user_answers": [],
        }
    }


def chat_fn(user_input: str, state: dict):
    app: AppState = state["app_state"]

    # 리셋 트리거
    if user_input.strip() in ("reset", "리셋"):
        state = init_state()
        return [], state

    app["chat_history"].append(("user", user_input))
    app["user_input"] = user_input

    new_state = APP.invoke(app)
    state["app_state"] = new_state

    chat_display = [
        {"role": r, "content": c} for r, c in new_state.get("chat_history", [])
    ]
    return chat_display, state


# ----------------------
# 실행 진입점
# ----------------------
if __name__ == "__main__":
    ensure_db()
    APP = build_graph()

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        ### 🧩 멀티 에이전트 퀴즈/리포트 (LangGraph)
        - 학생 예: `1반 김영희 S25B002 010-0000-0000` → 확인 후 `퀴즈 시작`
        - 교수 예: `2025-07-07 2반 리포트 출력` / `오답 리포트` / `성적 리포트`
        - 초기화: `reset`
        """)

        chatbot = gr.Chatbot(
            label="대화",
            height=480,
            type="messages",
        )
        txt = gr.Textbox(placeholder="메시지를 입력하세요", show_label=False)
        st = gr.State(init_state())

        txt.submit(chat_fn, inputs=[txt, st], outputs=[chatbot, st])
        txt.submit(lambda: "", None, txt)

        demo.launch()
