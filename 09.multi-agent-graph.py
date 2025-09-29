#!/usr/bin/env python
# coding: utf-8

# # 명탐정 코난 매니아 판별기 (LangGraph)

# ## OpenAI LLM 준비 및 퀴즈 파일 지정
# * 환경 변수(`.env` 파일)에서 API Key 로딩
# * 개발 환경에서는 `gpt-4o-mini` 또는 `gpt-3.5-turbo`
# * 핵심 실습 환경이라 `gpt-4o` 사용

# In[ ]:


import re
import json
import random
import sqlite3
from datetime import datetime
from typing import Literal, TypedDict, Optional, Annotated

import gradio as gr
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from langchain_teddynote.graphs import visualize_graph

# 경로 및 상수
QUIZ_FILE = "data/quizzes.json"
APPLICANT_FILE = "data/applicants.json"
DB_FILE = "data/quiz_results.db"

QUIZ_COUNT = 3  # 퀴즈 문항
QUIZ_COMMANDS = ["퀴즈", "퀴즈 시작"]


load_dotenv()
llm = ChatOpenAI(model="gpt-4o", temperature=0.5)


# ## DB 초기화 및 데이터 로딩

# In[ ]:


# DB 초기화
def ensure_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS quiz_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            taken_at TEXT NOT NULL,
            student_class TEXT,
            student_name TEXT,
            student_id TEXT,
            student_phone TEXT,
            total_score INTEGER,
            total_count INTEGER,
            details_json TEXT
        )
        """
    )
    conn.commit()
    conn.close()

# 퀴즈 로딩 함수
def load_quizzes() -> list[dict]:
    with open(QUIZ_FILE, "r", encoding="utf-8") as f:
        all_q = json.load(f)
    return random.sample(all_q, QUIZ_COUNT)

# 지원자 로딩 함수
def load_applicants() -> list[dict]:
    with open(APPLICANT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ## 기본 데이터 정의
# 
# ### 데이터 모델 정의

# In[ ]:


# 역할 기반 접근 제어 모델 정의
class RoleRoute(BaseModel):
    """역할 기반 접근 제어를 위한 경로 모델입니다."""
    role: Literal["student", "professor", "unknown"]

# 지원자 정보 모델 정의
class ApplicantInfo(BaseModel):
    """지원자 정보를 담는 클래스입니다."""
    student_class: str = Field(description="지원자의 학급")
    student_name: str = Field(description="지원자의 이름")
    student_id: str = Field(description="지원자의 학번")
    student_phone: str = Field(description="지원자의 전화번호")

# 채점 결과 모델 정의
class GradingResult(BaseModel):
    """단일 문제에 대한 채점 결과를 상세히 담는 클래스입니다."""
    question_id: int = Field(description="문제의 고유 ID")
    question: str = Field(description="채점 대상 문제")
    correct_answer: str = Field(description="문제의 정답")
    user_answer: str = Field(description="사용자가 제출한 답변")
    is_correct: bool = Field(description="정답 여부")
    explanation: str = Field(description="정답에 대한 친절한 해설")

# 퀴즈 채점 결과 모델 정의
class FinalReport(BaseModel):
    """퀴즈의 모든 채점 결과와 최종 점수를 종합한 최종 보고서 클래스입니다."""
    results: list[GradingResult] = Field(description="각 문제별 채점 결과 리스트")
    total_score: str = Field(description="'총점: X/Y' 형식의 최종 점수 요약")

# 보고서 요청 모델 정의
class ReportRequest(BaseModel):
    """최종 보고서 생성을 위한 요청 모델입니다."""
    taken_date: Optional[str] = Field(None, description="YYYY-MM-DD 또는 YYYY.MM.DD")
    student_class: Optional[str] = Field(None, description="반 (예: '2반')")
    report_type: Literal["오답", "성적", "전체"] = "전체"

# LLM 출력 형식 지정
llm_with_role = llm.with_structured_output(RoleRoute)
llm_with_applicant = llm.with_structured_output(ApplicantInfo)
llm_with_report = llm.with_structured_output(FinalReport)


# ### 리듀서 함수 정의

# In[ ]:


# add_messages를 사용하지 않고 단순히 리스트를 합치는 함수
def reduce_list(left: list, right: list) -> list:
    """두 리스트를 합칩니다."""

    return left + right


# ### 상태 정의
# 그래프(workflow)가 관리할 상태 정보 클래스 `AppState(TypedDict)`

# In[ ]:


# 애플리케이션 상태 모델 정의
# 모든 필드를 선택적(total=False)으로 관리 => 상태 관리에 적합한 방식
class AppState(TypedDict, total=False):
    """
    애플리케이션의 전체 상태를 관리하는 중앙 저장소.
    Annotated를 사용하여 각 필드에 대한 설명을 타입 힌트에 포함합니다.
    """

    # --- 공통 및 초기 필드 ---
    user_input: Annotated[str, "사용자의 현재 입력값"]
    chat_history: Annotated[list[tuple[str, str]], "UI용 대화 기록 리스트", reduce_list]
    role: Annotated[Literal["student", "professor", "unknown"], "현재 사용자의 역할"]

    # --- 응시자(student) 흐름 관련 필드 ---
    applicant: Annotated[ApplicantInfo, "응시자 정보"]
    questions: Annotated[list[dict], "생성된 퀴즈 질문 목록"]
    quiz_index: Annotated[int, "현재 진행 중인 퀴즈의 인덱스"]
    user_answers: Annotated[list[str], "사용자가 제출한 답변 목록", reduce_list]
    grading_prompt: Annotated[str, "채점을 위해 LLM에 전달할 프롬프트"]
    final_report: Annotated[FinalReport, "최종 채점 결과 보고서"]

    # --- 교수(professor) 리포트 흐름 관련 필드 ---
    report_request: Annotated[ReportRequest, "교수가 요청한 리포트 상세 정보"]


# ## Agent 노드 함수 구현
# 
# **일반적인 함수** 이름은 **동사** 로 시작하고, **노드 함수** 인 경우 **역할(행위자)** 을 명시해서 작명 권고
# 
# ### 1.1. LLM에 의한 역할 분류 함수

# In[ ]:


def classify_role(text: str) -> Literal["student", "professor", "unknown"]:
    """ 사용자 입력을 분석하여 역할을 분류하는 함수입니다."""

    system_message = """  
    당신은 사용자 유형을 분류하는 매우 정확한 라우터입니다. 사용자의 입력을 보고 'student', 'professor', 'unknown' 중 하나로 분류해주세요.

    ## 분류 기준:
    1. 'student': 반, 이름, 학번 등 개인정보를 포함하여 퀴즈 응시를 시도하는 경우.
    2. 'professor': 날짜, 반, '리포트' 또는 '성적'과 같은 키워드를 포함하여 결과를 조회하려는 경우.
    3. 'unknown': 위 두 경우에 해당하지 않는 모든 애매한 경우.

    ## 예시:
    - 입력: "1반 홍길동 S25B001 010-1111-2222", 분류: 'student'
    - 입력: "2025-07-07 2반 성적 순위 리포트 좀 보여줘", 분류: 'professor'
    - 입력: "안녕하세요", 분류: 'unknown'
    - 입력: "퀴즈를 풀고 싶어요.", 분류: 'unknown' (퀴즈 응시를 원하지만, 식별 정보가 없으므로 'unknown' 처리 후 안내)

    ## 출력 형식:
        JSON {"role": "student|professor|unknown"} 한 값만 주세요.
    """

    prompt = ChatPromptTemplate.from_messages([
        ( "system", system_message.strip()),
        ("human", "{input_text}")
    ])

    try:
        response = (prompt | llm_with_role).invoke({"input_text": text})
        return response.role
    except Exception:
        return "unknown"


# ### 1.2. 역할에 따른 엔트리 판단 노드
# 
# ※ 엔트리 노드는 그래프 빌드 직전에 선언하여 선행 노드 참조할 수 있도록 구성

# In[ ]:


def entry_router(state: AppState) -> str:
    """역할 분류 및 진입점 라우터 노드 함수입니다."""

    user_input = state.get("user_input")
    if user_input:
        role = classify_role(user_input)
    else:
        role = "unknown"

    if role == "student":
        return "student_entry"
    elif role == "professor":
        return "professor_entry"
    else:
        return "unknown_entry"


# ### 1.3. 엔트리 판단을 위한 도움말 제공 노드

# In[ ]:


def entry_helper(state: AppState) -> AppState:
    """알 수 없는 역할에 대한 도움말 메시지를 출력하는 노드 함수입니다."""

    help_text = """
        학생은 '1반 김영희 S25B002 010-0000-0000' 처럼 본인 정보를 입력하세요.\n
        교수는 '2025-07-07 2반 리포트 출력'처럼 날짜와 반을 포함해 입력하세요.
    """
    return {
        "chat_history": [("assistant", help_text)]
    }


# ### 2.1. LLM에 의한 응시자 정보 추출 : 입력된 응시자 정보를 JSON 형식으로 전달

# In[ ]:


def parse_applicant_info(text: str) -> ApplicantInfo | None:
    """사용자 입력에서 지원자 정보를 추출하는 함수입니다."""

    system_message = """  
    아래 문장에서 반(student_class), 이름(student_name), 학번(student_id), 전화번호(student_phone)을 추출하세요.
    - 반: 숫자와 '반'이 포함된 문자열 (예: '1반', '2반') 
    - 이름: 한글로 된 이름
    - 학번: 'S'로 시작하는 영문자와 숫자의 조합    
    - 전화번호: 하이픈(-)이 포함될 수 있는 8개 이상의 숫자 형식

    ## 예시:
    - 입력: "1반 홍길동 S25B001 010-1111-2222"
    - 출력: {"student_class": "1반", "student_name": "홍길동", "student_id": "S25B001", "student_phone": "010-1111-2222"}
    """

    prompt = ChatPromptTemplate.from_messages([
        ( "system", system_message.strip()),
        ("human", "{input_text}")
    ])

    try:
        response = (prompt | llm.with_applicant).invoke({"input_text": text})
        if not response.student_name or not response.student_id:
            return None
        return response
    except Exception:
        return None


# ### 2.2. 응시자 정보 검증 노드

# In[ ]:


def applicant_validator(state: AppState) -> AppState:
    """추출한 응시자 정보로 등록된 사용자 여부 및 퀴즈 응시 여부를 확인하는 노드 함수입니다."""

    user_input = state.get("user_input", "")
    applicant = parse_applicant_info(user_input)
    if not applicant:
        return {
            "chat_history": [(
                "assistant",
                "응시자 정보를 인식하지 못했습니다. 예) 1반 김영희 S25B002 010-0000-0000"
            )]
        }

    # 등록된 응시자 확인
    try:
        roster = load_applicants()
    except Exception:
        roster = []

    exists = next((r for r in roster if r.get("student_id") == applicant.student_id), None)
    if not exists:
        return {
            "chat_history": [(
                "assistant",
                f"등록된 응시자를 찾지 못했습니다: {applicant.student_id}"
            )]
        }

    # 이미 응시했는지 확인
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT taken_at,total_score FROM quiz_results WHERE student_id=? ORDER BY id DESC LIMIT 1", (applicant.student_id))
    row = cur.fetchone()
    conn.close()
    if row:
        taken_at, total_score = row
        return {
            "chat_history": [(
                "assistant",
                f"이미 응시 기록이 있습니다. 응시일자: {taken_at}, 점수: {total_score}"
            )]
        }

    # 응시자 검증 통과
    return  {
        "applicant": applicant,
        "chat_history": [(
            "assistant",
            f"{applicant.student_class} {applicant.student_name}님, 퀴즈를 시작하려면 '퀴즈 시작'이라고 입력하세요."
        )],
    }


# ### 2.3. 응시자 퀴즈 출제 노드

# In[ ]:


def quiz_setter(state: AppState) -> AppState:
    """퀴즈 문항을 설정하는 노드 함수입니다."""

    questions = load_quizzes()
    if not questions:
        return {
            "chat_history": [(
                "assistant",
                "퀴즈를 불러오는 데 실패했거나 풀 수 있는 문제가 없습니다."
            )],
            "questions": [],
        }

    return {
        "questions": questions,
        "quiz_index": 0,
        "user_answers": [],
        "final_report": None,
        "chat_history": [(
            "assistant",
            f"퀴즈를 시작합니다. 총 {len(questions)}문항입니다."
        )],
    }


# ### 2.4. 퀴즈 진행 여부 판단 조건부 노드

# In[ ]:


def continue_quiz_condition(state: AppState) -> str:
    """퀴즈 진행 여부를 판단하는 조건부 노드 함수입니다."""

    questions = state.get("questions", [])
    quiz_index = state.get("quiz_index", 0)
    if not questions:
        return "quiz_setter"
    if quiz_index < len(questions):
        return "quiz_popper"
    else:
        return "quiz_grader"


# ### 2.5. 개별 퀴즈 출력 노드

# In[ ]:


def quiz_popper(state: AppState) -> AppState:
    """현재 quiz_index에 맞는 문제를 포맷하여 chat_history에 추가합니다."""

    quiz_index = state["quiz_index"]
    quiz = state["questions"][quiz_index]

    text = f"문제 {quiz_index + 1}: {quiz['question']}"
    if quiz["type"] == "multiple_choice":
        choices = [f"{i + 1}. {c}" for i, c in enumerate(quiz["choices"])]
        text += "\n" + "\n".join(choices)

    return {
        "chat_history": [("assistant", text)],
    }


# ### 개별 퀴즈 답변 저장 노드

# In[ ]:


def answer_collector(state: AppState) -> AppState:
    """사용자 답변을 처리하고 저장한 뒤, 다음 문제로 넘어갑니다."""

    quiz_index = state["quiz_index"]
    quiz = state["questions"][quiz_index]
    user_input = state["user_input"].strip()

    # 빈 입력일 경우 안내 메시지만 추가하고, 인덱스는 그대로 유지
    if not user_input:
        return {
            "chat_history": [("assistant", "답변을 입력해 주세요.")],
        }

    processed_answer = user_input
    if quiz["type"] == "multiple_choice":
        try:
            sel = int(user_input) - 1
            if 0 <= sel < len(quiz["choices"]):
                processed_answer = quiz["choices"][sel]
        except (ValueError, IndexError):
            pass

    return {
        "user_answers": [processed_answer],
        "quiz_index": quiz_index + 1
    }


# ### 2.6. 퀴즈 채점 요청 노드 

# In[ ]:


def grading_prompter(state: AppState) -> AppState:
    """채점을 위해 LLM에 전달할 프롬프트를 생성합니다."""

    questions = state["questions"]
    user_answers = state["user_answers"]

    prompt_buff = [
        "지금부터 아래의 문제와 정답, 그리고 사용자의 답변을 보고 채점을 시작해주세요."
    ]
    for i, (q, a) in enumerate(zip(questions, user_answers)):
        prompt_buff.append(f"\n--- 문제 {i + 1} ---")
        prompt_buff.append(f"문제: {q['question']}")
        if q["type"] == "multiple_choice":
            prompt_buff.append(f"선택지: {', '.join(q['choices'])}")
        prompt_buff.append(f"정답: {q['answer']}")
        prompt_buff.append(f"사용자 답변: {a}")

    return {
        "chat_history": [("assistant", "채점을 진행합니다...")],
        "grading_prompt": "\n".join(prompt_buff),
    }


# ### LLM 채점 및 파싱 노드
# * 상태(state) 갱신의 목적 : 
#     - 채점 대상 데이터를 `state["grading_prompt"]`를 LLM에 전달
#     - LLM의 채점 결과를 FinalReport 클래스 JSON 데이터로 수신하여 `state["final_report"]`에 저장
#     
# - `chain = prompt | llm_with_final_report` : 프롬프트와 FinalReport를 출력으로 하는 LLM을 LCEL 체인으로 연결

# In[ ]:


def grade_reporter(state: AppState) -> AppState:
    """LLM을 호출하여 채점하고 결과를 파싱합니다."""

    system_message = """
    당신은 '명탐정 코난' 퀴즈의 전문 채점관입니다. 주어진 문제, 정답, 사용자 답변을 바탕으로 채점해주세요. 
    각 문제에 대해 정답 여부를 판단하고 친절한 해설을 덧붙여주세요. 
    모든 채점이 끝나면, 마지막에는 '총점: X/Y' 형식으로 최종 점수를 반드시 요약해서 보여줘야 합니다. 
    반드시 지정된 JSON 형식으로만 답변해야 합니다."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("human", "{grading_data}"),
        ]
    )

    try:
        # 체인 생성
        chain = prompt | llm_with_report
        report = chain.invoke({"grading_data": state["grading_prompt"]})
        return {
            "final_report": report
        }
    except Exception as e:
        print(f"채점 중 오류 발생: {e}")
        error_report = FinalReport(results=[], total_score="채점 오류가 발생했습니다.")
        return {
            "final_report": error_report
        }


# ### 최종 리포트(FinalReport)를 사용자에게 보여줄 문자열로 변환 노드
# 상태(state) 갱신의 목적 : LLM이 제공한 최종 채점 결과를 문제별로 정오, 정답, 제출 답변, 해설을 보기 좋게 포매팅
# - 마지막에 총점도 포함
# - 안내 메시지(재시작 방법 등)도 추가
# - 결과를 `assistant` 메시지로 `chat_history`에 append
# 
# 워크플로우 마지막 단계로 구조화된 채점 결과 객체(`final_report_obj`)를 사람이 읽을 수 있는 문자열로 변환하여 `chat_history`에 추가함으로써 사용자에게 "마지막 답변"을 제공

# In[ ]:


def report_formatter(state: AppState) -> AppState:
    """파싱된 최종 리포트를 사용자에게 보여줄 문자열로 변환합니다."""

    final_report = state["final_report"]
    report_buff = ["채점이 완료되었습니다! 🎉\n"]

    if final_report and final_report.results:
        for i, res in enumerate(final_report.results):
            is_correct_text = "✅ 정답" if res.is_correct else "❌ 오답"
            report_buff.append(f"--- 문제 {i + 1} ---")
            report_buff.append(f"문제: {res.question}")
            report_buff.append(f"정답: {res.correct_answer}")
            report_buff.append(f"제출한 답변: {res.user_answer}")
            report_buff.append(f"결과: {is_correct_text}")
            report_buff.append(f"해설: {res.explanation}\n")
        report_buff.append(f"**{final_report.total_score}**")
    else:
        report_buff.append("채점 결과를 생성하는 데 실패했습니다.")

    report_buff.append("\n퀴즈를 다시 시작하려면 '퀴즈 시작'이라고 입력해주세요.")    
    return {
        "chat_history": [("assistant", "\n".join(report_buff))],
    }


# ### 리포트 결과를 DB에 저장

# In[ ]:


def grade_report_saver(state: AppState) -> AppState:
    """채점 결과를 DB에 저장하는 노드 함수입니다."""

    applicant = state.get("applicant")
    final_report = state.get("final_report")
    if applicant and len(final_report.results) > 0:
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        details = [r.model_dump() for r in final_report.results]
        cur.execute(
            """
            INSERT INTO quiz_results (taken_at,student_class,student_name,student_id,student_phone,total_score,total_count,details_json)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                applicant.student_class,
                applicant.student_name,
                applicant.student_id,
                applicant.student_phone,
                final_report.total_score,
                final_report.total_count,
                json.dumps(details, ensure_ascii=False),
            ),
        )
        conn.commit()
        conn.close()
        return {
            "chat_history": [("assistant", "채점 결과가 성공적으로 저장되었습니다.")]
        }    
    else:
        return {
            "chat_history": [("assistant", "채점 결과를 저장하는 데 실패했습니다.")]
        }


# ### 리포트 요청 유형 분석

# In[ ]:


def report_request_parser(state: AppState) -> AppState:
    """교수의 리포트 요청을 파싱하는 노드 함수입니다."""

    user_input = state.get("user_input", "")

    date_match = re.search(r"(\d{4}[-/]\d{2}[-/]\d{2})", user_input)
    taken_date = date_match.group(1) if date_match else ""

    class_match = re.search(r"(\d+반)", user_input)
    student_class = class_match.group(1) if class_match else ""

    if "오답" in user_input:
        report_type = "오답"
    elif "성적" in user_input:
        report_type = "성적"
    else:
        report_type = "전체"

    report_request = ReportRequest(taken_date=taken_date, student_class=student_class, report_type=report_type)
    return {
        "report_request": report_request
    }


# ### 리포트 생성

# In[ ]:


def fetch_quiz_results(report_request) -> list:
    """데이터베이스에서 조건에 맞는 퀴즈 결과 데이터를 조회합니다."""

    taken_date = (report_request.taken_date or "").replace("/", "-")
    student_class = report_request.student_class or ""

    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    sql = "SELECT student_name, student_id, student_class, total_score, total_count, details_json, taken_at FROM quiz_results WHERE 1=1"
    params = []

    if taken_date:
        sql += " AND taken_at LIKE ?"
        params.append(f"{taken_date}%")
    if student_class:
        sql += " AND student_class = ?"
        params.append(student_class)

    cur.execute(sql + " ORDER BY total_score DESC, taken_at ASC", params)
    rows = cur.fetchall()
    conn.close()
    return rows


# In[ ]:


def create_rank_table(rows: list) -> str:
    """퀴즈 결과 데이터를 바탕으로 성적 순위 마크다운 테이블을 생성합니다."""

    rank_table_parts = ["### 성적 순위 (높은 점수 우선)", "이름 | 학번 | 반 | 점수 | 일시", "---|---|---|---|---"]
    for s_name, s_id, s_class, t_score, t_count, _, taken_at in rows:
        rank_table_parts.append(f"{s_name} | {s_id} | {s_class} | {t_score}/{t_count} | {taken_at}")
    return "\n".join(rank_table_parts)


# In[ ]:


def create_wrong_answer_table(rows: list) -> str:
    """퀴즈 결과 데이터를 바탕으로 오답률 상위 마크다운 테이블을 생성합니다."""

    # 1. 문항별 오답 집계
    # agg = { "question_id": [오답 수, 총 응시 수] }
    agg: dict[str, list[int]] = {}
    for *_, details_json, _ in rows:
        try:
            details = json.loads(details_json)
            for d in details:
                qid = f"{d.get('question_id', '?')}.{d.get('question', '')[:16]}"
                is_correct = d.get("is_correct", False)

                if qid not in agg:
                    agg[qid] = [0, 0] # [incorrect_count, total_count]

                agg[qid][1] += 1 # 총 응시 수 증가
                if not is_correct:
                    agg[qid][0] += 1 # 오답 수 증가
        except (json.JSONDecodeError, TypeError):
            continue

    # 2. 오답률 계산 및 정렬
    items = []
    for qid, (wrong, total) in agg.items():
        rate = (wrong / total * 100) if total else 0.0
        items.append({"qid": qid, "wrong": wrong, "total": total, "rate": rate})

    # 오답률(rate) 기준으로 내림차순 정렬
    items.sort(key=lambda x: x["rate"], reverse=True)

    # 3. 마크다운 테이블 생성
    wrong_table_parts = ["\n### 오답률 상위 문항", "문항 | 오답수/응시수 | 오답률(%)", "---|---|---"]
    # 상위 20개만 표시
    for item in items[:20]:
        wrong_table_parts.append(f"{item['qid']} | {item['wrong']}/{item['total']} | {item['rate']:.1f}")

    return "\n".join(wrong_table_parts)


# In[ ]:


def report_generater(state: AppState) -> AppState:
    """요청된 조건에 맞는 리포트를 생성하는 노드 함수입니다."""

    report_request = state.get("report_request")
    if not report_request:
        return {"chat_history": [("assistant", "리포트 요청을 파싱하지 못했습니다.")]}

    # 1. 데이터베이스에서 결과 조회
    quiz_results = fetch_quiz_results(report_request)
    if not quiz_results:
        return {"chat_history": [("assistant", "해당 조건의 응시 기록이 없습니다.")]}

    # 2. 요청 타입에 따라 리포트 생성
    report_outputs = []
    report_type = report_request.report_type

    if report_type in ("성적", "전체"):
        rank_table_md = create_rank_table(quiz_results)
        report_outputs.append(rank_table_md)

    if report_type in ("오답", "전체"):
        wrong_table_md = create_wrong_answer_table(quiz_results)
        report_outputs.append(wrong_table_md)

    # 3. 최종 결과 반환
    final_report = "\n\n".join(report_outputs)
    return {"chat_history": [("assistant", final_report)]}


# ## StateGraph 정의 및 컴파일

# In[ ]:


graph = StateGraph(AppState)

# 노드 추가
graph.add_node("entry_helper", entry_helper)
graph.add_node("applicant_validator", applicant_validator)
graph.add_node("quiz_setter", quiz_setter)
graph.add_node("quiz_popper", quiz_popper)
graph.add_node("answer_collector", answer_collector)
graph.add_node("grading_prompter", grading_prompter)
graph.add_node("grade_reporter", grade_reporter)
graph.add_node("grade_report_saver", grade_report_saver)
graph.add_node("report_formatter", report_formatter)
graph.add_node("report_request_parser", report_request_parser)
graph.add_node("report_generater", report_generater)

# 조건부 진입점 설정
graph.set_conditional_entry_point(
    entry_router,
    {
        "student_entry": "applicant_validator",
        "professor_entry": "report_request_parser",
        "unknown_entry": "entry_helper",
    },
)

# 엣지 연결
graph.add_edge("quiz_setter", "quiz_popper")
graph.add_edge("quiz_popper", END)
graph.add_edge("entry_helper", END)

graph.add_conditional_edges(
    "answer_collector",
    continue_quiz_condition,
    {"continue_quiz": "quiz_popper", "grade_quiz": "grading_prompter"},
)
graph.add_edge("grading_prompter", "grade_reporter")
graph.add_edge("grade_reporter", "grade_report_saver")
graph.add_edge("grade_report_saver", "report_formatter")
graph.add_edge("report_request_parser", "report_generater")
graph.add_edge("report_formatter", END)
graph.add_edge("report_generater", END)

# 그래프 컴파일 : 퀴즈 애플리케이션 워크플로우 완성
quiz_app = graph.compile()


# ## 그래프 시각화

# In[ ]:


visualize_graph(quiz_app)


# ## UI 인터페이스 함수
# `init_state()` :
# - 앱을 초기화할 때 사용할 빈 상태(딕셔너리) 반환
# - `quiz_state` 아래에 문제/대화 이력만 먼저 준비
# - 다른 필드는 워크플로우 도중 필요에 따라 추가됨(예: quiz_index, user_answers, final_report 등)
# 
# `chat_fn(user_input, state)` : Gradio 챗봇 UI에서 실제로 호출되는 메인 처리 함수
# 

# In[ ]:


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


def chat_fn(user_input, state):
    app_state = state["app_state"]

    app_state["chat_history"].append(("user", user_input))
    app_state["user_input"] = user_input

    new_state = quiz_app.invoke(app_state)
    state["app_state"] = new_state

    chat_display = [
        {"role": role, "content": content}
        for role, content in new_state.get("chat_history", [])
    ]

    return chat_display, state


# ## Gradio UI 정의
# Gradio의 Blocks UI 구성 방식을 사용해 명탐정 코난 퀴즈 챗봇 인터페이스를 구성

# In[ ]:


# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    ### 🧩 멀티 에이전트 퀴즈/리포트 (LangGraph)
    - 학생 예: `1반 김영희 S25B002 010-0000-0000` → 확인 후 `퀴즈 시작`
    - 교수 예: `2025-07-07 2반 리포트 출력` / `오답 리포트` / `성적 리포트`
    """)

    chatbot = gr.Chatbot(
        label="명탐정 코난 퀴즈 챗봇",
        height=400,
        avatar_images=("data/avatar_user.png", "data/avatar_conan.png"),
        type="messages",
    )

    txt = gr.Textbox(placeholder="메시지를 입력해보세요!", show_label=False)
    state = gr.State(init_state())

    txt.submit(chat_fn, inputs=[txt, state], outputs=[chatbot, state])
    txt.submit(lambda: "", None, txt)

    demo.launch()


# In[ ]:


demo.close()


# -----
# ** End of Documents **
