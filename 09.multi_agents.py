#!/usr/bin/env python
# coding: utf-8

# # 문제은행 기반 퀴즈 출제 및 채점 에이전트 (Multi Agents )
# 
# ## 시스템 개요
# 
# - 본 시스템은 LangGraph 기반 Agent 3종으로 구성되며, 각 Agent는 역할별로 분리되어 협력한다.
# - 모든 데이터는 data/ 디렉토리 하에 JSON 파일로 관리된다.
# - 시스템은 사용자의 입력을 분석하여 **응시자(학생)**인지 교수인지 식별하며, 이에 따라 분기된 처리 로직을 수행한다.
# 
# ### 파일 데이터 구조
# 
# 1. `quizzes.json` : 문제은행 퀴즈로, 선다형(multiple_choice)인 경우 선택지(choices)를 갖게 된다.
# 
# ```json
# [
#   {
#     "id": "Q001",
#     "type": "short_answer",
#     "question": "코난의 본명은 무엇인가요?",
#     "answer": "쿠도 신이치"
#   },
#   {
#     "id": "Q002",
#     "type": "multiple_choice",
#     "question": "다음 중 코난이 사용하는 음성 변조기 발명자는 누구인가요?",
#     "choices": ["아가사 히로시 박사", "하시바 마사루", "하이바라 아이", "모리 코고로"],
#     "answer": "아가사 히로시 박사"
#   }
# ]
# ```
# 
# 2. `applicants.json` : 응시자 정보
# 
# ```json
# [
# 	{
# 		"class": "1반",
# 		"name": "홍길동",
# 		"student_id": "S25B001",
# 		"phone": "010-1234-5678"
# 	}
# ]
# ```
# 
# 3. `results/{YYYY-MM-DD}/{class}/result.json`: 응시 결과 저장
# 
# ```json
# [
#   {
#     "student_id": "S25B001",
#     "name": "홍길동",
#     "score": 8,
#     "answers": [
#       {"quiz_id": "Q001", "user_answer": "신이치", "is_correct": true},
#       ...
#     ]
#   }
# ]
# ```
#  
# 
# ### 에이전트 정의 및 기능
# 
# 1. Applicant Agent : 
#     * 역할: 응시자 정보 처리 및 퀴즈 시작 요청 승인
#     * 입력: 사용자 입력 문장 (예: “1반 김영희 S25B002 010-0000-0000”)
#     * 기능:
#         - LLM을 통해 문장에서 class, name, student_id, phone 정보를 추출
#         - applicants.json과 대조하여 응시자 존재 여부 및 응시 가능 여부 확인
#         - 이미 응시한 경우 results/{날짜}/{반}/result.json에서 해당 학생 ID 확인하여 오류 메시지 전송
#     * 출력: 퀴즈 Agent로 전달할 응시자 정보 객체 반환 또는 오류 메시지
# 
# 2. Quiz Agent : 
#     * 역할: 퀴즈 출제, 응답 수집, LLM 채점
#     * 입력: 인증된 응시자 정보
#     * 기능:
#         - quizzes.json에서 10개의 랜덤 문제 선택 (quiz_id 포함)
#         - 퀴즈 시작 후 10문제 순차 출제 및 사용자 응답 수집
#         - 응답 완료 시, 다음 정보를 LLM에 전달하여 채점 수행하고 채점 결과 저장: 점수, 정오표, 응시자 정보
#         ```json
#         {
#           "student": { ... },
#           "quiz": [{ "id": "Q001", "question": "...", "answer": "...", "user_answer": "..." }, ...]
#         }
# 				```
#     * 출력: results/{YYYY-MM-DD}/{class}/result.json에 누적 저장
# 
# 3. Report Agent : 
#     * 역할: 교수의 요청에 따라 반별 오답 및 성적 리포트 출력
#     * 입력: 교수 입력 (예: “2025-07-07 2반 리포트 출력”)
#     * 기능:
# 		    - 응시일자 및 반에 따라 저장된 결과 파일 로딩
#         - 리포트 유형에 따라 다음 수행: 오답 리포트(문제별 오답률 집계 후 높은 순서로 정렬), 성적 순위 리포트(점수 기준으로 응시자 정렬)
#         - LLM 또는 파이썬 내장 로직을 활용하여 표 형태 출력
#     * 출력: Markdown 또는 Gradio-friendly 표 형태의 응답 리포트

# ## AI Agent 구현
# 
# ### 1. 초기 설정

# In[40]:


import gradio as gr
import random
import json
import os
import datetime
from dotenv import load_dotenv
from typing import List, TypedDict, Literal, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, END

# .env 파일에서 환경 변수 로드
load_dotenv()

# 파일 경로 및 퀴즈 개수 설정
QUIZ_FILE = "data/quizzes.json"
APPLICANTS_FILE = "data/applicants.json"
RESULTS_DIR = "results"
QUIZ_COUNT = 3  # 요구사항에 따라 10문제로 설정

# LLM 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ### 2. 데이터 모델 정의

# In[41]:


# Pydantic 모델 정의
class UserType(BaseModel):
    """사용자 유형 분류를 위한 Pydantic 모델"""
    user_type: Literal["student", "professor", "unknown"] = Field(description="사용자의 유형. 'student', 'professor', 'unknown' 중 하나")


class ApplicantInfo(BaseModel):
    """응시자 정보 추출을 위한 Pydantic 모델"""
    class_name: str = Field(description="응시자의 반. 예: '1반'")
    name: str = Field(description="응시자의 이름. 예: '홍길동'")
    student_id: str = Field(description="응시자의 학번. 예: 'S25B001'")
    phone: str = Field(description="응시자의 전화번호. 예: '010-1234-5678'")


class ReportRequest(BaseModel):
    """리포트 요청 분석을 위한 Pydantic 모델"""
    date: str = Field(description="조회할 날짜 (YYYY-MM-DD 형식)")
    class_name: str = Field(description="조회할 반 이름 (예: '1반', '2반')")
    report_type: Literal["성적", "오답"] = Field(description="요청한 리포트의 종류")


class GradingResult(BaseModel):
    """개별 문제 채점 결과"""
    quiz_id: str
    question: str
    correct_answer: str
    user_answer: str
    is_correct: bool
    explanation: str


class FinalReport(BaseModel):
    """전체 퀴즈에 대한 최종 리포트"""
    results: List[GradingResult]
    total_score: str


# ### 3. Graph 상태 정의

# In[42]:


# LangGraph의 상태(State) 정의
class SystemState(TypedDict):
    user_type: Optional[Literal["student", "professor", "unknown"]]
    user_input: str
    chat_history: List[tuple]
    
    # 학생 관련 상태
    applicant_info: Optional[ApplicantInfo]
    is_applicant_valid: bool
    validation_message: str
    questions: List[dict]
    user_answers: List[str]
    quiz_index: int
    final_report: Optional[FinalReport]

    # 교수 관련 상태
    report_request: Optional[ReportRequest]
    report_output: Optional[str]


# ### 4. 유틸리티 함수 (데이터 로딩 및 저장)

# In[43]:


# 헬퍼 함수 (데이터 로딩/저장)

def load_json(file_path):
    """JSON 파일을 로드하는 함수"""
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, file_path):
    """JSON 데이터를 저장하는 함수"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_quiz():
    """퀴즈 파일에서 랜덤 문제를 로드하는 함수"""
    all_q = load_json(QUIZ_FILE)
    if not all_q: return []
    # 퀴즈 개수가 QUIZ_COUNT 미만일 경우 가능한 만큼만 샘플링
    count = min(len(all_q), QUIZ_COUNT)
    return random.sample(all_q, count)


# ### 5. Agent 노드 함수 정의
# 
# - `route_user_type` : 사용자 입력 분석

# In[44]:


def route_user_type(state: SystemState) -> Literal["student_flow", "professor_flow", "unknown_flow"]:
    """사용자 입력을 분석하여 학생, 교수, 또는 알 수 없는 요청으로 라우팅"""
    parser = PydanticOutputParser(pydantic_object=UserType)
    
    # LLM이 역할을 더 잘 이해하도록 예시(Few-shot)를 포함한 프롬프트로 수정
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 사용자 유형을 분류하는 매우 정확한 라우터입니다. 사용자의 입력을 보고 'student', 'professor', 'unknown' 중 하나로 분류해주세요.

## 분류 기준:
1. 'student': 반, 이름, 학번 등 개인정보를 포함하여 퀴즈 응시를 시도하는 경우.
2. 'professor': 날짜, 반 이름, '리포트' 또는 '성적'과 같은 키워드를 포함하여 결과를 조회하려는 경우.
3. 'unknown': 위 두 경우에 해당하지 않는 모든 애매한 경우.

## 예시:
- 입력: "1반 김코난 S25B007 010-1111-2222"
  분류: 'student'
- 입력: "2025-07-07 2반 성적 순위 리포트 좀 보여줘"
  분류: 'professor'
- 입력: "안녕하세요"
  분류: 'unknown'
- 입력: "퀴즈를 풀고 싶어요."
  분류: 'unknown' (퀴즈 응시를 원하지만, 식별 정보가 없으므로 'unknown' 처리 후 안내)
"""),
        ("human", "사용자 입력: {user_input}\n\n{format_instructions}")
    ])
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({"user_input": state["user_input"], "format_instructions": parser.get_format_instructions()})
        user_type = result.user_type
    except Exception as e:
        print(f"라우팅 중 오류 발생: {e}")
        user_type = "unknown"

    state["user_type"] = user_type
    # state['chat_history'].append(('system', f"라우팅 결과: {user_type}")) # 디버깅용 로그
    
    if user_type == "student":
        return "student_flow"
    elif user_type == "professor":
        return "professor_flow"
    else:
        return "unknown_flow"


# - Applicant Agent Nodes

# In[45]:


def extract_applicant_info(state: SystemState) -> SystemState:
    """LLM을 사용해 사용자 입력에서 응시자 정보를 추출"""
    parser = PydanticOutputParser(pydantic_object=ApplicantInfo)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 사용자 정보 추출기입니다. 주어진 문장에서 반, 이름, 학번, 전화번호를 정확히 추출하여 JSON 형식으로 반환하세요."),
        ("human", "사용자 정보: {user_input}\n{format_instructions}")
    ])
    chain = prompt | llm | parser
    try:
        info = chain.invoke({"user_input": state["user_input"], "format_instructions": parser.get_format_instructions()})
        state["applicant_info"] = info
    except Exception as e:
        state["validation_message"] = f"정보 추출에 실패했습니다. 입력 형식을 확인해주세요. (오류: {e})"
        state["applicant_info"] = None
    return state

def validate_applicant(state: SystemState) -> SystemState:
    """추출된 정보를 바탕으로 응시자 자격 검증"""
    if not state["applicant_info"]:
        state["is_applicant_valid"] = False
        return state

    info = state["applicant_info"]
    applicants = load_json(APPLICANTS_FILE)
    if not applicants:
        state["is_applicant_valid"] = False
        state["validation_message"] = "오류: 응시자 명단 파일(applicants.json)을 찾을 수 없습니다."
        return state

    is_registered = any(
        str(app["student_id"]) == str(info.student_id) and \
        app["name"] == info.name and \
        str(app["class"]) == str(info.class_name)
        for app in applicants
    )

    if not is_registered:
        state["is_applicant_valid"] = False
        state["validation_message"] = "등록되지 않은 응시자입니다. 반, 이름, 학번을 다시 확인해주세요."
        return state
    
    today = datetime.date.today().strftime("%Y-%m-%d")
    result_path = os.path.join(RESULTS_DIR, today, str(info.class_name), "result.json")
    
    results_today = load_json(result_path)
    if results_today and any(str(res["student_id"]) == str(info.student_id) for res in results_today):
        state["is_applicant_valid"] = False
        state["validation_message"] = f"{info.name}님은 오늘 이미 퀴즈에 응시하셨습니다. 내일 다시 시도해주세요."
        return state

    state["is_applicant_valid"] = True
    state["validation_message"] = f"인증되었습니다. {info.name}님, 퀴즈를 시작합니다!"
    return state

def handle_validation_result(state: SystemState) -> SystemState:
    """검증 결과를 chat_history에 추가"""
    state["chat_history"].append(("assistant", state["validation_message"]))
    return state

def decide_to_start_quiz(state: SystemState) -> str:
    """응시자 검증 결과에 따라 퀴즈 시작 여부 결정"""
    return "start_quiz" if state["is_applicant_valid"] else "end_student_flow"


# - Quiz Agent Nodes

# In[46]:


def start_quiz(state: SystemState) -> SystemState:
    questions = load_quiz()
    if not questions:
        state["chat_history"].append(("assistant", "퀴즈를 불러오는 데 실패했습니다."))
        state["questions"] = []
    else:
        state["questions"] = questions
        state["quiz_index"] = 0
        state["user_answers"] = []
    return state

def ask_question(state: SystemState) -> SystemState:
    idx = state["quiz_index"]
    q = state["questions"][idx]
    text = f"문제 {idx + 1}/{len(state['questions'])}: {q['question']}"
    if q["type"] == "multiple_choice":
        choices = [f"{i + 1}. {c}" for i, c in enumerate(q["choices"])]
        text += "\n" + "\n".join(choices)
    state["chat_history"].append(("assistant", text))
    state["user_input"] = "" # 이전 사용자 입력 초기화
    return state

def process_and_store_answer(state: SystemState) -> SystemState:
    """퀴즈 답변 처리 및 저장"""
    idx = state["quiz_index"]
    q = state["questions"][idx]
    user_input = state["user_input"].strip()

    processed_answer = user_input
    if q["type"] == "multiple_choice":
        try:
            sel = int(user_input) - 1
            if 0 <= sel < len(q["choices"]):
                processed_answer = q["choices"][sel]
        except (ValueError, IndexError):
            pass # 숫자가 아니거나 범위 밖이면 원본 텍스트를 답변으로 간주

    state["user_answers"].append(processed_answer)
    state["quiz_index"] += 1
    return state

def should_continue_quiz(state: SystemState) -> str:
    """퀴즈 계속 진행 또는 채점 시작 결정"""
    if state.get("quiz_index", 0) < len(state.get("questions", [])):
        return "ask_question"
    else:
        return "grade_quiz"

# grade_and_save_results 함수를 아래 내용으로 교체

def grade_and_save_results(state: SystemState) -> SystemState:
    """LLM으로 채점하고 결과를 파일에 저장"""
    # 1. 채점 프롬프트 준비
    grading_input_parts = []
    for i, (q, a) in enumerate(zip(state["questions"], state["user_answers"])):
        part = (f"문제 {i+1}:\n"
                f"ID: {q['id']}\n"
                f"질문: {q['question']}\n"
                f"정답: {q['answer']}\n"
                f"사용자 답변: {a}\n---")
        grading_input_parts.append(part)
    grading_input_str = "\n".join(grading_input_parts)

    # 2. LLM 채점
    parser = PydanticOutputParser(pydantic_object=FinalReport)
    system_message = "당신은 '명탐정 코난' 퀴즈 전문 채점관입니다. 주어진 문제, 정답, 사용자 답변을 바탕으로 채점하고, 각 문제의 ID를 포함하여 JSON 형식으로 결과를 반환하세요. 정답 여부를 판단하고 친절한 해설을 덧붙여주세요. 마지막에는 '총점: X/Y' 형식으로 최종 점수를 요약해야 합니다."
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "{grading_data}\n\n{format_instructions}")
    ])
    chain = prompt | llm | parser

    try:
        report = chain.invoke({
            "grading_data": grading_input_str,
            "format_instructions": parser.get_format_instructions(),
        })
        state["final_report"] = report
        
        # 3. 결과 저장
        today = datetime.date.today().strftime("%Y-%m-%d")
        info = state["applicant_info"]
        result_path = os.path.join(RESULTS_DIR, today, str(info.class_name), "result.json")
        
        all_results = load_json(result_path) or []
        score = sum(1 for res in report.results if res.is_correct)

        # 새 결과 데이터 생성
        new_result = {
            "student_id": info.student_id,
            "name": info.name,
            "score": score,
            "answers": [res.model_dump() for res in report.results]
        }
        all_results.append(new_result)
        save_json(all_results, result_path)

    except Exception as e:
        print(f"채점 및 저장 중 오류 발생: {e}")
        state["final_report"] = FinalReport(results=[], total_score="채점 오류가 발생했습니다.")
        state["chat_history"].append(("assistant", "채점 중 오류가 발생하여 결과를 저장하지 못했습니다."))

    return state

def format_final_report(state: SystemState) -> SystemState:
    """최종 채점 결과를 사용자에게 보여주기 위해 포맷팅"""
    report = state["final_report"]
    parts = ["채점이 완료되었습니다! 🎉\n"]
    if report and report.results:
        for i, res in enumerate(report.results):
            is_correct_text = "✅ 정답" if res.is_correct else "❌ 오답"
            parts.append(f"--- 문제 {i + 1} ---\n"
                         f"문제: {res.question}\n"
                         f"정답: {res.correct_answer}\n"
                         f"제출한 답변: {res.user_answer}\n"
                         f"결과: {is_correct_text}\n"
                         f"해설: {res.explanation}\n")
        parts.append(f"**{report.total_score}**")
    else:
        parts.append("채점 결과를 생성하는 데 실패했습니다.")
    
    parts.append("\n\n새로운 응시자는 정보를 입력해주세요. (예: 1반 홍길동 S25B001 010-1234-5678)")
    state["chat_history"].append(("assistant", "\n".join(parts)))
    return state


# - Report Agent Nodes

# In[47]:


def decide_to_generate_report(state: SystemState) -> str:
    """리포트 요청이 유효한지 확인하여 다음 단계를 결정합니다."""
    if state.get("report_request"):
        return "generate_report"
    else:
        return "display_report"

def extract_report_request(state: SystemState) -> SystemState:
    """LLM을 사용해 리포트 요청 정보 추출"""
    parser = PydanticOutputParser(pydantic_object=ReportRequest)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 교수님의 리포트 요청을 분석하여 JSON 형식으로 추출하는 정확한 비서입니다.
요청에서 날짜(YYYY-MM-DD), 반 이름, 그리고 리포트 종류를 정확히 추출해야 합니다.
리포트 종류(report_type)는 반드시 "성적" 또는 "오답" 두 가지 중 하나여야 합니다.

## 예시:
- 입력: "오늘 2반 성적 순위 리포트 좀 보여줘" (오늘이 2025-07-07이라고 가정)
  추출: {{"date": "2025-07-07", "class_name": "2반", "report_type": "성적"}}
- 입력: "2025-07-06 1반 오답 리포트"
  추출: {{"date": "2025-07-06", "class_name": "1반", "report_type": "오답"}}
- 입력: "1반 성적"
  추출: (날짜 정보가 없어 실패해야 함)
"""),
        ("human", "오늘 날짜: {today}\n교수님 요청: {user_input}\n\n{format_instructions}")
    ])
    chain = prompt | llm | parser
    try:
        today_str = datetime.date.today().strftime("%Y-%m-%d")
        req = chain.invoke({
            "today": today_str,
            "user_input": state["user_input"], 
            "format_instructions": parser.get_format_instructions()
        })
        state["report_request"] = req
    except Exception as e:
        state["report_output"] = f"리포트 요청을 이해하지 못했습니다. 'YYYY-MM-DD N반 [성적 순위/오답 리포트]' 형식으로 요청해주세요. (오류: {e})"
        state["report_request"] = None
    return state

# generate_report 함수를 아래 내용으로 교체합니다.

def generate_report(state: SystemState) -> SystemState:
    """요청에 따라 리포트를 생성 (오답 리포트에 '정답' 컬럼 추가)"""
    # 1. 어떤 경우에도 KeyError가 발생하지 않도록 기본 오류 메시지로 'report_output'을 초기화합니다.
    state["report_output"] = "리포트를 생성하는 중 알 수 없는 오류가 발생했습니다."

    try:
        # 2. 로직 시작: report_request 키가 있는지 다시 한번 확인 (방어적 코딩)
        if not state.get("report_request"):
            state["report_output"] = "리포트 요청 정보가 없어 생성에 실패했습니다."
            return state

        req = state["report_request"]
        result_path = os.path.join(RESULTS_DIR, req.date, req.class_name, "result.json")
        data = load_json(result_path)

        if not data:
            state["report_output"] = f"{req.date}의 {req.class_name} 응시 결과가 없습니다."
            return state

        # 3. 리포트 유형에 따라 생성
        if req.report_type == "성적":
            sorted_data = sorted(data, key=lambda x: x["score"], reverse=True)
            report_lines = [f"### {req.date} {req.class_name} 성적 순위 리포트\n",
                            "| 순위 | 학번 | 이름 | 점수 |",
                            "|:---:|:---:|:---:|:---:|"]
            total_questions = len(sorted_data[0]['answers']) if sorted_data else 0
            for i, student in enumerate(sorted_data):
                report_lines.append(f"| {i+1} | {student['student_id']} | {student['name']} | {student['score']}/{total_questions} |")
            state["report_output"] = "\n".join(report_lines)

        elif req.report_type == "오답":
            # --- '오답 리포트' 로직 수정 시작 ---
            quiz_errors = {}
            total_students = len(data)
            for student in data:
                for answer in student["answers"]:
                    if not answer["is_correct"]:
                        qid = answer["quiz_id"]
                        if qid not in quiz_errors:
                            # [수정 1] 'correct_answer'도 함께 저장
                            quiz_errors[qid] = {
                                "question": answer["question"],
                                "correct_answer": answer["correct_answer"], 
                                "count": 0
                            }
                        quiz_errors[qid]["count"] += 1
            
            sorted_errors = sorted(quiz_errors.items(), key=lambda item: item[1]["count"], reverse=True)
            
            # [수정 2] 테이블 헤더에 '정답' 컬럼 추가
            report_lines = [f"### {req.date} {req.class_name} 오답률 TOP 3 리포트 (총 {total_students}명 응시)\n",
                            "| 순위 | 문제 | 정답 | 오답자 수 | 오답률 |",
                            "|:---:|:---|:---|:---:|:---:|"]
            
            if not sorted_errors:
                report_lines.append("| - | 오답이 기록된 문제가 없습니다. | - | - | - |")
            else:
                for i, (qid, details) in enumerate(sorted_errors[:3]):
                    error_rate = (details['count'] / total_students) * 100
                    # [수정 3] 테이블 행에 'correct_answer' 내용 추가
                    report_lines.append(f"| {i+1} | {details['question']} | {details['correct_answer']} | {details['count']}명 | {error_rate:.1f}% |")
            
            state["report_output"] = "\n".join(report_lines)
            # --- '오답 리포트' 로직 수정 끝 ---

        else:
            state["report_output"] = f"알 수 없는 리포트 유형입니다: '{req.report_type}'. 시스템 관리자에게 문의하세요."

    except Exception as e:
        print(f"리포트 생성 중 예외 발생: {e}")
        state["report_output"] = f"리포트 생성에 실패했습니다. (오류: {e})"

    return state

def display_report(state: SystemState) -> SystemState:
    """생성된 리포트를 chat_history에 추가"""
    state["chat_history"].append(("assistant", state["report_output"]))
    return state


# - handle_unknown_request nodes

# In[48]:


def handle_unknown_request(state: SystemState) -> SystemState:
    """알 수 없는 요청 처리"""
    message = "요청을 이해하지 못했습니다.\n" \
              "퀴즈에 응시하시려면 '1반 홍길동 S25B001 010-1234-5678'과 같이 정보를 입력해주세요.\n" \
              "리포트가 필요하시면 '2025-07-07 1반 성적 순위 리포트'와 같이 요청해주세요."
    state["chat_history"].append(("assistant", message))
    return state


# ### LangGraph 워크플로우 구성

# In[49]:


workflow = StateGraph(SystemState)

# 노드 추가
workflow.add_node("extract_applicant_info", extract_applicant_info)
workflow.add_node("validate_applicant", validate_applicant)
workflow.add_node("handle_validation_result", handle_validation_result)
workflow.add_node("start_quiz", start_quiz)
workflow.add_node("ask_question", ask_question)
workflow.add_node("process_answer", process_and_store_answer)
workflow.add_node("grade_and_save", grade_and_save_results)
workflow.add_node("format_final_report", format_final_report)
workflow.add_node("extract_report_request", extract_report_request)
workflow.add_node("generate_report", generate_report)
workflow.add_node("display_report", display_report)
workflow.add_node("handle_unknown_request", handle_unknown_request)

# 진입점 설정 (사용자 유형에 따라 분기)
workflow.set_conditional_entry_point(
    route_user_type,
    {
        "student_flow": "extract_applicant_info",
        "professor_flow": "extract_report_request",
        "unknown_flow": "handle_unknown_request",
    }
)

# 학생(Student) 워크플로우 엣지 연결
workflow.add_edge("extract_applicant_info", "validate_applicant")
workflow.add_edge("validate_applicant", "handle_validation_result")
workflow.add_conditional_edges(
    "handle_validation_result",
    decide_to_start_quiz,
    {"start_quiz": "start_quiz", "end_student_flow": END}
)
workflow.add_edge("start_quiz", "ask_question")
# 퀴즈 루프: 답변을 받으면 다음 문제로 가거나 채점으로 넘어감
workflow.add_node("router_after_answer", lambda state: state) # Dummy node
workflow.add_edge("process_answer", "router_after_answer")
workflow.add_conditional_edges(
    "router_after_answer",
    should_continue_quiz,
    {"ask_question": "ask_question", "grade_quiz": "grade_and_save"}
)
workflow.add_edge("ask_question", END) # 퀴즈 질문 후 사용자 입력을 기다림
workflow.add_edge("grade_and_save", "format_final_report")
workflow.add_edge("format_final_report", END)

# Professor(교수) 워크플로우 엣지 연결
workflow.add_conditional_edges(
    "extract_report_request",  # 시작 노드
    decide_to_generate_report, # 판단 함수
    {
        "generate_report": "generate_report", # 판단 결과가 "generate_report"이면 generate_report 노드로 이동
        "display_report": "display_report",   # 판단 결과가 "display_report"이면 display_report 노드로 이동
    }
)
# 이제 generate_report는 항상 유효한 요청을 받을 때만 실행됩니다.
workflow.add_edge("generate_report", "display_report")
workflow.add_edge("display_report", END)

# 알 수 없는 요청 처리
workflow.add_edge("handle_unknown_request", END)

# 그래프 컴파일
multi_agent_app = workflow.compile()


# ### 초기 상태 설정 및 UI 인터페이스 로직

# In[50]:


def init_state():
    """초기 상태를 생성하는 함수"""
    return {"system_state": {
        "chat_history": [],
        "user_input": "",
        "user_type": None
    }}

def chat_fn(user_input, state):
    """Gradio 상호작용을 처리하는 메인 함수"""
    system_state = state["system_state"]

    # 1. UI 상태(딕셔너리 리스트)에 사용자 입력 추가
    # state에 저장된 chat_history는 항상 딕셔너리 리스트 형식이라고 가정
    chat_history_dicts = system_state.get("chat_history", [])
    if not isinstance(chat_history_dicts, list): # 혹시 모를 비정상 상태 대비
        chat_history_dicts = []
    chat_history_dicts.append({"role": "user", "content": user_input})

    # 2. 내부 로직용 상태(튜플 리스트) 생성
    # 내부 로직(LangGraph)은 튜플 리스트 형식의 기록을 사용
    internal_history_tuples = [(item['role'], item.get('content', '')) for item in chat_history_dicts]
    
    # 내부 로직에 전달할 상태 객체 생성
    internal_state = {
        **system_state,
        "chat_history": internal_history_tuples,
        "user_input": user_input,
    }

    # 3. 퀴즈 진행 여부에 따라 분기
    is_in_quiz = internal_state.get("questions") and not internal_state.get("final_report")
    if is_in_quiz:
        quiz_state = process_and_store_answer(internal_state)
        next_step = should_continue_quiz(quiz_state)
        if next_step == "ask_question":
            final_internal_state = ask_question(quiz_state)
        else: # "grade_quiz"
            graded_state = grade_and_save_results(quiz_state)
            final_internal_state = format_final_report(graded_state)
    else: # 새로운 요청 (라우팅부터 시작)
        # 이전 상태 초기화 (채팅 기록은 유지)
        current_history = internal_state["chat_history"]
        clean_state = init_state()['system_state']
        clean_state['chat_history'] = current_history
        clean_state['user_input'] = user_input
        
        final_internal_state = multi_agent_app.invoke(clean_state)

    # 4. 최종 상태를 UI 형식(딕셔너리 리스트)으로 변환하여 저장 및 반환
    final_chat_history_tuples = final_internal_state.get("chat_history", [])
    
    final_chat_display = []
    for role, content in final_chat_history_tuples:
        if role == "user":
            final_chat_display.append({"role": "user", "content": content})
        elif role == "assistant":
            final_chat_display.append({"role": "assistant", "content": content})
    
    # UI에 표시될 기록과 다음 state에 저장될 기록을 모두 딕셔너리 형식으로 통일
    final_internal_state["chat_history"] = final_chat_display
    state["system_state"] = final_internal_state

    return final_chat_display, state


# ### Gradio UI 구성

# In[51]:


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 멀티 에이전트 퀴즈 시스템")
    gr.Markdown(
        " **응시자(학생)** : `1반 홍길동 S25B001 010-1234-5678` 형식으로 정보를 입력하여 퀴즈를 시작하세요.\n\n "
        " **교수** : `2025-07-07 1반 성적 순위 리포트` 형식으로 리포트를 요청하세요."
    )

    chatbot = gr.Chatbot(
        label="퀴즈 및 리포트 챗봇",
        height=400,
        type="messages"
    )
    
    chatbot_display = gr.JSON(visible=False)

    txt = gr.Textbox(placeholder="정보를 입력하거나 리포트를 요청하세요...", show_label=False)
    state = gr.State(init_state())

    def update_chatbot_ui(json_data):
        return json_data

    chatbot_display.change(update_chatbot_ui, chatbot_display, chatbot)

    txt.submit(chat_fn, inputs=[txt, state], outputs=[chatbot_display, state])
    txt.submit(lambda: "", None, txt)

demo.launch()

