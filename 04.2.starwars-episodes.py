# 스타워즈 에피소드 시대적 배경에 따라 감상하기

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# TODO:  정의영화 스타워즈 전체 에피소드의 개봉연도, 제목, 줄거리 배경 연도를 출력하는 llm 체인 구성하기 - 결과 출력
# TODO:  응답 결과를 이용하여 스타워즈 배경 연도에 따라 영화 제목과 개봉연도를 출력하는 코드 작성 - 결과 출력
