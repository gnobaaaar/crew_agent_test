"""
질문 수집 에이전트 (Question Collector)
사용자의 질문을 분석하고 검색에 최적화된 키워드를 추출합니다.
"""

from crewai import Agent, Task
from crewai.tools import BaseTool
from typing import Dict, Any, List

from utils.config_loader import config_loader
from utils.logger import logger, log_info, log_success, log_error


class QuestionAnalysisTool(BaseTool):
    """질문 분석 도구"""
    
    name: str = "question_analysis"
    description: str = "사용자 질문을 분석하여 핵심 키워드와 의도를 추출합니다"
    
    def _run(self, question: str) -> Dict[str, Any]:
        """
        질문을 분석하여 구조화된 정보를 반환합니다
        
        Args:
            question (str): 사용자 질문
            
        Returns:
            Dict[str, Any]: 분석 결과
        """
        try:
            # 질문 기본 정보
            analysis = {
                "original_question": question,
                "question_length": len(question),
                "question_type": self._classify_question_type(question),
                "keywords": self._extract_keywords(question),
                "search_queries": self._generate_search_queries(question),
                "priority_level": self._assess_priority(question)
            }
            
            log_info(f"질문 분석 완료: {len(analysis['keywords'])}개 키워드 추출")
            return analysis
            
        except Exception as e:
            log_error(f"질문 분석 실패: {str(e)}")
            return {
                "original_question": question,
                "error": str(e),
                "keywords": [],
                "search_queries": [question]
            }
    
    def _classify_question_type(self, question: str) -> str:
        """질문 유형을 분류합니다"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['어떻게', 'how', '방법']):
            return "how_to"
        elif any(word in question_lower for word in ['무엇', 'what', '뭐']):
            return "what_is"
        elif any(word in question_lower for word in ['왜', 'why', '이유']):
            return "why"
        elif any(word in question_lower for word in ['언제', 'when', '시기']):
            return "when"
        elif any(word in question_lower for word in ['어디', 'where', '위치']):
            return "where"
        else:
            return "general"
    
    def _extract_keywords(self, question: str) -> List[str]:
        """질문에서 핵심 키워드를 추출합니다"""
        # 간단한 키워드 추출 (실제로는 더 정교한 NLP 기법 사용 가능)
        import re
        
        # 불용어 제거
        stop_words = {
            '이', '그', '저', '것', '들', '에', '를', '을', '의', '가', '은', '는', 
            '으로', '로', '에서', '와', '과', '하고', '그리고', '또는', '하지만',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'
        }
        
        # 단어 추출 (한글, 영문, 숫자)
        words = re.findall(r'[가-힣a-zA-Z0-9]+', question)
        
        # 불용어 제거 및 길이 필터링
        keywords = [word for word in words 
                   if word.lower() not in stop_words and len(word) > 1]
        
        # 중복 제거
        keywords = list(set(keywords))
        
        return keywords[:10]  # 최대 10개 키워드
    
    def _generate_search_queries(self, question: str) -> List[str]:
        """검색용 쿼리를 생성합니다"""
        keywords = self._extract_keywords(question)
        
        queries = [question]  # 원본 질문
        
        # 키워드 조합으로 추가 쿼리 생성
        if len(keywords) >= 2:
            # 상위 키워드들로 조합 생성
            for i in range(min(3, len(keywords))):
                for j in range(i+1, min(3, len(keywords))):
                    query = f"{keywords[i]} {keywords[j]}"
                    queries.append(query)
        
        return queries[:5]  # 최대 5개 쿼리
    
    def _assess_priority(self, question: str) -> str:
        """질문의 우선순위를 평가합니다"""
        urgent_words = ['긴급', '급함', '빨리', '즉시', 'urgent', 'asap']
        important_words = ['중요', '필수', '반드시', 'important', 'critical']
        
        question_lower = question.lower()
        
        if any(word in question_lower for word in urgent_words):
            return "urgent"
        elif any(word in question_lower for word in important_words):
            return "high"
        else:
            return "normal"


class QuestionCollectorAgent:
    """질문 수집 에이전트 클래스"""
    
    def __init__(self):
        """QuestionCollectorAgent 초기화"""
        
        # 설정 로드
        agents_config = config_loader.get_agents_config()
        collector_config = agents_config.get("question_collector", {})
        
        # 도구 초기화
        self.analysis_tool = QuestionAnalysisTool()
        
        # CrewAI 에이전트 생성 (LangChain Gemini 사용)
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        # Gemini LLM 설정
        api_config = config_loader.get_api_config()
        gemini_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_config.get("gemini_api_key"),
            temperature=0.7
        )
        
        self.agent = Agent(
            role=collector_config.get("role", "질문 분석 전문가"),
            goal=collector_config.get("goal", "사용자의 질문을 분석하고 검색에 최적화된 키워드를 추출합니다"),
            backstory=collector_config.get("backstory", "질문 분석 전문가입니다"),
            verbose=collector_config.get("verbose", True),
            allow_delegation=collector_config.get("allow_delegation", False),
            tools=[self.analysis_tool],
            llm=gemini_llm
        )
        
        log_success("질문 수집 에이전트 초기화 완료")
    
    def create_analysis_task(self, question: str) -> Task:
        """
        질문 분석 태스크를 생성합니다
        
        Args:
            question (str): 분석할 질문
            
        Returns:
            Task: CrewAI 태스크 객체
        """
        task_description = f"""
        다음 사용자 질문을 분석해주세요:
        
        질문: "{question}"
        
        분석해야 할 항목:
        1. 질문의 핵심 키워드 추출
        2. 질문 유형 분류 (how_to, what_is, why, when, where, general)
        3. 검색에 최적화된 쿼리 생성
        4. 질문의 우선순위 평가
        
        결과는 구조화된 형태로 제공해주세요.
        """
        
        task = Task(
            description=task_description,
            agent=self.agent,
            expected_output="질문 분석 결과를 JSON 형태로 반환"
        )
        
        return task
    
    def analyze_question(self, question: str) -> Dict[str, Any]:
        """
        질문을 분석합니다
        
        Args:
            question (str): 분석할 질문
            
        Returns:
            Dict[str, Any]: 분석 결과
        """
        log_info(f"질문 분석 시작: '{question[:50]}...'")
        
        try:
            # 도구를 직접 사용하여 분석
            result = self.analysis_tool._run(question)
            
            log_success("질문 분석 완료")
            return result
            
        except Exception as e:
            log_error(f"질문 분석 실패: {str(e)}")
            return {
                "original_question": question,
                "error": str(e),
                "keywords": [],
                "search_queries": [question]
            }
    
    def get_search_queries(self, question: str) -> List[str]:
        """
        질문에서 검색 쿼리를 추출합니다
        
        Args:
            question (str): 원본 질문
            
        Returns:
            List[str]: 검색 쿼리 목록
        """
        analysis = self.analyze_question(question)
        return analysis.get("search_queries", [question])
    
    def get_keywords(self, question: str) -> List[str]:
        """
        질문에서 키워드를 추출합니다
        
        Args:
            question (str): 원본 질문
            
        Returns:
            List[str]: 키워드 목록
        """
        analysis = self.analyze_question(question)
        return analysis.get("keywords", [])
    
    def get_question_type(self, question: str) -> str:
        """
        질문 유형을 반환합니다
        
        Args:
            question (str): 원본 질문
            
        Returns:
            str: 질문 유형
        """
        analysis = self.analyze_question(question)
        return analysis.get("question_type", "general")