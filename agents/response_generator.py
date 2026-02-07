"""
답변 생성 에이전트 (Response Generator)
검색된 정보를 바탕으로 사용자에게 정확하고 유용한 답변을 제공합니다.
"""

from crewai import Agent, Task
from crewai.tools import BaseTool
from typing import Dict, Any, List
from langchain_google_genai import ChatGoogleGenerativeAI
import json

from utils.config_loader import config_loader
from utils.logger import logger, log_info, log_success, log_error, log_warning


class AnswerGenerationTool(BaseTool):
    """답변 생성 도구"""
    
    name: str = "answer_generation"
    description: str = "검색된 문서 내용을 바탕으로 사용자 질문에 대한 답변을 생성합니다"
    
    def __init__(self):
        super().__init__()
    
    def _run(self, question: str, context_documents: List[str], question_type: str = "general") -> Dict[str, Any]:
        """
        답변을 생성합니다
        
        Args:
            question (str): 사용자 질문
            context_documents (List[str]): 참고 문서들
            question_type (str): 질문 유형
            
        Returns:
            Dict[str, Any]: 생성된 답변과 메타데이터
        """
        try:
            log_info(f"답변 생성 시작: '{question[:50]}...'")
            
            # 컨텍스트 문서 준비
            if not context_documents:
                return self._generate_no_context_response(question)
            
            # LangChain Gemini 모델 생성
            api_config = config_loader.get_api_config()
            model = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=api_config.get("gemini_api_key"),
                temperature=0.7
            )
            
            # 프롬프트 생성
            prompt = self._create_answer_prompt(question, context_documents, question_type)
            
            # LangChain Gemini로 답변 생성
            response = model.invoke(prompt)
            
            if not response.content:
                raise Exception("LangChain Gemini에서 응답을 받지 못했습니다")
            
            # 답변을 마크다운 형식으로 후처리
            markdown_answer = self._format_answer_as_markdown(response.content.strip(), context_documents)
            
            # 답변 후처리
            answer_data = {
                "question": question,
                "answer": markdown_answer,
                "question_type": question_type,
                "context_used": len(context_documents),
                "confidence_level": self._assess_confidence(response.content, context_documents),
                "sources_count": len(context_documents),
                "model_used": "gemini-2.0-flash"
            }
            
            log_success("답변 생성 완료")
            return answer_data
            
        except Exception as e:
            log_error(f"답변 생성 실패: {str(e)}")
            return {
                "question": question,
                "answer": f"죄송합니다. 답변 생성 중 오류가 발생했습니다: {str(e)}",
                "error": str(e),
                "confidence_level": "low"
            }
    
    def _create_answer_prompt(self, question: str, context_documents: List[str], question_type: str) -> str:
        """답변 생성용 프롬프트를 생성합니다"""
        
        # 컨텍스트 문서들을 하나의 텍스트로 결합
        context_text = "\n\n---\n\n".join(context_documents)
        
        # 질문 유형별 지시사항
        type_instructions = {
            "how_to": "단계별로 명확하게 설명해주세요.",
            "what_is": "정의와 특징을 중심으로 설명해주세요.",
            "why": "이유와 원인을 논리적으로 설명해주세요.",
            "when": "시기나 조건을 명확히 제시해주세요.",
            "where": "위치나 장소 정보를 구체적으로 제공해주세요.",
            "general": "포괄적이고 유용한 정보를 제공해주세요."
        }
        
        instruction = type_instructions.get(question_type, type_instructions["general"])
        
        prompt = f"""
당신은 문서 기반 질문 답변 전문가입니다. 제공된 문서 내용을 바탕으로 사용자의 질문에 정확하고 도움이 되는 답변을 제공해주세요.

**사용자 질문:**
{question}

**참고 문서:**
{context_text}

**답변 지침:**
1. 제공된 문서 내용만을 바탕으로 답변하세요
2. {instruction}
3. 답변은 한국어로 작성하세요
4. 명확하고 이해하기 쉽게 설명하세요
5. 문서에 없는 내용은 추측하지 마세요
6. 불확실한 부분이 있다면 명시하세요
7. 답변은 일반 텍스트로 작성하세요 (마크다운 헤더나 특수 문자 사용 금지)

**답변:**
"""
        
        return prompt
    
    def _generate_no_context_response(self, question: str) -> Dict[str, Any]:
        """컨텍스트가 없을 때의 응답을 생성합니다"""
        
        no_context_prompt = f"""
사용자가 다음과 같은 질문을 했지만, 관련된 문서를 찾을 수 없습니다:

질문: {question}

이 상황에서 도움이 될 수 있는 일반적인 조언이나 다른 접근 방법을 제안해주세요.
답변은 한국어로 작성하고, 친근하고 도움이 되는 톤으로 해주세요.
"""
        
        try:
            # LangChain Gemini 모델 생성
            api_config = config_loader.get_api_config()
            model = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=api_config.get("gemini_api_key"),
                temperature=0.7
            )
            
            response = model.invoke(no_context_prompt)
            
            return {
                "question": question,
                "answer": response.content.strip() if response.content else "관련 정보를 찾을 수 없어 답변을 생성할 수 없습니다.",
                "context_used": 0,
                "confidence_level": "low",
                "no_context": True
            }
            
        except Exception as e:
            return {
                "question": question,
                "answer": "죄송합니다. 관련 문서를 찾을 수 없어 답변을 제공할 수 없습니다. 다른 키워드로 질문해보시거나, 더 구체적인 질문을 해주세요.",
                "error": str(e),
                "confidence_level": "low",
                "no_context": True
            }
    
    def _assess_confidence(self, answer: str, context_documents: List[str]) -> str:
        """답변의 신뢰도를 평가합니다"""
        
        # 간단한 신뢰도 평가 로직
        if not context_documents:
            return "low"
        
        # 답변 길이와 컨텍스트 수를 기반으로 평가
        answer_length = len(answer)
        context_count = len(context_documents)
        
        if answer_length > 200 and context_count >= 3:
            return "high"
        elif answer_length > 100 and context_count >= 2:
            return "medium"
        else:
            return "low"
    
    def _format_answer_as_markdown(self, answer: str, context_documents: List[Dict[str, Any]]) -> str:
        """
        답변을 마크다운 형식으로 포맷팅합니다
        
        Args:
            answer (str): 원본 답변
            context_documents (List[Dict[str, Any]]): 참조 문서들
            
        Returns:
            str: 마크다운 형식의 답변
        """
        try:
            # 기본 답변 정리
            formatted_answer = answer.strip()
            
            # 마크다운 형식으로 구조화
            markdown_parts = []
            
            # 답변 내용
            markdown_parts.append("## 📝 답변\n\n")
            markdown_parts.append(formatted_answer)
            markdown_parts.append("\n")
            
            # 참조 문서 정보 추가
            if context_documents:
                markdown_parts.append("\n---\n\n")
                markdown_parts.append("## 📚 참조 문서\n\n")
                
                for i, doc in enumerate(context_documents[:3], 1):  # 최대 3개만 표시
                    if isinstance(doc, dict):
                        source = doc.get('metadata', {}).get('source_file', '알 수 없음')
                        chunk_id = doc.get('metadata', {}).get('chunk_id', 0)
                    else:
                        source = '문서'
                        chunk_id = i
                    
                    # 파일명만 추출
                    if '/' in str(source):
                        filename = str(source).split('/')[-1]
                    else:
                        filename = str(source)
                    
                    markdown_parts.append(f"- **문서 {i}**: `{filename}` (청크 {chunk_id})\n")
                
                markdown_parts.append("\n")
            
            # 추가 정보
            markdown_parts.append("---\n\n")
            markdown_parts.append("💡 **도움말**: 더 구체적인 질문을 하시면 더 정확한 답변을 드릴 수 있습니다.\n")
            
            return "".join(markdown_parts)
            
        except Exception as e:
            log_error(f"마크다운 포맷팅 실패: {str(e)}")
            return f"## 📝 답변\n\n{answer}\n\n---\n💡 **참고**: 답변 포맷팅 중 오류가 발생했습니다.\n"


class ResponseFormatterTool(BaseTool):
    """답변 포맷팅 도구"""
    
    name: str = "response_formatter"
    description: str = "생성된 답변을 사용자 친화적인 형태로 포맷팅합니다"
    
    def _run(self, answer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        답변을 포맷팅합니다
        
        Args:
            answer_data (Dict[str, Any]): 원본 답변 데이터
            
        Returns:
            Dict[str, Any]: 포맷팅된 답변
        """
        try:
            formatted_response = {
                "question": answer_data.get("question", ""),
                "answer": self._format_answer_text(answer_data.get("answer", "")),
                "metadata": {
                    "confidence": answer_data.get("confidence_level", "medium"),
                    "sources_used": answer_data.get("sources_count", 0),
                    "question_type": answer_data.get("question_type", "general"),
                    "model": answer_data.get("model_used", "unknown")
                },
                "suggestions": self._generate_suggestions(answer_data)
            }
            
            return formatted_response
            
        except Exception as e:
            log_error(f"답변 포맷팅 실패: {str(e)}")
            return {
                "question": answer_data.get("question", ""),
                "answer": answer_data.get("answer", "답변을 포맷팅할 수 없습니다."),
                "error": str(e)
            }
    
    def _format_answer_text(self, answer: str) -> str:
        """답변 텍스트를 포맷팅합니다"""
        
        # 기본 정리
        formatted = answer.strip()
        
        # 연속된 공백 제거
        import re
        formatted = re.sub(r'\s+', ' ', formatted)
        
        # 문단 구분 개선
        formatted = re.sub(r'\n\s*\n', '\n\n', formatted)
        
        return formatted
    
    def _format_answer_as_markdown(self, answer: str, context_documents: List[Dict[str, Any]]) -> str:
        """
        답변을 마크다운 형식으로 포맷팅합니다
        
        Args:
            answer (str): 원본 답변
            context_documents (List[Dict[str, Any]]): 참조 문서들
            
        Returns:
            str: 마크다운 형식의 답변
        """
        try:
            # 기본 답변 정리
            formatted_answer = answer.strip()
            
            # 마크다운 형식으로 구조화
            markdown_parts = []
            
            # 답변 내용
            markdown_parts.append("## 📝 답변\n")
            markdown_parts.append(formatted_answer)
            markdown_parts.append("\n")
            
            # 참조 문서 정보 추가
            if context_documents:
                markdown_parts.append("\n---\n")
                markdown_parts.append("## 📚 참조 문서\n")
                
                for i, doc in enumerate(context_documents[:3], 1):  # 최대 3개만 표시
                    source = doc.get('metadata', {}).get('source_file', '알 수 없음')
                    chunk_id = doc.get('metadata', {}).get('chunk_id', 0)
                    
                    # 파일명만 추출
                    if '/' in source:
                        filename = source.split('/')[-1]
                    else:
                        filename = source
                    
                    markdown_parts.append(f"**{i}.** `{filename}` (청크 {chunk_id})\n")
                
                markdown_parts.append("\n")
            
            # 추가 정보
            markdown_parts.append("---\n")
            markdown_parts.append("💡 **도움말**: 더 구체적인 질문을 하시면 더 정확한 답변을 드릴 수 있습니다.\n")
            
            return "".join(markdown_parts)
            
        except Exception as e:
            log_error(f"마크다운 포맷팅 실패: {str(e)}")
            return f"## 📝 답변\n\n{answer}\n\n---\n💡 **참고**: 답변 포맷팅 중 오류가 발생했습니다.\n"
    
    def _generate_suggestions(self, answer_data: Dict[str, Any]) -> List[str]:
        """추가 제안사항을 생성합니다"""
        
        suggestions = []
        
        confidence = answer_data.get("confidence_level", "medium")
        sources_count = answer_data.get("sources_count", 0)
        
        if confidence == "low":
            suggestions.append("더 구체적인 질문을 해보세요")
            suggestions.append("다른 키워드로 검색해보세요")
        
        if sources_count == 0:
            suggestions.append("관련 문서를 추가로 업로드해보세요")
        
        if sources_count < 3:
            suggestions.append("더 많은 참고 자료가 있다면 더 정확한 답변을 제공할 수 있습니다")
        
        return suggestions


class ResponseGeneratorAgent:
    """답변 생성 에이전트 클래스"""
    
    def __init__(self):
        """ResponseGeneratorAgent 초기화"""
        
        # 설정 로드
        agents_config = config_loader.get_agents_config()
        generator_config = agents_config.get("response_generator", {})
        
        # 도구 초기화
        self.answer_tool = AnswerGenerationTool()
        self.formatter_tool = ResponseFormatterTool()
        
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
            role=generator_config.get("role", "답변 생성 전문가"),
            goal=generator_config.get("goal", "검색된 정보를 바탕으로 사용자에게 정확하고 유용한 답변을 제공합니다"),
            backstory=generator_config.get("backstory", "답변 생성 전문가입니다"),
            verbose=generator_config.get("verbose", True),
            allow_delegation=generator_config.get("allow_delegation", False),
            tools=[self.answer_tool, self.formatter_tool],
            llm=gemini_llm
        )
        
        log_success("답변 생성 에이전트 초기화 완료")
    
    def create_generation_task(self, question: str, context_documents: List[str], question_type: str = "general") -> Task:
        """
        답변 생성 태스크를 생성합니다
        
        Args:
            question (str): 사용자 질문
            context_documents (List[str]): 참고 문서들
            question_type (str): 질문 유형
            
        Returns:
            Task: CrewAI 태스크 객체
        """
        context_summary = f"{len(context_documents)}개의 참고 문서" if context_documents else "참고 문서 없음"
        
        task_description = f"""
        다음 질문에 대한 답변을 생성해주세요:
        
        질문: "{question}"
        질문 유형: {question_type}
        참고 자료: {context_summary}
        
        수행해야 할 작업:
        1. 제공된 문서 내용을 분석
        2. 질문에 가장 적합한 답변 생성
        3. 답변의 신뢰도 평가
        4. 사용자 친화적인 형태로 포맷팅
        
        정확하고 도움이 되는 답변을 제공해주세요.
        """
        
        task = Task(
            description=task_description,
            agent=self.agent,
            expected_output="구조화된 답변과 메타데이터를 포함한 완전한 응답"
        )
        
        return task
    
    def generate_answer(self, question: str, context_documents: List[str], question_type: str = "general") -> Dict[str, Any]:
        """
        답변을 생성합니다
        
        Args:
            question (str): 사용자 질문
            context_documents (List[str]): 참고 문서들
            question_type (str): 질문 유형
            
        Returns:
            Dict[str, Any]: 생성된 답변
        """
        log_info(f"답변 생성 시작: '{question[:50]}...'")
        
        try:
            # 답변 생성
            answer_data = self.answer_tool._run(question, context_documents, question_type)
            
            # 답변 포맷팅
            formatted_response = self.formatter_tool._run(answer_data)
            
            log_success("답변 생성 및 포맷팅 완료")
            return formatted_response
            
        except Exception as e:
            log_error(f"답변 생성 실패: {str(e)}")
            return {
                "question": question,
                "answer": f"죄송합니다. 답변 생성 중 오류가 발생했습니다: {str(e)}",
                "error": str(e),
                "metadata": {
                    "confidence": "low",
                    "sources_used": 0
                }
            }
    
    def get_simple_answer(self, question: str, context_documents: List[str]) -> str:
        """
        간단한 텍스트 답변만 반환합니다
        
        Args:
            question (str): 사용자 질문
            context_documents (List[str]): 참고 문서들
            
        Returns:
            str: 답변 텍스트
        """
        result = self.generate_answer(question, context_documents)
        return result.get("answer", "답변을 생성할 수 없습니다.")
    
    def evaluate_answer_quality(self, answer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        답변 품질을 평가합니다
        
        Args:
            answer_data (Dict[str, Any]): 답변 데이터
            
        Returns:
            Dict[str, Any]: 품질 평가 결과
        """
        answer = answer_data.get("answer", "")
        metadata = answer_data.get("metadata", {})
        
        evaluation = {
            "length_score": min(len(answer) / 200, 1.0),  # 적절한 길이
            "confidence_score": {
                "high": 1.0,
                "medium": 0.7,
                "low": 0.3
            }.get(metadata.get("confidence", "low"), 0.3),
            "sources_score": min(metadata.get("sources_used", 0) / 3, 1.0),  # 충분한 소스
            "overall_quality": "good" if metadata.get("confidence") == "high" else "fair"
        }
        
        evaluation["total_score"] = (
            evaluation["length_score"] * 0.3 +
            evaluation["confidence_score"] * 0.5 +
            evaluation["sources_score"] * 0.2
        )
        
        return evaluation