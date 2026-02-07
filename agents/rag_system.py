"""
RAG 시스템 통합 관리자
3개 에이전트를 조율하여 완전한 RAG 워크플로우를 실행합니다.
"""

from typing import Dict, Any, List
import time

from .question_collector import QuestionCollectorAgent
from .knowledge_retriever import KnowledgeRetrieverAgent
from .response_generator import ResponseGeneratorAgent

from utils.logger import logger, log_step, log_success, log_error, log_info, log_warning


class RAGSystem:
    """RAG 시스템 통합 관리 클래스"""
    
    def __init__(self):
        """RAGSystem 초기화"""
        
        log_step("RAG 시스템 초기화", "3개 에이전트 통합 시스템")
        
        try:
            # 각 에이전트 초기화
            log_info("에이전트들 초기화 중...")
            
            self.question_collector = QuestionCollectorAgent()
            self.knowledge_retriever = KnowledgeRetrieverAgent()
            self.response_generator = ResponseGeneratorAgent()
            
            log_success("RAG 시스템 초기화 완료")
            
        except Exception as e:
            log_error(f"RAG 시스템 초기화 실패: {str(e)}")
            raise
    
    def process_question(self, question: str, max_results: int = 5) -> Dict[str, Any]:
        """
        질문을 전체 RAG 파이프라인으로 처리합니다
        
        Args:
            question (str): 사용자 질문
            max_results (int): 검색할 최대 문서 수
            
        Returns:
            Dict[str, Any]: 완전한 RAG 응답
        """
        start_time = time.time()
        
        log_step("RAG 질문 처리 시작", f"질문: '{question[:50]}...'")
        
        try:
            # 1단계: 질문 분석
            log_info("1단계: 질문 분석 및 키워드 추출")
            question_analysis = self.question_collector.analyze_question(question)
            
            if "error" in question_analysis:
                log_warning(f"질문 분석 중 오류: {question_analysis['error']}")
            
            search_queries = question_analysis.get("search_queries", [question])
            question_type = question_analysis.get("question_type", "general")
            keywords = question_analysis.get("keywords", [])
            
            log_info(f"추출된 키워드: {', '.join(keywords[:5])}")
            log_info(f"질문 유형: {question_type}")
            log_info(f"검색 쿼리: {len(search_queries)}개")
            
            # 2단계: 문서 검색
            log_info("2단계: 관련 문서 검색")
            
            # 데이터베이스 상태 확인
            db_status = self.knowledge_retriever.check_database_status()
            
            if not db_status.get("database_ready", False):
                log_warning("데이터베이스가 준비되지 않음")
                return self._create_no_database_response(question, question_analysis, db_status)
            
            # 문서 검색 실행
            search_results = self.knowledge_retriever.search_documents(search_queries, max_results)
            
            if search_results.get("total_unique_results", 0) == 0:
                log_warning("관련 문서를 찾을 수 없음")
                return self._create_no_results_response(question, question_analysis, search_results)
            
            # 검색된 문서 내용 추출
            context_documents = self.knowledge_retriever.get_relevant_content(search_queries, max_results)
            
            log_info(f"검색된 문서: {len(context_documents)}개")
            
            # 3단계: 답변 생성
            log_info("3단계: 답변 생성")
            
            final_answer = self.response_generator.generate_answer(
                question=question,
                context_documents=context_documents,
                question_type=question_type
            )
            
            # 4단계: 결과 통합
            processing_time = time.time() - start_time
            
            complete_response = {
                "question": question,
                "answer": final_answer.get("answer", "답변을 생성할 수 없습니다."),
                "metadata": {
                    "processing_time": round(processing_time, 2),
                    "question_analysis": question_analysis,
                    "search_results": {
                        "total_results": search_results.get("total_unique_results", 0),
                        "queries_used": search_queries,
                        "sources_found": len(context_documents)
                    },
                    "answer_metadata": final_answer.get("metadata", {}),
                    "suggestions": final_answer.get("suggestions", [])
                },
                "success": True
            }
            
            log_success(f"RAG 처리 완료 ({processing_time:.2f}초)")
            return complete_response
            
        except Exception as e:
            processing_time = time.time() - start_time
            log_error(f"RAG 처리 실패: {str(e)}")
            
            return {
                "question": question,
                "answer": f"죄송합니다. 질문 처리 중 오류가 발생했습니다: {str(e)}",
                "error": str(e),
                "metadata": {
                    "processing_time": round(processing_time, 2),
                    "success": False
                },
                "success": False
            }
    
    def _create_no_database_response(self, question: str, question_analysis: Dict[str, Any], db_status: Dict[str, Any]) -> Dict[str, Any]:
        """데이터베이스가 비어있을 때의 응답을 생성합니다"""
        
        return {
            "question": question,
            "answer": "죄송합니다. 현재 검색할 수 있는 문서가 없습니다. PDF 파일을 먼저 업로드하고 처리해주세요.",
            "metadata": {
                "question_analysis": question_analysis,
                "database_status": db_status,
                "suggestions": [
                    "PDF 파일을 data/pdfs/ 폴더에 추가해주세요",
                    "문서 처리 파이프라인을 실행해주세요",
                    "시스템 상태를 확인해주세요"
                ]
            },
            "success": False,
            "no_database": True
        }
    
    def _create_no_results_response(self, question: str, question_analysis: Dict[str, Any], search_results: Dict[str, Any]) -> Dict[str, Any]:
        """검색 결과가 없을 때의 응답을 생성합니다"""
        
        # 컨텍스트 없이 일반적인 답변 시도
        general_answer = self.response_generator.generate_answer(
            question=question,
            context_documents=[],
            question_type=question_analysis.get("question_type", "general")
        )
        
        return {
            "question": question,
            "answer": general_answer.get("answer", "관련 정보를 찾을 수 없어 답변을 제공할 수 없습니다."),
            "metadata": {
                "question_analysis": question_analysis,
                "search_results": search_results,
                "answer_metadata": general_answer.get("metadata", {}),
                "suggestions": [
                    "다른 키워드로 질문해보세요",
                    "더 구체적인 질문을 해보세요",
                    "관련 문서를 추가로 업로드해보세요"
                ]
            },
            "success": True,
            "no_results": True
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        RAG 시스템 전체 상태를 반환합니다
        
        Returns:
            Dict[str, Any]: 시스템 상태 정보
        """
        try:
            # 데이터베이스 상태 확인
            db_status = self.knowledge_retriever.check_database_status()
            
            # 각 에이전트 상태 확인
            agents_status = {
                "question_collector": "ready",
                "knowledge_retriever": "ready" if db_status.get("database_ready") else "no_data",
                "response_generator": "ready"
            }
            
            system_status = {
                "overall_status": "ready" if all(status == "ready" for status in agents_status.values()) else "partial",
                "agents": agents_status,
                "database": db_status,
                "capabilities": {
                    "question_analysis": True,
                    "document_search": db_status.get("database_ready", False),
                    "answer_generation": True
                }
            }
            
            return system_status
            
        except Exception as e:
            log_error(f"시스템 상태 확인 실패: {str(e)}")
            return {
                "overall_status": "error",
                "error": str(e)
            }
    
    def test_system(self, test_question: str = "CrewAI는 무엇인가요?") -> Dict[str, Any]:
        """
        시스템 전체 기능을 테스트합니다
        
        Args:
            test_question (str): 테스트용 질문
            
        Returns:
            Dict[str, Any]: 테스트 결과
        """
        log_step("RAG 시스템 테스트", f"테스트 질문: '{test_question}'")
        
        try:
            # 시스템 상태 확인
            status = self.get_system_status()
            
            if status["overall_status"] == "error":
                return {
                    "test_passed": False,
                    "error": "시스템 상태 확인 실패",
                    "status": status
                }
            
            # 테스트 질문 처리
            test_result = self.process_question(test_question)
            
            # 테스트 결과 평가
            test_evaluation = {
                "test_passed": test_result.get("success", False),
                "test_question": test_question,
                "response_generated": bool(test_result.get("answer")),
                "processing_time": test_result.get("metadata", {}).get("processing_time", 0),
                "agents_working": {
                    "question_collector": bool(test_result.get("metadata", {}).get("question_analysis")),
                    "knowledge_retriever": test_result.get("metadata", {}).get("search_results", {}).get("total_results", 0) > 0,
                    "response_generator": bool(test_result.get("answer"))
                },
                "full_response": test_result,
                "system_status": status
            }
            
            if test_evaluation["test_passed"]:
                log_success("RAG 시스템 테스트 통과")
            else:
                log_warning("RAG 시스템 테스트 부분 통과")
            
            return test_evaluation
            
        except Exception as e:
            log_error(f"RAG 시스템 테스트 실패: {str(e)}")
            return {
                "test_passed": False,
                "error": str(e),
                "test_question": test_question
            }
    
    def interactive_chat(self):
        """
        대화형 채팅 인터페이스를 시작합니다
        """
        log_step("대화형 RAG 채팅 시작", "질문을 입력하세요 (종료: 'quit' 또는 'exit')")
        
        print("\n" + "="*60)
        print("🤖 CrewAI RAG 채팅 시스템")
        print("질문을 입력하세요. 종료하려면 'quit' 또는 'exit'를 입력하세요.")
        print("="*60 + "\n")
        
        while True:
            try:
                # 사용자 입력 받기
                user_input = input("\n💬 질문: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', '종료', '나가기']:
                    print("\n👋 채팅을 종료합니다. 감사합니다!")
                    break
                
                # 특별 명령어 처리
                if user_input.lower() in ['status', '상태']:
                    status = self.get_system_status()
                    print(f"\n📊 시스템 상태: {status['overall_status']}")
                    print(f"📚 데이터베이스: {status['database'].get('total_chunks', 0)}개 청크")
                    continue
                
                # 질문 처리
                print("\n🔍 처리 중...")
                response = self.process_question(user_input)
                
                # 답변 출력 (마크다운 형식)
                print(f"\n🤖 답변:")
                print("=" * 60)
                
                # 마크다운을 콘솔에서 보기 좋게 출력
                self._print_markdown(response["answer"])
                
                print("=" * 60)
                
                # 메타데이터 출력 (간단히)
                metadata = response.get("metadata", {})
                if metadata:
                    print(f"⏱️  처리 시간: {metadata.get('processing_time', 0)}초")
                    search_info = metadata.get("search_results", {})
                    if search_info:
                        print(f"📄 검색된 문서: {search_info.get('sources_found', 0)}개")
                
            except KeyboardInterrupt:
                print("\n\n👋 채팅을 종료합니다.")
                break
            except Exception as e:
                print(f"\n❌ 오류 발생: {str(e)}")
                continue
    
    def _print_markdown(self, markdown_text: str):
        """
        마크다운 텍스트를 콘솔에서 보기 좋게 출력합니다
        
        Args:
            markdown_text (str): 마크다운 형식의 텍스트
        """
        try:
            import re
            
            lines = markdown_text.split('\n')
            
            for line in lines:
                # 헤더 처리
                if line.startswith('## '):
                    header_text = line[3:].strip()
                    print(f"\n{header_text}")
                    print("─" * min(50, len(header_text.encode('utf-8'))))
                elif line.startswith('# '):
                    header_text = line[2:].strip()
                    print(f"\n{header_text}")
                    print("═" * min(50, len(header_text.encode('utf-8'))))
                
                # 구분선 처리
                elif line.strip() == '---':
                    print("\n" + "─" * 50)
                
                # 볼드 텍스트 처리
                elif '**' in line:
                    # **텍스트** -> 텍스트 (굵게 표시 대신 그대로)
                    formatted_line = re.sub(r'\*\*(.*?)\*\*', r'[\1]', line)
                    print(formatted_line)
                
                # 코드 블록 처리
                elif '`' in line:
                    # `코드` -> [코드]
                    formatted_line = re.sub(r'`(.*?)`', r'[\1]', line)
                    print(formatted_line)
                
                # 일반 텍스트
                else:
                    if line.strip():  # 빈 줄이 아닌 경우만
                        print(line)
                    else:
                        print()  # 빈 줄 유지
                        
        except Exception as e:
            # 마크다운 파싱 실패 시 원본 텍스트 출력
            print(markdown_text)