"""
지식 검색 에이전트 (Knowledge Retriever)
벡터 데이터베이스에서 관련 문서를 검색하고 정보를 수집합니다.
"""

from crewai import Agent, Task
from crewai.tools import BaseTool
from typing import Dict, Any, List
import json

from utils.config_loader import config_loader
from utils.document_processor import DocumentProcessor
from utils.logger import logger, log_info, log_success, log_error, log_warning


class DocumentSearchTool(BaseTool):
    """문서 검색 도구"""
    
    name: str = "document_search"
    description: str = "벡터 데이터베이스에서 관련 문서를 검색합니다"
    
    def _run(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        문서를 검색합니다
        
        Args:
            query (str): 검색 쿼리
            n_results (int): 반환할 결과 수
            
        Returns:
            Dict[str, Any]: 검색 결과
        """
        try:
            log_info(f"문서 검색 실행: '{query[:50]}...'")
            
            # DocumentProcessor 인스턴스 생성
            processor = DocumentProcessor()
            
            # 문서 검색
            results = processor.search_documents(query, n_results)
            
            # 결과 포맷팅
            formatted_results = {
                "query": query,
                "total_results": results["total_results"],
                "search_results": []
            }
            
            for result in results["results"]:
                formatted_result = {
                    "rank": result["rank"],
                    "content": result["full_content"],
                    "preview": result["content"],
                    "source_file": result["source_file"],
                    "similarity_score": result["similarity_score"],
                    "chunk_id": result["chunk_id"]
                }
                formatted_results["search_results"].append(formatted_result)
            
            log_success(f"문서 검색 완료: {len(formatted_results['search_results'])}개 결과")
            return formatted_results
            
        except Exception as e:
            log_error(f"문서 검색 실패: {str(e)}")
            return {
                "query": query,
                "total_results": 0,
                "search_results": [],
                "error": str(e)
            }


class MultiQuerySearchTool(BaseTool):
    """다중 쿼리 검색 도구"""
    
    name: str = "multi_query_search"
    description: str = "여러 검색 쿼리를 사용하여 포괄적인 문서 검색을 수행합니다"
    
    def _run(self, queries: List[str], n_results_per_query: int = 3) -> Dict[str, Any]:
        """
        여러 쿼리로 문서를 검색합니다
        
        Args:
            queries (List[str]): 검색 쿼리 목록
            n_results_per_query (int): 쿼리당 결과 수
            
        Returns:
            Dict[str, Any]: 통합 검색 결과
        """
        try:
            log_info(f"다중 쿼리 검색 시작: {len(queries)}개 쿼리")
            
            # DocumentSearchTool 인스턴스 생성
            search_tool = DocumentSearchTool()
            
            all_results = []
            seen_content = set()  # 중복 제거용
            
            for query in queries:
                search_result = search_tool._run(query, n_results_per_query)
                
                for result in search_result.get("search_results", []):
                    content_hash = hash(result["content"][:100])  # 내용 일부로 해시 생성
                    
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        result["source_query"] = query  # 어떤 쿼리에서 나온 결과인지 표시
                        all_results.append(result)
            
            # 유사도 점수로 정렬
            all_results.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            # 상위 결과만 선택
            top_results = all_results[:10]
            
            combined_result = {
                "queries": queries,
                "total_unique_results": len(top_results),
                "search_results": top_results
            }
            
            log_success(f"다중 쿼리 검색 완료: {len(top_results)}개 고유 결과")
            return combined_result
            
        except Exception as e:
            log_error(f"다중 쿼리 검색 실패: {str(e)}")
            return {
                "queries": queries,
                "total_unique_results": 0,
                "search_results": [],
                "error": str(e)
            }


class KnowledgeRetrieverAgent:
    """지식 검색 에이전트 클래스"""
    
    def __init__(self):
        """KnowledgeRetrieverAgent 초기화"""
        
        # 설정 로드
        agents_config = config_loader.get_agents_config()
        retriever_config = agents_config.get("knowledge_retriever", {})
        
        # 도구 초기화
        self.search_tool = DocumentSearchTool()
        self.multi_search_tool = MultiQuerySearchTool()
        
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
            role=retriever_config.get("role", "지식 검색 전문가"),
            goal=retriever_config.get("goal", "벡터 데이터베이스에서 관련 문서를 검색하고 정보를 수집합니다"),
            backstory=retriever_config.get("backstory", "지식 검색 전문가입니다"),
            verbose=retriever_config.get("verbose", True),
            allow_delegation=retriever_config.get("allow_delegation", False),
            tools=[self.search_tool, self.multi_search_tool],
            llm=gemini_llm
        )
        
        log_success("지식 검색 에이전트 초기화 완료")
    
    def create_search_task(self, queries: List[str]) -> Task:
        """
        검색 태스크를 생성합니다
        
        Args:
            queries (List[str]): 검색 쿼리 목록
            
        Returns:
            Task: CrewAI 태스크 객체
        """
        queries_str = ", ".join([f'"{q}"' for q in queries])
        
        task_description = f"""
        다음 검색 쿼리들을 사용하여 관련 문서를 검색해주세요:
        
        검색 쿼리: {queries_str}
        
        수행해야 할 작업:
        1. 각 쿼리로 문서 검색 실행
        2. 검색 결과의 관련성 평가
        3. 중복 제거 및 결과 통합
        4. 유사도 점수 기준으로 정렬
        
        가장 관련성 높은 문서들을 선별하여 반환해주세요.
        """
        
        task = Task(
            description=task_description,
            agent=self.agent,
            expected_output="검색된 문서들의 내용과 메타데이터를 포함한 구조화된 결과"
        )
        
        return task
    
    def search_documents(self, queries: List[str], max_results: int = 10) -> Dict[str, Any]:
        """
        문서를 검색합니다
        
        Args:
            queries (List[str]): 검색 쿼리 목록
            max_results (int): 최대 결과 수
            
        Returns:
            Dict[str, Any]: 검색 결과
        """
        log_info(f"문서 검색 시작: {len(queries)}개 쿼리")
        
        try:
            if len(queries) == 1:
                # 단일 쿼리 검색
                result = self.search_tool._run(queries[0], max_results)
            else:
                # 다중 쿼리 검색
                result = self.multi_search_tool._run(queries, max_results // len(queries))
            
            log_success("문서 검색 완료")
            return result
            
        except Exception as e:
            log_error(f"문서 검색 실패: {str(e)}")
            return {
                "queries": queries,
                "total_unique_results": 0,
                "search_results": [],
                "error": str(e)
            }
    
    def get_relevant_content(self, queries: List[str], max_results: int = 5) -> List[str]:
        """
        관련 문서 내용을 텍스트 리스트로 반환합니다
        
        Args:
            queries (List[str]): 검색 쿼리 목록
            max_results (int): 최대 결과 수
            
        Returns:
            List[str]: 관련 문서 내용 리스트
        """
        search_result = self.search_documents(queries, max_results)
        
        contents = []
        for result in search_result.get("search_results", []):
            content = result.get("content", "")
            if content and content not in contents:  # 중복 제거
                contents.append(content)
        
        return contents
    
    def get_search_summary(self, queries: List[str]) -> Dict[str, Any]:
        """
        검색 결과 요약을 반환합니다
        
        Args:
            queries (List[str]): 검색 쿼리 목록
            
        Returns:
            Dict[str, Any]: 검색 요약
        """
        search_result = self.search_documents(queries)
        
        if not search_result.get("search_results"):
            return {
                "queries": queries,
                "found_results": False,
                "summary": "관련 문서를 찾을 수 없습니다.",
                "suggestions": ["다른 키워드로 검색해보세요", "질문을 더 구체적으로 작성해보세요"]
            }
        
        results = search_result["search_results"]
        
        # 요약 정보 생성
        summary = {
            "queries": queries,
            "found_results": True,
            "total_results": len(results),
            "best_match_score": results[0]["similarity_score"] if results else 0,
            "source_files": list(set([r["source_file"] for r in results])),
            "content_preview": results[0]["preview"] if results else "",
            "all_contents": [r["content"] for r in results]
        }
        
        return summary
    
    def check_database_status(self) -> Dict[str, Any]:
        """
        데이터베이스 상태를 확인합니다
        
        Returns:
            Dict[str, Any]: 데이터베이스 상태 정보
        """
        try:
            processor = DocumentProcessor()
            status = processor.get_system_status()
            
            db_status = {
                "database_ready": True,
                "total_chunks": status["embedding_database"].get("total_chunks", 0),
                "unique_documents": status["embedding_database"].get("unique_documents", 0),
                "collection_name": status["embedding_database"].get("collection_name", ""),
                "embedding_model": status["embedding_database"].get("embedding_model", "")
            }
            
            if db_status["total_chunks"] == 0:
                db_status["database_ready"] = False
                db_status["message"] = "데이터베이스가 비어있습니다. 문서를 먼저 추가해주세요."
            
            return db_status
            
        except Exception as e:
            log_error(f"데이터베이스 상태 확인 실패: {str(e)}")
            return {
                "database_ready": False,
                "error": str(e),
                "message": "데이터베이스 연결에 실패했습니다."
            }