"""
문서 처리 파이프라인
PDF → Markdown → 임베딩 → Chroma DB 전체 과정을 관리합니다.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional

from .pdf_converter import PDFConverter
from .embedding_manager import EmbeddingManager
from .config_loader import config_loader
from .logger import logger, log_step, log_success, log_warning, log_error, log_info


class DocumentProcessor:
    """문서 처리 파이프라인 관리 클래스"""
    
    def __init__(self):
        """DocumentProcessor 초기화"""
        
        # 설정 로드
        db_config = config_loader.get_database_config()
        paths_config = config_loader.get_paths_config()
        processing_config = config_loader.get_config("config").get("processing", {})
        
        # PDF 변환기 초기화
        self.pdf_converter = PDFConverter(
            input_dir=paths_config.get("pdf_input", "./data/pdfs"),
            output_dir=paths_config.get("markdown_output", "./data/markdown")
        )
        
        # 임베딩 관리자 초기화
        self.embedding_manager = EmbeddingManager(
            db_path=db_config.get("chroma_db_path", "./data/chroma_db"),
            collection_name=db_config.get("collection_name", "rag_documents"),
            embedding_model=db_config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
            chunk_size=processing_config.get("chunk_size", 1000),
            chunk_overlap=processing_config.get("chunk_overlap", 200)
        )
        
        log_success("문서 처리 파이프라인 초기화 완료")
    
    def process_single_document(self, pdf_path: Path) -> bool:
        """
        단일 PDF 문서를 전체 파이프라인으로 처리합니다
        
        Args:
            pdf_path (Path): 처리할 PDF 파일 경로
            
        Returns:
            bool: 처리 성공 여부
        """
        log_step(f"문서 처리 시작", f"파일: {pdf_path.name}")
        
        try:
            # 1. PDF → Markdown 변환
            log_info("1단계: PDF → Markdown 변환")
            markdown_path = self.pdf_converter.convert_single_pdf(pdf_path)
            
            if not markdown_path:
                log_error("PDF 변환 실패")
                return False
            
            # 2. Markdown 내용 읽기
            log_info("2단계: Markdown 내용 로드")
            with open(markdown_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                log_error("Markdown 내용이 비어있습니다")
                return False
            
            log_info(f"문서 길이: {len(content):,} 문자")
            
            # 3. 임베딩 생성 및 DB 저장
            log_info("3단계: 임베딩 생성 및 DB 저장")
            success = self.embedding_manager.add_document(pdf_path, content)
            
            if not success:
                log_error("임베딩 저장 실패")
                return False
            
            log_success(f"문서 처리 완료: {pdf_path.name}")
            return True
            
        except Exception as e:
            log_error(f"문서 처리 중 오류 발생: {str(e)}")
            return False
    
    def process_all_documents(self) -> Dict[str, Any]:
        """
        모든 PDF 문서를 처리합니다
        
        Returns:
            Dict[str, Any]: 처리 결과 통계
        """
        log_step("전체 문서 처리 시작", "PDF 디렉토리의 모든 파일 처리")
        
        # PDF 파일 목록 가져오기
        pdf_files = list(self.pdf_converter.input_dir.glob("*.pdf"))
        
        if not pdf_files:
            log_warning(f"처리할 PDF 파일이 없습니다: {self.pdf_converter.input_dir}")
            return {
                "total_files": 0,
                "processed_files": 0,
                "failed_files": 0,
                "success_rate": 0
            }
        
        log_info(f"발견된 PDF 파일: {len(pdf_files)}개")
        
        # 각 파일 처리
        processed_count = 0
        failed_files = []
        
        for pdf_file in pdf_files:
            success = self.process_single_document(pdf_file)
            if success:
                processed_count += 1
            else:
                failed_files.append(pdf_file.name)
        
        # 결과 통계
        success_rate = (processed_count / len(pdf_files)) * 100 if pdf_files else 0
        
        result = {
            "total_files": len(pdf_files),
            "processed_files": processed_count,
            "failed_files": len(failed_files),
            "failed_file_names": failed_files,
            "success_rate": success_rate
        }
        
        # 결과 출력
        log_success(f"전체 처리 완료: {processed_count}/{len(pdf_files)} 파일 성공 ({success_rate:.1f}%)")
        
        if failed_files:
            log_warning(f"실패한 파일들: {', '.join(failed_files)}")
        
        return result
    
    def search_documents(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        문서에서 관련 내용을 검색합니다
        
        Args:
            query (str): 검색 쿼리
            n_results (int): 반환할 결과 수
            
        Returns:
            Dict[str, Any]: 검색 결과
        """
        log_info(f"문서 검색: '{query[:50]}...'")
        
        results = self.embedding_manager.search_similar(query, n_results)
        
        # 결과 포맷팅
        formatted_results = []
        
        if results["documents"] and results["documents"][0]:
            for i, (doc, metadata, distance) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )):
                formatted_result = {
                    "rank": i + 1,
                    "content": doc[:200] + "..." if len(doc) > 200 else doc,
                    "full_content": doc,
                    "source_file": metadata.get("file_name", "Unknown"),
                    "chunk_id": metadata.get("chunk_id", 0),
                    "similarity_score": 1 - distance,  # 거리를 유사도로 변환
                    "metadata": metadata
                }
                formatted_results.append(formatted_result)
        
        search_result = {
            "query": query,
            "total_results": len(formatted_results),
            "results": formatted_results
        }
        
        log_success(f"검색 완료: {len(formatted_results)}개 결과")
        return search_result
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        시스템 상태 정보를 반환합니다
        
        Returns:
            Dict[str, Any]: 시스템 상태
        """
        # PDF 변환 통계
        conversion_stats = self.pdf_converter.get_conversion_stats()
        
        # 임베딩 DB 통계
        collection_stats = self.embedding_manager.get_collection_stats()
        
        return {
            "pdf_conversion": conversion_stats,
            "embedding_database": collection_stats,
            "pipeline_status": "ready"
        }
    
    def reset_system(self) -> bool:
        """
        시스템을 초기화합니다 (임베딩 DB 리셋)
        
        Returns:
            bool: 성공 여부
        """
        log_warning("시스템 초기화 시작...")
        
        success = self.embedding_manager.reset_collection()
        
        if success:
            log_success("시스템 초기화 완료")
        else:
            log_error("시스템 초기화 실패")
        
        return success
    
    def create_sample_pdf(self) -> Path:
        """
        테스트용 샘플 PDF를 생성합니다 (텍스트 파일로 대체)
        
        Returns:
            Path: 생성된 샘플 파일 경로
        """
        sample_content = """# CrewAI RAG 시스템 테스트 문서

## 개요
이 문서는 CrewAI RAG 시스템의 PDF 처리 파이프라인을 테스트하기 위한 샘플 문서입니다.

## 주요 기능
1. PDF 텍스트 추출
2. 마크다운 변환
3. 텍스트 청킹
4. 임베딩 생성
5. 벡터 데이터베이스 저장

## 기술 스택
- CrewAI: 멀티 에이전트 프레임워크
- Chroma DB: 벡터 데이터베이스
- Sentence Transformers: 임베딩 모델
- PyMuPDF: PDF 처리

## 테스트 시나리오
이 문서를 통해 다음과 같은 기능들을 테스트할 수 있습니다:
- 한국어 텍스트 처리
- 긴 문서의 청킹
- 유사도 검색
- 메타데이터 관리

## 결론
CrewAI RAG 시스템은 PDF 문서를 효과적으로 처리하고 검색 가능한 형태로 변환합니다.
"""
        
        sample_path = self.pdf_converter.output_dir / "sample_test.md"
        
        with open(sample_path, 'w', encoding='utf-8') as f:
            f.write(sample_content)
        
        log_success(f"샘플 문서 생성: {sample_path}")
        return sample_path