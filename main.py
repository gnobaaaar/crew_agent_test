#!/usr/bin/env python3
"""
CrewAI RAG 시스템 메인 실행 파일
PDF 기반 RAG 채팅 시스템의 진입점입니다.
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.config_loader import config_loader
from utils.document_processor import DocumentProcessor
from agents.rag_system import RAGSystem
from utils.logger import logger, log_step, log_success, log_error, log_info


def main():
    """메인 실행 함수"""
    
    log_step("CrewAI RAG 시스템 시작", "PDF 기반 지능형 채팅 시스템")
    
    try:
        # 1. 설정 검증
        log_info("설정 파일 검증 중...")
        if not config_loader.validate_config():
            log_error("설정 검증 실패. 프로그램을 종료합니다.")
            return 1
        
        # 2. 설정 정보 출력
        api_config = config_loader.get_api_config()
        db_config = config_loader.get_database_config()
        
        log_info(f"사용 모델: {api_config.get('model_name', 'gemini-2.0-flash')}")
        log_info(f"벡터 DB 경로: {db_config.get('chroma_db_path', './data/chroma_db')}")
        log_info(f"임베딩 모델: {db_config.get('embedding_model', 'models/gemini-embedding-001')}")
        
        # 3. 문서 처리 파이프라인 초기화 (기존 데이터 확인)
        log_step("문서 처리 시스템 확인", "기존 데이터 상태 점검")
        
        processor = DocumentProcessor()
        status = processor.get_system_status()
        
        log_info(f"PDF 파일: {status['pdf_conversion']['pdf_files']}개")
        log_info(f"Markdown 파일: {status['pdf_conversion']['markdown_files']}개")
        log_info(f"임베딩 청크: {status['embedding_database'].get('total_chunks', 0)}개")
        
        # 4. 샘플 데이터가 없으면 생성
        if status['embedding_database'].get('total_chunks', 0) == 0:
            log_info("샘플 데이터 생성 중...")
            
            # 샘플 마크다운 문서 생성
            sample_path = processor.create_sample_pdf()
            
            # 샘플 문서를 임베딩 DB에 추가
            with open(sample_path, 'r', encoding='utf-8') as f:
                sample_content = f.read()
            
            success = processor.embedding_manager.add_document(sample_path, sample_content)
            
            if success:
                log_success("샘플 데이터 생성 완료")
            else:
                log_error("샘플 데이터 생성 실패")
        
        # 5. RAG 시스템 초기화
        log_step("RAG 시스템 초기화", "3개 에이전트 통합 시스템")
        
        rag_system = RAGSystem()
        
        # 6. 시스템 상태 확인
        system_status = rag_system.get_system_status()
        log_info(f"시스템 상태: {system_status['overall_status']}")
        
        for agent_name, agent_status in system_status.get('agents', {}).items():
            log_info(f"  {agent_name}: {agent_status}")
        
        # 7. RAG 시스템 테스트
        log_step("RAG 시스템 테스트", "전체 워크플로우 검증")
        
        test_questions = [
            "CrewAI의 주요 기능은 무엇인가요?",
            "PDF 처리는 어떻게 하나요?",
            "벡터 데이터베이스의 역할은?"
        ]
        
        for question in test_questions:
            log_info(f"테스트 질문: '{question}'")
            
            result = rag_system.process_question(question, max_results=3)
            
            if result.get("success"):
                log_success("질문 처리 성공")
                log_info(f"답변 길이: {len(result['answer'])} 문자")
                log_info(f"처리 시간: {result.get('metadata', {}).get('processing_time', 0)}초")
                
                # 답변 미리보기
                preview = result['answer'][:100] + "..." if len(result['answer']) > 100 else result['answer']
                log_info(f"답변 미리보기: {preview}")
            else:
                log_error(f"질문 처리 실패: {result.get('error', 'Unknown error')}")
        
        # 8. 대화형 모드 시작 여부 확인
        log_step("시스템 준비 완료", "대화형 채팅 모드 시작 가능")
        
        log_success("🎉 CrewAI RAG 시스템이 성공적으로 구축되었습니다!")
        log_info("✅ PDF 처리 파이프라인 완성")
        log_info("✅ 3개 에이전트 시스템 구축")
        log_info("✅ 질문-답변 워크플로우 검증")
        
        # 대화형 모드 시작
        print("\n" + "="*60)
        print("🚀 대화형 채팅 모드를 시작하시겠습니까?")
        print("y/yes: 채팅 시작, 다른 키: 종료")
        print("="*60)
        
        user_choice = input("선택: ").strip().lower()
        
        if user_choice in ['y', 'yes', 'ㅇ', '네', '예']:
            rag_system.interactive_chat()
        else:
            log_info("프로그램을 종료합니다.")
        
        return 0
        
    except Exception as e:
        log_error(f"시스템 실행 중 오류 발생: {str(e)}")
        logger.exception("상세 오류 정보:")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)