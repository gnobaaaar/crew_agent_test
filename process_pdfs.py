#!/usr/bin/env python3
"""
PDF 파일들을 수동으로 처리하는 스크립트
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.document_processor import DocumentProcessor
from utils.logger import logger, log_step, log_success, log_error, log_info


def main():
    """PDF 파일들을 처리합니다"""
    
    log_step("PDF 파일 처리", "data/pdfs/ 폴더의 모든 PDF 파일 처리")
    
    try:
        # DocumentProcessor 초기화
        processor = DocumentProcessor()
        
        # 모든 PDF 파일 처리
        result = processor.process_all_documents()
        
        log_info(f"처리 결과:")
        log_info(f"  전체 파일: {result['total_files']}개")
        log_info(f"  성공: {result['processed_files']}개")
        log_info(f"  실패: {result['failed_files']}개")
        log_info(f"  성공률: {result['success_rate']:.1f}%")
        
        if result['failed_file_names']:
            log_error(f"실패한 파일들: {', '.join(result['failed_file_names'])}")
        
        # 최종 상태 확인
        status = processor.get_system_status()
        log_success(f"시스템 상태 업데이트 완료")
        log_info(f"총 임베딩 청크: {status['embedding_database'].get('total_chunks', 0)}개")
        
        return 0
        
    except Exception as e:
        log_error(f"PDF 처리 중 오류 발생: {str(e)}")
        logger.exception("상세 오류 정보:")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)