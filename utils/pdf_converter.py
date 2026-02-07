"""
PDF → Markdown 변환 유틸리티
PDF 파일을 읽어서 마크다운 형식으로 변환합니다.
"""

import os
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Optional
import re

from .logger import logger, log_info, log_success, log_warning, log_error


class PDFConverter:
    """PDF를 Markdown으로 변환하는 클래스"""
    
    def __init__(self, input_dir: str = "./data/pdfs", output_dir: str = "./data/markdown"):
        """
        PDFConverter 초기화
        
        Args:
            input_dir (str): PDF 파일이 있는 디렉토리
            output_dir (str): 변환된 마크다운을 저장할 디렉토리
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # 출력 디렉토리 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        log_info(f"PDF 입력 디렉토리: {self.input_dir}")
        log_info(f"Markdown 출력 디렉토리: {self.output_dir}")
    
    def extract_text_from_pdf(self, pdf_path: Path) -> List[Dict[str, str]]:
        """
        PDF 파일에서 텍스트를 추출합니다
        
        Args:
            pdf_path (Path): PDF 파일 경로
            
        Returns:
            List[Dict[str, str]]: 페이지별 텍스트 정보
        """
        pages_content = []
        
        try:
            # PDF 문서 열기
            doc = fitz.open(pdf_path)
            log_info(f"PDF 파일 열기 성공: {pdf_path.name} ({len(doc)} 페이지)")
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # 텍스트 추출
                text = page.get_text()
                
                # 빈 페이지 건너뛰기
                if not text.strip():
                    log_warning(f"페이지 {page_num + 1}: 빈 페이지 건너뛰기")
                    continue
                
                # 페이지 정보 저장
                page_info = {
                    "page_number": page_num + 1,
                    "text": text.strip(),
                    "char_count": len(text.strip())
                }
                
                pages_content.append(page_info)
                log_info(f"페이지 {page_num + 1}: {len(text.strip())} 문자 추출")
            
            doc.close()
            log_success(f"PDF 텍스트 추출 완료: {len(pages_content)} 페이지")
            
        except Exception as e:
            log_error(f"PDF 텍스트 추출 실패: {str(e)}")
            return []
        
        return pages_content
    
    def clean_text(self, text: str) -> str:
        """
        추출된 텍스트를 정리합니다
        
        Args:
            text (str): 원본 텍스트
            
        Returns:
            str: 정리된 텍스트
        """
        # 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        # 연속된 줄바꿈을 단일 줄바꿈으로
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # 특수 문자 정리 (선택적)
        text = text.replace('\x0c', '')  # 폼 피드 제거
        text = text.replace('\r', '\n')  # 캐리지 리턴을 줄바꿈으로
        
        return text.strip()
    
    def convert_to_markdown(self, pages_content: List[Dict[str, str]], pdf_name: str) -> str:
        """
        페이지 내용을 마크다운 형식으로 변환합니다
        
        Args:
            pages_content (List[Dict[str, str]]): 페이지별 텍스트 정보
            pdf_name (str): PDF 파일명
            
        Returns:
            str: 마크다운 형식의 텍스트
        """
        markdown_content = []
        
        # 문서 제목 추가
        title = pdf_name.replace('.pdf', '').replace('_', ' ').title()
        markdown_content.append(f"# {title}\n")
        markdown_content.append(f"*원본 파일: {pdf_name}*\n")
        markdown_content.append("---\n")
        
        # 페이지별 내용 추가
        for page_info in pages_content:
            page_num = page_info["page_number"]
            text = self.clean_text(page_info["text"])
            
            # 페이지 구분자 추가
            markdown_content.append(f"\n## 페이지 {page_num}\n")
            markdown_content.append(f"{text}\n")
        
        # 문서 정보 추가
        total_pages = len(pages_content)
        total_chars = sum(page["char_count"] for page in pages_content)
        
        markdown_content.append("\n---")
        markdown_content.append(f"\n**문서 정보:**")
        markdown_content.append(f"- 총 페이지: {total_pages}")
        markdown_content.append(f"- 총 문자 수: {total_chars:,}")
        markdown_content.append(f"- 변환 일시: {Path().cwd()}")
        
        return "\n".join(markdown_content)
    
    def convert_single_pdf(self, pdf_path: Path) -> Optional[Path]:
        """
        단일 PDF 파일을 마크다운으로 변환합니다
        
        Args:
            pdf_path (Path): PDF 파일 경로
            
        Returns:
            Optional[Path]: 생성된 마크다운 파일 경로
        """
        log_info(f"PDF 변환 시작: {pdf_path.name}")
        
        # 텍스트 추출
        pages_content = self.extract_text_from_pdf(pdf_path)
        
        if not pages_content:
            log_error(f"텍스트 추출 실패: {pdf_path.name}")
            return None
        
        # 마크다운 변환
        markdown_text = self.convert_to_markdown(pages_content, pdf_path.name)
        
        # 마크다운 파일 저장
        output_filename = pdf_path.stem + ".md"
        output_path = self.output_dir / output_filename
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_text)
            
            log_success(f"마크다운 변환 완료: {output_filename}")
            return output_path
            
        except Exception as e:
            log_error(f"마크다운 파일 저장 실패: {str(e)}")
            return None
    
    def convert_all_pdfs(self) -> List[Path]:
        """
        입력 디렉토리의 모든 PDF 파일을 변환합니다
        
        Returns:
            List[Path]: 생성된 마크다운 파일 경로 목록
        """
        pdf_files = list(self.input_dir.glob("*.pdf"))
        
        if not pdf_files:
            log_warning(f"PDF 파일을 찾을 수 없습니다: {self.input_dir}")
            return []
        
        log_info(f"발견된 PDF 파일: {len(pdf_files)}개")
        
        converted_files = []
        
        for pdf_file in pdf_files:
            result = self.convert_single_pdf(pdf_file)
            if result:
                converted_files.append(result)
        
        log_success(f"전체 변환 완료: {len(converted_files)}/{len(pdf_files)} 파일")
        return converted_files
    
    def get_conversion_stats(self) -> Dict[str, int]:
        """
        변환 통계 정보를 반환합니다
        
        Returns:
            Dict[str, int]: 통계 정보
        """
        pdf_count = len(list(self.input_dir.glob("*.pdf")))
        md_count = len(list(self.output_dir.glob("*.md")))
        
        return {
            "pdf_files": pdf_count,
            "markdown_files": md_count,
            "conversion_rate": (md_count / pdf_count * 100) if pdf_count > 0 else 0
        }