"""
로깅 설정 유틸리티
프로젝트 전체에서 사용할 로거를 설정합니다.
"""

import logging
import sys
from pathlib import Path
from rich.logging import RichHandler
from rich.console import Console

# Rich 콘솔 설정
console = Console()


def setup_logger(name: str = "crewai_rag", level: str = "INFO") -> logging.Logger:
    """
    프로젝트용 로거를 설정합니다
    
    Args:
        name (str): 로거 이름
        level (str): 로그 레벨 (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        logging.Logger: 설정된 로거
    """
    
    # 로거 생성
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # 핸들러가 이미 있으면 제거 (중복 방지)
    if logger.handlers:
        logger.handlers.clear()
    
    # Rich 핸들러 설정 (콘솔 출력용)
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True
    )
    
    # 포맷터 설정
    formatter = logging.Formatter(
        fmt="%(message)s",
        datefmt="[%X]"
    )
    rich_handler.setFormatter(formatter)
    
    # 핸들러 추가
    logger.addHandler(rich_handler)
    
    # 상위 로거로 전파 방지
    logger.propagate = False
    
    return logger


def setup_file_logger(name: str = "crewai_rag", log_file: str = "logs/app.log") -> logging.Logger:
    """
    파일 출력용 로거를 추가로 설정합니다
    
    Args:
        name (str): 로거 이름
        log_file (str): 로그 파일 경로
        
    Returns:
        logging.Logger: 설정된 로거
    """
    
    logger = logging.getLogger(name)
    
    # 로그 디렉토리 생성
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 파일 핸들러 설정
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # 파일용 포맷터 (더 상세한 정보 포함)
    file_formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    
    # 파일 핸들러 추가
    logger.addHandler(file_handler)
    
    return logger


# 기본 로거 인스턴스 생성
logger = setup_logger()


def log_step(step_name: str, description: str = ""):
    """
    단계별 진행 상황을 로그로 출력합니다
    
    Args:
        step_name (str): 단계 이름
        description (str): 단계 설명
    """
    separator = "=" * 50
    logger.info(f"\n{separator}")
    logger.info(f"🚀 [bold blue]{step_name}[/bold blue]")
    if description:
        logger.info(f"📝 {description}")
    logger.info(f"{separator}")


def log_success(message: str):
    """성공 메시지를 로그로 출력합니다"""
    logger.info(f"✅ [bold green]{message}[/bold green]")


def log_warning(message: str):
    """경고 메시지를 로그로 출력합니다"""
    logger.warning(f"⚠️  [bold yellow]{message}[/bold yellow]")


def log_error(message: str):
    """에러 메시지를 로그로 출력합니다"""
    logger.error(f"❌ [bold red]{message}[/bold red]")


def log_info(message: str):
    """정보 메시지를 로그로 출력합니다"""
    logger.info(f"ℹ️  {message}")