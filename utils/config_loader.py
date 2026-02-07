"""
설정 파일 로더 유틸리티
YAML 설정 파일과 환경 변수를 로드하고 관리합니다.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv


class ConfigLoader:
    """설정 파일과 환경 변수를 로드하는 클래스"""
    
    def __init__(self, config_dir: str = "config"):
        """
        ConfigLoader 초기화
        
        Args:
            config_dir (str): 설정 파일이 있는 디렉토리 경로
        """
        self.config_dir = Path(config_dir)
        self.config_data = {}
        
        # .env 파일 로드
        load_dotenv()
        
        # 설정 파일들 로드
        self._load_configs()
    
    def _load_configs(self):
        """모든 YAML 설정 파일을 로드합니다"""
        config_files = [
            "config.yaml",
            "agents_config.yaml"
        ]
        
        for config_file in config_files:
            config_path = self.config_dir / config_file
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    file_data = yaml.safe_load(f)
                    # 파일명에서 확장자 제거하여 키로 사용
                    key = config_file.replace('.yaml', '')
                    self.config_data[key] = file_data
                    print(f"✅ {config_file} 로드 완료")
            else:
                print(f"⚠️  {config_file} 파일을 찾을 수 없습니다")
    
    def get_config(self, config_name: str) -> Dict[str, Any]:
        """
        특정 설정 파일의 데이터를 반환합니다
        
        Args:
            config_name (str): 설정 파일명 (확장자 제외)
            
        Returns:
            Dict[str, Any]: 설정 데이터
        """
        return self.config_data.get(config_name, {})
    
    def get_api_config(self) -> Dict[str, Any]:
        """API 관련 설정을 반환합니다"""
        config = self.get_config("config")
        api_config = config.get("api", {})
        
        # 환경 변수에서 API 키 로드
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            api_config["gemini_api_key"] = gemini_key
        
        return api_config
    
    def get_database_config(self) -> Dict[str, Any]:
        """데이터베이스 관련 설정을 반환합니다"""
        config = self.get_config("config")
        return config.get("database", {})
    
    def get_agents_config(self) -> Dict[str, Any]:
        """에이전트 관련 설정을 반환합니다"""
        return self.get_config("agents_config")
    
    def get_paths_config(self) -> Dict[str, Any]:
        """파일 경로 관련 설정을 반환합니다"""
        config = self.get_config("config")
        return config.get("paths", {})
    
    def validate_config(self) -> bool:
        """
        필수 설정이 모두 있는지 확인합니다
        
        Returns:
            bool: 설정이 유효하면 True
        """
        api_config = self.get_api_config()
        
        # Gemini API 키 확인
        if not api_config.get("gemini_api_key"):
            print("❌ GEMINI_API_KEY 환경 변수가 설정되지 않았습니다")
            return False
        
        # 필수 디렉토리 확인
        paths_config = self.get_paths_config()
        required_paths = ["pdf_input", "markdown_output"]
        
        for path_key in required_paths:
            path_value = paths_config.get(path_key)
            if path_value:
                Path(path_value).mkdir(parents=True, exist_ok=True)
        
        print("✅ 설정 검증 완료")
        return True


# 전역 설정 로더 인스턴스
config_loader = ConfigLoader()