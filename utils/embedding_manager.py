"""
임베딩 관리 유틸리티
문서를 청킹하고 임베딩을 생성하여 Chroma DB에 저장합니다.
LangChain Google GenAI를 사용합니다.
"""

import os
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import re

from .logger import logger, log_info, log_success, log_warning, log_error, log_step
from .config_loader import config_loader


class EmbeddingManager:
    """문서 임베딩 및 벡터 DB 관리 클래스"""
    
    def __init__(self, 
                 db_path: str = "./data/chroma_db",
                 collection_name: str = "rag_documents",
                 embedding_model: str = "models/gemini-embedding-001",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        EmbeddingManager 초기화
        
        Args:
            db_path (str): Chroma DB 저장 경로
            collection_name (str): 컬렉션 이름
            embedding_model (str): Gemini 임베딩 모델명
            chunk_size (int): 청크 크기
            chunk_overlap (int): 청크 겹침 크기
        """
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 디렉토리 생성
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # LangChain Google GenAI 임베딩 설정
        api_config = config_loader.get_api_config()
        api_key = api_config.get("gemini_api_key")
        
        if not api_key:
            raise ValueError("GEMINI_API_KEY가 설정되지 않았습니다")
        
        # 임베딩 모델 시도 순서: 공식 문서 기준
        embedding_models = [
            "gemini-embedding-001"  # 공식 문서에서 확인된 유일한 임베딩 모델
        ]
        
        embedding_model_used = None
        for model_name in embedding_models:
            try:
                test_model = GoogleGenerativeAIEmbeddings(
                    model=model_name,
                    google_api_key=api_key
                )
                # 테스트 임베딩으로 모델 검증
                test_embedding = test_model.embed_query("test")
                self.embedding_model = test_model
                embedding_model_used = model_name
                log_success(f"LangChain Gemini 임베딩 설정 완료: {model_name}")
                break
            except Exception as e:
                log_warning(f"임베딩 모델 {model_name} 실패: {str(e)}")
                continue
        
        if embedding_model_used is None:
            raise ValueError("사용 가능한 Gemini 임베딩 모델을 찾을 수 없습니다")
        
        # Chroma DB 클라이언트 초기화
        self._init_chroma_client()
        
        log_info(f"Chroma DB 경로: {self.db_path}")
        log_info(f"컬렉션명: {self.collection_name}")
        log_info(f"청크 크기: {self.chunk_size}, 겹침: {self.chunk_overlap}")
    
    def _init_chroma_client(self):
        """Chroma DB 클라이언트를 초기화합니다"""
        try:
            # Chroma 클라이언트 생성
            self.client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # 컬렉션 생성 또는 가져오기
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                log_info(f"기존 컬렉션 로드: {self.collection_name}")
            except:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "RAG 시스템용 문서 컬렉션"}
                )
                log_success(f"새 컬렉션 생성: {self.collection_name}")
            
        except Exception as e:
            log_error(f"Chroma DB 초기화 실패: {str(e)}")
            raise
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        텍스트를 청크로 분할합니다
        
        Args:
            text (str): 분할할 텍스트
            metadata (Dict[str, Any]): 메타데이터
            
        Returns:
            List[Dict[str, Any]]: 청크 정보 리스트
        """
        if not text.strip():
            return []
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            # 청크 끝 위치 계산
            end = start + self.chunk_size
            
            # 문장 경계에서 자르기 (가능한 경우)
            if end < len(text):
                # 마지막 문장 끝을 찾기
                sentence_end = text.rfind('.', start, end)
                if sentence_end > start + self.chunk_size // 2:
                    end = sentence_end + 1
                else:
                    # 단어 경계에서 자르기
                    word_end = text.rfind(' ', start, end)
                    if word_end > start + self.chunk_size // 2:
                        end = word_end
            
            # 청크 텍스트 추출
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunk_info = {
                    "text": chunk_text,
                    "chunk_id": chunk_id,
                    "start_pos": start,
                    "end_pos": end,
                    "char_count": len(chunk_text),
                    "metadata": metadata or {}
                }
                chunks.append(chunk_info)
                chunk_id += 1
            
            # 다음 청크 시작 위치 (겹침 고려)
            start = max(start + 1, end - self.chunk_overlap)
            
            # 무한 루프 방지
            if start >= len(text):
                break
        
        log_info(f"텍스트 청킹 완료: {len(chunks)}개 청크 생성")
        return chunks
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        텍스트 리스트에 대한 임베딩을 LangChain Gemini로 생성합니다
        
        Args:
            texts (List[str]): 임베딩할 텍스트 리스트
            
        Returns:
            List[List[float]]: 임베딩 벡터 리스트
        """
        try:
            log_info(f"LangChain Gemini 임베딩 생성 중: {len(texts)}개 텍스트")
            
            # LangChain 임베딩 생성
            embeddings = self.embedding_model.embed_documents(texts)
            
            log_success(f"LangChain Gemini 임베딩 생성 완료: {len(embeddings)}개")
            return embeddings
            
        except Exception as e:
            log_error(f"LangChain Gemini 임베딩 생성 실패: {str(e)}")
            return []
    
    def add_document(self, file_path: Path, content: str) -> bool:
        """
        문서를 청킹하고 임베딩하여 DB에 저장합니다
        
        Args:
            file_path (Path): 문서 파일 경로
            content (str): 문서 내용
            
        Returns:
            bool: 성공 여부
        """
        try:
            log_step(f"문서 처리 시작", f"파일: {file_path.name}")
            
            # 메타데이터 생성
            metadata = {
                "source_file": str(file_path),
                "file_name": file_path.name,
                "file_type": file_path.suffix,
                "char_count": len(content)
            }
            
            # 텍스트 청킹
            chunks = self.chunk_text(content, metadata)
            
            if not chunks:
                log_warning("청크가 생성되지 않았습니다")
                return False
            
            # 청크 텍스트 추출
            chunk_texts = [chunk["text"] for chunk in chunks]
            
            # 임베딩 생성
            embeddings = self.generate_embeddings(chunk_texts)
            
            if not embeddings:
                log_error("임베딩 생성 실패")
                return False
            
            # DB에 저장할 데이터 준비
            ids = [f"{file_path.stem}_{chunk['chunk_id']}_{uuid.uuid4().hex[:8]}" 
                   for chunk in chunks]
            
            metadatas = []
            for chunk in chunks:
                chunk_metadata = chunk["metadata"].copy()
                chunk_metadata.update({
                    "chunk_id": chunk["chunk_id"],
                    "start_pos": chunk["start_pos"],
                    "end_pos": chunk["end_pos"],
                    "char_count": chunk["char_count"]
                })
                metadatas.append(chunk_metadata)
            
            # Chroma DB에 저장
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=chunk_texts,
                metadatas=metadatas
            )
            
            log_success(f"문서 저장 완료: {len(chunks)}개 청크")
            return True
            
        except Exception as e:
            log_error(f"문서 처리 실패: {str(e)}")
            return False
    
    def search_similar(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        유사한 문서를 검색합니다
        
        Args:
            query (str): 검색 쿼리
            n_results (int): 반환할 결과 수
            
        Returns:
            Dict[str, Any]: 검색 결과
        """
        try:
            log_info(f"유사 문서 검색: '{query[:50]}...'")
            
            # 쿼리 임베딩 생성 (LangChain 사용)
            query_embedding = self.embedding_model.embed_query(query)
            
            # 유사 문서 검색
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            log_success(f"검색 완료: {len(results['documents'][0])}개 결과")
            return results
            
        except Exception as e:
            log_error(f"검색 실패: {str(e)}")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        컬렉션 통계 정보를 반환합니다
        
        Returns:
            Dict[str, Any]: 통계 정보
        """
        try:
            count = self.collection.count()
            
            # 샘플 데이터로 추가 정보 수집
            sample = self.collection.peek(limit=10)
            
            unique_sources = set()
            if sample["metadatas"]:
                for metadata in sample["metadatas"]:
                    if "source_file" in metadata:
                        unique_sources.add(metadata["source_file"])
            
            return {
                "total_chunks": count,
                "unique_documents": len(unique_sources),
                "collection_name": self.collection_name,
                "embedding_model": "LangChain GoogleGenerativeAIEmbeddings",
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap
            }
            
        except Exception as e:
            log_error(f"통계 정보 수집 실패: {str(e)}")
            return {}
    
    def reset_collection(self) -> bool:
        """
        컬렉션을 초기화합니다
        
        Returns:
            bool: 성공 여부
        """
        try:
            log_warning("컬렉션 초기화 중...")
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "RAG 시스템용 문서 컬렉션"}
            )
            log_success("컬렉션 초기화 완료")
            return True
        except Exception as e:
            log_error(f"컬렉션 초기화 실패: {str(e)}")
            return False