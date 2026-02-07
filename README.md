# CrewAI RAG 시스템

CrewAI를 활용한 PDF 기반 RAG(Retrieval-Augmented Generation) 채팅 시스템입니다.

## 🎯 프로젝트 개요

PDF 문서를 마크다운으로 변환하고 임베딩하여, 사용자의 질문에 대해 관련 정보를 검색하고 답변을 생성하는 지능형 채팅 시스템입니다.

## 🏗️ 시스템 아키텍처

### 3개의 전문 에이전트
1. **질문 수집 에이전트**: 사용자 질문 분석 및 키워드 추출
2. **지식 검색 에이전트**: 벡터 DB에서 관련 문서 검색  
3. **답변 생성 에이전트**: 검색된 정보 기반 답변 생성

### 기술 스택
- **CrewAI**: 멀티 에이전트 오케스트레이션
- **Chroma DB**: 벡터 데이터베이스 (로컬 저장)
- **Gemini 2.0 Flash**: 대화형 AI 모델
- **Gemini Embedding**: 텍스트 임베딩 모델
- **Python 3.10+**: 메인 개발 언어
- **uv**: 모던 Python 패키지 관리

## 📁 프로젝트 구조

```
crewai-rag-system/
├── agents/                  # 에이전트 모듈들
│   ├── question_collector.py
│   ├── knowledge_retriever.py
│   ├── response_generator.py
│   └── rag_system.py
├── utils/                   # 유틸리티 함수들
│   ├── config_loader.py
│   ├── document_processor.py
│   ├── embedding_manager.py
│   ├── logger.py
│   └── pdf_converter.py
├── config/                  # 설정 파일들
│   ├── config.yaml
│   └── agents_config.yaml
├── data/                    # 데이터 저장소 (gitignore)
│   ├── pdfs/               # 원본 PDF 파일들
│   ├── markdown/           # 변환된 마크다운 파일들
│   └── chroma_db/          # Chroma 벡터 DB
├── main.py                  # 메인 실행 파일
├── process_pdfs.py          # PDF 처리 스크립트
└── pyproject.toml           # 프로젝트 설정
```

## 🚀 설치 및 실행

### 1. 저장소 클론
```bash
git clone <repository-url>
cd crewai-rag-system
```

### 2. uv 설치 (아직 설치하지 않은 경우)
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 또는 Homebrew
brew install uv

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 3. 프로젝트 의존성 설치
```bash
# 가상환경 생성 및 의존성 설치
uv sync
```

### 4. 환경 변수 설정
```bash
# .env.example을 복사하여 .env 파일 생성
cp .env.example .env

# .env 파일을 편집하여 실제 API 키 입력
# GEMINI_API_KEY="your_actual_api_key_here"
```

**Gemini API 키 발급 방법:**
1. [Google AI Studio](https://makersuite.google.com/app/apikey) 접속
2. "Create API Key" 클릭
3. 발급받은 키를 `.env` 파일에 입력

### 5. 데이터 폴더 생성
```bash
mkdir -p data/pdfs data/markdown data/chroma_db
```

### 6. PDF 파일 준비
PDF 파일들을 `data/pdfs/` 폴더에 저장합니다.

### 7. 시스템 실행
```bash
# PDF 파일 처리 (최초 1회)
uv run python process_pdfs.py

# RAG 채팅 시스템 실행
uv run python main.py
```

## 💬 사용 방법

1. 시스템 실행 후 대화형 모드 선택 (y/yes)
2. 질문 입력 (예: "istio-cni란 무엇인가요?")
3. 시스템이 관련 문서를 검색하고 마크다운 형식으로 답변 제공
4. 종료하려면 `quit` 또는 `exit` 입력

## 🔧 주요 기능

- ✅ PDF → Markdown 자동 변환
- ✅ 텍스트 청킹 및 벡터 임베딩
- ✅ 의미 기반 문서 검색
- ✅ 3개 에이전트 협업 시스템
- ✅ 마크다운 형식 답변 출력
- ✅ 대화형 채팅 인터페이스
- ✅ 한국어 완전 지원

## 📝 개발 정보

- **Python 버전**: 3.10+
- **패키지 관리**: uv
- **AI 모델**: Gemini 2.0 Flash, Gemini Embedding 001
- **벡터 DB**: Chroma DB
- **PDF 처리**: PyMuPDF

## 🤝 기여하기

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 📞 문의

프로젝트 관련 문의사항이 있으시면 이슈를 생성해주세요.