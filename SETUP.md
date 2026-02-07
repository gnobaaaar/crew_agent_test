# 빠른 설정 가이드

## 🚀 5분 만에 시작하기

### 1. 저장소 클론 및 이동
```bash
git clone <your-repository-url>
cd crewai-rag-system
```

### 2. 환경 설정
```bash
# uv 설치 (없는 경우)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 의존성 설치
uv sync

# 환경 변수 설정
cp .env.example .env
# .env 파일을 편집하여 GEMINI_API_KEY 입력
```

### 3. 데이터 폴더 생성
```bash
mkdir -p data/pdfs data/markdown data/chroma_db
```

### 4. PDF 파일 추가
PDF 파일을 `data/pdfs/` 폴더에 복사

### 5. 실행
```bash
# PDF 처리
uv run python process_pdfs.py

# 채팅 시작
uv run python main.py
```

## 🔑 API 키 발급

1. [Google AI Studio](https://makersuite.google.com/app/apikey) 접속
2. "Create API Key" 클릭
3. 발급받은 키를 `.env` 파일에 입력

## 💡 사용 팁

- 처음 실행 시 PDF 처리에 시간이 걸릴 수 있습니다
- 질문은 한국어로 입력하세요
- 종료하려면 `quit` 또는 `exit` 입력
- 시스템 상태 확인: `status` 입력