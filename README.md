# crewai_qa

대화형 문서 QA에서 에이전트 오케스트레이션 구조에 따른 정확도와 운영 비용 비교  
**CrewAI 실험 코드 — 제2저자(김희경) 담당**

---

## 논문 정보

| 항목 | 내용 |
|------|------|
| 제목 | 대화형 문서 QA에서 에이전트 오케스트레이션 구조에 따른 정확도와 운영 비용 비교 |
| 학술대회 | 한국IT서비스학회 2026 춘계학술대회 |
| 저자 | 문근영·김희경·주승현·허대영 (국민대학교) |
| 본 repo 담당 | 제2저자 김희경 (CrewAI 프레임워크 실험) |

---

## 실험 개요

- **비교 대상 프레임워크**: LangChain Agent / **CrewAI** / LangGraph
- **오케스트레이션 구조**: 역할 기반 순차 실행 (Agent + Task + Crew 분리 구조)
- **데이터셋**: 한국어 행정문서 QA 20문항 + FinQA 20문항 = 총 40문항
- **실행 횟수**: 40문항 × 3회 반복 = 120회
- **LLM**: `gemini-2.5-flash` (temperature=0, max_tokens=1024)
- **측정 지표**: TSR(Task Success Rate) · 토큰 사용량 · 실행 시간

---

## 폴더 구조

```
crewai_qa/
├── .env                         ← API 키 (직접 생성 필요, repo 미포함)
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt             ← 패키지 목록
├── crewai_agent.py              ← 실험 메인 코드
├── tool_api.py                  ← 공통 Tool API (문서 검색·계산·검증)
├── questions.csv                ← 40문항 목록 (질문·문서·정답 포함)
├── gold_answers.csv             ← 정답지 (계산 수식 포함)
├── evaluation_schema.json       ← 평가 스키마 (판정 기준 정의)
├── logs/
│   └── call_log.jsonl           ← 도구 호출 로그 (자동 기록)
├── results/                     ← 실험 결과 저장 (자동 생성)
│   └── results_crewai.jsonl
└── experiment_data/             ← 데이터셋 (repo에 포함)
    ├── 한국어데이터/
    │   └── sample/
    │       ├── TL_span_extraction.json
    │       ├── TL_span_extraction_how.json
    │       ├── TL_tableqa.json
    │       ├── TL_text_entailment.json
    │       ├── TS_span_extraction.json
    │       ├── TS_span_extraction_how.json
    │       ├── TS_tableqa.json
    │       └── TS_text_entailment.json
    └── FinQA-main/
        └── code/
            └── evaluate/
                └── test.json
```

> 데이터셋(`experiment_data/`)이 repo에 포함되어 있으므로 **별도 파일 수령 없이** clone만 하면 바로 실행할 수 있습니다.

---

## 설치 및 실행

### STEP 1 — Python 버전 확인

```powershell
python --version
```

**Python 3.10 이상** 필요. 버전이 낮으면 https://www.python.org/downloads 에서 설치하세요.

---

### STEP 2 — repo clone

```powershell
cd C:\Users\사용자이름\Documents
git clone https://github.com/Asakhan/crewai_qa.git
cd crewai_qa
```

---

### STEP 3 — 가상환경 생성 및 활성화

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

> ⚠ 오류가 나면 먼저 실행 후 재시도:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

활성화 성공 시 터미널 앞에 `(.venv)` 가 붙습니다.

```
(.venv) PS C:\Users\...\crewai_qa>
```

---

### STEP 4 — 패키지 설치

```powershell
pip install -r requirements.txt
```

설치 확인:

```powershell
pip show crewai
```

`Version: 1.14.4` 가 출력되면 완료입니다.

---

### STEP 5 — .env 파일 생성

`crewai_qa/` 폴더 안에 `.env` 파일을 만드세요.

```
GOOGLE_API_KEY=AIza여기에본인구글키붙여넣기
```

**Google API 키 발급 방법:**
1. https://aistudio.google.com/apikey 접속 (Google 계정 로그인)
2. **Create API key** 클릭
3. 생성된 키(`AIza...`) 복사 후 `.env` 파일에 붙여넣기

> ⚠ `.env` 파일은 절대 GitHub에 올리지 마세요. `.gitignore`에 등록되어 있습니다.

---

### STEP 6 — 동작 확인

```powershell
python tool_api.py
```

아래처럼 출력되면 모든 파일과 데이터 연결이 정상입니다.

```
============================================================
Tool API 동작 확인
============================================================

[1] 문서 DB 로드: 40개 문항
[2] retrieve_document('Q001')
  status   : ok
  dataset  : Korean_Admin_QA
  question : 2020년도 기준 70대 이상 어르신의 스마트폰 보유율은 몇 퍼센트야
  document : [출처: 서울특별시청] ...
...
모든 동작 확인 완료.
```

---

### STEP 7 — 실험 실행

```powershell
python crewai_agent.py
```

실행하면 아래처럼 진행됩니다.

```
============================================================
CrewAI 실험 시작
============================================================
총 40문항 × 3회 반복 = 120회 실행
결과 저장: results/results_crewai.jsonl

[Q001_CrewAI_01] 실행 중... ✓ (4.2초, 2466tokens, success=1)
[Q001_CrewAI_02] 실행 중... ✓ (3.9초, 2451tokens, success=1)
[Q001_CrewAI_03] 실행 중... ✓ (4.1초, 2458tokens, success=1)
...
```

완료 후 `results/results_crewai.jsonl` 파일을 제1저자(문근영)에게 전달하세요.

> 💡 **중단 후 재시작 가능**: 실험 도중 종료되어도 괜찮습니다.  
> 다시 실행하면 **성공(success=1)한 항목은 자동으로 건너뛰고** 실패한 항목만 재실행합니다.

---

## 오류 발생 시 해결법

| 오류 메시지 | 원인 | 해결 |
|------------|------|------|
| `(.venv) 가 안 붙음` | 가상환경 비활성화 | STEP 3 다시 실행 |
| `ModuleNotFoundError: crewai` | 패키지 미설치 | `pip install -r requirements.txt` |
| `FileNotFoundError: questions.csv` | clone 오류 | STEP 2 재확인 |
| `FileNotFoundError: test.json` | clone 오류 | STEP 2 재확인 |
| `GOOGLE_API_KEY 없음` | .env 파일 오류 | STEP 5 재확인 |
| `Google Gen AI native provider not available` | `google-genai` 패키지 미설치 | `pip install -r requirements.txt` 재실행 |
| `Event stack depth limit (100) exceeded` | 이전 실행 비정상 종료로 내부 스택 누적 | **터미널을 완전히 닫고** 새로 열어서 재실행 |
| `503 UNAVAILABLE` | Gemini 서버 일시 과부하 | 자동 재시도 처리됨 (최대 5회, 60초 간격) |
| `429 RESOURCE_EXHAUSTED` | API 일일 무료 한도 초과 | 다음날 재실행 (자정에 한도 리셋) |

---

## 주의 사항

- `.env` 파일(API 키)과 `results/` 폴더(실험 결과)는 GitHub에 올리지 마세요.
- 실험 결과 파일(`results/results_crewai.jsonl`)은 완료 후 제1저자에게 직접 전달하세요.
- `Event stack depth limit` 오류가 발생하면 반드시 **터미널을 완전히 닫고** 새 터미널에서 재실행하세요. 단순히 `Ctrl+C` 후 재실행하면 같은 오류가 반복됩니다.
- 실험 중 503 오류가 많으면 Gemini 서버 과부하 상태입니다. 잠시 기다렸다가 재실행하면 됩니다.
