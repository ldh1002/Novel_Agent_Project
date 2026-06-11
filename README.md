# 📚 장편소설 독서 가이드 지능형 AI 에이전트

## 💡 프로젝트 개요
장편 소설 독서 중 과거의 복선이나 인물을 잊어버려 몰입도가 떨어지는 문제를 해결하기 위한 프라이빗 AI 에이전트입니다. 기존 챗봇의 치명적 단점인 '미래 내용 스포일러(Spoiler)'를 완벽하게 차단하며, 독자의 현재 진도에 맞춘 맞춤형 가이드와 인물 관계도 시각화(Tools) 기능을 제공합니다.

## 🛠️ 핵심 기능 (Key Features)
1. **딥 스포일러 가드레일 (Context & LLM Filtering)**
   - 독자의 현재 진도(Chapter)를 시스템이 인지하여 미래 챕터 데이터의 검색을 물리적으로 1차 차단합니다.
   - LLM 검열관 프롬프트를 통한 2차 검증으로 은유적 복선 유출 및 미래 뉘앙스까지 100% 방어합니다.
2. **지능형 도구 호출 (Agentic Tools)**
   - 대화 중 인물 관계 파악이 필요할 경우, AI가 스스로 판단하여 NetworkX 시각화 도구를 호출하고 관계도 이미지(`relationship_map.png`)를 자동 생성합니다.
3. **오토 인제스트 파이프라인 (Auto Ingest)**
   - 새로운 텍스트 파일을 폴더에 넣기만 하면, 정규식을 통해 챕터를 스스로 분석하고 Vector DB(Chroma)에 적재하는 자동화 시스템을 구축했습니다.
4. **장기 메모리 및 UX 개선**
   - 독자의 잦은 질문 키워드를 로컬 JSON에 영구 누적하여 다음 세션에 기억하는 개인화 기능을 제공합니다.

## 💻 기술 스택 (Tech Stack)
- **Language:** Python
- **LLM Engine:** Local LM Studio (Gemma / Llama-3 등 호환)
- **RAG & Vector DB:** LangChain, ChromaDB, HuggingFace Embeddings
- **Visualization:** NetworkX, Matplotlib

## 🚀 실행 방법 (How to Run)
1. 필요 라이브러리 설치
```bash
pip install langchain langchain-chroma langchain-huggingface langchain-openai networkx matplotlib
```
2. 도서 데이터베이스 빌드 (최초 1회 또는 도서 추가 시)
```
Bash
python build_db.py
```
최초 실행시, 생성된 raw_books폴더에 가지고 있는 도서 txt 파일을 생성된 저장 

3. 에이전트 실행 및 대화 시작
```   
Bash
python ask_agent.py
```
---
[프로젝트 발표 자료 보기 (PDF)](장편소설 독서용 지능형 AI 에이전트 최종발표.pdf)
