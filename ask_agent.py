import os
import sys
import warnings
import logging

# =====================================================================
# 1. 1차 방어막: 환경 변수 및 로깅 강제 종료
# =====================================================================
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

import json
import glob
import networkx as nx
import matplotlib.pyplot as plt
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# =====================================================================
# [기능 1] 장기 메모리 관리
# =====================================================================
MEMORY_FILE = "user_memory.json"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"frequent_keywords": []}

def save_memory(memory_data):
    with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(memory_data, f, ensure_ascii=False, indent=2)

memory = load_memory()

# =====================================================================
# 시스템 초기화 (로딩 중 경고문 완벽 묵살)
# =====================================================================
print("📚 에이전트 초기화 중... (데이터베이스 및 모델 연결)")

# 🌟 2차 방어막: 블랙홀(os.devnull)을 열어 모델 로딩 중 발생하는 모든 출력을 집어삼킴
original_stdout = sys.stdout
original_stderr = sys.stderr
blackhole = open(os.devnull, 'w')

sys.stdout = blackhole
sys.stderr = blackhole

try:
    # 이 안에서 발생하는 모든 로딩 바, 경고, 에러는 화면에 뜨지 않음
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
finally:
    # 모델 로딩이 끝나면 다시 터미널 출력을 정상화시킴
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    blackhole.close()

vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)

# 온도는 의도/검열용(0.0)과 추론/창의용(0.3)으로 분리
llm_strict = ChatOpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio", temperature=0.0)
llm_creative = ChatOpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio", temperature=0.3)

print("\n=======================================================")
print("🕵️‍♂️ [지능형] 장편소설 독서 가이드 에이전트 (Tools 탑재) 준비 완료!")
print("=======================================================")

if memory["frequent_keywords"]:
    print(f"💡 [장기 메모리 작동] 지난번에 {', '.join(memory['frequent_keywords'][:3])} 등에 대해 물어보셨군요. 기억하고 있습니다!")

# =====================================================================
# [UX 개선 1] 도서 자동 검색 및 객관식 선택 메뉴
# =====================================================================
PROCESSED_FOLDER = "./processed_books"
book_files = glob.glob(os.path.join(PROCESSED_FOLDER, "*.txt"))

if not book_files:
    print("\n⚠️ DB에 저장된 책이 없습니다! 'build_db.py'를 먼저 실행해서 책을 넣어주세요.")
    exit()

print("\n📚 [현재 DB에 보관 중인 도서 목록]")
available_books = []
for i, file_path in enumerate(book_files, 1):
    file_name = os.path.basename(file_path)
    book_title = os.path.splitext(file_name)[0].replace(" ", "_")
    available_books.append(book_title)
    print(f"  {i}. {book_title.replace('_', ' ')}")

while True:
    choice = input("\n👉 읽고 계신 책의 번호를 선택하세요: ").strip()
    try:
        choice_idx = int(choice) - 1
        if 0 <= choice_idx < len(available_books):
            selected_book = available_books[choice_idx]
            break
        else:
            print("⚠️ 목록에 있는 번호를 선택해 주세요.")
    except ValueError:
        print("⚠️ 문자가 아닌 '숫자'로 번호를 입력해 주세요.")

# =====================================================================
# [UX 개선 2] 진도 입력 강제 (빈칸 및 오타 방어)
# =====================================================================
while True:
    chapter_input = input(f"\n👉 [{selected_book.replace('_', ' ')}] 현재 몇 장(Chapter)을 읽고 계신가요? (숫자만): ").strip()
    
    if not chapter_input:
        print("⚠️ 값을 입력하지 않으셨습니다. 현재 진도를 반드시 입력해 주세요!")
        continue
        
    try:
        current_chapter = int(chapter_input)
        if current_chapter < 1:
            print("⚠️ 1 이상의 숫자를 입력해 주세요.")
            continue
        break
    except ValueError:
        print("⚠️ '숫자'만 입력하셔야 합니다. (예: 2)")

print(f"\n✅ 도서 및 진도 설정 완료! (도서: {selected_book.replace('_', ' ')} / 진도: {current_chapter}장)")
print("✅ 이제 자유롭게 질문해 주세요. (인물 관계도를 원하시면 '관계도 그려줘'라고 해보세요! / 종료: 'q')\n")

# =====================================================================
# 메인 채팅 루프 시작
# =====================================================================
while True:
    question = input("🧑‍💻 독자: ").strip()
    if not question: 
        continue
        
    if question.lower() == 'q':
        print(f"🕵️‍♂️ 에이전트: [{selected_book.replace('_', ' ')}] 독서를 마치셨군요! 다음에 또 뵙겠습니다.")
        break
        
    print("\n🕵️‍♂️ 에이전트 내부 프로세스 가동 중...")

    # =====================================================================
    # [기능 2] 의도 분석
    # =====================================================================
    intent_prompt = PromptTemplate.from_template("""
    다음 독자의 질문을 분석해서 의도를 추출해.
    의도는 반드시 [내용요약], [인물추적], [관계도시각화], [일반질문] 중 하나로만 대답해.
    
    질문: {question}
    답변 형식:
    의도: [의도]
    키워드: [키워드]
    """)
    intent_response = (intent_prompt | llm_strict).invoke({"question": question}).content
    print(f"  [1. 의도 분석 완료] {intent_response.strip().replace(chr(10), ' ')}")

    if "키워드:" in intent_response:
        keyword = intent_response.split("키워드:")[1].strip()
        if keyword and keyword not in memory["frequent_keywords"]:
            memory["frequent_keywords"].insert(0, keyword)
            save_memory(memory)

    # =====================================================================
    # [기능 3 & 8-1] 메타데이터 필터링 및 Tools 가동
    # =====================================================================
    if "관계도시각화" in intent_response:
        print("  [🛠️ Tools 가동] AI가 파이썬 시각화 도구를 직접 호출합니다...")
        
        all_docs = vectordb.get(
            where={
                "$and": [
                    {"book_title": selected_book},
                    {"chapter_num": {"$lte": current_chapter}}
                ]
            }
        )
        
        context = ""
        if all_docs and all_docs['documents']:
             recent_docs = all_docs['documents'][::-1][:3] 
             context = "\n\n".join(recent_docs)
             context = context[:12000]

        if not context:
            print("⚠️ 시각화할 수 있는 데이터가 부족합니다.")
            continue
            
        tool_prompt = PromptTemplate.from_template("""
        제공된 소설 내용에서 주요 인물들의 관계를 추출해.
        반드시 아래 형식으로만 출력해. (파이썬 코드로 파싱할 거라 부연 설명 절대 금지)
        (단, 그래프 시각화 시 한글 깨짐 방지를 위해 인물 이름과 관계는 반드시 '영어'로 작성할 것)
        
        형식: Name1|Name2|Relationship
        예시: Sherlock|Watson|Partner
        
        [소설 내용]
        {context}
        """)
        relations_text = (tool_prompt | llm_strict).invoke({"context": context}).content
        
        G = nx.Graph()
        for line in relations_text.strip().split('\n'):
            if '|' in line:
                parts = line.split('|')
                if len(parts) >= 3:
                    G.add_edge(parts[0].strip(), parts[1].strip(), label=parts[2].strip())
                    
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightcoral', node_size=2500, font_size=10, font_weight='bold')
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        
        plt.title(f"Character Relationship Map (Chap 1~{current_chapter})")
        plt.savefig("relationship_map.png")
        plt.close()
        
        print("\n================ [Tools 실행 완료] ================\n")
        print("📊 'relationship_map.png' 파일로 인물 관계도 이미지가 성공적으로 저장되었습니다!")
        print("   프로젝트 폴더를 열어서 생성된 이미지를 확인해 주세요.")
        print("\n====================================================\n")
        continue 

    else:
        docs = vectordb.similarity_search(question, k=5, filter={"book_title": selected_book})
        filtered_docs = [doc for doc in docs if doc.metadata["chapter_num"] <= current_chapter]
        context = "\n\n".join([doc.page_content for doc in filtered_docs])

        if not context:
            print("\n================ [최종 답변] ================\n")
            print("⚠️ 아직 읽지 않으신 미래의 내용이거나 존재하지 않는 내용입니다.")
            print("\n=============================================\n")
            continue

        # =====================================================================
        # [기능 4] CoT 추론
        # =====================================================================
        cot_prompt = PromptTemplate.from_template("""
        아래 [소설 내용]을 바탕으로 질문에 답하되, 반드시 <생각> 태그 안에 인과관계를 먼저 조립하고, <답변> 태그에 최종 답변을 작성해.
        [소설 내용] {context}
        질문: {question}
        """)
        print("  [2. CoT 추론 진행 중] 사건의 인과관계를 조립하고 있습니다...")
        draft_response = (cot_prompt | llm_creative).invoke({"context": context, "question": question}).content

        # =====================================================================
        # [기능 5] 🛡️ 딥 스포일러 가드레일 (문맥 100% 필터링)
        # =====================================================================
        verify_prompt = PromptTemplate.from_template("""
        너는 최고 수준의 무자비한 '스포일러 검열관(Spoiler Guard)'이야.
        독자는 현재 소설의 {current_chapter}장까지만 읽은 상태야.

        아래 [AI 답변 초안]을 극도로 깐깐하게 검토해서 다음 세 가지 기준을 통과하는지 확인해.

        [검열 기준]
        1. 명시적 스포일러: {current_chapter}장 이후에 발생할 사건, 인물의 죽음, 범인의 정체 등이 직접적으로 언급되었는가?
        2. 암시적 스포일러(복선 유출): 독자가 앞으로 스스로 추리해야 할 내용을 AI가 미리 결론지어 주거나, 나중에 밝혀질 반전을 은연중에 암시(뉘앙스)하고 있는가?
        3. 환각 및 외부 지식 개입: 제공된 텍스트에는 없는 소설의 뒷이야기를 AI가 자신의 기존 지식으로 채워 넣었는가?

        [행동 지침]
        - 위 기준 중 단 하나라도 의심되는 문장이 있다면, 그 문장을 완전히 삭제해.
        - 그리고 삭제한 자리나 답변의 끝에 "현재 진도({current_chapter}장)까지의 내용으로는 아직 명확한 진실을 알 수 없습니다. 앞으로의 전개를 기대해 주세요! 😊"라는 뉘앙스로 안전하게 방어하는 문장을 추가해.
        - 문제가 없다면 초안의 <답변> 내용을 자연스럽게 정돈하여 최종 대답으로 출력해.
        (반드시 한국어로 출력할 것)

        [AI 답변 초안]
        {draft}
        """)
        print("  [3. 딥 스포일러 가드레일 가동] 미래 뉘앙스 및 복선 유출 차단 심층 검열 중...")
        final_response = (verify_prompt | llm_strict).invoke({
            "current_chapter": current_chapter,
            "draft": draft_response
        }).content

        print("\n================ [최종 답변] ================\n")
        print(final_response)
        print("\n=============================================\n")