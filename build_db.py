import os
import re
import shutil
import glob
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# =====================================================================
# 1. 자동화 폴더 시스템 설정
# =====================================================================
RAW_FOLDER = "./raw_books"             # 새 책을 넣는 폴더 (Inbox)
PROCESSED_FOLDER = "./processed_books" # 처리가 완료된 책이 이동할 폴더 (Outbox)

# 폴더가 없으면 자동으로 생성
os.makedirs(RAW_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# =====================================================================
# 2. 챕터 자동 분석 알고리즘 (Auto-Splitter)
# =====================================================================
def auto_split_chapters(text):
    # 구텐베르크 등 다양한 책에서 쓰이는 대표적인 챕터 패턴들
    patterns = [
        r'\n(?=CHAPTER\s+[IVX]+|\d+)',          # CHAPTER I, CHAPTER 1
        r'\n(?=Chapter\s+[IVX]+|\d+)',          # Chapter I, Chapter 1
        r'\n(?=PART\s+[IVX]+)',                 # PART I
        r'\n(?=[IVX]+\.\s+[A-Z])',              # I. Title (셜록 홈즈 스타일)
        r'\n\n(?=[A-Z\s]{5,})\n\n'              # 대문자 제목만 덩그러니 있는 경우
    ]
    
    best_chunks = [text] # 기본값: 통째로 하나
    max_chunks = 0
    
    # 여러 패턴을 돌려보고 가장 합리적으로 분할되는 패턴 찾기
    for pattern in patterns:
        chunks = re.split(pattern, text)
        # 챕터가 3개 이상, 150개 이하로 나뉘면 성공적인 패턴으로 간주
        if 3 < len(chunks) < 150 and len(chunks) > max_chunks:
            best_chunks = chunks
            max_chunks = len(chunks)
            
    return best_chunks

# =====================================================================
# 3. 메인 자동화 파이프라인 (Automated Ingest)
# =====================================================================
print("🔄 [자동화 파이프라인] 도서 인제스트 시스템 가동 시작...")

# raw_books 폴더 안의 모든 .txt 파일 찾기
target_files = glob.glob(os.path.join(RAW_FOLDER, "*.txt"))

if not target_files:
    print(f"⚠️ '{RAW_FOLDER}' 폴더에 처리할 새로운 텍스트 파일이 없습니다.")
    print("새로운 책(.txt)을 넣고 다시 실행해 주세요!")
    exit()

documents = []
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

for file_path in target_files:
    # 파일명에서 확장자를 제외하고 책 제목으로 사용 ("A Study in Scarlet.txt" -> "A Study in Scarlet")
    file_name = os.path.basename(file_path)
    book_title = os.path.splitext(file_name)[0].replace(" ", "_") # 공백은 언더바로 처리
    
    print(f"\n📥 [새 도서 감지] '{book_title}' 분석 및 처리 중...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    # 구텐베르크 껍데기(라이선스 문구) 자동 제거 시도
    if "*** START OF THE PROJECT GUTENBERG" in raw_text:
        novel_text = raw_text.split("*** START OF THE PROJECT GUTENBERG")[1].split("*** END OF THE PROJECT GUTENBERG")[0]
    else:
        novel_text = raw_text

    # Auto-Splitter 작동
    chapters = auto_split_chapters(novel_text)
    
    valid_chunks = 0
    for i, text_chunk in enumerate(chapters[1:], start=1):
        clean_text = text_chunk.strip()
        if len(clean_text) < 100: continue
            
        doc = Document(
            page_content=clean_text,
            metadata={
                "book_title": book_title,
                "chapter_num": i
            }
        )
        documents.append(doc)
        valid_chunks += 1
        
    print(f"  ➔ 성공적으로 {valid_chunks}개의 챕터로 자동 분할되었습니다!")
    
    # 처리가 끝난 파일은 processed_books 폴더로 이동 (보관)
    destination = os.path.join(PROCESSED_FOLDER, file_name)
    shutil.move(file_path, destination)
    print(f"  ➔ '{file_name}' 파일이 처리 완료 폴더로 이동되었습니다.")

# =====================================================================
# 4. 벡터 DB에 일괄 업데이트
# =====================================================================
if documents:
    print("\n💾 추출된 모든 데이터를 ChromaDB에 병합(저장) 중입니다...")
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory="./chroma_db"
    )
    print("✅ 데이터베이스 업데이트가 완벽하게 완료되었습니다!")