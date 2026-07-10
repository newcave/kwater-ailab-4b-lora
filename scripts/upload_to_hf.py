# -*- coding: utf-8 -*-
"""Colab에서 실행: 학습된 LoRA 어댑터를 Hugging Face Hub에 업로드.

사전 준비:
  1) huggingface.co/settings/tokens 에서 write 권한 토큰 생성
  2) 아래 REPO_ID 확인 후 실행 (기본 private=True 권장)
"""
from huggingface_hub import login, HfApi

REPO_ID = "newcave/kwater-ailab-4b-lora"
ADAPTER_DIR = "kwater-4b-table-lora-final"      # STEP 7 저장 폴더
LOGO_PATH = None                                 # 예: "/content/drive/MyDrive/AI_Lab_logo.jpg"
MODEL_CARD = None                                # 예: "/content/MODEL_CARD_HF.md"
PRIVATE = True                                   # 내부 검토 전까지 True 권장

login()  # 토큰 입력 프롬프트

api = HfApi()
api.create_repo(REPO_ID, private=PRIVATE, exist_ok=True)
api.upload_folder(folder_path=ADAPTER_DIR, repo_id=REPO_ID)
if MODEL_CARD:
    api.upload_file(path_or_fileobj=MODEL_CARD, path_in_repo="README.md", repo_id=REPO_ID)
if LOGO_PATH:
    api.upload_file(path_or_fileobj=LOGO_PATH, path_in_repo="AI_Lab_logo.jpg", repo_id=REPO_ID)

print(f"업로드 완료: https://huggingface.co/{REPO_ID}")
