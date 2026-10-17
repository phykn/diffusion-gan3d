# CLAUDE.md

## 가상환경

- Python은 항상 저장소의 `.venv`를 사용한다. 시스템 `python`/`python3`는 쓰지 않는다.
- 실행 예:
  - 테스트: `.venv/bin/python -m pytest -q`
  - 린트: `.venv/bin/ruff check .`
  - 포맷 검사: `.venv/bin/ruff format --check .`
  - 스크립트: `.venv/bin/python run_train_1st.py ...`

## Git

- 브랜치는 `main` 하나만 유지한다. 작업용 브랜치나 PR을 만들지 않는다.
- 커밋은 `main`에 바로 하고, 푸시는 `git push origin main`으로 한다.
