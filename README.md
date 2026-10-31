# Week 9 Graded Mini Project: Mini-RAG

Author: Om Prakash Bhardwaj

## Setup

1. Install Python 3.10 or later.
2. From this folder, create and activate a virtual environment, then install dependencies:

	```powershell
	py -m venv .venv
	.\.venv\Scripts\Activate.ps1
	pip install openai qdrant-client python-dotenv
	```

3. Set these variables in your local environment before running the pipeline:
	- `OPENAI_API_KEY`: your Vocareum-issued key
	- `OPENAI_BASE_URL`: the Vocareum OpenAI-compatible endpoint
	- `OPENAI_MODEL`: `gpt-4o-mini`
	- `QDRANT_URL`: your Qdrant cluster URL
	- `QDRANT_API_KEY`: your Qdrant API key

	Keep credential values private. They are not included in this submission.

## Run

Run commands from this folder:

```powershell
python mp2_rag.py ingest
python mp2_rag.py validate
python mp2_rag.py ask
```

`ingest` loads the five stories, chunks and embeds them, and stores the vectors in Qdrant. It recreates the `mp2_sherlock` collection before ingestion. `validate` runs the two predefined questions and the three learner-authored questions. `ask` starts the interactive question loop.

## Submission files

- `mp2_rag.py`: end-to-end RAG pipeline
- `data/learner_questions.jsonl`: three learner-authored validation questions
- `mp2_reflection.md`: four-section project reflection
- `mp2_validation.txt`: recorded ingestion and validation output
- `README.md`: setup and run instructions
