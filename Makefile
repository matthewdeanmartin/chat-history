all:
	@echo "Nothing to do by default"
	@echo "Try 'make run'"

run:
	uv run uvicorn chat_history.app:app --reload --port 8080

#install:
#	uv run sync
