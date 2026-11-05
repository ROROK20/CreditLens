# CreditLens — developer convenience targets
# Usage: `make <target>`. Run `make help` (or just `make`) to list targets.

.DEFAULT_GOAL := help
.PHONY: help install test lint api frontend frontend-build snapshot capture offline demo demo-local

help: ## List available targets
	@echo "CreditLens — available make targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install backend with explainability + dev extras (editable)
	pip install -e ".[xai,dev]"

test: ## Run the backend test suite with coverage
	pytest tests/ --cov=creditlens -v

lint: ## Byte-compile the package as a lightweight lint/sanity check
	python -m compileall creditlens

api: ## Run the FastAPI server on port 8002
	uvicorn creditlens.api.app:app --reload --port 8002

frontend: ## Install deps and run the Next.js web demo (port 3003)
	cd frontend && npm install && npm run dev

frontend-build: ## Production build of the Next.js frontend
	cd frontend && npm install && npm run build

snapshot: ## Record the engine's model and audit for the no-backend page
	python frontend/scripts/record_engine.py

capture: ## Re-take the README screenshot and the social image together, from the live stack
	node e2e/capture.mjs

offline: ## Prove a production build with no public URLs calls no engine (real browser)
	node e2e/offline.mjs

demo: ## One-command full stack via Docker (API :8002 + web :3003)
	docker compose up --build

demo-local: ## Full stack without Docker: API on :8002 + Next dev on :3003
	@echo "Starting CreditLens API on :8002 and frontend on :3003 ..."
	@echo "API:      http://localhost:8002/health"
	@echo "Frontend: http://localhost:3003"
	uvicorn creditlens.api.app:app --port 8002 & \
		API_PID=$$!; \
		trap "kill $$API_PID 2>/dev/null" EXIT INT TERM; \
		cd frontend && npm install && NEXT_PUBLIC_API_URL=http://localhost:8002 npm run dev
