# usage:  source run_arctan.sh        (from the repo root, with the venv active)
# Clears every MEMORY_* / FACT_* setting left over in this terminal, then sets the arctan configuration and starts the proxy.
for v in $(env | grep -E '^(MEMORY_|FACT_|MANAGER|CONTEXT_MANAGER|PAIR_FACTOR|STANDIN_)' | cut -d= -f1); do unset "$v"; done
export MOCK_LLM=1                  # remove this line and `export GEMINI_API_KEY=your_key` instead for real models
export MEMORY_SCALE=arctan         # plain arctan: big jump at the start, flattens towards the cap
export MEMORY_BUDGET=1500          # base budget at prompt 0
export MEMORY_BUDGET_MAX=6000      # hard cap it approaches
export MEMORY_SCALE_BEND=40        # smaller = bigger early jump and earlier flattening
export MEMORY_SCALE_ARC=1          # 1 = pure arctan
export MEMORY_SCALE_START=0        # growth begins at prompt 0 (the default for arctan)
export MEMORY_SCALE_FRAC=0         # no full-history guard (the default for arctan)
echo "proxy settings:"; env | grep -E '^(MEMORY_|MOCK_LLM)' | sort
uvicorn main:app --port 8000
