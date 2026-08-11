PYTHON_VERSION := 3.14
DATA_DIR := src/terraform_docs_mcp/_data

.PHONY: help bootstrap sync index build install test typecheck check clean distclean

help:
	@echo "bootstrap  fetch submodules and sparse-checkout just the docs"
	@echo "sync       install dependencies into .venv"
	@echo "index      build the search index into $(DATA_DIR)"
	@echo "build      build the wheel (runs index first)"
	@echo "install    install the tool locally via uv"
	@echo "test       run the test suite"
	@echo "typecheck  run basedpyright against pyrightconfig.json"
	@echo "check      typecheck + test"
	@echo "clean      remove the generated index"

# Sparse-checkout config is not recorded in .gitmodules, so a fresh clone gets
# the full ~430 MB of provider source until this runs. Restricting to
# website/docs brings that down to ~38 MB. Cone mode keeps root files, which is
# how we still get each provider's LICENSE.
bootstrap:
	git submodule update --init --depth 1
	@for m in terraform-provider-aws terraform-provider-google; do \
		git -C $$m sparse-checkout init --cone; \
		git -C $$m sparse-checkout set website/docs; \
	done
	@du -sh terraform-provider-aws terraform-provider-google

sync:
	uv sync --all-groups

index: sync
	uv run python -m terraform_docs_mcp.build_index

# uv_build runs no build hooks, so `index` cannot be triggered from inside
# `uv build`. Make enforces the ordering instead; index.py additionally fails
# loudly at runtime if _data/ is missing.
build: index
	uv build --wheel

install: build
	uv tool install --reinstall --python $(PYTHON_VERSION) .

test: sync
	uv run pytest -q

# Uses the repo's pyrightconfig.json (standard mode, reportArgumentType=error).
typecheck: sync
	uv run basedpyright src tests

check: typecheck test

# Exercise the stdio MCP server end to end: handshake, tool discovery, and an
# actual tool call. Point it at whatever your client launches, e.g.
#   make probe PROBE_ARGS='--command "uvx --from=./dist/*.whl terraform-docs serve"'
PROBE_ARGS ?=
probe: sync
	uv run terraform-docs-probe $(PROBE_ARGS)

clean:
	rm -rf $(DATA_DIR) dist

distclean: clean
	rm -rf .venv
