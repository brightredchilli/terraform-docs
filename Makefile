PYTHON_VERSION := 3.14
DATA_DIR := src/terraform_docs_mcp/_data
MANIFEST := $(DATA_DIR)/manifest.json

# Named explicitly rather than globbed, so it can be a real Make target and
# `dist/` holding an older version does not confuse the rule.
VERSION := $(shell sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml)
WHEEL := dist/terraform_docs_mcp-$(VERSION)-py3-none-any.whl

# Rebuild even when the manifest says nothing changed:
#   make index FORCE=1
#   make build FORCE=1
# Note it is FORCE=1 and not `--force`: make parses a leading `--` as its own
# option and errors out, so a variable override is the only way through to the
# recipe.
FORCE ?=
BUILD_FLAGS := $(if $(FORCE),--force,)

.PHONY: help bootstrap sync index reindex build install test typecheck check probe clean distclean ALWAYS

help:
	@echo "bootstrap  fetch submodules and sparse-checkout just the docs"
	@echo "sync       install dependencies into .venv"
	@echo "index      rebuild the search index in $(DATA_DIR) if any input changed"
	@echo "reindex    rebuild it unconditionally (same as: make index FORCE=1)"
	@echo "build      build the wheel (re-indexes first if needed)"
	@echo "install    install the tool locally via uv (always reinstalls)"
	@echo "test       run the test suite"
	@echo "typecheck  run basedpyright against pyrightconfig.json"
	@echo "check      typecheck + test"
	@echo "probe      exercise the stdio MCP server end to end"
	@echo "clean      remove the generated index"

bootstrap:
	git submodule update --init --depth 1

sync:
	uv sync --all-groups

# The manifest records the provider commits, a checksum of everything under
# src/, and the model id -- and is written last, so its presence also means the
# build finished. That makes it both the provenance record and the target Make
# can hang the rest of the build off.
#
# ALWAYS forces the recipe to run on every `make index`, but the recipe is a
# fingerprint comparison rather than a build: it returns in well under a second
# unless an input actually changed. Plain mtime prerequisites cannot do this
# job, in both directions -- `git checkout` rewrites unchanged files (which
# would trigger a needless 96-second rebuild), and `git submodule update`
# replaces thousands of documents without touching any file in this repo (which
# would be missed entirely).
$(MANIFEST): ALWAYS | sync
	uv run python -m terraform_docs_mcp.build_index $(BUILD_FLAGS)

ALWAYS:

index: $(MANIFEST)

# So the escape hatch is discoverable from `make help`.
reindex:
	@$(MAKE) index FORCE=1

# uv_build runs no build hooks, so `index` cannot be triggered from inside
# `uv build`. Make enforces the ordering instead; index.py additionally fails
# loudly at runtime if _data/ is missing.
#
# Depending on the manifest by mtime means a no-op `make index` leaves this
# alone too, so 70 MB is not repacked for nothing.
$(WHEEL): $(MANIFEST)
	uv build --wheel

build: $(WHEEL)

# Phony, so it always reinstalls even when the wheel is unchanged -- that is
# usually the whole reason for running it. Installs the built wheel rather than
# `.` so the artifact is packed once, not twice.
install: $(WHEEL)
	uv tool install --reinstall --python $(PYTHON_VERSION) $(WHEEL)

test: sync
	uv run pytest -q

# Uses the repo's pyrightconfig.json (standard mode, reportArgumentType=error).
typecheck: sync
	uv run basedpyright src tests

check: typecheck test

# Exercise the stdio MCP server end to end: handshake, tool discovery, and an
# actual tool call. Point it at whatever your client launches, e.g.
#   make probe PROBE_ARGS='--command "uvx --from=./dist/*.whl terraform-docs-mcp serve"'
#
# Run as a module rather than a console script: the probe is a development
# tool and is deliberately not shipped as an entry point.
PROBE_ARGS ?=
probe: sync
	uv run python -m terraform_docs_mcp.probe $(PROBE_ARGS)

clean:
	rm -rf $(DATA_DIR) dist

distclean: clean
	rm -rf .venv
