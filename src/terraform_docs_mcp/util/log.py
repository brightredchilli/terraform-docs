"""One shared, pre-configured structured logger.

Import the module and call it directly -- no per-module
``structlog.get_logger(__name__)`` boilerplate at every call site:

    from terraform_docs_mcp.util import log

    log.info("documents_loaded", count=4336, duration_s=1.0)

Structured on purpose: a short, stable event name as the first argument, real
values as keyword arguments -- not an f-string. That is what actually makes
structlog worth using over ``print``; wrapping the same interpolated strings
in ``log.info(f"...")`` would keep the thing this replaces.
"""

from __future__ import annotations

import logging

import structlog

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

_logger = structlog.get_logger()

# Rebound as module-level names rather than exporting `_logger` itself, so
# `from terraform_docs_mcp.util import log; log.info(...)` resolves `info` as
# an attribute of this *module* -- which is what the module-as-namespace
# calling convention above actually needs.
debug = _logger.debug
info = _logger.info
warning = _logger.warning
error = _logger.error
critical = _logger.critical
exception = _logger.exception
