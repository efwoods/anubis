"""Reference connector stubs for the data-analysis deep agent.

Each module exposes `make_tool(backend)` returning a `@tool` decorated callable
bound to the deepagents backend so its parquet upload uses the active sandbox.
"""
