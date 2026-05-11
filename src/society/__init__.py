"""Diverse Actor-Critic Society package.

Submodules are intentionally not imported eagerly.  The natural prompt layer
uses ``src.society.agent_registry`` for enum definitions, and eager package
exports would create circular imports between prompts and deliberation.
"""
