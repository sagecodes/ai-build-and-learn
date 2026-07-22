"""
compat.py — Import this before any ragas import to stub broken optional deps.

ragas internally imports ChatVertexAI from langchain_community.chat_models.vertexai.
That module was removed in langchain-community 0.3.x, and the replacement
(langchain_google_vertexai) uses gRPC which crashes on this VM's CPU (no AVX512).
We never call ChatVertexAI — a stub class is all ragas needs to load.
"""

import sys
import types


def _stub(module_name: str, **attrs) -> None:
    if module_name not in sys.modules:
        mod = types.ModuleType(module_name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[module_name] = mod


_stub(
    "langchain_community.chat_models.vertexai",
    ChatVertexAI=type("ChatVertexAI", (), {}),
)
