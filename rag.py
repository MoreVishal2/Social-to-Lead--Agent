"""
agent/rag.py
RAG (Retrieval-Augmented Generation) pipeline for AutoStream agent.
Loads the local JSON knowledge base and builds a formatted context string
that is injected into the LLM system prompt at runtime.
"""

import json
from pathlib import Path


def load_knowledge_base() -> dict:
    """Load and return the AutoStream knowledge base from JSON."""
    kb_path = Path(__file__).parent / "autostream_kb.json"
    with open(kb_path, "r") as f:
        return json.load(f)


def build_rag_context() -> str:
    """
    Convert the knowledge base into a structured text context
    for injection into the LLM system prompt.
    """
    kb = load_knowledge_base()

    company = kb["company"]
    pricing = kb["pricing"]
    policies = kb["policies"]

    basic = pricing["basic_plan"]
    pro = pricing["pro_plan"]

    context = f"""
=== AUTOSTREAM KNOWLEDGE BASE ===

COMPANY
-------
Name: {company['name']}
Tagline: {company['tagline']}
Description: {company['description']}

PRICING PLANS
-------------
1. {basic['name']} — ${basic['price_monthly']}/month
   Features:
   {chr(10).join(f'   - {f}' for f in basic['features'])}

2. {pro['name']} — ${pro['price_monthly']}/month
   Features:
   {chr(10).join(f'   - {f}' for f in pro['features'])}

COMPANY POLICIES
----------------
- Refund Policy: {policies['refund_policy']}
- Support Policy: {policies['support_policy']}
=================================
""".strip()

    return context


def retrieve_relevant_context(user_query: str) -> str:
    """
    Simple keyword-based retrieval to return the most relevant section.
    For this project, we return the full KB context since it's compact.
    In production, this would use vector similarity search (e.g., FAISS, Chroma).
    """
    return build_rag_context()
