import json
from pathlib import Path


def load_knowledge_base() -> dict:
    kb_path = Path(__file__).parent / "autostream_kb.json"
    with open(kb_path, "r") as f:
        return json.load(f)


def build_rag_context() -> str:
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
    return build_rag_context()
