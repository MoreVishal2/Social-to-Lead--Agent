# AutoStream AI Agent 🎬

A production-grade Conversational AI Agent for **AutoStream**, a fictional SaaS platform offering automated video editing tools for content creators. Built with **LangGraph**, **Gemini 1.5 Flash**, and a **RAG pipeline** backed by a local JSON knowledge base.

---

## Features

| Capability | Implementation |
|---|---|
| Intent classification | LLM-powered (greeting / product_qa / high_intent) |
| Product Q&A | RAG pipeline from `knowledge_base/autostream_kb.json` |
| Lead capture | Fires `mock_lead_capture()` only when all 3 fields collected |
| State management | LangGraph `StateGraph` with `AgentState` TypedDict |
| Multi-turn memory | Full conversation history passed in every LLM call |
| LLM | Gemini 1.5 Flash via `langchain-google-genai` |

---

## Project Structure

```
autostream/
├── main.py                         # CLI entry point
├── requirements.txt
├── .env.example
├── README.md
├── knowledge_base/
│   └── autostream_kb.json          # Pricing, features, policies (RAG source)
├── agent/
│   ├── __init__.py
│   ├── autostream_agent.py         # LangGraph graph, nodes, state, intent classifier
│   └── rag.py                      # RAG pipeline (load KB → build context string)
└── tools/
    ├── __init__.py
    └── lead_capture.py             # mock_lead_capture() tool
```

---

## Quickstart

### 1. Clone and enter the project

```bash
git clone https://github.com/your-username/autostream-agent.git
cd autostream-agent
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your Anthropic API key

```bash
cp .env.example .env
# Open .env and replace the placeholder with your real key:
# GOOGLE_API_KEY=sk-ant-...
```

### 5. Run the agent

```bash
python main.py
```

You'll see the welcome prompt from **Alex**, the AutoStream assistant. Type your messages and press Enter.

---

## Example Conversation

```
👤 You: Hi, can you tell me about your pricing?

🤖 Alex: Hey there! AutoStream has two plans:
  • Basic ($29/mo) — 10 videos/month, 720p, email support
  • Pro ($79/mo)   — Unlimited videos, 4K, AI captions, 24/7 support
Would you like a deeper breakdown of either plan?

👤 You: That Pro plan sounds great. I want to sign up for my YouTube channel.

🤖 Alex: Awesome! I'd love to get you started. Could I get your full name first?

👤 You: Jane Smith

🤖 Alex: Great to meet you, Jane! 👋  What's the best email address to send your sign-up link to?

👤 You: jane@example.com

🤖 Alex: Almost there! Which creator platform do you primarily post on?

👤 You: YouTube

===================================================
  ✅  Lead captured successfully!
===================================================
  Name      : Jane Smith
  Email     : jane@example.com
  Platform  : Youtube
  Lead ID   : LEAD-04221
===================================================

🤖 Alex: 🎉 You're all set, Jane! We'll reach out to jane@example.com shortly...
```

---

## Architecture Explanation (~200 words)

### Why LangGraph?

LangGraph was chosen over AutoGen because it provides **explicit, inspectable state machines** — perfect for an agent that must track multi-step, conditional flows (greet → answer questions → collect lead fields one at a time → fire tool). Unlike an AutoGen multi-agent setup where control flow is emergent and harder to audit, LangGraph lets us define exactly what happens in each node and when edges fire. This maps cleanly onto the assignment's requirement for a `classify → respond` loop with a guarded tool call.

### How State is Managed

The `AgentState` TypedDict carries the entire conversation across turns: the full `messages` list (LangChain `HumanMessage`/`AIMessage` objects), the detected `current_intent`, individual lead fields (`lead_name`, `lead_email`, `lead_platform`), a `lead_captured` boolean flag, and `awaiting_field` to track which piece of information the agent last asked for. LangGraph merges node return values into this state dict after every step using the `add_messages` reducer for the messages list and direct overwrite for scalar fields. Because the full history is passed to the LLM on every turn, Claude maintains conversational context across 5–6+ turns without any external memory store.

### RAG Pipeline

The knowledge base lives in `knowledge_base/autostream_kb.json`. The `rag.py` module loads it at runtime, formats it into a structured text block, and injects it into the LLM system prompt on every call. This ensures the model always answers from ground truth rather than hallucinating prices or features.

---

## WhatsApp Integration via Webhooks

To deploy this agent on WhatsApp, the following approach is recommended:

### Architecture

```
WhatsApp User
    │
    ▼
WhatsApp Business API (Meta Cloud API)
    │  POST /webhook  (incoming message events)
    ▼
FastAPI / Flask Webhook Server  ←── your server (e.g. on Railway, Render, or AWS Lambda)
    │
    ├── Verifies webhook challenge (GET) for Meta's initial handshake
    │
    ├── Receives message payload (POST), extracts sender phone + message text
    │
    ├── Looks up or creates AgentState for that phone number (Redis / DynamoDB)
    │
    ├── Appends HumanMessage, invokes LangGraph graph
    │
    └── POSTs reply back via WhatsApp Cloud API
              https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages
```

### Key Steps

1. **Meta Developer Portal** — Create an app, enable the WhatsApp Business product, obtain a `PHONE_NUMBER_ID` and permanent `WHATSAPP_TOKEN`.

2. **Webhook endpoint** — Build a `/webhook` route:
   - `GET /webhook` — respond to Meta's verification challenge (`hub.challenge`).
   - `POST /webhook` — parse the `entry[0].changes[0].value.messages[0]` payload, extract `from` (phone) and `text.body`.

3. **Per-user state persistence** — Store `AgentState` keyed by phone number in **Redis** (for low-latency) or **DynamoDB** (for serverless). Deserialise on each incoming message, invoke the graph, re-serialise and save.

4. **Reply delivery** — After the graph returns, send the AI message back via:
   ```http
   POST https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages
   Authorization: Bearer {WHATSAPP_TOKEN}
   Content-Type: application/json

   {
     "messaging_product": "whatsapp",
     "to": "{user_phone}",
     "type": "text",
     "text": { "body": "{agent_reply}" }
   }
   ```

5. **Hosting** — Deploy on any HTTPS-capable host (Railway, Render, AWS Lambda + API Gateway). Meta requires a verified HTTPS webhook URL.

6. **Secrets** — Store `GOOGLE_API_KEY`, `WHATSAPP_TOKEN`, and `PHONE_NUMBER_ID` as environment secrets (never in source code).

---

## License

MIT — feel free to adapt for your own SaaS agent projects.
