# AutoStream AI Agent 🎬
A Conversational AI Agent for AutoStream (a fictional SaaS platform offering automated video editing tools for content creators) . 
Built with LangGraph, gemini-embedding-001, gemini-2.5-flash, and a RAG pipeline backed by a local JSON knowledge base.

Intent classification : LLM-powered (greeting / product_qa / high_intent) 
Product Q&A : RAG pipeline from `knowledge_base/autostream_kb.json` 
Lead capture : Fires `mock_lead_capture()` only when all 3 fields collected 
Multi-turn memory : Full conversation history passed in every LLM call 

---

## Quickstart

### 1. Clone and enter the project
```bash
git clone https://github.com/MoreVishal2/Social-to-Lead--Agent
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
### 4. Create .env and set your GOOGLE_API_KEY 
```bash
GOOGLE_API_KEY= Your_API-Key #inside .env
```
### 5. Run the agent
```bash
python main.py
```

---

## Architecture

LangGraph was chosen over AutoGen because it provides explicit, inspectable state machines. It allows you to define exactly what happens in each node and when edges are triggered. This maps cleanly onto the requirement of a classify → respond loop with a strictly guarded tool call.

The AgentState TypedDict carries the entire conversation across turns — including the full messages list, the detected current_intent, individual lead fields (lead_name, lead_email, lead_platform), a lead_captured boolean, and awaiting_field to track which piece of information the agent last requested.

LangGraph merges node return values into this state dictionary after every step. The full conversation history is passed to the LLM on each turn, allowing Gemini to maintain context across all turns without requiring an external memory store.

The knowledge base is stored in autostream_kb.json, and the rag.py module loads it at runtime.

Intent is classified using vector embeddings. User messages are converted into embeddings and compared with example sentences for each intent category (greeting, product_qa, high_intent) using cosine similarity.

---

## WhatsApp Integration via Webhooks

To integrate this agent with WhatsApp, we use Meta's WhatsApp Cloud API 
with a webhook-based architecture. The agent is deployed as a FastAPI server 
which exposes two endpoints — a GET endpoint for one-time verification with 
Meta, and a POST endpoint that receives incoming messages.

Every time a user sends a WhatsApp message, Meta automatically POSTs the 
message payload to our registered URL. The server extracts the phone number 
and message text, loads that user's conversation state, runs it through the 
LangGraph agent, and sends the reply back to Meta's Graph API which delivers 
it to the user on WhatsApp. Each user's AgentState is stored independently 
keyed by phone number, so multiple users can have separate conversations 
simultaneously.

For deployment, the server can either be hosted on Railway.app (permanent 
URL, no PC required) or run locally using uvicorn and exposed via ngrok .
Environment variables are never stored in GitHub — 
they are configured directly in the deployment dashboard.