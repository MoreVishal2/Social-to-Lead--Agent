import os
import re
import numpy as np
from typing import Annotated, TypedDict, Optional
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from rag import retrieve_relevant_context
from lead_capture import mock_lead_capture

load_dotenv()


# ─────────────────────────────────────────────
#  STATE DEFINITION
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # Full conversation history

    # Intent tracking
    current_intent: str    # greeting | product_qa | lead_capture | done

    # Lead capture fields (populated one at a time)
    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]

    # Flag: has the lead been captured (tool fired)?
    lead_captured: bool

    # Which field is the agent currently asking for?
    awaiting_field: Optional[str]             # name | email | platform | None


# ─────────────────────────────────────────────
#  LLM INITIALISATION
# ─────────────────────────────────────────────

def get_llm():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY not set. Please add it to your .env file."
        )
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.4,
        max_output_tokens=1024,
    )


# ─────────────────────────────────────────────
#  INTENT CLASSIFICATION
# ─────────────────────────────────────────────


# ── Intent examples — what each intent "sounds like" ──────────────
INTENT_EXAMPLES = {
    "greeting": [
        "hi", "hello", "hey there", "good morning", "what's up",
        "howdy", "greetings", "hi there"
    ],
    "product_qa": [
        "what are your plans", "tell me about pricing",
        "how much does it cost", "what features do you have",
        "what is the refund policy", "explain the basic plan",
        "what is included in pro", "how does it work",
        "what is the difference between plans"
    ],
    "high_intent": [
        "i want to sign up", "sign me up", "i'll take the pro plan",
        "this looks great i want it", "the pro plan sounds perfect for me",
        "basic is good for me", "i am interested in subscribing",
        "let's get started", "i want the pro", "sounds good i'm in",
        "the pro is better i want that", "that works for me",
        "i'm ready to get started", "i want to try it"
    ]
}

_embedder = None
_intent_embeddings = None

def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    return _embedder

def _build_intent_embeddings():
    global _intent_embeddings
    if _intent_embeddings is not None:
        return
    embedder = _get_embedder()
    _intent_embeddings = {}
    for intent, examples in INTENT_EXAMPLES.items():
        vecs = embedder.embed_documents(examples)
        # Store the mean vector for this intent
        _intent_embeddings[intent] = np.mean(vecs, axis=0)

def classify_intent(user_message: str) -> str:
    """
    Semantic intent classifier using cosine similarity.
    Understands meaning — handles typos, indirect phrasing,
    and unusual word choices without any keyword lists.
    """
    _build_intent_embeddings()
    embedder = _get_embedder()

    user_vec = np.array(embedder.embed_query(user_message))

    best_intent = "product_qa"
    best_score = -1

    for intent, intent_vec in _intent_embeddings.items():
        # Cosine similarity
        score = np.dot(user_vec, intent_vec) / (
            np.linalg.norm(user_vec) * np.linalg.norm(intent_vec)
        )
        if score > best_score:
            best_score = score
            best_intent = intent

    return best_intent

# ─────────────────────────────────────────────
#  HELPER: EXTRACT LEAD FIELDS FROM FREE TEXT
# ─────────────────────────────────────────────

def extract_email(text: str) -> Optional[str]:
    match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    return match.group(0) if match else None


KNOWN_PLATFORMS = [
    "youtube", "instagram", "tiktok", "twitter", "x", "facebook",
    "twitch", "linkedin", "snapchat", "pinterest", "vimeo",
]

def extract_platform(text: str) -> Optional[str]:
    words = text.lower().split()
    for p in KNOWN_PLATFORMS:
        if p in words:
            return p.capitalize()
    return None


# ─────────────────────────────────────────────
#  GRAPH NODES
# ─────────────────────────────────────────────

def classify_node(state: AgentState) -> dict:
    last_human = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        "",
    )

    # Allow user to exit lead_capture if they ask for product info
    escape_phrases = [
        "tell me", "what is", "how much", "explain", "what are",
        "first", "actually", "wait", "no ", "not yet", "hold on",
        "before that", "instead", "can you tell", "i want to know"
    ]
    current = state.get("current_intent", "")
    if current == "done":
        return {}
    if current == "lead_capture":
        msg_lower = last_human.lower()
        if any(phrase in msg_lower for phrase in escape_phrases):
            # User wants info, not to continue sign-up
            return {"current_intent": "product_qa", "awaiting_field": None}
        return {}

    intent = classify_intent(last_human)

    # Map high_intent → lead_capture to trigger the capture flow
    if intent == "high_intent":
        intent = "lead_capture"

    return {"current_intent": intent}



def respond_node(state: AgentState) -> dict:
    llm = get_llm()
    rag_context = retrieve_relevant_context("")
    intent = state.get("current_intent", "product_qa")

    # ── Lead capture flow ──────────────────────────────────────────────
    if intent == "lead_capture":
        last_human = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            "",
        )
        awaiting = state.get("awaiting_field")
        name = state.get("lead_name")
        email = state.get("lead_email")
        platform = state.get("lead_platform")

        # Parse the user's latest reply for the field we were awaiting
        if awaiting == "name" and not name:
            # Extract name: strip common filler words
            candidate = last_human.strip().strip(".,!").title()
            # Heuristic: if it's 1-4 words and no @ sign, accept as name
            if 1 <= len(candidate.split()) <= 4 and "@" not in candidate:
                name = candidate

        if awaiting == "email" and not email:
            extracted = extract_email(last_human)
            if extracted:
                email = extracted

        if awaiting == "platform" and not platform:
            extracted = extract_platform(last_human)
            if extracted:
                platform = extracted
            else:
                # Accept whatever they typed as platform name
                platform = last_human.strip().title()

        # Also try to extract all three from the very first high-intent message
        if not name and not awaiting:
            candidate = last_human.strip().strip(".,!")
            words = candidate.split()
            if 1 <= len(words) <= 4 and "@" not in candidate:
                pass  # don't auto-extract name from first message – ask explicitly

        if not email:
            extracted = extract_email(last_human)
            if extracted:
                email = extracted

        if not platform:
            extracted = extract_platform(last_human)
            if extracted:
                platform = extracted

        updates = {
            "lead_name": name,
            "lead_email": email,
            "lead_platform": platform,
        }

        # Check if we have everything — fire the tool
        if name and email and platform and not state.get("lead_captured"):
            result = mock_lead_capture(name, email, platform)
            reply = (
                        f"🎉 You're all set, **{name}**!\n"
                        f"Welcome to AutoStream! 🚀\n"
                        f"We will reach out to you shortly.\n\n"
                        f"Thank you for taking interest in AutoStream. "
                    )
            

            updates["lead_captured"] = True
            updates["awaiting_field"] = None
            updates["current_intent"] = "done"
            updates["messages"] = [AIMessage(content=reply)]
            return updates



        # Determine which field to ask for next
        if not name:
            # Detect which plan the user mentioned, if any
            last_human_lower = last_human.lower()
            if "basic" in last_human_lower:
                plan_mention = "the **Basic Plan** ($29/month)"
            elif "pro" in last_human_lower:
                plan_mention = "the **Pro Plan** ($79/month)"
            else:
                plan_mention = "an AutoStream plan"
            reply = (
                f"Great choice! 🎬 It sounds like you're interested in {plan_mention}.\n"
                "Would you like me to get you set up? If yes, just share your **full name**!"
            )
            updates["awaiting_field"] = "name"
        elif not email:        
                extracted = extract_email(last_human)
                if awaiting == "email" and not extracted:
                    # User already tried but gave invalid email
                    reply = (f"Hmm, that doesn't look like a valid email address. 🤔\n\n"
                        f"Please enter a valid email in the format **name@example.com** so we can reach you.")
                else:
                    reply = (f"Great to meet you, **{name}**! 👋\n\n"
                        "What's the best **email address** to send your sign-up link to?")
                updates["awaiting_field"] = "email"



        elif not platform:
            reply = (
                "Almost there! Which **creator platform** do you primarily post on? "
                "(e.g. YouTube, Instagram, TikTok…)"
            )
            updates["awaiting_field"] = "platform"
        else:
            reply = "One moment while I get everything set up for you…"

        updates["messages"] = [AIMessage(content=reply)]
        return updates

    # ── General conversation (greeting / product_qa) ───────────────────
    system_prompt = f"""You are Alex, a friendly sales assistant for AutoStream.

STRICT RULES:
- ONLY use the facts in the KNOWLEDGE BASE section below. Never invent prices or features.
- For ANY question about pricing, plans, or policies, quote the exact details from the knowledge base.
- Keep replies concise (2-4 sentences). Be warm and helpful.
- If the user seems interested in signing up, encourage them but do not ask for their details yet.

KNOWLEDGE BASE — use this as your single source of truth:
{rag_context}

IMPORTANT: If asked about plans, always mention BOTH plans with their exact prices and features from above.
"""

    # Build message list for LLM (history + system)
    lc_messages = [SystemMessage(content=system_prompt)] + state["messages"]

    response = llm.invoke(lc_messages)
    return {"messages": [AIMessage(content=response.content)]}


def router(state: AgentState) -> str:
    return "respond"


# ─────────────────────────────────────────────
#  GRAPH ASSEMBLY
# ─────────────────────────────────────────────

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("classify", classify_node)
    graph.add_node("respond", respond_node)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "respond")
    graph.add_edge("respond", END)

    return graph.compile()


# ─────────────────────────────────────────────
#  INITIAL STATE FACTORY
# ─────────────────────────────────────────────

def initial_state() -> AgentState:
    return AgentState(
        messages=[],
        current_intent="",
        lead_name=None,
        lead_email=None,
        lead_platform=None,
        lead_captured=False,
        awaiting_field=None,
    )