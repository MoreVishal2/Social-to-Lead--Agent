"""
tools/lead_capture.py
Mock lead capture tool for AutoStream agent.
Only called when all three fields (name, email, platform) are collected.
"""

import json
from datetime import datetime


def mock_lead_capture(name: str, email: str, platform: str) -> dict:
    """
    Simulates capturing a high-intent lead into a CRM/database.
    
    Args:
        name:     Full name of the lead
        email:    Email address of the lead
        platform: Creator platform (YouTube, Instagram, TikTok, etc.)

    Returns:
        dict with success status and confirmation details
    """
    timestamp = datetime.now().isoformat()

    # Simulated CRM write
    lead_record = {
        "id": f"LEAD-{abs(hash(email)) % 100000:05d}",
        "name": name,
        "email": email,
        "platform": platform,
        "captured_at": timestamp,
        "source": "AutoStream Conversational Agent",
        "intent": "high",
        "plan_interest": "Pro",
    }

    # Console output (as required by spec)
    print(f"\n{'='*55}")
    print(f"  ✅  Lead captured successfully!")
    print(f"{'='*55}")
    print(f"  Name      : {name}")
    print(f"  Email     : {email}")
    print(f"  Platform  : {platform}")
    print(f"  Lead ID   : {lead_record['id']}")
    print(f"  Timestamp : {timestamp}")
    print(f"{'='*55}\n")

    return {
        "success": True,
        "lead_id": lead_record["id"],
        "message": f"Lead captured successfully for {name}",
        "record": lead_record,
    }
