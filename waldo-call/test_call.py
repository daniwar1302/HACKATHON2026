"""Place one test call to the operator through Inkbox's hosted voice agent.

Run on Waldo: ~/HACKATHON2026/waldo-call/test_call.sh
Uses the shared iMessage line (no phone number needed) and contact_scoped authority.
"""
import os
import sys
import time

from inkbox import CallOrigin, Inkbox
from inkbox.phone.types import CallMode

TO = os.environ.get("WALDO_OPERATOR_PHONE")
if not TO:
    sys.exit("WALDO_OPERATOR_PHONE is not set in ~/.waldo-ask/env")

REASON = (
    "You are Waldo, a robot in a robotics lab, calling your operator to test phone calling. "
    "Say: Hi, this is Waldo. I am testing my calling. In the real version I would tell you I am stuck "
    "lifting a cup and ask where to grab it. Then ask if they can hear you clearly, and ask one question: "
    "should I grab the cup by the handle or the rim? Thank them, say goodbye and hang up. "
    "Keep it under one minute. Do not promise or do anything else."
)

with Inkbox() as client:
    identity = client.get_identity(os.environ.get("WALDO_ASK_IDENTITY", "waldo-agent"))
    call = identity.place_call(
        to_number=TO,
        origination=CallOrigin.SHARED_IMESSAGE_NUMBER,
        mode=CallMode.HOSTED_AGENT,
        hosted_agent_authority_mode="contact_scoped",
        on_voicemail="hang_up",
        reason=REASON,
    )
    print("placed:", call)
    call_id = str(getattr(call, "id", "") or getattr(getattr(call, "call", None), "id", ""))
    if not call_id:
        sys.exit("no call id returned; check the call in the Inkbox console")

    # Wait for the call to finish, then print the transcript.
    for _ in range(60):
        time.sleep(3)
        status = client.calls.get(call_id).status
        print("status:", status)
        if str(status).split(".")[-1].lower() in {"completed", "ended", "failed", "no_answer", "busy", "canceled"}:
            break
    for seg in identity.list_transcripts(call_id):
        print(f"[{seg.party}] {seg.text}")
