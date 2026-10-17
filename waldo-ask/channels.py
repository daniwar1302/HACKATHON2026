"""Ways to reach the operator through Inkbox: email, iMessage, SMS.

A channel sends one question and later lists the operator's replies to it.
Only replies from the configured operator address, sent after the question,
count; everything else is ignored.
"""
from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass
class Sent:
    """What a channel needs to find replies to one question later."""
    channel: str
    code: str
    sent_at: datetime
    ref: str | None = None  # email thread id / iMessage conversation id


class Channel(Protocol):
    name: str

    def send(self, code: str, subject: str, body: str, image: bytes | None) -> Sent: ...

    def replies(self, sent: Sent) -> list[str]:
        """Operator replies to `sent`, oldest first."""


_QUOTE_START = re.compile(r"^(On .+ wrote:|-----Original Message-----|From: )", re.I)


def strip_quoted(body: str) -> str:
    """Drop the quoted original from an email reply, keeping what the operator typed."""
    kept = []
    for line in body.splitlines():
        if _QUOTE_START.match(line.strip()):
            break
        if not line.lstrip().startswith(">"):
            kept.append(line)
    return "\n".join(kept).strip()


def _same_address(a: str | None, b: str) -> bool:
    if not a:
        return False
    match = re.search(r"<([^>]+)>", a)
    return (match.group(1) if match else a).strip().lower() == b.strip().lower()


def _digits(number: str | None) -> str:
    return re.sub(r"\D", "", number or "")


class EmailChannel:
    name = "email"

    def __init__(self, identity, to: str):
        self.identity, self.to = identity, to

    def send(self, code, subject, body, image):
        attachments, html = None, None
        if image:
            attachments = [{
                "filename": "waldo.jpg",
                "content_type": "image/jpeg",
                "content_base64": base64.b64encode(image).decode(),
                "content_id": "waldo-frame",
            }]
            html = "<p>" + body.replace("\n", "<br>") + '</p><img src="cid:waldo-frame" width="480">'
        msg = self.identity.send_email(
            to=[self.to], subject=f"{subject} [{code}]", body_text=body, body_html=html,
            attachments=attachments, idempotency_key=f"waldo-ask-{code}",
        )
        return Sent(self.name, code, msg.created_at, str(msg.thread_id) if msg.thread_id else None)

    def replies(self, sent):
        from inkbox.mail.types import MessageDirection

        found = []
        for msg in self.identity.iter_emails(direction=MessageDirection.INBOUND, page_size=20):
            if msg.created_at <= sent.sent_at:
                break  # newest first; everything after this is older than the question
            if not _same_address(msg.from_address, self.to):
                continue
            in_thread = sent.ref is not None and str(msg.thread_id) == sent.ref
            if not in_thread and sent.code not in (msg.subject or ""):
                continue
            detail = self.identity.get_message(str(msg.id))
            text = strip_quoted(detail.body_text or msg.snippet or "")
            if text:
                found.append(text)
        return list(reversed(found))


class IMessageChannel:
    name = "imessage"

    def __init__(self, identity, to: str):
        self.identity, self.to = identity, to

    def send(self, code, subject, body, image):
        media = None
        if image:
            upload = self.identity.upload_imessage_media(content=image, filename="waldo.jpg", content_type="image/jpeg")
            media = [upload.media_url]
        msg = self.identity.send_imessage(
            to=self.to, text=f"{body}\n({code})", media_urls=media, idempotency_key=f"waldo-ask-{code}",
        )
        return Sent(self.name, code, msg.created_at, str(msg.conversation_id))

    def replies(self, sent):
        msgs = self.identity.list_imessages(conversation_id=sent.ref, limit=20)
        found = [
            m.content for m in msgs
            if m.direction == "inbound" and m.created_at > sent.sent_at and m.content
            and _digits(m.remote_number) == _digits(self.to) and not m.is_blocked
        ]
        return list(reversed(found))


class SmsChannel:
    """Plain SMS. No photo: MMS needs a public image URL, which Waldo doesn't have."""
    name = "sms"

    def __init__(self, identity, to: str):
        self.identity, self.to = identity, to

    def send(self, code, subject, body, image):
        msg = self.identity.send_text(to=self.to, text=f"{body}\n({code})", idempotency_key=f"waldo-ask-{code}")
        return Sent(self.name, code, msg.created_at)

    def replies(self, sent):
        msgs = self.identity.list_texts(limit=20)
        found = [
            m.text for m in msgs
            if m.direction == "inbound" and m.created_at > sent.sent_at and m.text
            and _digits(m.remote_phone_number) == _digits(self.to)
        ]
        return list(reversed(found))


class ChannelError(RuntimeError):
    """The operator could not be reached (call not answered, failed, etc.)."""


CALL_DONE = {"completed", "ended", "failed", "no_answer", "busy", "canceled", "cancelled"}


class CallChannel:
    """Phone call through Inkbox's hosted voice agent, over the shared iMessage line.

    The voice agent is briefed with the question; the answer is what the operator
    said, read from the transcript once the call ends. The photo goes by iMessage.
    """
    name = "call"

    def __init__(self, identity, calls, to: str):
        self.identity, self.calls, self.to = identity, calls, to

    def send(self, code, subject, body, image):
        from inkbox import CallOrigin
        from inkbox.phone.types import CallMode

        if image:
            try:
                upload = self.identity.upload_imessage_media(content=image, filename="waldo.jpg", content_type="image/jpeg")
                self.identity.send_imessage(to=self.to, text=f"Photo for my call ({code})", media_urls=[upload.media_url])
            except Exception:
                pass  # the call matters more than the photo
        call = self.identity.place_call(
            to_number=self.to,
            origination=CallOrigin.SHARED_IMESSAGE_NUMBER,
            mode=CallMode.HOSTED_AGENT,
            hosted_agent_authority_mode="contact_scoped",
            on_voicemail="hang_up",
            reason=(
                "You are Waldo, a robot in a robotics lab, calling your operator because you need help. "
                f"Explain briefly, then ask the question and listen to the answer.\n\n{body}\n\n"
                "If the answer is unclear, ask one short follow-up. Repeat the instruction back, thank them, "
                "and hang up. Do not promise anything or take any other action."
            ),
        )
        return Sent(self.name, code, call.created_at, str(call.id))

    def replies(self, sent):
        status = str(self.calls.get(sent.ref).status).split(".")[-1].lower()
        if status not in CALL_DONE:
            return []
        said = [s.text for s in self.identity.list_transcripts(sent.ref) if str(s.party).endswith("remote") and s.text]
        if not said:
            raise ChannelError(f"call {status} without an answer")
        return [" ".join(said)]


def make_channel(kind: str, agent_handle: str, to: str) -> Channel:
    from inkbox import Inkbox

    client = Inkbox()  # key from INKBOX_API_KEY or ~/.inkbox/config
    identity = client.get_identity(agent_handle)
    if kind == "call":
        return CallChannel(identity, client.calls, to)
    channels = {"email": EmailChannel, "imessage": IMessageChannel, "sms": SmsChannel}
    if kind not in channels:
        raise ValueError(f"WALDO_ASK_CHANNEL must be one of {sorted([*channels, 'call'])}")
    return channels[kind](identity, to)
