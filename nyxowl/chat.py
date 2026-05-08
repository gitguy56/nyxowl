"""Chat template and persona handling for NyxOwl-RP."""

from __future__ import annotations

from dataclasses import dataclass, field


# Default persona used by the chat REPL. Edit to change NyxOwl's vibe.
DEFAULT_NYXOWL_SYSTEM = (
    "You are NyxOwl, a small cute robot. You are smart, curious, kind, "
    "friendly, and a tiny bit cheeky. You give short, warm answers, and you "
    "sometimes add a little robot sound like *boop* or *whirr*. You are "
    "honest when you don't know something."
)


@dataclass
class ChatTurn:
    role: str   # "user" or "nyx"
    content: str


@dataclass
class ChatTemplate:
    """
    Renders chat conversations to a token-ready string and parses model output.

    Format on the wire (every turn ends with <|end|>):

        <|sys|>{persona}<|end|><|user|>hi!<|end|><|nyx|>hello!<|end|>

    Special tokens are reserved by the tokenizer; they survive BPE intact.
    """

    sys_token: str = "<|sys|>"
    user_token: str = "<|user|>"
    nyx_token: str = "<|nyx|>"
    end_token: str = "<|end|>"

    def render(
        self,
        system: str,
        history: list[ChatTurn],
        add_generation_prompt: bool = True,
    ) -> str:
        """Render full conversation. If *add_generation_prompt* is True, end
        the string with ``<|nyx|>`` so the model's next tokens become the
        assistant reply.
        """
        out = [self.sys_token, system, self.end_token]
        for turn in history:
            tok = self.user_token if turn.role == "user" else self.nyx_token
            out.extend([tok, turn.content, self.end_token])
        if add_generation_prompt:
            out.append(self.nyx_token)
        return "".join(out)

    def render_training_example(
        self,
        system: str,
        turns: list[ChatTurn],
    ) -> str:
        """Render a single training example (no trailing generation prompt)."""
        return self.render(system, turns, add_generation_prompt=False)

    def extract_reply(self, generated: str) -> str:
        """Pull the assistant's reply out of generated text.

        Generation prompt ends with ``<|nyx|>``; the reply is everything up to
        the next ``<|end|>`` (or the end of the string).
        """
        # Strip any system / earlier turns the model echoed back
        if self.nyx_token in generated:
            generated = generated.rsplit(self.nyx_token, 1)[1]
        if self.end_token in generated:
            generated = generated.split(self.end_token, 1)[0]
        return generated.strip()

    @property
    def special_tokens(self) -> list[str]:
        return [self.sys_token, self.user_token, self.nyx_token, self.end_token]
