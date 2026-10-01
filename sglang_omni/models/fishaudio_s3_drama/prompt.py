# SPDX-License-Identifier: Apache-2.0
"""Build S3-drama reference-bracket-v1 prompt rows from chat messages."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, get_args

import torch
from tokenizers import Tokenizer

MessageRole = Literal["system", "user", "assistant"]

TTS_INSTRUCTION = "convert the provided text to speech"
REFERENCE_INSTRUCTION = (
    "convert the provided text to speech reference to the following:"
)
MESSAGE_END = "<|im_end|>\n"
GENERATION_START = "<|im_start|>assistant\n<|voice|>"
SEMANTIC_BEGIN_TOKEN = "<|semantic:0|>"
DEFAULT_SPEAKER_PREFIX = "<|speaker:0|> "
# note (SunskyXH): only the builder writes these; other tokens in caller text
# are content.
STRUCTURAL_TOKENS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|voice|>",
    "<|text|>",
    "<|interleave|>",
    "<|semantic:",
)
MAX_SPEAKER_ID = 15
# note (SunskyXH): the spellings the tokenizer has, <|speaker:0|> to <|speaker:15|>.
SPEAKER_IDS = frozenset(str(speaker_id) for speaker_id in range(MAX_SPEAKER_ID + 1))
SPEAKER_TAG_PATTERN = re.compile(r"<\|speaker:([0-9]+)\|>")
# note (SunskyXH): the Unicode White_Space property. str.strip() with no argument also
# strips U+001C..U+001F, which would change the empty check and the speaker rule.
UNICODE_WHITE_SPACE = "".join(
    map(
        chr,
        (*range(0x9, 0xE), 0x20, 0x85, 0xA0, 0x1680, *range(0x2000, 0x200B))
        + (0x2028, 0x2029, 0x202F, 0x205F, 0x3000),
    )
)


@dataclass(kw_only=True)
class TextPart:
    text: str


@dataclass(kw_only=True)
class AudioCodesPart:
    """VQ codes of one clip: an int32 or int64 [num_codebooks, frames] tensor."""

    audio_codes: torch.Tensor


@dataclass(kw_only=True)
class ReferencePart:
    """A voice reference clip; only the system message takes one."""

    transcript: str
    audio_codes: torch.Tensor


@dataclass(kw_only=True)
class PromptMessage:
    role: MessageRole
    parts: list[TextPart | AudioCodesPart | ReferencePart]


def validate_caller_text(text: str, location: str) -> None:
    """Reject empty text, structural tokens and speaker tags the tokenizer lacks."""
    if not text.strip(UNICODE_WHITE_SPACE):
        raise ValueError(f"{location}: text must not be empty")
    else:
        pass
    for token in STRUCTURAL_TOKENS:
        if token in text:
            raise ValueError(
                f"{location}: {token} is written by the prompt builder and must "
                f"not appear in caller text"
            )
        else:
            pass
    for speaker_marker in re.finditer(r"<\|speaker", text):
        speaker_tag = SPEAKER_TAG_PATTERN.match(text, speaker_marker.start())
        if speaker_tag is None or speaker_tag.group(1) not in SPEAKER_IDS:
            raise ValueError(
                f"{location}: speaker tags must be written <|speaker:N|> with N in "
                f"0..{MAX_SPEAKER_ID} and no leading zero; got {text!r}"
            )
        else:
            pass


def validate_audio_codes(
    audio_codes: torch.Tensor, num_codebooks: int, codebook_size: int, location: str
) -> None:
    if (
        audio_codes.dtype not in (torch.int32, torch.int64)
        or audio_codes.ndim != 2
        or audio_codes.shape[0] != num_codebooks
        or audio_codes.shape[1] == 0
    ):
        raise ValueError(
            f"{location}: audio codes must be an int32 or int64 [{num_codebooks}, "
            f"frames] tensor with at least one frame; got {audio_codes.dtype} "
            f"{tuple(audio_codes.shape)}"
        )
    elif audio_codes.min() < 0 or audio_codes.max() >= codebook_size:
        raise ValueError(f"{location}: audio codes must be in [0, {codebook_size})")
    else:
        pass


def build_prompt_rows(
    messages: list[PromptMessage],
    tokenizer: Tokenizer,
    *,
    num_codebooks: int,
    codebook_size: int,
    should_tag_user_text: bool,
) -> torch.Tensor:
    """Encode messages as CPU int64 rows that end in the open assistant voice turn.

    Each literal and part is tokenized on its own, because tokenizing the joined
    string changes ids at the seams.

    Args:
        should_tag_user_text: True when user text is a line to speak, so text
            without a leading speaker tag gets <|speaker:0|>; False keeps it
            verbatim, since an instruction must stay untagged.
    """
    if num_codebooks <= 0 or codebook_size <= 0:
        raise ValueError(
            f"num_codebooks and codebook_size must be positive; got "
            f"{num_codebooks} and {codebook_size}"
        )
    elif not messages:
        raise ValueError("messages must not be empty")
    elif any(message.role == "system" for message in messages[1:]):
        raise ValueError("a system message is only allowed as the first message")
    elif messages[-1].role != "user":
        raise ValueError("the last message must be a user message")
    else:
        pass
    semantic_begin_id = tokenizer.token_to_id(SEMANTIC_BEGIN_TOKEN)
    if semantic_begin_id is None:
        raise ValueError(f"the tokenizer has no {SEMANTIC_BEGIN_TOKEN} token")
    else:
        pass

    segments: list[str | torch.Tensor] = []
    if messages[0].role == "system":
        pass
    else:
        segments += ["<|im_start|>system\n", TTS_INSTRUCTION, MESSAGE_END]
    for message_index, message in enumerate(messages):
        if message.role not in get_args(MessageRole):
            raise ValueError(
                f"messages[{message_index}]: unknown role {message.role!r}"
            )
        elif not message.parts:
            raise ValueError(f"messages[{message_index}]: parts must not be empty")
        else:
            pass
        has_audio = any(isinstance(part, AudioCodesPart) for part in message.parts)
        is_voice_turn = message.role == "assistant" and has_audio
        segments.append(
            f"<|im_start|>{message.role}\n" + ("<|voice|>" if is_voice_turn else "")
        )
        if message.role == "system" and not any(
            isinstance(part, TextPart) for part in message.parts
        ):
            segments.append(REFERENCE_INSTRUCTION)
        else:
            pass

        reference_count = 0
        for part_index, part in enumerate(message.parts):
            location = f"messages[{message_index}].parts[{part_index}]"
            if isinstance(part, ReferencePart) and message.role != "system":
                raise ValueError(
                    f"{location}: a reference part is only allowed in the system "
                    f"message"
                )
            elif isinstance(part, AudioCodesPart) and message.role == "system":
                raise ValueError(
                    f"{location}: the system message takes text and reference "
                    f"parts only"
                )
            elif (
                isinstance(part, TextPart)
                and message.role == "system"
                and reference_count > 0
            ):
                raise ValueError(
                    f"{location}: system text must come before reference parts"
                )
            else:
                pass

            if isinstance(part, TextPart):
                validate_caller_text(part.text, location)
                text_body = part.text.lstrip(UNICODE_WHITE_SPACE)
                if (
                    message.role == "user"
                    and should_tag_user_text
                    and SPEAKER_TAG_PATTERN.match(text_body) is None
                ):
                    segments.append(DEFAULT_SPEAKER_PREFIX + part.text)
                else:
                    segments.append(part.text)
            elif isinstance(part, AudioCodesPart):
                validate_audio_codes(
                    part.audio_codes, num_codebooks, codebook_size, location
                )
                segments.append(part.audio_codes)
            else:
                validate_caller_text(part.transcript, location)
                validate_audio_codes(
                    part.audio_codes, num_codebooks, codebook_size, location
                )
                reference_count += 1
                # note (SunskyXH): the tag must follow "Text: " directly, so a tag
                # behind leading white space moves in front of it.
                transcript_body = part.transcript.lstrip(UNICODE_WHITE_SPACE)
                leading_speaker_tag = SPEAKER_TAG_PATTERN.match(transcript_body)
                if leading_speaker_tag is None:
                    transcript = DEFAULT_SPEAKER_PREFIX + part.transcript
                else:
                    leading_white_space = part.transcript[
                        : len(part.transcript) - len(transcript_body)
                    ]
                    transcript = (
                        leading_speaker_tag.group(0)
                        + leading_white_space
                        + transcript_body[leading_speaker_tag.end() :]
                    )
                segments += [
                    f"\n\n[Reference {reference_count}]\nText: {transcript}\nSpeech: ",
                    part.audio_codes,
                ]
        segments.append(MESSAGE_END)
    segments.append(GENERATION_START)

    rows: list[torch.Tensor] = []
    for segment in segments:
        if isinstance(segment, str):
            token_ids = torch.tensor(
                tokenizer.encode(segment).ids, dtype=torch.long, device="cpu"
            )
            text_rows = torch.zeros(
                (token_ids.numel(), num_codebooks + 1), dtype=torch.long, device="cpu"
            )
            text_rows[:, 0] = token_ids
            rows.append(text_rows)
        else:
            audio_codes = segment.to(device="cpu", dtype=torch.long)
            rows.append(torch.cat([audio_codes[:1] + semantic_begin_id, audio_codes]).T)
    return torch.cat(rows)
