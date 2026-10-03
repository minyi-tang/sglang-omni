# SPDX-License-Identifier: Apache-2.0
"""Build S3-drama prompt rows: reference-bracket-v1 chat messages and speech infill."""

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
INFILL_INSTRUCTION = (
    "fill in the missing part of the speech; the suffix follows <|fim_suffix|> and "
    "the prefix follows <|fim_prefix|>"
)
MESSAGE_END = "<|im_end|>\n"
GENERATION_START = "<|im_start|>assistant\n<|voice|>"
AUDIO_SEPARATOR = "\n<|audio_separator|>\n"
SEMANTIC_BEGIN_TOKEN = "<|semantic:0|>"
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
class AudioSeparatorPart:
    """Separates the lines of a user turn or the takes of an assistant turn."""


PromptPart = TextPart | AudioCodesPart | ReferencePart | AudioSeparatorPart


@dataclass(kw_only=True)
class PromptMessage:
    role: MessageRole
    parts: list[PromptPart]


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


def with_speaker_tag(text: str, default_speaker_id: int) -> str:
    """Tag text with default_speaker_id unless a tag leads it after white space."""
    if SPEAKER_TAG_PATTERN.match(text.lstrip(UNICODE_WHITE_SPACE)) is None:
        return f"<|speaker:{default_speaker_id}|> {text}"
    else:
        return text


def references_from_speaker_groups(
    speaker_groups: list[list[ReferencePart]],
) -> list[ReferencePart]:
    """Flatten grouped references; an untagged transcript takes its group's speaker id.

    Empty groups take no speaker id.
    """
    non_empty_groups = [group for group in speaker_groups if group]
    return [
        ReferencePart(
            transcript=with_speaker_tag(reference.transcript, speaker_id),
            audio_codes=reference.audio_codes,
        )
        for speaker_id, group in enumerate(non_empty_groups)
        for reference in group
    ]


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
            elif (
                isinstance(part, AudioCodesPart | AudioSeparatorPart)
                and message.role == "system"
            ):
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

            if isinstance(part, TextPart | AudioSeparatorPart):
                if isinstance(part, TextPart):
                    validate_caller_text(part.text, location)
                    is_spoken_line = message.role == "user" and should_tag_user_text
                    text = (
                        with_speaker_tag(part.text, 0) if is_spoken_line else part.text
                    )
                else:
                    text = AUDIO_SEPARATOR
                previous_part = (
                    message.parts[part_index - 1] if part_index > 0 else None
                )
                # note (SunskyXH): the separated lines of a user turn tokenize as one
                # string, so a separator joins the text on both sides.
                if message.role == "user" and (
                    isinstance(previous_part, AudioSeparatorPart)
                    or (
                        isinstance(part, AudioSeparatorPart)
                        and isinstance(previous_part, TextPart)
                    )
                ):
                    previous_text = segments.pop()
                    assert isinstance(previous_text, str)
                    segments.append(previous_text + text)
                else:
                    segments.append(text)
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
                tagged_transcript = with_speaker_tag(part.transcript, 0)
                transcript_body = tagged_transcript.lstrip(UNICODE_WHITE_SPACE)
                leading_speaker_tag = SPEAKER_TAG_PATTERN.match(transcript_body)
                assert leading_speaker_tag is not None
                leading_white_space = tagged_transcript[
                    : len(tagged_transcript) - len(transcript_body)
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


def build_infill_prompt_rows(
    transcript: str,
    tokenizer: Tokenizer,
    *,
    prefix_audio_codes: torch.Tensor,
    suffix_audio_codes: torch.Tensor,
    target_frame_count: int | None,
    num_codebooks: int,
    codebook_size: int,
) -> torch.Tensor:
    """Encode a speech infill prompt whose answer fills the gap between kept codes.

    The suffix codes come before the prefix codes, so the prefix touches the
    generated frames.

    Args:
        transcript: the text of the whole take; line breaks fold into spaces.
        prefix_audio_codes: the kept codes before the gap, [num_codebooks, frames];
            like suffix_audio_codes, a side without frames is left out.
        target_frame_count: the length hint for the gap; None or 0 gives no hint.
    """
    # note (SunskyXH): the model reads an infill transcript as a single line.
    folded_transcript = re.sub(r"\r\n?|\n", " ", transcript.strip(UNICODE_WHITE_SPACE))
    validate_caller_text(folded_transcript, "transcript")
    user_parts: list[PromptPart] = [
        TextPart(text=with_speaker_tag(folded_transcript, 0))
    ]
    if target_frame_count is not None and (
        isinstance(target_frame_count, bool)
        or not isinstance(target_frame_count, int)
        or target_frame_count < 0
    ):
        raise ValueError(
            f"target_frame_count must be a non-negative int or None; got "
            f"{target_frame_count!r}"
        )
    elif target_frame_count is None or target_frame_count == 0:
        pass
    else:
        user_parts.append(TextPart(text=f"[audio tokens: {target_frame_count}]"))
    user_parts.append(TextPart(text="<|fim_suffix|>"))
    if suffix_audio_codes.numel() > 0:
        user_parts.append(AudioCodesPart(audio_codes=suffix_audio_codes))
    else:
        pass
    user_parts.append(TextPart(text="<|fim_prefix|>"))
    if prefix_audio_codes.numel() > 0:
        user_parts.append(AudioCodesPart(audio_codes=prefix_audio_codes))
    else:
        pass
    messages = [
        PromptMessage(role="system", parts=[TextPart(text=INFILL_INSTRUCTION)]),
        PromptMessage(role="user", parts=user_parts),
    ]
    return build_prompt_rows(
        messages,
        tokenizer,
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
        should_tag_user_text=False,
    )
