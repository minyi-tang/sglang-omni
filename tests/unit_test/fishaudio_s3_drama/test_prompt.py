# SPDX-License-Identifier: Apache-2.0
"""S3-drama reference-bracket-v1 prompt rows: byte layout, speaker rules, rejections."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal, get_args

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers

from sglang_omni.models.fishaudio_s3_drama.prompt import (
    AudioCodesPart,
    MessageRole,
    PromptMessage,
    ReferencePart,
    TextPart,
    build_prompt_rows,
)

NUM_CODEBOOKS = 10
CODEBOOK_SIZE = 4096
# Protocol literals are written out here, not imported, so a changed literal fails.
SYSTEM = "<|im_start|>system\n"
USER = "<|im_start|>user\n"
ASSISTANT = "<|im_start|>assistant\n"
VOICE_TURN = "<|im_start|>assistant\n<|voice|>"
END = "<|im_end|>\n"
TTS_INSTRUCTION = "convert the provided text to speech"
REFERENCE_INSTRUCTION = (
    "convert the provided text to speech reference to the following:"
)
DEFAULT_SYSTEM_TURN: list[str | torch.Tensor] = [SYSTEM, TTS_INSTRUCTION, END]
# Where the text or codes under test sit in a conversation.
TextCarrier = Literal["system", "reference", "assistant", "user"]
CodesCarrier = Literal["user", "reference"]


@pytest.fixture(scope="module")
def toy_tokenizer() -> Tokenizer:
    """Byte-level BPE whose one merge joins two newlines, so a newline seam tells
    per-segment encoding apart from encoding the joined string."""
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocabulary = {symbol: token_id for token_id, symbol in enumerate(alphabet)}
    # Ċ is the byte-level symbol of a newline.
    vocabulary["ĊĊ"] = len(vocabulary)
    tokenizer = Tokenizer(models.BPE(vocab=vocabulary, merges=[("Ċ", "Ċ")]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=False
    )
    tokenizer.add_special_tokens(
        ["<|im_start|>", "<|im_end|>", "<|voice|>"]
        + [f"<|speaker:{speaker_id}|>" for speaker_id in range(16)]
        + [f"<|semantic:{code}|>" for code in range(CODEBOOK_SIZE)]
    )
    return tokenizer


@pytest.fixture(scope="module")
def real_tokenizer() -> Tokenizer:
    """The private tokenizer FISH_S3_DRAMA_TOKENIZER names; skipped when absent."""
    configured_path = os.environ.get("FISH_S3_DRAMA_TOKENIZER")
    if configured_path is None:
        pytest.skip("FISH_S3_DRAMA_TOKENIZER is not set")
    else:
        pass
    tokenizer_path = Path(configured_path).expanduser()
    if not tokenizer_path.is_file():
        pytest.skip(f"FISH_S3_DRAMA_TOKENIZER names no file: {tokenizer_path}")
    else:
        pass
    return Tokenizer.from_file(str(tokenizer_path))


def clip_codes(frame_count: int, seed: int) -> torch.Tensor:
    """[NUM_CODEBOOKS, frame_count] codes whose values name the clip."""
    codebook_offsets = 100 * torch.arange(NUM_CODEBOOKS).unsqueeze(1)
    return seed + codebook_offsets + torch.arange(frame_count)


def said(role: MessageRole, text: str) -> PromptMessage:
    return PromptMessage(role=role, parts=[TextPart(text=text)])


def reference(transcript: str, audio_codes: torch.Tensor) -> ReferencePart:
    return ReferencePart(transcript=transcript, audio_codes=audio_codes)


def system_with(*parts: TextPart | ReferencePart) -> PromptMessage:
    return PromptMessage(role="system", parts=list(parts))


def user_with(*parts: TextPart | AudioCodesPart | ReferencePart) -> PromptMessage:
    return PromptMessage(role="user", parts=list(parts))


def take(audio_codes: torch.Tensor) -> PromptMessage:
    return PromptMessage(
        role="assistant", parts=[AudioCodesPart(audio_codes=audio_codes)]
    )


def reference_header(reference_number: int, transcript_text: str) -> str:
    return f"\n\n[Reference {reference_number}]\nText: {transcript_text}\nSpeech: "


def build(
    tokenizer: Tokenizer, messages: list[PromptMessage], *, should_tag_user_text: bool
) -> torch.Tensor:
    return build_prompt_rows(
        messages,
        tokenizer,
        num_codebooks=NUM_CODEBOOKS,
        codebook_size=CODEBOOK_SIZE,
        should_tag_user_text=should_tag_user_text,
    )


def expected_rows(
    tokenizer: Tokenizer, segments: list[str | torch.Tensor]
) -> torch.Tensor:
    """Rows of hand-written protocol segments: one per token, then one per frame."""
    semantic_begin_id = tokenizer.token_to_id("<|semantic:0|>")
    rows: list[list[int]] = []
    for segment in segments:
        if isinstance(segment, str):
            token_ids = tokenizer.encode(segment).ids
            rows += [[token_id] + [0] * NUM_CODEBOOKS for token_id in token_ids]
        else:
            frames = segment.T.tolist()
            rows += [[semantic_begin_id + frame[0], *frame] for frame in frames]
    return torch.tensor(rows, dtype=torch.long)


DIRECTORS_SCENE = "场景：深夜电台，主播 Alice 疲惫低声。\n请按导演提示重新演绎。"
DIRECTORS_NOTE = "再轻声些，声音和嘴巴再放松、虚一点"
LAYOUT_CASES = [
    pytest.param(
        [said("user", "hello")],
        False,
        DEFAULT_SYSTEM_TURN + [USER, "hello", END, VOICE_TURN],
        id="no-system-message-uses-the-default-instruction",
    ),
    pytest.param(
        [
            system_with(reference("only clip", clip_codes(4, 1))),
            said("user", "read this"),
        ],
        False,
        [SYSTEM, REFERENCE_INSTRUCTION, reference_header(1, "<|speaker:0|> only clip")]
        + [clip_codes(4, 1), END, USER, "read this", END, VOICE_TURN],
        id="one-reference",
    ),
    pytest.param(
        [
            system_with(
                reference("first clip", clip_codes(4, 1)),
                reference("<|speaker:1|> second clip", clip_codes(5, 2)),
                reference("<|speaker:2|> third clip", clip_codes(6, 3)),
            ),
            said("user", "read this"),
        ],
        False,
        [SYSTEM, REFERENCE_INSTRUCTION]
        + [reference_header(1, "<|speaker:0|> first clip"), clip_codes(4, 1)]
        + [reference_header(2, "<|speaker:1|> second clip"), clip_codes(5, 2)]
        + [reference_header(3, "<|speaker:2|> third clip"), clip_codes(6, 3)]
        + [END, USER, "read this", END, VOICE_TURN],
        id="three-references-take-their-speakers-from-the-transcripts",
    ),
    pytest.param(
        [
            system_with(
                reference("First complete clip.", clip_codes(8, 10)),
                reference(
                    "<|speaker:1|> A reply. <|speaker:0|> A switch.", clip_codes(5, 40)
                ),
            ),
            said("user", "hello"),
        ],
        True,
        [SYSTEM, REFERENCE_INSTRUCTION]
        + [reference_header(1, "<|speaker:0|> First complete clip."), clip_codes(8, 10)]
        + [reference_header(2, "<|speaker:1|> A reply. <|speaker:0|> A switch.")]
        + [clip_codes(5, 40), END, USER, "<|speaker:0|> hello", END, VOICE_TURN],
        id="tts-references-keep-an-inner-speaker-switch",
    ),
    pytest.param(
        [
            system_with(
                TextPart(text="两人深夜电台对谈，Alice 疲惫低声。\n参考如下："),
                reference("参考转写", clip_codes(4, 1)),
            ),
            said("user", "再来一次"),
        ],
        False,
        [SYSTEM, "两人深夜电台对谈，Alice 疲惫低声。\n参考如下："]
        + [reference_header(1, "<|speaker:0|> 参考转写"), clip_codes(4, 1), END]
        + [USER, "再来一次", END, VOICE_TURN],
        id="free-form-multi-line-system-text",
    ),
    pytest.param(
        [
            system_with(
                TextPart(text="read this line like a tired detective"),
                reference("clip", clip_codes(3, 10)),
            ),
            said("user", "hello"),
        ],
        False,
        [SYSTEM, "read this line like a tired detective"]
        + [reference_header(1, "<|speaker:0|> clip"), clip_codes(3, 10), END]
        + [USER, "hello", END, VOICE_TURN],
        id="custom-instruction-replaces-only-the-instruction-line",
    ),
    pytest.param(
        [
            said("user", "<|speaker:0|> plain line"),
            take(clip_codes(6, 100)),
            said("user", "<|speaker:0|> [softer versus previous] plain line"),
            take(clip_codes(7, 200)),
            said("user", "<|speaker:0|> now shout it"),
        ],
        True,
        DEFAULT_SYSTEM_TURN
        + [USER, "<|speaker:0|> plain line", END]
        + [VOICE_TURN, clip_codes(6, 100), END]
        + [USER, "<|speaker:0|> [softer versus previous] plain line", END]
        + [VOICE_TURN, clip_codes(7, 200), END]
        + [USER, "<|speaker:0|> now shout it", END, VOICE_TURN],
        id="tts-prior-turns-replay-before-the-current-text",
    ),
    pytest.param(
        [
            user_with(
                TextPart(text="<|speaker:0|> make this one deeper"),
                AudioCodesPart(audio_codes=clip_codes(9, 300)),
            ),
            said("user", "<|speaker:0|> go"),
        ],
        True,
        DEFAULT_SYSTEM_TURN
        + [USER, "<|speaker:0|> make this one deeper", clip_codes(9, 300), END]
        + [USER, "<|speaker:0|> go", END, VOICE_TURN],
        id="user-audio-follows-its-text-without-voice",
    ),
    pytest.param(
        [
            said("system", "scene"),
            said("assistant", "kept as written"),
            said("user", "<|speaker:3|> line"),
        ],
        False,
        [SYSTEM, "scene", END, ASSISTANT, "kept as written", END]
        + [USER, "<|speaker:3|> line", END, VOICE_TURN],
        id="assistant-text-has-no-voice",
    ),
    pytest.param(
        [
            said("system", DIRECTORS_SCENE),
            said("user", "我从来没有那样说过。"),
            take(clip_codes(6, 9)),
            said("user", DIRECTORS_NOTE),
        ],
        False,
        [SYSTEM, DIRECTORS_SCENE, END, USER, "我从来没有那样说过。", END]
        + [VOICE_TURN, clip_codes(6, 9), END, USER, DIRECTORS_NOTE, END, VOICE_TURN],
        id="chat-mode-replays-a-take-and-leaves-a-directors-note-untagged",
    ),
]


@pytest.mark.parametrize(("messages", "should_tag_user_text", "segments"), LAYOUT_CASES)
def test_prompt_layout(
    toy_tokenizer: Tokenizer,
    messages: list[PromptMessage],
    should_tag_user_text: bool,
    segments: list[str | torch.Tensor],
) -> None:
    rows = build(toy_tokenizer, messages, should_tag_user_text=should_tag_user_text)

    assert torch.equal(rows, expected_rows(toy_tokenizer, segments))


@pytest.mark.parametrize(
    ("user_text", "spoken_text"),
    [
        ("hello", "<|speaker:0|> hello"),
        ("<|speaker:2|> hi", "<|speaker:2|> hi"),
        ("<|speaker:15|> x", "<|speaker:15|> x"),
        (" <|speaker:1|> hi", " <|speaker:1|> hi"),
        ("\n<|speaker:1|> hi", "\n<|speaker:1|> hi"),
        (
            "\N{IDEOGRAPHIC SPACE}<|speaker:1|> hi",
            "\N{IDEOGRAPHIC SPACE}<|speaker:1|> hi",
        ),
        ("a <|speaker:3|> b", "<|speaker:0|> a <|speaker:3|> b"),
        # U+001C is not Unicode White_Space, so the tag does not lead the text.
        ("\x1c<|speaker:1|> hi", "<|speaker:0|> \x1c<|speaker:1|> hi"),
    ],
)
def test_tts_mode_tags_untagged_user_text_and_chat_mode_keeps_it(
    toy_tokenizer: Tokenizer, user_text: str, spoken_text: str
) -> None:
    messages = [said("user", user_text)]

    tagged = build(toy_tokenizer, messages, should_tag_user_text=True)
    verbatim = build(toy_tokenizer, messages, should_tag_user_text=False)

    tagged_segments = DEFAULT_SYSTEM_TURN + [USER, spoken_text, END, VOICE_TURN]
    verbatim_segments = DEFAULT_SYSTEM_TURN + [USER, user_text, END, VOICE_TURN]
    assert torch.equal(tagged, expected_rows(toy_tokenizer, tagged_segments))
    assert torch.equal(verbatim, expected_rows(toy_tokenizer, verbatim_segments))


def test_tts_mode_tags_only_user_text(toy_tokenizer: Tokenizer) -> None:
    note_take = PromptMessage(
        role="assistant",
        parts=[TextPart(text="note"), AudioCodesPart(audio_codes=clip_codes(2, 1))],
    )
    messages = [said("system", "scene"), said("user", "line"), note_take]
    messages.append(said("user", "again"))

    rows = build(toy_tokenizer, messages, should_tag_user_text=True)

    segments = [SYSTEM, "scene", END, USER, "<|speaker:0|> line", END]
    segments += [VOICE_TURN, "note", clip_codes(2, 1), END]
    segments += [USER, "<|speaker:0|> again", END, VOICE_TURN]
    assert torch.equal(rows, expected_rows(toy_tokenizer, segments))


@pytest.mark.parametrize(
    ("transcript", "header_text"),
    [
        ("  padded  ", "<|speaker:0|>   padded  "),
        ("  <|speaker:2|> tagged", "<|speaker:2|>   tagged"),
        ("\n<|speaker:3|> line", "<|speaker:3|>\n line"),
        (
            "\N{IDEOGRAPHIC SPACE}<|speaker:4|> x",
            "<|speaker:4|>\N{IDEOGRAPHIC SPACE} x",
        ),
        # U+001C is not Unicode White_Space, so the tag is not leading.
        ("\x1c<|speaker:2|> x", "<|speaker:0|> \x1c<|speaker:2|> x"),
    ],
)
def test_reference_tag_moves_in_front_of_leading_white_space(
    toy_tokenizer: Tokenizer, transcript: str, header_text: str
) -> None:
    messages = [
        system_with(reference(transcript, clip_codes(1, 1))),
        said("user", "hi"),
    ]

    rows = build(toy_tokenizer, messages, should_tag_user_text=False)

    segments = [SYSTEM, REFERENCE_INSTRUCTION, reference_header(1, header_text)]
    segments += [clip_codes(1, 1), END, USER, "hi", END, VOICE_TURN]
    assert torch.equal(rows, expected_rows(toy_tokenizer, segments))


def test_reference_numbers_count_past_nine(toy_tokenizer: Tokenizer) -> None:
    reference_numbers = range(1, 18)
    references = [
        reference(f"clip {reference_number}", clip_codes(1, reference_number))
        for reference_number in reference_numbers
    ]
    messages = [system_with(*references), said("user", "hi")]

    rows = build(toy_tokenizer, messages, should_tag_user_text=False)

    segments: list[str | torch.Tensor] = [SYSTEM, REFERENCE_INSTRUCTION]
    for reference_number in reference_numbers:
        transcript_text = f"<|speaker:0|> clip {reference_number}"
        segments.append(reference_header(reference_number, transcript_text))
        segments.append(clip_codes(1, reference_number))
    segments += [END, USER, "hi", END, VOICE_TURN]
    assert torch.equal(rows, expected_rows(toy_tokenizer, segments))


def test_rows_are_cpu_int64_with_the_semantic_id_then_every_codebook(
    toy_tokenizer: Tokenizer,
) -> None:
    """Rows stay on the CPU even when a caller sets another default device."""
    # Codebook k holds frames [2k, 2k + 1].
    audio_codes = torch.arange(2 * NUM_CODEBOOKS).reshape(NUM_CODEBOOKS, 2)
    messages = [
        user_with(TextPart(text="say this"), AudioCodesPart(audio_codes=audio_codes))
    ]

    with torch.device("meta"):
        rows = build(toy_tokenizer, messages, should_tag_user_text=False)

    semantic_begin_id = toy_tokenizer.token_to_id("<|semantic:0|>")
    is_audio_row = rows[:, 1:].any(dim=1)
    assert rows.dtype == torch.long
    assert rows.device.type == "cpu"
    assert rows.shape[1] == 11
    assert rows[is_audio_row].tolist() == [
        [semantic_begin_id, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18],
        [semantic_begin_id + 1, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
    ]
    text_segments = DEFAULT_SYSTEM_TURN + [USER, "say this", END, VOICE_TURN]
    text_token_ids = [toy_tokenizer.encode(segment).ids for segment in text_segments]
    assert rows[~is_audio_row, 0].tolist() == sum(text_token_ids, [])


@pytest.mark.parametrize(
    ("messages", "segments", "joined_segments"),
    [
        pytest.param(
            [
                system_with(
                    TextPart(text="read it slowly\n"),
                    reference("clip", clip_codes(1, 1)),
                ),
                said("user", "hi"),
            ],
            [SYSTEM, "read it slowly\n", reference_header(1, "<|speaker:0|> clip")]
            + [clip_codes(1, 1), END, USER, "hi", END, VOICE_TURN],
            [SYSTEM, "read it slowly\n" + reference_header(1, "<|speaker:0|> clip")]
            + [clip_codes(1, 1), END, USER, "hi", END, VOICE_TURN],
            id="instruction-then-reference-header",
        ),
        pytest.param(
            [said("user", "\nline")],
            DEFAULT_SYSTEM_TURN + [USER, "\nline", END, VOICE_TURN],
            DEFAULT_SYSTEM_TURN + [USER + "\nline", END, VOICE_TURN],
            id="role-header-then-text",
        ),
        pytest.param(
            [user_with(TextPart(text="a\n"), TextPart(text="\nb"))],
            DEFAULT_SYSTEM_TURN + [USER, "a\n", "\nb", END, VOICE_TURN],
            DEFAULT_SYSTEM_TURN + [USER, "a\n\nb", END, VOICE_TURN],
            id="adjacent-text-parts",
        ),
    ],
)
def test_each_literal_and_part_is_tokenized_on_its_own(
    toy_tokenizer: Tokenizer,
    messages: list[PromptMessage],
    segments: list[str | torch.Tensor],
    joined_segments: list[str | torch.Tensor],
) -> None:
    rows = build(toy_tokenizer, messages, should_tag_user_text=False)

    assert torch.equal(rows, expected_rows(toy_tokenizer, segments))
    assert not torch.equal(rows, expected_rows(toy_tokenizer, joined_segments))


@pytest.mark.parametrize(
    "text",
    [
        "Direction: [wearily] keep it under your breath\n"
        "Line: <whisper>I never said that.</whisper>",
        "first chunk<|endofturn|>lookahead",
        "hi <|audio_separator|> there",
        "<|fim_prefix|> hi",
        "<|speaker:15|> x",
        # U+001F is not Unicode White_Space, so this text is not empty.
        "\x1f",
    ],
)
def test_other_markup_is_caller_content(toy_tokenizer: Tokenizer, text: str) -> None:
    rows = build(toy_tokenizer, [said("user", text)], should_tag_user_text=False)

    segments = DEFAULT_SYSTEM_TURN + [USER, text, END, VOICE_TURN]
    assert torch.equal(rows, expected_rows(toy_tokenizer, segments))


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_codes_take_both_ends_of_the_codebook_in_int32_and_int64(
    toy_tokenizer: Tokenizer, dtype: torch.dtype
) -> None:
    audio_codes = torch.tensor([[0, CODEBOOK_SIZE - 1]] * NUM_CODEBOOKS, dtype=dtype)
    messages = [user_with(TextPart(text="hi"), AudioCodesPart(audio_codes=audio_codes))]

    rows = build(toy_tokenizer, messages, should_tag_user_text=False)

    segments = DEFAULT_SYSTEM_TURN + [USER, "hi", audio_codes, END, VOICE_TURN]
    assert rows.dtype == torch.long
    assert torch.equal(rows, expected_rows(toy_tokenizer, segments))


BLANK_TEXT = " \n\t\N{IDEOGRAPHIC SPACE}"
REJECTED_CASES = [
    pytest.param([], "messages must not be empty", id="no-messages"),
    pytest.param(
        [said("user", "hi"), said("system", "s")],
        "a system message is only allowed as the first message",
        id="system-not-first",
    ),
    pytest.param(
        [said("system", "s"), said("system", "s"), said("user", "hi")],
        "a system message is only allowed as the first message",
        id="two-system-messages",
    ),
    pytest.param(
        [said("user", "hi"), said("assistant", "done")],
        "the last message must be a user message",
        id="ends-on-an-assistant-message",
    ),
    pytest.param(
        [said("user", "hi"), said("tool", "x"), said("user", "hi")],
        "messages[1]: unknown role 'tool'",
        id="unknown-role",
    ),
    pytest.param([user_with()], "messages[0]: parts must not be empty", id="no-parts"),
    pytest.param(
        [user_with(TextPart(text="ok"), TextPart(text=BLANK_TEXT))],
        "messages[0].parts[1]: text must not be empty",
        id="blank-text-part",
    ),
    pytest.param(
        [system_with(reference(BLANK_TEXT, clip_codes(1, 1))), said("user", "hi")],
        "messages[0].parts[0]: text must not be empty",
        id="blank-transcript",
    ),
    pytest.param(
        [user_with(TextPart(text="hi"), reference("clip", clip_codes(1, 1)))],
        "messages[0].parts[1]: a reference part is only allowed in the system message",
        id="reference-outside-system",
    ),
    pytest.param(
        [
            PromptMessage(
                role="system", parts=[AudioCodesPart(audio_codes=clip_codes(1, 1))]
            ),
            said("user", "hi"),
        ],
        "messages[0].parts[0]: the system message takes text and reference parts only",
        id="audio-in-system",
    ),
    pytest.param(
        [
            system_with(
                TextPart(text="scene"),
                reference("clip", clip_codes(1, 1)),
                TextPart(text="more"),
            ),
            said("user", "hi"),
        ],
        "messages[0].parts[2]: system text must come before reference parts",
        id="system-text-after-reference",
    ),
]


@pytest.mark.parametrize(("messages", "error"), REJECTED_CASES)
def test_malformed_conversations_are_rejected(
    toy_tokenizer: Tokenizer, messages: list[PromptMessage], error: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(error)):
        build(toy_tokenizer, messages, should_tag_user_text=False)


def conversation_carrying(text: str, carrier: TextCarrier) -> list[PromptMessage]:
    """A valid conversation whose system, reference, assistant or user text is text."""
    return [
        system_with(
            TextPart(text=text if carrier == "system" else "scene"),
            reference(text if carrier == "reference" else "clip", clip_codes(1, 1)),
        ),
        said("user", "line"),
        said("assistant", text if carrier == "assistant" else "reply"),
        said("user", text if carrier == "user" else "again"),
    ]


STRUCTURAL_TOKENS = [
    "<|im_start|>",
    "<|im_end|>",
    "<|voice|>",
    "<|text|>",
    "<|interleave|>",
    "<|semantic:3|>",
]
MALFORMED_SPEAKER_TEXTS = [
    "<|speaker:99|> x",
    "<|speaker:16|> x",
    "a <|speaker:99|> b",
    "<|speaker:01|> x",
    "<|speaker:|> x",
    "<|speaker0|> x",
    "hi <|speaker",
    "<|speaker:abc|> x",
    "<|speaker:1> x",
    "  <|speaker:abc|> hi",
]


@pytest.mark.parametrize("carrier", get_args(TextCarrier))
@pytest.mark.parametrize("token", STRUCTURAL_TOKENS)
def test_structural_tokens_in_caller_text_are_rejected(
    toy_tokenizer: Tokenizer, carrier: TextCarrier, token: str
) -> None:
    messages = conversation_carrying(f"hi {token} there", carrier)

    with pytest.raises(ValueError, match="is written by the prompt builder") as error:
        build(toy_tokenizer, messages, should_tag_user_text=True)
    assert token.removesuffix("3|>") in str(error.value)


@pytest.mark.parametrize("carrier", get_args(TextCarrier))
@pytest.mark.parametrize("text", MALFORMED_SPEAKER_TEXTS)
def test_speaker_tags_the_tokenizer_lacks_are_rejected(
    toy_tokenizer: Tokenizer, carrier: TextCarrier, text: str
) -> None:
    messages = conversation_carrying(text, carrier)

    with pytest.raises(ValueError, match="speaker tags must be written"):
        build(toy_tokenizer, messages, should_tag_user_text=True)


def codes_with_one_value(codebook_index: int, value: int) -> torch.Tensor:
    audio_codes = torch.zeros((NUM_CODEBOOKS, 2), dtype=torch.long)
    audio_codes[codebook_index, 1] = value
    return audio_codes


@pytest.mark.parametrize("carrier", get_args(CodesCarrier))
@pytest.mark.parametrize(
    ("audio_codes", "error"),
    [
        pytest.param(clip_codes(1, 1)[:-1], "int32 or int64", id="nine-codebooks"),
        pytest.param(
            torch.zeros((NUM_CODEBOOKS, 0), dtype=torch.long),
            "int32 or int64",
            id="no-frames",
        ),
        pytest.param(clip_codes(1, 1)[0], "int32 or int64", id="one-dimension"),
        pytest.param(clip_codes(1, 1).float(), "int32 or int64", id="float-codes"),
        pytest.param(codes_with_one_value(0, -1), "in [0, 4096)", id="negative"),
        pytest.param(
            codes_with_one_value(0, CODEBOOK_SIZE), "in [0, 4096)", id="codebook-size"
        ),
        pytest.param(
            codes_with_one_value(5, CODEBOOK_SIZE), "in [0, 4096)", id="later-codebook"
        ),
    ],
)
def test_malformed_codes_are_rejected(
    toy_tokenizer: Tokenizer,
    carrier: CodesCarrier,
    audio_codes: torch.Tensor,
    error: str,
) -> None:
    if carrier == "reference":
        messages = [system_with(reference("clip", audio_codes)), said("user", "hi")]
    else:
        part = AudioCodesPart(audio_codes=audio_codes)
        messages = [user_with(TextPart(text="hi"), part)]

    with pytest.raises(ValueError, match=re.escape(error)):
        build(toy_tokenizer, messages, should_tag_user_text=False)


@pytest.mark.parametrize(
    ("num_codebooks", "codebook_size"), [(0, CODEBOOK_SIZE), (NUM_CODEBOOKS, 0)]
)
def test_codebook_parameters_must_be_positive(
    toy_tokenizer: Tokenizer, num_codebooks: int, codebook_size: int
) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        build_prompt_rows(
            [said("user", "hi")],
            toy_tokenizer,
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            should_tag_user_text=False,
        )


def test_real_tokenizer_special_tokens_are_single_contiguous_ids(
    real_tokenizer: Tokenizer,
) -> None:
    for token in ["<|im_start|>", "<|im_end|>", "<|voice|>"]:
        assert real_tokenizer.encode(token).ids == [real_tokenizer.token_to_id(token)]
    speaker_begin_id = real_tokenizer.token_to_id("<|speaker:0|>")
    for speaker_id in range(16):
        speaker_token_ids = real_tokenizer.encode(f"<|speaker:{speaker_id}|>").ids
        assert speaker_token_ids == [speaker_begin_id + speaker_id]
    semantic_begin_id = real_tokenizer.token_to_id("<|semantic:0|>")
    for code in range(CODEBOOK_SIZE):
        semantic_token_ids = real_tokenizer.encode(f"<|semantic:{code}|>").ids
        assert semantic_token_ids == [semantic_begin_id + code]
    assert real_tokenizer.token_to_id("<|speaker:16|>") is None
    assert real_tokenizer.token_to_id(f"<|semantic:{CODEBOOK_SIZE}|>") is None


def test_real_tokenizer_rows_follow_the_segment_boundaries(
    real_tokenizer: Tokenizer,
) -> None:
    header = reference_header(1, "<|speaker:0|> clip")
    messages = [
        system_with(
            TextPart(text="Read it slowly.\n"), reference("clip", clip_codes(2, 1))
        ),
        user_with(TextPart(text="\nfirst line\n"), TextPart(text="\nsecond line")),
    ]

    chat_rows = build(real_tokenizer, messages, should_tag_user_text=False)
    tts_rows = build(real_tokenizer, [said("user", "hello")], should_tag_user_text=True)

    segments = [SYSTEM, "Read it slowly.\n", header, clip_codes(2, 1), END, USER]
    segments += ["\nfirst line\n", "\nsecond line", END, VOICE_TURN]
    assert torch.equal(chat_rows, expected_rows(real_tokenizer, segments))
    seams = [
        ("Read it slowly.\n", header),
        (USER, "\nfirst line\n"),
        ("\nfirst line\n", "\nsecond line"),
    ]
    for left, right in seams:
        joined_token_ids = real_tokenizer.encode(left + right).ids
        left_token_ids = real_tokenizer.encode(left).ids
        right_token_ids = real_tokenizer.encode(right).ids
        assert joined_token_ids != left_token_ids + right_token_ids
    assert chat_rows.shape[1] == NUM_CODEBOOKS + 1
    assert chat_rows[0, 0] == real_tokenizer.token_to_id("<|im_start|>")
    assert chat_rows[-1, 0] == real_tokenizer.token_to_id("<|voice|>")
    tts_segments = DEFAULT_SYSTEM_TURN + [USER, "<|speaker:0|> hello", END, VOICE_TURN]
    assert torch.equal(tts_rows, expected_rows(real_tokenizer, tts_segments))
