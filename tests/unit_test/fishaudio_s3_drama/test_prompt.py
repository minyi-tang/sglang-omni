# SPDX-License-Identifier: Apache-2.0
"""S3-drama prompt rows: chat and infill byte layouts, speaker rules, rejections."""

from __future__ import annotations

import itertools
import os
import re
from pathlib import Path
from typing import Literal, get_args

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers

from sglang_omni.models.fishaudio_s3_drama.prompt import (
    AudioCodesPart,
    AudioSeparatorPart,
    MessageRole,
    PromptMessage,
    PromptPart,
    ReferencePart,
    TextPart,
    build_infill_prompt_rows,
    build_prompt_rows,
    references_from_speaker_groups,
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
INFILL_INSTRUCTION = (
    "fill in the missing part of the speech; the suffix follows <|fim_suffix|> and "
    "the prefix follows <|fim_prefix|>"
)
SEPARATOR = "\n<|audio_separator|>\n"
DEFAULT_SYSTEM_TURN: list[str | torch.Tensor] = [SYSTEM, TTS_INSTRUCTION, END]
INFILL_SYSTEM_TURN: list[str | torch.Tensor] = [SYSTEM, INFILL_INSTRUCTION, END]
# Where the text or codes under test sit in a conversation.
TextCarrier = Literal["system", "reference", "assistant", "user"]
CodesCarrier = Literal["user", "reference"]


@pytest.fixture(scope="module")
def toy_tokenizer() -> Tokenizer:
    """Byte-level BPE whose merges join two newlines and "t[", so a newline seam,
    or an infill transcript ending in "t" before its length hint, tells
    per-segment encoding apart from encoding the joined string."""
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocabulary = {symbol: token_id for token_id, symbol in enumerate(alphabet)}
    # Ċ is the byte-level symbol of a newline.
    vocabulary["ĊĊ"] = len(vocabulary)
    vocabulary["t["] = len(vocabulary)
    tokenizer = Tokenizer(models.BPE(vocab=vocabulary, merges=[("Ċ", "Ċ"), ("t", "[")]))
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


def user_with(*parts: PromptPart) -> PromptMessage:
    return PromptMessage(role="user", parts=list(parts))


def assistant_with(*parts: PromptPart) -> PromptMessage:
    return PromptMessage(role="assistant", parts=list(parts))


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


DIRECTORS_SCENE = "场景：雨后的天台，两位老友重逢。\n请按备注调整这一条。"
DIRECTORS_NOTE = "语速放慢，尾音再收一点"
IMITATE_INSTRUCTION = "Say the line below the way this clip sounds: "
LAYOUT_CASES = [
    pytest.param(
        [said("user", "hello")],
        False,
        DEFAULT_SYSTEM_TURN + [USER, "hello", END, VOICE_TURN],
        id="no-system-message-uses-the-default-instruction",
    ),
    pytest.param(
        [
            system_with(reference("lone take", clip_codes(4, 1))),
            said("user", "say it"),
        ],
        False,
        [SYSTEM, REFERENCE_INSTRUCTION, reference_header(1, "<|speaker:0|> lone take")]
        + [clip_codes(4, 1), END, USER, "say it", END, VOICE_TURN],
        id="one-reference",
    ),
    pytest.param(
        [
            system_with(
                reference("opening take", clip_codes(4, 1)),
                reference("<|speaker:1|> answer take", clip_codes(5, 2)),
                reference("<|speaker:2|> closing take", clip_codes(6, 3)),
            ),
            said("user", "say it"),
        ],
        False,
        [SYSTEM, REFERENCE_INSTRUCTION]
        + [reference_header(1, "<|speaker:0|> opening take"), clip_codes(4, 1)]
        + [reference_header(2, "<|speaker:1|> answer take"), clip_codes(5, 2)]
        + [reference_header(3, "<|speaker:2|> closing take"), clip_codes(6, 3)]
        + [END, USER, "say it", END, VOICE_TURN],
        id="three-references-take-their-speakers-from-the-transcripts",
    ),
    pytest.param(
        [
            system_with(
                reference("A whole sample.", clip_codes(8, 10)),
                reference(
                    "<|speaker:1|> One voice. <|speaker:0|> Another voice.",
                    clip_codes(5, 40),
                ),
            ),
            said("user", "hello"),
        ],
        True,
        [SYSTEM, REFERENCE_INSTRUCTION]
        + [reference_header(1, "<|speaker:0|> A whole sample."), clip_codes(8, 10)]
        + [reference_header(2, "<|speaker:1|> One voice. <|speaker:0|> Another voice.")]
        + [clip_codes(5, 40), END, USER, "<|speaker:0|> hello", END, VOICE_TURN],
        id="tts-references-keep-an-inner-speaker-switch",
    ),
    pytest.param(
        [
            system_with(
                TextPart(text="午后访谈节目，主持人语气轻快。\n示例如下："),
                reference("示例转写", clip_codes(4, 1)),
            ),
            said("user", "再说一遍"),
        ],
        False,
        [SYSTEM, "午后访谈节目，主持人语气轻快。\n示例如下："]
        + [reference_header(1, "<|speaker:0|> 示例转写"), clip_codes(4, 1), END]
        + [USER, "再说一遍", END, VOICE_TURN],
        id="free-form-multi-line-system-text",
    ),
    pytest.param(
        [
            system_with(
                TextPart(text="speak this line like a sleepy captain"),
                reference("clip", clip_codes(3, 10)),
            ),
            said("user", "hello"),
        ],
        False,
        [SYSTEM, "speak this line like a sleepy captain"]
        + [reference_header(1, "<|speaker:0|> clip"), clip_codes(3, 10), END]
        + [USER, "hello", END, VOICE_TURN],
        id="custom-instruction-replaces-only-the-instruction-line",
    ),
    pytest.param(
        [
            said("user", "<|speaker:0|> the base line"),
            take(clip_codes(6, 100)),
            said("user", "<|speaker:0|> [warmer than before] the base line"),
            take(clip_codes(7, 200)),
            said("user", "<|speaker:0|> now whisper it"),
        ],
        True,
        DEFAULT_SYSTEM_TURN
        + [USER, "<|speaker:0|> the base line", END]
        + [VOICE_TURN, clip_codes(6, 100), END]
        + [USER, "<|speaker:0|> [warmer than before] the base line", END]
        + [VOICE_TURN, clip_codes(7, 200), END]
        + [USER, "<|speaker:0|> now whisper it", END, VOICE_TURN],
        id="tts-prior-turns-replay-before-the-current-text",
    ),
    pytest.param(
        [
            user_with(
                TextPart(text="<|speaker:0|> make this one brighter"),
                AudioCodesPart(audio_codes=clip_codes(9, 300)),
            ),
            said("user", "<|speaker:0|> next"),
        ],
        True,
        DEFAULT_SYSTEM_TURN
        + [USER, "<|speaker:0|> make this one brighter", clip_codes(9, 300), END]
        + [USER, "<|speaker:0|> next", END, VOICE_TURN],
        id="user-audio-follows-its-text-without-voice",
    ),
    pytest.param(
        [
            said("system", "scene"),
            said("assistant", "plain reply text"),
            said("user", "<|speaker:3|> my line"),
        ],
        False,
        [SYSTEM, "scene", END, ASSISTANT, "plain reply text", END]
        + [USER, "<|speaker:3|> my line", END, VOICE_TURN],
        id="assistant-text-has-no-voice",
    ),
    pytest.param(
        [
            said("system", DIRECTORS_SCENE),
            said("user", "这件事我真的不知道。"),
            take(clip_codes(6, 9)),
            said("user", DIRECTORS_NOTE),
        ],
        False,
        [SYSTEM, DIRECTORS_SCENE, END, USER, "这件事我真的不知道。", END]
        + [VOICE_TURN, clip_codes(6, 9), END, USER, DIRECTORS_NOTE, END, VOICE_TURN],
        id="chat-mode-replays-a-take-and-leaves-a-directors-note-untagged",
    ),
    pytest.param(
        [
            said("system", "场景：雨夜的车站"),
            said("user", "我等了你很久。"),
            take(clip_codes(4, 3)),
            said("user", "再冷淡一点"),
        ],
        False,
        [SYSTEM, "场景：雨夜的车站", END, USER, "我等了你很久。", END]
        + [VOICE_TURN, clip_codes(4, 3), END, USER, "再冷淡一点", END, VOICE_TURN],
        id="chat-mode-replays-two-user-assistant-pairs",
    ),
    pytest.param(
        [
            user_with(
                TextPart(text=IMITATE_INSTRUCTION),
                AudioCodesPart(audio_codes=clip_codes(6, 2)),
                TextPart(text="See you tomorrow."),
            )
        ],
        False,
        DEFAULT_SYSTEM_TURN
        + [USER, IMITATE_INSTRUCTION, clip_codes(6, 2), "See you tomorrow.", END]
        + [VOICE_TURN],
        id="imitate-user-turn-has-audio-between-its-texts",
    ),
    pytest.param(
        [
            user_with(
                TextPart(text="甲：大家早上好！"),
                AudioSeparatorPart(),
                TextPart(text="乙：早，今天开会吗？"),
            ),
            assistant_with(
                AudioCodesPart(audio_codes=clip_codes(4, 4)),
                AudioSeparatorPart(),
                AudioCodesPart(audio_codes=clip_codes(5, 5)),
            ),
            said("user", "第二句再来一遍"),
        ],
        False,
        DEFAULT_SYSTEM_TURN
        + [USER, f"甲：大家早上好！{SEPARATOR}乙：早，今天开会吗？", END]
        + [VOICE_TURN, clip_codes(4, 4), SEPARATOR, clip_codes(5, 5), END]
        + [USER, "第二句再来一遍", END, VOICE_TURN],
        id="dub-joins-user-lines-and-separates-assistant-takes",
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


def grouped_references(speaker_groups: list[list[str]]) -> list[ReferencePart]:
    """Grouped references whose codes number the clips in reading order from 1."""
    clip_numbers = itertools.count(1)
    return references_from_speaker_groups(
        [
            [
                reference(transcript, clip_codes(1, next(clip_numbers)))
                for transcript in group
            ]
            for group in speaker_groups
        ]
    )


@pytest.mark.parametrize(
    ("speaker_groups", "header_texts"),
    [
        pytest.param(
            [["voice one", "voice two"]],
            ["<|speaker:0|> voice one", "<|speaker:0|> voice two"],
            id="one-group-is-speaker-0",
        ),
        pytest.param(
            [["<|speaker:9|> tagged voice"], ["plain voice"]],
            ["<|speaker:9|> tagged voice", "<|speaker:1|> plain voice"],
            id="a-tagged-transcript-keeps-its-speaker",
        ),
        pytest.param(
            [[], ["hi"], [" <|speaker:5|> hi"], ["x", "y"]],
            ["<|speaker:0|> hi", "<|speaker:5|>  hi"]
            + ["<|speaker:2|> x", "<|speaker:2|> y"],
            id="empty-groups-take-no-speaker-id",
        ),
        pytest.param([["   "]], ["<|speaker:0|>    "], id="blank-transcript"),
    ],
)
def test_speaker_groups_tag_untagged_transcripts_with_the_group_number(
    toy_tokenizer: Tokenizer, speaker_groups: list[list[str]], header_texts: list[str]
) -> None:
    references = grouped_references(speaker_groups)
    messages = [system_with(*references), said("user", "hello")]

    rows = build(toy_tokenizer, messages, should_tag_user_text=True)

    segments: list[str | torch.Tensor] = [SYSTEM, REFERENCE_INSTRUCTION]
    for clip_number, header_text in enumerate(header_texts, start=1):
        segments.append(reference_header(clip_number, header_text))
        segments.append(clip_codes(1, clip_number))
    segments += [END, USER, "<|speaker:0|> hello", END, VOICE_TURN]
    assert torch.equal(rows, expected_rows(toy_tokenizer, segments))


@pytest.mark.parametrize(
    "speaker_groups",
    [
        pytest.param([["<|speaker:16|> voice"]], id="speaker-16"),
        pytest.param([["<|speaker:abc|> voice"]], id="malformed-tag"),
        pytest.param([["clip"]] * 17, id="an-untagged-seventeenth-group"),
    ],
)
def test_speaker_groups_reject_tags_the_tokenizer_lacks(
    toy_tokenizer: Tokenizer, speaker_groups: list[list[str]]
) -> None:
    messages = [system_with(*grouped_references(speaker_groups)), said("user", "hi")]

    with pytest.raises(ValueError, match="speaker tags must be written"):
        build(toy_tokenizer, messages, should_tag_user_text=True)


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
        pytest.param(
            [
                said("user", "line"),
                assistant_with(
                    AudioCodesPart(audio_codes=clip_codes(2, 1)),
                    TextPart(text="note\n"),
                    AudioSeparatorPart(),
                    AudioCodesPart(audio_codes=clip_codes(2, 2)),
                ),
                said("user", "again"),
            ],
            DEFAULT_SYSTEM_TURN
            + [USER, "line", END, VOICE_TURN, clip_codes(2, 1), "note\n", SEPARATOR]
            + [clip_codes(2, 2), END, USER, "again", END, VOICE_TURN],
            DEFAULT_SYSTEM_TURN
            + [USER, "line", END, VOICE_TURN, clip_codes(2, 1), "note\n" + SEPARATOR]
            + [clip_codes(2, 2), END, USER, "again", END, VOICE_TURN],
            id="assistant-separator",
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
    ("user_parts", "segments", "split_segments"),
    [
        pytest.param(
            [TextPart(text="a\n"), AudioSeparatorPart(), TextPart(text="\nb")],
            [f"a\n{SEPARATOR}\nb"],
            ["a\n", SEPARATOR, "\nb"],
            id="between-two-lines",
        ),
        pytest.param(
            [
                TextPart(text="a\n"),
                TextPart(text="\nb\n"),
                AudioSeparatorPart(),
                TextPart(text="\nc"),
            ],
            ["a\n", f"\nb\n{SEPARATOR}\nc"],
            ["a\n", "\nb\n", SEPARATOR, "\nc"],
            id="not-the-line-before-the-last",
        ),
        pytest.param(
            [
                TextPart(text="a\n"),
                AudioSeparatorPart(),
                TextPart(text="\nb"),
                TextPart(text="\nc"),
            ],
            [f"a\n{SEPARATOR}\nb", "\nc"],
            ["a\n", SEPARATOR, "\nb", "\nc"],
            id="not-the-line-after-the-next",
        ),
        pytest.param(
            [AudioSeparatorPart(), TextPart(text="\nb")],
            [f"{SEPARATOR}\nb"],
            [SEPARATOR, "\nb"],
            id="leading",
        ),
        pytest.param(
            [TextPart(text="a\n"), AudioSeparatorPart()],
            [f"a\n{SEPARATOR}"],
            ["a\n", SEPARATOR],
            id="trailing",
        ),
        pytest.param(
            [
                TextPart(text="a\n"),
                AudioSeparatorPart(),
                AudioCodesPart(audio_codes=clip_codes(2, 1)),
                TextPart(text="\nb"),
            ],
            [f"a\n{SEPARATOR}", clip_codes(2, 1), "\nb"],
            ["a\n", SEPARATOR, clip_codes(2, 1), "\nb"],
            id="audio-ends-the-line",
        ),
        pytest.param(
            [
                AudioCodesPart(audio_codes=clip_codes(2, 1)),
                AudioSeparatorPart(),
                TextPart(text="\nb"),
            ],
            [clip_codes(2, 1), f"{SEPARATOR}\nb"],
            [clip_codes(2, 1), SEPARATOR, "\nb"],
            id="audio-before-the-separator",
        ),
    ],
)
def test_user_separator_joins_the_text_around_it(
    toy_tokenizer: Tokenizer,
    user_parts: list[PromptPart],
    segments: list[str | torch.Tensor],
    split_segments: list[str | torch.Tensor],
) -> None:
    rows = build(toy_tokenizer, [user_with(*user_parts)], should_tag_user_text=False)

    turn_start = DEFAULT_SYSTEM_TURN + [USER]
    turn_end: list[str | torch.Tensor] = [END, VOICE_TURN]
    expected = expected_rows(toy_tokenizer, turn_start + segments + turn_end)
    split = expected_rows(toy_tokenizer, turn_start + split_segments + turn_end)
    assert torch.equal(rows, expected)
    assert not torch.equal(rows, split)


def test_tts_mode_tags_each_line_before_a_separator_joins_them(
    toy_tokenizer: Tokenizer,
) -> None:
    """The builder composes its two rules; no trained layout pins this one."""
    user_parts = [
        TextPart(text="first line\n"),
        AudioSeparatorPart(),
        TextPart(text="<|speaker:1|> second line"),
    ]

    rows = build(toy_tokenizer, [user_with(*user_parts)], should_tag_user_text=True)

    turn_start = DEFAULT_SYSTEM_TURN + [USER]
    turn_end: list[str | torch.Tensor] = [END, VOICE_TURN]
    joined_lines = f"<|speaker:0|> first line\n{SEPARATOR}<|speaker:1|> second line"
    split_lines = ["<|speaker:0|> first line\n", SEPARATOR, "<|speaker:1|> second line"]
    expected = expected_rows(toy_tokenizer, turn_start + [joined_lines] + turn_end)
    split = expected_rows(toy_tokenizer, turn_start + split_lines + turn_end)
    assert torch.equal(rows, expected)
    assert not torch.equal(rows, split)


@pytest.mark.parametrize(
    "text",
    [
        "Direction: [quietly] hold back a little\n"
        "Line: <soft>We should go home now.</soft>",
        "current part<|endofturn|>next part",
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


NO_FRAMES = torch.zeros((NUM_CODEBOOKS, 0), dtype=torch.long)


def build_infill(
    tokenizer: Tokenizer,
    transcript: str,
    *,
    prefix_audio_codes: torch.Tensor,
    suffix_audio_codes: torch.Tensor,
    target_frame_count: int | None,
) -> torch.Tensor:
    return build_infill_prompt_rows(
        transcript,
        tokenizer,
        prefix_audio_codes=prefix_audio_codes,
        suffix_audio_codes=suffix_audio_codes,
        target_frame_count=target_frame_count,
        num_codebooks=NUM_CODEBOOKS,
        codebook_size=CODEBOOK_SIZE,
    )


def test_infill_puts_the_suffix_before_the_prefix(toy_tokenizer: Tokenizer) -> None:
    rows = build_infill(
        toy_tokenizer,
        "<|speaker:0|> a quick infill test",
        prefix_audio_codes=clip_codes(5, 10),
        suffix_audio_codes=clip_codes(3, 50),
        target_frame_count=37,
    )

    segments = INFILL_SYSTEM_TURN + [USER, "<|speaker:0|> a quick infill test"]
    segments += ["[audio tokens: 37]", "<|fim_suffix|>", clip_codes(3, 50)]
    segments += ["<|fim_prefix|>", clip_codes(5, 10), END, VOICE_TURN]
    joined_segments = INFILL_SYSTEM_TURN + [USER]
    joined_segments += ["<|speaker:0|> a quick infill test[audio tokens: 37]"]
    joined_segments += ["<|fim_suffix|>", clip_codes(3, 50)]
    joined_segments += ["<|fim_prefix|>", clip_codes(5, 10), END, VOICE_TURN]
    assert torch.equal(rows, expected_rows(toy_tokenizer, segments))
    assert not torch.equal(rows, expected_rows(toy_tokenizer, joined_segments))


@pytest.mark.parametrize(
    ("target_frame_count", "hint_segments"),
    [(None, []), (0, []), (1, ["[audio tokens: 1]"])],
)
def test_infill_hints_only_a_positive_frame_count(
    toy_tokenizer: Tokenizer,
    target_frame_count: int | None,
    hint_segments: list[str | torch.Tensor],
) -> None:
    rows = build_infill(
        toy_tokenizer,
        "<|speaker:0|> a quick infill test",
        prefix_audio_codes=clip_codes(2, 1),
        suffix_audio_codes=NO_FRAMES,
        target_frame_count=target_frame_count,
    )

    segments = INFILL_SYSTEM_TURN + [USER, "<|speaker:0|> a quick infill test"]
    segments += hint_segments + ["<|fim_suffix|>", "<|fim_prefix|>", clip_codes(2, 1)]
    segments += [END, VOICE_TURN]
    assert torch.equal(rows, expected_rows(toy_tokenizer, segments))


@pytest.mark.parametrize(
    "target_frame_count", [True, False, 1.5, float("nan"), float("inf"), -1]
)
def test_infill_rejects_a_frame_count_that_is_not_a_non_negative_int(
    toy_tokenizer: Tokenizer, target_frame_count: float
) -> None:
    with pytest.raises(
        ValueError, match="target_frame_count must be a non-negative int or None"
    ):
        build_infill(
            toy_tokenizer,
            "line",
            prefix_audio_codes=clip_codes(2, 1),
            suffix_audio_codes=NO_FRAMES,
            target_frame_count=target_frame_count,
        )


@pytest.mark.parametrize(
    ("prefix_audio_codes", "suffix_audio_codes", "fill_segments"),
    [
        pytest.param(
            clip_codes(5, 10),
            NO_FRAMES,
            ["<|fim_suffix|>", "<|fim_prefix|>", clip_codes(5, 10)],
            id="no-suffix",
        ),
        pytest.param(
            NO_FRAMES,
            clip_codes(3, 50),
            ["<|fim_suffix|>", clip_codes(3, 50), "<|fim_prefix|>"],
            id="no-prefix",
        ),
        pytest.param(
            NO_FRAMES, NO_FRAMES, ["<|fim_suffix|>", "<|fim_prefix|>"], id="neither"
        ),
    ],
)
def test_infill_leaves_out_a_side_without_frames(
    toy_tokenizer: Tokenizer,
    prefix_audio_codes: torch.Tensor,
    suffix_audio_codes: torch.Tensor,
    fill_segments: list[str | torch.Tensor],
) -> None:
    rows = build_infill(
        toy_tokenizer,
        "<|speaker:0|> regenerate the end",
        prefix_audio_codes=prefix_audio_codes,
        suffix_audio_codes=suffix_audio_codes,
        target_frame_count=None,
    )

    segments = INFILL_SYSTEM_TURN + [USER, "<|speaker:0|> regenerate the end"]
    segments += fill_segments + [END, VOICE_TURN]
    assert torch.equal(rows, expected_rows(toy_tokenizer, segments))


@pytest.mark.parametrize(
    ("transcript", "spoken_transcript"),
    [
        ("a full take", "<|speaker:0|> a full take"),
        ("  one\r\ntwo\rthree\nfour  ", "<|speaker:0|> one two three four"),
        ("x\r\r\ny", "<|speaker:0|> x  y"),
        (" <|speaker:2|> x", "<|speaker:2|> x"),
        ("one <|speaker:3|> two", "<|speaker:0|> one <|speaker:3|> two"),
    ],
)
def test_infill_folds_line_breaks_and_tags_the_transcript(
    toy_tokenizer: Tokenizer, transcript: str, spoken_transcript: str
) -> None:
    rows = build_infill(
        toy_tokenizer,
        transcript,
        prefix_audio_codes=clip_codes(2, 1),
        suffix_audio_codes=NO_FRAMES,
        target_frame_count=None,
    )

    segments = INFILL_SYSTEM_TURN + [USER, spoken_transcript, "<|fim_suffix|>"]
    segments += ["<|fim_prefix|>", clip_codes(2, 1), END, VOICE_TURN]
    assert torch.equal(rows, expected_rows(toy_tokenizer, segments))


@pytest.mark.parametrize(
    ("transcript", "error"),
    [
        pytest.param(" \r\n\t", "transcript: text must not be empty", id="blank"),
        pytest.param(
            "<|speaker:16|> line",
            "transcript: speaker tags must be written",
            id="speaker-16",
        ),
        pytest.param(
            "line <|im_end|>",
            "transcript: <|im_end|> is written by the prompt builder",
            id="structural-token",
        ),
    ],
)
def test_infill_rejects_a_bad_transcript(
    toy_tokenizer: Tokenizer, transcript: str, error: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(error)):
        build_infill(
            toy_tokenizer,
            transcript,
            prefix_audio_codes=clip_codes(2, 1),
            suffix_audio_codes=NO_FRAMES,
            target_frame_count=None,
        )


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
            PromptMessage(role="system", parts=[AudioSeparatorPart()]),
            said("user", "hi"),
        ],
        "messages[0].parts[0]: the system message takes text and reference parts only",
        id="separator-in-system",
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
    for token in [
        "<|im_start|>",
        "<|im_end|>",
        "<|voice|>",
        "<|fim_prefix|>",
        "<|fim_suffix|>",
        "<|audio_separator|>",
        "<|endofturn|>",
    ]:
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


def test_real_tokenizer_rows_keep_the_infill_and_separator_seams(
    real_tokenizer: Tokenizer,
) -> None:
    dub_messages = [
        user_with(
            TextPart(text="第一句。\n"), AudioSeparatorPart(), TextPart(text="\n第二句")
        ),
        assistant_with(
            AudioCodesPart(audio_codes=clip_codes(2, 3)),
            AudioSeparatorPart(),
            AudioCodesPart(audio_codes=clip_codes(2, 4)),
        ),
        said("user", "再说一遍"),
    ]

    infill_rows = build_infill(
        real_tokenizer,
        "第一句。",
        prefix_audio_codes=clip_codes(2, 1),
        suffix_audio_codes=clip_codes(3, 2),
        target_frame_count=37,
    )
    dub_rows = build(real_tokenizer, dub_messages, should_tag_user_text=False)

    infill_segments = INFILL_SYSTEM_TURN + [USER, "<|speaker:0|> 第一句。"]
    infill_segments += ["[audio tokens: 37]", "<|fim_suffix|>", clip_codes(3, 2)]
    infill_segments += ["<|fim_prefix|>", clip_codes(2, 1), END, VOICE_TURN]
    dub_segments = DEFAULT_SYSTEM_TURN + [USER, f"第一句。\n{SEPARATOR}\n第二句", END]
    dub_segments += [VOICE_TURN, clip_codes(2, 3), SEPARATOR, clip_codes(2, 4), END]
    dub_segments += [USER, "再说一遍", END, VOICE_TURN]
    assert torch.equal(infill_rows, expected_rows(real_tokenizer, infill_segments))
    assert torch.equal(dub_rows, expected_rows(real_tokenizer, dub_segments))
    seams = [
        ("<|speaker:0|> 第一句。", "[audio tokens: 37]"),
        ("第一句。\n", SEPARATOR),
        (SEPARATOR, "\n第二句"),
    ]
    for left, right in seams:
        joined_token_ids = real_tokenizer.encode(left + right).ids
        left_token_ids = real_tokenizer.encode(left).ids
        right_token_ids = real_tokenizer.encode(right).ids
        assert joined_token_ids != left_token_ids + right_token_ids
