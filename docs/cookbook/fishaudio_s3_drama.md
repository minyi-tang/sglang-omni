# Fish Audio S3-drama

Fish Audio S3-drama is a text-to-speech model served through `/v1/audio/speech`. The server runs it as three stages:
preprocessing builds the prompt and encodes reference audio into codec tokens on the GPU, the TTS engine generates
audio frames, and the vocoder decodes the frames into audio in its own process.

## Launch the Server

```bash
sgl-omni serve \
  --model-path <checkpoint> \
  --port 8000
```

`<checkpoint>` is a local directory holding the model's `config.json`, `tokenizer.json` and weights.
`examples/configs/s3_drama_tts.yaml` holds the same setup as a config file.

## Synthesize Speech

The server supports plain text-to-speech, optionally cloning a voice from reference audio, returned as one response or
as a stream.

Plain TTS:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, how are you?"
  }' \
  --output output.wav
```

Voice cloning, with a reference clip and its transcript:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Get the trust fund to the bank early.",
    "references": [{
      "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }]
  }' \
  --output output.wav
```

A reference may give its audio as `audio_path` (a path or URL) or inline as base64 `data` (with its `media_type`, WAV by
default), or give its codec tokens as `vq_codes` instead of audio; `ref_audio` and `ref_text` are a shorthand for one
reference. Several references of the same speaker may be given.

Streaming:

```bash
curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Get the trust fund to the bank early.",
    "stream": true,
    "response_format": "pcm"
  }' \
  --output output.pcm
```

Sampling fields: `temperature` (default 1.0), `top_p` (default 0.9) and `top_k` (default 30). `max_new_tokens` counts
audio frames; a request that sets none generates until the end of speech, up to the engine's cap
(`--tts_engine.factory.max_new_tokens`, 2048 frames by default) and the remaining context. A value the request sets
above the cap or beyond the context left after the prompt fails the request with HTTP 400 (see below).

### Requests the Server Refuses

These requests fail with HTTP 400:

- `instructions`, `language`, `seed` or `stage_params` with any value other than `null`; the same holds for the codec
  fields `x_vector_only_mode`, `stream_codec_output`, `suppress_bootstrap_silence`, `token_count`, `duration_tokens`
  and `initial_codec_chunk_frames`.
- A `task_type` other than `Base`.
- A `voice` other than `default` or the name of an [uploaded voice](../basic_usage/tts.md#uploaded-voices).
- More than one speaker: a speaker tag other than speaker 0 in `input` or in a reference transcript.
- Infill: the server has no infill request, and an `infill` field is refused as an unknown field.
- A `repetition_penalty` other than 1.0.
- A value out of range: `max_new_tokens` below 1, above the engine's cap or beyond the room the prompt leaves in the
  context; a `temperature` that is negative or not finite; a `top_p` outside (0, 1]; a `top_k` outside [1, 2147483647].

## Request Limits

### Reference Audio Length

Each reference given as audio may be at most 60 seconds long. The limit applies to every reference in the request
separately, and a longer one fails the request with HTTP 400 before it is encoded:

```text
S3-drama request: references.1: reference audio lasts 75.3 s, longer than the 60 s limit; shorten it or raise max_reference_audio_seconds
```

`references.N` names the reference by its position in the request's `references` list, counted from 0.

Shorten that reference, or raise the limit when you start the server, for example to 90 seconds:

```bash
sgl-omni serve \
  --model-path <checkpoint> \
  --preprocessing.factory.max_reference_audio_seconds 90 \
  --preprocessing.total_reserve_bytes 9GiB
```

or in a config file:

```yaml
stages:
  preprocessing:
    factory:
      max_reference_audio_seconds: 90
    total_reserve_bytes: 9GiB
```

The GPU memory the preprocessing stage needs grows with the length of the reference it encodes, and the stage is
capped at `total_reserve_bytes` (6 GiB by default, sized for the 60-second limit). When you raise the limit, raise the
reserve with it, measured on your GPU; a reference that does not fit the reserve fails with an out-of-memory error in
the preprocessing stage. The budgets of all stages on the GPU must still fit the card, which the server checks at
startup.

### Prompt Length

A prompt holds the tokens of the prompt template, the input text and the reference transcripts, plus one row for
every codec frame of its reference audio, about 21.5 rows per second of audio (the codec turns 44.1 kHz audio into
frames of 2048 samples). A prompt may have at most 8192 rows;
a longer one fails the request with HTTP 400:

```text
S3-drama request: a prompt of 8193 rows exceeds the maximum allowed length of 8192 rows
```

The cap is the engine's `max_prompt_rows`, which preprocessing applies too; change it on the engine with
`--tts_engine.factory.max_prompt_rows`. A prompt and its `max_new_tokens` must also fit the engine's context length.

## Attention Backend

S3-drama runs its attention on the `triton` backend only. The server refuses to start with any other attention backend,
for example `--tts_engine.engine.attention_backend fa3`.

## Deployment

On one GPU the default placement works as is: every stage runs on GPU 0, with the vocoder in its own process.

With two GPUs, put the vocoder on the second one, so its decoding no longer shares a GPU with the engine:

```bash
sgl-omni serve \
  --model-path <checkpoint> \
  --tts_engine.gpu 0 \
  --vocoder.gpu 1
```

The server must see both GPUs. The vocoder already runs in its own process, so nothing else needs to change.

When the engine and the vocoder share one GPU, `--mps on` lets their processes share it through CUDA MPS (see
[Same-GPU Data Parallelism with CUDA MPS](../basic_usage/mps_dp.md)). It can raise throughput under high concurrency
but may make a single stream slower; measure it with your own load before you keep it.

## Defaults and How to Change Them

| Default | What it does | Change it with |
|---|---|---|
| Decode CUDA graph | Replays decode steps as CUDA graphs. | `--tts_engine.engine.cuda_graph_backend_decode disabled`; `--tts_engine.engine.disable_cuda_graph true` turns off every engine CUDA graph |
| Breakable prefill CUDA graph, up to 2048 tokens | Replays the backbone and the codebook tower's prefill as one breakable CUDA graph, captured at startup for prefill batches of up to 2048 tokens; a larger prefill batch runs eagerly. It shortens the time to the first audio, adds to startup time and takes GPU memory for the graphs; the KV cache keeps its size. | `--tts_engine.engine.cuda_graph_backend_prefill disabled`; the token cap with `--tts_engine.engine.cuda_graph_max_bs_prefill` |
| First-frame CUDA graphs, up to 16 requests | Replays the step that samples each request's first frame as a CUDA graph, captured at startup for prefills of 1 to 16 requests (at most the decode CUDA graph's largest batch); a prefill of more requests samples its first frames eagerly. It needs the decode CUDA graph, shortens each prefill, and adds to startup time and GPU memory. | `--tts_engine.factory.first_frame_cuda_graphs false`; the request cap with `--tts_engine.factory.first_frame_cuda_graph_max_requests` |
| `prefill_decode_interval` 4 | After a prefill, new prefills wait for the next 4 scheduling rounds, so running streams keep decoding. | `--tts_engine.engine.prefill_decode_interval 0` schedules prefills without the wait |
| Triton sampling | Samples each frame's main token and its codes with Triton kernels, one per codebook head and one for the main token's two draws, instead of the PyTorch sampler. The top-k, top-p and repetition-aware rules are the same, but floating-point rounding can keep a candidate in one sampler and drop it in the other when its cumulative probability lies at top_p; the random numbers come from a counter that each sampling step advances, so the same seed draws different tokens than the PyTorch sampler. Without Triton or a CUDA device, or with a codebook size or main-token candidate count the kernels do not cover, the PyTorch sampler runs. | `--tts_engine.factory.triton_sampling false` |
| Fused codebook tower layers, up to 16 rows | Runs each codebook tower layer of a decode step, and of a prefill's first-frame step, as four Triton kernels when the step has at most 16 rows, eagerly or inside the step's CUDA graph: the code-embedding mix and its projection; the attention input norm and projections with the rotary embedding and the KV cache write; the attention output gate and projection; the MLP's norm, gate-up projection and activation. The attention, the down projection and the final add-norm run as before. Steps of more rows and the tower's prefill keep the module path. The kernels add in a different order than the module path, so the tower's outputs can differ in the last bits and the sampled frames can differ. Without Triton or a CUDA device, with deterministic inference, or with tower weights or a KV cache the kernels do not cover, the module path runs. | `--tts_engine.factory.tower_fused_layers false` |
| Vocoder window CUDA graphs | Replays the decode of each streaming window length at batch 1 as a CUDA graph; batches and longer windows decode eagerly. | `--vocoder.factory.window_cuda_graphs false` |
| Vocoder fp16 decoding | Decodes frames into audio in fp16. | `--vocoder.factory.decode_dtype float32` |

The same settings in a config file:

```yaml
stages:
  tts_engine:
    engine:
      cuda_graph_backend_prefill: disabled
      prefill_decode_interval: 0
  vocoder:
    factory:
      window_cuda_graphs: false
      decode_dtype: float32
```
