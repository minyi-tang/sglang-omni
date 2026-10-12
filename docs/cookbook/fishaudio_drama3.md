# Fish Audio Drama 3

**Fish Audio Drama 3** is Fish Audio's new-generation text-to-speech model, built for multi-character scenes and long-form content. It supports multi-speaker generation, voice cloning and natural-language control.

- **35B parameters, 3B active per token**: MoE text backbone with hybrid (full + linear) attention
- **Codebook-MTP audio decoder**: 10 codebooks decoded causally over the whole sequence
- **FP8 MoE experts** with the remaining weights in bf16
- **Speaker tags** `<|speaker:0|>` … `<|speaker:15|>` — one text can carry up to 16 characters
- **Long text** is chunked automatically and can be streamed per chunk

## Installation and Startup

Detailed instructions are in [Installation](../get_started/installation.md). From the repository root:

```bash
uv pip install --prerelease=allow -v -e ".[fishaudio]"
```

Download the checkpoint, then start the server. This command places preprocessing, the engine, and the vocoder on GPU 0. A stage moves to another GPU only when its `--<stage>.gpu` flag says so; see [Deployment](#deployment).

```bash
hf download fishaudio/drama-3

sgl-omni serve \
  --model-path fishaudio/drama-3 \
  --port 8000
```

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
  --output output_voice_cloning.wav
```

Streaming:

```bash
curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Get the trust fund to the bank early.",
    "stream": true,
    "response_format": "pcm"
  }' \
  --output output_streaming.pcm
```

## Serving Limits

### Reference Audio Length

Each reference given as audio may be at most 60 seconds long. The limit applies to every reference in the request separately, and a longer one fails the request with HTTP 400 before it is encoded:

```text
Drama 3 request: references.1: reference audio lasts 75.3 s, longer than the 60 s limit; shorten it or raise max_reference_audio_seconds
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

The GPU memory the preprocessing stage needs grows with the length of the reference it encodes, and the stage is capped at `total_reserve_bytes` (6 GiB by default, sized for the 60-second limit). When you raise the limit, raise the reserve with it, measured on your GPU; a reference that does not fit the reserve fails with an out-of-memory error in
the preprocessing stage. The budgets of all stages on the GPU must still fit the card, which the server checks at startup.

### Prompt Length

A prompt holds the tokens of the prompt template, the input text and the reference transcripts, plus one row for every codec frame of its reference audio, about 21.5 rows per second of audio (the codec turns 44.1 kHz audio into frames of 2048 samples). A prompt may have at most 8192 rows; a longer one fails the request with HTTP 400:

```text
Drama 3 request: a prompt of 8193 rows exceeds the maximum allowed length of 8192 rows
```

Change the cap with `--tts_engine.factory.max_prompt_rows`. A prompt and its `max_new_tokens` must also fit the engine's context length.

## Attention Backend

Drama 3 runs its attention on the `triton` backend only. The server refuses to start with any other attention backend, for example `--tts_engine.engine.attention_backend fa3`.

## Deployment

On one GPU the default placement works as is: every stage runs on GPU 0, with the vocoder in its own process. With two GPUs, put the vocoder on the second one, so its decoding no longer shares a GPU with the engine:

```bash
sgl-omni serve \
  --model-path <checkpoint> \
  --tts_engine.gpu 0 \
  --vocoder.gpu 1
```

When the engine and the vocoder share one GPU, `--mps on` lets their processes share it through CUDA MPS (see [Same-GPU Data Parallelism with CUDA MPS](../basic_usage/mps_dp.md)). It can raise throughput under high concurrency but may make a single stream slower; measure it with your own load before you keep it.

## High Concurrency

The defaults run at most 16 requests at a time on the engine and give it 0.70 of the GPU's memory. For more streams at once, raise the request cap; the measurements below also raise the engine's share:

```bash
sgl-omni serve \
  --model-path <checkpoint> \
  --tts_engine.gpu_memory_fraction 0.85 \
  --tts_engine.factory.max_running_requests 256
```

These values are sized for an H200: the engine's and the vocoder's shares and the preprocessing reserve must fit one GPU together, which the server checks at startup.

### Throughput and Time to First Audio

Measured on H200 GPUs with the settings above, the vocoder's default share and streamed requests, 1,088 requests per concurrency level, each cloning a voice from one reference given as `vq_codes` with its transcript, with `max_new_tokens` 500, `temperature` 1.0, `top_p` 0.9 and `top_k` 30. Each figure is the average over six server starts; the throughput counts every frame the engine generates, including the end-of-speech frame that ends a stream:

| Vocoder | Concurrent streams | Median time to first audio | 95th percentile | Throughput (generated frames/s) |
|---|---|---|---|---|
| On the engine's GPU | 128 | 685 ms | 1,412 ms | 2,177 |
| On the engine's GPU | 256 | 3,403 ms | 4,255 ms | 2,422 |
| On a second GPU | 128 | 401 ms | 995 ms | 3,692 |
| On a second GPU | 256 | 2,146 ms | 2,469 ms | 4,160 |

With the vocoder on the engine's GPU, at 256 streams the vocoder runs at full load and each new stream's first audio waits in its queue. When you need many concurrent streams and a short time to first audio together, put the vocoder on a second GPU, as under [Deployment](#deployment): at 256 streams that gave about 72% more throughput and a 37% shorter median time to first audio.

### Non-Streaming Requests

The vocoder decodes a non-streaming request's audio in one piece after the engine finishes it, batching requests that
finish together up to a frame limit it sets at startup from its memory share (`--vocoder.gpu_memory_fraction`, 0.10
by default) and logs:

```text
Drama 3 vocoder: a non-streaming decode takes up to <frames> padded frames of <bytes> bytes each; share <share> GiB, held <held> GiB
```

The limit does not count the memory the vocoder process holds outside PyTorch's allocator, so it cannot guarantee that the process stays within its share. With the engine's share raised to 0.85 on the same GPU, concurrent non-streaming requests can run out of GPU memory. To avoid that, stream those requests (`"stream": true`), keep the engine's default share of 0.70 on the GPU that runs the vocoder, or put the vocoder on a second GPU, as under [Deployment](#deployment).

## Defaults Parameters

| Default | What it does | Change it with |
|---|---|---|
| Decode CUDA graph | Replays decode steps as CUDA graphs. | `--tts_engine.engine.cuda_graph_backend_decode disabled`; `--tts_engine.engine.disable_cuda_graph true` turns off every engine CUDA graph |
| Breakable prefill CUDA graph, up to 2048 tokens | Replays prefill batches of up to 2048 tokens, backbone and codebook tower together, as one breakable CUDA graph; larger batches run eagerly. Shortens the time to the first audio at the cost of startup time and GPU memory; the KV cache keeps its size. | `--tts_engine.engine.cuda_graph_backend_prefill disabled`; the token cap with `--tts_engine.engine.cuda_graph_max_bs_prefill` |
| First-frame CUDA graphs, up to 16 requests | Replays the step that samples each request's first frame as a CUDA graph for prefills of up to 16 requests (at most the decode CUDA graph's largest batch); larger prefills sample eagerly. Needs the decode CUDA graph; shortens each prefill at the cost of startup time and GPU memory. | `--tts_engine.factory.first_frame_cuda_graphs false`; the request cap with `--tts_engine.factory.first_frame_cuda_graph_max_requests` |
| `prefill_decode_interval` 4 | After a prefill, new prefills wait for the next 4 scheduling rounds, so running streams keep decoding. | `--tts_engine.engine.prefill_decode_interval 0` schedules prefills without the wait |
| Triton sampling | Samples each frame's main token and codes with Triton kernels instead of the PyTorch sampler, under the same top-k, top-p and repetition-aware rules. Using the same seed does not guarantee the same tokens as the PyTorch sampler. Rounding can keep or drop a candidate whose cumulative probability lies at top_p. | `--tts_engine.factory.triton_sampling false` |
| Fused codebook tower layers, up to 16 rows | Runs the codebook tower layers of decode and first-frame steps of up to 16 rows as fused Triton kernels, eagerly or inside the CUDA graphs; larger steps and the tower's prefill keep the module path. The kernels add in a different order, so the tower's outputs can differ in the last bits and the sampled frames can differ. | `--tts_engine.factory.tower_fused_layers false` |
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

Triton sampling and the fused codebook tower layers have been validated on Hopper GPUs only, so they run only on compute capability 9.0; on any other GPU no setting turns them on. Both also fall back without Triton or a CUDA device. Triton sampling falls back to the PyTorch sampler when its kernels do not cover the codebook size or the main-token candidate count, and the fused layers fall back to the module path with deterministic inference or when their kernels do not cover the tower weights or the KV cache. The startup log names the path each one takes and, for a fallback, the reason, for example:

```text
Drama 3 codebook sampling: PyTorch, untested GPU sm_120
Drama 3 tower fused layers: off, untested GPU sm_120
```
