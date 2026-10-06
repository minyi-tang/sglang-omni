# Fish Audio S3-drama

Fish Audio S3-drama is a text-to-speech model served through `/v1/audio/speech`. Reference audio sent with a request
is encoded into codec tokens on the GPU by the preprocessing stage.

## Reference Audio Length

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

## Prefill CUDA Graph

The engine replays the backbone's prefill as a breakable CUDA graph, captured at startup for prefill batches of up to
2048 tokens; a larger prefill batch runs eagerly. This shortens the time to the first audio. On an H200 it adds about
9 seconds to startup and leaves about 0.75 GB less free GPU memory after graph capture; the KV cache keeps its size.

To run the prefill eagerly instead:

```bash
sgl-omni serve \
  --model-path <checkpoint> \
  --tts_engine.engine.cuda_graph_backend_prefill disabled
```

or in a config file:

```yaml
stages:
  tts_engine:
    engine:
      cuda_graph_backend_prefill: disabled
```
