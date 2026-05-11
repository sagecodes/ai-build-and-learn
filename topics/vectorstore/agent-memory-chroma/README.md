# Agent Memory: Chroma + Gemma 4 + HF Hub

Week 1.5 of the vector-stores series. Same Chroma + Gemma + Flyte 2 stack as
`rag-chroma-flyte`, but the store is **read+write**: it's the agent's
long-running memory of the user, not a static doc corpus.

```
                    ┌─────────────────────────────┐
                    │    HuggingFace Hub          │
                    │  sagecodes/agent-mem        │
                    │  ── memory.tar.gz (commits) │
                    └────────────▲────────────────┘
                                 │
                    on_startup  │  on_shutdown / "💾 Save to HF"
                    (snapshot   │  (tar + upload_file)
                     download)  │
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│  agent-memory-chat  (Flyte AppEnvironment, Knative-served)        │
│                                                                   │
│   per turn:                                                       │
│   1. user msg → BGE encode → Chroma top-k                         │
│   2. Gemma 4 stream answer with memories injected as context      │
│   3. 2nd Gemma call: extract atomic facts → JSON                  │
│   4. embed + write back to Chroma                                 │
└──────────────────────────────────────────────────────────────────┘
```

## How memory persists

Chroma's persist dir lives at `/tmp/agent_memory_chroma` inside the pod.
Knative scales the pod down after 30 min idle, which wipes pod-local disk.
To keep memory:

- **`@env.on_startup`**: pulls `memory.tar.gz` from `sagecodes/agent-mem`
  via `hf_hub_download`, untars into the persist dir. Missing file: start fresh.
- **`@env.on_shutdown`**: Knative SIGTERMs the pod (30s grace), the hook
  tars the persist dir and `upload_file`s it back as a new commit.
- **"💾 Save to HF" button**: same upload, on demand. Useful for explicit
  checkpoints during a livestream and as a safety net if a crash skips the
  shutdown hook.

Each save is a new commit, so HF gives you free memory versioning. You can
scrub back through "what did the agent remember on Tuesday."

## Files

| File | What it does |
|------|--------------|
| `chat_app.py` | Single Gradio `AppEnvironment` with `@on_startup` / `@server` / `@on_shutdown`. All HF + Chroma logic in one file. |
| `requirements.txt` | Local deps. The pod image is built from these via `with_pip_packages`. |

## Prereqs

- Gemma 4 vLLM is already deployed (sibling project: `topics/gemma4/gemma4-dgx-devbox/vllm_server.py`).
- HF token secret scoped to the same project/domain the app deploys into:

   ```bash
   flyte create secret HF_TOKEN --project flytesnacks --domain development
   ```

   The token must have **write** scope (a default read-only token returns 401 when the app tries to push the snapshot). The `-p` / `-d` flags matter: a secret created without them lives at a different scope and the app pod won't see it.
- HF model repo exists: `sagecodes/agent-mem` (created via the HF web UI). Empty is fine. First-run will create the tarball on first save.

## Deploy

```bash
cd topics/vectorstore/agent-memory-chroma
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt

python chat_app.py
# → Agent memory chat deployed: http://agent-memory-chat-flytesnacks-development.localhost:30081/
```

## Demo flow for the stream

1. Open the URL. Status panel shows `Memories in store: 0`.
2. **Turn 1:** "Hey, I'm Sage. I'm building a Flyte demo for tonight's stream and I prefer terse responses."
   - Watch the "Written this turn" panel light up with 2-3 facts.
   - Status counter ticks up.
3. **Turn 2:** "What do you know about me?"
   - "Retrieved this turn" panel shows the facts from turn 1.
   - Gemma's reply draws on them.
4. Click **"💾 Save to HF"**. Status flips to `✅ Saved to <commit URL>`.
5. (Optional drama) **Hit Ctrl-C on the deploy script, redeploy.** Pod cold-starts, on_startup pulls the tarball back, status reads `Memories in store: N`. Ask "what do you know about me?". Same answers.

## Toggles in the UI

- **Use memory**: disable to compare retrieval-on vs. retrieval-off.
- **Enable thinking**: same toggle as the other Gemma chats. Off is faster.
- **Top-k memories**: how many to inject per turn. 5 is the default.

## Memory extraction prompt (in `chat_app.py`)

After every assistant reply, a second small Gemma call (thinking off, temp 0)
runs with this system prompt:

> You extract durable facts about the user from a single exchange. Output ONLY
> a JSON array of strings. Each string is one atomic fact, preference, or
> decision the user explicitly stated or strongly implied. Skip questions,
> speculation, and trivial pleasantries. If nothing is worth remembering,
> output [].

Then a regex grabs the first `[...]` block and `json.loads` it. If parsing
fails, no memories get written; the chat still works.

## Troubleshooting

**"💾 Save to HF" returns 401 / `RepositoryNotFoundError`**: almost always one of:

1. **Token isn't scoped to the same project/domain as the app.** A bare `flyte create secret HF_TOKEN` creates the secret at a different scope; the pod can't see it. Recreate with the project + domain flags:

   ```bash
   flyte delete secret HF_TOKEN -p flytesnacks -d development   # if one exists at this scope
   flyte create secret HF_TOKEN -p flytesnacks -d development
   ```

   Then redeploy the app so the running pod picks up the new secret.

2. **Token is read-only.** HF returns 401 with a "Repository Not Found" message rather than 403, so it looks like a missing repo when it's actually missing permission. Recreate the token at https://huggingface.co/settings/tokens with **Write** role (or fine-grained with write access to `sagecodes/agent-mem` specifically), then re-set the Flyte secret as above.

3. **`repo_type` mismatch.** The error URL contains `/api/models/...` if `repo_type="model"`; if your repo is actually a Space or Dataset, flip `HF_MEMORY_REPO_TYPE` at the top of `chat_app.py`.

**Memories panel stays empty even when the user said something obvious.** Check the chat-pod logs for `[memory] extraction call failed` or JSON parse
errors. The prompt is finicky; adjust `EXTRACTION_SYSTEM` / `EXTRACTION_EXAMPLES`
in `chat_app.py`.

**Pod scales down but no commit lands.** Knative grace period (default 30s)
might be too short for slow uploads. Use the manual save button as a backstop.

**Cold-start says `No prior tarball … starting fresh`.** First deploy, or
the repo is empty. Save once and the next cold start will restore it.

## Next ideas

- **Memory-decay pruner**: Flyte 2 cron task that nightly removes memories
  the agent hasn't retrieved in N days.
- **Multi-user**: add `entity_id` metadata to each memory and filter on
  query / write.
- **Show provenance**: store the originating turn id with each memory and
  link back to the conversation that produced it.
- **Always-on Chroma server**: separate Flyte AppEnvironment with
  `replicas=(1,1)` so memory survives across chat-app restarts without the
  HF round-trip.
