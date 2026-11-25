# 🦄 UniCoRn: A Benchmark for Unified Compression and Reranking

Context compressors are usually evaluated with reranking held fixed: a frozen
reranker selects the documents, and the compressor is scored only on how it
prunes them. UniCoRn evaluates both jobs together, since a compressor that
prunes well but ranks poorly is not useful in a retrieval stack.

The same compressor is held fixed across three tiers; only the retrieval regime
changes.

| Tier | Retrieval | What each query provides | What it measures |
| --- | --- | --- | --- |
| Unit | none | one document | pruning quality, isolated from ranking |
| Integration | fixed | 50 frozen candidates | reranking and pruning together |
| System | live | candidates fetched per agent call | behaviour inside an agent loop |

## Install

```bash
git clone https://github.com/byunsj/unicorn-benchmark && cd unicorn-benchmark
uv venv --python 3.12 && uv pip install -e ".[gpu]"
```

The datasets live on the Hugging Face Hub, so authenticate once with
`hf auth login`. Dropping the `[gpu]` extra installs the data layer only, which
is enough to browse the tiers but not to run them.

FlashAttention-2 is optional and left out of the extras because it compiles
against the installed torch. Every method loads without it, falling back to the
attention path that ships with transformers, though a greedy decode can then
differ from the reported numbers on a borderline query. Add it for the exact
setup the results were produced with:

```bash
uv pip install flash-attn --no-build-isolation
```

Evaluation needs one GPU: 24 GB covers compression for every method except
Comp-S and OSCAR, and adding `--judge` wants 80 GB, mostly for the judge.
Activate the virtualenv rather than calling its `python` directly, since vLLM
builds an attention kernel in a subprocess and needs `ninja` on `PATH`.

## Run an evaluation

```bash
unicorn --method comp-h --tier integration --dataset hotpotqa
```

This reranks the 50 candidates, prunes the top 5, and writes one JSONL record
per query plus a summary. It reports compression metrics only: how much was
removed, and how much of the gold evidence survived.

Omitting `--dataset` runs every dataset in the tier, which is the usual way to
reproduce a table:

```bash
unicorn --method comp-h --tier integration     # all 17 datasets
unicorn --method comp-h --tier unit            # all 10 probes
```

Answer accuracy is off by default, since it needs two more models. `--judge`
turns it on, generating an answer from the compressed context and scoring it:

```bash
unicorn --method comp-h --tier integration --judge
```

Answers come from Qwen3.5-9B and are scored by Qwen3.5-35B-A3B, which is asked a
single Yes/No question and read from that token's logits rather than from its
text. This keeps the score continuous and independent of how the judge phrases
its verdict, and matches Bergen's `LLMeval` so numbers stay comparable with
published work. Both models are swappable via `--generator-model` and
`--judge-model`.

The three stages load one model at a time and free the GPU in between, so a
threshold sweep pays each model-load once rather than once per threshold.

## Setting the threshold

Every compressor scores each sentence and keeps those above a threshold, so the
threshold *is* the compression/quality tradeoff. Each method ships with a
default that `--threshold` overrides:

```bash
unicorn --method comp-h --tier integration --dataset hotpotqa --threshold 0.5
```

Passing several values sweeps them in one run, loading the model once:

```bash
unicorn --method comp-h --tier integration --dataset conditionalqa --threshold 0.1 0.3 0.5 0.7
```

```
conditionalqa    t=0.1   gold@5=100.0  evidence= 77.2  compression= 95.2
conditionalqa    t=0.3   gold@5=100.0  evidence= 59.4  compression= 97.8
conditionalqa    t=0.5   gold@5=100.0  evidence= 32.5  compression= 98.9
conditionalqa    t=0.7   gold@5=100.0  evidence= 19.4  compression= 99.6
```

Compression is the fraction of tokens removed and evidence recall the fraction
of gold spans that survived, so a good compressor is one whose evidence recall
decays slowly as compression climbs. Adding `--judge` appends an accuracy column
to these rows, which traces the whole quality curve for roughly the cost of one
judged run. Thresholds are not comparable across methods, since each model's
scores are calibrated differently.

That non-comparability is why the paper's headline tables compare methods at
*matched compression* rather than at a matched threshold: each method is shown
at whichever threshold lands nearest the target rate. Reproducing them means
sweeping and then selecting the point closest to that rate. 

`--top-k` (documents kept) and `--rerank-k` (candidates considered, default 50)
are the other two knobs; `--no-prune` reranks without compressing, isolating the
ranking half of the task. `--top-k` defaults to 5 on the integration tier and 1
on the unit tier, where each query has a single document and nothing to rank.

## System tier

The first two tiers hand the compressor a fixed pool. The system tier does not:
an agent decides what to search for, the retriever answers live, and whatever
the compressor returns shapes the agent's next query. Pruning a crucial sentence
therefore does not only cost a point, it sends the agent down a different path.

The tier includes BrowseComp-Plus (830 multi-hop queries over a
100k-document corpus). It wants **two GPUs**: one for the agent model, one for
the retriever and compressor, which is how the published numbers were produced.

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve openai/gpt-oss-20b --port 8000 --gpu-memory-utilization 0.9
CUDA_VISIBLE_DEVICES=1 unicorn-system --method comp-h --limit 20 --no-grade
```

The published runs used two B200s, and the compressor's peak allocation is set
by the longest documents a search returns. On an 80 GB card that peak does not
fit on BrowseComp-Plus, so lower `--batch-size` until it does. Batching only
changes how rows are grouped into a forward pass, not what is computed, so
results hold; leaving it at 0 keeps each method's own default, which is what the
published numbers used.

Watch the `failed=` counter either way. A search that raises is reported to the
agent and the run continues, so a run whose searches all failed summarises as
zero compression rather than as an error. A few failures per run are normal:
the agent sometimes calls the tool with arguments it invented. Many failures
mean the compressor is running out of memory.

Grading is a separate step because the grader does not fit beside the agent
server. Stop the server first, then:

```bash
unicorn-system --method comp-h --grade-only results/system/comp-h_system_t0.3.jsonl
```

Alongside accuracy this reports **recall** (was the agent ever shown
the documents needed to justify its answer), **search calls** (how much work the
compression forced it to do), and **calibration error** (did its stated
confidence match reality). High recall with low accuracy means the agent was
handed the right document but not the right sentence.

Recall here is document-level, and is not the same metric as the evidence recall
reported for the other two tiers: BrowseComp-Plus labels whole documents as
evidence rather than annotating spans inside them. It is counted over the
documents the tool actually returned after compression, so a compressor that
drops a relevant document is penalised for it.

Retrieval uses `Qwen3-Embedding-8B` against the published index. Since the index
vectors are already normalised, an exhaustive inner-product scan on the GPU is
exact and fast at this corpus size, so no FAISS or Tevatron dependency is
needed.

### OfficeQA

The other system-tier dataset asks 246 financial questions whose answers are
figures buried in Treasury Bulletin tables. Its headline metric is therefore
agreement with the reference number, reported at 5%, 1% and 0.1% tolerance and
at exact match, using OfficeQA's own scorer:

```bash
unicorn-system --method comp-h --dataset officeqa --limit 20 --no-grade
```

The paper embedded the agent's sub-queries with a proprietary model. 
The index published here is a rebuild with `Qwen3-Embedding-8B`, the same open model the tier already uses.
Questions, answers and relevance judgements are unchanged, but the retrieval
space is not, so OfficeQA numbers produced here are a reproduction variant and
do not line up with the paper's table. Comparisons between compressors on this
index are still sound, since every method searches the same corpus.

The questions and answers stay at [`databricks/officeqa`](https://huggingface.co/datasets/databricks/officeqa),
which is access-gated by its authors; accept the gate there once and the
benchmark reads them from the source. The corpus, judgements and index are at
[`unicorn-rerank/unicorn-officeqa`](https://huggingface.co/datasets/unicorn-rerank/unicorn-officeqa),
whose card reports what the open embedder retrieves.

## Compressors

| Method | Default threshold | Notes |
| --- | --- | --- |
| `comp-h` | 0.3 | ours, extractive; reranks and prunes in one pass |
| `comp-s` | — | ours, soft compression into memory embeddings |
| `provence` | 0.1 | DeBERTa-v3 |
| `xprovence` | 0.3 | BGE-M3-v2 |
| `open-provence` | 0.1 | Large, 310M |
| `zerank-DSLR` | 0.4 | sentence-level selection, calibrated |
| `zilliz`, `zilliz+bge` | 0.3 | semantic highlight |
| `llmlingua-2`, `llmlingua-2+bge` | 0.5 | XLM-RoBERTa-large |
| `longllmlingua` | 0.55 | LLaMA-2-7B |
| `oscar` | — | Qwen2-7B, soft compression |
| `bge-reranker` | — | rerank only, no compression |
| `gold-only` | — | oracle; see below |

### Soft compression

`comp-s` and `oscar` do not prune text. They compress each selected document into
memory embeddings and answer from those with their own decoder, so `--threshold`
does not apply, the shared generator is not used, and `--generator-model` is
ignored. Both read a document as 128-token windows and keep its leading 16, which
is what bounds their memory; a document longer than about 1,800 tokens reaches the
decoder only in part.

Two consequences for the metrics. They report no evidence recall, since there is
no compressed string in which to locate the gold spans. And their compression
ratio is the rate at which they turn document tokens into memory embeddings —
one per 16 tokens of every selected document — rather than a measurement of
surviving text. `memory_vectors` in each record is the complement: how much the
decoder was actually handed, after the window cap.

They are also the two methods the system tier cannot run, since an agent has to
read its search results to decide what to search for next.

### The oracle

`gold-only` is the ceiling the other rows are measured against. It skips both
retrieval and compression: every gold document annotated for the query goes to
the generator in full, so `--threshold` and `--top-k` have no effect.

```bash
unicorn --method gold-only --tier integration --judge
```

## Metrics

Each run records, per query:

- **evidence recall** — fraction of gold evidence spans surviving compression,
  weighted by annotator agreement where the dataset provides it. Unit and
  integration tiers only; the system tier reports document-level retrieval
  recall instead, since it has no span annotations.
- **compression ratio** — fraction of tokens removed, in both the compressor's
  and the generator's tokenizer; for soft compression, the rate at which
  documents become memory embeddings
- **ranking** — whether gold reached the top 5, and at what rank
- **answer accuracy** — the judge's score, when `--judge` is passed

A rerank-only baseline reports a slightly negative compression ratio, because
the context carries a per-document header that the raw documents do not.

