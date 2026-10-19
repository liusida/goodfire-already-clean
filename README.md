# Already Clean reproduction

This project starts from the dense `Qwen/Qwen3-32B` model used in Part 3
(“Exploring Language Models”) of *Neural Networks Are Already Clean*. The
smoke test pins the Hugging Face model revision for repeatability.

## Project layout

```text
.
├── model_daemon.py              # resident model process and script runner
├── jit_neuron_lens.py           # reusable just-in-time neuron lens
├── notebooks/
│   └── jit_neuron_lens_demo.ipynb
├── scripts/
│   ├── qwen32_layer7/           # Part 3 medical-prompt reproduction
│   ├── qwen35_weekdays/         # small-model weekday experiments
│   └── mlp_measurement/         # corpus-level MLP distributions
└── outputs/
    ├── layer7/                  # Qwen3-32B Layer 7 artifacts
    ├── weekdays/                # Qwen3.5-0.8B weekday artifacts
    ├── openwebtext/             # OpenWebText profile reports
    ├── jit_neuron_lens/         # live calibration/index/profile cache
    └── smoke/                   # small development-test caches
```

Run commands below from the project root. Generated data stays under
`outputs/`; reusable code and experiment entry points stay outside it.

Create/synchronize the environment and run the smoke test with:

```bash
uv sync
uv run python scripts/qwen32_layer7/hello_qwen.py
```

The first run downloads roughly 65.5 GB of BF16 model weights into the
Hugging Face cache. Model tensors are loaded through `gb10-load-llm`, which
uses the GB10 CPU-page-touch workaround before moving the model to CUDA.

## Keep the model resident

The project's local `model_daemon.py` can keep the loaded model in CUDA memory
while experiment scripts change between runs. It was initially copied from the
standalone ModelDaemon at `/home/liusida/code/ModelDaemon` so it can now be
customized for neuron-capture experiments here.

Start the daemon in one terminal:

```bash
uv run --no-sync python model_daemon.py serve
```

Run the smoke test through it from another terminal:

```bash
uv run --no-sync python model_daemon.py run scripts/qwen32_layer7/hello_qwen.py
```

The first request loads `Qwen3-32B`. Later requests reuse that exact model
object until the daemon is stopped with Ctrl+C. In a local test, the cold
daemon request took 151.54 seconds and the warm request took 4.60 seconds.

## Capture the medical case-study neuron

`scripts/qwen32_layer7/capture_layer7.py` uses the blog's exact 79-token raw prompt and captures the
input to Layer 7's MLP `down_proj`. This is the complete post-SwiGLU activation
with shape `[1, 79, 25600]`, covering every native MLP neuron at every prompt
position. It also extracts the full 79-position trace for target neuron
`L7/N2026+`.

With the daemon running, execute:

```bash
uv run --no-sync python model_daemon.py run scripts/qwen32_layer7/capture_layer7.py
```

The script writes the BF16 tensor artifact to
`outputs/layer7/layer7_mlp_activations.pt` and readable token/target-neuron metadata to
`outputs/layer7/layer7_mlp_activations.json`. Hooks are always removed after the
forward pass so they cannot leak into later daemon requests.

Plot the saved `L7/N2026` trace without starting the daemon:

```bash
uv run python scripts/qwen32_layer7/plot_neuron_activity.py
```

The plot is saved to
`outputs/layer7/sampled_neurons/layer7_n2026_activity.png`.

Generate the same plot for 10 reproducibly sampled Layer 7 neurons:

```bash
uv run --no-sync python scripts/qwen32_layer7/plot_sampled_neurons.py
```

The target plot, 10 random-neuron plots, the four neurons with a higher signed
mean than N2026, and `manifest.json` are written to
`outputs/layer7/sampled_neurons/`. The default random seed is 42 and N2026 is
excluded from the random sample. All neuron plots use identical x-axis limits
computed across the complete comparison set.

Plot the distribution of each Layer 7 neuron's signed mean activation across
the 79-token prompt, with N2026 marked:

```bash
uv run --no-sync python scripts/qwen32_layer7/plot_activation_distribution.py
```

The plot and JSON summary are written alongside the neuron plots in
`outputs/layer7/sampled_neurons/`.

## Prompt variations

Capture every Layer 7 neuron for the original prompt plus five controlled
variations (changed symptoms, changed question, different medical topic,
software debugging, and weather):

```bash
uv run --no-sync python model_daemon.py run scripts/qwen32_layer7/capture_layer7_variations.py
```

The combined BF16 activation artifact is written to
`outputs/layer7/layer7_prompt_variations.pt`, with readable prompts, tokenization,
N2026 traces, next-token predictions, and top-mean neuron summaries in the
adjacent JSON file.

Plot N2026 across tokens and the all-neuron mean-activation distribution for
each of the six prompts:

```bash
uv run --no-sync python scripts/qwen32_layer7/plot_prompt_variations.py
```

The 12 plots use shared axes within each plot type and are written to
`outputs/layer7/prompt_variation_plots/` with a comparison manifest.

## Qwen3.5-0.8B weekday dataset

`scripts/qwen35_weekdays/capture_qwen35_weekdays.py` builds 10 sentence templates and substitutes each
weekday from Monday through Sunday, producing 70 prompts. Every weekday is one
token. For each prompt it captures every post-SwiGLU MLP activation at every
token and all 24 model layers:

```bash
uv run --no-sync python scripts/qwen35_weekdays/capture_qwen35_weekdays.py
```

The BF16 artifact is stored at `outputs/weekdays/qwen35_08b_weekday_activations.pt`.
Each prompt record contains an activation tensor shaped
`[24, sequence_length, 3584]`, plus its tokenization and weekday-token position.
A readable index without the large tensors is saved alongside it as JSON.

## Minimal MLP distribution measurement

Measure Layer 0 post-SwiGLU activations over exactly 10,000 sampled
OpenWebText tokens and plot the signed raw values and absolute magnitudes:

```bash
uv run python scripts/mlp_measurement/measure_mlp_distributions.py \
  --dataset en \
  --layers 0
```

`--dataset` accepts `en` for OpenWebText, `cn` for BAAI CCI3-Data, or
`both`. With `both`, `--num-tokens` is the budget for each corpus, not the
combined budget. CCI3-Data is gated: first accept its terms at
<https://huggingface.co/datasets/BAAI/CCI3-Data> and authenticate locally with
`hf auth login`.

`--model` accepts a full Hugging Face model ID or the short aliases `0.8B` and
`9B`. For example, use Qwen3.5-9B-Base with:

```bash
uv run python scripts/mlp_measurement/measure_mlp_distributions.py \
  --model 9B \
  --layers 0 \
  --neurons all \
  --num-tokens 10000
```

Use `--revision` to pin a non-default model to a branch, tag, or commit hash.
The default 0.8B model remains pinned to the revision used by this project.

`--layers` also accepts `all`, a bracketed list such as `[0,1,2]`, or a
comma-separated list such as `0,1,2`.

`--neurons` uses the same syntax and defaults to `all`. For example, measure
two neurons across three layers with `--layers '[0,1,2]' --neurons '[2,279]'`.
Explicit neuron lists are captured together in one shared corpus pass and then
analyzed independently, so this example writes separate `neuron_0002/` and
`neuron_0279/` results beneath every selected layer without repeating model
inference for each neuron. Selected activations are retained as BF16 on CPU;
one million tokens require about 2 MB per neuron per layer. A single explicit
neuron uses the same folder layout. `--neurons all` retains the pooled
layer-level two-pass behavior because retaining and independently reporting all
neurons would be impractical.

Results are written beneath `outputs/mlp_measurement/<model-name>/layer_00/`
(for example, `outputs/mlp_measurement/Qwen3.5-9B-Base/layer_00/`), with run
metadata at the model-directory level. Corpus-specific artifacts are stored in
`en/` and `cn/` beneath the selected layer or single-neuron directory. Each
result also includes `extreme_contexts.json`, containing the most positive and
most negative occurrences for every selected neuron, with the top ten stored
as ranked lists for each tail. Use `--context-tokens 24` to control how many
source tokens are retained on each side. `report.md` embeds the clean
`raw_distribution.png`, reports the raw activation mean, population variance,
population skewness, population kurtosis, and the standardized ICA log-cosh
non-Gaussianity score, and presents those top-ten contexts with the measured
neuron's activation color-coded across every context token; the ranked extreme
token is outlined and exact values appear on hover. The same statistics, plus excess kurtosis, the log-cosh
expectation and Gaussian reference contrast, and the existing counts and
extrema, are saved in `details.json`. No annotated or magnitude-distribution
PNG is produced.

### Per-neuron metric distributions

Measure each Layer 0 MLP neuron independently over Chinese web text, then plot
histograms across all neurons for mean, population variance, population
skewness, population kurtosis, and the standardized ICA log-cosh score:

```bash
uv run python scripts/mlp_measurement/measure_neuron_metrics.py \
  --model 9B \
  --dataset cn \
  --layers 0 \
  --num-tokens 1000000
```

This uses two inference passes: the first accumulates each neuron's raw moments,
and the second standardizes each neuron with its measured mean and variance and
accumulates the ICA contrast. Results are written to
`outputs/mlp_measurement/<model>/layer_00/neuron_metrics/<dataset>/`, including
the complete metric tensors in `neuron_metrics.pt`, full-range and central-99%
histogram panels, reusable histogram counts in `metric_histograms.pt`,
the 10 most negative- and most positive-skewness neurons in
`skewness_extremes.json`, the 10 lowest- and highest-kurtosis neurons in
`kurtosis_extremes.json`, the 10 lowest and highest ICA log-cosh scores in
`log_cosh_extremes.json`,
`details.json`, and `report.md`.

After an individual-neuron run finishes, assemble its small square raw
histograms into one HTML gallery without loading the model again:

```bash
uv run python scripts/mlp_measurement/render_neuron_histogram_gallery.py \
  --model Qwen/Qwen3.5-9B-Base \
  --layer 20 \
  --dataset en
```

By default the script reads the neuron list from the matching run metadata,
falling back to discovering completed individual-neuron directories. The HTML
and its square plot assets are written in the selected `layer_XX/` directory.

Project the 10 lowest- and 10 highest-kurtosis neurons' `down_proj` columns
directly through the vocabulary unembedding and render their shared-scale
square histograms and strongest signed tokens:

```bash
uv run python scripts/mlp_measurement/project_down_directions_to_vocab.py \
  --model Qwen/Qwen3.5-9B-Base \
  --layer 20 \
  --dataset en
```

This computes `lm_head.weight @ down_proj.weight[:, neuron]` exactly, without
the model's final RMSNorm, and saves both the complete projection tensor and an
HTML gallery beneath `layer_20/down_projection_kurtosis_extremes/en/`.

### Targeted L10:N7404 probe

Test the hypothesis that Layer 10 neuron 7404 responds to scholarly
bibliography/reference formatting and journal-title spans using 75 positive,
factorial, contrast, negative-control, token-boundary, multilingual, and
adversarial prompts:

```bash
uv run python scripts/mlp_measurement/probe_l10_n7404.py
```

The script captures the raw post-SwiGLU activation at every token and writes
`results.json`, `summary.csv`, and an activation-colored `report.html` beneath
`outputs/mlp_measurement/Qwen3.5-9B-Base/l10_n7404_probe/`.

Probe the strong negative tail of Layer 10 neuron 1376 with controlled
abbreviated/expanded journal-name and context contrasts:

```bash
uv run python scripts/mlp_measurement/probe_l10_n1376.py
```

The negative-tail-sorted report is written beneath
`outputs/mlp_measurement/Qwen3.5-9B-Base/l10_n1376_probe/`.

Capture every post-SwiGLU MLP neuron at the final name token in 10 natural
Michael Jordan prompts, 13 matched famous-name counterfactuals, and six
Jordan-meaning controls, then rank strong and Jordan-selective neurons without
rerunning the model:

```bash
uv run python scripts/mlp_measurement/capture_michael_jordan_neurons.py
uv run python scripts/mlp_measurement/analyze_michael_jordan_neurons.py
```

The activation tensor, prompt metadata, and analysis CSVs are written beneath
`outputs/mlp_measurement/Qwen3.5-9B-Base/michael_jordan_name_probe/`.

Render token-level signed activation heatmaps for the four leading
Michael-Jordan/Air-Jordan candidate neurons:

```bash
uv run python scripts/mlp_measurement/render_jordan_candidate_heatmaps.py
```

Run an adversarial all-layer/all-neuron search for Michael-Jordan-selective
features while penalizing known Jordan, basketball, dunk, entity-completion,
and numeric/product false positives:

```bash
uv run python scripts/mlp_measurement/search_michael_jordan_neuron.py
```

### Neuron activation maximization

Alternate continuous input-embedding optimization with nearest-vocabulary
projection for both activation tails, then refine the final discrete sequence.
Each snapped sequence initializes the next continuous round. The objective is
the target neuron's mean activation across every sequence position, and the
saved result includes every round and the complete per-position traces:

```bash
uv run python scripts/mlp_measurement/maximize_neuron_activation.py \
  --layer 0 \
  --neuron 0 \
  --alternating-rounds 3
```

The JSON result is saved inside the corresponding
`layer_XX/neuron_NNNN/` directory. `--steps` applies to each alternating round.
Continuous scores are upper-bound searches; the snapped and refined scores are
produced by actual token sequences.

### Per-neuron corpus extremes

Measure every selected layer's strongest positive activation and strongest
negative activation for each of its 3,584 MLP neurons:

```bash
uv run python scripts/mlp_measurement/measure_neuron_extremes.py \
  --dataset both \
  --layers 0 \
  --num-tokens 1000000
```

With `both`, the script measures the requested number of tokens independently
for OpenWebText and CCI3-Data, then writes `en`, `cn`, and combined `both`
results beneath `layer_XX/neuron_extremes/`. Each result contains signed and
linear-magnitude distribution plots, complete per-neuron extrema tensors, and a
JSON summary. N0 is marked by default; use `--highlight-neurons 0,552` to mark
additional neurons.
