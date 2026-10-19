from __future__ import annotations

import hashlib
import heapq
import html
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
from datasets import load_dataset
from gb10_load_llm import load_model_to_cuda
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
MODEL_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
WEEKDAY_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2"]


class _StopAfterIndexedLayer(Exception):
    """Internal signal used to skip layers after a single indexed layer."""


@dataclass
class SentenceAnalysis:
    sentence: str
    tokens: list[str]
    token_ids: list[int]
    features: list[list[dict[str, float | int]]]
    profile_loader: Callable[[int, int, str], "NeuronProfile"] | None = field(
        default=None, repr=False, compare=False
    )

    @property
    def feature_options(self) -> list[tuple[int, int]]:
        return sorted(
            {
                (int(feature["layer"]), int(feature["neuron"]))
                for token_features in self.features
                for feature in token_features
            }
        )

    def _repr_html_(self) -> str:
        cards = []
        for token, features in zip(self.tokens, self.features, strict=True):
            peak = max((abs(float(item["z_score"])) for item in features), default=0.0)
            rows = "".join(
                f'<div class="nla-score-row"><div class="nla-feature'
                f'{" nla-weak" if peak and abs(float(item["z_score"])) / peak < 0.5 else ""}" '
                f'title="L{int(item["layer"])}:N{int(item["neuron"])}; '
                f'z {float(item["z_score"]):+.4f}" '
                f'style="--width:{(100 * abs(float(item["z_score"])) / peak if peak else 0):.1f}%">'
                f'<span>{int(item["neuron"])}</span>'
                f'<strong>{float(item["z_score"]):+.1f}σ</strong></div></div>'
                for item in features
            )
            token_text = token.strip() or "␠"
            cards.append(
                '<article class="nla-card">'
                f'<header>{html.escape(token_text)}</header>'
                f'<div class="nla-features">{rows}</div></article>'
            )
        return (
            '<style>'
            '.nla-rail{display:grid;grid-template-columns:repeat(auto-fill,minmax(100px,1fr));gap:10px;'
            'padding:2px;color:#151922;'
            'font-family:Inter,ui-sans-serif,system-ui,sans-serif}'
            '.nla-card{min-width:0;border:1px solid #cbd3df;border-radius:8px;background:#fff;padding:6px;'
            'text-align:center;box-shadow:0 1px 2px rgba(20,25,34,.04)}'
            '.nla-card header{min-height:25px;margin:1px 0 6px;text-align:center;font-size:13px;font-weight:850;'
            'white-space:nowrap;overflow:hidden;text-overflow:ellipsis}'
            '.nla-score-row{position:relative;min-height:24px;margin-top:3px}'
            '.nla-feature{display:flex;width:100%;min-height:24px;align-items:center;justify-content:space-between;'
            'gap:5px;padding:3px 5px;overflow:hidden;border:1px solid #d5dce7;border-radius:5px;'
            'background:linear-gradient(90deg,#e7ebf1 0,#e7ebf1 var(--width),transparent var(--width),'
            'transparent 100%);font:700 10px ui-monospace,SFMono-Regular,Consolas,monospace}'
            '.nla-feature strong{font:500 10px ui-monospace,SFMono-Regular,Consolas,monospace;'
            'font-variant-numeric:tabular-nums}.nla-weak{opacity:.35}'
            '</style><div class="nla-rail">' + "".join(cards) + "</div>"
        )

    def _ipython_display_(self) -> None:
        """Display clickable neuron badges and an occurrence panel in notebooks."""
        if self.profile_loader is None:
            from IPython.display import HTML, display

            display(HTML(self._repr_html_()))
            return

        import ipywidgets as widgets
        from IPython.display import HTML, clear_output, display

        style = widgets.HTML(
            """<style>
            .nla-widget-card{border:1px solid #cbd3df;border-radius:8px;padding:6px;background:#fff}
            .nla-widget-token{text-align:center;font:850 13px Inter,system-ui,sans-serif;
              min-height:25px;margin:1px 0 4px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
            .nla-neuron-button button{min-height:24px;padding:2px 5px;border:1px solid #d5dce7;
              border-radius:5px;background:#e7ebf1;font:700 10px ui-monospace,monospace;text-align:left}
            .nla-neuron-button.weak{opacity:.35}
            </style>"""
        )
        output = widgets.Output(layout=widgets.Layout(width="100%"))
        cards = []
        for token, features in zip(self.tokens, self.features, strict=True):
            peak = max((abs(float(item["z_score"])) for item in features), default=0.0)
            buttons = []
            for item in features:
                layer = int(item["layer"])
                neuron = int(item["neuron"])
                z_score = float(item["z_score"])
                tail = "positive" if z_score >= 0 else "negative"
                button = widgets.Button(
                    description=f"{neuron}   {z_score:+.1f}σ",
                    tooltip=f"Profile L{layer}:N{neuron} · {tail} tail",
                    layout=widgets.Layout(width="100%", height="24px"),
                )
                button.add_class("nla-neuron-button")
                if peak and abs(z_score) / peak < 0.5:
                    button.add_class("weak")

                def show_profile(
                    _button: widgets.Button, *, layer: int = layer,
                    neuron: int = neuron, tail: str = tail
                ) -> None:
                    with output:
                        clear_output(wait=True)
                        display(HTML(f"<p><strong>Profiling L{layer}:N{neuron}…</strong></p>"))
                    try:
                        profile = self.profile_loader(layer, neuron, tail)
                    except Exception as error:
                        with output:
                            clear_output(wait=True)
                            display(HTML(
                                '<div style="padding:10px;border:1px solid #e2b8b8;border-radius:7px;'
                                f'color:#8b1e1e;background:#fff7f7">{html.escape(str(error))}</div>'
                            ))
                        return
                    with output:
                        clear_output(wait=True)
                        display(profile)

                button.on_click(show_profile)
                buttons.append(button)
            token_text = html.escape(token.strip() or "␠")
            card = widgets.VBox(
                [widgets.HTML(f'<div class="nla-widget-token">{token_text}</div>'), *buttons],
                layout=widgets.Layout(width="112px"),
            )
            card.add_class("nla-widget-card")
            cards.append(card)
        rail = widgets.Box(
            cards,
            layout=widgets.Layout(
                display="flex", flex_flow="row wrap", gap="10px", width="100%"
            ),
        )
        display(widgets.VBox([style, rail, output], layout=widgets.Layout(width="100%")))


@dataclass
class NeuronProfile:
    layer: int
    neuron: int
    samples_processed: int
    occurrences: list[dict[str, Any]]

    def _repr_html_(self) -> str:
        maximum = max(
            (
                abs(float(token["z_score"]))
                for item in self.occurrences
                for token in item["context_tokens"]
            ),
            default=1.0,
        )

        def color(z_score: float) -> str:
            # A square-root scale keeps smaller contextual activations visible.
            t = max(0.0, min(1.0, abs(z_score) / max(maximum, 1e-9))) ** 0.5
            return f"rgb(255,{round(255 - 46 * t)},{round(255 - 185 * t)})"

        has_tails = any("tail" in item for item in self.occurrences)
        present_tails = {
            str(item.get("tail")) for item in self.occurrences if "tail" in item
        }
        tail_groups = (
            [
                (tail, f"{tail.title()} tail")
                for tail in ("positive", "negative")
                if tail in present_tails
            ]
            if has_tails else [("all", "Top occurrences")]
        )
        sections = []
        for tail, label in tail_groups:
            items = [
                item for item in self.occurrences
                if tail == "all" or item.get("tail") == tail
            ]
            cards = []
            for rank, item in enumerate(items, start=1):
                context = "".join(
                    f'<span class="nlp-token{" nlp-active" if token["is_active"] else ""}" '
                    f'style="background:{color(float(token["z_score"]))}" '
                    f'title="z {float(token["z_score"]):+.4f}; activation '
                    f'{float(token["activation"]):+.6f}; token {token["token_position"]}">'
                    f'{html.escape(token["text"])}</span>'
                    for token in item["context_tokens"]
                )
                sign_class = " nlp-negative" if float(item["z_score"]) < 0 else ""
                peak_count = int(item.get("occurrence_count", 1))
                peak_label = f'<span>{peak_count} peaks</span>' if peak_count > 1 else ""
                cards.append(
                    '<article class="nlp-card">'
                    '<header class="nlp-meta">'
                    f'<span class="nlp-rank">#{rank}</span>'
                    f'<strong class="nlp-score{sign_class}">{float(item["z_score"]):+.3f}σ</strong>'
                    f'<span>raw&nbsp;{float(item["activation"]):+.5f}</span>'
                    f'{peak_label}'
                    f'<span class="nlp-sample">sample&nbsp;{int(item["sample_index"]):,}</span>'
                    '</header>'
                    f'<div class="nlp-context">{context}</div>'
                    '</article>'
                )
            sections.append(
                f'<section class="nlp-tail nlp-{tail}"><h4>{label} '
                f'<span>{len(items)} samples</span></h4><div class="nlp-list">'
                + "".join(cards) + "</div></section>"
            )
        return (
            '<style>'
            '.nlp-root{max-width:1180px;color:#18212b;font-family:Inter,ui-sans-serif,system-ui,sans-serif}'
            '.nlp-heading{display:flex;align-items:baseline;gap:10px;margin:6px 0 14px}'
            '.nlp-heading h3{font-size:20px;margin:0}.nlp-heading span{color:#667381;font-size:13px}'
            '.nlp-tail{margin-top:16px}.nlp-tail h4{margin:0 0 8px;font-size:14px}'
            '.nlp-tail h4 span{margin-left:6px;color:#7a8693;font-size:11px;font-weight:500}'
            '.nlp-negative h4{color:#315b9a}.nlp-positive h4{color:#a43b12}'
            '.nlp-list{display:grid;gap:10px}.nlp-card{overflow:hidden;border:1px solid #dbe2e8;'
            'border-radius:10px;background:#fff;box-shadow:0 1px 2px rgba(22,34,48,.04)}'
            '.nlp-meta{display:flex;align-items:center;gap:12px;padding:8px 12px;background:#f6f8fa;'
            'border-bottom:1px solid #e6ebef;color:#667381;font:12px ui-monospace,SFMono-Regular,monospace}'
            '.nlp-rank{display:inline-grid;place-items:center;min-width:30px;height:24px;border-radius:12px;'
            'background:#e4eaf0;color:#33404d;font-weight:700}.nlp-score{color:#a43b12;font-size:14px}'
            '.nlp-score.nlp-negative{color:#315b9a}.nlp-sample{margin-left:auto}'
            '.nlp-context{padding:13px 15px;font:13px/1.9 ui-monospace,SFMono-Regular,Consolas,monospace;'
            'text-align:left;white-space:pre-wrap;overflow-wrap:anywhere}'
            '.nlp-token{padding:2px 1px;border-radius:3px}.nlp-active{position:relative;font-weight:800;'
            'outline:2px solid #d18b00;outline-offset:1px;z-index:1}'
            '</style>'
            '<div class="nlp-root"><div class="nlp-heading">'
            f'<h3>L{self.layer}:N{self.neuron}</h3><span>separate activation tails · '
            f'{self.samples_processed:,} OpenWebText documents</span></div>'
            + "".join(sections) + "</div>"
        )


class JITNeuronLens:
    def __init__(
        self,
        cache_dir: str | Path = "outputs/jit_neuron_lens",
        calibration_path: str | Path | None = None,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.calibration_path = Path(calibration_path or self.cache_dir / "calibration.pt")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        self.model, _ = load_model_to_cuda(
            AutoModelForCausalLM,
            MODEL_ID,
            revision=MODEL_REVISION,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            touch="auto",
            return_touched=True,
        )
        self.model.eval()
        self.num_layers = len(self.model.model.layers)
        self.width = int(self.model.config.intermediate_size)
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None
        self.minimum: torch.Tensor | None = None
        self.maximum: torch.Tensor | None = None
        if self.calibration_path.exists():
            state = torch.load(self.calibration_path, map_location="cpu", weights_only=True)
            self.mean, self.std = state["mean"], state["std"]
            self.minimum, self.maximum = state.get("minimum"), state.get("maximum")

    def calibrate(
        self,
        num_samples: int = 10_000,
        max_length: int = 256,
        batch_size: int = 16,
        seed: int = 123,
        shuffle_buffer: int = 50_000,
    ) -> Path:
        sums = torch.zeros(self.num_layers, self.width, dtype=torch.float64)
        square_sums = torch.zeros_like(sums)
        minima = torch.full((self.num_layers, self.width), torch.inf)
        maxima = torch.full((self.num_layers, self.width), -torch.inf)
        token_count = 0
        batch_mask: torch.Tensor | None = None
        handles = []

        for layer, module in enumerate(self.model.model.layers):
            def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], *, layer=layer) -> None:
                assert batch_mask is not None
                values = inputs[0].detach().float()
                mask = batch_mask[:, :, None]
                sums[layer] += (values * mask).sum((0, 1)).double().cpu()
                square_sums[layer] += (values.square() * mask).sum((0, 1)).double().cpu()
                minima[layer] = torch.minimum(
                    minima[layer], values.masked_fill(mask == 0, torch.inf).amin((0, 1)).cpu()
                )
                maxima[layer] = torch.maximum(
                    maxima[layer], values.masked_fill(mask == 0, -torch.inf).amax((0, 1)).cpu()
                )

            handles.append(module.mlp.down_proj.register_forward_pre_hook(hook))

        stream = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
        iterator = iter(stream.shuffle(seed=seed, buffer_size=shuffle_buffer).take(num_samples))
        processed = 0
        started = time.monotonic()
        try:
            while processed < num_samples:
                texts = []
                for _ in range(min(batch_size, num_samples - processed)):
                    try:
                        text = next(iterator)["text"]
                    except StopIteration:
                        break
                    if text.strip():
                        texts.append(text)
                if not texts:
                    break
                encoded = self.tokenizer(
                    texts, add_special_tokens=False, truncation=True, max_length=max_length,
                    padding=True, return_tensors="pt"
                )
                batch_mask = encoded.attention_mask.to(self.model.device, dtype=torch.float32)
                with torch.inference_mode():
                    self.model.model(
                        input_ids=encoded.input_ids.to(self.model.device),
                        attention_mask=encoded.attention_mask.to(self.model.device),
                        use_cache=False,
                    )
                count = int(encoded.attention_mask.sum())
                token_count += count
                processed += len(texts)
                if processed % 1_000 < batch_size:
                    print(f"calibration {processed:,}/{num_samples:,} · {token_count:,} tokens")
        finally:
            for handle in handles:
                handle.remove()

        mean = sums / token_count
        variance = torch.clamp(square_sums / token_count - mean.square(), min=1e-12)
        self.mean = mean.float()
        self.std = variance.sqrt().float()
        self.minimum = minima
        self.maximum = maxima
        torch.save(
            {
                "mean": self.mean,
                "std": self.std,
                "minimum": self.minimum,
                "maximum": self.maximum,
                "samples": processed,
                "tokens": token_count,
                "seed": seed,
                "max_length": max_length,
                "elapsed_seconds": time.monotonic() - started,
            },
            self.calibration_path,
        )
        print(f"saved calibration: {self.calibration_path.resolve()}")
        return self.calibration_path

    def analyze(self, sentence: str, layer: int, top_k: int = 3) -> SentenceAnalysis:
        if self.mean is None or self.std is None:
            raise RuntimeError("Run lens.calibrate() once before sentence analysis")
        if not 0 <= layer < self.num_layers:
            raise ValueError(f"layer must be in [0, {self.num_layers - 1}], got {layer}")
        captured: dict[int, torch.Tensor] = {}
        def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            captured[layer] = inputs[0].detach().float().cpu()[0]

        handle = self.model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(hook)
        encoded = self.tokenizer(sentence, add_special_tokens=False, return_tensors="pt")
        try:
            with torch.inference_mode():
                self.model.model(input_ids=encoded.input_ids.to(self.model.device), use_cache=False)
        finally:
            handle.remove()
        activations = captured[layer]
        z_by_token = (activations - self.mean[layer]) / self.std[layer]
        _, top_indices = torch.topk(z_by_token.abs(), top_k, dim=-1)
        features = []
        for position in range(activations.shape[0]):
            token_features = []
            for neuron_index in top_indices[position]:
                neuron = int(neuron_index)
                token_features.append(
                    {
                        "layer": layer,
                        "neuron": neuron,
                        "z_score": float(z_by_token[position, neuron]),
                        "activation": float(activations[position, neuron]),
                    }
                )
            features.append(token_features)
        ids = encoded.input_ids[0].tolist()
        return SentenceAnalysis(
            sentence=sentence,
            tokens=[self.tokenizer.decode([token_id]) for token_id in ids],
            token_ids=ids,
            features=features,
            profile_loader=lambda selected_layer, neuron, tail: self.profile_indexed_neuron(
                layer=selected_layer, neuron=neuron, tail=tail
            ),
        )

    def build_occurrence_index(
        self,
        num_samples: int = 100_000,
        candidates_per_neuron: int = 20,
        max_length: int = 256,
        batch_size: int = 32,
        seed: int = 42,
        shuffle_buffer: int = 100_000,
        layers: str | int | list[int] = "all",
        force: bool = False,
    ) -> Path:
        """Build exact positive- and negative-tail indexes for selected layers."""
        if self.mean is None or self.std is None:
            raise RuntimeError("Run lens.calibrate() once before building the index")
        if layers == "all":
            selected_layers = list(range(self.num_layers))
            scope = "all"
        elif isinstance(layers, int) and not isinstance(layers, bool):
            selected_layers = [layers]
            scope = f"L{layers}"
        elif isinstance(layers, list) and layers:
            if any(not isinstance(layer, int) or isinstance(layer, bool) for layer in layers):
                raise ValueError("every item in layers must be an integer")
            selected_layers = sorted(set(layers))
            scope = "L" + "-".join(str(layer) for layer in selected_layers)
        else:
            raise ValueError(
                "layers must be 'all', one integer, or a non-empty list of integers"
            )
        if any(not 0 <= layer < self.num_layers for layer in selected_layers):
            raise ValueError(f"layer numbers must be from 0 to {self.num_layers - 1}")
        stem = f"n{num_samples}-m{max_length}-s{seed}-k{candidates_per_neuron}"
        index_path = self.cache_dir / f"occurrence-index-v2-tails-{scope}-{stem}.pt"
        corpus_path = self.cache_dir / f"token-corpus-n{num_samples}-m{max_length}-s{seed}.pt"
        if index_path.exists() and corpus_path.exists() and not force:
            print(f"using existing index: {index_path.resolve()}")
            return index_path

        device = self.model.device
        means = self.mean.to(device)
        stds = self.std.to(device)
        def empty_tail() -> dict[str, torch.Tensor]:
            return {
                "rank_value": torch.full(
                    (candidates_per_neuron, self.width), -torch.inf, device=device
                ),
                "z_score": torch.zeros((candidates_per_neuron, self.width), device=device),
                "sample": torch.full(
                    (candidates_per_neuron, self.width), -1, dtype=torch.int32, device=device
                ),
                "position": torch.zeros(
                    (candidates_per_neuron, self.width), dtype=torch.int16, device=device
                ),
            }

        states = {
            layer: {"positive": empty_tail(), "negative": empty_tail()}
            for layer in selected_layers
        }
        batch_state: dict[str, Any] = {}
        handles = []
        stop_after_layer = None if layers == "all" else max(selected_layers)
        for layer in selected_layers:
            module = self.model.model.layers[layer]
            def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], *, layer=layer) -> None:
                values = inputs[0].detach().float()
                batch, sequence, _ = values.shape
                z = (values - means[layer]) / stds[layer]
                flat_z = z.reshape(batch * sequence, self.width)
                valid = batch_state["mask"].reshape(-1)
                local_k = min(candidates_per_neuron, int(valid.sum()))
                for tail, rank_values in (
                    ("positive", flat_z),
                    ("negative", -flat_z),
                ):
                    ranked = rank_values.masked_fill(~valid[:, None], -torch.inf)
                    local_rank, local_offset = torch.topk(ranked, local_k, dim=0)
                    local_z = torch.gather(flat_z, 0, local_offset)
                    local_sample = (
                        batch_state["base"]
                        + local_offset.div(sequence, rounding_mode="floor")
                    ).to(torch.int32)
                    local_position = local_offset.remainder(sequence).to(torch.int16)

                    state = states[layer][tail]
                    combined_rank = torch.cat((state["rank_value"], local_rank), dim=0)
                    combined_z = torch.cat((state["z_score"], local_z), dim=0)
                    combined_sample = torch.cat((state["sample"], local_sample), dim=0)
                    combined_position = torch.cat((state["position"], local_position), dim=0)
                    best_rank, selection = torch.topk(
                        combined_rank, candidates_per_neuron, dim=0
                    )
                    state["rank_value"] = best_rank
                    state["z_score"] = torch.gather(combined_z, 0, selection)
                    state["sample"] = torch.gather(combined_sample, 0, selection)
                    state["position"] = torch.gather(combined_position, 0, selection)
                if layer == stop_after_layer:
                    raise _StopAfterIndexedLayer

            handles.append(module.mlp.down_proj.register_forward_pre_hook(hook))

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise RuntimeError("Tokenizer must define a padding token")
        corpus_ids = torch.full(
            (num_samples, max_length), pad_id, dtype=torch.int32
        )
        corpus_lengths = torch.zeros(num_samples, dtype=torch.int16)
        stream = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
        iterator = iter(stream.shuffle(seed=seed, buffer_size=shuffle_buffer).take(num_samples))
        processed = 0
        started = time.monotonic()
        progress = tqdm(
            total=num_samples,
            desc="Building occurrence index",
            unit="doc",
            dynamic_ncols=True,
        )
        try:
            while processed < num_samples:
                texts = []
                for _ in range(min(batch_size, num_samples - processed)):
                    try:
                        text = next(iterator)["text"]
                    except StopIteration:
                        break
                    if text.strip():
                        texts.append(text)
                if not texts:
                    break
                encoded = self.tokenizer(
                    texts, add_special_tokens=False, truncation=True,
                    max_length=max_length, padding=True, return_tensors="pt"
                )
                size, sequence = encoded.input_ids.shape
                corpus_ids[processed : processed + size, :sequence] = encoded.input_ids.to(torch.int32)
                corpus_lengths[processed : processed + size] = encoded.attention_mask.sum(1).to(torch.int16)
                batch_state["mask"] = encoded.attention_mask.to(device, dtype=torch.bool)
                batch_state["base"] = processed
                try:
                    with torch.inference_mode():
                        self.model.model(
                            input_ids=encoded.input_ids.to(device),
                            attention_mask=encoded.attention_mask.to(device),
                            use_cache=False,
                        )
                except _StopAfterIndexedLayer:
                    if stop_after_layer is None:
                        raise
                processed += size
                progress.update(size)
        finally:
            progress.close()
            for handle in handles:
                handle.remove()

        torch.save(
            {
                "input_ids": corpus_ids[:processed],
                "lengths": corpus_lengths[:processed],
                "metadata": {
                    "model_id": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                    "num_samples": processed,
                    "requested_num_samples": num_samples,
                    "max_length": max_length,
                    "seed": seed,
                    "shuffle_buffer": shuffle_buffer,
                },
            },
            corpus_path,
        )
        torch.save(
            {
                **{
                    f"{tail}_{field}": torch.stack(
                        [states[layer][tail][field].cpu() for layer in selected_layers]
                    )
                    for tail in ("positive", "negative")
                    for field in ("z_score", "sample", "position")
                },
                "metadata": {
                    "index_version": 2,
                    "model_id": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                    "num_samples": processed,
                    "requested_num_samples": num_samples,
                    "max_length": max_length,
                    "seed": seed,
                    "shuffle_buffer": shuffle_buffer,
                    "layers": selected_layers,
                    "candidates_per_neuron": candidates_per_neuron,
                    "ranking": "separate positive and negative per-neuron z-score tails",
                    "corpus_path": str(corpus_path.resolve()),
                    "elapsed_seconds": time.monotonic() - started,
                },
            },
            index_path,
        )
        print(f"saved occurrence index: {index_path.resolve()}")
        print(f"saved token corpus: {corpus_path.resolve()}")
        return index_path

    def _find_occurrence_index(
        self, layer: int, num_samples: int, max_length: int, seed: int, top_k: int
    ) -> Path | None:
        for path in sorted(self.cache_dir.glob("occurrence-index-*.pt")):
            state = torch.load(path, map_location="cpu", weights_only=True)
            metadata = state["metadata"]
            indexed_layers = metadata.get("layers", list(range(self.num_layers)))
            if (
                int(metadata.get("index_version", 1)) >= 2
                and int(metadata.get("requested_num_samples", metadata["num_samples"])) == num_samples
                and int(metadata["max_length"]) == max_length
                and int(metadata["seed"]) == seed
                and int(metadata["candidates_per_neuron"]) >= top_k
                and layer in indexed_layers
            ):
                return path
        return None

    def profile_indexed_neuron(
        self,
        layer: int,
        neuron: int,
        top_k: int = 10,
        context_tokens: int = 24,
        tail: str | None = None,
    ) -> NeuronProfile:
        """Profile a neuron using the largest compatible two-tail index."""
        if tail not in {None, "positive", "negative"}:
            raise ValueError("tail must be 'positive', 'negative', or None")
        matches: list[tuple[int, Path, dict[str, Any]]] = []
        for path in self.cache_dir.glob("occurrence-index-v2-tails-*.pt"):
            state = torch.load(path, map_location="cpu", weights_only=True)
            metadata = state["metadata"]
            if (
                layer in metadata.get("layers", [])
                and int(metadata["candidates_per_neuron"]) >= top_k
            ):
                sample_count = int(
                    metadata.get("requested_num_samples", metadata["num_samples"])
                )
                matches.append((sample_count, path, metadata))
        if not matches:
            raise RuntimeError(
                f"No positive/negative occurrence index contains layer {layer}. "
                f"Run lens.build_occurrence_index(layers={layer}) first."
            )
        _, index_path, metadata = max(matches, key=lambda item: item[0])
        profile = self.profile_neuron(
            layer=layer,
            neuron=neuron,
            num_samples=int(
                metadata.get("requested_num_samples", metadata["num_samples"])
            ),
            top_k=top_k,
            max_length=int(metadata["max_length"]),
            context_tokens=context_tokens,
            seed=int(metadata["seed"]),
        )
        if tail is None:
            return profile
        return NeuronProfile(
            layer=profile.layer,
            neuron=profile.neuron,
            samples_processed=profile.samples_processed,
            occurrences=[
                occurrence for occurrence in profile.occurrences
                if occurrence.get("tail") == tail
            ],
        )

    def _profile_from_index(
        self,
        index_path: Path,
        layer: int,
        neuron: int,
        top_k: int,
        context_tokens: int,
    ) -> NeuronProfile:
        index = torch.load(index_path, map_location="cpu", weights_only=True)
        metadata = index["metadata"]
        corpus = torch.load(
            metadata["corpus_path"], map_location="cpu", weights_only=True
        )
        indexed_layers = metadata.get("layers", list(range(self.num_layers)))
        stored_layer = indexed_layers.index(layer)
        tail_positions: dict[str, dict[int, list[int]]] = {}
        for tail in ("positive", "negative"):
            candidate_samples = index[f"{tail}_sample"][stored_layer, :, neuron].tolist()
            candidate_positions = index[f"{tail}_position"][stored_layer, :, neuron].tolist()
            sample_positions: dict[int, list[int]] = {}
            for sample_id, position in zip(
                candidate_samples, candidate_positions, strict=True
            ):
                if sample_id < 0:
                    continue
                if sample_id not in sample_positions:
                    if len(sample_positions) >= top_k:
                        continue
                    sample_positions[sample_id] = []
                if position not in sample_positions[sample_id]:
                    sample_positions[sample_id].append(position)
            tail_positions[tail] = sample_positions

        unique_samples = list(
            dict.fromkeys(
                sample_id
                for tail in ("positive", "negative")
                for sample_id in tail_positions[tail]
            )
        )
        sample_ids = torch.tensor(unique_samples, dtype=torch.long)
        if len(sample_ids) == 0:
            return NeuronProfile(layer, neuron, int(metadata["num_samples"]), [])
        lengths = corpus["lengths"][sample_ids].to(torch.long)
        input_ids = corpus["input_ids"][sample_ids].to(torch.long)
        attention_mask = (
            torch.arange(input_ids.shape[1])[None, :] < lengths[:, None]
        ).to(torch.long)

        captured: torch.Tensor | None = None
        def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            nonlocal captured
            captured = inputs[0][..., neuron].detach().float().cpu()
            raise _StopAfterIndexedLayer

        handle = self.model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(hook)
        try:
            try:
                with torch.inference_mode():
                    self.model.model(
                        input_ids=input_ids.to(self.model.device),
                        attention_mask=attention_mask.to(self.model.device),
                        use_cache=False,
                    )
            except _StopAfterIndexedLayer:
                pass
        finally:
            handle.remove()
        assert captured is not None
        z_values = (captured - float(self.mean[layer, neuron])) / float(self.std[layer, neuron])
        occurrences = []
        sample_rows = {sample_id: row for row, sample_id in enumerate(unique_samples)}
        for tail in ("positive", "negative"):
            for sample_id, positions in tail_positions[tail].items():
                row = sample_rows[sample_id]
                length = int(lengths[row])
                active_positions = sorted(positions)
                chooser = max if tail == "positive" else min
                position = chooser(active_positions, key=lambda p: float(z_values[row, p]))
                ids = input_ids[row, :length].tolist()
                start = max(0, min(active_positions) - context_tokens)
                end = min(length, max(active_positions) + context_tokens + 1)
                active_position_set = set(active_positions)
                tokens = [
                    {
                        "text": self.tokenizer.decode([ids[p]]),
                        "token_id": ids[p],
                        "token_position": p,
                        "activation": float(captured[row, p]),
                        "z_score": float(z_values[row, p]),
                        "is_active": p in active_position_set,
                    }
                    for p in range(start, end)
                ]
                occurrences.append(
                    {
                        "tail": tail,
                        "activation": float(captured[row, position]),
                        "z_score": float(z_values[row, position]),
                        "sample_index": sample_id,
                        "token_position": position,
                        "active_positions": active_positions,
                        "occurrence_count": len(active_positions),
                        "context_tokens": tokens,
                    }
                )
        return NeuronProfile(
            layer, neuron, int(metadata["num_samples"]), occurrences
        )

    def profile_neuron(
        self,
        layer: int,
        neuron: int,
        num_samples: int = 100_000,
        top_k: int = 10,
        max_length: int = 256,
        batch_size: int = 32,
        context_tokens: int = 24,
        seed: int = 42,
        shuffle_buffer: int = 100_000,
        force: bool = False,
    ) -> NeuronProfile:
        if self.mean is None or self.std is None:
            raise RuntimeError("Run lens.calibrate() once before profiling a neuron")
        config = f"tails-v5-grouped-L{layer}N{neuron}-n{num_samples}-k{top_k}-m{max_length}-s{seed}"
        cache_path = self.cache_dir / f"profile-{hashlib.sha1(config.encode()).hexdigest()[:12]}.json"
        if cache_path.exists() and not force:
            data = json.loads(cache_path.read_text())
            return NeuronProfile(layer, neuron, data["samples_processed"], data["occurrences"])

        index_path = self._find_occurrence_index(
            layer=layer, num_samples=num_samples, max_length=max_length,
            seed=seed, top_k=top_k
        )
        if index_path is not None:
            print(f"using occurrence index: {index_path.resolve()}")
            profile = self._profile_from_index(
                index_path, layer, neuron, top_k, context_tokens
            )
            cache_path.write_text(
                json.dumps(
                    {
                        "samples_processed": profile.samples_processed,
                        "occurrences": profile.occurrences,
                        "source_index": str(index_path),
                    }
                )
            )
            return profile

        captured: torch.Tensor | None = None
        def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            nonlocal captured
            captured = inputs[0][..., neuron].detach().float().cpu()

        handle = self.model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(hook)
        stream = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
        iterator = iter(stream.shuffle(seed=seed, buffer_size=shuffle_buffer).take(num_samples))
        heap: list[tuple[float, int, dict[str, Any]]] = []
        serial = 0
        processed = 0
        try:
            while processed < num_samples:
                texts = []
                for _ in range(min(batch_size, num_samples - processed)):
                    try:
                        text = next(iterator)["text"]
                    except StopIteration:
                        break
                    if text.strip():
                        texts.append(text)
                if not texts:
                    break
                encoded = self.tokenizer(
                    texts, add_special_tokens=False, truncation=True, max_length=max_length,
                    padding=True, return_tensors="pt"
                )
                with torch.inference_mode():
                    self.model.model(
                        input_ids=encoded.input_ids.to(self.model.device),
                        attention_mask=encoded.attention_mask.to(self.model.device), use_cache=False
                    )
                assert captured is not None
                values = captured.clone()
                neuron_mean = float(self.mean[layer, neuron])
                neuron_std = float(self.std[layer, neuron])
                z_values = (values - neuron_mean) / neuron_std
                z_values[encoded.attention_mask == 0] = 0
                local_k = min(top_k, int(encoded.attention_mask.sum()))
                magnitudes, offsets = torch.topk(z_values.abs().reshape(-1), local_k)
                for magnitude, offset in zip(magnitudes.tolist(), offsets.tolist(), strict=True):
                    if len(heap) >= top_k and magnitude <= heap[0][0]:
                        continue
                    row = offset // values.shape[1]
                    position = offset % values.shape[1]
                    activation = float(values[row, position])
                    z_score = float(z_values[row, position])
                    length = int(encoded.attention_mask[row].sum())
                    ids = encoded.input_ids[row, :length].tolist()
                    start, end = max(0, position-context_tokens), min(length, position+context_tokens+1)
                    tokens = [
                        {
                            "text": self.tokenizer.decode([ids[p]]), "token_id": ids[p],
                            "token_position": p, "activation": float(values[row, p]),
                            "z_score": float(z_values[row, p]),
                            "is_active": p == position,
                        }
                        for p in range(start, end)
                    ]
                    item = {
                        "activation": activation, "z_score": z_score,
                        "sample_index": processed + row,
                        "token_position": position, "context_tokens": tokens,
                    }
                    serial += 1
                    candidate = (magnitude, serial, item)
                    if len(heap) < top_k:
                        heapq.heappush(heap, candidate)
                    else:
                        heapq.heapreplace(heap, candidate)
                processed += len(texts)
                if processed % 1_000 < batch_size:
                    print(f"L{layer}:N{neuron} profile {processed:,}/{num_samples:,}")
        finally:
            handle.remove()
        occurrences = [item for _, _, item in sorted(heap, reverse=True)]
        cache_path.write_text(json.dumps({"samples_processed": processed, "occurrences": occurrences}))
        return NeuronProfile(layer, neuron, processed, occurrences)

