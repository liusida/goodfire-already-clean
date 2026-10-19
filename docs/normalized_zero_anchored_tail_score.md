# Normalized Zero-Anchored Tail Score

## Purpose

The normalized zero-anchored tail score measures whether a neuron has a small
number of consistently large activations relative to its ordinary activation
scale.

It is designed for post-SwiGLU neuron activations, where zero has a direct
interpretation: at zero, the neuron contributes nothing through its
down-projection direction. The score therefore measures tail magnitude from
zero rather than from the sample mean or median.

The score is useful alongside kurtosis:

- Excess kurtosis measures global non-Gaussianity and is highly sensitive to
  rare extremes.
- The normalized zero-anchored tail score gives those extremes an interpretable
  magnitude relative to the neuron's bulk activity.
- The two directional components show whether the rare activity is positive or
  negative.

## Definition

Let a neuron have activations

\[
a_1, a_2, \ldots, a_N.
\]

Sort them in ascending order:

\[
a_{(1)} \leq a_{(2)} \leq \cdots \leq a_{(N)}.
\]

Choose a recurrence count \(k\). The current Pile-10k experiment uses
\(k=20\).

Define the negative and positive raw tail magnitudes as

\[
R_- = \max\left(-a_{(k)}, 0\right)
\]

and

\[
R_+ = \max\left(a_{(N-k+1)}, 0\right).
\]

Thus, \(R_-\) is the magnitude of the 20th-most-negative activation and
\(R_+\) is the 20th-most-positive activation. At least \(k\) observed token
positions reach the corresponding magnitude.

Define the neuron's bulk activation scale as

\[
B = Q_{0.90}(|a|),
\]

the 90th percentile of absolute activation magnitude. The directional,
normalized tail scores are

\[
T_- = \frac{R_-}{B}, \qquad T_+ = \frac{R_+}{B}.
\]

The unified score is

\[
\boxed{T = \max(T_-, T_+)}.
\]

The winning direction is negative when \(T_- > T_+\), and positive otherwise.

## Tail asymmetry

Directional asymmetry is summarized by

\[
A = \frac{T_+ - T_-}{T_+ + T_-}.
\]

Because both directional scores have the same denominator, this is equivalent
to computing the expression from \(R_+\) and \(R_-\).

- \(A \approx -1\): strongly negative-tail dominated.
- \(A \approx 0\): similar tail magnitudes on both sides.
- \(A \approx +1\): strongly positive-tail dominated.

## Why these choices

### Zero anchor

Mean- or median-centered measures treat the center of the observed distribution
as the reference point. For post-SwiGLU activations, zero is more meaningful:
it represents no contribution from that neuron. Anchoring at zero also handles
distributions whose dense bulk is slightly shifted away from zero.

### The kth extreme

Using the maximum would allow a single anomalous token to determine the score.
Using the kth extreme requires recurrence: with \(k=20\), twenty token positions
must attain at least the reported raw magnitude.

### Bulk normalization

Raw activation magnitudes are not comparable across neurons under multiplicative
rescaling. If all activations are multiplied by a positive constant \(c\), then
both the raw tail magnitude and \(Q_{0.90}(|a|)\) are multiplied by \(c\), so
their ratio is unchanged:

\[
\frac{cR}{cB} = \frac{R}{B}.
\]

The 90th percentile is far enough into the bulk to provide a stable scale, but
not so deep into the tail that a rare detached component usually determines the
denominator. This avoids the failure observed with a fixed 99.99th-percentile
"shoulder": if a rare component contains more than 0.01% of positions, that
shoulder can fall inside the component it is meant to detect.

## Interpretation

A score of \(T=150\) means that the winning-side 20th-extreme magnitude is 150
times the neuron's 90th-percentile absolute activation magnitude. It does not
mean that the activation is 150 standard deviations from the mean.

In the 100-neuron Qwen3-32B Layer 0 experiment:

| Neuron | Score | Rank | Direction | Interpretation |
| --- | ---: | ---: | --- | --- |
| N10361 | 150.14 | 19/100 | Negative | A large recurrent negative component relative to a small bulk scale |
| N18862 | 208.00 | 13/100 | Positive | A sparse, detached positive activation regime |
| N17931 | 10.62 | 99/100 | Negative | A comparatively smooth distribution without a remote tail component |

These examples illustrate why normalization matters: raw magnitude places
N10361 high and N17931 low, while the bulk normalization makes that comparison
invariant to multiplicative neuron scale.

## Current implementation

The implementation is in
[`scripts/qwen32_layer0/render_neuron_sample_html.py`](../scripts/qwen32_layer0/render_neuron_sample_html.py).

For the saved Layer 0 scan:

- Model: Qwen3-32B, Layer 0.
- Dataset: Pile-10k.
- Positions per neuron: 15,392,600.
- Number of sampled neurons: 100.
- Tail recurrence count: \(k=20\).
- Extreme cutoffs: exact over all positions.
- Bulk scale: estimated from 1,026,174 deterministically strided positions per
  neuron, using stride 15.
- Stored score field: `robust_tail_metrics.unified_tail_score`.
- Directional fields: `negative_normalized_zero_tail` and
  `positive_normalized_zero_tail`.
- Raw fields: `negative_zero_anchored_tail` and
  `positive_zero_anchored_tail`.
- Bulk field: `bulk_absolute_scale_estimate`.

The interactive report is
[`outputs/layer0/sample_100_seed_0_pile10k/neurons.html`](../outputs/layer0/sample_100_seed_0_pile10k/neurons.html).
The histogram-only ranking is
[`outputs/layer0/sample_100_seed_0_pile10k/histograms_by_normalized_tail.html`](../outputs/layer0/sample_100_seed_0_pile10k/histograms_by_normalized_tail.html).

## Limitations

### It measures relative tail magnitude, not interpretability

A high score identifies recurrent activations that are large relative to the
bulk. The corresponding contexts must still be examined to determine whether
the tail represents a coherent and interpretable feature.

### It does not directly measure density separation

A continuous heavy tail can receive a high score even without a low-density gap
between the bulk and the extremes. Histograms and extreme contexts remain useful
for distinguishing a detached rare mode from a smooth heavy tail.

### The recurrence count depends on dataset size

A fixed \(k=20\) corresponds to a different empirical frequency when \(N\)
changes. Comparisons across datasets should keep both \(N\) and \(k\) fixed, or
define \(k\) from a fixed tail probability.

### Near-zero bulk scales require care

If \(Q_{0.90}(|a|)=0\), the normalized score is undefined. This can occur for a
neuron that is exactly zero on at least 90% of positions. Such neurons should be
reported separately or evaluated with a higher bulk quantile; silently adding a
global epsilon would make scores depend on an arbitrary scale.

### The score excludes the down-projection magnitude

This score describes the activation distribution. It does not measure the size
of the neuron's final residual-stream contribution. A functional-effect metric
could additionally multiply the raw tail magnitude by the norm of the neuron's
down-projection vector.

## Recommended reporting

Report the following together:

1. Normalized zero-anchored tail score \(T\).
2. Winning direction.
3. Tail asymmetry \(A\).
4. Raw directional magnitudes \(R_-\) and \(R_+\).
5. Bulk scale \(Q_{0.90}(|a|)\).
6. Excess kurtosis and signed skewness.
7. The values of \(N\) and \(k\).

This keeps the headline score concise while preserving enough information to
identify scale effects, directional structure, and differences between global
non-Gaussianity and recurrent extreme activation.
