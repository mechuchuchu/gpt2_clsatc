# Random-init GPT-2: Comparing Learning Scope Across Transformer Components

## 1. Objective

Train a randomly initialized GPT-2 model on **2B tokens from FineWeb-Edu** and investigate how language-modeling performance changes as the set of trainable components is gradually expanded.

Instead of immediately fine-tuning the entire Transformer, we compare three training regimes:

1. **Embedding-only**
2. **Embedding + Attention**
3. **Full training**

All experiments start from the **same random initialization** and use the same data and optimization setup.

The main question is:

> How much language-modeling capability can be learned by training only embeddings, by additionally training attention, and finally by training the full Transformer?

## 2. Base Model

* Architecture: GPT-2 Small
* Parameters: ~124M
* Layers: 12
* Hidden size: 768
* Attention heads: 12
* Context length: 1024
* Tokenizer: GPT-2 BPE
* Dataset: FineWeb-Edu
* Training budget: 2B tokens
* Initialization: random
* Optimizer: AdamW
* Learning rate: 6e-4
* Weight decay: 0.1
* LR schedule: warmup + cosine decay
* Precision: BF16

## 3. Training Regimes

### A. Embedding

Trainable:

* `wte`: token embeddings
* `wpe`: positional embeddings
* `ln_f`: final LayerNorm

Frozen:

* All Transformer blocks
* `ln_1`
* `ln_2`
* Attention
* MLP

GPT-2 uses tied input/output embeddings, so `wte` and `lm_head` share the same parameters.

```text
wte + wpe
    ↓
12 × frozen Transformer blocks
    ↓
ln_f
    ↓
lm_head (= wte)
```

### B. Attention

Trainable:

* `wte`
* `wpe`
* `ln_1` in every block
* Attention in every block
* `ln_2` in every block
* `ln_f`

Frozen:

* MLP in every block

```text
wte + wpe
    ↓
12 × [trainable LN + Attention + trainable LN
          + frozen MLP]
    ↓
ln_f
    ↓
lm_head (= wte)
```

Once attention is made trainable, both LayerNorms in the corresponding Transformer block are also made trainable.

### C. Full

All model parameters are trainable.

```text
wte + wpe
    ↓
12 × [LN + Attention + LN + MLP]
    ↓
ln_f
    ↓
lm_head (= wte)
```

## 4. Controlled Variables

The following are kept identical across all three experiments:

1. GPT-2 architecture
2. Random initialization
3. FineWeb-Edu data
4. Data ordering
5. Sequence length
6. Batch size
7. Optimizer
8. Learning-rate schedule
9. Total training budget: 2B tokens

Therefore, the primary experimental variable is **which model components are allowed to learn**.

## 5. Main Comparison

The central progression is:

```text
Embedding
    ↓
+ Attention
    ↓
+ MLP
```

This allows us to investigate:

* How well can a randomly initialized Transformer perform when only its embeddings are trained?
* How much does learning attention improve language modeling?
* How much additional benefit comes from learning the MLP?
* What is the relative contribution of attention and MLP learning during pretraining?

## 6. Evaluation

The primary metrics are:

* Training loss
* Validation loss
* Perplexity

Metrics should be recorded against the **number of training tokens**, rather than only against optimization steps.

For example:

```text
100M → 200M → 300M → ... → 2B tokens
```

At each evaluation point, all three models are evaluated on the same held-out validation set.

The main visualization should be **validation loss vs. tokens seen**, with all three training regimes plotted on the same curve.

## 7. Additional Measurements

For each experiment, record:

* Number of trainable parameters
* Training loss
* Validation loss
* Perplexity
* Training throughput (tokens/sec)
* GPU memory usage
* Final checkpoint

## 8. Hypothesis

The experiment is exploratory, so the outcome should not be assumed in advance.

The main quantities of interest are:

1. The performance achievable by training embeddings alone.
2. The performance jump from adding trainable attention.
3. The additional improvement obtained by training the MLP.

In particular, the experiment asks whether **attention provides most of the useful computational adaptation**, or whether substantial additional capability emerges only once the MLP layers are also trained.

## 9. Interpretation

This is a controlled ablation of GPT-2 pretraining from random initialization.

Rather than asking only whether a Transformer can learn language, the experiment asks:

> **Which components of a randomly initialized Transformer need to be learned in order for useful language-modeling behavior to emerge?**

The progression from **Embedding → Attention → Full** provides a simple way to study the emergence of language-modeling capability as increasingly expressive parts of the Transformer are allowed to adapt.
