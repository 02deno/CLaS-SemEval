# CLaS literature notes

Notes for the course literature review on SemEval 2027 Task 7, CLaS: Cross-Lingual Alignment and Steering of Large Language Models. The task page is [clas-semeval2027.github.io](https://clas-semeval2027.github.io/clas.semeval2027/task/). The official code is [d-gurgurov/CLaS-SemEval-2027](https://github.com/d-gurgurov/CLaS-SemEval-2027), cloned in this project under `third_party/CLaS-SemEval-2027`. PDFs live in `Articles/`.

## 1. The task

A question arrives in a source language. The model must answer that same question, coherently, in a specified target language. The two models that count for the official ranking are `meta-llama/Llama-3.1-8B-Instruct` and `CohereLabs/tiny-aya-global`.

The method has to be an inference-time intervention on the model's internal computations. Appending "Respond in Turkish" (or the same instruction in any language) is the organizers' prompting baseline. It is useful for comparison and is not eligible for the ranking.

Tracks:

- T1, high-resource steering. Sources and targets are `ar`, `de`, `en`, `es`, `fr`, `ja`, `zh`, `ru`, `hi`, `ko`. Same-language pairs are excluded, so there are 90 pairs. One ranking per model.
- T2, low-resource steering. The same 10 sources into `cs`, `el`, `fa`, `vi`, `uk`, `mt`, `no`, `sw`, `th`, `tr`. 100 pairs. Turkish is a target. One ranking per model.
- T3, full multi-language steering. All 20 languages into one of `ja`, `ru`, or `de` (19 sources each). Six rankings: model times target language.
- Bonus, multilingual refusal. Refuse the question, and write the refusal in the target language. The 10 high-resource languages, including same-language pairs (100 pairs). The scoring script is `evaluate_refusal.py`. The written protocol may still change.

Validation data is CLaS-Bench: 70 parallel open-ended questions per language. FLORES-200 dev and devtest lines are the training text for finding language-related components. The hidden test set is 100 new questions per language. Organizers run submitted code on A100 and H100 GPUs. Generation should be deterministic: greedy decoding, `max_tokens=1024`, chat template with the generation prompt.

Registration deadline is 31 December 2026. Evaluation runs 10–31 January 2027. Each team gets two test submissions per track and one final submission per track.

## 2. Inference time

A transformer writes an answer one token at a time. Two clocks matter:

- Training time is when the weights change: pretraining, instruction tuning, or fine-tuning.
- Inference time is when the weights are already frozen and the model is only answering.

Inside each layer the running hidden state of a token is the residual stream, a vector of a few thousand numbers. Activation steering leaves the weights alone and adds a fixed vector to that state while the forward pass is running:

```
h_layer = h_layer + alpha * v_language / ||v_language||
```

`v_language` is computed earlier, from text whose language we know (FLORES sentences). `alpha` is a strength. Too small, and the model keeps the source language. Too large, and the text may look like the target language while no longer answering the question.

Prompting is also inference, but it only changes the input string. This task asks for the edit inside the network, because the scientific question is whether language is a direction in the residual stream.

## 3. How the vector is built

CLaS-Bench's strongest method is DiffMean on the residual stream (Gurgurov et al., 2026, following Marks and Tegmark, 2023, and Rimsky et al., 2024). Average the hidden state of the target language and of a contrast language at one layer, then subtract:

```
v_layer = mean(h | target) - mean(h | contrast)
```

At answer time the vector is L2-normalized and added with strength `alpha`. In the paper the contrast is the source language. A source-independent alternative, which this project can test, replaces the source mean with the mean of the other languages. That second vector can be reused for every source of a given target.

The same paper also tried:

- Prompting in English, and prompting with the instruction written in the target language.
- LAPE neurons (Tang et al., 2024): neurons with low cross-lingual entropy, then add or replace their activation with the target-language average.
- A linear probe trained to separate the target language from other languages; the weight vector is the steering direction.
- PCA and LDA directions in the residual stream.
- DiffMean inside a sparse autoencoder's latent space, then decode back.

On Llama-3.1-8B-Instruct, mean language steering score across 32 languages was about 84.5 for DiffMean, 80.1 for LAPE neurons, 67.7 and 67.3 for the two prompting baselines, 48.6 for probes, 42.3 for SAE DiffMean, 23.6 for LDA, and 15.1 for PCA (Table 2 of the paper). DiffMean is the result to beat and the first method this codebase implements.

Two findings shape the experiments:

- Language-specific structure shows up mainly in later layers (roughly 16–32 on the 8B Llama), and steering directions cluster by language family.
- Steering is uneven. In that table, Maltese is about 15.6 and Swahili about 49.8, while Turkish is about 84.5 and German about 95.4. A course result is more informative on the weak languages than on another easy high-resource pair.

There is a real disagreement about depth. CLaS-Bench finds language geometry in later layers. Kang et al. (BlackboxNLP 2026) find language steering more reliable in earlier layers, and Cegin et al. (EMNLP 2026) find early-layer language steering helpful for synthetic data. A layer sweep is an experiment, not a settled setting.

## 4. Scores

The official evaluator (`third_party/CLaS-SemEval-2027/evaluate.py`) reports three numbers.

- LFS, language forcing success. Fraction of outputs whose language ID is the target language, in `[0, 1]`. The identifier is fastText (`facebook/fasttext-language-identification`). If confidence is below 0.9, Chinese, Japanese, and Korean can still be accepted from the script. Other low-confidence outputs count as a miss.
- OR, output relevance. A judge model (`Qwen/Qwen3-8B` in the script) scores the answer 0, 1, or 2 for whether it addresses the question. The judge is told to ignore the language. The mean score is divided by 2 so OR is in `[0, 1]`.
- LSS, language steering score. The harmonic mean:

```
LSS = 2 * LFS * OR / (LFS + OR)
```

If either term is 0, LSS is 0. Fluency in the wrong language and gibberish in the right language both lose.

The bonus script (`evaluate_refusal.py`) reports a refusal rate from an LLM judge, language-forcing success, and their harmonic mean.

Our generator writes the JSON shape that `evaluate.py` already reads. See `src/clas_steer/generate.py`.

## 5. Papers in this folder

Read them in this order.

### Task definition

Gurgurov, Al Ghussin, Bäumel, Chou, Schramowski, Mosbach, van Genabith, and Ostermann. 2026. CLaS-Bench: A Cross-Lingual Alignment and Steering Benchmark. Findings of ACL 2026. `Articles/2601.08331v1.pdf`. Anthology: [2026.findings-acl.1086](https://aclanthology.org/2026.findings-acl.1086/).

Introduces 70 parallel questions in 32 languages and the LFS / OR / LSS protocol. The shared task uses a subset of those languages, splits them into T1–T3, and swaps Aya-Expanse-8B for `tiny-aya-global`. Everything else in this review is a method paper sitting on top of this benchmark.

### The steering operation

Rimsky, Gabrieli, Schulz, Tong, Hubinger, and Turner. 2024. Steering Llama 2 via Contrastive Activation Addition. ACL 2024. `Articles/2024.acl-long.828.pdf`.

Contrastive Activation Addition (CAA). For a behavior, run contrast pairs (behavior present versus absent), read the residual stream at one layer, average the difference, and add that vector at every position after the prompt. Demonstrated on Llama 2 Chat for behaviors such as sycophancy, refusal, and hallucination, with a multiplier that sets the strength. A language vector is the same construction: positive examples are sentences in the target language, negative examples are sentences in the contrast language.

Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee, and Nanda. 2024. Refusal in Language Models Is Mediated by a Single Direction. NeurIPS 2024. `Articles/NeurIPS-2024-refusal-in-language-models-is-mediated-by-a-single-direction-Paper-Conference.pdf`.

Difference-in-means between harmful and harmless prompts yields one direction. Adding it makes the model refuse harmless requests. Ablating it makes the model comply with harmful ones, across 13 chat models up to 72B. This is the bonus-track operator: a refusal direction, composed with a language direction so the refusal is written in the target language. The paper is also a warning that the direction is brittle. An adversarial suffix can suppress it.

Stolfo, Balachandran, Yousefi, Horvitz, and Nushi. 2025. Improving Instruction-Following in Language Models through Activation Steering. ICLR 2025. `Articles/ICLR-2025-improving-instruction-following-in-language-models-through-activation-steering-Paper-Conference.pdf`.

The contrast is "instruction present" versus "instruction absent" for constraints such as format, length, and required words. Several instruction vectors can be added in one forward pass. Vectors taken from an instruction-tuned model transfer to the base model. Relevant here because the shared task forbids putting the language instruction in the prompt: the vector stands in for that instruction.

### Language as a direction

Maraia, Ranaldi, Valentino, and Zanzotto. 2026. Can Activation Steering Generalize Across Languages? A Study on Syllogistic Reasoning in Language Models. EACL 2026. `Articles/2026.eacl-long.125.pdf`.

An English steering vector for formal syllogistic reasoning still helps in other languages (they report gains up to about 36 points) and barely hurts language-model loss. The behavior is reasoning, not output language. The useful claim for us is narrower: a direction estimated in one language can fire on inputs in another. A Turkish vector should be tested on non-Turkish questions, which is exactly T2 and T3.

Mahmoud, Semage, Karimpanal, and Rana. 2026. Improving Multilingual Language Models by Aligning Representations through Steering. LREC 2026. `Articles/2026.lrec-1.164.pdf`.

Learns one vector, added at a single layer, to realign non-English hidden states. They compare prompting, supervised fine-tuning, in-context learning, projection mapping, and translation pipelines, and report that the vector matches production translation systems at much lower cost. Their objective is task accuracy after alignment, not "the whole answer must be in language X." Treat it as a cousin method: one cheap vector can replace a heavier adaptation, which is the practical motivation of the shared task.

Kirtane and Huang. 2026. Language Steering for Multilingual In-Context Learning. `Articles/2602.02326v3.pdf`.

Language vectors are mean activation differences between parallel source and target sentences, added as an offset during inference. English few-shot demonstrations plus a target-language query improve when the vector is applied, with no weight update, across 19 languages and three models. Related languages transfer more of the gain than distant ones, and the vectors cluster by family. This is the reason to try a Russian donor vector for Ukrainian, a German donor for Norwegian, and an Arabic donor for Persian.

Cegin, Gurgurov, Al Ghussin, and Ostermann. 2026. Want Better Synthetic Data? Steer It: Activation Steering for Low-Resource Language Generation. EMNLP 2026. `Articles/2026.emnlp-main.1290.pdf`.

Two vectors for data generation: a language-identity vector, and a quality vector that contrasts human-written text with backtranslation. Early-layer steering improved diversity and often the downstream classifier, especially for low-resource languages, in both zero-shot and few-shot prompts. The early-layer result is the one that conflicts with CLaS-Bench's later-layer analysis.

Kang, Gurgurov, Bäumel, van Genabith, and Ostermann. 2026. Compositional Multilingual and Behavioral Attribute Steering. BlackboxNLP 2026. `Articles/2026.blackboxnlp-1.38.pdf`.

Language, jailbreak, and conciseness vectors can be added in one forward pass if each is injected at its own best layer. Single-attribute steering works only inside the right layer and strength. Language preferred earlier layers; jailbreak and conciseness preferred middle layers. The vectors are approximately orthogonal, which is why addition does not always destroy one of the attributes. This is the bonus-track sketch: Arditi's refusal direction at a middle layer, the language vector at its own layer.

Al Ghussin, Gurgurov, Bäumel, van Genabith, Schramowski, and Ostermann. 2026. Multilingual Steering by Design: Multilingual Sparse Autoencoders and Principled Layer Selection. `Articles/2605.23036v1.pdf`.

English-only sparse autoencoders steer language unreliably. SAEs trained on multilingual data give cleaner cross-lingual features. They pick the layer where multilingual alignment and language separability overlap, instead of sweeping every layer. Evaluated on Llama-3.1-8B and Gemma-2-9B for translation and cross-lingual summarization. Strong, and heavy: it needs a trained SAE. Leave it until DiffMean runs.

Michaelov, Amo Alonso, Chang, and Levy. 2026. Target-Language Generation in Multilingual Models: Activation Steering and Optimal Control. `Articles/2609.16967v1.pdf`.

Splits target-language quality into language adherence, linguistic coherence, and semantic coherence, then combines them. Casts steering as optimal control (LISECO, following Cheng et al., 2024) and compares it with DiffMean. They report performance at least as good as DiffMean on most models they tried, with less searching over `alpha`. A reasonable second method if a hand-chosen strength is unstable.

Nelwan and Wicaksono. 2026. Deployable Per-Instance Multi-Layer Activation Steering for Large Language Models. `Articles/2608.08829v1.pdf`.

The best layer set changes from input to input. Steering too many layers collapses fluency. A greedy rule that ranks layers by marginal effect recovers most of an oracle, but it needs the gold label, so they train a prompt-only ranker and a gate that stops adding layers. Too much machinery for the first milestone. Cite it as the reason to sweep layers by language family rather than freeze one global layer.

## 6. Papers that are not in the folder

CLaS-Bench uses these as the actual operators. They are the background citations, not extra PDFs in `Articles/`.

- Tang et al. 2024. Language-specific neurons: the LAPE method (activation probability, entropy filter, then add or replace).
- Marks and Tegmark. 2023. The geometry of truth: difference-in-means as a way to read a concept direction out of the residual stream.
- Turner et al. 2023. Activation addition. Zou et al. 2023. Representation engineering. The broader framing of inference-time control.
- Park, Lee, et al. The linear representation hypothesis: many concepts behave like directions, which is why adding one vector can move a behavior.
- Wendler et al. 2024 and Zhao et al. 2024. Multilingual models often map non-English text into an English-like space in the middle layers, then map back out. That picture is why middle versus late layers is an empirical question rather than a default.

## 7. What to try, in order

Each step uses the same code path in `src/clas_steer/`.

1. Reproduce DiffMean on a slice: `en?de`, `en?tr`, `ar?en`, `en?sw`, `en?mt`, `ru?uk`. Compare LFS, OR, and LSS with the official prompting baseline. This checks the hook before any new claim.
2. Sweep layers `{8, 12, 16, 20, 24, 28}` and strengths `{1, 2, 4, 8}` on that slice. The papers disagree about depth; the sweep is the result.
3. Compare a pair-specific vector, `mean(target) - mean(source)`, with a target vector, `mean(target) - mean(other languages)`. The second one is what a full T1/T2/T3 run would actually ship.
4. Family transfer on weak targets. Donors suggested by the clustering papers: Ukrainian from Russian, Norwegian from German, Persian from Arabic. Also measure cosine similarity between the vectors.
5. Bonus track, after the first three work. Add a refusal direction (Arditi et al.) at a middle layer and the language vector at the language layer (Kang et al.). Score with `evaluate_refusal.py`.

Not in the first code drop: training a new SAE, PCA, LDA, and a per-instance layer ranker. The benchmark already shows PCA and LDA far behind DiffMean, and the SAE and ranker are separate projects.

## 8. How this repo maps onto the papers

| Piece | Paper idea | Code |
| --- | --- | --- |
| Mean residual at the last real token | CAA / DiffMean activation collection | `clas_steer.collect.collect_residual_means` |
| `mean(target) - mean(contrast)`, then L2-normalize | CLaS-Bench DiffMean | `clas_steer.vectors.diffmean` |
| `h + alpha * v` on chosen decoder layers | Inference-time residual intervention | `clas_steer.hooks.SteeringHook` |
| Greedy chat generation, official JSON | Shared-task submission format | `clas_steer.generate` |
| Slice `en-de`, `en-tr`, `ar-en`, `en-sw`, `en-mt`, `ru-uk` | First reproduction experiment | `python src/clas_steer/run_slice.py --dry-run` |

`--contrast pair` is the paper's source-versus-target vector. `--contrast others` is experiment 3. Scoring stays in the official `evaluate.py`; this package does not reimplement language ID or the judge.
