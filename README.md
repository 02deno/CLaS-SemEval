# CLaS-SemEval

Course project for SemEval 2027 Task 7 (CLaS). A question comes in one language. A frozen language model has to answer it in another language. The intervention is inference-time activation steering. A prompt that says "respond in Turkish" is the organizers' comparison baseline, and it is not the method this code implements.

## Read first

- Task notes, the inference-time explanation, and paper summaries: [docs/literature_review.md](docs/literature_review.md)
- Organizers' code ships in this repo at commit `65a43da900bb18a1d078f9c39b19b0a056cf944e`: [third_party/CLaS-SemEval-2027](third_party/CLaS-SemEval-2027)

## Setup

```bash
pip install -r requirements.txt
python third_party/CLaS-SemEval-2027/download_data.py --out data
```

`download_data.py` uses the Hugging Face `datasets` library from the official requirements file, and it needs a network connection. It writes `data/validation/<lang>.txt` (CLaS-Bench questions) and `data/train/<lang>.txt` (FLORES lines). Steering itself uses PyTorch and Transformers. Llama-3.1-8B needs a GPU. The dry run does not.

## Dry run

```bash
python src/clas_steer/run_slice.py --dry-run
```

Prints the default slice (`en→de`, `en→tr`, `ar→en`, `en→sw`, `en→mt`, `ru→uk`), the track of each pair, and how many question and FLORES lines are on disk. It does not download or load a model.

## DiffMean generation

```bash
python src/clas_steer/run_slice.py --layers 28 --alpha 1 --num-questions 2 --num-sentences 128
```

Layer indices are 0-based. Llama-3.1-8B-Instruct has 32 layers, numbered 0 through 31.

The run estimates a mean residual at the last real token of each FLORES sentence, builds a unit vector `mean(target) - mean(source)`, and adds `alpha` times that vector inside the chosen decoder layers during greedy chat generation. Outputs land in:

```text
generations/<model>/<track>/diffmean/<src>_to_<tgt>.json
```

Track 3 uses `generations/<model>/t3/<target>/diffmean/`, matching the official directory layout.

`--contrast pair` is the CLaS-Bench vector (target minus source). `--contrast others` subtracts the mean of the other languages in the run, so one target vector can be reused across sources. `--layers 8,12,16,20,24,28` and a few values of `--alpha` are the layer and strength sweep from the literature notes.

The JSON fields are the ones `evaluate.py` reads at the pinned commit: `source_language`, `target_language`, and each result's `output` and `original_prompt`. The user turn is the raw question.

## Scoring

Run the official script from its own checkout so `import clas` resolves:

```bash
cd third_party/CLaS-SemEval-2027
python evaluate.py --gen-dir ../../generations/Llama-3.1-8B-Instruct/t1/diffmean
```

`--skip-judge` reports language-forcing success only. Output relevance loads the judge model through vLLM, which is listed in the official `requirements.txt` and is separate from this steering environment.

## Checks

```bash
python -m unittest tests.test_scaffold
```

This checks the default slice, the JSON fields the official scorer reads, the pinned commit, the dry run, and the DiffMean plus hook arithmetic when PyTorch is installed.
