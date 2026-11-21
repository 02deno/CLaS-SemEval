"""Greedy chat generation in the official CLaS JSON shape.

Pinned to ``third_party/CLaS-SemEval-2027`` commit
``65a43da900bb18a1d078f9c39b19b0a056cf944e``.

``evaluate.py`` globs ``*_to_*.json`` and reads:

- ``source_language``, ``target_language`` (ISO 639-1 codes)
- ``results[i]["output"]`` for language identification
- ``results[i]["original_prompt"]`` as the question shown to the judge

``run_baseline.py`` also stores ``model``, ``method``, ``track``,
``results[i]["question_idx"]``, and ``results[i]["input"]``. This module
writes those too. For steering, ``input`` is the raw question: the user
turn does not contain a "respond in X" instruction. The judge ignores
``input`` and scores ``original_prompt`` against ``output``.

Generation matches the shared-task rule: chat template, generation prompt,
``do_sample=False``, and ``max_new_tokens=1024`` unless the caller overrides it.
PyTorch is imported only inside the functions that load or run a model, so
building a record does not require a GPU stack.
"""

from __future__ import annotations

from clas_steer.official import EVALUATOR_RESULT_KEYS, EVALUATOR_TOP_KEYS


def build_record(
    source: str,
    target: str,
    model_name: str,
    method: str,
    track: str,
    questions: list[str],
    outputs: list[str],
) -> dict:
    """One language-pair file. ``input`` stays the raw question on purpose."""
    if len(questions) != len(outputs):
        raise ValueError(
            f"{source}->{target}: {len(questions)} questions and {len(outputs)} outputs"
        )
    record = {
        "source_language": source,
        "target_language": target,
        "model": model_name,
        "method": method,
        "track": track,
        "results": [
            {
                "question_idx": index,
                "input": question,
                "output": output,
                "original_prompt": question,
            }
            for index, (question, output) in enumerate(zip(questions, outputs))
        ],
    }
    missing_top = [key for key in EVALUATOR_TOP_KEYS if key not in record]
    missing_row = []
    if record["results"]:
        missing_row = [
            key for key in EVALUATOR_RESULT_KEYS if key not in record["results"][0]
        ]
    if missing_top or missing_row:
        raise RuntimeError(f"record is missing {missing_top + missing_row}")
    return record


def _encode_questions(tokenizer, questions: list[str]):
    """Tokenize a batch of user turns.

    The chat template already includes special tokens, so the string fallback
    encodes with ``add_special_tokens=False`` and does not add a second BOS.
    """
    messages = [[{"role": "user", "content": question}] for question in questions]
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
            return_dict=True,
        )
        if "input_ids" not in encoded:
            raise TypeError("chat template did not return input_ids")
    except (TypeError, ValueError, KeyError):
        rendered = [render_user_turn(tokenizer, question) for question in questions]
        encoded = tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
    return {
        key: value
        for key, value in encoded.items()
        if key in {"input_ids", "attention_mask"}
    }


def render_user_turn(tokenizer, question: str) -> str:
    """Chat template for a single user question, with the generation prompt."""
    if not getattr(tokenizer, "chat_template", None):
        name = getattr(tokenizer, "name_or_path", "tokenizer")
        raise RuntimeError(f"{name} has no chat template")
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def load_causal_lm(model_name: str, trust_remote_code: bool = False):
    """Load a frozen causal LM. bfloat16 on CUDA, float32 on CPU."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    common = {
        "trust_remote_code": trust_remote_code,
    }
    if torch.cuda.is_available():
        common["device_map"] = "auto"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, **common
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, **common
        )
    if not torch.cuda.is_available():
        model.to("cpu")
    model.eval()
    return model, tokenizer


def generate_answers(
    model,
    tokenizer,
    questions: list[str],
    steering,
    max_new_tokens: int = 1024,
    batch_size: int = 4,
) -> list[str]:
    """Greedy decoding. ``steering`` is a context manager, usually ``SteeringHook``."""
    import torch

    if not questions:
        return []
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = model.get_input_embeddings().weight.device
    previous_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    outputs: list[str] = []
    was_training = model.training
    model.eval()
    try:
        for start in range(0, len(questions), batch_size):
            batch = questions[start : start + batch_size]
            encoded = _encode_questions(tokenizer, batch)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            prompt_width = encoded["input_ids"].shape[1]
            with steering, torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                )
            for row in generated:
                new_tokens = row[prompt_width:]
                text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                outputs.append(text.strip())
    finally:
        tokenizer.padding_side = previous_side
        model.train(was_training)
    return outputs
