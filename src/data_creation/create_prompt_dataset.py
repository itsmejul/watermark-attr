"""Generate the prompt files in data/prompts using OpenAI Batch API.

Closed keyspace (64000 samples):
    prefix_10.json    first 10 tokens of each watermarked text T_w
    titles_1.json     titles with exactly 1 word replaced
    titles_2.json     titles with exactly 2 words replaced
    titles_3.json     titles with exactly 3 words replaced
    questions.json    [q1, q2] pairs of questions

Open keyspace (1000 held-out negatives):
    prefix_10_open.json, titles_{1,2,3}_open.json, questions_open.json

Each task is split into batches of <50,000 requests due to OpenAI rate limits, 
all batches are started simultaneously and polled until done. 
Batch IDs and intermediate outputs are cached in _batch_work/ in case the run fails.
Generated questions that are empty or too long are regenerated.

Requires OPENAI_API_KEY (environment or .env), data/seeded_dataset.jsonl,
and data/t_ws/combined_t_ws.json.

usage:
    PYTHONPATH=. python src/data_creation/create_prompt_dataset.py # both sets
    PYTHONPATH=. python src/data_creation/create_prompt_dataset.py --set open
    PYTHONPATH=. python src/data_creation/create_prompt_dataset.py --set closed
"""

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.util.filereader import (
    load_abstracts,
    load_open_keyspace_set,
    load_path_file,
    load_titles,
    write_path_file,
)

sys.stdout.reconfigure(line_buffering=True)
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

N_CLOSED = 64000
N_OPEN = 1000
MAX_BATCH_REQUESTS = 50000
MODEL = "gpt-4o-mini"
POLL_INTERVAL_S = 60
PREFIX_LEN = 10

MAX_QUESTION_LEN = 250
REPAIR_WORKERS = 50
MAX_REPAIR_ROUNDS = 3

WORK_DIR = Path(__file__).parent / "_batch_work"
BATCH_IDS_FILE = WORK_DIR / "batch_ids.json"

PERTURB_PROMPTS = {
    "titles_1": """You are given a single paper title. Replace exactly one word with a suitable synonym or similar word.
Rules:
- The replacement must be a different word (not just different capitalization or punctuation)
- You must replace exactly one word
- Return only the modified title, nothing else. No explanation.""",
    "titles_2": """You are given a single paper title. Replace exactly two words with suitable synonyms or similar words.
Rules:
- Each replacement must be a different word (not just different capitalization or punctuation)
- You must replace exactly two words
- Return only the modified title, nothing else. No explanation.""",
    "titles_3": """You are given a single paper title. Replace exactly three words with suitable synonyms or similar words.
Rules:
- Each replacement must be a different word (not just different capitalization or punctuation)
- you must replace exactly three words
- Return only the modified title, nothing else. No explanation.""",
}

QUESTIONS_TASK = "questions"
QUESTIONS_PROMPT = """You are given the abstract of a research paper.
Generate exactly TWO distinct questions whose answers are contained in this paper.

Rules for each question:
- Sound like a natural knowledge or exam question, NOT a question about the paper itself.
- Never reference "the paper", "the authors", "the study", or ask about methodology or experimental details.
- Ask about the underlying concepts and findings.

The two questions must be clearly different from each other in both content and phrasing
(do not paraphrase the same question twice).

Return a JSON object with exactly two keys, "q1" and "q2", whose values are the two
questions as plain strings. Do not include any other keys or commentary."""

PROMPT_FOR = {**PERTURB_PROMPTS, QUESTIONS_TASK: QUESTIONS_PROMPT}
TITLE_TASKS = ("titles_1", "titles_2", "titles_3")


def _load_batch_ids():
    if BATCH_IDS_FILE.is_file():
        return json.loads(BATCH_IDS_FILE.read_text())
    return {}


def _save_batch_ids(batch_ids):
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    BATCH_IDS_FILE.write_text(json.dumps(batch_ids, indent=2))


def split_chunks(items):
    """Split into the fewest balanced chunks of at most MAX_BATCH_REQUESTS."""
    k = max(1, math.ceil(len(items) / MAX_BATCH_REQUESTS))
    size = math.ceil(len(items) / k)
    return [items[i: i + size] for i in range(0, len(items), size)]


def submit_batch(key, task, inputs, json_mode):
    """Write the request file for one chunk and submit it."""
    body_extra = {"response_format": {"type": "json_object"}} if json_mode else {}
    jsonl_path = WORK_DIR / f"batch_input_{key}.jsonl"
    with open(jsonl_path, "w") as f:
        for i, text in enumerate(inputs):
            line = {
                "custom_id": f"{key}_{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": PROMPT_FOR[task]},
                        {"role": "user", "content": text},
                    ],
                    **body_extra,
                },
            }
            f.write(json.dumps(line) + "\n")

    batch_file = client.files.create(file=open(jsonl_path, "rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"[{key}] submitted: {batch.id}")
    return batch.id


def poll_until_done(batch_ids):
    """Poll every batch until all are finished. Returns dict: key -> batch."""
    pending = dict(batch_ids)
    done = {}
    while pending:
        for key, batch_id in list(pending.items()):
            batch = client.batches.retrieve(batch_id)
            c = batch.request_counts
            print(f"[{key}] {batch.status} — {c.completed}/{c.total} completed, {c.failed} failed")
            if batch.status in ("completed", "failed", "expired", "cancelled"):
                done[key] = batch
                del pending[key]
        if pending:
            time.sleep(POLL_INTERVAL_S)
    return done


def collect_raw(batch, expected_len):
    """Collect a completed batch and return raw contents ordered by index."""
    if batch.status != "completed":
        if batch.error_file_id:
            print(client.files.content(batch.error_file_id).text[:1000])
        raise RuntimeError(f"batch ended with status: {batch.status}")
    if batch.error_file_id:
        print("partial errors:")
        print(client.files.content(batch.error_file_id).text[:500])

    result_text = client.files.content(batch.output_file_id).text
    by_idx = {}
    for line in result_text.strip().split("\n"):
        r = json.loads(line)
        idx = int(r["custom_id"].split("_")[-1])
        by_idx[idx] = r["response"]["body"]["choices"][0]["message"]["content"].strip()

    assert len(by_idx) == expected_len, f"result count mismatch: got {len(by_idx)}, expected {expected_len}"
    return [by_idx[i] for i in range(expected_len)]


def parse_two_questions(content):
    """Parse a JSON response into [q1, q2]."""
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        return [None, None]
    q1 = obj.get("q1") or obj.get("question1") or obj.get("Q1")
    q2 = obj.get("q2") or obj.get("question2") or obj.get("Q2")
    return [
        q1.strip() if isinstance(q1, str) else None,
        q2.strip() if isinstance(q2, str) else None,
    ]


def question_is_valid(entry):
    """A valid entry is [q1, q2] with two non-empty strings under the max length."""
    if not isinstance(entry, list) or len(entry) != 2:
        return False
    for q in entry:
        if not isinstance(q, str) or q.strip() == "" or len(q) > MAX_QUESTION_LEN:
            return False
    return True


def regenerate_one(abstract):
    """Single call to regenerate a [q1, q2] pair for one abstract."""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": QUESTIONS_PROMPT},
            {"role": "user", "content": abstract},
        ],
        response_format={"type": "json_object"},
    )
    return parse_two_questions(resp.choices[0].message.content.strip())


def repair_questions(questions, abstracts):
    """Regenerate any invalid entry, repeating until max num of retries is reached."""
    for round_num in range(1, MAX_REPAIR_ROUNDS + 1):
        bad = [i for i, e in enumerate(questions) if not question_is_valid(e)]
        if not bad:
            print("all question entries valid")
            return questions
        print(f"repair round {round_num}: regenerating {len(bad)} invalid entries")
        with ThreadPoolExecutor(max_workers=REPAIR_WORKERS) as pool:
            futures = {pool.submit(regenerate_one, abstracts[i]): i for i in bad}
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    questions[i] = fut.result()
                except Exception as e:
                    print(f"  idx={i} failed: {e}")
    leftover = sum(1 for e in questions if not question_is_valid(e))
    print(f"repair finished with {leftover} still-invalid entries (left as-is)")
    return questions


def run_task(task, suffix, inputs, json_mode, batch_ids):
    """Submit (if needed), poll, collect and combine all chunks for one task."""
    chunks = split_chunks(inputs)
    keys = [f"{task}{suffix}_chunk{c}" for c in range(len(chunks))]

    for key, chunk in zip(keys, chunks):
        if key in batch_ids or (WORK_DIR / f"{key}.json").is_file():
            continue
        batch_ids[key] = submit_batch(key, task, chunk, json_mode)
        _save_batch_ids(batch_ids)

    to_poll = {
        key: batch_ids[key]
        for key in keys
        if key in batch_ids and not (WORK_DIR / f"{key}.json").is_file()
    }
    completed = poll_until_done(to_poll) if to_poll else {}

    combined = []
    for key, chunk in zip(keys, chunks):
        cached = WORK_DIR / f"{key}.json"
        if cached.is_file():
            part = json.loads(cached.read_text())
        else:
            raw = collect_raw(completed[key], expected_len=len(chunk))
            part = [parse_two_questions(c) for c in raw] if json_mode else raw
            cached.write_text(json.dumps(part, ensure_ascii=False))
        combined += part
    return combined


def build_prefix(texts):
    """First PREFIX_LEN tokens of each text as a string, using
    the watermark model's tokenizer."""
    from transformers import AutoTokenizer

    wm_config = load_path_file(["data"], "watermark_config.json")
    tokenizer = AutoTokenizer.from_pretrained(wm_config["watermark_model"])
    tokenizer.pad_token = tokenizer.eos_token
    return [
        tokenizer.decode(
            tokenizer.encode(text, max_length=PREFIX_LEN, truncation=True),
            skip_special_tokens=True,
        )
        for text in texts
    ]


def generate_set(suffix, titles, abstracts, batch_ids):
    """Run all four LLM tasks for one set and write the output files."""
    for task in TITLE_TASKS:
        combined = run_task(task, suffix, titles, json_mode=False, batch_ids=batch_ids)
        write_path_file(["data", "prompts"], f"{task}{suffix}.json", combined)
        print(f"wrote data/prompts/{task}{suffix}.json ({len(combined)} entries)")

    questions = run_task(QUESTIONS_TASK, suffix, abstracts, json_mode=True, batch_ids=batch_ids)
    questions = repair_questions(questions, abstracts)
    write_path_file(["data", "prompts"], f"{QUESTIONS_TASK}{suffix}.json", questions)
    print(f"wrote data/prompts/{QUESTIONS_TASK}{suffix}.json ({len(questions)} entries)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", choices=["closed", "open", "both"], default="both")
    args = parser.parse_args()

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    batch_ids = _load_batch_ids()

    if args.set in ("closed", "both"):
        print("=== closed set (64000 samples) ===")
        titles = load_titles(N_CLOSED)
        abstracts = load_abstracts(N_CLOSED)

        t_ws = load_path_file(["data", "t_ws"], "combined_t_ws.json")
        prefix = build_prefix(t_ws)
        write_path_file(["data", "prompts"], "prefix_10.json", prefix)
        print(f"wrote data/prompts/prefix_10.json ({len(prefix)} entries)")

        generate_set("", titles, abstracts, batch_ids)

    if args.set in ("open", "both"):
        print("=== open keyspace (1000 held-out samples) ===")
        open_set = load_open_keyspace_set(n_samples=N_OPEN)
        titles, abstracts = open_set["titles"], open_set["abstracts"]

        prefix = build_prefix(abstracts)
        write_path_file(["data", "prompts"], "prefix_10_open.json", prefix)
        print(f"wrote data/prompts/prefix_10_open.json ({len(prefix)} entries)")

        generate_set("_open", titles, abstracts, batch_ids)


if __name__ == "__main__":
    main()
