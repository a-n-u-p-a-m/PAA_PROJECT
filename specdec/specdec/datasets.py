"""The three benchmarks of Dutta et al. (EACL 2026), at their exact sizes.

    HUMAN-EVAL   all 164 prompts
    GSM-8K       256 randomly sampled prompts
    CNN-DM       256 randomly sampled prompts

Dataset choice is not cosmetic here. Dutta et al.'s central observation is
that energy savings depend on the dataset as much as on the strategy: code
completion is low-entropy, so the draft model tracks the target and
acceptance is high, while summarisation is not. Since rejected drafts are
pure wasted energy, acceptance rate is the bridge between their model-level
result and our component-level one -- and it is only comparable if the
prompts are the same.

Their sampling seed is not published, so ours is fixed at 42 and recorded in
every result file. Absolute numbers will differ slightly from the paper's;
the trends across datasets are what we compare.

Needs network on first use (datasets are cached afterwards). Without it,
``--offline`` falls back to a small bundled sample so the pipeline can still
be exercised -- those runs are labelled and must not be reported.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

DatasetName = Literal["humaneval", "gsm8k", "cnndm"]

PAPER_SIZES: dict[str, int] = {"humaneval": 164, "gsm8k": 256, "cnndm": 256}
PAPER_SEED = 42

#: How each dataset's raw record becomes a prompt. CNN-DM needs an
#: instruction wrapper because the raw record is a bare news article.
CNNDM_INSTRUCTION = (
    "Summarize the following news article in three sentences.\n\n{article}\n\nSummary:"
)


@dataclass(frozen=True)
class Sample:
    dataset: str
    index: int
    prompt: str
    #: True when this came from the bundled offline fallback, not the real set
    synthetic: bool = False


# -- offline fallback ------------------------------------------------------

_FALLBACK: dict[str, list[str]] = {
    "humaneval": [
        'def has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    """\n',
        'def separate_paren_groups(paren_string: str) -> List[str]:\n    """ Input to this function is a string containing multiple groups of nested parentheses.\n    Your goal is to separate those groups into separate strings and return the list of those.\n    """\n',
        'def truncate_number(number: float) -> float:\n    """ Given a positive floating point number, it can be decomposed into\n    an integer part and decimals. Return the decimal part of the number.\n    """\n',
        'def below_zero(operations: List[int]) -> bool:\n    """ Detect if at any point the balance of an account falls below zero.\n    """\n',
    ],
    "gsm8k": [
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need?",
        "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
    ],
    "cnndm": [
        CNNDM_INSTRUCTION.format(
            article=(
                "The city council voted on Tuesday to approve funding for a new light rail "
                "line connecting the northern suburbs to the central business district. "
                "The project, estimated at 2.1 billion dollars, has been under discussion "
                "for nearly a decade. Construction is expected to begin next spring and "
                "finish in 2031. Opponents argued the money would be better spent on bus "
                "services, which reach more neighbourhoods at a fraction of the cost."
            )
        ),
        CNNDM_INSTRUCTION.format(
            article=(
                "Researchers reported that a long-running study of coastal erosion has "
                "found the shoreline retreating faster than models predicted. Measurements "
                "collected over eighteen years show an average loss of 1.4 metres a year, "
                "roughly double the rate assumed in regional planning documents. The team "
                "cautioned that the figure varies widely by location."
            )
        ),
    ],
}


def _load_offline(name: str, limit: int | None) -> list[Sample]:
    pool = _FALLBACK[name]
    n = min(limit or len(pool), len(pool))
    return [Sample(name, i, pool[i], synthetic=True) for i in range(n)]


# -- real loaders ----------------------------------------------------------


#: Hub ids to try, in order. The namespaced form is canonical; the bare name
#: is the legacy alias. This matters from ``datasets`` 3.0, which removed
#: script-based loaders -- ``cnn_dailymail`` was one, and the bare name can
#: fail with "Dataset scripts are no longer supported" on a fresh install.
HUB_IDS: dict[str, tuple[tuple[str, ...], str | None]] = {
    "humaneval": (("openai/openai_humaneval", "openai_humaneval"), None),
    "gsm8k": (("openai/gsm8k", "gsm8k"), "main"),
    "cnndm": (("abisee/cnn_dailymail", "cnn_dailymail"), "3.0.0"),
}


def _hub_load(name: str, split: str = "test"):
    """Try each known id for ``name`` and return the first that loads."""
    from datasets import load_dataset

    ids, config = HUB_IDS[name]
    errors = []
    for hub_id in ids:
        try:
            return (load_dataset(hub_id, config, split=split) if config
                    else load_dataset(hub_id, split=split))
        except Exception as exc:
            errors.append(f"{hub_id}: {type(exc).__name__}: {exc}")
    raise RuntimeError("none of the known hub ids loaded:\n  " + "\n  ".join(errors))


def _sampled_indices(n: int, limit: int | None, seed: int) -> list[int]:
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    return idx[: limit or n]


def _load_humaneval(limit: int | None) -> list[Sample]:
    ds = _hub_load("humaneval")
    rows = list(ds)[: limit or len(ds)]
    return [Sample("humaneval", i, r["prompt"]) for i, r in enumerate(rows)]


def _load_gsm8k(limit: int | None, seed: int) -> list[Sample]:
    ds = _hub_load("gsm8k")
    return [Sample("gsm8k", i, ds[i]["question"])
            for i in _sampled_indices(len(ds), limit, seed)]


def _load_cnndm(limit: int | None, seed: int) -> list[Sample]:
    ds = _hub_load("cnndm")
    return [Sample("cnndm", i, CNNDM_INSTRUCTION.format(article=ds[i]["article"]))
            for i in _sampled_indices(len(ds), limit, seed)]


def load(
    name: DatasetName,
    limit: int | None = None,
    seed: int = PAPER_SEED,
    offline: bool = False,
) -> list[Sample]:
    """Load one benchmark. ``limit=None`` uses the paper's sample count."""
    if name not in PAPER_SIZES:
        raise KeyError(f"unknown dataset {name!r}; have {list(PAPER_SIZES)}")
    if limit is None:
        limit = PAPER_SIZES[name]
    if offline:
        return _load_offline(name, limit)

    try:
        if name == "humaneval":
            return _load_humaneval(limit)
        if name == "gsm8k":
            return _load_gsm8k(limit, seed)
        return _load_cnndm(limit, seed)
    except Exception as exc:
        raise RuntimeError(
            f"could not load {name!r}: {exc}\n"
            "Needs network access the first time (`pip install datasets`). "
            "Pass --offline to run against the small bundled sample instead -- "
            "useful for debugging, not for any number that goes in the report."
        ) from exc


def load_all(
    names: list[str] | None = None,
    limit: int | None = None,
    seed: int = PAPER_SEED,
    offline: bool = False,
) -> list[Sample]:
    out: list[Sample] = []
    for name in names or list(PAPER_SIZES):
        out.extend(load(name, limit=limit, seed=seed, offline=offline))  # type: ignore[arg-type]
    return out
