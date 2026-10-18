#!/usr/bin/env python3
"""Replay one prompt through the draft-verify loop and show every token.

The sweep records how many tokens each round proposed and accepted, not
which ones, so it cannot be read back as text. This runs a single prompt
with token capture on and renders the result three ways: a coloured
terminal trace, a standalone HTML page, and JSON.

    python trace_spec.py --config qwen2.5-7b --quant float16 --k 4
    python trace_spec.py --sample-index 2 --k 5 --max-new-tokens 64
    python trace_spec.py --prompt "What is 17 * 24?" --k 4 --no-compare

Every drafted token is shown with what became of it:

    accepted   the target agreed; the token is kept
    rejected   the target disagreed here; this token and everything
               drafted after it is thrown away
    discarded  drafted after the rejection point, so never even judged
    bonus      the target's own token for the position after the
               accepted prefix -- free, it comes from the same pass

Energy lives in the discarded column: those forward passes happened.
"""
from __future__ import annotations

import argparse
import html as _html
import json
import pathlib
import sys
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch  # noqa: E402

from specdec import engine  # noqa: E402
from specdec import datasets as ds  # noqa: E402
from specdec.configs import DEFAULT_DATASET, DEFAULT_PAIR, MODEL_PAIRS  # noqa: E402
from run_baseline import encode, load_pair  # noqa: E402

RESET, GREEN, RED, GREY, BLUE, BOLD = (
    "\033[0m", "\033[32m", "\033[31m", "\033[90m", "\033[36m", "\033[1m",
)


def visible(s: str) -> str:
    """Make whitespace legible without changing token boundaries."""
    return s.replace("\n", "⏎").replace("\t", "→").replace(" ", "·")


def classify(rec) -> list[dict]:
    """Per-drafted-token fate, reconstructed from one RoundRecord."""
    k, a = rec.proposed, rec.accepted
    out = []
    for i, tid in enumerate(rec.draft_ids):
        if i < a:
            fate = "accepted"
        elif i == a:
            fate = "rejected"
        else:
            fate = "discarded"
        out.append({"id": int(tid), "fate": fate, "position": i})
    # emitted = drafts[:accepted] + [bonus], possibly cut by the budget or EOS
    bonus = None
    if len(rec.emitted_ids) == a + 1:
        bonus = int(rec.emitted_ids[-1])
    return out, bonus


def render_terminal(rounds, tok, prompt_text, header):
    print(f"\n{BOLD}{header}{RESET}\n")
    print(f"{BOLD}prompt{RESET}  {prompt_text.strip()[:400]}")
    print(f"\n{GREEN}accepted{RESET}  {RED}rejected{RESET}  "
          f"{GREY}discarded{RESET}  {BLUE}bonus (from target){RESET}\n")
    colour = {"accepted": GREEN, "rejected": RED, "discarded": GREY}
    for r in rounds:
        chips = []
        for t in r["drafts"]:
            chips.append(f"{colour[t['fate']]}{visible(t['text'])}{RESET}")
        if r["bonus"] is not None:
            chips.append(f"{BLUE}{visible(r['bonus_text'])}{RESET}")
        bar = " ".join(chips) if chips else f"{BLUE}{visible(r['bonus_text'] or '')}{RESET}"
        print(f"  r{r['index']:<3} K={r['proposed']} acc={r['accepted']}/{r['proposed']}"
              f"  draft {r['draft_ms']:6.1f} ms  verify {r['verify_ms']:6.1f} ms   {bar}")


CSS = """
:root{--bg:#ffffff;--fg:#1a2332;--mut:#6a7a8c;--line:#e3e8ee;--card:#f5f7fa;
--acc:#1f7a4d;--accbg:#e4f4ec;--rej:#c2512f;--rejbg:#fdeae3;
--dis:#8a93a0;--disbg:#eef1f4;--bon:#1f3864;--bonbg:#e4ecf9;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
--bg:#12161c;--fg:#e8edf3;--mut:#93a1b1;--line:#28303a;--card:#1a2029;
--acc:#6ddba0;--accbg:#133224;--rej:#f09276;--rejbg:#36190f;
--dis:#7d8795;--disbg:#20262e;--bon:#9dc0f5;--bonbg:#14213a;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:16px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;}
.wrap{max-width:1100px;margin:0 auto;padding:40px 16px 80px}
h1{font-size:30px;margin:0 0 6px;letter-spacing:-.01em}
.sub{color:var(--mut);margin:0 0 28px;font-size:15px}
.meta{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:28px}
.meta div{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:10px 14px;font-size:14px}
.meta b{display:block;font-size:20px;font-weight:650}
.prompt{background:var(--card);border:1px solid var(--line);border-radius:12px;
padding:18px 20px;white-space:pre-wrap;font-size:15px;margin-bottom:26px}
.key{display:flex;flex-wrap:wrap;gap:18px;margin-bottom:22px;font-size:14px;color:var(--mut)}
.key span{display:inline-flex;align-items:center;gap:7px}
.dot{width:12px;height:12px;border-radius:3px;display:inline-block}
.round{border:1px solid var(--line);border-radius:12px;margin-bottom:10px;overflow:hidden}
.rhead{display:flex;flex-wrap:wrap;gap:16px;align-items:baseline;
background:var(--card);padding:9px 16px;font-size:13px;color:var(--mut);
border-bottom:1px solid var(--line)}
.rhead b{color:var(--fg);font-size:14px}
.toks{display:flex;flex-wrap:wrap;gap:6px;padding:14px 16px}
.t{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:14px;
padding:4px 9px;border-radius:7px;white-space:pre;border:1px solid transparent}
.accepted{background:var(--accbg);color:var(--acc);border-color:var(--acc)}
.rejected{background:var(--rejbg);color:var(--rej);border-color:var(--rej);
text-decoration:line-through}
.discarded{background:var(--disbg);color:var(--dis);text-decoration:line-through}
.bonus{background:var(--bonbg);color:var(--bon);border-color:var(--bon)}
.out{background:var(--card);border:1px solid var(--line);border-radius:12px;
padding:18px 20px;white-space:pre-wrap;margin-top:28px;font-size:15px}
h2{font-size:19px;margin:34px 0 12px}
@media(max-width:640px){.wrap{padding:24px 16px 60px}h1{font-size:24px}}
"""


def render_html(path, rounds, meta, prompt_text, output_text):
    def chip(text, cls):
        return f'<span class="t {cls}">{_html.escape(visible(text))}</span>'

    body = []
    for r in rounds:
        chips = [chip(t["text"], t["fate"]) for t in r["drafts"]]
        if r["bonus"] is not None:
            chips.append(chip(r["bonus_text"], "bonus"))
        body.append(
            '<div class="round"><div class="rhead">'
            f'<b>round {r["index"]}</b>'
            f'<span>K = {r["proposed"]}</span>'
            f'<span>accepted {r["accepted"]}/{r["proposed"]}</span>'
            f'<span>emitted {r["emitted"]}</span>'
            f'<span>draft {r["draft_ms"]:.1f} ms</span>'
            f'<span>verify {r["verify_ms"]:.1f} ms</span>'
            '</div>'
            f'<div class="toks">{"".join(chips)}</div></div>'
        )

    cards = "".join(
        f"<div>{k}<b>{v}</b></div>" for k, v in meta["cards"].items()
    )
    key = (
        '<div class="key">'
        '<span><i class="dot" style="background:var(--acc)"></i>accepted — target agreed</span>'
        '<span><i class="dot" style="background:var(--rej)"></i>rejected — target disagreed here</span>'
        '<span><i class="dot" style="background:var(--dis)"></i>discarded — drafted after the rejection, never judged</span>'
        '<span><i class="dot" style="background:var(--bon)"></i>bonus — the target\'s own next token, free</span>'
        '</div>'
    )
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Speculative Decoding Trace</title><style>{CSS}</style></head><body>
<div class="wrap">
<h1>Speculative decoding, token by token</h1>
<p class="sub">{_html.escape(meta['subtitle'])}</p>
<div class="meta">{cards}</div>
<h2>Prompt</h2>
<div class="prompt">{_html.escape(prompt_text.strip())}</div>
<h2>Rounds</h2>
{key}
{''.join(body)}
<h2>Final output</h2>
<div class="out">{_html.escape(output_text.strip())}</div>
</div></body></html>"""
    pathlib.Path(path).write_text(doc, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_PAIR, choices=sorted(MODEL_PAIRS))
    ap.add_argument("--quant", default="float16",
                    choices=["float16", "bfloat16", "float32", "nf4"])
    ap.add_argument("--compute-dtype", default="float16")
    ap.add_argument("--draft-precision", default="float16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dataset", default=DEFAULT_DATASET, choices=list(ds.PAPER_SIZES))
    ap.add_argument("--sample-index", type=int, default=0,
                    help="which prompt from the seeded sample (default 0)")
    ap.add_argument("--prompt", default=None, help="use this text instead")
    ap.add_argument("--seed", type=int, default=ds.PAPER_SEED)
    ap.add_argument("--k", default="4", help="draft length or strategy, e.g. 4, coga-5, dyga-4")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--no-chat-template", action="store_true")
    ap.add_argument("--no-compare", action="store_true",
                    help="skip the greedy cross-check (halves the runtime)")
    ap.add_argument("--offline", action="store_true",
                    help="use the built-in sample prompts, no dataset download")
    ap.add_argument("--synthetic", action="store_true",
                    help="two tiny random models; smoke-tests the tracer offline")
    ap.add_argument("--html", default="results/trace.html")
    ap.add_argument("--json", dest="json_out", default="results/trace.json")
    args = ap.parse_args()

    pair = MODEL_PAIRS[args.config]
    if args.synthetic:
        from run_baseline import load_synthetic
        tok, draft, target = load_synthetic(
            args.device, "float32" if args.quant == "nf4" else args.quant)
        args.no_chat_template = True
    else:
        tok, draft, target = load_pair(pair, args)

    if args.prompt is not None:
        prompt_text, sample_id = args.prompt, "custom"
    else:
        samples = ds.load(args.dataset, limit=args.sample_index + 1,
                          seed=args.seed, offline=args.offline or args.synthetic)
        s = samples[args.sample_index]
        prompt_text, sample_id = s.prompt, getattr(s, "index", args.sample_index)

    ids = encode(tok, prompt_text, args.device, not args.no_chat_template)
    eos = tok.eos_token_id

    engine.TRACE_TOKENS = True
    res = engine.speculative_generate(
        draft, target, ids, k=args.k,
        max_new_tokens=args.max_new_tokens, do_sample=False, eos_token_id=eos,
    )
    engine.TRACE_TOKENS = False

    identical = None
    if not args.no_compare:
        ref = engine.autoregressive_generate(
            target, ids, max_new_tokens=args.max_new_tokens,
            do_sample=False, eos_token_id=eos,
        )
        a = res.sequences[0, res.prompt_len:].tolist()
        b = ref.sequences[0, ref.prompt_len:].tolist()
        identical = a == b

    rounds, wasted, drafted = [], 0, 0
    for rec in res.rounds:
        toks, bonus = classify(rec)
        for t in toks:
            t["text"] = tok.decode([t["id"]])
        drafted += len(toks)
        wasted += sum(1 for t in toks if t["fate"] != "accepted")
        rounds.append({
            "index": rec.index, "proposed": rec.proposed, "accepted": rec.accepted,
            "emitted": rec.emitted, "draft_ms": rec.draft_ms, "verify_ms": rec.verify_ms,
            "drafts": toks, "bonus": bonus,
            "bonus_text": tok.decode([bonus]) if bonus is not None else "",
        })

    output_text = tok.decode(res.sequences[0, res.prompt_len:], skip_special_tokens=True)
    accept_pct = 100.0 * (drafted - wasted) / drafted if drafted else 0.0
    waste_pct = 100.0 * wasted / drafted if drafted else 0.0

    header = (f"{pair.label}  |  {args.quant}  |  {args.k}  |  "
              f"{args.dataset} sample {sample_id}")
    render_terminal(rounds, tok, prompt_text, header)
    print(f"\n  {len(res.rounds)} rounds, {res.new_tokens} tokens out, "
          f"{drafted} drafted, {wasted} thrown away ({waste_pct:.1f}%)")
    if identical is not None:
        print(f"  lossless check: {'IDENTICAL to greedy' if identical else 'DIVERGED'}")
    print(f"\n{output_text.strip()[:600]}\n")

    meta = {
        "subtitle": header + (
            "  ·  output identical to plain greedy" if identical else ""),
        "cards": {
            "tokens out": res.new_tokens,
            "rounds": len(res.rounds),
            "drafted": drafted,
            "accepted": f"{accept_pct:.0f}%",
            "thrown away": f"{wasted} ({waste_pct:.0f}%)",
            "tokens / round": f"{res.new_tokens / max(1, len(res.rounds)):.2f}",
        },
    }
    for p in (args.html, args.json_out):
        pathlib.Path(p).parent.mkdir(parents=True, exist_ok=True)
    render_html(args.html, rounds, meta, prompt_text, output_text)
    pathlib.Path(args.json_out).write_text(json.dumps({
        "pair": pair.key, "quant": args.quant, "strategy": str(args.k),
        "dataset": args.dataset, "sample": sample_id, "prompt": prompt_text,
        "output": output_text, "identical_to_greedy": identical,
        "drafted": drafted, "wasted": wasted, "rounds": rounds,
    }, indent=1), encoding="utf-8")
    print(f"  wrote {args.html}\n  wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
