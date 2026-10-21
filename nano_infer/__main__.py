import argparse
import json
from pathlib import Path

from .labs import LABS
from .model import TinyModel, decode, encode, greedy
from .verification import shape_tour, verification_report
from .training import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="CPU-sized lessons on DeepInfer design")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="run a lesson and check its numerical contract")
    demo.add_argument("lesson", choices=["all", *LABS], default="all", nargs="?")
    demo.add_argument("--json", type=Path, help="write reproducible lesson results")
    demo.add_argument("--trace", action="store_true", help="print full round/batch traces")
    for name, help_text in (("tour", "observe actual prefill/decode tensor shapes"),
                            ("verify", "check an independent toy oracle and emit evidence")):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--json", type=Path, help="write the full result")
    generate = sub.add_parser("generate", help="generate text from a trained checkpoint or seeded weights")
    generate.add_argument("--checkpoint", type=Path)
    generate.add_argument("--prompt", default="nano")
    generate.add_argument("--max-tokens", type=int, default=24)
    args = parser.parse_args()
    if args.command == "generate":
        try:
            model = load_checkpoint(args.checkpoint) if args.checkpoint else TinyModel()
            tokens = greedy(model, encode(args.prompt), args.max_tokens)
        except ValueError as exc:
            parser.error(str(exc))
        print(f"Model: {args.checkpoint or 'seeded initialization'}")
        print(args.prompt + decode(tokens))
        return
    if args.command == "tour":
        report = shape_tour()
    elif args.command == "verify":
        report = verification_report()
    else:
        selected = LABS if args.lesson == "all" else {args.lesson: LABS[args.lesson]}
        report = {name: run() for name, run in selected.items()}
    text = json.dumps(report, indent=2, ensure_ascii=False)
    def compact(value):
        if isinstance(value, dict):
            return {k: compact(v) for k, v in value.items() if k not in ("rounds", "batches")}
        return value
    print(text if getattr(args, "trace", False) else json.dumps(compact(report), indent=2, ensure_ascii=False))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
