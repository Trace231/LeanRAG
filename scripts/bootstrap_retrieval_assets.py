import argparse
from pathlib import Path

from lean_dojo_v2.agent.lean_agent import LeanAgent

url = "https://github.com/durant42040/lean4-example"
commit = "005de00d03f1aaa32cb2923d5e3cbaf0b954a192"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap retrieval assets (trace + train retriever)."
    )
    parser.add_argument("--url", default=url)
    parser.add_argument("--commit", default=commit)
    parser.add_argument("--database-path", default="dynamic_database.json")
    parser.add_argument(
        "--build-deps",
        action="store_true",
        help="Enable full dependency tracing. Default is False (noDeps).",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Only trace/export merged data. Skip retriever training.",
    )
    args = parser.parse_args()

    agent = LeanAgent(database_path=args.database_path)

    # Avoid BaseAgent.setup_github_repository() because LeanAgent forces
    # build_deps=True there. We expose it as a CLI knob for stable noDeps runs.
    traced_repo = agent.trace_repository(
        url=args.url, commit=args.commit, build_deps=args.build_deps
    )
    agent.add_repository(traced_repo)
    # Always export merged data so corpus.jsonl is available for proving.
    merged_path = Path("raid/data/merged")
    agent.database.export_merged_data([traced_repo], merged_path)

    if args.skip_train:
        print(
            f"bootstrap done (trace/export only, build_deps={args.build_deps}) "
            f"for {args.url}@{args.commit}"
        )
        return

    agent.train()
    print(
        f"bootstrap done (build_deps={args.build_deps}, trained=True) "
        f"for {args.url}@{args.commit}"
    )


if __name__ == "__main__":
    main()
