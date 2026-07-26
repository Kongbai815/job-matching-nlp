import argparse
import json
from pathlib import Path

from .core import JobMatchingSystem


def main():
    parser = argparse.ArgumentParser(description="Run the MSBA job matching prototype.")
    parser.add_argument("--query", help="Advisor/student query text.")
    parser.add_argument("--input-json", help="JSON file with a query field.")
    parser.add_argument("--data", default="data/data_jobs_msba_project_sample_100k.csv", help="Path to project CSV.")
    parser.add_argument("--top-k", type=int, default=6, help="Number of retrieved postings.")
    parser.add_argument("--output", default="outputs/system_output.json", help="Where to write JSON output.")
    args = parser.parse_args()

    if args.input_json:
        payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
        query = payload.get("query") or payload.get("text") or json.dumps(payload)
    elif args.query:
        query = args.query
    else:
        raise SystemExit("Provide --query or --input-json.")

    system = JobMatchingSystem(data_path=args.data, top_k=args.top_k)
    result = system.run(query, top_k=args.top_k)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
