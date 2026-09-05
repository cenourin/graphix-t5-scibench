import json
import argparse

# ScienceBenchmark ships 3 DBs (cordis, oncomx, sdss), each with its own
# seed.json (human-written), synth.json (template-generated) and dev.json.
# Mirrors merge_train.py's train_spider+train_others concatenation, but
# generalized across the 3 DBs and the train/dev split.
DBS = ["cordis", "oncomx", "sdss"]


def load_and_merge(base_dir, dbs, splits):
    merged = []
    for db in dbs:
        for split in splits:
            path = f"{base_dir}/{db}/{split}.json"
            examples = json.load(open(path, "r"))
            merged.extend(examples)
    return merged


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--base_dir",
        type=str,
        default="data_all_in/data/sciencebenchmark",
        help="directory containing cordis/oncomx/sdss subfolders",
    )
    arg_parser.add_argument("--train_output_path", type=str, required=True)
    arg_parser.add_argument("--dev_output_path", type=str, required=True)
    args = arg_parser.parse_args()

    train = load_and_merge(args.base_dir, DBS, ["seed", "synth"])
    dev = load_and_merge(args.base_dir, DBS, ["dev"])

    json.dump(train, open(args.train_output_path, "w"), indent=4)
    json.dump(dev, open(args.dev_output_path, "w"), indent=4)
    print(f"merged ScienceBenchmark: train={len(train)} examples, dev={len(dev)} examples")
