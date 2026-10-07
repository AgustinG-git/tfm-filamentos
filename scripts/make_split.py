"""Genera splits/split_v1.json una sola vez.

Uso (local o en Kaggle):

    python scripts/make_split.py --annotations <ruta al JSON de train>

Después se hace commit de splits/split_v1.json y nunca se regenera.
"""

import argparse
import json

from filseg import contract
from filseg.gt import build_meta, load_coco
from filseg.split import make_split, save_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, help="JSON COCO de train")
    parser.add_argument("--out", default=contract.SPLIT_PATH)
    args = parser.parse_args()

    meta = build_meta(load_coco(args.annotations))
    split = make_split(meta, annotations_path=args.annotations)
    path = save_split(split, args.out)

    print(f"Split guardado en {path}")
    print(json.dumps(split["summary"], indent=1))


if __name__ == "__main__":
    main()
