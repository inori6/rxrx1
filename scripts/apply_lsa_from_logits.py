from pathlib import Path
import argparse
import pandas as pd
import yaml

from rxrx1.data.manifest import read_manifest,create_label_to_index
from rxrx1.inference.lsa import lsa_predict

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True)
    p.add_argument("--logits",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()

    root=Path(__file__).resolve().parents[1]
    cfg=yaml.safe_load((root/args.config).read_text())

    train_manifest=read_manifest(root/cfg["data"]["train_manifest"])
    label_to_index=create_label_to_index(train_manifest)

    predictions=pd.read_pickle(root/args.logits)

    submission,assignments=lsa_predict(
        predictions,
        train_manifest,
        label_to_index,
    )

    mapping=pd.read_csv(root/"data/sirna_id_map.csv")
    sirna_to_official=dict(zip(mapping["sirna"],mapping["sirna_id"]))

    submission["sirna"]=submission["sirna"].map(sirna_to_official)
    assert not submission["sirna"].isna().any()
    submission["sirna"]=submission["sirna"].astype(int)

    sample=pd.read_csv(root/"data/raw/sample_submission.csv")

    invalid={
        "HUVEC-18_3_D23",
        "RPE-09_2_J16",
    }

    sample=sample[
        ~sample["id_code"].astype(str).isin(invalid)
    ].copy()

    pred=dict(zip(
        submission["id_code"].astype(str),
        submission["sirna"],
    ))

    sample["sirna"]=sample["id_code"].astype(str).map(pred)

    assert not sample["sirna"].isna().any()
    sample["sirna"]=sample["sirna"].astype(int)

    out=root/args.output
    out.parent.mkdir(parents=True,exist_ok=True)
    sample.to_csv(out,index=False)

    print("plate assignments:",len(assignments))
    print("saved:",out)
    print("rows:",len(sample))

if __name__=="__main__":
    main()
