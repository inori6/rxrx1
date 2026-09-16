from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import torch,yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from rxrx1.data.dataset import RxRxDataset
from rxrx1.data.manifest import read_manifest,create_label_to_index
from rxrx1.data.transforms import prepare_transforms
from rxrx1.inference.lsa import lsa_predict
from rxrx1.models.factory import build_model
from predict_test_submit import (
    filter_valid_test_wells,
    expand_complete_test_sites,
    build_inference_normalizer,
    split_normalizer,
)

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--threshold",type=float,default=0.96)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--previous-state",default=None)
    p.add_argument("--save-logits",action="store_true")
    return p.parse_args()

def main():
    args=parse_args()
    root=Path(__file__).resolve().parents[1]
    cfg=yaml.safe_load((root/args.config).read_text())
    out=root/args.output_dir
    out.mkdir(parents=True,exist_ok=True)

    train_manifest=read_manifest(root/cfg["data"]["train_manifest"])
    label_to_index=create_label_to_index(train_manifest)
    index_to_label={v:k for k,v in label_to_index.items()}

    raw_test=pd.read_csv(root/"data/raw/test.csv")
    sample=pd.read_csv(root/"data/raw/sample_submission.csv")
    raw_test,sample=filter_valid_test_wells(raw_test,sample)

    test_root=root/"data/raw/test"
    test_sites=expand_complete_test_sites(raw_test,test_root).copy()
    test_sites["cell_type"]=test_sites["experiment"].str.split("-").str[0]

    dummy_label=next(iter(label_to_index))
    infer_manifest=test_sites.copy()
    infer_manifest["sirna"]=dummy_label

    _,test_transform=prepare_transforms(cfg)

    normalizer=build_inference_normalizer(
        config=cfg,
        test_manifest=infer_manifest,
        label_to_index=label_to_index,
        root=root,
        norm_stats_source="train",
    )
    image_normalizer,batch_normalizer=split_normalizer(normalizer)

    ds=RxRxDataset(
        infer_manifest,
        test_root,
        label_to_index,
        transform=test_transform,
        normalizer=image_normalizer,
    )

    loader=DataLoader(
        ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg=dict(cfg["model"])
    model_cfg["pretrained"]=False

    model=build_model(
        model_config=model_cfg,
        num_classes=len(label_to_index),
        metric=cfg.get("metric") or {},
    ).to(device)

    state=torch.load(
        root/args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    amp_enabled=device.type=="cuda"
    amp_dtype=(
        torch.bfloat16
        if amp_enabled and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    logit_sums={}
    site_counts={}

    with torch.inference_mode():
        for batch in tqdm(loader,desc="RAW inference"):
            images=batch["image"].to(device)

            metadata={
                "cell_type_idx":batch["cell_type_idx"].to(device),
                "well_position":batch["well_position"].to(device),
            }

            if batch_normalizer is not None:
                images=batch_normalizer(images,batch)

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                outputs=model(images,metadata)
                logits=outputs[0] if isinstance(outputs,tuple) else outputs

            logits=logits.float().cpu()

            for id_code,logit in zip(batch["id_code"],logits):
                id_code=str(id_code)

                if id_code not in logit_sums:
                    logit_sums[id_code]=logit.clone()
                    site_counts[id_code]=1
                else:
                    logit_sums[id_code]+=logit
                    site_counts[id_code]+=1

    well_meta=(
        raw_test[
            ["id_code","experiment","plate","well"]
        ]
        .drop_duplicates("id_code")
        .set_index("id_code")
    )

    rows=[]

    for id_code,total in logit_sums.items():
        mean_logits=(total/site_counts[id_code]).numpy()
        meta=well_meta.loc[id_code]

        rows.append({
            "id_code":id_code,
            "experiment":meta["experiment"],
            "plate":int(meta["plate"]),
            "well":meta["well"],
            "logits":mean_logits,
        })

    pred=pd.DataFrame(rows)

    logits=np.stack(pred["logits"].to_numpy())
    probs=torch.softmax(torch.from_numpy(logits),dim=1).numpy()

    raw_idx=probs.argmax(axis=1)
    pred["raw_pred"]=[index_to_label[int(i)] for i in raw_idx]
    pred["confidence"]=probs.max(axis=1)

    lsa_submission,assignments=lsa_predict(
        pred,
        train_manifest,
        label_to_index,
    )

    lsa_submission=lsa_submission.rename(
        columns={"sirna":"lsa_pred"}
    )

    state_df=pred[
        [
            "id_code",
            "experiment",
            "plate",
            "well",
            "raw_pred",
            "confidence",
        ]
    ].merge(
        lsa_submission,
        on="id_code",
        how="left",
        validate="one_to_one",
    )

    state_df["agreement"]=state_df["raw_pred"]==state_df["lsa_pred"]
    state_df["high_confidence"]=state_df["confidence"]>=args.threshold
    state_df["selected"]=state_df["agreement"] & state_df["high_confidence"]

    state_df["status"]=np.select(
        [
            state_df["selected"],
            state_df["agreement"],
        ],
        [
            "selected",
            "low_confidence",
        ],
        default="conflict",
    )

    total=len(state_df)
    selected=int(state_df["selected"].sum())
    agreement=int(state_df["agreement"].sum())
    high_conf=int(state_df["high_confidence"].sum())
    conflict=int((~state_df["agreement"]).sum())
    low_conf=int(
        (
            state_df["agreement"]
            & ~state_df["high_confidence"]
        ).sum()
    )

    flip_rate=None

    if args.previous_state:
        prev=pd.read_csv(root/args.previous_state)[
            ["id_code","lsa_pred"]
        ].rename(
            columns={"lsa_pred":"previous_lsa_pred"}
        )

        tmp=state_df.merge(prev,on="id_code",how="inner")

        if len(tmp):
            flip_rate=float(
                (tmp["lsa_pred"]!=tmp["previous_lsa_pred"]).mean()
            )

    summary={
        "threshold":args.threshold,
        "total":total,
        "selected_count":selected,
        "selected_rate":selected/total,
        "low_conf_count":low_conf,
        "conflict_count":conflict,
        "raw_lsa_agreement_count":agreement,
        "raw_lsa_agreement_rate":agreement/total,
        "confidence_ge_threshold_count":high_conf,
        "confidence_ge_threshold_rate":high_conf/total,
        "mean_confidence":float(state_df["confidence"].mean()),
        "label_flip_rate":flip_rate,
        "plate_assignments":len(assignments),
    }

    edges=np.linspace(0,1,51)
    counts,_=np.histogram(state_df["confidence"],bins=edges)

    hist=pd.DataFrame({
        "left":edges[:-1],
        "right":edges[1:],
        "count":counts,
        "fraction":counts/total,
    })

    selected_wells=(
        state_df.loc[
            state_df["selected"],
            ["id_code","lsa_pred","confidence"],
        ]
        .rename(
            columns={
                "lsa_pred":"sirna",
                "confidence":"pseudo_confidence",
            }
        )
    )

    pseudo_manifest=(
        test_sites
        .merge(
            selected_wells,
            on="id_code",
            how="inner",
            validate="many_to_one",
        )
    )

    pseudo_manifest["source"]="test"
    pseudo_manifest["is_pseudo"]=True
    pseudo_manifest["pseudo_round"]=1

    state_df.to_csv(out/"state.csv",index=False)
    hist.to_csv(out/"confidence_histogram.csv",index=False)
    selected_wells.to_csv(out/"selected_wells.csv",index=False)
    pseudo_manifest.to_csv(out/"pseudo_manifest.csv",index=False)

    class_counts=(
        selected_wells["sirna"]
        .value_counts()
        .rename_axis("sirna")
        .reset_index(name="well_count")
    )
    class_counts.to_csv(out/"selected_class_counts.csv",index=False)

    (out/"summary.json").write_text(
        json.dumps(summary,indent=2),
        encoding="utf-8",
    )

    if args.save_logits:
        pred.to_pickle(out/"well_logits.pkl")

    print()
    print("="*70)
    for k,v in summary.items():
        print(f"{k}: {v}")
    print("selected_classes:",selected_wells["sirna"].nunique())
    print("pseudo_site_rows:",len(pseudo_manifest))
    print("="*70)

if __name__=="__main__":
    main()
