from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--stage-dir",required=True)
    p.add_argument("--stage",required=True)
    p.add_argument("--threshold",type=float,default=0.96)
    p.add_argument("--history",required=True)
    p.add_argument("--previous-state",default=None)
    return p.parse_args()

def main():
    args=parse_args()

    stage_dir=Path(args.stage_dir)
    state_path=stage_dir/"state.csv"
    history_path=Path(args.history)

    df=pd.read_csv(state_path)

    required={
        "id_code",
        "raw_pred",
        "lsa_pred",
        "confidence",
    }
    missing=required-set(df.columns)

    if missing:
        raise ValueError(
            f"state.csv missing columns: {sorted(missing)}"
        )

    df["agreement"]=df["raw_pred"]==df["lsa_pred"]
    df["high_confidence"]=df["confidence"]>=args.threshold
    df["selected"]=df["agreement"] & df["high_confidence"]

    df["status"]=np.select(
        [
            df["selected"],
            df["agreement"],
        ],
        [
            "selected",
            "low_confidence",
        ],
        default="conflict",
    )

    total=len(df)
    selected=int(df["selected"].sum())
    agreement=int(df["agreement"].sum())
    high_conf=int(df["high_confidence"].sum())
    conflict=int((~df["agreement"]).sum())
    low_conf=int(
        (
            df["agreement"]
            & ~df["high_confidence"]
        ).sum()
    )

    retained_selected=None
    new_selected=None
    dropped_selected=None
    label_flip_rate=None

    if args.previous_state:
        prev=pd.read_csv(args.previous_state)

        prev["agreement"]=(
            prev["raw_pred"]
            == prev["lsa_pred"]
        )
        prev["high_confidence"]=(
            prev["confidence"]
            >= args.threshold
        )
        prev["selected"]=(
            prev["agreement"]
            & prev["high_confidence"]
        )

        old=prev[
            ["id_code","lsa_pred","selected"]
        ].rename(
            columns={
                "lsa_pred":"previous_lsa_pred",
                "selected":"previous_selected",
            }
        )

        cmp=df.merge(
            old,
            on="id_code",
            how="inner",
            validate="one_to_one",
        )

        retained_selected=int(
            (
                cmp["previous_selected"]
                & cmp["selected"]
            ).sum()
        )

        new_selected=int(
            (
                ~cmp["previous_selected"]
                & cmp["selected"]
            ).sum()
        )

        dropped_selected=int(
            (
                cmp["previous_selected"]
                & ~cmp["selected"]
            ).sum()
        )

        label_flip_rate=float(
            (
                cmp["lsa_pred"]
                != cmp["previous_lsa_pred"]
            ).mean()
        )

    summary={
        "stage":args.stage,
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
        "mean_confidence":float(df["confidence"].mean()),
        "retained_selected":retained_selected,
        "new_selected":new_selected,
        "dropped_selected":dropped_selected,
        "label_flip_rate":label_flip_rate,
    }

    # 0.02 confidence bins
    edges=np.linspace(0.0,1.0,51)
    counts,_=np.histogram(
        df["confidence"],
        bins=edges,
    )

    hist=pd.DataFrame({
        "left":edges[:-1],
        "right":edges[1:],
        "count":counts,
        "fraction":counts/total,
    })

    hist.to_csv(
        stage_dir/"confidence_histogram.csv",
        index=False,
    )

    # confidence distribution PNG
    fig,ax=plt.subplots(figsize=(11,6))

    ax.hist(
        df["confidence"],
        bins=edges,
    )

    ax.axvline(
        args.threshold,
        linestyle="--",
        linewidth=1.5,
        label=f"threshold = {args.threshold:.2f}",
    )

    ax.set_title(
        f"Pseudo-label Confidence Distribution — {args.stage}"
    )
    ax.set_xlabel("Raw prediction confidence")
    ax.set_ylabel("Number of wells")
    ax.set_xlim(0,1)
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        stage_dir/"confidence_distribution.png",
        dpi=180,
    )
    plt.close(fig)

    # overwrite state with standardized columns
    df.to_csv(
        stage_dir/"state.csv",
        index=False,
    )

    (stage_dir/"summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    history_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    row=pd.DataFrame([summary])

    if history_path.exists():
        history=pd.read_csv(history_path)
        history=history[
            history["stage"].astype(str)
            != str(args.stage)
        ]
        history=pd.concat(
            [history,row],
            ignore_index=True,
        )
    else:
        history=row

    order={
        "teacher":0,
        "epoch3":3,
        "epoch5":5,
    }

    history["_order"]=(
        history["stage"]
        .map(order)
        .fillna(999)
    )

    history=(
        history
        .sort_values("_order")
        .drop(columns="_order")
    )

    history.to_csv(
        history_path,
        index=False,
    )

    print()
    print("="*70)
    for k,v in summary.items():
        print(f"{k}: {v}")
    print()
    print(
        "plot:",
        stage_dir/"confidence_distribution.png",
    )
    print(
        "history:",
        history_path,
    )
    print("="*70)

if __name__=="__main__":
    main()
