from pathlib import Path
import argparse,os,time
import numpy as np,pandas as pd,torch,yaml
from torch.utils.data import DataLoader

from rxrx1.data.dataset import MultiRootRxRxDataset
from rxrx1.data.sampler import MixedPKBatchSampler
from rxrx1.data.manifest import read_manifest,create_label_to_index
from rxrx1.data.transforms import prepare_transforms,build_batch_transform
from rxrx1.data.normalization import build_normalizer
from rxrx1.models.factory import build_model
from rxrx1.training.criterion import build_criterion
from rxrx1.training.optimizers import build_optimizer
from rxrx1.training.schedulers import build_scheduler
from rxrx1.training.trainer import train_one_epoch
from rxrx1.training.checkpoint import save_checkpoint,load_checkpoint
from rxrx1.utils.seed import set_seed
from rxrx1.utils.logger import setup_logger
from rxrx1.utils.tracking import setup_wandb,update_wandb_git_info,fail_wandb
from predict_test_submit import filter_valid_test_wells,expand_complete_test_sites


def split_normalizer(n):
    if n is None:return None,None
    s=getattr(n,"apply_to","image")
    if s=="image":return n,None
    if s=="batch":return None,n
    raise ValueError(s)


def as_bool(s):
    if s.dtype==bool:return s
    return s.astype(str).str.lower().isin(["true","1","yes"])


def prepare_curriculum(root,cfg):
    pcfg=cfg["pseudo"]["curriculum"]
    n=int(pcfg.get("stages",10))

    state=pd.read_csv(root/pcfg["state"])
    required={"id_code","lsa_pred","confidence","selected","agreement"}
    miss=required-set(state.columns)
    if miss:raise ValueError(f"state missing columns: {sorted(miss)}")

    state=state.copy()
    state["selected"]=as_bool(state["selected"])
    state["agreement"]=as_bool(state["agreement"])
    state["id_code"]=state["id_code"].astype(str)

    initial=state[state["selected"]].copy()
    remaining=state[~state["selected"]].copy()

    # Easier / safer pseudo first:
    # agreement samples by confidence, then RAW/LSA conflicts.
    agree=(
        remaining[remaining["agreement"]]
        .sort_values("confidence",ascending=False)
        .reset_index(drop=True)
    )

    conflict=(
        remaining[~remaining["agreement"]]
        .sort_values("confidence",ascending=False)
        .reset_index(drop=True)
    )

    shards=[]

    # Stage 1-6: agreement only
    for idx in np.array_split(np.arange(len(agree)),6):
        shards.append(agree.iloc[idx].copy())

    # Stage 7-10: conflict only
    for idx in np.array_split(np.arange(len(conflict)),4):
        shards.append(conflict.iloc[idx].copy())

    raw=pd.read_csv(root/"data/raw/test.csv")
    sample=pd.read_csv(root/"data/raw/sample_submission.csv")
    raw,_=filter_valid_test_wells(raw,sample)

    test_root=root/"data/raw/test"
    sites=expand_complete_test_sites(raw,test_root).copy()
    sites["cell_type"]=sites["experiment"].str.split("-").str[0]
    sites["id_code"]=sites["id_code"].astype(str)

    out=root/pcfg["output_dir"]
    stage_dir=out/"stages"
    stage_dir.mkdir(parents=True,exist_ok=True)

    stages=[]
    rows=[]

    accumulated=[initial]

    for stage,shard in enumerate(shards,1):
        accumulated.append(shard)
        wells=pd.concat(accumulated,ignore_index=True)

        labels=wells[
            ["id_code","lsa_pred","confidence"]
        ].rename(columns={
            "lsa_pred":"sirna",
            "confidence":"pseudo_confidence",
        })

        manifest=sites.merge(
            labels,
            on="id_code",
            how="inner",
            validate="many_to_one",
        )

        if manifest["id_code"].nunique()!=len(wells):
            missing=set(wells["id_code"])-set(manifest["id_code"])
            raise RuntimeError(
                f"stage {stage}: {len(missing)} wells missing from images"
            )

        manifest["source"]="test"
        manifest["is_pseudo"]=True
        manifest["curriculum_stage"]=stage

        path=stage_dir/f"stage_{stage:02d}.csv"
        manifest.to_csv(path,index=False)
        stages.append(manifest)

        rows.append({
            "stage":stage,
            "new_wells":len(shard),
            "cumulative_pseudo_wells":len(wells),
            "cumulative_site_rows":len(manifest),
            "new_agreement_rate":float(shard["agreement"].mean()),
            "new_conf_max":float(shard["confidence"].max()),
            "new_conf_mean":float(shard["confidence"].mean()),
            "new_conf_min":float(shard["confidence"].min()),
        })

    summary=pd.DataFrame(rows)
    summary.to_csv(out/"stage_summary.csv",index=False)

    print("\nInitial pseudo wells:",len(initial))
    print("Remaining wells:",len(remaining))
    print(summary.to_string(index=False))

    return stages,summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True)
    p.add_argument("--prepare-only",action="store_true")
    args=p.parse_args()

    root=Path(__file__).resolve().parents[1]
    cfg=yaml.safe_load((root/args.config).read_text())
    set_seed(cfg["experiment"]["seed"])

    stages,summary=prepare_curriculum(root,cfg)

    if args.prepare_only:return

    device=torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    name=cfg["experiment"]["name"]
    logger=setup_logger(
        name,
        root/cfg["logging"]["log_dir"]/f"{name}.log",
        cfg["logging"]["level"],
    )

    run=setup_wandb(cfg,root)
    update_wandb_git_info(
        run,
        os.getenv("RXRX1_GIT_COMMIT"),
        os.getenv("RXRX1_GIT_REF"),
    )

    try:
        base=read_manifest(
            root/cfg["data"]["train_manifest"]
        ).copy()

        base["source"]="train"
        base["is_pseudo"]=False

        label_to_index=create_label_to_index(base)

        train_tf,_=prepare_transforms(cfg)

        batch_cfg=(
            (cfg.get("transform") or {}).get("batch")
            or []
        )

        batch_tf=build_batch_transform(
            batch_cfg,
            len(label_to_index),
        )

        normalizer=build_normalizer(
            cfg,
            split="train",
            project_root=root,
            logger=logger,
        )

        image_norm,batch_norm=split_normalizer(
            normalizer
        )

        pcfg=cfg["pseudo"]

        train_root=Path(
            pcfg.get(
                "train_image_root",
                "data/raw/train",
            )
        )

        test_root=Path(
            pcfg.get(
                "pseudo_image_root",
                "data/raw/test",
            )
        )

        if not train_root.is_absolute():
            train_root=root/train_root
        if not test_root.is_absolute():
            test_root=root/test_root

        bs=int(cfg["data"]["batch_size"])
        nw=int(cfg["data"]["num_workers"])
        k=int(
            (cfg["data"].get("pk_sampler") or {})
            .get("k",4)
        )
        pseudo_k=int(
            pcfg.get("pseudo_per_class",1)
        )
        seed=int(cfg["experiment"]["seed"])

        def make_loader(pseudo,stage):
            manifest=pd.concat(
                [base,pseudo],
                ignore_index=True,
            )

            ds=MultiRootRxRxDataset(
                manifest,
                image_roots={
                    "train":train_root,
                    "test":test_root,
                },
                label_to_index=label_to_index,
                transform=train_tf,
                normalizer=image_norm,
            )

            sampler=MixedPKBatchSampler(
                labels=ds.manifest["sirna"].tolist(),
                is_pseudo=ds.manifest["is_pseudo"].tolist(),
                batch_size=bs,
                k=k,
                pseudo_k=pseudo_k,
                seed=seed,
            )

            sampler.set_epoch(stage-1)

            return DataLoader(
                ds,
                batch_sampler=sampler,
                num_workers=nw,
                pin_memory=torch.cuda.is_available(),
            )

        # MixedPK epoch length depends on real training data,
        # therefore every stage has the same number of optimizer steps.
        first_loader=make_loader(stages[0],1)
        steps_per_stage=len(first_loader)
        del first_loader

        for i,s in enumerate(stages[1:],2):
            l=make_loader(s,i)
            if len(l)!=steps_per_stage:
                raise RuntimeError(
                    f"Stage {i} steps changed: "
                    f"{len(l)} != {steps_per_stage}"
                )
            del l

        model_cfg=dict(cfg["model"])
        model_cfg["pretrained"]=False

        model=build_model(
            model_config=model_cfg,
            num_classes=len(label_to_index),
            metric=cfg.get("metric") or {},
        ).to(device)

        criterion=build_criterion(cfg).to(device)

        tcfg=cfg["training"]
        init_from=tcfg.get("init_from")
        resume_from=tcfg.get("resume_from")

        if init_from and resume_from:
            raise ValueError(
                "init_from and resume_from cannot both be set"
            )

        optimizer=build_optimizer(model,cfg)

        scheduler=build_scheduler(
            optimizer=optimizer,
            config=cfg,
            epochs=len(stages),
            steps_per_epoch=steps_per_stage,
        )

        start_stage=0

        if resume_from:
            ckpt=load_checkpoint(
                path=root/resume_from,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                total_epochs=len(stages),
                steps_per_epoch=steps_per_stage,
                device=device,
            )

            start_stage=int(
                (ckpt.get("training_state") or {})
                .get("curriculum_stage",ckpt["epoch"])
            )

            logger.info(
                "Resumed curriculum | completed_stage=%d",
                start_stage,
            )

        elif init_from:
            state=torch.load(
                root/init_from,
                map_location=device,
                weights_only=False,
            )

            model.load_state_dict(
                state["model_state_dict"]
            )

            logger.info(
                "Initialized model weights | path=%s | source_epoch=%s",
                root/init_from,
                state.get("epoch"),
            )

        out=(
            root
            / cfg["checkpoint"]["dir"]
            / name
        )
        out.mkdir(parents=True,exist_ok=True)

        amp=bool(
            (tcfg.get("amp") or {})
            .get("enabled",False)
        )

        logger.info(
            "Curriculum | stages=%d | steps_per_stage=%d "
            "| total_steps=%d",
            len(stages),
            steps_per_stage,
            len(stages)*steps_per_stage,
        )

        for stage,pseudo in enumerate(stages,1):
            if stage<=start_stage:
                continue

            loader=make_loader(
                pseudo,
                stage,
            )

            row=summary.iloc[stage-1]

            logger.info(
                "Stage %02d start | new_wells=%d "
                "| cumulative_pseudo_wells=%d "
                "| agreement=%.4f "
                "| new_conf=%.5f..%.5f",
                stage,
                row.new_wells,
                row.cumulative_pseudo_wells,
                row.new_agreement_rate,
                row.new_conf_min,
                row.new_conf_max,
            )

            t=time.perf_counter()

            loss,acc=train_one_epoch(
                model=model,
                loader=loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                scheduler=scheduler,
                batch_normalizer=batch_norm,
                batch_transform=batch_tf,
                amp_enabled=amp,
            )

            minutes=(
                time.perf_counter()-t
            )/60

            training_state={
                "curriculum_stage":stage,
                "curriculum_stages":len(stages),
                "initial_pseudo_wells":int(
                    summary.iloc[0][
                        "cumulative_pseudo_wells"
                    ]
                    - summary.iloc[0]["new_wells"]
                ),
            }

            ckpt=out/f"stage_{stage:02d}.pt"

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=stage,
                path=ckpt,
                scheduler=scheduler,
                total_epochs=len(stages),
                steps_per_epoch=steps_per_stage,
                training_state=training_state,
            )

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=stage,
                path=out/"last.pt",
                scheduler=scheduler,
                total_epochs=len(stages),
                steps_per_epoch=steps_per_stage,
                training_state=training_state,
            )

            logger.info(
                "Stage %02d done | loss=%.5f "
                "| acc=%.5f | %.2f min | %s",
                stage,
                loss,
                acc,
                minutes,
                ckpt,
            )

            if run is not None:
                run.log({
                    "curriculum/stage":stage,
                    "curriculum/new_wells":
                        int(row.new_wells),
                    "curriculum/cumulative_pseudo_wells":
                        int(row.cumulative_pseudo_wells),
                    "curriculum/new_agreement_rate":
                        float(row.new_agreement_rate),
                    "curriculum/new_conf_min":
                        float(row.new_conf_min),
                    "curriculum/new_conf_mean":
                        float(row.new_conf_mean),
                    "curriculum/new_conf_max":
                        float(row.new_conf_max),
                    "train/loss":loss,
                    "train/acc":acc,
                    "runtime/stage_minutes":minutes,
                    "lr":
                        optimizer.param_groups[0]["lr"],
                })

        if run is not None:
            run.summary[
                "curriculum_completed_stages"
            ]=len(stages)
            run.finish()

        print("\nDONE:",out)

    except Exception:
        fail_wandb(run)
        raise


if __name__=="__main__":
    main()
