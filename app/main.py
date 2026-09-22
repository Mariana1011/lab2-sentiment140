from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from typing import List, Union
from pathlib import Path
import re
import joblib
import mlflow
from mlflow import MlflowClient
import pandas as pd

MLFLOW_URI = "http://localhost:5000"
EXPERIMENT_NAME = "nlp-lab2-sentiment140"
MODEL_NAME = "sentiment140"
MODEL_ALIAS = "champion"

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_registry_uri(MLFLOW_URI)

client = MlflowClient(
    tracking_uri=MLFLOW_URI,
    registry_uri=MLFLOW_URI
)

app = FastAPI(
    title="Sentiment140 API",
    version="1.0.0"
)

model = None
vectorizer = None
champion_info = None


URL_RE = re.compile(
    r"https?://\S+|www\.\S+",
    flags=re.IGNORECASE
)
MENTION_RE = re.compile(r"@\w+")
SPACE_RE = re.compile(r"\s+")


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = URL_RE.sub("url", text)
    text = MENTION_RE.sub("user", text)
    text = SPACE_RE.sub(" ", text).strip()

    # P_ELONGATION
    text = re.sub(
        r"(.)\1{2,}",
        r"\1\1",
        text
    )

    return text


def load_champion():
    global model, vectorizer, champion_info

    champion_info = client.get_model_version_by_alias(
        MODEL_NAME,
        MODEL_ALIAS
    )

    root = Path(
        mlflow.artifacts.download_artifacts(
            artifact_uri=f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        )
    )

    model = joblib.load(
        root / "artifacts" / "final_model.joblib"
    )

    vectorizer = joblib.load(
        root / "artifacts" / "vectorizer.joblib"
    )


try:
    load_champion()
    print("✅ champion cargado correctamente")
except Exception as e:
    print("❌ Error cargando champion:", e)


class PredictRequest(BaseModel):
    text: Union[str, List[str]]

    @field_validator("text")
    @classmethod
    def validate_text(cls, value):
        texts = [value] if isinstance(value, str) else value

        if not 1 <= len(texts) <= 32:
            raise ValueError(
                "Debe enviar entre 1 y 32 textos."
            )

        for text in texts:
            if not isinstance(text, str):
                raise ValueError(
                    "Todos los elementos deben ser strings."
                )

            if len(text) == 0:
                raise ValueError(
                    "Los textos no pueden estar vacíos."
                )

            if len(text) > 1000:
                raise ValueError(
                    "Cada texto debe tener máximo 1000 caracteres."
                )

        return value


def get_experiment_or_503():
    try:
        exp = client.get_experiment_by_name(
            EXPERIMENT_NAME
        )

        if exp is None:
            raise Exception()

        return exp

    except Exception:
        raise HTTPException(
            status_code=503,
            detail="mlflow_unavailable"
        )



@app.get("/health")
def health():
    try:
        champion = client.get_model_version_by_alias(
            MODEL_NAME,
            MODEL_ALIAS
        )

        # Comprobamos también que el run exista realmente
        client.get_run(champion.run_id)

        return {
            "status": "ok",
            "model_run_id": champion.run_id
        }

    except Exception:
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=503,
            content={
                "status": "unavailable",
                "model_run_id": None
            }
        )


@app.post("/api/v1/predict")
def predict(request: PredictRequest):

    global model, vectorizer, champion_info

    if model is None or vectorizer is None:
        try:
            load_champion()
        except Exception:
            raise HTTPException(
                status_code=503,
                detail="model_unavailable"
            )

    texts = (
        [request.text]
        if isinstance(request.text, str)
        else request.text
    )

    clean_texts = [
        preprocess_text(t)
        for t in texts
    ]

    X = vectorizer.transform(clean_texts)
    preds = model.predict(X)

    label_map = {
        0: "negative",
        1: "positive"
    }

    predictions = [
        label_map[int(pred)]
        for pred in preds
    ]

    return {
        "model_run_id": champion_info.run_id,
        "predictions": predictions
    }


# ============================================================
# FUNCIONES AUXILIARES DE AUDITORÍA
# ============================================================

def _presented_runs(exp):
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        max_results=5000
    )

    return [
        r for r in runs
        if r.data.tags.get("lab_run_type")
        in {"protocol", "experiment", "final"}
    ]


def _artifacts_recursive(run_id):
    result = []

    def walk(path=None):
        for item in client.list_artifacts(run_id, path):
            if item.is_dir:
                walk(item.path)
            else:
                result.append(item.path)

    walk()
    return sorted(result)


def _configuration(run_id):
    import json

    try:
        p = client.download_artifacts(
            run_id,
            "run/configuration.json"
        )

        with open(p, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        if not isinstance(cfg, dict):
            return None

        expected = {
            "preprocessing",
            "representation",
            "classifier"
        }

        if set(cfg.keys()) != expected:
            return None

        return cfg

    except Exception:
        return None


def _protocol_run(exp):
    protocols = [
        r for r in _presented_runs(exp)
        if r.data.tags.get("lab_run_type") == "protocol"
    ]

    if len(protocols) != 1:
        raise HTTPException(
            status_code=409,
            detail="protocol_not_unique"
        )

    return protocols[0]


def _members(protocol_run):
    import csv

    try:
        p = client.download_artifacts(
            protocol_run.info.run_id,
            "protocol/members.csv"
        )

        result = []

        with open(
            p,
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as f:
            reader = csv.DictReader(f)

            for row in reader:
                result.append({
                    "member_id": row["member_id"],
                    "notebook_arn": row["notebook_arn"]
                })

        return result

    except Exception:
        return []


def _provenance_ok(run):
    import json

    try:
        p = client.download_artifacts(
            run.info.run_id,
            "provenance/sagemaker-resource-metadata.json"
        )

        with open(p, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        resource_arn = metadata.get("ResourceArn")

        return (
            resource_arn is not None
            and resource_arn
            == run.data.tags.get("notebook_arn")
        )

    except Exception:
        return False


def _experiment_valid(
    run,
    protocol_run_id,
    valid_pairs
):
    tags = run.data.tags
    metrics = run.data.metrics

    if run.info.status != "FINISHED":
        return False

    if tags.get("lab_run_type") != "experiment":
        return False

    required_tags = [
        "lab_protocol_run_id",
        "lab_experiment_id",
        "lab_stage",
        "lab_member_id",
        "lab_configuration_id",
        "notebook_arn"
    ]

    if any(not tags.get(x) for x in required_tags):
        return False

    required_metrics = [
        "macro_f1_fold_0",
        "macro_f1_fold_1",
        "macro_f1_fold_2",
        "macro_f1_mean",
        "macro_f1_std"
    ]

    if any(x not in metrics for x in required_metrics):
        return False

    if tags.get("lab_protocol_run_id") != protocol_run_id:
        return False

    pair = (
        tags.get("lab_member_id"),
        tags.get("notebook_arn")
    )

    if pair not in valid_pairs:
        return False

    if _configuration(run.info.run_id) is None:
        return False

    if not _provenance_ok(run):
        return False

    if tags.get("lab_stage") == "ablation":

        if not tags.get("lab_ablation_parent_run_id"):
            return False

        if not run.data.params.get(
            "ablation_reverted_decision"
        ):
            return False

        if "macro_f1_delta" not in metrics:
            return False

    return True


@app.get("/audit/protocol")
def audit_protocol():

    exp = get_experiment_or_503()

    try:
        r = _protocol_run(exp)
        p = r.data.params

        return {
            "protocol_run_id": r.info.run_id,
            "dataset_id": p["dataset_id"],
            "dataset_revision": p["dataset_revision"],
            "sampling_strategy": p["sampling_strategy"],
            "sample_size": int(p["sample_size"]),
            "random_seed": int(p["random_seed"]),
            "cv_strategy": p["cv_strategy"],
            "cv_folds": int(p["cv_folds"]),
            "cv_shuffle":
                str(p["cv_shuffle"]).lower() == "true",
            "partitions_artifact":
                "protocol/partitions.csv",
            "members_artifact":
                "protocol/members.csv"
        }

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=503,
            detail="mlflow_unavailable"
        )


@app.get("/audit/runs")
def audit_runs():

    exp = get_experiment_or_503()

    try:
        runs = _presented_runs(exp)

        output = []

        for r in sorted(
            runs,
            key=lambda x: x.info.start_time or 0
        ):
            output.append({
                "run_id": r.info.run_id,
                "status": r.info.status,
                "run_type": r.data.tags.get(
                    "lab_run_type"
                ),
                "params": dict(r.data.params),
                "metrics": {
                    k: float(v)
                    for k, v
                    in r.data.metrics.items()
                },
                "tags": {
                    str(k): str(v)
                    for k, v
                    in r.data.tags.items()
                },
                "artifacts":
                    _artifacts_recursive(
                        r.info.run_id
                    ),
                "configuration":
                    (
                        None
                        if r.data.tags.get(
                            "lab_run_type"
                        ) == "protocol"
                        else _configuration(
                            r.info.run_id
                        )
                    )
            })

        return {
            "runs": output
        }

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=503,
            detail="mlflow_unavailable"
        )


@app.get("/audit/contributions")
def audit_contributions():

    exp = get_experiment_or_503()

    try:
        protocol = _protocol_run(exp)

        members_source = _members(protocol)

        valid_pairs = {
            (
                m["member_id"],
                m["notebook_arn"]
            )
            for m in members_source
        }

        members = {}

        for m in members_source:
            members[m["member_id"]] = {
                "member_id": m["member_id"],
                "notebook_arn": m["notebook_arn"],
                "run_ids": [],
                "counted_run_ids": [],
                "configuration_ids": [],
                "stages": []
            }

        invalid_run_ids = []
        unattributed_run_ids = []

        for r in _presented_runs(exp):

            if r.data.tags.get(
                "lab_run_type"
            ) != "experiment":
                continue

            member_id = r.data.tags.get(
                "lab_member_id"
            )

            notebook_arn = r.data.tags.get(
                "notebook_arn"
            )

            pair = (
                member_id,
                notebook_arn
            )

            attributable = pair in valid_pairs

            if not attributable:
                unattributed_run_ids.append(
                    r.info.run_id
                )

            valid = _experiment_valid(
                r,
                protocol.info.run_id,
                valid_pairs
            )

            if not valid:
                invalid_run_ids.append(
                    r.info.run_id
                )
                continue

            data = members[member_id]

            data["run_ids"].append(
                r.info.run_id
            )

            # T0 y B0 no cuentan para el mínimo
            experiment_id = r.data.tags.get(
                "lab_experiment_id"
            )

            if experiment_id not in {"T0", "B0"}:

                data["counted_run_ids"].append(
                    r.info.run_id
                )

                configuration_id = (
                    r.data.tags.get(
                        "lab_configuration_id"
                    )
                )

                if configuration_id:
                    data[
                        "configuration_ids"
                    ].append(
                        configuration_id
                    )

                stage = r.data.tags.get(
                    "lab_stage"
                )

                if stage:
                    data["stages"].append(
                        stage
                    )

        result = []

        for member_id in sorted(members):

            data = members[member_id]

            data["run_ids"] = sorted(
                set(data["run_ids"])
            )

            data["counted_run_ids"] = sorted(
                set(data["counted_run_ids"])
            )

            data["configuration_ids"] = sorted(
                set(data["configuration_ids"])
            )

            data["stages"] = sorted(
                set(data["stages"])
            )

            data["valid_configurations"] = len(
                data["configuration_ids"]
            )

            data["stage_count"] = len(
                data["stages"]
            )

            data["meets_requirement"] = (
                data["valid_configurations"] >= 3
                and data["stage_count"] >= 2
            )

            result.append(data)

        return {
            "members": result,
            "invalid_run_ids":
                sorted(set(invalid_run_ids)),
            "unattributed_run_ids":
                sorted(set(unattributed_run_ids))
        }

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=503,
            detail="mlflow_unavailable"
        )


@app.get("/audit/model")
def audit_model():

    exp = get_experiment_or_503()

    try:
        champion = client.get_model_version_by_alias(
            MODEL_NAME,
            MODEL_ALIAS
        )

        run = client.get_run(
            champion.run_id
        )

        protocol = _protocol_run(exp)

        configuration = _configuration(
            run.info.run_id
        )

        selected_run_id = (
            run.data.tags.get(
                "lab_selected_experiment_run_id"
            )
        )

        if run.info.status != "FINISHED":
            raise HTTPException(
                status_code=409,
                detail="champion_invalid"
            )

        if (
            run.data.tags.get("lab_run_type")
            != "final"
        ):
            raise HTTPException(
                status_code=409,
                detail="champion_invalid"
            )

        if (
            run.data.tags.get(
                "lab_protocol_run_id"
            )
            != protocol.info.run_id
        ):
            raise HTTPException(
                status_code=409,
                detail="champion_invalid"
            )

        if configuration is None:
            raise HTTPException(
                status_code=409,
                detail="champion_invalid"
            )

        if not selected_run_id:
            raise HTTPException(
                status_code=409,
                detail="champion_invalid"
            )

        selected = client.get_run(
            selected_run_id
        )

        if (
            selected.data.tags.get(
                "lab_run_type"
            )
            != "experiment"
        ):
            raise HTTPException(
                status_code=409,
                detail="champion_invalid"
            )

        return {
            "model_name": MODEL_NAME,
            "alias": MODEL_ALIAS,
            "version": int(champion.version),
            "run_id": champion.run_id,
            "protocol_run_id":
                run.data.tags.get(
                    "lab_protocol_run_id"
                ),
            "selected_experiment_run_id":
                selected_run_id,
            "configuration_id":
                run.data.tags.get(
                    "lab_configuration_id"
                ),
            "configuration":
                configuration,
            "training_size":
                int(
                    run.data.params[
                        "training_size"
                    ]
                ),
            "test_macro_f1":
                float(
                    run.data.metrics[
                        "test_macro_f1"
                    ]
                )
        }

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=503,
            detail="mlflow_unavailable"
        )
