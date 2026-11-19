import logging
import math
import re
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "pokemon_type_xgboost.joblib"
FEATURES = (
    "height_dm",
    "weight_hg",
    "base_experience",
    "hp",
    "attack",
    "defense",
)
FEATURE_LABELS = {
    "height_dm": "Height (dm)",
    "weight_hg": "Weight (hg)",
    "base_experience": "Base experience",
    "hp": "HP",
    "attack": "Attack",
    "defense": "Defense",
}

app = Flask(__name__, static_folder="statics")
app.logger.setLevel(logging.INFO)
model = joblib.load(MODEL_PATH)["model"]


def normalize_feature_name(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def get_model_features():
    model_features = getattr(model, "feature_names_in_", None)
    if model_features is None:
        return list(FEATURES)

    aliases = {
        "heightdm": "height_dm",
        "weighthg": "weight_hg",
        "hp": "hp",
        "attack": "attack",
        "defense": "defense",
        "baseexperience": "base_experience",
        "baseexp": "base_experience",
        "experience": "base_experience",
    }
    resolved = []
    for name in model_features:
        feature = aliases.get(normalize_feature_name(name))
        if feature is None:
            raise ValueError(
                f"The model expects an unsupported input feature: {name}"
            )
        resolved.append(feature)

    if len(resolved) != len(FEATURES) or set(resolved) != set(FEATURES):
        raise ValueError(
            "The model must use Height, Weight, Base experience, HP, Attack, and Defense."
        )
    return resolved


MODEL_FEATURES = get_model_features()


@app.get("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html", values={})

    values = {}
    errors = []
    for feature in FEATURES:
        raw_value = request.form.get(feature, "").strip()
        try:
            value = float(raw_value)
        except ValueError:
            errors.append(f"Enter a valid number for {FEATURE_LABELS[feature]}.")
            continue

        if not math.isfinite(value) or value < 0:
            errors.append(f"{FEATURE_LABELS[feature]} must be a non-negative number.")
            continue
        values[feature] = value

    if errors:
        return (
            render_template(
                "predict.html",
                values=request.form,
                errors=errors,
            ),
            400,
        )

    sample = pd.DataFrame(
        [[values[feature] for feature in MODEL_FEATURES]],
        columns=MODEL_FEATURES,
    )
    prediction = model.predict(sample)[0]
    if hasattr(prediction, "item"):
        prediction = prediction.item()

    return render_template(
        "result.html",
        prediction=str(prediction),
        values=values,
    )


if __name__ == "__main__":
    app.run()
