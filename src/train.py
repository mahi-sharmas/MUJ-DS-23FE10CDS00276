"""Step 3 of the pipeline: train the authorship classifier, evaluate it on the
unseen test books, save results (metrics, confusion matrix) and the model."""
import json

import joblib
import matplotlib
matplotlib.use("Agg")                      # save plots to files (no screen needed)
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import find_names, mask_names, style_matrix
from src.utils import ensure_dir, load_config


def build_model(style_cols, cfg, use_tfidf=True):
    """Style features (scaled) [+ character n-gram TF-IDF] -> Logistic Regression."""
    m = cfg["model"]
    blocks = [("style", StandardScaler(), style_cols)]
    weights = {"style": 1.0}
    if use_tfidf:
        tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=tuple(m["tfidf_ngram_range"]),
                                sublinear_tf=True, min_df=2, max_features=m["tfidf_max_features"])
        blocks.append(("tfidf", tfidf, "text"))
        weights = {"style": m["style_weight"], "tfidf": 1.0}
    features = ColumnTransformer(blocks, transformer_weights=weights)
    return make_pipeline(features, LogisticRegression(max_iter=m["max_iter"]))


def make_inputs(texts, names):
    """Style features from the text + name-masked text for TF-IDF, in one table."""
    X = style_matrix(texts)
    X["text"] = [mask_names(t, names) for t in texts]
    return X


def main():
    cfg = load_config()
    out_dir = ensure_dir(cfg["model"]["results_dir"])
    df = pd.read_csv("data/chunks.csv")
    train, test = df[df.split == "train"], df[df.split == "test"]

    names = find_names(train["text"], cfg["features"]["name_min_count"], cfg["features"]["name_cap_ratio"])
    X_train, X_test = make_inputs(train["text"].tolist(), names), make_inputs(test["text"].tolist(), names)
    style_cols = [c for c in X_train.columns if c != "text"]

    # Compare: pure style vs style + TF-IDF
    results, models = {}, {}
    for label, use_tfidf in [("style_only", False), ("style_plus_tfidf", True)]:
        model = build_model(style_cols, cfg, use_tfidf).fit(X_train, train["author"])
        pred = model.predict(X_test)
        results[label] = {"accuracy": round(accuracy_score(test["author"], pred), 3),
                          "macro_f1": round(f1_score(test["author"], pred, average="macro"), 3)}
        models[label] = (model, pred)
        print(f"{label:18s} accuracy={results[label]['accuracy']:.3f}  macro-F1={results[label]['macro_f1']:.3f}")

    final_model, final_pred = models["style_plus_tfidf"]
    print("\n" + classification_report(test["author"], final_pred, digits=3))

    # Save metrics + confusion matrix
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(test["author"], final_pred, ax=ax, colorbar=False,
                                            xticks_rotation=45, cmap="Blues")
    ax.set_title("Confusion matrix - unseen test books")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)

    # Top style habits per author (what the model learned)
    lr = final_model[-1]
    coefs = pd.DataFrame(lr.coef_[:, :len(style_cols)], index=lr.classes_, columns=style_cols)
    lines = [f"{a:10s} " + ", ".join(coefs.loc[a].nlargest(5).index) for a in coefs.index]
    (out_dir / "top_style_features.txt").write_text("\n".join(lines))
    print("Top style features per author:\n" + "\n".join(lines))

    # Save everything the app needs in one file
    style_train = X_train[style_cols]
    bundle = {"model": final_model, "names": names, "style_cols": style_cols,
              "author_avg": style_train.groupby(train["author"].values).mean(),
              "overall_avg": style_train.mean()}
    ensure_dir("models")
    joblib.dump(bundle, cfg["model"]["model_path"])
    print(f"\nSaved model to {cfg['model']['model_path']} and results to {out_dir}/")


if __name__ == "__main__":
    main()
