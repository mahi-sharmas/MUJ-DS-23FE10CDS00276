"""The demo: paste your writing -> closest classic author + an explanation
(classifier finds the style habits, Gemini explains them)."""
import joblib
import pandas as pd

from src.features import mask_names, normalize_punctuation, style_features
from src.llm import Explainer, load_prompts
from src.utils import load_config

CFG = load_config()
BUNDLE = joblib.load(CFG["model"]["model_path"])
AUTHOR_NAMES = {k: v["name"] for k, v in CFG["data"]["authors"].items()}
EXPLAINER = Explainer(CFG, load_prompts())


def describe(feat):
    """Turn a feature name like 'fw_upon' into readable words."""
    if feat.startswith("fw_"):
        return f'uses of the word "{feat[3:]}" per 100 words'
    if feat.startswith("p_"):
        return f"{feat[2:]} marks per 100 words"
    return {"avg_sent_len": "average sentence length (words)",
            "std_sent_len": "variation in sentence length",
            "avg_word_len": "average word length (letters)"}.get(feat, feat)


def analyze(text):
    """Run the classifier on the text and collect the facts to explain."""
    words = normalize_punctuation(text).split()
    if not any(c.isalpha() for c in text):
        return None
    words = words[:CFG["app"]["max_words"]]
    clean = " ".join(words)

    feats = style_features(clean)
    X = pd.DataFrame([feats])[BUNDLE["style_cols"]]
    X["text"] = mask_names(clean, BUNDLE["names"])

    model = BUNDLE["model"]
    probs = dict(zip(model.classes_, model.predict_proba(X)[0]))
    best = max(probs, key=probs.get)

    # How much each style feature pushed toward the winner = weight x scaled value
    scaler = model[0].named_transformers_["style"]
    z = pd.Series(scaler.transform(X[BUNDLE["style_cols"]])[0], index=BUNDLE["style_cols"])
    coef = pd.Series(model[-1].coef_[list(model.classes_).index(best)][:len(z)], index=z.index)
    push = (coef * z).drop("type_token_ratio").nlargest(3)   # TTR is unreliable for short texts

    name = AUTHOR_NAMES[best]
    habits = [f"- {describe(f)}: you {feats[f]:.2f} | {name} {BUNDLE['author_avg'].loc[best, f]:.2f} "
              f"| all authors {BUNDLE['overall_avg'][f]:.2f}" for f in push.index]
    top3 = sorted(probs.items(), key=lambda kv: -kv[1])[:3]
    return {"author": name, "n_words": len(words), "habits": "\n".join(habits),
            "top_matches": ", ".join(f"{AUTHOR_NAMES[a]} {p:.0%}" for a, p in top3),
            "probs": {AUTHOR_NAMES[a]: float(p) for a, p in probs.items()}}


def who_do_i_sound_like(text):
    """Gradio callback: returns (author probabilities, explanation markdown)."""
    facts = analyze(text)
    if facts is None:
        return {}, "Please paste some text."
    probs = facts.pop("probs")
    explanation = EXPLAINER.explain(**facts)
    if explanation is None:                  # LLM unavailable -> plain fallback, demo never breaks
        explanation = (f"**You write most like {facts['author']}** ({facts['top_matches']})\n\n"
                       f"Style habits behind this match:\n{facts['habits']}\n\n"
                       "_(AI explanation unavailable - showing raw analysis.)_")
    if facts["n_words"] < CFG["app"]["warn_below_words"]:
        explanation += f"\n\n⚠️ Only {facts['n_words']} words - results get more reliable above ~200."
    return probs, explanation


def build_app():
    import gradio as gr
    return gr.Interface(
        fn=who_do_i_sound_like,
        inputs=gr.Textbox(lines=12, label="Paste your writing (200+ words works best)"),
        outputs=[gr.Label(num_top_classes=3, label="Closest authors"), gr.Markdown(label="Why")],
        title="Who does your writing sound like?",
        description="Style-based authorship attribution (classical NLP) + an AI explanation (LLM via Groq).",
        flagging_mode="never",
    )


if __name__ == "__main__":
    build_app().launch()
