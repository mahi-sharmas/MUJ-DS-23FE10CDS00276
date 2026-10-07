"""The demo: paste your writing -> closest classic author + an explanation
(the classifier finds the style habits, the LLM explains them)."""
import json
from pathlib import Path

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


LEAF = ('<svg viewBox="0 0 24 24" width="{s}" height="{s}"><path d="M12 2C6.5 6 4.5 12 6.5 18'
        'c1.6 2.6 5.5 4 5.5 4s3.9-1.4 5.5-4C19.5 12 17.5 6 12 2z" fill="{c}"/></svg>')
SPINE_COLOURS = ["#9B3F6E", "#B4604F", "#C58A4A", "#7E3460", "#8E4A62",
                 "#6E2C55", "#A35A58", "#7A3B5C", "#94506F", "#6B2F57"]


def shelf_html(probs):
    """Draw every author as a book spine: the taller the spine, the closer the match."""
    ranked = sorted(probs.items(), key=lambda kv: -kv[1])
    top_name, top_p = ranked[0]
    spines = []
    for i, (name, p) in enumerate(ranked):
        height = 86 + 136 * (p / top_p)
        width = [56, 46, 42][i] if i < 3 else 30
        label = name if i == 0 else name.split()[-1]
        pct = f"<span class='pct'>{p:.0%}</span>" if i < 3 else ""
        ribbon = "<span class='ribbon'></span>" if i == 0 else ""
        cls = "spine top" if i == 0 else ("spine near" if i < 3 else "spine")
        spines.append(f"<div class='{cls}' style='height:{height:.0f}px;width:{width}px;"
                      f"background:{SPINE_COLOURS[i % len(SPINE_COLOURS)]}'>{ribbon}"
                      f"<span class='title'>{label}</span>{pct}</div>")
    strength = "a clear match" if top_p >= 0.45 else ("a fair match" if top_p >= 0.30 else "a tentative match")
    return (f"<div class='bookplate'><p class='ex'>Ex libris</p><p class='who'>{top_name}</p>"
            f"<p class='how'>{strength} · {top_p:.0%}</p></div>"
            f"<p class='label'>Taller spine · closer match</p>"
            f"<div class='shelf'>{''.join(spines)}</div><div class='plank'></div>")


EMPTY_SHELF = ("<div class='bookplate empty'><p class='ex'>Ex libris</p><p class='who'>· · ·</p>"
               "<p class='how'>your kindred author appears here</p></div>"
               "<p class='label'>Taller spine · closer match</p>"
               "<div class='shelf'>" + "".join(
                   f"<div class='spine ghost' style='height:{h}px;width:{w}px'></div>"
                   for h, w in [(150, 40), (120, 34), (170, 44), (100, 30), (135, 36), (90, 28), (115, 32)])
               + "</div><div class='plank'></div>")
EMPTY_WHY = "*Paste your writing and press **Find my author**.*"


def who_do_i_sound_like(text):
    """Gradio callback: returns (bookshelf HTML, explanation markdown)."""
    facts = analyze(text)
    if facts is None:
        return EMPTY_SHELF, "Please paste some text first."
    probs = facts.pop("probs")
    explanation = EXPLAINER.explain(**facts)
    if explanation is None:                  # LLM unavailable -> plain fallback, demo never breaks
        explanation = (f"**Style habits behind this match:**\n{facts['habits']}\n\n"
                       "_(AI explanation unavailable - showing the raw analysis.)_")
    explanation = explanation.replace("\n*Tip:", "\n\n*Tip:")   # keep the tip out of the bullet list
    if facts["n_words"] < CFG["app"]["warn_below_words"]:
        explanation += f"\n\n*Only {facts['n_words']} words - results get more reliable above ~200.*"
    return shelf_html(probs), explanation


def word_count_note(text):
    """Live word counter shown under the text box."""
    n = len(text.split())
    if n == 0:
        return "<p class='wc'>0 words</p>"
    if n < CFG["app"]["warn_below_words"]:
        return f"<p class='wc'>{n} words · a little more makes the reading surer</p>"
    return f"<p class='wc'>{n} words · a good long read</p>"


def how_it_works_html():
    """One-line model summary for the footer (reads the saved metrics)."""
    acc = ""
    metrics_file = Path(CFG["model"]["results_dir"]) / "metrics.json"
    if metrics_file.exists():
        m = json.loads(metrics_file.read_text())["style_plus_tfidf"]
        acc = f" · {m['accuracy']:.0%} right on books it has never seen"
    return (f"<div id='how'><span class='h'>How it works</span>"
            f"<span>91 style measurements · trained on 40 novels by {len(AUTHOR_NAMES)} authors{acc}"
            f" · explained by an LLM ({CFG['llm']['model']})</span></div>")


SAMPLE_CLASSIC = (
    "No one who had ever seen Catherine Morland in her infancy would have supposed her born to be an heroine. "
    "Her situation in life, the character of her father and mother, her own person and disposition, were all "
    "equally against her. Her father was a clergyman, without being neglected, or poor, and a very respectable man, "
    "though his name was Richard--and he had never been handsome. He had a considerable independence besides two "
    "good livings--and he was not in the least addicted to locking up his daughters. Her mother was a woman of "
    "useful plain sense, with a good temper, and, what is more remarkable, with a good constitution. She had three "
    "sons before Catherine was born; and instead of dying in bringing the latter into the world, as anybody might "
    "expect, she still lived on--lived to have six children more--to see them growing up around her, and to enjoy "
    "excellent health herself. A family of ten children will be always called a fine family, where there are heads "
    "and arms and legs enough for the number; but the Morlands had little other right to the word, for they were in "
    "general very plain, and Catherine, for many years of her life, as plain as any.")

SAMPLE_MODERN = (
    "I didn't plan to stay up this late, but here we are. The project was supposed to be simple: take some old "
    "books, teach a computer to recognise who wrote them, and build a little app around it. Simple, right? Then the "
    "model names changed, the API went down, and a dash turned out to be three different characters depending on "
    "who typed the book. Honestly, I think I learned more from the things that broke than from the things that "
    "worked. That's probably the real lesson here. You plan, you test, something odd happens, and you figure out "
    "why. Then you fix it and move on. My friends think I'm mad for caring about semicolons at midnight, and maybe "
    "they're right. But there's something satisfying about watching a program read a paragraph and say, with a "
    "straight face, that you write like a Victorian novelist. I'm not sure whether that's a compliment or an "
    "insult. Either way, I'm going to finish this, push it to GitHub, and sleep for about twelve hours.")

HERO = (
    "<div id='hero'>"
    + "".join(f"<span class='leaf' style='left:{x}%;animation-duration:{d}s;animation-delay:-{dl}s'>"
              + LEAF.format(s=sz, c=c) + "</span>"
              for x, d, dl, sz, c in [(6, 19, 3, 22, "#EFA07A"), (24, 23, 12, 18, "#D98C9A"),
                                      (61, 17, 7, 24, "#D8A657"), (83, 21, 15, 20, "#C8694C"),
                                      (93, 25, 1, 16, "#EFA07A")])
    + "<p class='kicker'>A stylometry reading room</p>"
      "<h1>Who does your writing <em>sound like?</em></h1>"
      "<div class='rule'><span></span>❦<span></span></div>"
      "<p class='sub'>Pour a cup of tea, paste a page of your writing, and meet the classic novelist "
      "whose voice is closest to yours.</p></div>")

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,500;0,600;1,500&family=Lora:ital,wght@0,400;0,600;1,400&display=swap');

/* ---------- page ---------- */
body, gradio-app {
  background:
    radial-gradient(circle at 12% 55%, rgba(216,166,87,.30), transparent 32%),
    radial-gradient(circle at 88% 70%, rgba(239,160,122,.28), transparent 34%),
    radial-gradient(circle at 50% 40%, rgba(155,63,110,.45), transparent 30%),
    radial-gradient(ellipse 90% 60% at 50% 0%, #5B2340 0%, #3A1530 55%, #220C1D 100%) !important;
  background-attachment: fixed !important; background-color: #220C1D !important;
}
.gradio-container {
  --body-text-color: #F7E9D7 !important; --block-background-fill: transparent !important;
  --block-border-width: 0px !important; --block-shadow: none !important;
  --background-fill-primary: transparent !important; --background-fill-secondary: transparent !important;
  max-width: 1180px !important; margin: auto !important; position: relative;
  background: transparent !important; font-family: 'Lora', Georgia, serif !important; color: #F7E9D7 !important;
}
.gradio-container > * { position: relative; z-index: 1; }
footer { display: none !important; }

/* ---------- hero ---------- */
#hero { position: relative; text-align: center; padding: 46px 12px 30px; overflow: visible; }
#hero .kicker { margin: 0; font-family: 'Cormorant Garamond', serif; font-size: 15px; letter-spacing: .32em;
                text-transform: uppercase; color: #D8A657; }
#hero h1 { margin: 10px auto 0; max-width: 820px; font-family: 'Cormorant Garamond', serif; font-weight: 600;
           font-size: clamp(38px, 6vw, 62px); line-height: 1.05; color: #F7E9D7; }
#hero h1 em { font-weight: 500; color: #EFA07A; }
#hero .rule { display: flex; justify-content: center; align-items: center; gap: 14px; color: #D8A657; margin: 14px 0; }
#hero .rule span { display: block; width: 90px; height: 1px; background: rgba(216,166,87,.6); }
#hero .sub { margin: 0 auto; max-width: 560px; font-style: italic; font-size: 18px; color: #E3C9BC; }
@keyframes leaf-fall { 0% { transform: translate(0,-40px) rotate(0); opacity: 0 } 10% { opacity: .9 }
                       100% { transform: translate(80px, 1100px) rotate(540deg); opacity: 0 } }
#hero .leaf { position: absolute; top: 0; animation: leaf-fall linear infinite; pointer-events: none; }
@media (prefers-reduced-motion: reduce) { #hero .leaf { animation: none; opacity: .6 } }

/* ---------- glass panels ---------- */
.glass { background: rgba(255,238,220,.08) !important; border: 1px solid rgba(255,222,190,.22) !important;
         border-radius: 18px !important; padding: 26px 28px !important;
         backdrop-filter: blur(22px) saturate(140%); -webkit-backdrop-filter: blur(22px) saturate(140%);
         box-shadow: 0 20px 50px rgba(20,4,16,.45), inset 0 1px 0 rgba(255,240,225,.18) !important; }
.glass .block, .glass .form, .glass .wrap { background: transparent !important; border: none !important; box-shadow: none !important; }
.panel-head { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; }
.panel-head h2 { margin: 0; font-family: 'Cormorant Garamond', serif; font-weight: 600; font-size: 30px; color: #F7E9D7; }
.panel-head span { font-family: 'Cormorant Garamond', serif; font-size: 14px; letter-spacing: .22em;
                   text-transform: uppercase; color: #D8A657; }

/* ---------- text box ---------- */
#manuscript textarea {
  background-color: rgba(30,8,24,.45) !important; color: #F7E9D7 !important;
  border: 1px solid rgba(255,222,190,.2) !important; border-radius: 12px !important;
  font-family: 'Lora', Georgia, serif !important; font-size: 16px !important; line-height: 28px !important;
  padding: 14px 18px !important;
  background-image: repeating-linear-gradient(180deg, transparent 0 27px, rgba(216,166,87,.13) 27px 28px) !important;
  background-attachment: local !important;
}
#manuscript textarea::placeholder { color: #A98B95 !important; }
#manuscript textarea:focus { outline: 2px solid #D8A657 !important; }
.wc { margin: 4px 2px 0; font-size: 14px; color: #D8A657; }

/* ---------- buttons ---------- */
.ghost-btn { background: transparent !important; color: #F3C3A6 !important; border: 1px solid rgba(239,160,122,.6) !important;
             border-radius: 10px !important; font-family: 'Lora', Georgia, serif !important; box-shadow: none !important; }
.ghost-btn:hover { background: rgba(239,160,122,.12) !important; }
#clear-btn { background: rgba(255,238,220,.08) !important; color: #E3C9BC !important;
             border: 1px solid rgba(255,222,190,.2) !important; border-radius: 10px !important; box-shadow: none !important; }
#find-btn { background: #D8A657 !important; color: #2A0F24 !important; border: none !important; border-radius: 10px !important;
            font-family: 'Cormorant Garamond', serif !important; font-weight: 600 !important; font-size: 21px !important;
            letter-spacing: .04em; box-shadow: 0 6px 20px rgba(216,166,87,.35) !important; }
#find-btn:hover { background: #E7BC72 !important; }

/* ---------- results: bookplate + shelf ---------- */
.inner-frame { border: 1px solid rgba(216,166,87,.45); border-radius: 12px; padding: 18px 20px 6px; }
.bookplate { margin: 6px auto 18px; width: fit-content; text-align: center; padding: 14px 36px 12px;
             border: 1px solid rgba(216,166,87,.6); border-radius: 50% / 24%;
             box-shadow: inset 0 0 0 5px rgba(30,8,24,.2), inset 0 0 0 6px rgba(239,160,122,.4); }
.bookplate .ex { margin: 0; font-family: 'Cormorant Garamond', serif; font-size: 13px; letter-spacing: .4em;
                 text-transform: uppercase; color: #D8A657; }
.bookplate .who { margin: 2px 0 0; font-family: 'Cormorant Garamond', serif; font-style: italic; font-size: 44px;
                  line-height: 1.1; color: #F0C987; }
.bookplate .how { margin: 4px 0 0; font-size: 14px; color: #E3C9BC; }
.bookplate.empty .who { color: rgba(240,201,135,.5); }
.label { margin: 0 0 8px; font-family: 'Cormorant Garamond', serif; font-size: 13px; letter-spacing: .3em;
         text-transform: uppercase; color: #D8A657; }
.shelf { display: flex; align-items: flex-end; justify-content: center; gap: 6px; height: 232px; padding: 0 6px; }
.spine { position: relative; box-sizing: border-box; border-radius: 4px 4px 2px 2px; padding: 8px 0;
         display: flex; flex-direction: column; align-items: center; justify-content: space-between;
         box-shadow: inset 3px 0 0 rgba(255,235,215,.12), inset -3px 0 0 rgba(20,4,16,.2);
         transform-origin: bottom; animation: rise .7s ease-out both; }
.spine .title { writing-mode: vertical-rl; transform: rotate(180deg); font-family: 'Cormorant Garamond', serif;
                font-weight: 600; font-size: 13px; color: #F3D9C9; white-space: nowrap; overflow: hidden; }
.spine.near .title { font-size: 15px; color: #FFE9D8; }
.spine.top { box-shadow: inset 4px 0 0 rgba(255,235,215,.16), inset -4px 0 0 rgba(20,4,16,.22), 0 0 26px rgba(216,166,87,.5); }
.spine.top .title { font-size: 19px; color: #FBE3B4; letter-spacing: .05em; }
.spine .pct { font-size: 11px; color: #FFE9D8; }
.spine.top .pct { font-size: 12px; color: #FBE3B4; }
.spine .ribbon { position: absolute; top: -6px; right: 10px; width: 9px; height: 34px; background: #EFA07A;
                 clip-path: polygon(0 0, 100% 0, 100% 100%, 50% 80%, 0 100%); }
.spine.ghost { background: rgba(255,238,220,.07); border: 1px dashed rgba(255,222,190,.2); animation: none; }
@keyframes rise { from { transform: scaleY(0); opacity: 0 } to { transform: scaleY(1); opacity: 1 } }
.plank { height: 12px; margin: 0 -4px 6px; border-radius: 3px; background: #8A5238;
         box-shadow: 0 6px 14px rgba(20,4,16,.45), inset 0 2px 0 rgba(255,225,190,.22); }

/* ---------- explanation ---------- */
#why { background: rgba(30,8,24,.38) !important; border: 1px solid rgba(255,222,190,.16) !important;
       border-radius: 12px !important; padding: 16px 20px !important; min-height: 90px; }
#why, #why p, #why li { color: #F7E9D7 !important; font-size: 15px; line-height: 1.65; }
#why strong { color: #F0C987 !important; }
#why em { color: #F3C3A6; }

/* ---------- footer ---------- */
#how { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 6px 24px; align-items: center;
       border-top: 1px solid rgba(255,222,190,.22); border-bottom: 1px solid rgba(255,222,190,.22);
       padding: 14px 4px; margin: 8px 0 30px; font-size: 14px; color: #E3C9BC; }
#how .h { font-family: 'Cormorant Garamond', serif; font-size: 20px; color: #F0C987; }
"""


def build_app():
    import gradio as gr
    theme = gr.themes.Base(font=[gr.themes.GoogleFont("Lora"), "Georgia", "serif"], radius_size="lg")

    with gr.Blocks(theme=theme, css=CSS, title="Who does your writing sound like?") as demo:
        gr.HTML(HERO)
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, elem_classes="glass"):
                gr.HTML("<div class='panel-head'><h2>Your manuscript</h2><span>Page i</span></div>")
                text = gr.Textbox(lines=12, max_lines=22, show_label=False, elem_id="manuscript",
                                  placeholder="Paste 200 or more words: an essay, a story, a long letter...")
                count = gr.HTML(word_count_note(""))
                with gr.Row():
                    classic_btn = gr.Button("Read a classic sample", size="sm", elem_classes="ghost-btn")
                    modern_btn = gr.Button("Read a modern sample", size="sm", elem_classes="ghost-btn")
                with gr.Row():
                    clear_btn = gr.Button("Clear", elem_id="clear-btn", scale=1)
                    find_btn = gr.Button("Find my author", elem_id="find-btn", scale=2)

            with gr.Column(scale=1, elem_classes="glass"):
                gr.HTML("<div class='panel-head'><h2>Your kindred author</h2><span>Ex libris</span></div>")
                shelf = gr.HTML(EMPTY_SHELF, elem_classes="inner-frame")
                why = gr.Markdown(EMPTY_WHY, elem_id="why")

        gr.HTML(how_it_works_html())

        text.change(word_count_note, text, count)
        classic_btn.click(lambda: SAMPLE_CLASSIC, None, text)
        modern_btn.click(lambda: SAMPLE_MODERN, None, text)
        clear_btn.click(lambda: ("", EMPTY_SHELF, EMPTY_WHY), None, [text, shelf, why])
        find_btn.click(who_do_i_sound_like, text, [shelf, why])
    return demo


if __name__ == "__main__":
    build_app().launch()
