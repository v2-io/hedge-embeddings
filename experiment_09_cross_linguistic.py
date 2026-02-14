"""
Experiment 09: Cross-Linguistic Probability Geometry

The strongest possible test of whether the probability axis is a property
of language universally (not just English): train the axis on English
Mosteller data, then project hedge expressions from German, French,
Spanish, and Chinese onto it — zero-shot, no language-specific calibration.

If "wahrscheinlich" (German "probably") projects near 70% on an axis it
was never trained on, the structure is language-agnostic.

Uses bge-m3 (XLM-RoBERTa, 100+ languages, 1024d) as the multilingual
embedding model.

Also tests: train axis on German/French/etc. expressions independently,
check alignment with the English axis.

Usage:
  python3 experiment_09_cross_linguistic.py [model_name]
  Default model: bge-m3
"""

import sys
import numpy as np
import requests
from scipy import stats

MODEL = sys.argv[1] if len(sys.argv) > 1 else "bge-m3"
OLLAMA_URL = "http://localhost:11434/api/embed"


def embed_texts(texts, model=MODEL):
    resp = requests.post(OLLAMA_URL, json={"model": model, "input": texts})
    resp.raise_for_status()
    return np.array(resp.json()["embeddings"], dtype=np.float64)


def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def train_axis(diffs, medians, lam=0.1):
    medians_c = (medians - medians.mean()) / medians.std()
    dim = diffs.shape[1]
    w = np.linalg.solve(diffs.T @ diffs + lam * np.eye(dim), diffs.T @ medians_c)
    w = normalize(w)
    projections = diffs @ w
    slope, intercept, _, _, _ = stats.linregress(projections, medians)
    return w, slope, intercept


def loo_evaluate(diffs, medians):
    n = len(medians)
    loo_preds = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        w, slope, intercept = train_axis(diffs[mask], medians[mask])
        loo_preds[i] = float(np.clip(slope * (diffs[i] @ w) + intercept, 0, 100))
    r, _ = stats.spearmanr(loo_preds, medians)
    mae = np.mean(np.abs(loo_preds - medians))
    return r, mae


# ═══════════════════════════════════════════════════════════════════════
# English (Mosteller training data)
# ═══════════════════════════════════════════════════════════════════════

EN_PREDICATIVE = {
    "certain": 99.6, "almost certain": 90.2, "very likely": 87.5,
    "likely": 71.1, "probable": 70.2, "very probable": 89.7,
    "possible": 38.5, "unlikely": 17.2, "very unlikely": 5.0,
    "improbable": 12.5, "very improbable": 4.8, "impossible": 0.3,
}
EN_PRED_TEMPLATE = "It is {PHRASE} that the experiment will succeed"
EN_BARE = "The experiment will succeed"

EN_MODAL = {
    "certainly": 99.6, "almost certainly": 90.2, "probably": 70.2,
    "likely": 71.1, "possibly": 38.5, "unlikely": 17.2,
    "very unlikely": 5.0, "definitely": 99.6, "perhaps": 38.5,
    "undoubtedly": 95.0,
}
EN_MODAL_TEMPLATE = "The experiment will {PHRASE} succeed"


# ═══════════════════════════════════════════════════════════════════════
# German
# ═══════════════════════════════════════════════════════════════════════

DE_PREDICATIVE = {
    "sicher": 99.6,            # certain
    "fast sicher": 90.2,       # almost certain
    "sehr wahrscheinlich": 87.5,  # very likely
    "wahrscheinlich": 71.1,    # likely/probable
    "möglich": 38.5,           # possible
    "unwahrscheinlich": 17.2,  # unlikely
    "sehr unwahrscheinlich": 5.0,  # very unlikely
    "unmöglich": 0.3,          # impossible
}
DE_PRED_TEMPLATE = "Es ist {PHRASE}, dass das Experiment gelingen wird"
DE_BARE = "Das Experiment wird gelingen"

DE_MODAL = {
    "sicherlich": 99.6,        # certainly
    "wahrscheinlich": 70.2,    # probably
    "möglicherweise": 38.5,    # possibly
    "vielleicht": 38.5,        # perhaps/maybe
    "vermutlich": 70.2,        # presumably
    "bestimmt": 95.0,          # definitely
    "zweifellos": 95.0,        # undoubtedly
}
DE_MODAL_TEMPLATE = "Das Experiment wird {PHRASE} gelingen"


# ═══════════════════════════════════════════════════════════════════════
# French
# ═══════════════════════════════════════════════════════════════════════

FR_PREDICATIVE = {
    "certain": 99.6,            # certain
    "presque certain": 90.2,    # almost certain
    "très probable": 87.5,      # very likely
    "probable": 71.1,           # likely/probable
    "possible": 38.5,           # possible
    "improbable": 17.2,         # unlikely
    "très improbable": 5.0,     # very unlikely
    "impossible": 0.3,          # impossible
}
FR_PRED_TEMPLATE = "Il est {PHRASE} que l'expérience réussira"
FR_BARE = "L'expérience réussira"

FR_MODAL = {
    "certainement": 99.6,      # certainly
    "probablement": 70.2,      # probably
    "peut-être": 38.5,         # perhaps
    "possiblement": 38.5,      # possibly
    "vraisemblablement": 70.2, # presumably
    "sans doute": 75.0,        # without doubt (but actually ~probably in French)
    "indubitablement": 95.0,   # undoubtedly
}
FR_MODAL_TEMPLATE = "L'expérience va {PHRASE} réussir"


# ═══════════════════════════════════════════════════════════════════════
# Spanish
# ═══════════════════════════════════════════════════════════════════════

ES_PREDICATIVE = {
    "seguro": 99.6,             # certain
    "casi seguro": 90.2,        # almost certain
    "muy probable": 87.5,       # very likely
    "probable": 71.1,           # likely/probable
    "posible": 38.5,            # possible
    "improbable": 17.2,         # unlikely
    "muy improbable": 5.0,      # very unlikely
    "imposible": 0.3,           # impossible
}
ES_PRED_TEMPLATE = "Es {PHRASE} que el experimento tenga éxito"
ES_BARE = "El experimento tendrá éxito"

ES_MODAL = {
    "ciertamente": 99.6,        # certainly
    "probablemente": 70.2,      # probably
    "posiblemente": 38.5,       # possibly
    "quizás": 38.5,             # perhaps
    "tal vez": 38.5,            # maybe
    "seguramente": 85.0,        # surely
    "indudablemente": 95.0,     # undoubtedly
}
ES_MODAL_TEMPLATE = "El experimento {PHRASE} tendrá éxito"


# ═══════════════════════════════════════════════════════════════════════
# Chinese (Mandarin)
# ═══════════════════════════════════════════════════════════════════════

ZH_PREDICATIVE = {
    "确定的": 99.6,        # certain (quèdìng de)
    "几乎确定的": 90.2,    # almost certain
    "很可能的": 87.5,      # very likely
    "可能的": 71.1,        # likely/probable (kěnéng de)
    "有可能的": 38.5,      # possible
    "不太可能的": 17.2,    # unlikely
    "非常不可能的": 5.0,   # very unlikely
    "不可能的": 0.3,       # impossible
}
ZH_PRED_TEMPLATE = "实验成功是{PHRASE}"
ZH_BARE = "实验会成功"

ZH_MODAL = {
    "肯定": 99.6,          # certainly (kěndìng)
    "大概": 65.0,          # probably/roughly (dàgài)
    "可能": 50.0,          # possibly/maybe (kěnéng)
    "也许": 38.5,          # perhaps (yěxǔ)
    "或许": 38.5,          # perhaps (huòxǔ)
    "一定": 95.0,          # definitely (yīdìng)
    "八成": 80.0,          # 80% / very likely (colloquial, bā chéng)
}
ZH_MODAL_TEMPLATE = "实验{PHRASE}会成功"


# ═══════════════════════════════════════════════════════════════════════
# Japanese
# ═══════════════════════════════════════════════════════════════════════

JA_PREDICATIVE = {
    "確実な": 99.6,        # certain (kakujitsu na)
    "ほぼ確実な": 90.2,    # almost certain
    "非常にありそうな": 87.5,  # very likely
    "ありそうな": 71.1,    # likely (arisō na)
    "あり得る": 38.5,      # possible (arieru)
    "ありそうにない": 17.2,  # unlikely
    "非常にありそうにない": 5.0,  # very unlikely
    "不可能な": 0.3,       # impossible (fukanō na)
}
JA_PRED_TEMPLATE = "実験が成功するのは{PHRASE}ことだ"
JA_BARE = "実験は成功する"

JA_MODAL = {
    "きっと": 95.0,        # surely/definitely (kitto)
    "確かに": 90.0,        # certainly (tashika ni)
    "おそらく": 70.2,      # probably (osoraku)
    "たぶん": 65.0,        # probably/maybe (tabun)
    "もしかしたら": 38.5,   # possibly/perhaps (moshikashitara)
    "ひょっとしたら": 30.0,  # by some chance (hyotto shitara)
    "まさか": 5.0,         # surely not / no way (masaka) — low probability
}
JA_MODAL_TEMPLATE = "実験は{PHRASE}成功する"


# ═══════════════════════════════════════════════════════════════════════
# Korean
# ═══════════════════════════════════════════════════════════════════════

KO_PREDICATIVE = {
    "확실한": 99.6,        # certain (hwaksilhan)
    "거의 확실한": 90.2,    # almost certain
    "매우 가능성이 높은": 87.5,  # very likely
    "가능성이 높은": 71.1,  # likely
    "가능한": 38.5,        # possible (ganeunghan)
    "가능성이 낮은": 17.2,  # unlikely
    "매우 가능성이 낮은": 5.0,  # very unlikely
    "불가능한": 0.3,       # impossible (bulganeunghan)
}
KO_PRED_TEMPLATE = "실험이 성공하는 것은 {PHRASE} 일이다"
KO_BARE = "실험은 성공할 것이다"

KO_MODAL = {
    "분명히": 95.0,        # clearly/certainly (bunmyeonghi)
    "확실히": 99.6,        # certainly (hwaksilhi)
    "아마": 70.2,          # probably (ama)
    "아마도": 65.0,        # probably (amado)
    "혹시": 38.5,          # possibly/perhaps (hoksi)
    "어쩌면": 38.5,        # perhaps/maybe (eojjeomyeon)
}
KO_MODAL_TEMPLATE = "실험은 {PHRASE} 성공할 것이다"


# ═══════════════════════════════════════════════════════════════════════
# Arabic
# ═══════════════════════════════════════════════════════════════════════

AR_PREDICATIVE = {
    "مؤكد": 99.6,            # certain (mu'akkad)
    "شبه مؤكد": 90.2,        # almost certain
    "مرجح جداً": 87.5,       # very likely
    "مرجح": 71.1,            # likely (murajjaḥ)
    "ممكن": 38.5,            # possible (mumkin)
    "غير مرجح": 17.2,        # unlikely
    "غير مرجح جداً": 5.0,    # very unlikely
    "مستحيل": 0.3,           # impossible (mustaḥīl)
}
AR_PRED_TEMPLATE = "من {PHRASE} أن التجربة ستنجح"
AR_BARE = "التجربة ستنجح"

AR_MODAL = {
    "بالتأكيد": 99.6,        # certainly (bi-t-ta'kīd)
    "على الأرجح": 75.0,      # most likely ('ala al-arjaḥ)
    "ربما": 50.0,            # perhaps/maybe (rubbamā)
    "من المحتمل": 65.0,      # probably (min al-muḥtamal)
    "قد": 45.0,              # might/may (qad) — modal particle
    "بالكاد": 10.0,          # hardly (bi-l-kād)
}
AR_MODAL_TEMPLATE = "{PHRASE} ستنجح التجربة"


# ═══════════════════════════════════════════════════════════════════════
# Hindi
# ═══════════════════════════════════════════════════════════════════════

HI_PREDICATIVE = {
    "निश्चित": 99.6,           # certain (niśchit)
    "लगभग निश्चित": 90.2,     # almost certain
    "बहुत संभावित": 87.5,      # very likely
    "संभावित": 71.1,           # likely (sambhāvit)
    "संभव": 38.5,             # possible (sambhav)
    "असंभावित": 17.2,          # unlikely
    "बहुत असंभावित": 5.0,      # very unlikely
    "असंभव": 0.3,             # impossible (asambhav)
}
HI_PRED_TEMPLATE = "प्रयोग सफल होना {PHRASE} है"
HI_BARE = "प्रयोग सफल होगा"

HI_MODAL = {
    "निश्चित रूप से": 99.6,    # certainly
    "शायद": 50.0,              # perhaps/maybe (śāyad)
    "संभवतः": 65.0,            # probably (sambhavataḥ)
    "कदाचित": 30.0,            # perhaps/perchance (kadācit)
    "ज़रूर": 90.0,              # surely (zarūr)
    "मुश्किल से": 10.0,         # hardly (muśkil se)
}
HI_MODAL_TEMPLATE = "प्रयोग {PHRASE} सफल होगा"


# ═══════════════════════════════════════════════════════════════════════
# Language configs
# ═══════════════════════════════════════════════════════════════════════

LANGUAGES = {
    "English": {
        "predicative": (EN_PREDICATIVE, EN_PRED_TEMPLATE, EN_BARE),
        "modal": (EN_MODAL, EN_MODAL_TEMPLATE, EN_BARE),
    },
    "German": {
        "predicative": (DE_PREDICATIVE, DE_PRED_TEMPLATE, DE_BARE),
        "modal": (DE_MODAL, DE_MODAL_TEMPLATE, DE_BARE),
    },
    "French": {
        "predicative": (FR_PREDICATIVE, FR_PRED_TEMPLATE, FR_BARE),
        "modal": (FR_MODAL, FR_MODAL_TEMPLATE, FR_BARE),
    },
    "Spanish": {
        "predicative": (ES_PREDICATIVE, ES_PRED_TEMPLATE, ES_BARE),
        "modal": (ES_MODAL, ES_MODAL_TEMPLATE, ES_BARE),
    },
    "Chinese": {
        "predicative": (ZH_PREDICATIVE, ZH_PRED_TEMPLATE, ZH_BARE),
        "modal": (ZH_MODAL, ZH_MODAL_TEMPLATE, ZH_BARE),
    },
    "Japanese": {
        "predicative": (JA_PREDICATIVE, JA_PRED_TEMPLATE, JA_BARE),
        "modal": (JA_MODAL, JA_MODAL_TEMPLATE, JA_BARE),
    },
    "Korean": {
        "predicative": (KO_PREDICATIVE, KO_PRED_TEMPLATE, KO_BARE),
        "modal": (KO_MODAL, KO_MODAL_TEMPLATE, KO_BARE),
    },
    "Arabic": {
        "predicative": (AR_PREDICATIVE, AR_PRED_TEMPLATE, AR_BARE),
        "modal": (AR_MODAL, AR_MODAL_TEMPLATE, AR_BARE),
    },
    "Hindi": {
        "predicative": (HI_PREDICATIVE, HI_PRED_TEMPLATE, HI_BARE),
        "modal": (HI_MODAL, HI_MODAL_TEMPLATE, HI_BARE),
    },
}


def embed_group(expressions, template, bare_text):
    """Embed a group of expressions and return (diffs, medians, phrases)."""
    phrases = list(expressions.keys())
    medians = np.array([expressions[p] for p in phrases])
    bare_emb = embed_texts([bare_text])[0]
    sentences = [template.replace("{PHRASE}", p) for p in phrases]
    embs = embed_texts(sentences)
    diffs = embs - bare_emb
    return diffs, medians, phrases, bare_emb


def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  Experiment 09: Cross-Linguistic Probability Geometry              ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print(f"║  Model: {MODEL:<58s}║")
    print("║                                                                    ║")
    print("║  Question: Does the English probability axis work for German,      ║")
    print("║  French, Spanish, and Chinese hedge expressions?                   ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    try:
        test = embed_texts(["test"])
        dim = test.shape[1]
        print(f"Model loaded. Dim: {dim}")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # ═══════════════════════════════════════════════════════════════════
    # Step 1: Train axes on English, verify they work
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "═" * 78)
    print("STEP 1: ENGLISH BASELINE (train and verify)")
    print("═" * 78)

    en_results = {}
    en_axes = {}
    for frame in ["predicative", "modal"]:
        exprs, template, bare = LANGUAGES["English"][frame]
        diffs, medians, phrases, bare_emb = embed_group(exprs, template, bare)
        w, sl, intc = train_axis(diffs, medians)
        preds = np.clip(sl * (diffs @ w) + intc, 0, 100)
        r_is, _ = stats.spearmanr(preds, medians)
        r_loo, mae_loo = loo_evaluate(diffs, medians)

        en_axes[frame] = (w, sl, intc)
        en_results[frame] = {"r_is": r_is, "r_loo": r_loo, "mae_loo": mae_loo}
        print(f"\n  English {frame}: in-sample ρ = {r_is:.3f}, "
              f"LOO ρ = {r_loo:.3f}, LOO MAE = {mae_loo:.1f}%")

    # ═══════════════════════════════════════════════════════════════════
    # Step 2: Zero-shot transfer — project other languages onto English axis
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "═" * 78)
    print("STEP 2: ZERO-SHOT CROSS-LINGUISTIC TRANSFER")
    print("  (English-trained axis applied to other languages)")
    print("═" * 78)

    cross_results = {}

    for lang_name, lang_data in LANGUAGES.items():
        if lang_name == "English":
            continue

        print(f"\n  {'─'*60}")
        print(f"  {lang_name}")
        print(f"  {'─'*60}")

        cross_results[lang_name] = {}

        for frame in ["predicative", "modal"]:
            if frame not in lang_data:
                continue

            exprs, template, bare = lang_data[frame]
            diffs, medians, phrases, bare_emb = embed_group(exprs, template, bare)

            # Project onto ENGLISH axis (zero-shot)
            en_w, en_sl, en_intc = en_axes[frame]
            preds = np.clip(en_sl * (diffs @ en_w) + en_intc, 0, 100)
            r_zero, p_zero = stats.spearmanr(preds, medians)
            mae_zero = np.mean(np.abs(preds - medians))

            cross_results[lang_name][frame] = {
                "zero_shot_rho": r_zero, "zero_shot_mae": mae_zero,
                "zero_shot_p": p_zero,
            }

            print(f"\n  {frame} (n={len(phrases)}):")
            print(f"    Zero-shot ρ = {r_zero:+.3f} (p = {p_zero:.2e}), "
                  f"MAE = {mae_zero:.1f}%")
            print(f"    Per-phrase:")
            sorted_idx = np.argsort(medians)[::-1]
            for i in sorted_idx:
                err = abs(preds[i] - medians[i])
                print(f"      {phrases[i]:25s}  Pred={preds[i]:5.1f}%  "
                      f"Expected={medians[i]:5.1f}%  Err={err:4.1f}%")

    # ═══════════════════════════════════════════════════════════════════
    # Step 3: Within-language axes and alignment with English
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "═" * 78)
    print("STEP 3: WITHIN-LANGUAGE AXES AND ALIGNMENT")
    print("  (Train per-language axis, measure cosine with English axis)")
    print("═" * 78)

    for lang_name, lang_data in LANGUAGES.items():
        if lang_name == "English":
            continue

        print(f"\n  {lang_name}:")

        for frame in ["predicative", "modal"]:
            if frame not in lang_data:
                continue

            exprs, template, bare = lang_data[frame]
            diffs, medians, phrases, bare_emb = embed_group(exprs, template, bare)

            # Train axis on THIS language's data
            w_lang, sl_lang, intc_lang = train_axis(diffs, medians)
            preds_lang = np.clip(sl_lang * (diffs @ w_lang) + intc_lang, 0, 100)
            r_lang, _ = stats.spearmanr(preds_lang, medians)

            # LOO within this language
            r_loo, mae_loo = loo_evaluate(diffs, medians)

            # Alignment with English axis
            en_w = en_axes[frame][0]
            cos_align = abs(float(en_w @ w_lang))

            cross_results[lang_name][frame].update({
                "within_rho": r_lang,
                "within_loo_rho": r_loo,
                "within_loo_mae": mae_loo,
                "axis_alignment": cos_align,
            })

            print(f"    {frame}:")
            print(f"      Within-language: ρ = {r_lang:.3f}, "
                  f"LOO ρ = {r_loo:.3f}, LOO MAE = {mae_loo:.1f}%")
            print(f"      Axis alignment with English: cos = {cos_align:.3f}")

    # ═══════════════════════════════════════════════════════════════════
    # Step 4: Cross-language axis transfer (German axis → French, etc.)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "═" * 78)
    print("STEP 4: CROSS-LANGUAGE AXIS PAIRWISE ALIGNMENT (predicative)")
    print("═" * 78)

    lang_axes = {"English": en_axes.get("predicative", (None,))[0]}
    for lang_name, lang_data in LANGUAGES.items():
        if lang_name == "English" or "predicative" not in lang_data:
            continue
        exprs, template, bare = lang_data["predicative"]
        diffs, medians, phrases, bare_emb = embed_group(exprs, template, bare)
        w, _, _ = train_axis(diffs, medians)
        lang_axes[lang_name] = w

    # Pairwise alignment matrix
    lang_names = list(lang_axes.keys())
    print(f"\n  {'':12s}", end="")
    for name in lang_names:
        print(f"  {name[:6]:>6s}", end="")
    print()
    print(f"  {'─'*12}", end="")
    for _ in lang_names:
        print(f"  {'─'*6}", end="")
    print()

    for i, name_i in enumerate(lang_names):
        print(f"  {name_i:12s}", end="")
        for j, name_j in enumerate(lang_names):
            if i == j:
                print(f"  {'1.000':>6s}", end="")
            else:
                cos = abs(float(lang_axes[name_i] @ lang_axes[name_j]))
                print(f"  {cos:5.3f}", end="")
        print()

    # ═══════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "═" * 78)
    print("SUMMARY")
    print("═" * 78)

    print(f"\n  English baseline (bge-m3):")
    for frame, res in en_results.items():
        print(f"    {frame}: LOO ρ = {res['r_loo']:.3f}, MAE = {res['mae_loo']:.1f}%")

    print(f"\n  Zero-shot transfer (English axis → other languages):")
    print(f"  {'Language':<12s}  {'Frame':<12s}  {'Zero-shot ρ':>12s}  {'MAE':>6s}  {'Axis align':>10s}")
    print(f"  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*6}  {'─'*10}")
    for lang_name, frames in cross_results.items():
        for frame, res in frames.items():
            align = res.get("axis_alignment", "—")
            align_str = f"{align:.3f}" if isinstance(align, float) else align
            print(f"  {lang_name:<12s}  {frame:<12s}  {res['zero_shot_rho']:+11.3f}  "
                  f"{res['zero_shot_mae']:5.1f}%  {align_str:>10s}")

    # Verdict
    print()
    zero_rhos = [res[frame]["zero_shot_rho"]
                 for lang, res in cross_results.items()
                 for frame in res]
    mean_zero = np.mean([abs(r) for r in zero_rhos])
    alignments = [res[frame].get("axis_alignment", 0)
                  for lang, res in cross_results.items()
                  for frame in res if "axis_alignment" in res[frame]]
    mean_align = np.mean(alignments) if alignments else 0

    if mean_zero > 0.7:
        print(f"  STRONG TRANSFER: Mean zero-shot |ρ| = {mean_zero:.3f}")
        print(f"  The English probability axis works for other languages.")
    elif mean_zero > 0.5:
        print(f"  MODERATE TRANSFER: Mean zero-shot |ρ| = {mean_zero:.3f}")
        print(f"  Partial cross-linguistic structure; some language-specific variation.")
    else:
        print(f"  WEAK TRANSFER: Mean zero-shot |ρ| = {mean_zero:.3f}")
        print(f"  The axis is largely language-specific.")

    if mean_align > 0.5:
        print(f"  Mean axis alignment (cos): {mean_align:.3f} — axes point in similar directions.")
    elif mean_align > 0.3:
        print(f"  Mean axis alignment (cos): {mean_align:.3f} — partial directional overlap.")
    else:
        print(f"  Mean axis alignment (cos): {mean_align:.3f} — axes are largely independent.")

    print()


if __name__ == "__main__":
    main()
