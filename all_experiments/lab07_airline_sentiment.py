"""Lab 8: Airline Tweet Sentiment Classification.
Install: python -m pip install pandas numpy scikit-learn matplotlib
Run: python lab08_airline_sentiment.py
Default CSV: all_datasets/lab07-Tweets.csv
"""
import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
    f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay)

LABELS = ["negative", "neutral", "positive"]
SEED = 42

def clean_text(x):
    x = re.sub(r"https?://\S+|www\\.\S+", " URL ", str(x), flags=re.I)
    x = re.sub(r"@\w+", " USER ", x)
    return re.sub(r"\s+", " ", x.lower()).strip()

def vectorizer(ngram):
    return CountVectorizer(stop_words="english", min_df=2,
                           max_features=20000, ngram_range=ngram)

def metrics(name, y, pred):
    p, r, f, _ = precision_recall_fscore_support(
        y, pred, labels=LABELS, average="macro", zero_division=0)
    return {"model": name, "accuracy": accuracy_score(y, pred),
            "macro_precision": p, "macro_recall": r,
            "macro_f1": f1_score(y, pred, labels=LABELS,
                                 average="macro", zero_division=0)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="all_datasets/lab07-Tweets.csv")
    ap.add_argument("--out", default="lab8_results")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.data)

    # Standardise common CSV header variants.
    mapping = {}
    for c in raw.columns:
        k = str(c).strip().lower()
        if k in ["tweet_id", "tweetid", "id"]: mapping[c] = "tweet_id"
        elif k in ["text", "tweet", "content"]: mapping[c] = "text"
        elif k in ["airline_sentiment", "sentiment", "label"]: mapping[c] = "airline_sentiment"
        elif k in ["airline", "airline_name"]: mapping[c] = "airline"
    df = raw.rename(columns=mapping)
    required = ["tweet_id", "text", "airline_sentiment", "airline"]
    missing = [c for c in required if c not in df.columns]
    if missing: raise ValueError(f"Missing columns {missing}. Found: {list(df.columns)}")
    df = df[required].dropna().copy()
    df["tweet_id"] = df["tweet_id"].astype(str)
    df["airline_sentiment"] = df["airline_sentiment"].str.lower().str.strip()
    exact_dupes = int(df["text"].duplicated().sum())
    df["clean_text"] = df["text"].map(clean_text)
    normalised_dupes = int(df["clean_text"].duplicated().sum())
    df = df.drop_duplicates("clean_text").reset_index(drop=True)
    if not set(df.airline_sentiment).issubset(set(LABELS)):
        raise ValueError("Sentiment labels must be negative, neutral, positive.")

    print("\\nQ1 — INSPECT, CLEAN AND DEDUPLICATE")
    print(f"Raw rows: {len(raw)} | exact duplicate texts: {exact_dupes}")
    print(f"Duplicates after normalisation: {normalised_dupes} | retained: {len(df)}")
    print(df.airline_sentiment.value_counts())
    print("Class shares:\\n", df.airline_sentiment.value_counts(normalize=True).round(3))

    # 70% train; split remaining 30% equally into validation and test.
    ix = np.arange(len(df))
    tr, rest = train_test_split(ix, test_size=.30, random_state=SEED,
        stratify=df.airline_sentiment)
    va, te = train_test_split(rest, test_size=.50, random_state=SEED,
        stratify=df.airline_sentiment.iloc[rest])
    split = np.full(len(df), "", dtype=object)
    split[tr], split[va], split[te] = "train", "validation", "test"
    pd.DataFrame({"tweet_id": df.tweet_id, "split": split}).to_csv(out/"split.csv", index=False)
    print("\\nQ2 — SPLIT COUNTS", pd.Series(split).value_counts())
    print(pd.crosstab(split, df.airline_sentiment))

    print("\\nQ3 — VOCABULARY SIZES (FIT TRAIN ONLY)")
    for ng in [(1,1), (1,2)]:
        v = vectorizer(ng); v.fit(df.clean_text.iloc[tr])
        print(f"{ng}: {len(v.vocabulary_):,} features")

    # Q4 majority baseline, evaluated on validation and later test.
    majority = df.airline_sentiment.iloc[tr].mode().iloc[0]
    yv = df.airline_sentiment.iloc[va]
    base_val = metrics("Majority baseline", yv, np.repeat(majority, len(va)))
    print("\\nQ4 — BASELINE", majority, base_val)

    # Q5 tune only on validation macro F1.
    tuning = []
    best = None
    for ng in [(1,1), (1,2)]:
        v = vectorizer(ng); Xtr = v.fit_transform(df.clean_text.iloc[tr])
        Xva = v.transform(df.clean_text.iloc[va])
        for alpha in [.3, .5, 1., 2.]:
            m = MultinomialNB(alpha=alpha).fit(Xtr, df.airline_sentiment.iloc[tr])
            pred = m.predict(Xva)
            score = f1_score(yv, pred, labels=LABELS, average="macro", zero_division=0)
            tuning.append({"ngram_range": str(ng), "alpha": alpha,
                           "validation_macro_f1": score, "features": Xtr.shape[1]})
            if best is None or score > best["score"]:
                best = {"ngram": ng, "alpha": alpha, "score": score}
    tune_df = pd.DataFrame(tuning).sort_values("validation_macro_f1", ascending=False)
    tune_df.to_csv(out/"validation_results.csv", index=False)
    print("\\nQ5 — VALIDATION GRID\\n", tune_df.to_string(index=False))
    print("Selected:", best)

    # Q6: fit final vectorizer/model on training subset, test once.
    v = vectorizer(best["ngram"])
    Xtr = v.fit_transform(df.clean_text.iloc[tr])
    Xte = v.transform(df.clean_text.iloc[te])
    model = MultinomialNB(alpha=best["alpha"]).fit(Xtr, df.airline_sentiment.iloc[tr])
    yt = df.airline_sentiment.iloc[te].to_numpy()
    pred = model.predict(Xte)
    prob = model.predict_proba(Xte)
    confidence = prob.max(axis=1)
    result = pd.DataFrame([
        metrics("Majority baseline", yt, np.repeat(majority, len(te))),
        metrics("Multinomial NB", yt, pred)])
    result.to_csv(out/"metrics.csv", index=False)
    print("\\nQ6 — TEST METRICS\\n", result.to_string(index=False))
    print("\\nPer-class report:\\n", classification_report(
        yt, pred, labels=LABELS, zero_division=0))
    pd.DataFrame(classification_report(yt, pred, labels=LABELS,
        output_dict=True, zero_division=0)).T.to_csv(out/"per_class_metrics.csv")

    cm = confusion_matrix(yt, pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(7,6))
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(
        ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("Multinomial Naive Bayes — Test Confusion Matrix")
    fig.tight_layout(); fig.savefig(out/"confusion_matrix.png", dpi=180); plt.close(fig)
    print("Confusion matrix (rows=true, columns=predicted):\\n",
          pd.DataFrame(cm, index=LABELS, columns=LABELS))

    test = df.iloc[te][["tweet_id","text","clean_text","airline","airline_sentiment"]].copy()
    test["predicted_sentiment"] = pred
    test["confidence"] = confidence
    test.to_csv(out/"test_predictions.csv", index=False)

    # Q7 subgroup analysis.
    airline = []
    for name, g in test.groupby("airline"):
        airline.append({"airline": name, "n_test": len(g),
            "accuracy": accuracy_score(g.airline_sentiment, g.predicted_sentiment),
            "macro_f1": f1_score(g.airline_sentiment, g.predicted_sentiment,
                labels=LABELS, average="macro", zero_division=0)})
    airline_df = pd.DataFrame(airline)
    airline_df.to_csv(out/"per_airline_f1.csv", index=False)
    print("\\nQ7 — AIRLINE SUBGROUP RESULTS\\n", airline_df.to_string(index=False))

    errors = test[test.airline_sentiment != test.predicted_sentiment].copy()
    errors = errors.sort_values(["airline_sentiment","confidence","tweet_id"],
                                ascending=[True,False,True])
    errors = errors.groupby("airline_sentiment", group_keys=False).head(4).copy()
    def error_tag(row):
        txt = row.clean_text
        if re.search(r"\b(not|no|never|n't|cannot|can't)\b", txt): return "negation"
        if "?" in row.text or re.search(r"\b(when|why|how|what|where)\b", txt): return "question_or_request"
        if len(txt.split()) <= 4: return "very_short_text"
        return "ambiguous_or_other"
    errors["error_category"] = errors.apply(error_tag, axis=1)
    errors.to_csv(out/"error_analysis.csv", index=False)
    print("\\nMost confident errors (up to four per true class):\\n",
          errors[["tweet_id","airline_sentiment","predicted_sentiment",
                  "confidence","error_category","text"]].to_string(index=False))

    print(f"\\nCompleted. Outputs saved in: {out.resolve()}")
    print("Files: split.csv, validation_results.csv, metrics.csv, per_class_metrics.csv,")
    print("test_predictions.csv, confusion_matrix.png, per_airline_f1.csv, error_analysis.csv")
    print("Note: scores may vary if your CSV version or preprocessing differs from the slides.")

if __name__ == "__main__":
    main()
