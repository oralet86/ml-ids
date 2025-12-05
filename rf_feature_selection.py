import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report

# Sabitler: ihtiyaç halinde buradan düzenleyin
CSV_PATH = "/Users/cemresudeakdag/Downloads/archive/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
LABEL_COLUMN = "Label"
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 300
USE_PERMUTATION = False
TOP_K = 30
OUTPUT_CSV = None
VERBOSE = True  # eklendi

def compute_feature_importance(
    csv_path: str,
    label_column: str = "Label",
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 300,
    use_permutation: bool = False,
    top_k: int = 30,
    output_csv: str | None = None,
    verbose: bool = True,
):
    # Veriyi oku
    df = pd.read_csv(csv_path, low_memory=False)

    # Kolon isimlerini normalize ederek gerçek etiket kolonunu bul
    col_norm = {c: c.strip() for c in df.columns}
    df.rename(columns=col_norm, inplace=True)
    normalized = [c.strip().lower() for c in df.columns]
    target_norm = label_column.strip().lower()
    try:
        label_col_actual = df.columns[normalized.index(target_norm)]
    except ValueError:
        if verbose:
            print("Mevcut kolonlar:", list(df.columns)[:50])
        raise ValueError(f"Etiket sütunu bulunamadı: {label_column}")

    # NaN/inf temizliği
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Etiket: BENIGN -> 0, diğerleri -> 1
    y = (df[label_col_actual].astype(str).str.strip() != "BENIGN").astype(int)

    # Sadece sayısal feature’lar
    X = df.drop(columns=[label_col_actual]).select_dtypes(include=[np.number])

    if X.empty:
        raise ValueError("Sayısal özellik bulunamadı.")

    # Sabit (tekil değerli) kolonları at
    nunique = X.nunique()
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        X = X.drop(columns=constant_cols)

    if X.shape[1] == 0:
        raise ValueError("Sabit olmayan sayısal özellik bulunamadı.")

    if verbose:
        benign_ratio = (y == 0).mean()
        print(f"Özellik sayısı: {X.shape[1]} | Örnek sayısı: {len(X)} | BENIGN oranı: {benign_ratio:.3f}")

    # Stratified train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=test_size, stratify=y.values, random_state=random_state
    )

    # RandomForest (dengesizlik için class_weight='balanced')
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        n_jobs=-1,
        random_state=random_state,
        class_weight="balanced",
        oob_score=False,
        bootstrap=True,
    )
    rf.fit(X_train, y_train)

    # Test metriği (hızlı kontrol)
    y_pred = rf.predict(X_test)
    if verbose:
        print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("Classification Report:\n", classification_report(y_test, y_pred, target_names=["BENIGN(0)", "ATTACK(1)"]))

    # Gini önemleri
    gini_importances = rf.feature_importances_
    feat_names = X.columns.tolist()
    imp_df = pd.DataFrame({
        "feature": feat_names,
        "gini_importance": gini_importances
    }).sort_values("gini_importance", ascending=False).reset_index(drop=True)

    # Permütasyon önemi (opsiyonel, daha yavaş)
    if use_permutation:
        perm = permutation_importance(
            rf, X_test, y_test, n_repeats=5, n_jobs=-1, random_state=random_state, scoring="accuracy"
        )
        imp_df["permutation_importance_mean"] = pd.Series(perm.importances_mean, index=feat_names)[imp_df["feature"]].values
        imp_df["permutation_importance_std"] = pd.Series(perm.importances_std, index=feat_names)[imp_df["feature"]].values
        # Permütasyon varsa ona göre sırala, yoksa Gini'ye göre
        imp_df = imp_df.sort_values(
            by="permutation_importance_mean", ascending=False, na_position="last"
        ).reset_index(drop=True)
    else:
        imp_df = imp_df.sort_values(by="gini_importance", ascending=False).reset_index(drop=True)

    # Sonuçların özeti
    print("\nEn önemli özellikler:")
    cols_to_show = ["feature", "gini_importance"]
    if use_permutation:
        cols_to_show += ["permutation_importance_mean", "permutation_importance_std"]
    print(imp_df.loc[: top_k - 1, cols_to_show].to_string(index=False))

    # Dışa aktarım
    if output_csv:
        imp_df.to_csv(output_csv, index=False)
        print(f"\nÖnem değerleri kaydedildi: {output_csv}")

def main():
    # Argparse kaldırıldı; sabitlerden çağır
    compute_feature_importance(
        csv_path=CSV_PATH,
        label_column=LABEL_COLUMN,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        n_estimators=N_ESTIMATORS,
        use_permutation=USE_PERMUTATION,
        top_k=TOP_K,
        output_csv=OUTPUT_CSV,
        verbose=VERBOSE,
    )

if __name__ == "__main__":
    main()