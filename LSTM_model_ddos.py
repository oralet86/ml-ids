import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

# Sabitler: ihtiyaç halinde buradan düzenleyin
CSV_PATH = "/Users/cemresudeakdag/Downloads/archive/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
LABEL_COLUMN = "Label"
TEST_SIZE = 0.1
K_FOLDS = 8
RANDOM_STATE = 42
VERBOSE = False

# Gini importance'a göre en önemli 22 özellik
IMPORTANT_FEATURES = [
    "Avg Fwd Segment Size",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Max",
    "Init_Win_bytes_forward",
    "Fwd IAT Std",
    "act_data_pkt_fwd",
    "Total Length of Fwd Packets",
    "Subflow Fwd Bytes",
    "Bwd Packet Length Mean",
    "Avg Bwd Segment Size",
    "Bwd Packet Length Max",
    "Fwd Header Length.1",
    "Destination Port",
    "Bwd Packet Length Std",
    "Subflow Fwd Packets",
    "Fwd IAT Total",
    "Total Fwd Packets",
    "Fwd Packet Length Std",
    "Total Backward Packets",
    "Subflow Bwd Packets",
    "Fwd Header Length",
    "Fwd IAT Max"
]

def build_lstm_model(hp, input_dim, timesteps=1):
    """Hyperparameter tuning için LSTM model builder"""
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(timesteps, input_dim)))
    
    # LSTM katman sayısını optimize et (1-3 arası)
    n_lstm_layers = hp.Int('n_lstm_layers', min_value=1, max_value=3, step=1)
    
    for i in range(n_lstm_layers):
        # Her LSTM katmanı için unit sayısını optimize et
        units = hp.Int(f'lstm_units_{i}', min_value=32, max_value=256, step=32)
        
        # Son LSTM katmanı hariç return_sequences=True
        return_seq = (i < n_lstm_layers - 1)
        model.add(keras.layers.LSTM(units, return_sequences=return_seq))
        
        # Dropout oranını optimize et
        dropout = hp.Float(f'lstm_dropout_{i}', min_value=0.2, max_value=0.5, step=0.1)
        model.add(keras.layers.Dropout(dropout))
    
    # Dense katman sayısını optimize et (0-2 arası)
    n_dense_layers = hp.Int('n_dense_layers', min_value=0, max_value=2, step=1)
    
    for i in range(n_dense_layers):
        units = hp.Int(f'dense_units_{i}', min_value=16, max_value=128, step=16)
        model.add(keras.layers.Dense(units, activation='relu'))
        
        dropout = hp.Float(f'dense_dropout_{i}', min_value=0.2, max_value=0.5, step=0.1)
        model.add(keras.layers.Dropout(dropout))
    
    # Çıkış katmanı
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    
    # Learning rate'i optimize et
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def preprocess_and_model_lstm(csv_path: str, label_column: str = "Label", test_size: float = 0.2, k_folds: int = 5, random_state: int = 42, verbose: bool = False):
    # Veriyi oku
    df = pd.read_csv(csv_path, low_memory=False)
    
    # Sütun isimlerini temizle (boşluk, özel karakterler)
    df.columns = df.columns.str.strip()
    
    if not verbose:
        tf.get_logger().setLevel("ERROR")
    np.random.seed(random_state)
    tf.random.set_seed(random_state)

    # NaN ve inf değerlerini çıkar
    before_len = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if verbose:
        print(f"Yüklenen satır: {before_len}, temizlenen satır: {before_len - len(df)}, kalan: {len(df)}")

    # Etiket encode: BENIGN -> 0, diğerleri -> 1
    label_column = label_column.strip()
    if label_column not in df.columns:
        raise ValueError(f"Etiket sütunu bulunamadı: {label_column}. Mevcut sütunlar: {df.columns.tolist()}")
    y = (df[label_column].astype(str) != "BENIGN").astype(int)
    if verbose:
        vc = y.value_counts().to_dict()
        print(f"Sınıf dağılımı (0=BENIGN,1=ATTACK): {vc}")

    # Özellikleri seç (etiket sütunu hariç, sayısal kolonlar)
    X = df.drop(columns=[label_column])
    X = X.select_dtypes(include=[np.number])
    if X.empty:
        raise ValueError("Sayısal özellik bulunamadı.")
    
    # Önemli özellikleri filtrele
    available_features = [f for f in IMPORTANT_FEATURES if f.strip() in X.columns]
    if len(available_features) == 0:
        if verbose:
            print(f"Aranan özellikler: {IMPORTANT_FEATURES}")
            print(f"Mevcut sütunlar: {X.columns.tolist()}")
        raise ValueError("Önemli özelliklerden hiçbiri veri setinde bulunamadı.")
    
    available_features_stripped = [f.strip() for f in available_features]
    X = X[available_features_stripped]
    
    if verbose:
        print(f"Kullanılan özellik sayısı: {X.shape[1]}/{len(IMPORTANT_FEATURES)}")
        print(f"Kullanılan özellikler: {available_features_stripped}")

    # CHRONOLOGICAL SPLIT: shuffle=False, stratify=None
    # İlk (1-test_size) train, son test_size test
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.values[:split_idx], X.values[split_idx:]
    y_train, y_test = y.values[:split_idx], y.values[split_idx:]
    
    if verbose:
        print(f"\nChronological split yapıldı:")
        print(f"Train boyutu: {len(X_train)}, Test boyutu: {len(X_test)}")
        print(f"Train sınıf dağılımı: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        print(f"Test sınıf dağılımı: {dict(zip(*np.unique(y_test, return_counts=True)))}")

    # MinMaxScaler - SADECE TRAIN'E FIT ET
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Sadece transform

    # SMOTE ile dengeleme (sadece train üzerinde)
    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
    if verbose:
        unique, counts = np.unique(y_train_bal, return_counts=True)
        print(f"SMOTE sonrası dağılım: {dict(zip(unique.tolist(), counts.tolist()))}")

    # LSTM için reshape: (samples, timesteps, features)
    X_train_bal_lstm = X_train_bal.reshape(X_train_bal.shape[0], 1, X_train_bal.shape[1])
    X_test_lstm = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

    # CHRONOLOGICAL K-FOLD: TimeSeriesSplit kullan
    # TimeSeriesSplit sıralı fold'lar oluşturur
    tscv = TimeSeriesSplit(n_splits=k_folds)
    cv_accs = []
    
    # Hyperparameter tuning için ilk fold'u kullan
    first_fold = True
    best_hps = None
    
    for train_idx, val_idx in tscv.split(X_train_bal):
        X_tr, X_val = X_train_bal_lstm[train_idx], X_train_bal_lstm[val_idx]
        y_tr, y_val = y_train_bal[train_idx], y_train_bal[val_idx]

        if first_fold:
            # Hyperparameter tuning
            print("\nLSTM Hyperparameter tuning başlıyor (TimeSeriesSplit ile)...")
            tuner = kt.Hyperband(
                lambda hp: build_lstm_model(hp, X_tr.shape[2], timesteps=1),
                objective='val_accuracy',
                max_epochs=30,
                factor=3,
                directory='tuner_results_lstm',
                project_name='lstm_ddos_tuning'
            )
            
            stop_early = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)
            
            tuner.search(
                X_tr, y_tr,
                epochs=30,
                batch_size=256,
                validation_data=(X_val, y_val),
                callbacks=[stop_early],
                verbose=0
            )
            
            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            
            print(f"\nEn iyi hyperparametreler:")
            print(f"  LSTM katman sayısı: {best_hps.get('n_lstm_layers')}")
            for i in range(best_hps.get('n_lstm_layers')):
                print(f"  LSTM Katman {i+1} - Units: {best_hps.get(f'lstm_units_{i}')}, Dropout: {best_hps.get(f'lstm_dropout_{i}'):.2f}")
            print(f"  Dense katman sayısı: {best_hps.get('n_dense_layers')}")
            for i in range(best_hps.get('n_dense_layers')):
                print(f"  Dense Katman {i+1} - Units: {best_hps.get(f'dense_units_{i}')}, Dropout: {best_hps.get(f'dense_dropout_{i}'):.2f}")
            print(f"  Learning rate: {best_hps.get('learning_rate'):.6f}\n")
            
            first_fold = False
        
        # En iyi hyperparametrelerle model oluştur ve eğit
        model = build_lstm_model(best_hps, X_tr.shape[2], timesteps=1)
        es = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)
        model.fit(X_tr, y_tr, epochs=20, batch_size=256, validation_data=(X_val, y_val), verbose=0, callbacks=[es])

        val_preds = (model.predict(X_val, verbose=0).ravel() >= 0.5).astype(int)
        cv_accs.append(accuracy_score(y_val, val_preds))

    print(f"K-fold doğrulama ortalama doğruluk ({k_folds}-fold TimeSeriesSplit): {np.mean(cv_accs):.4f} ± {np.std(cv_accs):.4f}")

    # Tüm dengelenmiş train üzerinde final modeli en iyi hyperparametrelerle eğit
    final_model = build_lstm_model(best_hps, X_train_bal_lstm.shape[2], timesteps=1)
    
    # Validation split için sıralı bölme (son %10)
    val_split_idx = int(len(X_train_bal_lstm) * 0.9)
    X_train_final = X_train_bal_lstm[:val_split_idx]
    y_train_final = y_train_bal[:val_split_idx]
    X_val_final = X_train_bal_lstm[val_split_idx:]
    y_val_final = y_train_bal[val_split_idx:]
    
    es_final = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)
    
    print("\nFinal LSTM model eğitimi başlıyor (chronological validation)...")
    final_model.fit(
        X_train_final, y_train_final,
        epochs=30,
        batch_size=256,
        validation_data=(X_val_final, y_val_final),
        verbose=1 if verbose else 0,
        callbacks=[es_final]
    )

    # Test değerlendirme
    test_preds = (final_model.predict(X_test_lstm, verbose=0).ravel() >= 0.5).astype(int)
    print("\nTest Accuracy:", f"{accuracy_score(y_test, test_preds):.4f}")
    print("Classification Report:\n", classification_report(y_test, test_preds, target_names=["BENIGN(0)", "ATTACK(1)"]))
    
    return final_model, best_hps

def main():
    preprocess_and_model_lstm(
        csv_path=CSV_PATH,
        label_column=LABEL_COLUMN,
        test_size=TEST_SIZE,
        k_folds=K_FOLDS,
        random_state=RANDOM_STATE,
        verbose=VERBOSE,
    )

if __name__ == "__main__":
    main()
