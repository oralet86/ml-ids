import os
import sys
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

# Sabitler: ihtiyaç halinde buradan düzenleyin
CSV_PATH = "/Users/cemresudeakdag/Downloads/archive/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
LABEL_COLUMN = "Label"
TEST_SIZE = 0.2
K_FOLDS = 5
RANDOM_STATE = 42
VERBOSE = False

# Hyperparameter sonuçları için kayıt dizini
HYPERPARAMS_SAVE_DIR = "/Users/cemresudeakdag/CICIDS2017/saved_hyperparams"
HYPERPARAMS_FILE = os.path.join(HYPERPARAMS_SAVE_DIR, "best_hyperparameters_ann.json")

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

def build_ann_model(hp, input_dim):
    """Hyperparameter tuning için ANN model builder"""
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(input_dim,)))
    
    # Katman sayısını optimize et (1-4 arası)
    n_layers = hp.Int('n_layers', min_value=1, max_value=4, step=1)
    
    for i in range(n_layers):
        # Her katman için unit sayısını optimize et
        units = hp.Int(f'units_{i}', min_value=16, max_value=512, step=16)
        
        # Aktivasyon fonksiyonunu optimize et
        activation = hp.Choice(f'activation_{i}', values=['relu', 'tanh', 'sigmoid'])
        model.add(keras.layers.Dense(units, activation=activation))
        
        # Batch Normalization ekle (opsiyonel)
        use_bn = hp.Boolean(f'use_bn_{i}')
        if use_bn:
            model.add(keras.layers.BatchNormalization())
        
        # Dropout oranını optimize et
        dropout = hp.Float(f'dropout_{i}', min_value=0.1, max_value=0.5, step=0.1)
        model.add(keras.layers.Dropout(dropout))
    
    # Çıkış katmanı
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    
    # Learning rate'i optimize et
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    
    # Optimizer seçimi
    optimizer_choice = hp.Choice('optimizer', values=['adam', 'rmsprop', 'sgd'])
    if optimizer_choice == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_choice == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def print_detailed_metrics(y_true, y_pred, y_pred_proba=None, dataset_name="Dataset"):
    """Print detailed classification metrics including confusion matrix"""
    print(f"\n{'='*60}")
    print(f"{dataset_name} - Detaylı Metrikler")
    print(f"{'='*60}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                 BENIGN  ATTACK")
    print(f"Actual BENIGN    {cm[0][0]:6d}  {cm[0][1]:6d}")
    print(f"       ATTACK    {cm[1][0]:6d}  {cm[1][1]:6d}")
    
    # Classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"\nSınıflandırma Metrikleri:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    # ROC-AUC (if probabilities available)
    roc_auc = None
    if y_pred_proba is not None:
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        print(f"  ROC-AUC:   {roc_auc:.4f}")
    
    # Regression-style metrics
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\nRegresyon Tarzı Metrikler:")
    print(f"  MSE (Mean Squared Error):     {mse:.4f}")
    print(f"  MAE (Mean Absolute Error):    {mae:.4f}")
    print(f"  R² Score:                     {r2:.4f}")
    
    print(f"{'='*60}\n")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'confusion_matrix': cm,
        'roc_auc': roc_auc
    }

def load_and_prepare_data(csv_path: str, label_column: str, feature_list: list, verbose: bool = False):
    """Load and prepare data from CSV file"""
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip()
    
    # NaN ve inf değerlerini çıkar
    before_len = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if verbose:
        print(f"  Yüklenen: {before_len}, temizlenen: {before_len - len(df)}, kalan: {len(df)}")
    
    # Etiket encode
    label_column = label_column.strip()
    if (label_column not in df.columns):
        raise ValueError(f"Etiket sütunu bulunamadı: {label_column}")
    
    y = (df[label_column].astype(str) != "BENIGN").astype(int)
    
    # Özellikleri seç
    X = df.drop(columns=[label_column])
    X = X.select_dtypes(include=[np.number])
    
    # Feature listesindeki özellikleri kullan
    available_features = [f.strip() for f in feature_list if f.strip() in X.columns]
    if len(available_features) == 0:
        raise ValueError("Önemli özelliklerden hiçbiri veri setinde bulunamadı.")
    
    X = X[available_features]
    
    if verbose:
        vc = y.value_counts().to_dict()
        print(f"  Sınıf dağılımı: BENIGN={vc.get(0, 0)}, ATTACK={vc.get(1, 0)}")
        print(f"  Özellik sayısı: {X.shape[1]}/{len(feature_list)}")
    
    return X.values, y.values, available_features

def save_hyperparameters(hyperparams_dict: dict, filepath: str):
    """Save best hyperparameters to JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(hyperparams_dict, f, indent=4)
    print(f"✅ Hyperparametreler kaydedildi: {filepath}")

def load_hyperparameters(filepath: str):
    """Load hyperparameters from JSON file"""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            hyperparams_dict = json.load(f)
        print(f"✅ Kaydedilmiş hyperparametreler yüklendi: {filepath}")
        return hyperparams_dict
    return None

def dict_to_hp_object(hyperparams_dict: dict, input_dim: int):
    """Convert dictionary to keras_tuner HyperParameters object"""
    hp = kt.HyperParameters()
    
    # Set all hyperparameters as fixed values
    hp.Fixed('n_layers', hyperparams_dict['n_layers'])
    hp.Fixed('learning_rate', hyperparams_dict['learning_rate'])
    hp.Fixed('optimizer', hyperparams_dict['optimizer'])
    
    for i in range(hyperparams_dict['n_layers']):
        hp.Fixed(f'units_{i}', hyperparams_dict[f'units_{i}'])
        hp.Fixed(f'activation_{i}', hyperparams_dict[f'activation_{i}'])
        hp.Fixed(f'use_bn_{i}', hyperparams_dict[f'use_bn_{i}'])
        hp.Fixed(f'dropout_{i}', hyperparams_dict[f'dropout_{i}'])
    
    return hp

def hp_object_to_dict(hp):
    """Convert keras_tuner HyperParameters object to dictionary"""
    hyperparams_dict = {
        'n_layers': hp.get('n_layers'),
        'learning_rate': hp.get('learning_rate'),
        'optimizer': hp.get('optimizer')
    }
    
    for i in range(hp.get('n_layers')):
        hyperparams_dict[f'units_{i}'] = hp.get(f'units_{i}')
        hyperparams_dict[f'activation_{i}'] = hp.get(f'activation_{i}')
        hyperparams_dict[f'use_bn_{i}'] = hp.get(f'use_bn_{i}')
        hyperparams_dict[f'dropout_{i}'] = hp.get(f'dropout_{i}')
    
    return hyperparams_dict

def preprocess_and_model_ann(csv_path: str, label_column: str = "Label", test_size: float = 0.2, k_folds: int = 5, random_state: int = 42, verbose: bool = False):
    # Veriyi oku
    if not verbose:
        tf.get_logger().setLevel("ERROR")
    np.random.seed(random_state)
    tf.random.set_seed(random_state)

    print(f"\n{'='*60}")
    print(f"Ana Veri Seti Yükleniyor: {os.path.basename(csv_path)}")
    print(f"{'='*60}")
    
    X, y, available_features = load_and_prepare_data(
        csv_path, label_column, IMPORTANT_FEATURES, verbose
    )

    # CHRONOLOGICAL SPLIT: shuffle=False, stratify=None
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    if verbose:
        print(f"\nChronological split yapıldı:")
        print(f"Train boyutu: {len(X_train)}, Test boyutu: {len(X_test)}")
        print(f"Train sınıf dağılımı: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        print(f"Test sınıf dağılımı: {dict(zip(*np.unique(y_test, return_counts=True)))}")

    # MinMaxScaler - SADECE train verisi üzerinde fit
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if verbose:
        print(f"\nScaler train verisi üzerinde fit edildi.")
        print(f"Train shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}")

    # SMOTE ile dengeleme (sadece train üzerinde)
    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
    if verbose:
        unique, counts = np.unique(y_train_bal, return_counts=True)
        print(f"SMOTE sonrası dağılım: {dict(zip(unique.tolist(), counts.tolist()))}")

    # CHRONOLOGICAL K-FOLD: TimeSeriesSplit kullan
    tscv = TimeSeriesSplit(n_splits=k_folds)
    cv_accs = []
    cv_metrics = []
    
    # Hyperparameter tuning için ilk fold'u kullan
    first_fold = True
    best_hps = None
    
    # Kaydedilmiş hyperparametreler var mı kontrol et
    saved_hyperparams = load_hyperparameters(HYPERPARAMS_FILE)
    
    if saved_hyperparams is not None:
        print("\n🔄 Kaydedilmiş hyperparametreler kullanılıyor, yeniden tuning yapılmayacak.")
        print(f"  Katman sayısı: {saved_hyperparams['n_layers']}")
        print(f"  Optimizer: {saved_hyperparams['optimizer']}")
        for i in range(saved_hyperparams['n_layers']):
            print(f"  Katman {i+1} - Units: {saved_hyperparams[f'units_{i}']}, Activation: {saved_hyperparams[f'activation_{i}']}, BN: {saved_hyperparams[f'use_bn_{i}']}, Dropout: {saved_hyperparams[f'dropout_{i}']:.2f}")
        print(f"  Learning rate: {saved_hyperparams['learning_rate']:.6f}\n")
        
        # Dictionary'den HyperParameters objesine çevir
        best_hps = dict_to_hp_object(saved_hyperparams, X_train_bal.shape[1])
        first_fold = False
    
    for train_idx, val_idx in tscv.split(X_train_bal):
        X_tr, X_val = X_train_bal[train_idx], X_train_bal[val_idx]
        y_tr, y_val = y_train_bal[train_idx], y_train_bal[val_idx]

        if first_fold:
            # Hyperparameter tuning
            print("\n🔍 ANN Hyperparameter tuning başlıyor (TimeSeriesSplit ile)...")
            print("⏳ Bu işlem biraz zaman alabilir...\n")
            
            tuner = kt.Hyperband(
                lambda hp: build_ann_model(hp, X_tr.shape[1]),
                objective='val_accuracy',
                max_epochs=30,
                factor=3,
                directory='tuner_results_ann',
                project_name='ann_ddos_tuning',
                overwrite=True
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
            
            print(f"\n✅ En iyi hyperparametreler bulundu:")
            print(f"  Katman sayısı: {best_hps.get('n_layers')}")
            print(f"  Optimizer: {best_hps.get('optimizer')}")
            for i in range(best_hps.get('n_layers')):
                print(f"  Katman {i+1} - Units: {best_hps.get(f'units_{i}')}, Activation: {best_hps.get(f'activation_{i}')}, BN: {best_hps.get(f'use_bn_{i}')}, Dropout: {best_hps.get(f'dropout_{i}'):.2f}")
            print(f"  Learning rate: {best_hps.get('learning_rate'):.6f}")
            
            # Hyperparametreleri kaydet
            hyperparams_dict = hp_object_to_dict(best_hps)
            save_hyperparameters(hyperparams_dict, HYPERPARAMS_FILE)
            
            first_fold = False
        
        # En iyi hyperparametrelerle model oluştur ve eğit
        model = build_ann_model(best_hps, X_tr.shape[1])
        es = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)
        model.fit(X_tr, y_tr, epochs=20, batch_size=256, validation_data=(X_val, y_val), verbose=0, callbacks=[es])

        val_preds_proba = model.predict(X_val, verbose=0).ravel()
        val_preds = (val_preds_proba >= 0.5).astype(int)
        
        # Collect metrics for this fold
        fold_acc = accuracy_score(y_val, val_preds)
        fold_f1 = f1_score(y_val, val_preds, zero_division=0)
        fold_precision = precision_score(y_val, val_preds, zero_division=0)
        fold_recall = recall_score(y_val, val_preds, zero_division=0)
        
        cv_accs.append(fold_acc)
        cv_metrics.append({
            'accuracy': fold_acc,
            'f1': fold_f1,
            'precision': fold_precision,
            'recall': fold_recall
        })

    # Print CV summary
    print(f"\n{'='*60}")
    print(f"K-Fold Cross Validation Özeti ({k_folds}-fold TimeSeriesSplit)")
    print(f"{'='*60}")
    print(f"Accuracy:  {np.mean(cv_accs):.4f} ± {np.std(cv_accs):.4f}")
    print(f"Precision: {np.mean([m['precision'] for m in cv_metrics]):.4f} ± {np.std([m['precision'] for m in cv_metrics]):.4f}")
    print(f"Recall:    {np.mean([m['recall'] for m in cv_metrics]):.4f} ± {np.std([m['recall'] for m in cv_metrics]):.4f}")
    print(f"F1-Score:  {np.mean([m['f1'] for m in cv_metrics]):.4f} ± {np.std([m['f1'] for m in cv_metrics]):.4f}")
    print(f"{'='*60}\n")

    # Tüm dengelenmiş train üzerinde final modeli en iyi hyperparametrelerle eğit
    final_model = build_ann_model(best_hps, X_train_bal.shape[1])
    
    # Validation split için sıralı bölme (son %10)
    val_split_idx = int(len(X_train_bal) * 0.9)
    X_train_final = X_train_bal[:val_split_idx]
    y_train_final = y_train_bal[:val_split_idx]
    X_val_final = X_train_bal[val_split_idx:]
    y_val_final = y_train_bal[val_split_idx:]
    
    es_final = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)
    
    print("Final ANN model eğitimi başlıyor (chronological validation)...")
    final_model.fit(
        X_train_final, y_train_final,
        epochs=50,
        batch_size=256,
        validation_data=(X_val_final, y_val_final),
        verbose=1 if verbose else 0,
        callbacks=[es_final]
    )

    # Test değerlendirme - Ana veri seti test split'i
    test_preds_proba = final_model.predict(X_test_scaled, verbose=0).ravel()
    test_preds = (test_preds_proba >= 0.5).astype(int)
    
    test_metrics = print_detailed_metrics(
        y_test, 
        test_preds, 
        test_preds_proba,
        dataset_name="Test Set (Ana Veri Seti - ANN)"
    )
    
    print("Detaylı Classification Report:")
    print(classification_report(y_test, test_preds, target_names=["BENIGN(0)", "ATTACK(1)"]))
    
    return final_model, scaler, test_metrics, best_hps

def main():
    preprocess_and_model_ann(
        csv_path=CSV_PATH,
        label_column=LABEL_COLUMN,
        test_size=TEST_SIZE,
        k_folds=K_FOLDS,
        random_state=RANDOM_STATE,
        verbose=VERBOSE
    )

if __name__ == "__main__":
    main()
