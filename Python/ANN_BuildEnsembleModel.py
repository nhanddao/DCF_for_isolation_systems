import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from pickle import dump
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers

# ================================================================
# INPUTS
# ================================================================
GMType = "Mix"
ModelVersion = "20261117"
# fd, Td, k1d, GMSetID, TM/T0, y0=DM/D0, TM, betaM, BM
XCol = np.array([0, 1, 2, 6, 7])
# XCol = np.array([1,7])
YCol = np.array([8])

DatFile = f"../Data/BM_Train_{GMType}_HighDM.txt"

# ================================================================
# ANN structure
# ================================================================
n_splits = 10
epochs = 200
patience = 25
l2_reg = 1e-6
import ANN_Best_Params
best_params= ANN_Best_Params.get_best_params(GMType=GMType,num_features=len(XCol))
hidden_units=best_params['hidden_units']
learning_rate= best_params['learning_rate']
batch_size= best_params['batch_size']
weight_decay= best_params['weight_decay']
DirToSaveModels = f"ANN_Models/{GMType}/{len(XCol)}Features_{ModelVersion}"
os.makedirs(DirToSaveModels, exist_ok=True)

# ================================================================
# Load & Scale Data
# ================================================================
AllDat = np.loadtxt(DatFile)
X_raw = AllDat[:, XCol]
y_raw = AllDat[:, YCol].ravel()

# Scalers
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X_raw)
y_scaled = scaler_y.fit_transform(y_raw.reshape(-1, 1)).ravel()

print(f"Data shape: X={X_scaled.shape}, y_scaled={y_scaled.shape}")
print(f"Original y → mean={y_raw.mean():.4f}, std={y_raw.std():.4f}")

# ================================================================
# Model Definition
# ================================================================
def build_model(input_dim):
    inputs = keras.Input(shape=(input_dim,))
    x = inputs
    
    for units in hidden_units:
        x = layers.Dense(
            units,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(l2_reg)
        )(x)
        # BatchNormalization removed (dropout=0.0)
    
    outputs = layers.Dense(1)(x)
    
    model = keras.Model(inputs, outputs)
    
    model.compile(
        optimizer=optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay),
        loss='mse',
        metrics=['mae']
    )
    return model

# ================================================================
# K-Fold Training
# ================================================================
kf = KFold(n_splits=n_splits, shuffle=True, random_state=None)

fold_rmses = []      # on scaled target
fold_rmses_orig = [] # on original scale
models = []
oof_predictions_scaled = np.zeros(len(y_scaled))

print(f"\nStarting {n_splits}-fold training with scaling...\n")

for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled), 1):
    print(f"Fold {fold}/{n_splits}")
    
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y_scaled[train_idx], y_scaled[val_idx]
    
    model = build_model(input_dim=X_scaled.shape[1])
    
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss', patience=patience, restore_best_weights=True, verbose=0
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=12, min_lr=1e-6, verbose=0
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],
        verbose=0,
        shuffle=True
    )
    
    # Predict on validation fold (scaled)
    y_pred_scaled = model.predict(X_val, batch_size=batch_size*2, verbose=0).ravel()
    
    # RMSE on scaled target
    rmse_scaled = np.sqrt(mean_squared_error(y_val, y_pred_scaled))
    
    # RMSE on original scale
    y_val_orig = scaler_y.inverse_transform(y_val.reshape(-1, 1)).ravel()
    y_pred_orig = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    rmse_orig = np.sqrt(mean_squared_error(y_val_orig, y_pred_orig))
    
    r2 = r2_score(y_val, y_pred_scaled)
    
    fold_rmses.append(rmse_scaled)
    fold_rmses_orig.append(rmse_orig)
    oof_predictions_scaled[val_idx] = y_pred_scaled
    models.append(model)
    
    print(f"   Scaled RMSE: {rmse_scaled:.5f} | Orig RMSE: {rmse_orig:.5f} | R²: {r2:.4f}")

# ================================================================
# Final Results
# ================================================================
print("\n" + "="*70)
print("FINAL CROSS-VALIDATION RESULTS")
print("="*70)
print(f"Scaled   RMSE : {np.mean(fold_rmses):.5f} ± {np.std(fold_rmses):.5f}")
print(f"Original RMSE : {np.mean(fold_rmses_orig):.5f} ± {np.std(fold_rmses_orig):.5f}")
print(f"OOF R²        : {r2_score(y_scaled, oof_predictions_scaled):.4f}")
print("="*70)

# ================================================================
# Save Scalers and Models
# ================================================================
dump(scaler_X, open(f"{DirToSaveModels}/InputScaler.pkl", 'wb'))
dump(scaler_y, open(f"{DirToSaveModels}/OutputScaler.pkl", 'wb'))

for i, model in enumerate(models, 1):
    model.save(f"{DirToSaveModels}/ANNModel_fold{i}.keras")

print(f"\nModels and scalers saved to: {DirToSaveModels}")

# ================================================================
# Plot OOF Predictions (Original Scale)
# ================================================================
oof_pred_orig = scaler_y.inverse_transform(oof_predictions_scaled.reshape(-1, 1)).ravel()

plt.figure(figsize=(8, 8))
plt.scatter(oof_pred_orig, y_raw, alpha=0.6, s=4)
plt.plot([y_raw.min(), y_raw.max()], [y_raw.min(), y_raw.max()], 'r--', lw=2)
plt.xlabel("Predicted Value")
plt.ylabel("True Value")
plt.title("Out-of-Fold Predictions (Original Scale)")
plt.grid(True, alpha=0.3)
plt.show()