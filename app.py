# AI Sport Elemző Program (Asztalitenisz)
# V4: Deep learning, odds-elemzés, játékosprofil, videómozgás-előkészítés + TTSwing integráció

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import cv2

# ------------------------------
# 1. Adatbetöltés / Előkészítés
# ------------------------------

def load_match_data(file_path):
    if not os.path.exists(file_path):
        st.error("Adatfájl nem található.")
        return None
    df = pd.read_csv(file_path)
    return df

def load_ttswing_data(file_path):
    if not os.path.exists(file_path):
        st.warning("TTSwing adatfájl nem található.")
        return None
    df = pd.read_csv(file_path)
    return df

# ------------------------------
# 2. Elemzés / Vizualizáció
# ------------------------------

def plot_win_distribution(df):
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='gyoztes', ax=ax)
    ax.set_title("Győztesek eloszlása (0 = vendég, 1 = hazai)")
    st.pyplot(fig)

def plot_ttswing_summary(df):
    st.write("🎾 TTSwing adatok összegzése:")
    st.write(df.describe())
    fig, ax = plt.subplots()
    sns.histplot(df['ax_mean'], bins=30, kde=True, ax=ax)
    ax.set_title("Ütések gyorsulása X-tengely mentén")
    st.pyplot(fig)

# ------------------------------
# 3. Klasszikus ML modell
# ------------------------------

def train_rf_model(df):
    X = df[["hazai_elozmeny", "vendeg_elozmeny", "hazai_gyozelmek", "vendeg_gyozelmek", "odds_hazai", "odds_vendeg"]]
    y = df["gyoztes"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc

# ------------------------------
# 4. Deep Learning modell
# ------------------------------

def train_dl_model(df):
    X = df[["hazai_elozmeny", "vendeg_elozmeny", "hazai_gyozelmek", "vendeg_gyozelmek", "odds_hazai", "odds_vendeg"]].values
    y = df["gyoztes"].values
    model = Sequential([
        Dense(64, activation='relu', input_shape=(6,)),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=50, batch_size=16, verbose=0)
    loss, acc = model.evaluate(X, y, verbose=0)
    return model, acc

# ------------------------------
# 5. Predikciós függvények
# ------------------------------

def predict_match_rf(model, features):
    input_data = np.array([features])
    pred = model.predict(input_data)
    prob = model.predict_proba(input_data)
    return pred[0], prob[0][1], prob[0][0]

def predict_match_dl(model, features):
    input_data = np.array([features])
    prob = model.predict(input_data)[0][0]
    return int(prob > 0.5), prob, 1 - prob

# ------------------------------
# 6. Streamlit Alkalmazás
# ------------------------------

def main():
    st.set_page_config(page_title="AI Asztalitenisz Elemző", layout="centered")
    st.title("🏓 AI Asztalitenisz Meccs Elemző")

    file_path = "asztalitenisz_meccsek.csv"
    df = load_match_data(file_path)

    swing_path = "ttswing_data.csv"
    swing_df = load_ttswing_data(swing_path)

    if df is not None:
        st.subheader("📊 Alap statisztikák")
        st.write(df.describe())
        plot_win_distribution(df)

        st.subheader("🧠 Modell Tanítása")
        with st.spinner("Tanítás folyamatban..."):
            rf_model, rf_acc = train_rf_model(df)
            dl_model, dl_acc = train_dl_model(df)
        st.success(f"Random Forest pontosság: {rf_acc*100:.2f}%")
        st.success(f"Deep Learning pontosság: {dl_acc*100:.2f}%")

        st.subheader("🔮 Meccs Predikció")
        hazai_forma = st.slider("Hazai forma (utóbbi győzelmek)", 0, 10, 5)
        vendeg_forma = st.slider("Vendég forma (utóbbi győzelmek)", 0, 10, 3)
        hazai_gy = st.number_input("Hazai összes győzelem", min_value=0, value=25)
        vendeg_gy = st.number_input("Vendég összes győzelem", min_value=0, value=20)
        odds_h = st.number_input("Hazai odds", min_value=1.0, value=1.80)
        odds_v = st.number_input("Vendég odds", min_value=1.0, value=2.00)

        features = [hazai_forma, vendeg_forma, hazai_gy, vendeg_gy, odds_h, odds_v]

        if st.button("Eredmény előrejelzése"):
            pred_rf, prob_h_rf, prob_v_rf = predict_match_rf(rf_model, features)
            pred_dl, prob_h_dl, prob_v_dl = predict_match_dl(dl_model, features)

            st.info(f"🧠 RF győztes: {'Hazai' if pred_rf == 1 else 'Vendég'} ({prob_h_rf*100:.1f}% vs {prob_v_rf*100:.1f}%)")
            st.info(f"🤖 DL győztes: {'Hazai' if pred_dl == 1 else 'Vendég'} ({prob_h_dl*100:.1f}% vs {prob_v_dl*100:.1f}%)")

    if swing_df is not None:
        st.subheader("🏓 TTSwing Ütéselemzés")
        plot_ttswing_summary(swing_df)

    st.subheader("📹 Videó Mozgásfeldolgozás (kísérleti)")
    st.markdown("Ez a funkció előkészíti a videók mozgásanalízisét asztaliteniszhez.")
    st.code(\"\"\"import cv2
cap = cv2.VideoCapture('meccs.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # OpenCV mozgás vagy labdakövetés itt
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()\"\"\", language='python')

if __name__ == "__main__":
    main()
