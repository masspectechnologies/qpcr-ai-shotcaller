import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import pickle
import os

class CurveClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(4)
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)

@st.cache_resource
def load_model():
    model = CurveClassifier()
    if os.path.exists('qpcr_model.pth'):
        model.load_state_dict(torch.load('qpcr_model.pth', map_location='cuda'))
        model.to('cuda')
        model.eval()
        st.success("AI model loaded!")
    else:
        st.error("Model file missing – run save in notebook first")
    return model

@st.cache_resource
def load_classes():
    if os.path.exists('classes.pkl'):
        with open('classes.pkl', 'rb') as f:
            return pickle.load(f)
    return ['False Positive', 'Inhibited', 'True Negative', 'Valid Positive']

model = load_model()
classes = load_classes()

st.title("🧬 qPCR AI Shot-Caller Portal")
st.write("Upload raw qPCR data – AI flags issues early!")

uploaded_file = st.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx'])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)
        st.dataframe(df.head(10))

        cycles = df.iloc[:, 0].values
        data = df.iloc[:, 1:].values.astype(float)

        baseline = data[:15].mean(axis=0, keepdims=True)
        data_norm = data - baseline

        tensor = torch.tensor(data_norm, dtype=torch.float32).unsqueeze(1).to('cuda')
        with torch.no_grad():
            outputs = model(tensor)
            preds = outputs.argmax(1).cpu().numpy()
            confs = outputs.softmax(1).max(1).values.cpu().numpy()

        fig, ax = plt.subplots(figsize=(12, 8))
        for i, col in enumerate(df.columns[1:]):
            color = 'green' if 'Valid' in classes[preds[i]] else 'red'
            ax.plot(cycles, data_norm[:, i], label=f"{col}: {classes[preds[i]]} ({confs[i]:.2f})", color=color)

        ax.set_xlabel("Cycle")
        ax.set_ylabel("Fluorescence")
        ax.set_title("AI Results")
        ax.legend(bbox_to_anchor=(1.05, 1))
        ax.grid(True)
        st.pyplot(fig)

        st.write("### Results")
        for i, col in enumerate(df.columns[1:]):
            st.write(f"**{col}**: {classes[preds[i]]} – {confs[i]:.2f} confidence")

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("Upload a file to start!")
    