# 🌌 Kepler Exoplanet Detection & Classification Dashboard

## 🔭 Overview
This project applies supervised machine learning to automate the vetting of NASA Kepler Objects of Interest (KOIs). It features a highly accurate **Random Forest Classifier** designed to distinguish genuine, confirmed exoplanets from false positives (such as eclipsing binary stars and instrumental noise) using purely physical and geometric transit parameters.

This system was developed as a final-year project for the Bachelor of Science (Physics Honours) program at Maharishi Markandeshwar (Deemed to be University).

## ✨ Features
* **AI Inference Engine:** Select any Kepler ID to dynamically fetch its physical parameters and run a real-time neural vetting classification.
* **Astrophysical Feature Engine:** Utilizes 12 derived metrics including Transit Model SNR, Planetary Radius, and Orbital Period.
* **Global Dataset Analytics:** Interactive Plotly visualizations mapping the physical boundaries between planets and false positives.
* **End-to-End Pipeline:** Includes data imputation, feature scaling via `StandardScaler`, and class-weighted balancing.

## 🚀 How to Run Locally
To run this application on your own machine:

1. Clone the repository:
   ```bash
 git clone https://github.com/udit454/Kepler-Exoplanet-Detection.git
