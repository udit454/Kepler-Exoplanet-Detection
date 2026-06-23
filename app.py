import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Exoplanet Dashboard",
    page_icon="🪐",
    layout="wide"
)

# ---------------- LOAD DATA AND TRAIN MODEL ----------------

@st.cache_data
def load_and_train_model():

    try:
        df = pd.read_csv("exoplanets_2018.csv")

    except FileNotFoundError:
        st.error(
            "exoplanets_2018.csv not found."
        )
        st.stop()

    # Remove candidates
    df = df[df['koi_disposition'] != 'CANDIDATE']

    # Target variable
    df['target'] = df['koi_disposition'].map(
        {
            'CONFIRMED': 1,
            'FALSE POSITIVE': 0
        }
    )

    features = [
        'koi_period',
        'koi_impact',
        'koi_duration',
        'koi_depth',
        'koi_prad',
        'koi_teq',
        'koi_insol',
        'koi_model_snr',
        'koi_steff',
        'koi_slogg',
        'koi_srad',
        'koi_kepmag'
    ]

    df_clean = df.dropna(
        subset=features + ['target', 'kepoi_name']
    )

    X = df_clean[features]
    y = df_clean['target']

    kepler_ids = sorted(
        df_clean['kepoi_name'].unique()
    )

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced'
    )

    rf_model.fit(X_scaled, y)

    return (
        df_clean,
        rf_model,
        scaler,
        features,
        kepler_ids
    )


df_clean, rf_model, scaler, features, kepler_ids = load_and_train_model()

# ---------------- TITLE ----------------

st.title("🪐 Exoplanet Detection and Classification Dashboard")

st.markdown("---")

# ---------------- TABS ----------------

tab1, tab2 = st.tabs(
    [
        "🔍 Prediction",
        "📈 Dataset Analysis"
    ]
)

# ==================================================
# TAB 1
# ==================================================

with tab1:

    selected_id = st.selectbox(
        "Select Kepler Object",
        kepler_ids
    )

    if st.button(
        "Run AI Analysis",
        type="primary"
    ):

        star_data = df_clean[
            df_clean['kepoi_name']
            == selected_id
        ].iloc[0]

        input_features = (
            star_data[features]
            .values
            .reshape(1, -1)
        )

        input_scaled = scaler.transform(
            input_features
        )

        prediction = rf_model.predict(
            input_scaled
        )[0]

        probability = rf_model.predict_proba(
            input_scaled
        )[0]

        st.markdown("## Result")

        if prediction == 1:

            confidence = probability[1] * 100

            st.success(
                "🌍 CONFIRMED EXOPLANET"
            )

        else:

            confidence = probability[0] * 100

            st.error(
                "❌ FALSE POSITIVE"
            )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.progress(
            int(confidence)
        )

        with st.expander(
            "Physical Parameters"
        ):

            st.json(
                {
                    "Orbital Period":
                        float(
                            star_data['koi_period']
                        ),

                    "Transit Depth":
                        float(
                            star_data['koi_depth']
                        ),

                    "Planet Radius":
                        float(
                            star_data['koi_prad']
                        ),

                    "Transit Duration":
                        float(
                            star_data['koi_duration']
                        ),

                    "Signal-to-Noise Ratio":
                        float(
                            star_data['koi_model_snr']
                        ),

                    "Stellar Temperature":
                        float(
                            star_data['koi_steff']
                        )
                }
            )

# ==================================================
# TAB 2
# ==================================================

with tab2:

    fig = px.scatter(
        df_clean,
        x='koi_period',
        y='koi_depth',
        color='koi_disposition',
        log_x=True,
        log_y=True,

        color_discrete_map={
            'CONFIRMED': '#00CC96',
            'FALSE POSITIVE': '#EF553B'
        },

        labels={
            'koi_period':
                'Orbital Period (Days)',

            'koi_depth':
                'Transit Depth (ppm)'
        },

        title='Transit Depth vs Orbital Period'
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )