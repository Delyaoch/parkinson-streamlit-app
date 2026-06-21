import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

st.set_page_config(
    page_title="Early Parkinson's Disease Detection",
    page_icon="🔬",
    layout="wide"
)

HEALTHY_COLOR = "#22C55E"
DISEASE_COLOR = "#EF4444"
BLUE = "#2563EB"
DARK = "#1E293B"

st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #F8FBFF 0%, #F3F8FB 100%);
}

.block-container {
    padding-top: 1.6rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #1E293B;
}

.subtitle {
    color: #64748B;
    font-size: 18px;
    margin-top: -8px;
    margin-bottom: 22px;
}

.hero-card {
    background: linear-gradient(135deg, #E0F2FE 0%, #ECFEFF 100%);
    border: 1px solid #BAE6FD;
    border-radius: 22px;
    padding: 28px 32px;
    margin-bottom: 26px;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
}

.hero-title {
    font-size: 38px;
    font-weight: 800;
    color: #0F172A;
    margin-bottom: 8px;
}

.hero-text {
    font-size: 17px;
    color: #475569;
}

.metric-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 18px;
    padding: 18px 20px;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
}

.metric-label {
    color: #64748B;
    font-size: 14px;
    margin-bottom: 6px;
}

.metric-value {
    color: #0F172A;
    font-size: 34px;
    font-weight: 750;
}

.info-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    padding: 22px;
    border-radius: 18px;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
    margin-bottom: 16px;
}

.success-box {
    background-color: #ECFDF5;
    color: #065F46;
    padding: 18px;
    border-radius: 16px;
    border-left: 6px solid #22C55E;
    font-weight: 600;
}

.warning-box {
    background-color: #FEF2F2;
    color: #991B1B;
    padding: 18px;
    border-radius: 16px;
    border-left: 6px solid #EF4444;
    font-weight: 600;
}

.blue-box {
    background-color: #EFF6FF;
    color: #1D4ED8;
    padding: 16px;
    border-radius: 14px;
    border-left: 5px solid #3B82F6;
    margin-bottom: 18px;
}

hr {
    border: none;
    border-top: 1px solid #E2E8F0;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv("data/parkinsons.data")
    df["patient_id"] = df["name"].str.extract(r"(S\\d+)")
    df["status_label"] = df["status"].map({
        0: "Здоровый",
        1: "Болезнь Паркинсона"
    })
    return df


@st.cache_resource
def load_model():
    return joblib.load("models/rf_model.joblib")


df = load_data()
rf_model = load_model()

feature_columns = [
    col for col in df.columns
    if col not in ["name", "status", "patient_id", "status_label"]
]

tab1, tab2, tab3 = st.tabs([
    "Описание данных и EDA",
    "Модели и сравнение",
    "Инференс"
])


with tab1:
    st.markdown("""
    <div class="hero-card">
        <div class="hero-title">🔬 Early Parkinson's Disease Detection</div>
        <div class="hero-text">
            Анализ голосовых характеристик пациентов и демонстрация модели машинного обучения
            для раннего выявления признаков болезни Паркинсона.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Записей</div>
            <div class="metric-value">{df.shape[0]}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Признаков</div>
            <div class="metric-value">{len(feature_columns)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Пациентов</div>
            <div class="metric-value">{df["patient_id"].nunique()}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Классов</div>
            <div class="metric-value">{df["status"].nunique()}</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.subheader("Баланс классов")

    class_counts = df["status_label"].value_counts().reset_index()
    class_counts.columns = ["Класс", "Количество"]

    fig = px.bar(
        class_counts,
        x="Класс",
        y="Количество",
        text="Количество",
        title="Распределение записей по классам",
        height=420,
        color="Класс",
        color_discrete_map={
            "Здоровый": HEALTHY_COLOR,
            "Болезнь Паркинсона": DISEASE_COLOR
        }
    )
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color=DARK),
        title_font=dict(size=18),
        bargap=0.35
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Фрагмент данных")
    st.dataframe(df.head(20), width="stretch", hide_index=True)

    st.subheader("Распределение ключевых признаков")

    feature = st.selectbox(
        "Выберите признак",
        [
            "MDVP:Fo(Hz)",
            "MDVP:Fhi(Hz)",
            "MDVP:Flo(Hz)",
            "MDVP:Jitter(%)",
            "MDVP:Shimmer",
            "NHR",
            "HNR",
            "RPDE",
            "DFA",
            "spread1",
            "spread2",
            "D2",
            "PPE",
        ]
    )

    fig_hist = px.histogram(
        df,
        x=feature,
        color="status_label",
        nbins=30,
        marginal="box",
        title=f"Распределение признака {feature}",
        height=520,
        color_discrete_map={
            "Здоровый": HEALTHY_COLOR,
            "Болезнь Паркинсона": DISEASE_COLOR
        }
    )
    fig_hist.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color=DARK),
        title_font=dict(size=18)
    )
    st.plotly_chart(fig_hist, width="stretch")

    st.subheader("Корреляционная матрица")

    numeric_df = df.drop(
        columns=["name", "patient_id", "status_label"]
    ).select_dtypes(include="number")

    corr = numeric_df.corr()

    fig_corr = px.imshow(
        corr,
        title="Корреляция числовых признаков",
        color_continuous_scale="RdBu_r",
        range_color=[-1, 1],
        aspect="equal",
        width=820,
        height=820
    )
    fig_corr.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color=DARK),
        title_font=dict(size=18),
        margin=dict(l=40, r=40, t=80, b=120)
    )

    left, center, right = st.columns([1, 4, 1])
    with center:
        st.plotly_chart(fig_corr, width="content")


with tab2:
    st.title("Модели и сравнение")
    st.markdown(
        "<div class='subtitle'>Сравнение качества моделей при разных подходах к подготовке данных</div>",
        unsafe_allow_html=True
    )

    metrics_agg = pd.DataFrame({
        "Модель": ["KNN", "XGBoost", "Random Forest", "Logistic Regression", "Decision Tree", "SVM"],
        "ROC-AUC": [1.000, 1.000, 1.000, 0.750, 0.750, 0.844],
        "F1": [0.857, 0.920, 0.920, 0.909, 0.920, 0.923],
        "Recall": [1.000, 1.000, 1.000, 1.000, 1.000, 1.000],
        "Precision": [0.750, 0.860, 0.857, 0.833, 0.857, 0.857]
    })

    metrics_no_agg = pd.DataFrame({
        "Модель": ["KNN", "XGBoost", "Random Forest", "Logistic Regression", "Decision Tree", "SVM"],
        "ROC-AUC": [0.809, 0.746, 0.980, None, 0.720, 0.892],
        "F1": [0.906, 0.908, 0.986, 0.845, 0.932, 0.952],
        "Recall": [0.967, 1.000, 0.972, None, 0.944, 0.968],
        "Precision": [0.828, 0.831, 1.000, None, 0.919, 0.938]
    })

    st.subheader("1. Модели с агрегацией данных")
    st.dataframe(metrics_agg, hide_index=True, width="stretch")

    fig_agg = px.bar(
        metrics_agg,
        x="Модель",
        y="F1",
        text="F1",
        title="F1-score моделей с агрегацией данных",
        height=430
    )
    fig_agg.update_traces(marker_color=BLUE)
    fig_agg.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color=DARK),
        title_font=dict(size=18),
        yaxis_range=[0, 1.05]
    )
    st.plotly_chart(fig_agg, width="stretch")

    st.subheader("2. Модели без агрегации данных")
    st.dataframe(metrics_no_agg, hide_index=True, width="stretch")

    fig_no_agg = px.bar(
        metrics_no_agg,
        x="Модель",
        y="F1",
        text="F1",
        title="F1-score моделей без агрегации данных",
        height=430
    )
    fig_no_agg.update_traces(marker_color=BLUE)
    fig_no_agg.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color=DARK),
        title_font=dict(size=18),
        yaxis_range=[0, 1.05]
    )
    st.plotly_chart(fig_no_agg, width="stretch")

    st.subheader("3. Итоговые выводы")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class='success-box'>
        Лучшая модель по F1-score: Random Forest<br>
        F1 = 0.986<br>
        ROC-AUC = 0.980<br>
        Precision = 1.000
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='info-card'>
        Подход без агрегации со сплитом по пациентам позволяет сохранить 195 записей
        и снизить риск утечки данных между train и test.
        </div>
        """, unsafe_allow_html=True)

    best_models = pd.DataFrame({
        "Показатель": [
            "Максимальный F1",
            "Максимальный Recall",
            "Максимальный Precision",
            "Модель для инференса"
        ],
        "Модель": [
            "Random Forest",
            "XGBoost",
            "Random Forest",
            "Random Forest"
        ],
        "Значение": [
            "F1 = 0.986",
            "Recall = 1.000",
            "Precision = 1.000",
            "Используется во вкладке инференса"
        ]
    })

    st.dataframe(best_models, hide_index=True, width="stretch")


with tab3:
    st.title("Инференс")
    st.markdown(
        "<div class='subtitle'>Проверка модели на введённых данных</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='blue-box'>Значения можно изменять вручную.</div>",
        unsafe_allow_html=True
    )

    input_values = {}

    col_left, col_right = st.columns(2)

    for i, col in enumerate(feature_columns):
        median_value = float(df[col].median())
        min_value = float(df[col].min())
        max_value = float(df[col].max())

        target_col = col_left if i % 2 == 0 else col_right

        with target_col:
            input_values[col] = st.number_input(
                label=col,
                min_value=min_value,
                max_value=max_value,
                value=median_value,
                format="%.6f"
            )

    input_df = pd.DataFrame([input_values])

    st.subheader("Введённые значения")
    st.dataframe(input_df, width="stretch", hide_index=True)

    if st.button("Вывести прогноз"):
        prediction = rf_model.predict(input_df)[0]

        probability = None
        if hasattr(rf_model, "predict_proba"):
            probability = rf_model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.markdown(
                "<div class='warning-box'>Прогноз модели: высокий риск болезни Паркинсона</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div class='success-box'>Прогноз модели: признаки болезни Паркинсона не выявлены</div>",
                unsafe_allow_html=True
            )

        if probability is not None:
            st.metric(
                "Вероятность класса Болезнь Паркинсона",
                f"{probability:.2%}"
            )

        st.caption(
            "Результат предназначен только для демонстрации работы модели "
            "и не является медицинским заключением."
        )