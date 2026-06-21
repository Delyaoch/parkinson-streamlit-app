import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

st.set_page_config(
    page_title="Early Parkinson's Disease Detection",
    page_icon="🔬",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #F8FAFC;
}

h1, h2, h3 {
    color: #0F172A;
}

.block-container {
    padding-top: 2rem;
}

.subtitle {
    color: #475569;
    font-size: 18px;
    margin-bottom: 20px;
}

.info-card {
    background: white;
    padding: 22px;
    border-radius: 16px;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.07);
    margin-bottom: 18px;
}

.success-box {
    background-color: #ECFDF5;
    color: #065F46;
    padding: 18px;
    border-radius: 14px;
    border-left: 6px solid #16A34A;
    font-weight: 600;
}

.warning-box {
    background-color: #FEF2F2;
    color: #991B1B;
    padding: 18px;
    border-radius: 14px;
    border-left: 6px solid #DC2626;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv("data/parkinsons.data")
    df["patient_id"] = df["name"].str.extract(r"(S\d+)")
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
    st.title("🔬 Early Parkinson's Disease Detection")
    st.markdown(
        "<div class='subtitle'>Раннее выявление болезни Паркинсона по голосовым характеристикам пациентов</div>",
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Записей", df.shape[0])
    col2.metric("Признаков", len(feature_columns))
    col3.metric("Пациентов", df["patient_id"].nunique())
    col4.metric("Классов", df["status"].nunique())

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
            "Здоровый": "#16A34A",
            "Болезнь Паркинсона": "#DC2626"
        }
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Фрагмент данных")
    st.dataframe(df.head(20), use_container_width=True)

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
            "Здоровый": "#16A34A",
            "Болезнь Паркинсона": "#DC2626"
        }
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Корреляционная матрица")

    numeric_df = df.drop(
        columns=["name", "patient_id", "status_label"]
    ).select_dtypes(include="number")

    corr = numeric_df.corr()

    fig_corr = px.imshow(
        corr,
        title="Корреляция числовых признаков",
        color_continuous_scale="RdBu_r",
        aspect="auto",
        width=950,
        height=620
    )
    fig_corr.update_layout(margin=dict(l=40, r=40, t=80, b=120))
    st.plotly_chart(fig_corr, use_container_width=False)


with tab2:
    st.title("Модели и сравнение")
    st.markdown(
        "<div class='subtitle'>Сравнение качества моделей при разных подходах к подготовке данных</div>",
        unsafe_allow_html=True
    )

    st.subheader("1. Модели с агрегацией данных")

    metrics_agg = pd.DataFrame({
        "Модель": ["KNN", "XGBoost", "Random Forest", "Logistic Regression", "Decision Tree", "SVM"],
        "ROC-AUC": [1.000, 1.000, 1.000, 0.750, 0.750, 0.844],
        "F1": [0.857, 0.920, 0.920, 0.909, 0.920, 0.923],
        "Recall": [1.000, 1.000, 1.000, 1.000, 1.000, 1.000],
        "Precision": [0.750, 0.860, 0.857, 0.833, 0.857, 0.857]
    })

    st.dataframe(metrics_agg, hide_index=True, use_container_width=True)

    fig_agg = px.bar(
        metrics_agg,
        x="Модель",
        y="F1",
        text="F1",
        title="F1-score моделей с агрегацией данных",
        height=430
    )
    fig_agg.update_traces(marker_color="#2563EB")
    st.plotly_chart(fig_agg, use_container_width=True)

    st.subheader("2. Модели без агрегации данных")

    metrics_no_agg = pd.DataFrame({
        "Модель": ["KNN", "XGBoost", "Random Forest", "Logistic Regression", "Decision Tree", "SVM"],
        "ROC-AUC": [0.809, 0.746, 0.980, None, 0.720, 0.892],
        "F1": [0.906, 0.908, 0.986, 0.845, 0.932, 0.952],
        "Recall": [0.967, 1.000, 0.972, None, 0.944, 0.968],
        "Precision": [0.828, 0.831, 1.000, None, 0.919, 0.938]
    })

    st.dataframe(metrics_no_agg, hide_index=True, use_container_width=True)

    fig_no_agg = px.bar(
        metrics_no_agg,
        x="Модель",
        y="F1",
        text="F1",
        title="F1-score моделей без агрегации данных",
        height=430
    )
    fig_no_agg.update_traces(marker_color="#2563EB")
    st.plotly_chart(fig_no_agg, use_container_width=True)

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
        Подход без агрегации со сплитом по пациентам позволяет избежать утечки данных между train и test.
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

    st.dataframe(best_models, hide_index=True, use_container_width=True)


with tab3:
    st.title("Инференс")
    st.markdown(
        "<div class='subtitle'>Проверка прогноза по голосовым признакам</div>",
        unsafe_allow_html=True
    )

    st.info(
        "Значения можно изменять вручную."
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
    st.dataframe(input_df, use_container_width=True)

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