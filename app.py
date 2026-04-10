import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import plotly.express as px

######### CONFIGURACIÓN
st.set_page_config(page_title="Violencia Digital", layout="wide")

######### ESTILO
st.markdown("""
<style>

/* Fondo general */
.main {
    background-color: #2B2A52;
}

/* PANEL PRINCIPAL */
.block-container {
    background-color: #BED3E6;
    padding: 2rem;
    border-radius: 20px;
}

/* TÍTULOS */
h1 {
    color: #2B2A52;
    text-align: center;
}

h2, h3 {
    color: #2B2A52;
}

/* TARJETAS KPI */
[data-testid="metric-container"] {
    background: white;
    padding: 15px;
    border-radius: 15px;
    text-align: center;
    color: #2B2A52;
    box-shadow: 0px 6px 15px rgba(0,0,0,0.2);
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background-color: #2B2A52;
    color: white;
}

/* GRÁFICAS */
.stPlotlyChart, .stPyplot {
    background-color: #2B2A52;
    border-radius: 15px;
    padding: 15px;
}

</style>
""", unsafe_allow_html=True)

######### TÍTULO
st.title("Algoritmos contra la violencia digital: identificación de perfiles de riesgo en mujeres mediante Machine Learning")
st.subheader("¿Le han enviado mensajes o publicado comentarios con insinuaciones sexuales, insultos u ofensas, a través del celular, correo electrónico o redes sociales?")

######### DATOS
df = pd.read_excel("AMBITOESCOLAR.xlsx")
df.columns = df.columns.str.strip().str.upper()

######### LIMPIEZA
df = df.fillna(df.median(numeric_only=True))
df = df.fillna("DESCONOCIDO")
df = df.drop(columns=[col for col in df.columns if "ID" in col], errors='ignore')

######### VARIABLE DE SALIDA
target = df.columns[-1]
df[target] = df[target].replace({2: 0})

######### FILTROS
st.sidebar.header("Filtros")

estado_sel = st.sidebar.selectbox("Estado", ["Todos"] + list(df["ESTADO"].unique()))

if "DOMINIO" in df.columns:
    dominio_sel = st.sidebar.selectbox("Zona", ["Todos"] + list(df["DOMINIO"].unique()))
else:
    dominio_sel = "Todos"

max_depth = st.sidebar.slider("Profundidad del árbol", 2, 10, 3)

######### FILTRADO
df_filtrado = df.copy()

if estado_sel != "Todos":
    df_filtrado = df_filtrado[df_filtrado["ESTADO"] == estado_sel]

if dominio_sel != "Todos":
    df_filtrado = df_filtrado[df_filtrado["DOMINIO"] == dominio_sel]

######### KPIs
col1, col2, col3 = st.columns(3)

total = len(df_filtrado)
violencia = df_filtrado[target].sum()
porcentaje = (violencia / total) * 100 if total > 0 else 0

col1.metric("Total mujeres", total)
col2.metric("Casos de violencia", int(violencia))
col3.metric("% violencia", f"{porcentaje:.2f}%")

######### MODELO
X = df_filtrado.drop(columns=[target])
y = df_filtrado[target]

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

modelo = DecisionTreeClassifier(
    max_depth=max_depth,
    class_weight='balanced',
    random_state=42
)

modelo.fit(X_train, y_train)

######### ÁRBOL DE DECISIÓN E IMPORTANCIA
colA, colB = st.columns([2,1])

with colA:
    st.subheader("Árbol de Decisión")
    fig, ax = plt.subplots(figsize=(10,5))
    plot_tree(modelo, feature_names=X.columns, class_names=["No","Sí"], filled=True, fontsize=6)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    st.pyplot(fig)

with colB:
    st.subheader("Variables más importantes")
    importancia = pd.Series(modelo.feature_importances_, index=X.columns)
    importancia = importancia.sort_values(ascending=True).tail(10)

    imp_df = importancia.reset_index()
    imp_df.columns = ["Variable", "Importancia"]

    fig_imp = px.bar(
        imp_df,
        x="Importancia",
        y="Variable",
        orientation='h',
        color="Importancia",
        color_continuous_scale="Purples"
    )

    fig_imp.update_layout(
        plot_bgcolor="#2B2A52",
        paper_bgcolor="#2B2A52",
        font=dict(color="white")
    )

    st.plotly_chart(fig_imp, use_container_width=True)

######### GRAFICA DE BARRAS DE PREDICCIONES
st.subheader("Predicciones del modelo: ¿Le han enviado mensajes o publicado comentarios con insinuaciones sexuales, insultos u ofensas, a través del celular, correo electrónico o redes sociales? (Sí vs No)")

df_pred = X_test.copy()
df_pred["REAL"] = y_test.values
df_pred["PREDICCION"] = modelo.predict(X_test)
df_pred["ESTADO"] = df_filtrado.loc[X_test.index, "ESTADO"]

df_group = df_pred.groupby(["ESTADO","PREDICCION"]).size().reset_index(name="Conteo")
df_group["PREDICCION"] = df_group["PREDICCION"].map({0:"No",1:"Sí"})

fig_pred = px.bar(
    df_group,
    x="ESTADO",
    y="Conteo",
    color="PREDICCION",
    barmode="group",
    color_discrete_map={"Sí":"#a21caf","No":"#6366f1"}
)

fig_pred.update_layout(
    plot_bgcolor="#2B2A52",
    paper_bgcolor="#2B2A52",
    font=dict(color="white"),
    xaxis=dict(tickangle=45)
)

st.plotly_chart(fig_pred, use_container_width=True)

#########  MATRIZ DE CONFUSIÓN Y TENDENCIA
colC, colD = st.columns(2)

with colC:
    st.subheader("Matriz de Confusión")
    cm = confusion_matrix(y_test, modelo.predict(X_test))
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', linewidths=1)
    fig_cm.patch.set_facecolor('white')
    ax_cm.set_facecolor('white')
    st.pyplot(fig_cm)

with colD:
    st.subheader("Tendencia")
    df_prob = df_filtrado.groupby("ESTADO")[target].mean().reset_index()
    df_prob[target] = (df_prob[target]*100).round(2)

    fig_line = px.line(
        df_prob,
        x="ESTADO",
        y=target,
        markers=True,
        color_discrete_sequence=["#a21caf"]
    )

    fig_line.update_layout(
        plot_bgcolor="#2B2A52",
        paper_bgcolor="#2B2A52",
        font=dict(color="white"),
        xaxis=dict(tickangle=45),
        yaxis_title="% de violencia"
    )

    st.plotly_chart(fig_line, use_container_width=True)

######### GRAFICA DE BARRAS POR ESTADO ESTADO
st.subheader("Comparación por Estado")

mapa_df = df_filtrado.groupby("ESTADO")[target].mean().reset_index()
mapa_df[target] = (mapa_df[target]*100).round(2)

fig_bar = px.bar(
    mapa_df.sort_values(by=target),
    x=target,
    y="ESTADO",
    orientation='h',
    color=target,
    color_continuous_scale="RdPu"
)

fig_bar.update_layout(
    plot_bgcolor="#2B2A52",
    paper_bgcolor="#2B2A52",
    font=dict(color="white"),
    xaxis_title="% de violencia"
)

st.plotly_chart(fig_bar, use_container_width=True)

######### INTERPRETACIÓN
st.subheader("Interpretación")

st.write("""
El dashboard muestra la incidencia de violencia digital en mujeres en porcentaje,
permitiendo comparar estados y analizar factores relevantes mediante un modelo de árbol de decisión.
""")