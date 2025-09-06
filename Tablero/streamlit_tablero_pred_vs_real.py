# streamlit run streamlit_tablero_pred_vs_real.py
# Tablero: Predicción vs Real — Maqueta según especificación del proyecto

import io
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# =============================
# 0) Configuración general
# =============================
st.set_page_config(
    page_title="Tablero Ventas — Predicción vs Real",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Tablero de Ventas — Predicción vs Realidad")

# =============================
# 1) Generar DATOS DE EJEMPLO
# =============================
@st.cache_data(show_spinner=False)
def generar_datos_ejemplo(
    dias: int = 180,
    n_partners: int = 5,
    paises: list[str] = ("Colombia", "Perú", "México"),
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    inicio = datetime.today().date() - timedelta(days=dias)
    fechas = pd.date_range(inicio, periods=dias, freq="D")

    partners = [f"Partner_{i+1}" for i in range(n_partners)]
    paises = list(paises)
    ciudades = {
        "Colombia": ["Bogotá", "Medellín", "Cali"],
        "Perú": ["Lima", "Arequipa"],
        "México": ["CDMX", "Guadalajara"],
    }

    filas = []
    tienda_id = 1
    for p in partners:
        for pais in paises:
            for ciudad in ciudades[pais]:
                # 2 tiendas por ciudad
                for _ in range(2):
                    store = f"{p[:3].upper()}-{pais[:2].upper()}-{tienda_id:03d}"
                    tienda_id += 1
                    # estacionalidad simple por día de la semana
                    base_week = rng.uniform(800, 2000, size=7)
                    trend = np.linspace(0.95, 1.05, len(fechas))
                    for i, d in enumerate(fechas):
                        dow = d.weekday()
                        ruido = rng.normal(1.0, 0.12)
                        ventas_reales = max(0, base_week[dow] * trend[i] * ruido)
                        # predicción = real con ruido controlado (mejorable)
                        pred_ruido = rng.normal(1.0, 0.10)
                        ventas_pred = max(0, ventas_reales * pred_ruido)
                        filas.append(
                            {
                                "fecha": pd.Timestamp(d),
                                "partner": p,
                                "pais": pais,
                                "ciudad": ciudad,
                                "tienda": store,
                                "ventas_reales": round(ventas_reales, 2),
                                "ventas_predichas": round(ventas_pred, 2),
                            }
                        )
    df = pd.DataFrame(filas)
    return df

# Datos de ejemplo (puedes reemplazar por carga real más abajo)
df = generar_datos_ejemplo()

with st.expander("Vista previa de los datos (ejemplo)"):
    st.dataframe(df.head(20), use_container_width=True)

# =============================
# (Opcional) Cargar datos reales
# =============================
with st.sidebar:
    st.header("📥 Cargar datos reales (opcional)")
    st.caption(
        "El tablero espera columnas: `fecha`, `partner`, `pais`, `ciudad`, `tienda`, `ventas_reales`, `ventas_predichas`."
    )
    up = st.file_uploader("Archivo CSV/XLSX", type=["csv", "xlsx", "xls"])
    sep = st.text_input("Separador CSV", value=",")

@st.cache_data(show_spinner=False)
def cargar_archivo(archivo, sep=","):
    if archivo is None:
        return None
    name = archivo.name.lower()
    if name.endswith(".csv"):
        tmp = pd.read_csv(archivo, sep=sep)
    else:
        tmp = pd.read_excel(archivo)
    # normalización mínima de nombres
    tmp = tmp.rename(columns={c: c.strip().lower() for c in tmp.columns})
    # asegurar columnas clave si vienen con mayúsculas
    rename_map = {
        "transaction date": "fecha",
        "partner name": "partner",
        "country": "pais",
        "city": "ciudad",
        "store name": "tienda",
        "local amount": "ventas_reales",
        "pred": "ventas_predichas",
    }
    for k, v in rename_map.items():
        if k in tmp.columns and v not in tmp.columns:
            tmp = tmp.rename(columns={k: v})
    # tipos
    if "fecha" in tmp.columns:
        tmp["fecha"] = pd.to_datetime(tmp["fecha"], errors="coerce")
    for col in ("ventas_reales", "ventas_predichas"):
        if col in tmp.columns:
            tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
    return tmp

real = cargar_archivo(up, sep)
if real is not None and not real.empty:
    # Usar datos reales cargados
    df = real.dropna(subset=["fecha"]).copy()
    st.success("Usando datos reales cargados.")

# =============================
# 2) FILTROS (en la parte superior)
# =============================
flt_area = st.container()
with flt_area:
    c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])

    # Rango de fechas
    min_f, max_f = df["fecha"].min().date(), df["fecha"].max().date()
    with c1:
        fecha_sel = st.date_input(
            "Rango de fechas",
            value=(min_f, max_f),
            min_value=min_f,
            max_value=max_f,
        )

        # Validación: debe haber 2 fechas
        if not isinstance(fecha_sel, (list, tuple)) or len(fecha_sel) != 2:
            st.warning("⚠️ Por favor selecciona un rango de fechas válido (inicio y fin).")
            st.stop()

        r0, r1 = fecha_sel

    # Marca aliada
    partners = sorted(df["partner"].astype(str).unique())
    with c2:
        sel_partner = st.multiselect("Marca aliada", partners, default=partners)

    # País
    paises = sorted(df["pais"].astype(str).unique())
    with c3:
        sel_pais = st.multiselect("País", paises, default=paises)

    # Tienda
    tiendas_all = df["tienda"].astype(str).unique().tolist()
    default_tiendas = tiendas_all if len(tiendas_all) <= 30 else tiendas_all[:30]
    with c4:
        sel_tienda = st.multiselect("Tienda", default_tiendas, default=default_tiendas)

mask = (
    (df["fecha"].dt.date >= r0)
    & (df["fecha"].dt.date <= r1)
    & (df["partner"].astype(str).isin(sel_partner))
    & (df["pais"].astype(str).isin(sel_pais))
    & (df["tienda"].astype(str).isin(sel_tienda))
)

df_f = df.loc[mask].copy()

# =============================
# 3) KPIs (tarjetas)
# =============================
col1, col2, col3 = st.columns(3)
ventas_reales_tot = float(df_f["ventas_reales"].sum()) if "ventas_reales" in df_f else 0.0
ventas_pred_tot = float(df_f["ventas_predichas"].sum()) if "ventas_predichas" in df_f else 0.0

# % Precisión: Accuracy tipo (1 - MAPE)
def calc_precision(y_true: pd.Series, y_pred: pd.Series) -> float:
    s = pd.concat([y_true, y_pred], axis=1).dropna()
    if s.empty:
        return np.nan
    y, yp = s.iloc[:, 0].values, s.iloc[:, 1].values
    denom = np.where(y == 0, np.nan, y)
    mape = np.nanmean(np.abs((y - yp) / denom))
    if np.isnan(mape):
        return np.nan
    return max(0.0, 1 - mape)

precision = calc_precision(df_f.get("ventas_reales", pd.Series(dtype=float)), df_f.get("ventas_predichas", pd.Series(dtype=float)))

with col1:
    st.metric("Ventas Reales (filtrado)", f"${ventas_reales_tot:,.0f}")
with col2:
    st.metric("Ventas Predichas (filtrado)", f"${ventas_pred_tot:,.0f}")
with col3:
    st.metric("% Precisión (1 - MAPE)", f"{(precision*100 if not np.isnan(precision) else 0):.2f}%")

# =============================
# 4) Serie temporal: Real vs Pred
# =============================
agg_ts = (
    df_f
    .groupby("fecha", as_index=False)[["ventas_reales", "ventas_predichas"]]
    .sum()
    .sort_values("fecha")
)
if not agg_ts.empty:
    ts_long = agg_ts.melt(id_vars=["fecha"], value_vars=["ventas_reales", "ventas_predichas"],
                          var_name="serie", value_name="valor")
    fig = px.line(ts_long, x="fecha", y="valor", color="serie", markers=True,
                  title="Ventas: Predicción vs Realidad (serie temporal)")
    fig.update_layout(xaxis_title="Fecha", yaxis_title="Ventas")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No hay datos en el rango/selección actual para graficar.")



# =============================
# 5) Exportación (CSV)
# =============================
col1, col2 = st.columns([3, 1])  # más espacio para la izquierda

with col1:
    st.subheader("Desglose por tienda")

with col2:
    csv = df_f.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Descargar CSV",
        data=csv,
        file_name="pred_vs_real.csv",
        mime="text/csv",
        use_container_width=True
    )


# =============================
# 6) Desglose por tienda (tabla)
# =============================
cols_show = ["partner", "pais", "ciudad", "tienda", "ventas_reales", "ventas_predichas"]

detalle = (
    df_f[cols_show]
    .groupby(["partner", "pais", "ciudad", "tienda"], as_index=False)
    .sum()
    .sort_values(["partner", "pais", "ciudad", "tienda"]) 
)

st.dataframe(detalle, use_container_width=True)

