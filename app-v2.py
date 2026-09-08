from pathlib import Path
import re
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import plotly.express as px

import warnings

# Silencia el aviso específico de nombres de columnas mezclados en PyArrow/Streamlit
warnings.filterwarnings("ignore", category=UserWarning, module="streamlit.dataframe_util")

# --- CONFIGURACIÓN DE PÁGINA STREAMLIT ---
st.set_page_config(
    page_title="Tablero Agroclimático Nacional",
    page_icon="🌦️",
    layout="wide",
)

# --- RUTAS Y CONSTANTES ---
BASE_DIR = Path(__file__).resolve().parent
DIR_DATOS_CRUDOS = BASE_DIR / "datos_crudos"
SHAPE_ESTACIONES = BASE_DIR / "estaciones_smn-inta_conv.shp"
SHAPE_DEPARTAMENTOS = BASE_DIR / "departamentos_argentina.shp"

DICC_MESES = {
    1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic"
}

# --- CLASIFICACIÓN HISTÓRICA ENSO (ONI / NOAA) ---
ENSO_ANUAL = {
    # El Niño
    1965: "Niño", 1968: "Niño", 1969: "Niño", 1972: "Niño", 1976: "Niño", 1977: "Niño",
    1982: "Niño", 1986: "Niño", 1987: "Niño", 1991: "Niño", 1994: "Niño", 1997: "Niño",
    2002: "Niño", 2004: "Niño", 2006: "Niño", 2009: "Niño", 2015: "Niño", 2018: "Niño", 2023: "Niño",
    # La Niña
    1970: "Niña", 1971: "Niña", 1973: "Niña", 1974: "Niña", 1975: "Niña", 1988: "Niña",
    1989: "Niña", 1998: "Niña", 1999: "Niña", 2000: "Niña", 2007: "Niña", 2008: "Niña",
    2010: "Niña", 2011: "Niña", 2017: "Niña", 2020: "Niña", 2021: "Niña", 2022: "Niña"
}

def obtener_fase_enso(anio):
    return ENSO_ANUAL.get(anio, "Neutro")

# --- FUNCIONES DE CARGA DE DATOS CON CACHÉ ---
@st.cache_data(show_spinner="Cargando capa de departamentos...")
def cargar_capa_departamentos():
    if not SHAPE_DEPARTAMENTOS.exists():
        return None
    gdf = gpd.read_file(SHAPE_DEPARTAMENTOS)
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return gdf

@st.cache_data(show_spinner="Cargando capas de estaciones...")
def cargar_estaciones():
    gdf = None
    if SHAPE_ESTACIONES.exists():
        gdf = gpd.read_file(SHAPE_ESTACIONES)
    else:
        csv_est = BASE_DIR / "estaciones_smn-inta_conv.csv"
        if csv_est.exists():
            df = pd.read_csv(csv_est)
            gdf = gpd.GeoDataFrame(
                df, geometry=gpd.points_from_xy(df.longitud, df.latitud), crs="EPSG:4326"
            )

    if gdf is None:
        return None

    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    col_code = next((c for c in ["codigo_nh", "codigo", "id", "nh"] if c in gdf.columns), gdf.columns[0])
    gdf["codigo_limpio"] = (
        gdf[col_code].astype(str).str.strip().str.split(".").str[0].str.zfill(5)
    )
    return gdf

@st.cache_data(show_spinner="Cargando datos climáticos optimizados...")
def cargar_datos_diarios():
    archivo_cache = BASE_DIR / "datos_cache.parquet"

    if not archivo_cache.exists():
        return pd.DataFrame()

    # Cargar solo las columnas estrictamente necesarias
    cols = ["codigo_limpio", "red", "fecha", "anio", "mes", "tmax", "tmin", "precip"]
    df = pd.read_parquet(archivo_cache, columns=cols)

    # Optimización extrema de memoria RAM
    df["anio"] = df["anio"].astype("int16")
    df["mes"] = df["mes"].astype("int8")
    df["tmax"] = df["tmax"].astype("float32")
    df["tmin"] = df["tmin"].astype("float32")
    df["precip"] = df["precip"].astype("float32")
    df["red"] = df["red"].astype("category")

    return df

# --- TITULO E INTERFAZ DE STREAMLIT ---
st.title("🌦️ Tablero Agroclimático Interactivo")
st.markdown("Análisis temporal y zonal de precipitación y temperaturas por red de estaciones.")

gdf_deptos = cargar_capa_departamentos()
gdf_estaciones = cargar_estaciones()
df_diario = cargar_datos_diarios()

if gdf_deptos is None or df_diario.empty or gdf_estaciones is None:
    st.error("No se pudieron cargar los datos base (capas geográficas o archivos .DAT). Verifique las rutas.")
    st.stop()

# --- FILTROS EN BARRA LATERAL ---
st.sidebar.header("⚙️ Parámetros de Consulta")

red_seleccionada = st.sidebar.multiselect(
    "Red de Estaciones:",
    options=["SMN", "INTA"],
    default=["SMN", "INTA"]
)

variables_dic = {
    "Precipitación (mm)": ("precip", "sum"),
    "Temperatura Máxima (°C)": ("tmax", "mean"),
    "Temperatura Mínima (°C)": ("tmin", "mean")
}
var_label = st.sidebar.selectbox("Variable climática:", list(variables_dic.keys()))
var_col, var_agregacion = variables_dic[var_label]

posibles_provs = ["provincia", "nam_prov", "fna_prov", "provinci", "ign_nam"]
col_prov = next((c for c in gdf_deptos.columns if c.lower() in posibles_provs), gdf_deptos.columns[0])

provincias = sorted(gdf_deptos[col_prov].dropna().unique().tolist())
prov_sel = st.sidebar.selectbox("Provincia:", provincias)

gdf_prov = gdf_deptos[gdf_deptos[col_prov] == prov_sel]

posibles_deptos = ["nam", "fna", "nombre", "departamento", "depto"]
col_depto = next((c for c in gdf_prov.columns if c.lower() in posibles_deptos), gdf_prov.columns[0])

deptos_disponibles = sorted(gdf_prov[col_depto].dropna().unique().tolist())
deptos_sel = st.sidebar.multiselect("Departamentos:", options=deptos_disponibles, default=deptos_disponibles[:1])

modo_fecha = st.sidebar.radio("Modo de selección temporal:", ["Por Años y Meses", "Por Rango de Fechas Diario"])

if modo_fecha == "Por Rango de Fechas Diario":
    min_f, max_f = df_diario["fecha"].min(), df_diario["fecha"].max()
    f_inicio, f_fin = st.sidebar.date_input("Rango de Fechas:", [min_f, max_f], min_value=min_f, max_value=max_f)
else:
    anios = sorted(df_diario["anio"].unique())
    anio_in, anio_fi = st.sidebar.select_slider("Rango de Años:", options=anios, value=(anios[0], anios[-1]))
    mes_in, mes_fi = st.sidebar.select_slider("Rango de Meses:", options=list(DICC_MESES.keys()), format_func=lambda x: DICC_MESES[x], value=(1, 12))

# --- FILTRADO SPATIAL Y TEMPORAL ---
gdf_deptos_sel = gdf_prov[gdf_prov[col_depto].isin(deptos_sel)]

if not gdf_deptos_sel.empty:
    estaciones_en_depto = gpd.sjoin(gdf_estaciones, gdf_deptos_sel, how="inner", predicate="intersects")
else:
    estaciones_en_depto = gdf_estaciones.copy()

estaciones_filtradas = estaciones_en_depto[estaciones_en_depto["red"].isin(red_seleccionada)].copy() if "red" in estaciones_en_depto.columns else estaciones_en_depto.copy()

df_filtrado = df_diario[df_diario["codigo_limpio"].isin(estaciones_filtradas["codigo_limpio"])].copy()

if modo_fecha == "Por Rango de Fechas Diario":
    df_filtrado = df_filtrado[(df_filtrado["fecha"] >= pd.to_datetime(f_inicio)) & (df_filtrado["fecha"] <= pd.to_datetime(f_fin))]
else:
    df_filtrado = df_filtrado[(df_filtrado["anio"] >= anio_in) & (df_filtrado["anio"] <= anio_fi)]
    if mes_in <= mes_fi:
        df_filtrado = df_filtrado[(df_filtrado["mes"] >= mes_in) & (df_filtrado["mes"] <= mes_fi)]
    else:
        df_filtrado = df_filtrado[(df_filtrado["mes"] >= mes_in) | (df_filtrado["mes"] <= mes_fi)]

df_merged = pd.merge(df_filtrado, estaciones_filtradas.drop(columns=["geometry"], errors="ignore"), on="codigo_limpio", how="inner")

if not df_merged.empty:
    df_merged["fase_enso"] = df_merged["anio"].apply(obtener_fase_enso)

# --- PESTAÑAS DEL DASHBOARD ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Gráfico Interactivo", 
    "📋 Tabla de Datos", 
    "📍 Estaciones Seleccionadas",
    "🌊 Fase ENSO"
])

# --- TAB 1: GRÁFICO INTERACTIVO ---
with tab1:
    if df_merged.empty:
        st.warning("No hay datos meteorológicos registrados para los filtros seleccionados en este departamento/período.")
    else:
        if modo_fecha == "Por Rango de Fechas Diario":
            df_chart = df_merged.groupby(["fecha", "red"])[var_col].agg(var_agregacion).reset_index()
            fig = px.line(
                df_chart, x="fecha", y=var_col, color="red",
                title=f"{var_label} diaria en {prov_sel} ({', '.join(deptos_sel)})",
                labels={"fecha": "Fecha", var_col: var_label, "red": "Red"}
            )
        else:
            df_chart = df_merged.groupby(["anio", "red"])[var_col].agg(var_agregacion).reset_index()
            fig = px.bar(
                df_chart, x="anio", y=var_col, color="red", barmode="group",
                title=f"{var_label} ({DICC_MESES[mes_in]}-{DICC_MESES[mes_fi]}) en {prov_sel} ({', '.join(deptos_sel)})",
                labels={"anio": "Año", var_col: var_label, "red": "Red"}
            )
        st.plotly_chart(fig, width='stretch')

# --- TAB 2: TABLA DE DATOS ---
with tab2:
    st.subheader("Datos consolidados")
    if df_merged.empty:
        st.info("No hay tabla disponible para esta selección.")
    else:
        posibles_nombres_est = ["nombre", "estacion", "station_nam", "nom_est", "codigo_nh"]
        col_nombre_est = next((c for c in df_merged.columns if c.lower() in posibles_nombres_est), "codigo_limpio")

        if modo_fecha == "Por Rango de Fechas Diario":
            pivoted = df_merged.pivot_table(index=["fecha", col_nombre_est], columns="red", values=var_col, aggfunc=var_agregacion).reset_index()
        else:
            pivoted = df_merged.pivot_table(index=["anio", col_nombre_est], columns=["mes", "red"], values=var_col, aggfunc=var_agregacion).reset_index()

        pivoted = pivoted.rename(columns={col_nombre_est: "Estación"})
        st.dataframe(pivoted.head(2000), width="stretch")
        st.caption("ℹ️ Mostrando las primeras 2.000 filas. Descarga el CSV para obtener el total completo.")

        csv = pivoted.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Descargar Tabla en CSV", csv, "datos_agronomicos.csv", "text/csv")

# --- TAB 3: ESTACIONES Y MAPA CON CENTRADO AUTOMÁTICO ---
import warnings

# --- TAB 3: ESTACIONES Y MAPA CON CENTRADO AUTOMÁTICO ---
with tab3:
    st.subheader("Estaciones incluidas en la consulta")

    if estaciones_filtradas.empty:
        st.warning("⚠️ No se encontraron estaciones geográficas asociadas al departamento seleccionado.")
    else:
        # Búsqueda amplia de la columna con el NOMBRE REAL de la estación
        posibles_nombres_est = [
            "nombre", "estacion", "station_nam", "nom_est", "nombre_est", 
            "nombre_estacion", "denominacion", "localidad", "nom_loc"
        ]
        col_nombre_est = next(
            (c for c in estaciones_filtradas.columns if c.lower() in posibles_nombres_est), 
            None
        )

        est_mapa = estaciones_filtradas.copy()

        # Extraer coordenadas evitando la advertencia de CRS geográfico
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            est_mapa["lon"] = est_mapa.geometry.centroid.x.round(4)
            est_mapa["lat"] = est_mapa.geometry.centroid.y.round(4)

        # Si no encontró ninguna columna de texto de nombre, asigna 'Estación <código>'
        if col_nombre_est:
            est_mapa["Nombre Estación"] = est_mapa[col_nombre_est]
        else:
            est_mapa["Nombre Estación"] = "Estación " + est_mapa["codigo_limpio"].astype(str)

        cols_mostrar = ["codigo_limpio", "Nombre Estación", "red", "lat", "lon"]
        cols_presentes = [c for c in cols_mostrar if c in est_mapa.columns]
        
        resumen_estaciones = (
            est_mapa[cols_presentes]
            .drop_duplicates(subset=["codigo_limpio"])
            .rename(columns={
                "codigo_limpio": "Código", 
                "red": "Red",
                "lat": "Latitud",
                "lon": "Longitud"
            })
        )

        st.metric("Total de Estaciones en el Área Seleccionada", len(resumen_estaciones))

        # Reordenar columnas para visualización clara
        cols_tabla_est = [c for c in ["Código", "Nombre Estación", "Red", "Latitud", "Longitud"] if c in resumen_estaciones.columns]
        
        # Compatibilidad de ancho para tablas
        try:
            st.dataframe(resumen_estaciones[cols_tabla_est], width='stretch')
        except Exception:
            st.dataframe(resumen_estaciones[cols_tabla_est], width='stretch')

        # Configuración del Mapa con PUNTOS MÁS GRANDES
        mapa_df = resumen_estaciones.dropna(subset=["Latitud", "Longitud"])
        if not mapa_df.empty:
            lat_centro = mapa_df["Latitud"].mean()
            lon_centro = mapa_df["Longitud"].mean()

            mapa_df["tamano_punto"] = 14

            params = {
                "data_frame": mapa_df,
                "lat": "Latitud",
                "lon": "Longitud",
                "hover_name": "Nombre Estación",
                "hover_data": [c for c in ["Código", "Red", "Latitud", "Longitud"] if c in mapa_df.columns],
                "color": "Red" if "Red" in mapa_df.columns else None,
                "size": "tamano_punto",
                "size_max": 16,
                "zoom": 7,
                "height": 520,
                "title": "Ubicación Geográfica de las Estaciones Seleccionadas"
            }

            if hasattr(px, "scatter_map"):
                fig_mapa = px.scatter_map(**params)
                fig_mapa.update_layout(
                    map_style="open-street-map",
                    map_center={"lat": lat_centro, "lon": lon_centro},
                    margin={"r": 0, "t": 40, "l": 0, "b": 0}
                )
            else:
                fig_mapa = px.scatter_mapbox(**params)
                fig_mapa.update_layout(
                    mapbox_style="open-street-map",
                    mapbox_center={"lat": lat_centro, "lon": lon_centro},
                    margin={"r": 0, "t": 40, "l": 0, "b": 0}
                )

            try:
                st.plotly_chart(fig_mapa, width='stretch')
            except Exception:
                st.plotly_chart(fig_mapa, width='stretch')
        else:
            st.info("Las estaciones seleccionadas no cuentan con coordenadas lat/lon válidas para graficar en el mapa.")

# --- TAB 4: ANÁLISIS DE FASES ENSO ---
with tab4:
    st.subheader("Análisis comparativo según Fase ENSO (El Niño / La Niña / Neutro)")
    
    if df_merged.empty:
        st.warning("No hay datos disponibles para la consulta seleccionada.")
    else:
        colores_enso = {
            "Niño": "#EF553B",    # Rojo
            "Niña": "#636EFA",    # Azul
            "Neutro": "#00CC96"   # Verde
        }
        
        df_enso = (
            df_merged.groupby(["anio", "fase_enso", "red"])[var_col]
            .agg(var_agregacion)
            .reset_index()
        )
        
        fig_enso_bar = px.bar(
            df_enso,
            x="anio",
            y=var_col,
            color="fase_enso",
            color_discrete_map=colores_enso,
            barmode="group",
            title=f"Comportamiento de {var_label} por Año y Fase ENSO en {prov_sel}",
            labels={"anio": "Año", var_col: var_label, "fase_enso": "Fase ENSO", "red": "Red"}
        )
        st.plotly_chart(fig_enso_bar, width='stretch')
        
        st.markdown("---")
        
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            st.markdown("##### Distribución Estadísticamente Comparada")
            fig_box = px.box(
                df_enso,
                x="fase_enso",
                y=var_col,
                color="fase_enso",
                color_discrete_map=colores_enso,
                points="all",
                title=f"Distribución de {var_label} según Fase ENSO",
                labels={"fase_enso": "Fase ENSO", var_col: var_label}
            )
            st.plotly_chart(fig_box, width='stretch')
            
        with col_c2:
            st.markdown("##### Resumen Numérico por Fase")
            resumen_fase = (
                df_enso.groupby("fase_enso")[var_col]
                .agg(["mean", "median", "std", "count"])
                .rename(columns={
                    "mean": "Promedio",
                    "median": "Mediana",
                    "std": "Desv. Estándar",
                    "count": "Años Registrados"
                })
                .reset_index()
            )
            st.dataframe(resumen_fase, width='stretch')
