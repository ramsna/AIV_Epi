# aiv_app_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import itertools
import os
import pydeck as pdk

# =========================
# Helpers
# =========================
def calcular_DPC(sec: str) -> pd.DataFrame:
    sec = sec.upper().replace("\n", "")
    aminoacidos = "ACDEFGHIKLMNPQRSTVWY"
    DPC = [''.join(p) for p in itertools.product(aminoacidos, repeat=2)]
    dpc = {d: 0 for d in DPC}
    for i in range(len(sec) - 1):
        par = sec[i:i + 2]
        if par in DPC:
            dpc[par] += 1
    total = len(sec) - 1
    if total > 0:
        for par in dpc:
            dpc[par] /= total
        dpc_vec = pd.DataFrame([dpc])
        dpc_vec.columns = list(range(400))
        return dpc_vec
    return pd.DataFrame([np.zeros(400)], columns=list(range(400)))

def detectar_sitio_clivaje(secuencia, motivos, ventana_max=14):
    secuencia = secuencia.upper()
    motivos_set = set(motivos['Cleavage_Site'].astype(str).str.strip())
    encontrados = []
    for i in range(len(secuencia) - 2):
        if secuencia[i:i + 3] == "GLF":
            for size in range(4, ventana_max + 1):
                inicio = i - size
                if inicio >= 0:
                    motivo = secuencia[inicio:i]
                    if motivo in motivos_set:
                        info = motivos[motivos['Cleavage_Site'].astype(str).str.strip() == motivo]
                        clado_tipo = info.iloc[0]['Clade_or_Type']
                        subtipo = info.iloc[0]['Subtype']
                        encontrados.append(
                            f"- Motivo: {motivo} | Subtipo: {subtipo} | Clado/Tipo: {clado_tipo}"
                        )
    return "\n".join(encontrados) if encontrados else "Ningún motivo detectado"
import os, zipfile
import streamlit as st
import gdown

# 👇 Reemplaza por el ID real de tu Drive
DRIVE_ID = "1CMLlczo-eWmFDVEChozS08W-JWwuFIFw/view?usp=sharing"
URL = f"https://drive.google.com/file/d/1CMLlczo-eWmFDVEChozS08W-JWwuFIFw/view?usp=sharing"

DEST_DIR = "modelos"
TMP_ZIP  = "modelos_tmp.zip"

@st.cache_data(show_spinner=True)
def ensure_modelos_drive():
    # Si ya están extraídos, usar directo
    necesarios = [
        "scaler.pkl",
        "SVM_best_model.pkl",
        "KNN_best_model.pkl",
        "cleavage_sites_H5_H7_extended.csv",
    ]
    if all(os.path.exists(os.path.join(DEST_DIR, f)) for f in necesarios):
        return DEST_DIR

    os.makedirs(DEST_DIR, exist_ok=True)

    # Descargar ZIP con progreso
    st.info("📦 Descargando modelos desde Google Drive…")
    gdown.download(URL, TMP_ZIP, quiet=False)

    # Extraer
    with zipfile.ZipFile(TMP_ZIP, "r") as z:
        z.extractall(DEST_DIR)

    # Limpiar
    try:
        os.remove(TMP_ZIP)
    except OSError:
        pass

    # Chequeo final
    if not all(os.path.exists(os.path.join(DEST_DIR, f)) for f in necesarios):
        raise RuntimeError("Faltan archivos de modelos luego de extraer el ZIP.")

    return DEST_DIR

# Llamalo donde cargás los modelos:
modelos_dir = ensure_modelos_drive()

@st.cache_resource(show_spinner=False)
def cargar_modelos_y_tablas(model_dir: str):
    # Ajustá nombres si tus ficheros tienen otros
    import joblib, pandas as pd, os
    scaler_subtipo = joblib.load(os.path.join(modelos_dir, "scaler.pkl"))
    model_subtipo  = joblib.load(os.path.join(modelos_dir, "SVM_best_model.pkl"))
    scaler_host    = joblib.load(os.path.join(modelos_dir, "scaler.pkl"))
    model_host     = joblib.load(os.path.join(modelos_dir, "KNN_best_model.pkl"))
    motivos        = pd.read_csv(os.path.join(modelos_dir, "cleavage_sites_H5_H7_extended.csv"))
    return scaler_subtipo, model_subtipo, scaler_host, model_host, motivos

def guardar_csv(df: pd.DataFrame, path_csv: str):
    df.to_csv(path_csv, index=False, encoding="utf-8")

def cargar_csv(path_csv: str, cols):
    if os.path.exists(path_csv):
        try:
            return pd.read_csv(path_csv, dtype=str).fillna("")
        except Exception:
            return pd.DataFrame(columns=cols, dtype=str)
    else:
        return pd.DataFrame(columns=cols, dtype=str)

# =========================
# UI
# =========================
st.set_page_config(page_title="Clasificador Influenza A – MGAP DILAVE", layout="wide")
st.title("🧬 Clasificador de Influenza A – MGAP DILAVE")

with st.sidebar:
    st.header("⚙️ Configuración")
    modelos_dir = st.text_input(
        "Carpeta de modelos",
        value="modelos",
        help="Debe contener scaler.pkl, SVM_best_model.pkl, KNN_best_model.pkl y cleavage_sites_H5_H7_extended.csv",
    )
    csv_path = st.text_input("Archivo CSV de resultados", value="resultados_influenza.csv")
    st.caption("Si ves advertencias de versión de scikit-learn, ocurre cuando los modelos se entrenaron con otra versión.")

# Cargar modelos/motivos
try:
    scaler_subtipo, model_subtipo, scaler_host, model_host, motivos = cargar_modelos_y_tablas(modelos_dir)
    modelos_ok = True
except Exception as e:
    modelos_ok = False
    st.error(f"No pude cargar modelos/archivos desde **{modelos_dir}**. Detalle: {e}")

# Estado de resultados
cols = ["ID", "Hospedero", "Predicho", "Subtipo", "Patogenicidad", "Lat", "Lon"]
if "resultados" not in st.session_state:
    st.session_state["resultados"] = cargar_csv(csv_path, cols)

# Layout principal
col_form, col_map = st.columns([0.38, 0.62], gap="large")

with col_form:
    st.subheader("📥 Datos de la muestra")

    id_muestra = st.text_input("ID de muestra")
    hosp_decl  = st.text_input("Hospedero declarado (texto libre)")
    c1, c2 = st.columns(2)
    with c1:
        lat = st.text_input("Latitud (ej: -34.9)")
    with c2:
        lon = st.text_input("Longitud (ej: -56.2)")
    sec = st.text_area("Secuencia FASTA (sin encabezado)", height=140, placeholder="Pegá aquí la secuencia…")

    puede_clasificar = modelos_ok and all([id_muestra.strip(), hosp_decl.strip(), lat.strip(), lon.strip(), sec.strip()])
    btn = st.button("🔍 Clasificar y agregar al mapa/tabla", use_container_width=True, disabled=not puede_clasificar)

    if btn:
        try:
            latf = float(lat)
            lonf = float(lon)

            DPC = calcular_DPC(sec)
            X1 = scaler_subtipo.transform(DPC)
            subtipo = f"H{int(model_subtipo.predict(X1)[0]) + 1}"

            X2 = scaler_host.transform(DPC)
            mapa_host = {0: "Aves", 1: "Cerdos", 2: "Humano"}
            host_pred = mapa_host[int(model_host.predict(X2)[0])]

            # Patogenicidad simple por motivo
            motivos_det = detectar_sitio_clivaje(sec, motivos) if subtipo in ["H5", "H7"] else ""
            if subtipo in ["H5", "H7"]:
                patogenicidad = "Alta" if ("RRR" in motivos_det or "KRR" in motivos_det) else "Baja"
            else:
                patogenicidad = "No aplica"

            nuevo = {
                "ID": id_muestra.strip(),
                "Hospedero": hosp_decl.strip(),
                "Predicho": host_pred,
                "Subtipo": subtipo,
                "Patogenicidad": patogenicidad,
                "Lat": f"{latf}",
                "Lon": f"{lonf}",
            }
            st.session_state["resultados"] = pd.concat(
                [st.session_state["resultados"], pd.DataFrame([nuevo])],
                ignore_index=True
            ).fillna("").astype(str)

            guardar_csv(st.session_state["resultados"], csv_path)

            st.success("✅ Clasificación agregada y guardada en el CSV.")
            with st.expander("Ver detalle de la clasificación agregada", expanded=True):
                st.write(f"**ID:** {id_muestra}")
                st.write(f"**Hospedero declarado:** {hosp_decl}")
                st.write(f"**Hospedero predicho:** {host_pred}")
                st.write(f"**Subtipo:** {subtipo}")
                st.write(f"**Patogenicidad:** {patogenicidad}")
                if motivos_det:
                    st.write("**Motivos detectados:**")
                    st.code(motivos_det, language="text")

        except Exception as e:
            st.error(f"Ocurrió un error al clasificar: {e}")

    st.markdown("---")
    st.subheader("📄 Resultados (CSV en disco)")
    st.dataframe(
        st.session_state["resultados"],
        width="stretch",   # reemplaza use_container_width
        hide_index=True
    )

with col_map:
    st.subheader("🗺️ Mapa")

    df_map = st.session_state["resultados"].copy()
    # Convertir coords a float válidas
    for c in ("Lat", "Lon"):
        df_map[c] = pd.to_numeric(df_map[c], errors="coerce")
    df_map = df_map.dropna(subset=["Lat", "Lon"])

    # Paleta por hospedero predicho
    def color_by_host(h):
        return {
            "Aves":   [66, 165, 245],   # azul
            "Cerdos": [239, 83, 80],    # rojo
            "Humano": [102, 187, 106],  # verde
        }.get(h, [25, 118, 210])

    if not df_map.empty:
        df_map["color"] = df_map["Predicho"].apply(color_by_host)

        # Vista inicial
        lat_center = float(df_map["Lat"].mean())
        lon_center = float(df_map["Lon"].mean())

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_map,
            get_position='[Lon, Lat]',
            get_radius=7000,             # radio en metros (ajustá a gusto)
            get_fill_color="color",
            pickable=True
        )

        tooltip = {
            "html": "<b>ID:</b> {ID} <br/>"
                    "<b>Hosp. declarado:</b> {Hospedero} <br/>"
                    "<b>Predicho:</b> {Predicho} <br/>"
                    "<b>Subtipo:</b> {Subtipo} <br/>"
                    "<b>Patogenicidad:</b> {Patogenicidad}",
            "style": {"backgroundColor": "white", "color": "black"}
        }

        st.pydeck_chart(pdk.Deck(
            initial_view_state=pdk.ViewState(
                latitude=lat_center, longitude=lon_center, zoom=4
            ),
            layers=[layer],
            tooltip=tooltip,
            map_style=None  # usa base por defecto (no requiere token)
        ))
    else:
        # Mapa vacío centrado en Uruguay como default
        st.pydeck_chart(pdk.Deck(
            initial_view_state=pdk.ViewState(latitude=-32.5, longitude=-55.8, zoom=4),
            layers=[],
            map_style=None
        ))
        st.info("Aún no hay puntos para mostrar. Agregá una muestra con coordenadas.")


