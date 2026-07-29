# aiv_app_streamlit.py
import os
from pathlib import Path

import itertools
import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import zipfile
import gdown
import requests  

st.set_page_config(page_title="Influenza A Virus Classifier", layout="wide")

# =========================
# Helpers de features y motivos
# =========================
def calcular_DPC(sec: str) -> pd.DataFrame:
    """Vector de 400 dipeptidos normalizados (20x20)."""
    if not isinstance(sec, str):
        sec = str(sec or "")
    sec = sec.upper().replace("\n", "").strip()
    aminoacidos = "ACDEFGHIKLMNPQRSTVWY"
    DPC = [''.join(p) for p in itertools.product(aminoacidos, repeat=2)]
    dpc = {d: 0 for d in DPC}
    if len(sec) < 2:
        return pd.DataFrame([np.zeros(400)], columns=list(range(400)))

    for i in range(len(sec) - 1):
        par = sec[i:i + 2]
        if par in dpc:
            dpc[par] += 1

    total = len(sec) - 1
    if total > 0:
        for k in dpc:
            dpc[k] /= total

    #Los modelos fueron entrenados con vectores de 400, se guardan igual
    dpc_vec = pd.DataFrame([list(dpc.values())], columns=list(range(400)))
    return dpc_vec


import re
import pandas as pd

def detectar_sitio_clivaje(secuencia: str, motivos: pd.DataFrame, ventana_max=14, tolerar_basico_final=True) -> str:
    """Busca motivos de clivaje inmediatamente antes de 'GLF' (ventana P4..P14)."""
    if motivos is None or motivos.empty:
        return "Tabla de motivos no cargada"

    # 1) Normalizar secuencia
    secuencia = re.sub(r'[^A-Z]', '', (secuencia or '').upper())  # quita \n, espacios, etc.

    # 2) Normalizar tabla
    m = motivos.copy()
    m["Cleavage_Site"] = m["Cleavage_Site"].astype(str).str.strip().str.upper()

    # índice rápido: motivo -> fila
    idx = {row.Cleavage_Site: row for _, row in m.iterrows()}

    encontrados = []
    for i in range(len(secuencia) - 2):
        if secuencia[i:i+3] == "GLF":
            # ventana de P4..P14 (4 a 14 aa antes de GLF)
            candidatos = []
            for size in range(4, ventana_max + 1):
                inicio = i - size
                if inicio < 0:
                    break
                motivo = secuencia[inicio:i]

                # Coincidencia exacta
                if motivo in idx:
                    candidatos.append(motivo)
                    continue

                # Coincidencia flexible: permite un R/K extra al final
                if tolerar_basico_final and motivo[:-1] in idx and motivo[-1] in "RK":
                    candidatos.append(motivo)

            if candidatos:
                # prioriza el más largo (más específico)
                motivo_sel = max(candidatos, key=len)
                clave = motivo_sel if motivo_sel in idx else motivo_sel[:-1]
                info = idx[clave]
                encontrados.append(
                    f"- Motivo: {motivo_sel} | Subtipo: {info.get('Subtype','NA')} | Clado/Tipo: {info.get('Clade_or_Type','NA')}"
                )

    return "\n".join(encontrados) if encontrados else "Ningún motivo detectado"



# =========================
# Descarga y preparación de modelos (Google Drive)
# =========================
DRIVE_ID = "1orIsijhlHdxrr8FjYnaEG6_5z24VOnFn"

@st.cache_data(show_spinner=True)
def ensure_modelos_drive() -> str:
    MODELOS_DIR = Path("modelos_v1")
    if MODELOS_DIR.exists() and any(MODELOS_DIR.iterdir()):
        return str(MODELOS_DIR)

    TMP_ZIP = Path("/tmp/modelos.zip")
    if TMP_ZIP.exists():
        TMP_ZIP.unlink()

    # 1) Intento por ID 
    try:
        ok = gdown.download(id=DRIVE_ID, output=str(TMP_ZIP), quiet=False, use_cookies=True)
        if not ok or not TMP_ZIP.exists() or TMP_ZIP.stat().st_size < 1024:
            raise RuntimeError("Descarga incompleta desde Drive (archivo muy pequeño).")
    except Exception as e_id:
        # 2) Fallback opcional (mirror HTTP)
        if not FALLBACK_URL:
            raise RuntimeError(f"Fallo Drive por ID: {e_id}. Configura FALLBACK_URL o revisa permisos/cuota.") from e_id
        r = requests.get(FALLBACK_URL, stream=True, timeout=60)
        r.raise_for_status()
        with open(TMP_ZIP, "wb") as f:
            for c in r.iter_content(1 << 20):
                if c:
                    f.write(c)
        if TMP_ZIP.stat().st_size < 1024:
            raise RuntimeError("Fallback HTTP descargó un archivo demasiado pequeño.")

    # 3) Descomprimir
    MODELOS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(TMP_ZIP, "r") as zf:
            zf.extractall(MODELOS_DIR)
    except zipfile.BadZipFile:
        raise RuntimeError("El archivo descargado no es un ZIP válido (¿Drive devolvió HTML?).")

    return str(MODELOS_DIR)


modelos_dir = ensure_modelos_drive()

@st.cache_resource(show_spinner=False)
def cargar_modelos_y_tablas(model_dir: str):
    scaler_subtipo = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    model_subtipo  = joblib.load(os.path.join(model_dir, "SVM_best_model.pkl"))
    scaler_host    = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    model_host     = joblib.load(os.path.join(model_dir, "KNN_best_model.pkl"))
    motivos        = pd.read_csv(os.path.join(model_dir, "cleavage_sites_H5_H7_extended.csv"))
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


# Ayuda simple arriba de la app
# ---------------------------
st.title("🧬 Influenza A Virus Classifier (Beta)")

# ---------------------------
# Ayuda compacta arriba de la app
# ---------------------------
if "show_help" not in st.session_state:
    st.session_state.show_help = True   # se muestra la primera vez

if st.session_state.show_help:
    with st.container(border=True):
        col_texto, col_cerrar = st.columns([0.92, 0.08])

        with col_texto:
            st.markdown(
                """
                <div style="font-size:0.80rem; line-height:1.25;">
                  <b>How to use the app</b><br>
                  1. Enter sample <b>ID</b>.<br>
                  2. Enter <b>Host declared</b>.<br>
                  3. Load <b>latitud</b> y <b>longitud</b>.<br>
                  4. Paste the <b>HA sequence</b> (without header).<br>
                  5. Click on <b>“Classify and add to Map/tabla”</b>.<br><br>
                  The app will display the predicted influenza A subtype, host of origin, pathogenicity for H5 and H7 subtypes, and the sample’s location on a map.
                </div>
                """,
                unsafe_allow_html=True
            )

        with col_cerrar:
            st.button("✖", key="cerrar_ayuda", on_click=lambda: st.session_state.__setitem__("show_help", False))

with st.sidebar:
    st.header("⚙️ Configuración")
    modelos_dir = st.text_input(
        "Carpeta de modelos",
        value=modelos_dir,
        help="Debe contener scaler.pkl, SVM_best_model.pkl, KNN_best_model.pkl y cleavage_sites_H5_H7_extended.csv",
    )
    csv_path = st.text_input("Archivo CSV de resultados", value="resultados_influenza.csv")
    st.caption("Si ves advertencias de versión de scikit-learn, es porque los modelos se entrenaron con otra versión.")

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
    st.subheader("📥 Sample Data")

    id_muestra = st.text_input("Sample ID")
    hosp_decl  = st.text_input("Host declared (free text)")
    c1, c2 = st.columns(2)
    with c1:
        lat = st.text_input("Latitud (ej: -34.9)")
    with c2:
        lon = st.text_input("Longitud (ej: -56.2)")
    sec = st.text_area("FASTA sequence (without header)", height=140, placeholder="Pegá aquí la secuencia…")

    puede_clasificar = modelos_ok and all([id_muestra.strip(), hosp_decl.strip(), lat.strip(), lon.strip(), sec.strip()])
    btn = st.button("🔍 Clssify and add to map/table", use_container_width=True, disabled=not puede_clasificar)

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

            motivos_det = detectar_sitio_clivaje(sec, motivos) if subtipo in ["H5", "H7"] else ""
            if subtipo in ["H5", "H7"]:
                patogenicidad = "Alta" if ("HPAI" in motivos_det) else "Baja"
            else:
                patogenicidad = "No aplica"

            nuevo = {
                "ID": id_muestra.strip(),
                "Host": hosp_decl.strip(),
                "Predicted Host": host_pred,
                "Subtype": subtipo,
                "Patogenicity": patogenicidad,
                "Lat": f"{latf}",
                "Lon": f"{lonf}",
            }
            st.session_state["resultados"] = pd.concat(
                [st.session_state["resultados"], pd.DataFrame([nuevo])],
                ignore_index=True
            ).fillna("").astype(str)

            guardar_csv(st.session_state["resultados"], csv_path)

            st.success("✅ Classification completed and saved")
            with st.expander("Ver detalle de la clasificación agregada", expanded=True):
                st.write(f"**ID:** {id_muestra}")
                st.write(f"**Declared Host:** {hosp_decl}")
                st.write(f"**Predicted host origin:** {host_pred}")
                st.write(f"**Subtype:** {subtipo}")
                st.write(f"**Patogenicity:** {patogenicidad}")
                if motivos_det:
                    st.write("**Detected Motivs:**")
                    st.code(motivos_det, language="text")

        except Exception as e:
            st.error(f"Ocurrió un error al clasificar: {e}")

    st.markdown("---")
    st.subheader("📄 Results")
    st.dataframe(
        st.session_state["resultados"],
        use_container_width=True,
        hide_index=True
    )

with col_map:
    st.subheader("🗺️ Map")

    df_map = st.session_state["resultados"].copy()
    # Convertir coords a float válidas
    for c in ("Lat", "Lon"):
        df_map[c] = pd.to_numeric(df_map[c], errors="coerce")
    df_map = df_map.dropna(subset=["Lat", "Lon"])

    def color_by_host(h):
        return {
            "Aves":   [66, 165, 245],
            "Cerdos": [239, 83, 80],
            "Humano": [102, 187, 106],
        }.get(h, [25, 118, 210])

    if not df_map.empty:
        df_map["color"] = df_map["Predicho"].apply(color_by_host)
        lat_center = float(df_map["Lat"].mean())
        lon_center = float(df_map["Lon"].mean())

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_map,
            get_position='[Lon, Lat]',
            get_radius=7000,
            get_fill_color="color",
            pickable=True
        )

        tooltip = {
            "html": "<b>ID:</b> {ID} <br/>"
            "<b>Declared host:</b> {Hospedero} <br/>"
            "<b>Predicted host:</b> {Predicho} <br/>"
            "<b>Subtype:</b> {Subtipo} <br/>"
            "<b>Pathogenicity:</b> {Patogenicidad}",
            "style": {"backgroundColor": "white", "color": "black"}
        }
        st.pydeck_chart(pdk.Deck(
            initial_view_state=pdk.ViewState(
                latitude=lat_center, longitude=lon_center, zoom=4
            ),
            layers=[layer],
            tooltip=tooltip,
            map_style=None
        ))
    else:
        st.pydeck_chart(pdk.Deck(
            initial_view_state=pdk.ViewState(latitude=-32.5, longitude=-55.8, zoom=4),
            layers=[],
            map_style=None
        ))
        st.info("No points to display yet. Add a sample with coordinates.")














































