import json
import os
import time
from io import StringIO

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

pd.set_option("future.no_silent_downcasting", True)

HEADERS = {"User-Agent": "Mozilla/5.0 (Educational bot - Mineria de Datos)"}
CACHE_POKEMON = "pokeapi_cache.json"
CACHE_SPECIES = "pokeapi_especies_cache.json"


st.set_page_config(page_title="Dashboard Pokemon - Mineria de Datos", layout="wide")
st.title("Dashboard Pokemon - Mineria de Datos")
st.caption("Vista simplificada para entrega: filtros, visualizaciones y hallazgos.")


def cargar_cache(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def guardar_cache(cache, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)


def obtener_pokemon_api(pokedex_id, cache, reintentos=3):
    key = str(pokedex_id)
    if key in cache:
        return cache[key]

    url = f"https://pokeapi.co/api/v2/pokemon/{pokedex_id}"
    for intento in range(1, reintentos + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=12)
            if resp.status_code == 200:
                data = resp.json()
                cache[key] = data
                return data
            if resp.status_code == 404:
                return None
        except requests.exceptions.RequestException:
            pass
        time.sleep(1.2 * intento)
    return None


def obtener_especie_api(pokedex_id, cache, reintentos=3):
    key = str(pokedex_id)
    if key in cache:
        return cache[key]

    url = f"https://pokeapi.co/api/v2/pokemon-species/{pokedex_id}"
    for intento in range(1, reintentos + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=12)
            if resp.status_code == 200:
                data = resp.json()
                cache[key] = data
                return data
            if resp.status_code == 404:
                return None
        except requests.exceptions.RequestException:
            pass
        time.sleep(1.2 * intento)
    return None


@st.cache_data(show_spinner=False)
def construir_dataset():
    url_lista = "https://www.wikidex.net/wiki/Lista_de_Pok%C3%A9mon"
    resp_lista = requests.get(url_lista, headers=HEADERS, timeout=20)
    tablas = pd.read_html(StringIO(resp_lista.text))

    lista_pokemon = pd.concat(
        [tablas[2], tablas[4], tablas[6], tablas[8], tablas[10], tablas[12], tablas[14], tablas[16], tablas[18]],
        ignore_index=True,
    )
    lista_pokemon = lista_pokemon[["#", "Nombre"]]
    lista_pokemon.drop_duplicates(subset="#", keep="first", inplace=True)

    cache_pokemon = cargar_cache(CACHE_POKEMON)
    cache_species = cargar_cache(CACHE_SPECIES)

    ids_pokedex = sorted(lista_pokemon["#"].dropna().astype(int).unique())

    rows = []
    for pid in ids_pokedex:
        data = obtener_pokemon_api(pid, cache_pokemon)
        if data is None:
            continue

        stats = {s["stat"]["name"]: s["base_stat"] for s in data["stats"]}
        tipos_api = [t["type"]["name"] for t in data["types"]]

        rows.append(
            {
                "#": data["id"],
                "nombre_en": data["name"],
                "altura_dm": data["height"],
                "peso_hg": data["weight"],
                "tipo_1": tipos_api[0] if len(tipos_api) > 0 else None,
                "tipo_2": tipos_api[1] if len(tipos_api) > 1 else None,
                "hp": stats.get("hp"),
                "ataque": stats.get("attack"),
                "defensa": stats.get("defense"),
                "ataque_esp": stats.get("special-attack"),
                "defensa_esp": stats.get("special-defense"),
                "velocidad": stats.get("speed"),
            }
        )

    guardar_cache(cache_pokemon, CACHE_POKEMON)
    pokemon = pd.DataFrame(rows)

    es_legendario = {}
    es_mitico = {}
    for pid in ids_pokedex:
        data = obtener_especie_api(pid, cache_species)
        if data is None:
            continue
        es_legendario[pid] = data["is_legendary"]
        es_mitico[pid] = data["is_mythical"]

    guardar_cache(cache_species, CACHE_SPECIES)

    traduccion_tipos = {
        "normal": "Normal",
        "fire": "Fuego",
        "water": "Agua",
        "electric": "Electrico",
        "grass": "Planta",
        "ice": "Hielo",
        "fighting": "Lucha",
        "poison": "Veneno",
        "ground": "Tierra",
        "flying": "Volador",
        "psychic": "Psiquico",
        "bug": "Bicho",
        "rock": "Roca",
        "ghost": "Fantasma",
        "dragon": "Dragon",
        "dark": "Siniestro",
        "steel": "Acero",
        "fairy": "Hada",
    }

    pokemon["legendario"] = pokemon["#"].map(es_legendario).fillna(False)
    pokemon["mitico"] = pokemon["#"].map(es_mitico).fillna(False)
    pokemon["tipo_1"] = pokemon["tipo_1"].map(traduccion_tipos)
    pokemon["tipo_2"] = pokemon["tipo_2"].map(traduccion_tipos, na_action="ignore")

    pokemon["altura_cm"] = pokemon["altura_dm"] * 10
    pokemon["peso_kg"] = pokemon["peso_hg"] / 10
    pokemon["suma_estadisticas"] = pokemon[["hp", "ataque", "defensa", "ataque_esp", "defensa_esp", "velocidad"]].sum(axis=1)

    nombre_por_id = dict(zip(lista_pokemon["#"], lista_pokemon["Nombre"]))
    pokemon["nombre_es"] = pokemon["#"].map(nombre_por_id).str.lower()

    pokemon["pseudo_legendario"] = (~pokemon["legendario"]) & (~pokemon["mitico"]) & (pokemon["suma_estadisticas"] == 600)

    return pokemon


with st.spinner("Construyendo dataset (primera ejecucion puede tardar)..."):
    df = construir_dataset()

st.sidebar.header("Filtros")

tipos_disponibles = sorted(df["tipo_1"].dropna().unique().tolist())
tipos_sel = st.sidebar.multiselect("Tipo principal", tipos_disponibles, default=tipos_disponibles)

min_stats = int(df["suma_estadisticas"].min())
max_stats = int(df["suma_estadisticas"].max())
rango_stats = st.sidebar.slider("Rango suma de estadisticas", min_stats, max_stats, (min_stats, max_stats))

filtro_legendario = st.sidebar.selectbox("Categoria", ["Todos", "Solo legendarios", "Solo no legendarios"])

filtrado = df[df["tipo_1"].isin(tipos_sel)].copy()
filtrado = filtrado[filtrado["suma_estadisticas"].between(rango_stats[0], rango_stats[1])]

if filtro_legendario == "Solo legendarios":
    filtrado = filtrado[filtrado["legendario"]]
elif filtro_legendario == "Solo no legendarios":
    filtrado = filtrado[~filtrado["legendario"]]

st.subheader("Resumen")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Registros", f"{len(filtrado):,}")
c2.metric("Tipos visibles", f"{filtrado['tipo_1'].nunique()}")
c3.metric("Prom. suma stats", f"{filtrado['suma_estadisticas'].mean():.1f}" if len(filtrado) else "-")
c4.metric("Pseudo-legendarios", f"{int(filtrado['pseudo_legendario'].sum())}" if len(filtrado) else "0")

st.subheader("Visualizaciones")

if len(filtrado) == 0:
    st.warning("No hay datos con los filtros seleccionados.")
else:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**1) Distribucion por tipo principal**")
        conteo_tipos = filtrado["tipo_1"].value_counts().head(10).sort_values(ascending=True)
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.barh(conteo_tipos.index, conteo_tipos.values)
        ax1.set_xlabel("Cantidad de Pokemon")
        ax1.set_ylabel("Tipo")
        ax1.set_title("Top 10 tipos mas frecuentes")
        st.pyplot(fig1)

    with col_b:
        st.markdown("**2) Ataque vs Velocidad**")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.scatter(filtrado["ataque"], filtrado["velocidad"], alpha=0.55)
        ax2.set_xlabel("Ataque")
        ax2.set_ylabel("Velocidad")
        ax2.set_title("Relacion entre ataque y velocidad")
        st.pyplot(fig2)

    st.markdown("**3) Suma de estadisticas por tipo (boxplot)**")
    tipos_top = filtrado["tipo_1"].value_counts().head(8).index.tolist()
    base_box = filtrado[filtrado["tipo_1"].isin(tipos_top)][["tipo_1", "suma_estadisticas"]]
    fig3, ax3 = plt.subplots(figsize=(11, 5))
    base_box.boxplot(column="suma_estadisticas", by="tipo_1", ax=ax3, rot=25)
    ax3.set_xlabel("Tipo")
    ax3.set_ylabel("Suma de estadisticas")
    ax3.set_title("Dispersion por tipo principal (top 8)")
    plt.suptitle("")
    plt.tight_layout()
    st.pyplot(fig3)

st.subheader("Hallazgos y conclusiones")

if len(filtrado) == 0:
    st.write("No se pueden generar hallazgos porque el filtro actual no tiene registros.")
else:
    tipo_mas_frecuente = filtrado["tipo_1"].value_counts().idxmax()
    prom_stats = filtrado["suma_estadisticas"].mean()
    pct_legend = (filtrado["legendario"].mean() * 100) if len(filtrado) else 0
    top_ataque = filtrado.sort_values("ataque", ascending=False).head(1)["nombre_es"].iloc[0]

    st.markdown(
        f"""
- El tipo mas frecuente en la seleccion actual es **{tipo_mas_frecuente}**.
- La suma promedio de estadisticas en el recorte filtrado es **{prom_stats:.1f}**.
- Los legendarios representan **{pct_legend:.1f}%** de los datos filtrados.
- El Pokemon con mayor ataque en esta vista es **{top_ataque}**.
        """
    )

st.subheader("Datos filtrados")
st.dataframe(
    filtrado[
        [
            "#",
            "nombre_es",
            "tipo_1",
            "tipo_2",
            "hp",
            "ataque",
            "defensa",
            "ataque_esp",
            "defensa_esp",
            "velocidad",
            "altura_cm",
            "peso_kg",
            "legendario",
            "mitico",
            "pseudo_legendario",
            "suma_estadisticas",
        ]
    ],
    use_container_width=True,
)
