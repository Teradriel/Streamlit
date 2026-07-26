import json
import os
import time
from io import StringIO

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

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

    features_combate = ["hp", "ataque", "defensa", "ataque_esp", "defensa_esp", "velocidad"]
    escalador = StandardScaler()
    combate_escalado = escalador.fit_transform(pokemon[features_combate])

    kmeans = KMeans(n_clusters=5, random_state=123, n_init=10)
    pokemon["cluster"] = kmeans.fit_predict(combate_escalado)

    nombres_cluster = {
        0: "Forma base / sin evolucionar",
        1: "Atacante especial",
        2: "Atacante fisico robusto",
        3: "Barredor veloz",
        4: "Muro defensivo",
    }
    pokemon["arquetipo"] = pokemon["cluster"].map(nombres_cluster)

    return pokemon


@st.cache_data(show_spinner=False)
def evaluar_modelo_supervisado(df_modelo):
    features_num = ["hp", "ataque", "defensa", "ataque_esp", "defensa_esp", "velocidad"]
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    tipo_encoded = encoder.fit_transform(df_modelo[["tipo_1"]])
    tipo_df = pd.DataFrame(tipo_encoded, columns=encoder.get_feature_names_out(), index=df_modelo.index)

    x = pd.concat([df_modelo[features_num], tipo_df], axis=1)
    y = df_modelo["pseudo_legendario"]

    modelo = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=123)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

    f1 = cross_val_score(modelo, x, y, cv=cv, scoring="f1").mean()
    precision = cross_val_score(modelo, x, y, cv=cv, scoring="precision").mean()
    recall = cross_val_score(modelo, x, y, cv=cv, scoring="recall").mean()

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "n_positivos": int(y.sum()),
        "n_total": int(len(y)),
    }


with st.spinner("Construyendo dataset (primera ejecucion puede tardar)..."):
    df = construir_dataset()

st.sidebar.header("Filtros")

tipos_disponibles = sorted(df["tipo_1"].dropna().unique().tolist())
tipos_sel = st.sidebar.multiselect("Tipo principal", tipos_disponibles, default=tipos_disponibles)

arquetipos_disponibles = ["Todos"] + sorted(df["arquetipo"].dropna().unique().tolist())
filtro_arquetipo = st.sidebar.selectbox("Arquetipo", arquetipos_disponibles)

min_stats = int(df["suma_estadisticas"].min())
max_stats = int(df["suma_estadisticas"].max())
rango_stats = st.sidebar.slider("Rango suma de estadisticas", min_stats, max_stats, (min_stats, max_stats))

filtro_legendario = st.sidebar.selectbox("Categoria", ["Todos", "Solo legendarios", "Solo no legendarios", "Solo pseudo-legendarios"])

filtrado = df[df["tipo_1"].isin(tipos_sel)].copy()
filtrado = filtrado[filtrado["suma_estadisticas"].between(rango_stats[0], rango_stats[1])]

if filtro_arquetipo != "Todos":
    filtrado = filtrado[filtrado["arquetipo"] == filtro_arquetipo]

if filtro_legendario == "Solo legendarios":
    filtrado = filtrado[filtrado["legendario"]]
elif filtro_legendario == "Solo pseudo-legendarios":
    filtrado = filtrado[filtrado["pseudo_legendario"]]
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
    tipos_top = filtrado["tipo_1"].value_counts().index.tolist()
    base_box = filtrado[filtrado["tipo_1"].isin(tipos_top)][["tipo_1", "suma_estadisticas"]]
    fig3, ax3 = plt.subplots(figsize=(11, 5))
    base_box.boxplot(column="suma_estadisticas", by="tipo_1", ax=ax3, rot=25)
    ax3.set_xlabel("Tipo")
    ax3.set_ylabel("Suma de estadisticas")
    ax3.set_title("Dispersion por tipo principal")
    plt.suptitle("")
    plt.tight_layout()
    st.pyplot(fig3)

    if filtro_arquetipo == "Todos":
        st.markdown("**4) Distribucion de arquetipos**")
        conteo_arquetipos = filtrado["arquetipo"].value_counts().sort_values(ascending=True)
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        ax4.barh(conteo_arquetipos.index, conteo_arquetipos.values)
        ax4.set_xlabel("Cantidad de Pokemon")
        ax4.set_ylabel("Arquetipo")
        ax4.set_title("Pokemon por arquetipo")
        st.pyplot(fig4)

st.subheader("Hallazgos y conclusiones")

if len(filtrado) == 0:
    st.write("No se pueden generar hallazgos porque el filtro actual no tiene registros.")
else:
    tipo_mas_frecuente = filtrado["tipo_1"].value_counts().idxmax()
    prom_stats = filtrado["suma_estadisticas"].mean()
    pct_legend = (filtrado["legendario"].mean() * 100) if len(filtrado) else 0
    top_ataque = filtrado.sort_values("ataque", ascending=False).head(1)["nombre_es"].iloc[0]
    top_defensa = filtrado.sort_values("defensa", ascending=False).head(1)["nombre_es"].iloc[0]
    top_velocidad = filtrado.sort_values("velocidad", ascending=False).head(1)["nombre_es"].iloc[0]
    top_hp = filtrado.sort_values("hp", ascending=False).head(1)["nombre_es"].iloc[0]
    top_ataque_esp = filtrado.sort_values("ataque_esp", ascending=False).head(1)["nombre_es"].iloc[0]
    top_defensa_esp = filtrado.sort_values("defensa_esp", ascending=False).head(1)["nombre_es"].iloc[0]
    corr_altura_velocidad = filtrado["altura_cm"].corr(filtrado["velocidad"])
    corr_peso_defensa = filtrado["peso_kg"].corr(filtrado["defensa"])
    hay_legendario_bicho = bool(((filtrado["legendario"]) & (filtrado["tipo_1"] == "Bicho")).any())

    st.markdown(
        f"""
- El tipo mas frecuente en la seleccion actual es **{tipo_mas_frecuente}**.
- El arquetipo mas frecuente en la seleccion actual es **{filtrado['arquetipo'].value_counts().idxmax()}**.
- La suma promedio de estadisticas en el recorte filtrado es **{prom_stats:.1f}**.
- Los legendarios representan **{pct_legend:.1f}%** de los datos filtrados.
- El Pokemon con mayor ataque en esta vista es **{top_ataque}**.
- El Pokemon con mayor defensa en esta vista es **{top_defensa}**.
- El Pokemon con mayor velocidad en esta vista es **{top_velocidad}**.
- El Pokemon con mayor vida en esta vista es **{top_hp}**.
- El Pokemon con mayor ataque especial en esta vista es **{top_ataque_esp}**.
- El Pokemon con mayor defensa especial en esta vista es **{top_defensa_esp}**.
- La relacion entre tamaño fisico y combate es debil: altura vs velocidad **{corr_altura_velocidad:.2f}** y peso vs defensa **{corr_peso_defensa:.2f}**.
- Esto coincide con el hallazgo extra del notebook original: el tamaño fisico aporta una pista menor, pero no separa por si solo los perfiles de combate.
- El filtro de arquetipos ayuda a ver que los Pokemon se agrupan mejor por perfil de stats que por tipo solamente.
- Legendarios de tipo Bicho en la vista actual: **{'si' if hay_legendario_bicho else 'no'}**.
        """
    )

st.subheader("Resultados de minería del proyecto original")

resultados_modelo = evaluar_modelo_supervisado(df)
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("F1 (CV 5 folds)", f"{resultados_modelo['f1']:.3f}")
col_m2.metric("Precision", f"{resultados_modelo['precision']:.3f}")
col_m3.metric("Recall", f"{resultados_modelo['recall']:.3f}")
col_m4.metric("Pseudo-legendarios", f"{resultados_modelo['n_positivos']}/{resultados_modelo['n_total']}")

st.caption(
    "Estos indicadores replican la etapa supervisada del notebook: "
    "arbol de decision con max_depth=3, class_weight='balanced' y validacion cruzada estratificada."
)

stats_cols = ["hp", "ataque", "defensa", "ataque_esp", "defensa_esp", "velocidad"]
perfil_stats = filtrado.groupby("arquetipo")[stats_cols].mean().round(1)

if len(perfil_stats) > 0:
    st.markdown("**Perfil promedio de stats por arquetipo (subset filtrado)**")
    fig5, ax5 = plt.subplots(figsize=(11, 4.8))
    perfil_stats.plot(kind="bar", ax=ax5)
    ax5.set_xlabel("Arquetipo")
    ax5.set_ylabel("Valor promedio")
    ax5.set_title("Comparacion de stats por arquetipo")
    ax5.tick_params(axis="x", labelrotation=20)
    ax5.legend(title="Stat", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    st.pyplot(fig5)

st.subheader("Datos filtrados")
st.dataframe(
    filtrado[
        [
            "#",
            "nombre_es",
            "tipo_1",
            "tipo_2",
            "arquetipo",
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
