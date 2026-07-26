# Mineria de datos - Final 2026

import streamlit as st
import requests
import matplotlib.pyplot as plt
import pandas as pd
import time
import json
import os
from io import StringIO
from bs4 import BeautifulSoup
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict, LeaveOneOut
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, silhouette_score
from sklearn.tree import export_text
from sklearn.cluster import KMeans

pd.set_option('future.no_silent_downcasting', True)

headers = {'User-Agent': 'Mozilla/5.0 (Educational bot - Mineria de Datos)'}

CACHE_PATH = 'pokeapi_cache.json'

CACHE_ESPECIES_PATH = 'pokeapi_especies_cache.json'

st.set_page_config(page_title='Mineria de Datos - Pokemon', layout='wide')
st.title('Mineria de datos - Final 2026')


def mostrar_info_df(df, titulo):
    buffer = StringIO()
    df.info(buf=buffer)
    st.subheader(titulo)
    st.text(buffer.getvalue())


def mostrar_df(titulo, df):
    st.subheader(titulo)
    st.dataframe(df, use_container_width=True)

def ver_robots(dominio):
    url = f'https://{dominio}/robots.txt'
    r = requests.get(url)
    if r.status_code == 200:
        st.write(f'=== robots.txt de {dominio} ===')
        st.write(r.text[:600])
    else:
        st.write(f'{dominio}: sin robots.txt (status {r.status_code})')
        
def cargar_cache(cache_path):
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def guardar_cache(cache, cache_path):
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False)

def obtener_pokemon_api(pokedex_id, cache, reintentos=3):
    """Obtiene los datos de un Pokémon desde PokeAPI, usando caché local.
    Devuelve el dict de datos o None si falla tras los reintentos."""
    clave = str(pokedex_id)

    if clave in cache:
        return cache[clave]

    url = f'https://pokeapi.co/api/v2/pokemon/{pokedex_id}'

    for intento in range(1, reintentos + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=10)
        except requests.exceptions.RequestException as e:
            st.write(f'[{pokedex_id}] error de conexión (intento {intento}/{reintentos}): {e}')
            time.sleep(1.5 * intento)  # backoff simple
            continue

        if resp.status_code == 200:
            datos = resp.json()
            cache[clave] = datos
            return datos
        elif resp.status_code == 404:
            st.write(f'[{pokedex_id}] no encontrado (404), se omite')
            return None
        else:
            st.write(f'[{pokedex_id}] status {resp.status_code} (intento {intento}/{reintentos})')
            time.sleep(1.5 * intento)

    st.write(f'[{pokedex_id}] falló tras {reintentos} intentos, se omite')
    return None

def obtener_especie_api(pokedex_id, cache, reintentos=3):
    clave = str(pokedex_id)
    if clave in cache:
        return cache[clave]

    url = f'https://pokeapi.co/api/v2/pokemon-species/{pokedex_id}'
    for intento in range(1, reintentos + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=10)
        except requests.exceptions.RequestException as e:
            st.write(f'[{pokedex_id}] error de conexión (intento {intento}/{reintentos}): {e}')
            time.sleep(1.5 * intento)
            continue

        if resp.status_code == 200:
            datos = resp.json()
            cache[clave] = datos
            return datos
        elif resp.status_code == 404:
            return None
        else:
            time.sleep(1.5 * intento)

    return None

ver_robots('www.wikidex.net')

url = 'https://www.wikidex.net/wiki/Lista_de_Pok%C3%A9mon'
respuesta = requests.get(url, headers=headers)

tablas_pokemon = pd.read_html(StringIO(respuesta.text))

st.write(f'Tablas encontradas: {len(tablas_pokemon)}')

for i, t in enumerate(tablas_pokemon[:50]):
    st.write(f'Tabla {i}: {t.shape} — columnas: {list(t.columns[:3])}')
    
lista_pokemon = pd.concat([tablas_pokemon[2], tablas_pokemon[4], tablas_pokemon[6], tablas_pokemon[8], tablas_pokemon[10], tablas_pokemon[12], tablas_pokemon[14], tablas_pokemon[16], tablas_pokemon[18]], ignore_index=True)

mostrar_info_df(lista_pokemon, 'Info de lista_pokemon (antes de limpieza)')

lista_pokemon = lista_pokemon[['#', 'Nombre']]

lista_pokemon.drop_duplicates(subset='#', keep='first', inplace=True)

lista_pokemon.to_csv('pokemon.csv', index=False, encoding='utf-8-sig')

url = 'https://www.wikidex.net/wiki/Tipo'
respuesta = requests.get(url, headers=headers)

tablas_tipos = pd.read_html(StringIO(respuesta.text)) 

st.write(f'Tablas encontradas: {len(tablas_tipos)}')

for i, t in enumerate(tablas_tipos[:50]):
    st.write(f'Tabla {i}: {t.shape} — columnas: {list(t.columns[:10])}')

url_tipos = 'https://www.wikidex.net/wiki/Tipo'

tipos_page = BeautifulSoup(requests.get(url_tipos, headers=headers).text, 'html.parser')

tabla_tipos = tipos_page.find('table', class_='tabpokemon')

primer_tipo = tipos_page.select_one('table.tabpokemon tbody tr td')

st.write(primer_tipo.prettify())

tabla_tipos = tipos_page.find('table', class_='tabpokemon')

tipos = tabla_tipos.find_all('tr')
tipos = [t.find('a').get_text() for t in tipos if t.find('a')]

tipos = [t.replace('Pelea/Lucha', 'Lucha') for t in tipos]
tipos = [t.replace('Insecto/Bicho', 'Bicho') for t in tipos]

tipos.sort()

st.write(tipos)

tabla_efectivos = tipos_page.find('table', class_='tablaTipos').find_all('td')[3:]

st.write(tabla_efectivos)

mapa_efectividad = {
    None: 1,
    'Poco eficaz': 0.5,
    'Supereficaz': 2,
    'Sin efecto': 0,
}

filas = []
for td in tabla_efectivos:
    clases = td.get('class', [])
    atacante = next(c[1:] for c in clases if c.startswith('r'))
    defensor = next(c[1:] for c in clases if c.startswith('c'))

    span_efecto = td.find('span', title=True)
    titulo = span_efecto['title'] if span_efecto else None

    efectividad = mapa_efectividad.get(titulo, 1)
    filas.append({'atacante': atacante, 'defensor': defensor, 'efectividad': efectividad})

df_largo = pd.DataFrame(filas)

tabla_efectividades = df_largo.pivot(index='atacante', columns='defensor', values='efectividad')

tabla_efectividades.index = tipos
tabla_efectividades.columns = tipos

mostrar_df('Tabla de efectividades por tipo', tabla_efectividades)

cache_pokeapi = cargar_cache(CACHE_PATH)
cache_especies = cargar_cache(CACHE_ESPECIES_PATH)
st.write(f'Registros en caché al iniciar: {len(cache_pokeapi)} de pokemon y {len(cache_especies)} de especies')

ids_pokedex = sorted(lista_pokemon['#'].dropna().astype(int).unique())
st.write(f'IDs a consultar en la API: {len(ids_pokedex)}')

registros_api = []
nuevos_pedidos = 0

for pid in ids_pokedex:
    ya_en_cache = str(pid) in cache_pokeapi
    datos = obtener_pokemon_api(pid, cache_pokeapi)

    if not ya_en_cache:
        nuevos_pedidos += 1
        time.sleep(0.5)
        if nuevos_pedidos % 50 == 0:
            guardar_cache(cache_pokeapi, CACHE_PATH)
            st.write(f'  ... {nuevos_pedidos} pedidos nuevos realizados, caché guardada')

    if datos is None:
        continue

    stats = {s['stat']['name']: s['base_stat'] for s in datos['stats']}
    tipos_api = [t['type']['name'] for t in datos['types']]

    registros_api.append({
        '#': datos['id'],
        'nombre_en': datos['name'],
        'altura_dm': datos['height'],       # decímetros
        'peso_hg': datos['weight'],         # hectogramos
        'experiencia_base': datos['base_experience'],
        'tipo_1': tipos_api[0] if len(tipos_api) > 0 else None,
        'tipo_2': tipos_api[1] if len(tipos_api) > 1 else None,
        'hp': stats.get('hp'),
        'ataque': stats.get('attack'),
        'defensa': stats.get('defense'),
        'ataque_esp': stats.get('special-attack'),
        'defensa_esp': stats.get('special-defense'),
        'velocidad': stats.get('speed'),
    })

guardar_cache(cache_pokeapi, CACHE_PATH)
st.write(f'Pedidos nuevos realizados en esta corrida: {nuevos_pedidos}')
st.write(f'Registros obtenidos: {len(registros_api)}')

pokemon_api = pd.DataFrame(registros_api)

cache_especies = cargar_cache(CACHE_ESPECIES_PATH)

es_legendario = {}
es_mitico = {}
nuevos_pedidos = 0

for pid in ids_pokedex:
    ya_en_cache = str(pid) in cache_especies
    datos = obtener_especie_api(pid, cache_especies)

    if not ya_en_cache:
        nuevos_pedidos += 1
        time.sleep(0.5)
        if nuevos_pedidos % 50 == 0:
            guardar_cache(cache_especies, CACHE_ESPECIES_PATH)
            st.write(f'  ... {nuevos_pedidos} pedidos nuevos, caché guardada')

    if datos is None:
        continue

    es_legendario[pid] = datos['is_legendary']
    es_mitico[pid] = datos['is_mythical']

guardar_cache(cache_especies, CACHE_ESPECIES_PATH)
st.write(f'Pedidos nuevos realizados: {nuevos_pedidos}')

pokemon_api['legendario'] = pokemon_api['#'].map(es_legendario)
pokemon_api['mitico'] = pokemon_api['#'].map(es_mitico)
mostrar_df('Muestra de legendarios y míticos', pokemon_api[['nombre_en', 'legendario', 'mitico']].head())

traduccion_tipos = {
    'normal': 'Normal', 'fire': 'Fuego', 'water': 'Agua', 'electric': 'Eléctrico',
    'grass': 'Planta', 'ice': 'Hielo', 'fighting': 'Lucha', 'poison': 'Veneno',
    'ground': 'Tierra', 'flying': 'Volador', 'psychic': 'Psíquico', 'bug': 'Bicho',
    'rock': 'Roca', 'ghost': 'Fantasma', 'dragon': 'Dragón', 'dark': 'Siniestro',
    'steel': 'Acero', 'fairy': 'Hada'
}

pokemon_api['tipo_1'] = pokemon_api['tipo_1'].map(traduccion_tipos)
pokemon_api['tipo_2'] = pokemon_api['tipo_2'].map(traduccion_tipos, na_action='ignore')

pokemon_api['altura_cm'] = pokemon_api['altura_dm'] * 10
pokemon_api['peso_kg'] = pokemon_api['peso_hg'] / 10 

pokemon_api['suma_estadisticas'] = pokemon_api[['hp', 'ataque', 'defensa', 'ataque_esp', 'defensa_esp', 'velocidad']].sum(axis=1)

pokemon_api = pokemon_api[['#', 'nombre_en', 'altura_cm', 'peso_kg', 'experiencia_base', 'tipo_1', 'tipo_2', 'hp', 'ataque', 'defensa', 'ataque_esp', 'defensa_esp', 'velocidad','legendario', 'mitico', 'suma_estadisticas']]

debilidades_por_tipo = {}
resistencias_por_tipo = {}

for tipo_defensor in tabla_efectividades.columns:
    columna_tipo = tabla_efectividades[tipo_defensor]
    tipos_que_pegan_fuerte = []
    tipos_que_pegan_debil = []

    for tipo_atacante, multiplicador in columna_tipo.items():
        if multiplicador > 1:
            tipos_que_pegan_fuerte.append(tipo_atacante)
        elif multiplicador < 1:
            tipos_que_pegan_debil.append(tipo_atacante)

    debilidades_por_tipo[tipo_defensor] = ', '.join(tipos_que_pegan_fuerte)
    resistencias_por_tipo[tipo_defensor] = ', '.join(tipos_que_pegan_debil)

lista_debil_contra = []
lista_resiste_a = []

for _, fila_pokemon in pokemon_api.iterrows():
    tipo_principal = fila_pokemon['tipo_1']
    lista_debil_contra.append(debilidades_por_tipo.get(tipo_principal, ''))
    lista_resiste_a.append(resistencias_por_tipo.get(tipo_principal, ''))

pokemon_api['debil_contra'] = lista_debil_contra
pokemon_api['resiste_a'] = lista_resiste_a

st.write(f'Shape del dataset integrado: {pokemon_api.shape}')
mostrar_df('Muestra del dataset integrado', pokemon_api[['nombre_en', 'tipo_1', 'debil_contra', 'resiste_a']].head())

# Agregar el nombre en español de cada Pokémon usando la lista obtenida por scraping
# convertir a minusculas para que coincida con la columna 'nombre_en' de la API

nombre_por_id = dict(zip(lista_pokemon['#'], lista_pokemon['Nombre']))
pokemon_api['nombre_es'] = pokemon_api['#'].map(nombre_por_id)

pokemon_api['nombre_es'] = pokemon_api['nombre_es'].str.lower()

# contar cuantas debilidades y resistencias tiene cada Pokémon

n_debilidades = []
n_resistencias = []

for _, fila in pokemon_api.iterrows():
    texto_debil = fila['debil_contra']
    if pd.isnull(texto_debil) or texto_debil.strip() == '':
        n_debilidades.append(0)
    else:
        partes_debil = texto_debil.split(', ')
        n_debilidades.append(len(partes_debil))

    texto_resiste = fila['resiste_a']
    if pd.isnull(texto_resiste) or texto_resiste.strip() == '':
        n_resistencias.append(0)
    else:
        partes_resiste = texto_resiste.split(', ')
        n_resistencias.append(len(partes_resiste))

pokemon_api['n_debilidades'] = n_debilidades
pokemon_api['n_resistencias'] = n_resistencias

mostrar_df('Primeras filas de pokemon_api', pokemon_api.head())

st.write(f'Shape del dataset de la API: {pokemon_api.shape}\n')
mostrar_info_df(pokemon_api, 'Info de pokemon_api (final)')

mostrar_df('Estadísticas descriptivas', pokemon_api.describe())

pokemon_api.to_csv('pokemon_integrado.csv', index=False, encoding='utf-8-sig')

pokemon_api['pseudo_legendario'] = (
    (~pokemon_api['legendario']) &
    (~pokemon_api['mitico']) &
    (pokemon_api['suma_estadisticas'] == 600)
)

st.write(pokemon_api['pseudo_legendario'].value_counts())
mostrar_df(
    'Pokémon pseudo-legendarios detectados',
    pokemon_api.loc[pokemon_api['pseudo_legendario'], ['nombre_es', 'tipo_1', 'tipo_2', 'suma_estadisticas']]
)

features_num = ['hp', 'ataque', 'defensa', 'ataque_esp', 'defensa_esp', 'velocidad']

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

tipo_encoded = encoder.fit_transform(pokemon_api[['tipo_1']])

col_names = encoder.get_feature_names_out()

tipo_df = pd.DataFrame(tipo_encoded, columns=col_names, index=pokemon_api.index)

X = pd.concat([pokemon_api[features_num], tipo_df], axis=1)
y = pokemon_api['pseudo_legendario']
 
st.write(X.shape)
mostrar_df('Muestra de features para clasificación', X.head())

# Usar DecisionTreeClassifier para predecir si un Pokémon es pseudo-legendario o no, usando validación cruzada, 
# con max_depth=3 y class_weight='balanced' para manejar el desbalance de clases.

modelo = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=123)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

scores_f1 = cross_val_score(modelo, X, y, cv=cv, scoring='f1')
st.write('F1 por pliegue:', scores_f1)
st.write('F1 promedio:', scores_f1.mean())

precision = cross_val_score(modelo, X, y, cv=cv, scoring='precision').mean()
recall = cross_val_score(modelo, X, y, cv=cv, scoring='recall').mean()

st.write(f'Precisión promedio: {precision:.3f}')
st.write(f'Recall promedio: {recall:.3f}')

loo_scores = cross_val_score(modelo, X, y, cv=LeaveOneOut(), scoring='f1')
st.write(f'F1 con Leave-One-Out: {loo_scores.mean():.3f}')

y_pred_loo = cross_val_predict(modelo, X, y, cv=LeaveOneOut())

st.write(f'F1 (LOO agregado): {f1_score(y, y_pred_loo):.3f}')
st.write(f'Precisión (LOO agregado): {precision_score(y, y_pred_loo):.3f}')
st.write(f'Recall (LOO agregado): {recall_score(y, y_pred_loo):.3f}')
st.write(confusion_matrix(y, y_pred_loo))

pred_todos = modelo.fit(X, y).predict(X)
falsos_positivos = pokemon_api[(pred_todos == True) & (y == False)]
mostrar_df(
    'Falsos positivos del modelo',
    falsos_positivos[['nombre_es', 'tipo_1', 'suma_estadisticas']].sort_values('suma_estadisticas', ascending=False)
)

modelo_final = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=123)
modelo_final.fit(X, y)

st.write(export_text(modelo_final, feature_names=list(X.columns)))

X_cluster = pokemon_api[features_num]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

inercias = []
siluetas = []
rango_k = range(2, 11)

for k in rango_k:
    km = KMeans(n_clusters=k, random_state=123, n_init=10)
    labels = km.fit_predict(X_scaled)
    inercias.append(km.inertia_)
    siluetas.append(silhouette_score(X_scaled, labels))

for k, inercia, silueta in zip(rango_k, inercias, siluetas):
    st.write(f'K={k}: inercia={inercia:.1f}, silueta={silueta:.3f}')
    
fig, ax1 = plt.subplots(figsize=(8, 5))

ax1.plot(rango_k, inercias, marker='o', color='tab:blue', label='Inercia')
ax1.set_xlabel('K (número de clusters)')
ax1.set_ylabel('Inercia', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.plot(rango_k, siluetas, marker='s', color='tab:orange', label='Silueta')
ax2.set_ylabel('Coeficiente de silueta', color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')

ax1.axvline(x=5, color='gray', linestyle='--', alpha=0.5)
plt.title('Selección de K: inercia vs. silueta')
fig.tight_layout()
st.pyplot(fig)

k_final = 5
modelo_kmeans = KMeans(n_clusters=k_final, random_state=123, n_init=10)
pokemon_api['cluster'] = modelo_kmeans.fit_predict(X_scaled)

perfil_clusters = pokemon_api.groupby('cluster')[features_num].mean().round(1)
perfil_clusters['n_pokemon'] = pokemon_api['cluster'].value_counts().sort_index()
mostrar_df('Perfil promedio por cluster', perfil_clusters)

distancias = modelo_kmeans.transform(X_scaled)
pokemon_api['dist_centroide'] = distancias.min(axis=1)

for c in range(k_final):
    st.write(f'\n--- Cluster {c} ---')
    st.write(pokemon_api[pokemon_api['cluster'] == c]
          .nsmallest(3, 'dist_centroide')[['nombre_es', 'hp', 'ataque', 'defensa', 'ataque_esp', 'defensa_esp', 'velocidad']])
    
nombres_cluster = {
    0: 'Forma base / sin evolucionar',
    1: 'Atacante especial',
    2: 'Atacante físico robusto',
    3: 'Barredor veloz',
    4: 'Muro defensivo',
}
pokemon_api['arquetipo'] = pokemon_api['cluster'].map(nombres_cluster)
st.write(pokemon_api['arquetipo'].value_counts())

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Ataque vs Velocidad
for c in range(k_final):
    mask = pokemon_api['cluster'] == c
    axes[0].scatter(pokemon_api.loc[mask, 'ataque'], pokemon_api.loc[mask, 'velocidad'],
                     label=nombres_cluster[c], alpha=0.6, s=20)
axes[0].set_xlabel('Ataque')
axes[0].set_ylabel('Velocidad')
axes[0].set_title('Ataque vs. Velocidad')
axes[0].legend(fontsize=8)

# Vida vs Defensa
for c in range(k_final):
    mask = pokemon_api['cluster'] == c
    axes[1].scatter(pokemon_api.loc[mask, 'hp'], pokemon_api.loc[mask, 'defensa'],
                     label=nombres_cluster[c], alpha=0.6, s=20)
axes[1].set_xlabel('HP')
axes[1].set_ylabel('Defensa')
axes[1].set_title('HP vs. Defensa')
axes[1].legend(fontsize=8)

plt.tight_layout()
st.pyplot(fig)

perfil_clusters[features_num].plot(kind='bar', figsize=(10, 6))
plt.title('Perfil promedio de stats por arquetipo')
plt.xlabel('Cluster')
plt.ylabel('Valor promedio')
plt.xticks(ticks=range(k_final), labels=[nombres_cluster[c] for c in range(k_final)], rotation=20, ha='right')
plt.legend(title='Stat', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
fig = plt.gcf()
st.pyplot(fig)

pokemon_api.to_csv('pokemon_api_integrado_con_clusters.csv', index=False, encoding='utf-8-sig')

mostrar_df(
    'Pseudo-legendarios por arquetipo',
    pokemon_api[pokemon_api['pseudo_legendario']][['nombre_es', 'arquetipo']]
)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

pokemon_api.boxplot(column='altura_cm', by='arquetipo', ax=axes[0], rot=20)
axes[0].set_title('Altura por arquetipo')
axes[0].set_xlabel('')
axes[0].set_ylabel('Altura (cm)')

pokemon_api.boxplot(column='peso_kg', by='arquetipo', ax=axes[1], rot=20)
axes[1].set_title('Peso por arquetipo')
axes[1].set_xlabel('')
axes[1].set_ylabel('Peso (kg)')

plt.suptitle('')
plt.tight_layout()
st.pyplot(fig)

# Estadísticas descriptivas de tamaño físico por arquetipo
resumen_fisico_por_arquetipo = pokemon_api.groupby('arquetipo')[['altura_cm', 'peso_kg']].agg(['mean', 'std']).round(1)
mostrar_df('Resumen físico por arquetipo', resumen_fisico_por_arquetipo)

# Correlación entre tamaño físico y stats de combate
correlaciones = pokemon_api[['altura_cm', 'peso_kg'] + features_num].corr()
mostrar_df(
    'Correlación entre tamaño físico y stats de combate',
    correlaciones[['altura_cm', 'peso_kg']].drop(['altura_cm', 'peso_kg'])
)

features_fisicas = ['altura_cm', 'peso_kg']

X_fisico = pokemon_api[features_fisicas]
scaler_fisico = StandardScaler()
X_fisico_scaled = scaler_fisico.fit_transform(X_fisico)

inercias_fisico = []
siluetas_fisico = []
rango_k_fisico = range(2, 11)

for k in rango_k_fisico:
    km = KMeans(n_clusters=k, random_state=123, n_init=10)
    labels = km.fit_predict(X_fisico_scaled)
    inercias_fisico.append(km.inertia_)
    siluetas_fisico.append(silhouette_score(X_fisico_scaled, labels))

for k, inercia, silueta in zip(rango_k_fisico, inercias_fisico, siluetas_fisico):
    st.write(f'K={k}: inercia={inercia:.1f}, silueta={silueta:.3f}')
    
fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(rango_k_fisico, inercias_fisico, marker='o', color='tab:blue', label='Inercia')
ax1.set_xlabel('K (número de clusters)')
ax1.set_ylabel('Inercia', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.plot(rango_k_fisico, siluetas_fisico, marker='s', color='tab:orange', label='Silueta')
ax2.set_ylabel('Coeficiente de silueta', color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')

plt.title('Selección de K: clustering físico (altura/peso)')
fig.tight_layout()
st.pyplot(fig)

k_fisico_final = 4  # ajustar según el gráfico de inercia/silueta
modelo_kmeans_fisico = KMeans(n_clusters=k_fisico_final, random_state=123, n_init=10)
pokemon_api['cluster_fisico'] = modelo_kmeans_fisico.fit_predict(X_fisico_scaled)

perfil_fisico = pokemon_api.groupby('cluster_fisico')[features_fisicas].mean().round(1)
perfil_fisico['n_pokemon'] = pokemon_api['cluster_fisico'].value_counts().sort_index()
mostrar_df('Perfil físico por cluster', perfil_fisico)

distancias_fisico = modelo_kmeans_fisico.transform(X_fisico_scaled)
pokemon_api['dist_centroide_fisico'] = distancias_fisico.min(axis=1)

for c in range(k_fisico_final):
    st.write(f'\n--- Cluster {c} ---')
    st.write(pokemon_api[pokemon_api['cluster_fisico'] == c]
          .nsmallest(3, 'dist_centroide_fisico')[['nombre_es', 'altura_cm', 'peso_kg']])
    
from sklearn.metrics import adjusted_rand_score

tabla_comparacion = pd.crosstab(pokemon_api['arquetipo'], pokemon_api['cluster_fisico'])
st.write(tabla_comparacion)

ari = adjusted_rand_score(pokemon_api['cluster'], pokemon_api['cluster_fisico'])
st.write(f'\nAdjusted Rand Index (combate vs físico): {ari:.3f}')

fig, ax = plt.subplots(figsize=(8, 6))
for arquetipo in pokemon_api['arquetipo'].unique():
    mask = pokemon_api['arquetipo'] == arquetipo
    ax.scatter(pokemon_api.loc[mask, 'altura_cm'], pokemon_api.loc[mask, 'peso_kg'],
               label=arquetipo, alpha=0.6, s=20)
ax.set_xlabel('Altura (cm)')
ax.set_ylabel('Peso (kg)')
ax.set_title('Altura vs. Peso coloreado por arquetipo de combate')
ax.legend(fontsize=8)
plt.tight_layout()
st.pyplot(fig)

