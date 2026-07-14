import streamlit as st
import pandas as pd
import kagglehub
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

try:
    import umap
    UMAP_DISPONIBLE = True
except ImportError:
    UMAP_DISPONIBLE = False

# Configuración de página con estilos CSS para eliminar paddings, scrollbar y aplicar colores personalizados
st.set_page_config(layout="wide", page_title="UFC Analytics", page_icon="🥊")

# Inicializar estados de selección
if "peleadores_seleccionados" not in st.session_state:
    st.session_state.peleadores_seleccionados = []
if "limpiar_contador" not in st.session_state:
    st.session_state.limpiar_contador = 0

# CSS para Azul Profundo (#0B132B) y Rojo UFC (#E63946)
st.markdown("""
    <style>
        /* Fondo de la aplicación */
        .stApp {
            background-color: #0B132B;
            color: #FFFFFF;
        }
        .block-container {padding-top: 1rem; padding-bottom: 0rem; padding-left: 2rem; padding-right: 2rem;}
        h1, h2, h3, h4, h5 {margin-top: 0.2rem !important; margin-bottom: 0.2rem !important; color: #FFFFFF !important;}
        hr {margin-top: 0.4rem !important; margin-bottom: 0.4rem !important; border-color: rgba(230, 57, 70, 0.3) !important;}
        
        /* Personalización de Tabs (Pestañas) */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background-color: rgba(255, 255, 255, 0.03);
            border-radius: 4px;
            padding: 2px;
        }
        .stTabs [data-baseweb="tab"] {
            padding-top: 4px;
            padding-bottom: 4px;
            color: #8D99AE !important;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            color: #E63946 !important;
            border-bottom-color: #E63946 !important;
            font-weight: bold;
        }
        
        /* Botón Primario en Rojo */
        .stButton>button[kind="primary"] {
            background-color: #E63946 !important;
            color: white !important;
            border: none !important;
            border-radius: 4px;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        .stButton>button[kind="primary"]:hover {
            background-color: #B51A2B !important;
            box-shadow: 0 0 10px rgba(230, 57, 70, 0.5);
        }
        
        /* Sidebar estilizado */
        section[data-testid="stSidebar"] {
            background-color: #1C2541 !important;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }
    </style>
""", unsafe_allow_html=True)

st.title("Dashboard :red[_UFC Analytics_]")

# 1. Carga de datos
@st.cache_data
def cargar_y_procesar_todo():
    path = kagglehub.dataset_download("mdabbert/ultimate-ufc-dataset")
    df_completo = pd.read_csv(f"{path}/ufc-master.csv")

    df_completo = df_completo.rename(columns={
        'R_draw': 'R_draws', 'B_draw': 'B_draws',
        'R_current_lose_streak': 'R_current_loss_streak', 'B_current_lose_streak': 'B_current_loss_streak',
        'R_win_by_KO/TKO': 'R_win_by_KO_TKO', 'B_win_by_KO/TKO': 'B_win_by_KO_TKO',
        'R_win_by_Submission': 'R_win_by_Submission', 'B_win_by_Submission': 'B_win_by_Submission'
    })

    for prefix in ['R_', 'B_']:
        dec_cols = [c for c in df_completo.columns if c.startswith(f"{prefix}win_by_Decision")]
        df_completo[f"{prefix}win_by_Decision_Total"] = df_completo[dec_cols].sum(axis=1) if dec_cols else 0

    cols_interes = [
        'fighter', 'avg_SIG_STR_landed', 'avg_SIG_STR_pct',
        'avg_SUB_ATT', 'avg_TD_landed', 'avg_TD_pct',
        'Height_cms', 'Reach_cms', 'Weight_lbs', 'age', 'Stance',
        'wins', 'losses', 'draws', 'current_win_streak', 'current_loss_streak',
        'win_by_KO_TKO', 'win_by_Submission', 'win_by_Decision_Total'
    ]

    df_red = df_completo[['R_' + col for col in cols_interes]].copy()
    df_red.columns = [col.replace('_Total', '') for col in cols_interes]
    df_blue = df_completo[['B_' + col for col in cols_interes]].copy()
    df_blue.columns = [col.replace('_Total', '') for col in cols_interes]

    df_red_blue = pd.concat([df_red, df_blue], axis=0, ignore_index=True).sort_values(by='age')

    df_acciones = df_red_blue.groupby('fighter').agg({
        'avg_SIG_STR_landed': 'mean', 'avg_SIG_STR_pct': 'mean',
        'avg_SUB_ATT': 'mean', 'avg_TD_landed': 'mean', 'avg_TD_pct': 'mean'
    }).reset_index()

    df_perfil = df_red_blue.groupby('fighter').agg({
        'Height_cms': 'last', 'Reach_cms': 'last', 'Weight_lbs': 'last', 'age': 'max', 'Stance': 'last',
        'wins': 'last', 'losses': 'last', 'draws': 'last', 'current_win_streak': 'last', 'current_loss_streak': 'last',
        'win_by_KO_TKO': 'sum', 'win_by_Submission': 'sum', 'win_by_Decision': 'sum'
    }).reset_index()

    r_filtros = df_completo[['R_fighter', 'weight_class', 'gender']].rename(columns={'R_fighter': 'fighter'})
    b_filtros = df_completo[['B_fighter', 'weight_class', 'gender']].rename(columns={'B_fighter': 'fighter'})
    fighters_info = pd.concat([r_filtros, b_filtros]).drop_duplicates(subset='fighter').dropna()
    fighters_info['gender'] = fighters_info['gender'].str.title()

    return df_completo, fighters_info, df_acciones, df_perfil

df_ufc, fighters_info, df_peleador_acciones, df_peleador_perfil = cargar_y_procesar_todo()

df_peleadores_consolidado = pd.merge(df_peleador_acciones, df_peleador_perfil, on='fighter', how='inner')
df_peleadores_consolidado = pd.merge(df_peleadores_consolidado, fighters_info, on='fighter', how='inner')

columnas_modelo = [
    'avg_SIG_STR_landed', 'avg_SIG_STR_pct', 'avg_SUB_ATT',
    'avg_TD_landed', 'avg_TD_pct', 'Height_cms', 'Reach_cms', 'age',
    'win_by_KO_TKO', 'win_by_Submission'
]

labels_metricas = {
    'avg_SIG_STR_landed': 'Golpes Conectados', 'avg_SIG_STR_pct': 'Precisión Golpeo',
    'avg_SUB_ATT': 'Intentos Sumisión', 'avg_TD_landed': 'Derribos Conectados',
    'avg_TD_pct': 'Precisión Derribo', 'Height_cms': 'Altura',
    'Reach_cms': 'Alcance', 'age': 'Edad',
    'win_by_KO_TKO': 'Victorias KO/TKO', 'win_by_Submission': 'Victorias Sumisión'
}

df_clean = df_peleadores_consolidado.dropna(subset=columnas_modelo).copy()

@st.cache_data(show_spinner=False)
def calcular_proyecciones(df_para_modelo: pd.DataFrame, columnas_modelo: list, umap_disponible: bool):
    scaler = StandardScaler()
    datos_escalados = scaler.fit_transform(df_para_modelo[columnas_modelo])

    pca = PCA(n_components=2)
    comp_pca = pca.fit_transform(datos_escalados)

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(df_para_modelo) - 1))
    comp_tsne = tsne.fit_transform(datos_escalados)

    resultado = {'escalados': datos_escalados, 'pca': comp_pca, 'tsne': comp_tsne, 'umap': None}
    if umap_disponible:
        reducer = umap.UMAP(n_components=2, random_state=42)
        resultado['umap'] = reducer.fit_transform(datos_escalados)
    return resultado

@st.cache_data(show_spinner=False)
def calcular_clusters(coordenadas: np.ndarray, k: int):
    kmeans_modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
    return kmeans_modelo.fit_predict(coordenadas)

if len(df_clean) > 10:
    proyecciones = calcular_proyecciones(df_clean, columnas_modelo, UMAP_DISPONIBLE)
    datos_escalados = proyecciones['escalados']
    df_clean['PCA_1'], df_clean['PCA_2'] = proyecciones['pca'][:, 0], proyecciones['pca'][:, 1]
    df_clean['TSNE_1'], df_clean['TSNE_2'] = proyecciones['tsne'][:, 0], proyecciones['tsne'][:, 1]

    if UMAP_DISPONIBLE and proyecciones['umap'] is not None:
        df_clean['UMAP_1'], df_clean['UMAP_2'] = proyecciones['umap'][:, 0], proyecciones['umap'][:, 1]

    with st.sidebar:
        st.header("🎛️ Configuración")
        espacio_proyeccion = st.selectbox("K-Means sobre:", options=["PCA", "t-SNE", "UMAP"] if UMAP_DISPONIBLE else ["PCA", "t-SNE"])
        k_clusters = st.slider("Arquetipos (K):", min_value=2, max_value=6, value=3)
        tipo_escala = st.selectbox("Escala:", options=["Datos Originales", "Datos Normalizados"])
        peleador_buscado = st.selectbox("Localizar peleador:", options=["Selecciona un Peleador"] + sorted(df_clean['fighter'].unique().tolist()))
        division_sel = st.selectbox("Filtrar por División:", options=["Todas"] + sorted(df_clean['weight_class'].unique().tolist()))

    if espacio_proyeccion == "PCA":
        coordenadas_clustering = df_clean[['PCA_1', 'PCA_2']].values
    elif espacio_proyeccion == "t-SNE":
        coordenadas_clustering = df_clean[['TSNE_1', 'TSNE_2']].values
    elif espacio_proyeccion == "UMAP":
        coordenadas_clustering = df_clean[['UMAP_1', 'UMAP_2']].values

    labels_cluster = calcular_clusters(coordenadas_clustering, k_clusters)
    df_clean['cluster_arquetipo'] = [f"Arquetipo {x + 1}" for x in labels_cluster]

    df_norm = pd.DataFrame(datos_escalados, columns=columnas_modelo)
    for col in ['fighter', 'weight_class', 'cluster_arquetipo', 'wins', 'losses', 'PCA_1', 'PCA_2', 'TSNE_1', 'TSNE_2', 'win_by_KO_TKO', 'win_by_Submission', 'win_by_Decision']:
        df_norm[col] = df_clean[col].values

    # Filtrado Dinámico Global
    df_clean_filtrado = df_clean.copy()
    df_norm_filtrado = df_norm.copy()
    if division_sel != "Todas":
        df_clean_filtrado = df_clean_filtrado[df_clean_filtrado['weight_class'] == division_sel]
        df_norm_filtrado = df_norm_filtrado[df_norm_filtrado['weight_class'] == division_sel]

    df_base_grafica = df_norm_filtrado.copy() if "Normalizados" in tipo_escala else df_clean_filtrado.copy()

    # Mantener tus colores de arquetipos sin tocar
    paleta_dinamica = px.colors.qualitative.D3
    arquetipos_unicos = sorted(df_clean['cluster_arquetipo'].unique().tolist())
    colores_arquetipo = {arq: paleta_dinamica[i % len(paleta_dinamica)] for i, arq in enumerate(arquetipos_unicos)}

    def generar_mapa_interactivo(dim_x, dim_y, titulo_x, titulo_y, key_identificador):
        # El identificador dinámico de clave evita que la selección antigua persista tras hacer clic en "Limpiar"
        key_dinamica = f"{key_identificador}_{st.session_state.limpiar_contador}"
        fig = go.Figure()
        fig.add_trace(go.Histogram2dContour(x=df_clean_filtrado[dim_x], y=df_clean_filtrado[dim_y], colorscale='Greys', opacity=0.1, showscale=False, hoverinfo='skip'))
        for arquetipo in sorted(df_clean_filtrado['cluster_arquetipo'].unique()):
            df_g = df_clean_filtrado[df_clean_filtrado['cluster_arquetipo'] == arquetipo]
            if not df_g.empty:
                fig.add_trace(go.Scatter(
                    x=df_g[dim_x], y=df_g[dim_y], mode='markers', name=arquetipo, text=df_g['fighter'],
                    customdata=np.stack((df_g['wins'], df_g['losses']), axis=-1),
                    marker=dict(size=5.5, color=colores_arquetipo[arquetipo], opacity=0.65, line=dict(width=0.3, color='white')),
                    hovertemplate="<b>%{text}</b><br>%{customdata[0]}W-%{customdata[1]}L<extra></extra>"
                ))
        if peleador_buscado != "Selecciona un Peleador":
            p_df = df_clean_filtrado[df_clean_filtrado['fighter'] == peleador_buscado]
            if not p_df.empty:
                # Detalle en rojo vibrante (#E63946) para el peleador localizado
                fig.add_trace(go.Scatter(x=p_df[dim_x], y=p_df[dim_y], mode='markers', marker=dict(size=11, color='#E63946', symbol='star'), name=peleador_buscado))

        fig.update_layout(
            xaxis=dict(title=dict(text=titulo_x, font=dict(size=9, color="white")), tickfont=dict(size=8, color="white")), 
            yaxis=dict(title=dict(text=titulo_y, font=dict(size=9, color="white")), tickfont=dict(size=8, color="white")),
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)', 
            height=220, 
            margin=dict(t=5, b=25, l=35, r=5), 
            showlegend=False
        )
        return st.plotly_chart(fig, use_container_width=True, key=key_dinamica, on_select="rerun")
    
    # --- RENDERIZADO LAYOUT COMPACTO ---
    col_izq, col_der = st.columns([1.1, 1.0])

    with col_izq:
        tab_pca, tab_tsne, tab_umap = st.tabs(["PCA", "t-SNE", "UMAP"])
        
        with tab_pca:
            ev_pca = generar_mapa_interactivo('PCA_1', 'PCA_2', "PC1", "PC2", "mapa_pca")
            if ev_pca and "selection" in ev_pca and ev_pca["selection"]["points"]:
                st.session_state.peleadores_seleccionados = [p["text"] for p in ev_pca["selection"]["points"] if "text" in p]
        with tab_tsne:
            ev_tsne = generar_mapa_interactivo('TSNE_1', 'TSNE_2', "t-SNE 1", "t-SNE 2", "mapa_tsne")
            if ev_tsne and "selection" in ev_tsne and ev_tsne["selection"]["points"]:
                st.session_state.peleadores_seleccionados = [p["text"] for p in ev_tsne["selection"]["points"] if "text" in p]
        with tab_umap:
            if UMAP_DISPONIBLE:
                ev_umap = generar_mapa_interactivo('UMAP_1', 'UMAP_2', "UMAP 1", "UMAP 2", "mapa_umap")
                if ev_umap and "selection" in ev_umap and ev_umap["selection"]["points"]:
                    st.session_state.peleadores_seleccionados = [p["text"] for p in ev_umap["selection"]["points"] if "text" in p]
        
        # Coordenadas paralelas optimizadas (Mapeo de colores estricto e intacto)
        metricas_parcoords = ['avg_SIG_STR_landed', 'avg_SIG_STR_pct', 'avg_SUB_ATT', 'avg_TD_landed', 'avg_TD_pct', 'Height_cms', 'Reach_cms', 'age']
        if not df_norm_filtrado.empty:
            df_perfil_arquetipos_z = df_norm_filtrado.groupby('cluster_arquetipo')[metricas_parcoords].mean().reset_index().sort_values('cluster_arquetipo')
            
            # Crear índice numérico para el mapeo discreto de la escala de color
            mapa_indice_arquetipo = {arq: i for i, arq in enumerate(sorted(df_perfil_arquetipos_z['cluster_arquetipo'].unique()))}
            df_perfil_arquetipos_z['_indice_color'] = df_perfil_arquetipos_z['cluster_arquetipo'].map(mapa_indice_arquetipo)
            
            n_arquetipos_actual = len(mapa_indice_arquetipo)
            colores_parcoords = [colores_arquetipo[arq] for arq in sorted(mapa_indice_arquetipo, key=mapa_indice_arquetipo.get)]
            
            # Escala de colores discreta bien formada para plotly
            if n_arquetipos_actual > 1:
                colorscale_parcoords = [[i / (n_arquetipos_actual - 1), c] for i, c in enumerate(colores_parcoords)]
            else:
                colorscale_parcoords = [[0, colores_parcoords[0]], [1, colores_parcoords[0]]]

            fig_parcoords = go.Figure(data=go.Parcoords(
                line=dict(
                    color=df_perfil_arquetipos_z['_indice_color'], 
                    colorscale=colorscale_parcoords, 
                    showscale=False
                ),
                dimensions=[dict(label=labels_metricas.get(col, col), values=df_perfil_arquetipos_z[col]) for col in metricas_parcoords],
                labelfont=dict(size=9, color='white'), 
                tickfont=dict(size=8, color='rgba(255,255,255,0.7)')
            ))
            # Ajuste de margen superior (t=45) para que los nombres de las variables se vean perfectos
            fig_parcoords.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=190, margin=dict(t=45, b=15, l=35, r=35))
            st.plotly_chart(fig_parcoords, use_container_width=True, key="parcoords_arquetipos")

    with col_der:
        st.space(50)
        col_cuadros, col_radar = st.columns(2)
        with col_cuadros:
            if not df_clean_filtrado.empty:
                df_perfil_arquetipos_orig = df_clean_filtrado.groupby('cluster_arquetipo')[columnas_modelo].mean().reset_index()
                df_perfil_arquetipos_z = df_norm_filtrado.groupby('cluster_arquetipo')[metricas_parcoords].mean().reset_index()
                
                for arquetipo in sorted(df_clean_filtrado['cluster_arquetipo'].unique()):
                    row_orig = df_perfil_arquetipos_orig[df_perfil_arquetipos_orig['cluster_arquetipo'] == arquetipo]
                    row_z = df_perfil_arquetipos_z[df_perfil_arquetipos_z['cluster_arquetipo'] == arquetipo]
                    if not row_z.empty:
                        color_hex = colores_arquetipo[arquetipo]
                        rgb = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                        id_metrica_top = row_z[['avg_SIG_STR_landed', 'avg_SIG_STR_pct', 'avg_SUB_ATT', 'avg_TD_landed', 'avg_TD_pct']].iloc[0].idxmax()
                        val_top = row_orig[id_metrica_top].iloc[0] * (100 if 'pct' in id_metrica_top and row_orig[id_metrica_top].iloc[0] <= 1.0 else 1)
                        
                        st.markdown(f"""
                            <div style="background-color: rgba({rgb[0]},{rgb[1]},{rgb[2]},0.1); border-left: 3px solid {color_hex}; padding: 3px 8px; border-radius: 4px; margin-bottom: 4px; line-height: 1.1;">
                                <span style="color:{color_hex}; font-weight:bold; font-size:0.85em;">{arquetipo}</span><br>
                                <span style="font-size:0.75em; opacity: 0.9; color: #FFFFFF;">★ {labels_metricas.get(id_metrica_top)} ({val_top:.1f}{'%' if 'pct' in id_metrica_top else ''})</span>
                            </div>
                        """, unsafe_allow_html=True)

        with col_radar:
            # Radar con dimensiones corregidas para evitar recortes de texto
            fig_radar = go.Figure()
            metricas_radar = ['avg_SIG_STR_landed', 'avg_SIG_STR_pct', 'avg_SUB_ATT', 'avg_TD_landed', 'avg_TD_pct']
            labels_radar = ['Golpes', 'Precisión%', 'Subm.', 'Derribos', 'TD Eff%']

            if st.session_state.peleadores_seleccionados:
                for p in st.session_state.peleadores_seleccionados:
                    df_p_radar = df_base_grafica[df_base_grafica['fighter'] == p]
                    if not df_p_radar.empty:
                        val = df_p_radar[metricas_radar].iloc[0].tolist()
                        if "Originales" in tipo_escala: val = [v*100 if i in [1,4] and v<=1.0 else v for i,v in enumerate(val)]
                        fig_radar.add_trace(go.Scatterpolar(r=val+[val[0]], theta=labels_radar+[labels_radar[0]], fill='toself', name=p))
            else:
                for arquetipo in sorted(df_clean_filtrado['cluster_arquetipo'].unique()):
                    df_c = df_base_grafica[df_base_grafica['cluster_arquetipo'] == arquetipo]
                    if not df_c.empty:
                        val = df_c[metricas_radar].mean().tolist()
                        if "Originales" in tipo_escala: val = [v*100 if i in [1,4] and v<=1.0 else v for i,v in enumerate(val)]
                        fig_radar.add_trace(go.Scatterpolar(r=val+[val[0]], theta=labels_radar+[labels_radar[0]], mode='lines', name=arquetipo, line=dict(color=colores_arquetipo[arquetipo], width=1.5)))

            # Modificado: Se amplía la altura a 210 e incrementamos los márgenes laterales y verticales para evitar textos cortados
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(size=7, color='white')), 
                    angularaxis=dict(tickfont=dict(size=8, color='white'))
                ),
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)', 
                height=210, 
                margin=dict(t=30, b=30, l=40, r=40), 
                showlegend=False
            )
            st.plotly_chart(fig_radar, use_container_width=True, key="radar_multivariable")

        st.space(50)
        if not df_clean_filtrado.empty:
            df_vic = df_clean_filtrado.groupby('cluster_arquetipo')[['win_by_KO_TKO', 'win_by_Submission', 'win_by_Decision']].sum().reset_index()
            df_melt = df_vic.melt(id_vars=['cluster_arquetipo'], value_vars=['win_by_KO_TKO', 'win_by_Submission', 'win_by_Decision'], var_name='Metodo', value_name='Total')
            df_melt['Metodo'] = df_melt['Metodo'].map({'win_by_KO_TKO': 'KO', 'win_by_Submission': 'SUB', 'win_by_Decision': 'DEC'})
            
            fig_equipos = px.bar(df_melt, x='Metodo', y='Total', color='cluster_arquetipo', barmode='group', color_discrete_map=colores_arquetipo, text_auto=True)
            fig_equipos.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=180, margin=dict(t=5, b=5, l=5, r=5),
                                      font=dict(size=9, color='white'), yaxis=dict(gridcolor='rgba(255,255,255,0.08)'), showlegend=False)
            st.plotly_chart(fig_equipos, use_container_width=True, key="metodos_victoria_equipos")

    # --- SECCIÓN INFERIOR: COMPARADOR INDIVIDUAL ---
    st.divider()
    if st.session_state.peleadores_seleccionados:
        col_cards_inf, col_tabs_inf = st.columns([1.0, 2.0])
        
        with col_cards_inf:
            for peleador in st.session_state.peleadores_seleccionados[:3]:
                datos_p = df_clean_filtrado[df_clean_filtrado['fighter'] == peleador]
                if not datos_p.empty:
                    row_p = datos_p.iloc[0]
                    # Obtenemos el arquetipo y su color correspondiente
                    arquetipo_p = row_p['cluster_arquetipo']
                    color_hex = colores_arquetipo.get(arquetipo_p, "#E63946") # Rojo por defecto si falla
                    
                    # Extraemos el RGB para el fondo semitransparente
                    rgb = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                    
                    st.markdown(f"""
                        <div style="
                            background-color: rgba({rgb[0]},{rgb[1]},{rgb[2]},0.08); 
                            border: 1px solid rgba({rgb[0]},{rgb[1]},{rgb[2]},0.3); 
                            border-left: 5px solid {color_hex}; 
                            padding: 6px 10px; 
                            border-radius: 4px; 
                            margin-bottom: 5px; 
                            font-size:0.78em; 
                            line-height:1.3;
                        ">
                            <b style="color: {color_hex}; font-size:1.1em;">{row_p['fighter']}</b> 
                            <span style="color: rgba(255,255,255,0.6); font-size:0.9em;">({arquetipo_p})</span><br>
                            <span style="color: #FFFFFF;">
                                <b>Div:</b> {row_p['weight_class']} | 
                                <b>Alt:</b> {row_p['Height_cms']:.0f}cm | 
                                <b>Alc:</b> {row_p['Reach_cms']:.0f}cm
                            </span>
                        </div>
                    """, unsafe_allow_html=True)
            
            # Botón Primario para limpiar selección
            if st.button("Limpiar Selección", type="primary"):
                st.session_state.peleadores_seleccionados = []
                st.session_state.limpiar_contador += 1
                st.rerun()

        with col_tabs_inf:
            df_analisis = df_base_grafica[df_base_grafica['fighter'].isin(st.session_state.peleadores_seleccionados)].copy()
            if not df_analisis.empty:
                tab_perf, tab_vics = st.tabs(["Atributos Tácticos", "Vías de Victoria"])
                
                with tab_perf:
                    if "Originales" in tipo_escala:
                        if df_analisis['avg_SIG_STR_pct'].max() <= 1.0: df_analisis['avg_SIG_STR_pct'] *= 100
                        if df_analisis['avg_TD_pct'].max() <= 1.0: df_analisis['avg_TD_pct'] *= 100
                    
                    fig_prec = go.Figure()
                    for idx, row in df_analisis.iterrows():
                        fig_prec.add_trace(go.Bar(x=[labels_metricas[c] for c in metricas_parcoords], y=[row[c] for c in metricas_parcoords], name=row['fighter']))
                    fig_prec.update_layout(barmode='group', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=140, margin=dict(t=5, b=5, l=5, r=5), font=dict(size=8, color='white'), showlegend=False)
                    st.plotly_chart(fig_prec, use_container_width=True, key="precision_barras")
                
                with tab_vics:
                    df_v_reales = df_clean_filtrado[df_clean_filtrado['fighter'].isin(st.session_state.peleadores_seleccionados)]
                    df_m_vic = df_v_reales.melt(id_vars=['fighter'], value_vars=['win_by_KO_TKO', 'win_by_Submission', 'win_by_Decision'], var_name='Metodo', value_name='Total')
                    df_m_vic['Metodo'] = df_m_vic['Metodo'].map({'win_by_KO_TKO': 'KO', 'win_by_Submission': 'SUB', 'win_by_Decision': 'DEC'})
                    fig_vic = px.bar(df_m_vic, x='Metodo', y='Total', color='fighter', barmode='group', text_auto=True)
                    fig_vic.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=140, margin=dict(t=5, b=5, l=5, r=5), font=dict(size=8, color='white'), showlegend=False)
                    st.plotly_chart(fig_vic, use_container_width=True, key="victorias_barras")
    else:
        st.caption("Usa la herramienta de selección (Caja o Lazo) de los mapas superiores para activar el comparador individual.")
else:
    st.error("Registros vectoriales insuficientes.")
