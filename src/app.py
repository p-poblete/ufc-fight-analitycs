import streamlit as st
import pandas as pd
import kagglehub
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(layout="wide")

st.title("Dashboard :red[_UFC_]")
st.divider()

def clasificar_arquetipo_base(row):
    if row['avg_SIG_STR_landed'] > 4.0:
        return "Striker"
    elif row['avg_TD_landed'] > 2.0:
        return "Grappler"
    return "Balanceado"

def calcular_retorno(odds, inversion, gano):
    if not gano:
        return -inversion
    if pd.isna(odds):
        return None
    if odds > 0:
        ganancia = inversion * (odds / 100)
    else:
        ganancia = inversion * (100 / abs(odds))
    return round(ganancia, 2)

def get_tipo_apuesta(odds):
    if pd.isna(odds):
        return None
    elif odds < -200:
        return 'Gran favorito'
    elif odds < 0:
        return 'Favorito'
    elif odds < 150:
        return "Pick'em"
    elif odds < 300:
        return 'Underdog'
    else:
        return 'Gran underdog'
    
def get_historial_apuestas(nombre, df_ufc, inversion, col_r, col_b):
    mask   = (df_ufc['R_fighter'] == nombre) | (df_ufc['B_fighter'] == nombre)
    peleas = df_ufc[mask].copy()

    registros = []
    for n_pelea, (_, row) in enumerate(peleas.iterrows(), start=1):
        lado = 'R' if row['R_fighter'] == nombre else 'B'
        winner = row['Winner']
        
        if winner == 'Draw':
            resultado, gano = 'Empate', False
        elif (winner == 'Red' and lado == 'R') or (winner == 'Blue' and lado == 'B'):
            resultado, gano = 'Victoria', True
        else:
            resultado, gano = 'Derrota', False

        odds    = row.get(col_r if lado == 'R' else col_b, None)
        retorno = calcular_retorno(odds, inversion, gano)
        tipo    = get_tipo_apuesta(odds)
        oponente= row['B_fighter'] if lado == 'R' else row['R_fighter']
        
        if pd.notna(row['date']):
            fecha = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])
        else:
            fecha = 'N/A'

        registros.append({
            'peleador' : nombre,
            'n_pelea'  : n_pelea,
            'resultado': resultado,
            'odds'     : odds,
            'tipo'     : tipo,
            'retorno'  : retorno,
            'oponente' : oponente,
            'fecha'    : fecha,
            'finish'   : row.get('finish', 'N/A'),
        })

    return pd.DataFrame(registros)

def get_historial_scatter(nombre, df_ufc):
    """
    Filtra y procesa los combates históricos de un peleador específico para
    estructurar las métricas de rendimiento por round y duración del combate.
    """
    mask   = (df_ufc['R_fighter'] == nombre) | (df_ufc['B_fighter'] == nombre)
    peleas = df_ufc[mask].copy()

    registros = []
    for n_pelea, (_, row) in enumerate(peleas.iterrows(), start=1):
        winner = row['Winner']
        
        if winner == 'Draw':
            resultado = 'Empate'
        elif (winner == 'Red'  and row['R_fighter'] == nombre) or \
             (winner == 'Blue' and row['B_fighter'] == nombre):
            resultado = 'Victoria'
        else:
            resultado = 'Derrota'

        lado = 'R' if row['R_fighter'] == nombre else 'B'
        
        dur_secs = row.get('total_fight_time_secs',    None)
        sig_str  = row.get(f'{lado}_avg_SIG_STR_landed',   None)
        sig_pct  = row.get(f'{lado}_avg_SIG_STR_pct',      None)
        td_land  = row.get(f'{lado}_avg_TD_landed',         None)
        sub_att  = row.get(f'{lado}_avg_SUB_ATT',           None)

        if pd.notna(row['date']):
            fecha = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])
        else:
            fecha = 'N/A'

        registros.append({
            'peleador'   : nombre,
            'n_pelea'    : n_pelea,
            'resultado'  : resultado,
            'dur_min'    : dur_secs / 60 if dur_secs else None,
            'sig_str'    : sig_str,
            'sig_str_pct': sig_pct,
            'td_landed'  : td_land,
            'sub_att'    : sub_att,
            'finish'     : row.get('finish', 'N/A'),
            'oponente'   : row['B_fighter'] if lado == 'R' else row['R_fighter'],
            'fecha'      : fecha,
        })

    return pd.DataFrame(registros)

def get_historial_tiempos(nombre, df_ufc):
    """
    Extrae la duración en minutos de todas las peleas del peleador.
    """
    mask = (df_ufc['R_fighter'] == nombre) | (df_ufc['B_fighter'] == nombre)
    peleas = df_ufc[mask].copy()
    
    registros = []
    for _, row in peleas.iterrows():
        dur_secs = row.get('total_fight_time_secs', None)
        if pd.isna(dur_secs) or dur_secs <= 0:
            continue
            
        registros.append({
            'peleador': nombre,
            'dur_min': dur_secs / 60
        })
    return pd.DataFrame(registros)

# 1. Carga de datos
@st.cache_data
def cargar_y_procesar_todo():
    path = kagglehub.dataset_download("mdabbert/ultimate-ufc-dataset")
    df_completo = pd.read_csv(f"{path}/ufc-master.csv")
    
    df_completo = df_completo.rename(columns={
        'R_draw': 'R_draws',
        'B_draw': 'B_draws',
        'R_current_lose_streak': 'R_current_loss_streak',
        'B_current_lose_streak': 'B_current_loss_streak'
    })

    cols_interes = [
        'fighter', 'avg_SIG_STR_landed', 'avg_SIG_STR_pct', 
        'avg_SUB_ATT', 'avg_TD_landed', 'avg_TD_pct', 
        'Height_cms', 'Reach_cms', 'Weight_lbs', 'age', 'Stance',
        'wins', 'losses', 'draws', 'current_win_streak', 'current_loss_streak'
    ]
    
    df_red = df_completo[['R_' + col for col in cols_interes]].copy()
    df_red.columns = cols_interes
    
    df_blue = df_completo[['B_' + col for col in cols_interes]].copy()
    df_blue.columns = cols_interes
    
    df_red_blue = pd.concat([df_red, df_blue], axis=0, ignore_index=True)
    
    df_red_blue = df_red_blue.sort_values(by='age', ascending=True)
    
    df_acciones = df_red_blue.groupby('fighter').agg({
        'avg_SIG_STR_landed': 'mean',
        'avg_SIG_STR_pct': 'mean',
        'avg_SUB_ATT': 'mean',
        'avg_TD_landed': 'mean',
        'avg_TD_pct': 'mean'
    }).reset_index()
    
    df_acciones['arquetipo_base'] = df_acciones.apply(clasificar_arquetipo_base, axis=1)
    
    df_perfil = df_red_blue.groupby('fighter').agg({
        'Height_cms': 'last',
        'Reach_cms': 'last',
        'Weight_lbs': 'last',
        'age': 'max',
        'Stance': 'last',
        'wins': 'last',
        'losses': 'last',
        'draws': 'last',
        'current_win_streak': 'last',
        'current_loss_streak': 'last'
    }).reset_index()
    
    r_filtros = df_completo[['R_fighter', 'weight_class', 'gender']].rename(columns={'R_fighter': 'fighter'})
    b_filtros = df_completo[['B_fighter', 'weight_class', 'gender']].rename(columns={'B_fighter': 'fighter'})
    fighters_info = pd.concat([r_filtros, b_filtros]).drop_duplicates(subset='fighter').dropna()
    fighters_info['gender'] = fighters_info['gender'].str.title()
    
    return df_completo, fighters_info, df_acciones, df_perfil

df_ufc, fighters_info, df_peleador_acciones, df_peleador_perfil = cargar_y_procesar_todo()

def generar_radar_chart(lista_peleadores, df_acciones):
    """
    Genera un gráfico de radar comparativo para los peleadores seleccionados.
    """
    cols_norm = ['avg_SIG_STR_landed', 'avg_SIG_STR_pct', 'avg_SUB_ATT', 'avg_TD_landed', 'avg_TD_pct']
    
    categorias = [
        'Sig. Strikes<br>lanzados',
        'Precisión<br>strike',
        'Intentos<br>sumisión',
        'Takedowns<br>lanzados',
        'Precisión<br>takedown'
    ]
    
    df_sel = df_acciones[df_acciones['fighter'].isin(lista_peleadores)].copy()
    
    maximos_globales = {col: df_acciones[col].max() for col in cols_norm}
    
    df_norm = df_sel.copy()
    for col in cols_norm:
        max_global = maximos_globales[col]
        df_norm[col] = df_norm[col] / max_global if max_global > 0 else 0
        
    colores = [
        ('#378ADD', 'rgba(55,138,221,0.10)'),
        ('#D85A30', 'rgba(216,90,48,0.10)'),
    ]
    
    fig = go.Figure()
    
    for i, (_, row_norm) in enumerate(df_norm.iterrows()):
        nombre   = row_norm['fighter']
        row_real = df_sel[df_sel['fighter'] == nombre].iloc[0]
        
        vals_norm = [row_norm[c] for c in cols_norm]
        tooltip   = [
            f'{cat.replace("<br>", " ")}: {row_real[col]:.2f}'
            for cat, col in zip(categorias, cols_norm)
        ]
        
        color_line, color_fill = colores[i % len(colores)]
        
        fig.add_trace(go.Scatterpolar(
            r             = vals_norm + [vals_norm[0]],
            theta         = categorias + [categorias[0]],
            name          = nombre,
            text          = tooltip + [tooltip[0]],
            hovertemplate = '%{text}<extra></extra>',
            line          = dict(color=color_line, width=2),
            fill          = 'toself',
            fillcolor     = color_fill,
            marker        = dict(size=5, color=color_line),
        ))
        
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(
                visible  = True,
                range    = [0,1],
                tickvals = [0.25, 0.5, 0.75, 1.0],
                ticktext = ['25%', '50%', '75%', '100%'],
                tickfont = dict(size=9, color='gray'),
                gridcolor= 'rgba(128,128,128,0.15)',
            ),
            angularaxis=dict(
                tickfont = dict(size=11),
                gridcolor= 'rgba(128,128,128,0.15)',
            ),
        ),
        legend=dict(orientation='h', yanchor='bottom', y=-0.15, xanchor='center', x=0.5),
        title=dict(text='Acciones de Combate', x=0, font=dict(size=14, color='gray')),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor ='rgba(0,0,0,0)',
        height=450,
        margin=dict(t=40, b=60, l=60, r=60),
    )
    
    return fig

def generar_donas_finalizaciones(lista_peleadores, df_ufc):
    """
    Genera un subplot horizontal con gráficos de dona sobre cómo finalizan sus peleas ganadas.
    """
    color_map = {
        'KO/TKO': '#D85A30', 'SUB': '#378ADD', 'DEC': '#1D9E75', 
        'S-DEC': '#9B59B6', 'M-DEC': '#F39C12', 'U-DEC': '#1D9E75', 
        'CNC': '#888780', 'DQ': '#888780',
    }

    rows = 1
    cols = 2
    specs = [[{'type': 'pie'}, {'type': 'pie'}]]

    fig = make_subplots(
        rows=rows, cols=cols,
        specs=specs,
        subplot_titles=lista_peleadores,
    )

    for i, nombre in enumerate(lista_peleadores):
        col_idx = i + 1
        
        mask = (df_ufc['R_fighter'] == nombre) | (df_ufc['B_fighter'] == nombre)
        peleas = df_ufc[mask][['finish', 'Winner', 'R_fighter', 'B_fighter']].copy()
        
        peleas_ganadas = peleas[
            ((peleas['Winner'] == 'Red')  & (peleas['R_fighter'] == nombre)) |
            ((peleas['Winner'] == 'Blue') & (peleas['B_fighter'] == nombre))
        ]
        
        finalizaciones = peleas_ganadas['finish'].value_counts()
        
        if finalizaciones.empty:
            continue
        
        labels = finalizaciones.index.tolist()
        values = finalizaciones.values.tolist()
        colors = [color_map.get(l, '#888780') for l in labels]
        
        fig.add_trace(
            go.Pie(
                labels=labels,
                values=values,
                name=nombre,
                marker=dict(colors=colors, line=dict(color='rgba(0,0,0,0.2)', width=1)),
                textinfo='label+percent',
                hovertemplate='<b>%{label}</b><br>Victorias: %{value}<br>%{percent}<extra>' + nombre + '</extra>',
                hole=0.4,
                textfont=dict(size=11),
                insidetextorientation='radial',
            ),
            row=1, col=col_idx
        )

    fig.update_layout(
        title=dict(
            text='Distribución de Finalizaciones',
            x=0, font=dict(size=14, color='gray'),
        ),
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=-0.1, xanchor='center', x=0.5),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=375,
        margin=dict(t=80, b=60, l=40, r=40),
    )
    return fig

def generar_donas_derrotas(lista_peleadores, df_ufc):
    """
    Genera un subplot horizontal con gráficos de dona sobre cómo perdieron sus peleas.
    """
    color_map = {
        'KO/TKO' : '#D85A30', 'SUB'   : '#378ADD', 'DEC'   : '#1D9E75',
        'S-DEC'   : '#9B59B6', 'M-DEC' : '#F39C12', 'U-DEC' : '#1D9E75',
        'CNC'     : '#888780', 'DQ'    : '#888780',
    }

    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'pie'}, {'type': 'pie'}]], subplot_titles=lista_peleadores)

    for i, nombre in enumerate(lista_peleadores):
        col_idx = i + 1
        mask = (df_ufc['R_fighter'] == nombre) | (df_ufc['B_fighter'] == nombre)
        peleas = df_ufc[mask][['finish', 'Winner', 'R_fighter', 'B_fighter']].copy()
        
        peleas_perdidas = peleas[
            ((peleas['Winner'] == 'Blue') & (peleas['R_fighter'] == nombre)) |
            ((peleas['Winner'] == 'Red')  & (peleas['B_fighter'] == nombre))
        ]
        
        derrotas = peleas_perdidas['finish'].value_counts()
        
        if derrotas.empty:
            fig.add_trace(
                go.Pie(
                    labels=['Sin derrotas'],values=[1],name=nombre,
                    marker=dict(colors=['#1D9E75'], line=dict(color='rgba(0,0,0,0.2)', width=1)),
                    textinfo='label', hovertemplate=f'<b>{nombre}</b> no tiene derrotas registradas<extra></extra>',
                    hole=0.4, textfont=dict(size=11),
                ),
                row=1, col=col_idx
            )
        else:
            labels = derrotas.index.tolist()
            values = derrotas.values.tolist()
            colors = [color_map.get(l, '#888780') for l in labels]
            
            fig.add_trace(
                go.Pie(
                    labels=labels, values=values, name=nombre,
                    marker=dict(colors=colors, line=dict(color='rgba(0,0,0,0.2)', width=1)),
                    textinfo='label+percent', hovertemplate='<b>%{label}</b><br>Derrotas: %{value}<br>%{percent}<extra>' + nombre + '</extra>',
                    hole=0.4, textfont=dict(size=11), insidetextorientation='radial',
                ),
                row=1, col=col_idx
            )

    fig.update_layout(
        title=dict(text='Distribución de Derrotas', x=0, font=dict(size=14, color='gray')),
        showlegend=True, legend=dict(orientation='h', yanchor='bottom', y=-0.1, xanchor='center', x=0.5),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=375, margin=dict(t=80, b=60, l=40, r=40),
    )
    return fig

def generar_chart_roi(lista_peleadores, df_ufc, tipo_odds, inversion=100):
    """
    Genera un gráfico Scatter que simula el retorno de inversión según el mercado seleccionado.
    """
    odds_config = {
        'odds'    : {'col_r': 'R_odds',     'col_b': 'B_odds',     'label': 'Odds generales'},
        'dec_odds': {'col_r': 'r_dec_odds', 'col_b': 'b_dec_odds', 'label': 'Odds decisión'},
        'sub_odds': {'col_r': 'r_sub_odds', 'col_b': 'b_sub_odds', 'label': 'Odds sumisión'},
        'ko_odds' : {'col_r': 'r_ko_odds',  'col_b': 'b_ko_odds',  'label': 'Odds KO/TKO'},
    }

    col_r = odds_config[tipo_odds]['col_r']
    col_b = odds_config[tipo_odds]['col_b']
    label = odds_config[tipo_odds]['label']

    colores_peleador = {
        lista_peleadores[0]: '#378ADD',
        lista_peleadores[1]: '#D85A30',
    }

    symbol_map = {'Victoria': 'circle', 'Derrota': 'x', 'Empate': 'diamond'}
    size_map   = {'Victoria': 10, 'Derrota': 13, 'Empate': 10}
    orden_tipo = ['Gran favorito', 'Favorito', "Pick'em", 'Underdog', 'Gran underdog']

    dfs = []
    for p in lista_peleadores:
        df_h = get_historial_apuestas(p, df_ufc, inversion, col_r, col_b)
        if not df_h.empty:
            dfs.append(df_h)
            
    if not dfs:
        return go.Figure()

    df_todos = pd.concat(dfs, ignore_index=True).dropna(subset=['retorno', 'tipo'])
    df_todos['tipo'] = pd.Categorical(df_todos['tipo'], categories=orden_tipo, ordered=True)

    fig = go.Figure()

    for resultado in ['Victoria', 'Empate', 'Derrota']:
        for nombre in lista_peleadores:
            grupo = df_todos[(df_todos['peleador'] == nombre) & (df_todos['resultado'] == resultado)]
            if grupo.empty:
                continue

            fig.add_trace(go.Scatter(
                x=grupo['retorno'],
                y=grupo['tipo'],
                mode='markers',
                name=nombre if resultado == 'Victoria' else None,
                legendgroup=nombre,
                showlegend=resultado == 'Victoria',
                marker=dict(
                    color=colores_peleador.get(nombre, '#888780'),
                    symbol=symbol_map[resultado],
                    size=size_map[resultado],
                    opacity=0.85,
                    line=dict(width=1.5, color='rgba(0,0,0,0.2)'),
                ),
                customdata=grupo[['oponente', 'fecha', 'finish', 'resultado', 'odds', 'n_pelea']].values,
                hovertemplate=(
                    f'<b>{nombre}</b> vs %{{customdata[0]}}<br>'
                    'Fecha         : %{customdata[1]}<br>'
                    'Resultado     : <b>%{customdata[3]}</b><br>'
                    'Finalización  : %{customdata[2]}<br>'
                    f'Odds ({label}): %{{customdata[4]}}<br>'
                    'Retorno neto  : $%{x:.2f}<br>'
                    'Pelea #%{customdata[5]}'
                    '<extra></extra>'
                ),
            ))

    fig.add_vline(x=0, line_dash='dash', line_color='rgba(128,128,128,0.5)', line_width=1.5,
                  annotation_text='Break-even', annotation_position='top', annotation_font=dict(size=9, color='gray'))
    fig.add_vline(x=-inversion, line_dash='dot', line_color='rgba(216,90,48,0.4)', line_width=1,
                  annotation_text=f'-${inversion}', annotation_position='bottom', annotation_font=dict(size=9, color='#D85A30'))

    for sym, lbl in [('circle','Victoria'), ('x','Derrota'), ('diamond','Empate')]:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(symbol=sym, size=9, color='gray'),
                                 name=lbl, showlegend=True, legendgroup='shapes'))

    fig.update_layout(
        title=dict(text=f'Historial de Retorno de Inversion Historico simulado — {label}<br><sup>Inversión constante de ${inversion} por combate</sup>', x=0, font=dict(size=14, color='gray')),
        xaxis=dict(title='Retorno neto (USD)', gridcolor='rgba(128,128,128,0.15)', zeroline=False, tickprefix='$'),
        yaxis=dict(title='Condición inicial de la cuota', gridcolor='rgba(128,128,128,0.15)', categoryorder='array', categoryarray=orden_tipo),
        legend=dict(orientation='h', yanchor='top', y=10, xanchor='center', x=1, font=dict(size=10)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=450,
        margin=dict(t=80, b=80, l=120, r=40),
    )
    return fig

def generar_scatter_rendimiento(lista_peleadores, df_ufc, metrica):
    """
    Genera un scatter plot para analizar una métrica física a lo largo del tiempo de pelea.
    """
    metrica_config = {
        'sig_str' : {'col': 'sig_str',   'label': 'Sig. Strikes por minuto'},
        'td_landed'  : {'col': 'td_landed',    'label': 'Takedowns lanzados'},
        'sig_str_pct': {'col': 'sig_str_pct',  'label': 'Precisión de strikes %'},
        'sub_att'    : {'col': 'sub_att',      'label': 'Intentos de sumisión'},
    }

    col_x    = metrica_config[metrica]['col']
    label_x  = metrica_config[metrica]['label']

    colores_peleador = {
        lista_peleadores[0]: '#378ADD',
        lista_peleadores[1]: '#D85A30',
    }

    symbol_map = {'Victoria': 'circle', 'Derrota': 'x', 'Empate': 'diamond'}
    size_map   = {'Victoria': 10, 'Derrota': 13, 'Empate': 10}

    dfs = []
    for p in lista_peleadores:
        df_h = get_historial_scatter(p, df_ufc)
        if not df_h.empty:
            dfs.append(df_h)
            
    if not dfs:
        return go.Figure()

    df_todos = pd.concat(dfs, ignore_index=True).dropna(subset=[col_x, 'dur_min'])
    
    fig = go.Figure()
    orden_resultado = ['Victoria', 'Empate', 'Derrota']

    for resultado in orden_resultado:
        for nombre in lista_peleadores:
            grupo = df_todos[(df_todos['peleador'] == nombre) & (df_todos['resultado'] == resultado)]
            if grupo.empty:
                continue

            color = colores_peleador.get(nombre, '#888780')

            fig.add_trace(go.Scatter(
                x    = grupo['dur_min'],
                y    = grupo[col_x],
                mode = 'markers',
                name            = nombre if resultado == 'Victoria' else None,
                legendgroup     = nombre,
                showlegend      = resultado == 'Victoria',
                marker = dict(
                    color   = color,
                    symbol  = symbol_map[resultado],
                    size    = size_map[resultado],
                    opacity = 0.85,
                    line    = dict(width = 1.5, color = 'rgba(0,0,0,0.25)'),
                ),
                customdata    = grupo[['oponente', 'fecha', 'finish', 'resultado', 'n_pelea']].values,
                hovertemplate = (
                    f'<b>{nombre}</b> vs %{{customdata[0]}}<br>'
                    'Fecha       : %{customdata[1]}<br>'
                    'Resultado   : <b>%{customdata[3]}</b><br>'
                    'Finalización: %{customdata[2]}<br>'
                    f'{label_x}: %{{y:.2f}}<br>'
                    'Duración    : %{x:.1f} min<br>'
                    'Pelea #%{customdata[4]}'
                    '<extra></extra>'
                ),
            ))

    for r in range(1, 6):
        fig.add_vline(x=r * 5, line_dash='dot', line_color='rgba(128,128,128,0.25)', line_width=1,
                      annotation_text=f'R{r}', annotation_position='top', annotation_font=dict(size=9, color='gray'))

    avg = df_todos[col_x].mean()
    fig.add_hline(y=avg, line_dash='dash', line_color='rgba(128,128,128,0.4)', line_width=1,
                  annotation_text=f'Promedio actual: {avg:.2f}', annotation_position='right', annotation_font=dict(size=9, color='gray'))

    for sym, lbl in [('circle','Victoria'), ('x','Derrota'), ('diamond','Empate')]:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(symbol=sym, size=9, color='gray'),
                                 name=lbl, showlegend=True, legendgroup='shapes'))

    fig.update_layout(
        title=dict(text=f'Evolución de Rendimiento Histórico<br><sup>X = Duración del combate · Y = {label_x}</sup>', x=0, font=dict(size=14, color='gray')),
        xaxis=dict(title='Duración acumulada (minutos)', gridcolor='rgba(128,128,128,0.15)', zeroline=False, range=[0,27]),
        yaxis=dict(title=label_x, gridcolor='rgba(128,128,128,0.15)', zeroline=False),
        legend=dict(orientation='h', yanchor='top', y=10, xanchor='center', x=1, font=dict(size=10)),
        paper_bgcolor = 'rgba(0,0,0,0)',
        plot_bgcolor  = 'rgba(0,0,0,0)',
        height=480,
        margin=dict(t=80, b=80, l=60, r=40),
    )
    return fig

def generar_chart_distribucion_tiempo(lista_peleadores, df_ufc):
    """
    Crea un gráfico de distribución acumulada/densidad para comparar la 
    duración de las peleas de ambos peleadores.
    """
    colores_peleador = {
        lista_peleadores[0]: '#378ADD',
        lista_peleadores[1]: '#D85A30',
    }
    
    fig = go.Figure()
    
    for nombre in lista_peleadores:
        df_tiempos = get_historial_tiempos(nombre, df_ufc)
        if df_tiempos.empty:
            continue
            
        color = colores_peleador.get(nombre, '#888780')
        
        fig.add_trace(go.Histogram(
            x=df_tiempos['dur_min'],
            name=nombre,
            histnorm='probability density',
            nbinsx=15,
            marker_color=color,
            opacity=0.45,
            autobinx=False,
            xbins=dict(start=0, end=25, size=2.5),
        ))
        
        promedio = df_tiempos['dur_min'].mean()
        fig.add_vline(
            x=promedio,
            line_dash='dash',
            line_color=color,
            line_width=2,
            annotation_text=f"Prom. {nombre.split()[-1]}: {promedio:.1f} min",
            annotation_position="top right",
            annotation_font=dict(size=9, color=color)
        )

    for r in range(1, 6):
        fig.add_vline(x=r * 5, line_dash='dot', line_color='rgba(128,128,128,0.25)', line_width=1,
                      annotation_text=f'R{r}', annotation_position='top', annotation_font=dict(size=9, color='gray'))

    fig.update_layout(
        title=dict(
            text='Distribución del Tiempo de Pelea<br>',
            x=0, font=dict(size=14, color='gray')
        ),
        xaxis=dict(
            title='Duración de la Pelea (Minutos)',
            gridcolor='rgba(128,128,128,0.15)',
            range=[0, 27],
            dtick=5
        ),
        yaxis=dict(
            title='Densidad de Frecuencia',
            gridcolor='rgba(128,128,128,0.15)',
            showticklabels=False 
        ),
        barmode='overlay', 
        legend=dict(orientation='h', yanchor='top', y=10, xanchor='center', x=1),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=420,
        margin=dict(t=80, b=60, l=40, r=40),
    )
    return fig

def generar_chart_record_barras(lista_peleadores, df_peleador_perfil):
    """
    Genera un gráfico de barras horizontales independientes donde cada barra
    lleva escrito su propio tipo de resultado (Victoria/Derrota/Empate) y se oculta la leyenda.
    """
    df_resumen = df_peleador_perfil[df_peleador_perfil['fighter'].isin(lista_peleadores)].copy()
    
    categorias = {
        'wins': {'label': 'Victoria', 'color': '#1D9E75'},
        'losses': {'label': 'Derrota', 'color': '#D85A30'},
        'draws': {'label': 'Empate', 'color': '#888780'}
    }
    
    fig = go.Figure()
    
    for col, info in categorias.items():
        valores = df_resumen[col].fillna(0).tolist() if col in df_resumen.columns else len(df_resumen)
        nombres = df_resumen['fighter'].tolist() if 'fighter' in df_resumen.columns else lista_peleadores
        
        textos_barras = [f"{info['label']}: {int(val)}" for val in valores]
        
        fig.add_trace(go.Bar(
            y=nombres,  
            x=valores,  
            orientation='h',  
            marker_color=info['color'],
            text=textos_barras,
            textposition='auto',       
            textfont=dict(weight='bold'),
            showlegend=False,
            hovertemplate=f'<b>%{{y}}</b><br>{info["label"]}: %{{x}}<extra></extra>'
        ))

    fig.update_layout(
        title=dict(
            text='Récord Histórico (UFC)',
            x=0, font=dict(size=14, color='gray')
        ),
        xaxis=dict(
            gridcolor='rgba(128,128,128,0.15)',
            zeroline=False
        ),
        yaxis=dict(
            gridcolor='rgba(128,128,128,0.11)',
            #autorange="reversed"
        ),
        barmode='group', 
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=320,
        margin=dict(t=60, b=40, l=140, r=60),
    )
    return fig

def generar_boxplot_tiempo_divisiones(df_ufc):
    """
    Genera un gráfico de cajas (Box Plot) para comparar la distribución de la
    duración de las peleas entre todas las divisiones de peso de la UFC.
    """
    if 'total_fight_time_secs' not in df_ufc.columns or 'weight_class' not in df_ufc.columns:
        return go.Figure()
        
    df_clean = df_ufc.dropna(subset=['total_fight_time_secs', 'weight_class']).copy()
    df_clean['dur_min'] = df_clean['total_fight_time_secs'] / 60

    conteos = df_clean['weight_class'].value_counts()
    divisiones_validas = conteos[conteos > 15].index
    df_clean = df_clean[df_clean['weight_class'].isin(divisiones_validas)]

    orden_divisiones = df_clean.groupby('weight_class')['dur_min'].median().sort_values().index

    fig = go.Figure()

    for division in orden_divisiones:
        df_div = df_clean[df_clean['weight_class'] == division]
        
        fig.add_trace(go.Box(
            y=df_div['dur_min'],
            name=division,
            boxpoints='outliers', 
            marker=dict(size=3, opacity=0.5),
            line=dict(width=1.5),
            notched=False
        ))

    for r in [0, 27]:
        fig.add_hline(
            y=r * 5, line_dash='dash', line_color='rgba(128,128,128,0.3)', line_width=1,
            annotation_text=f'Fin R{r}' if r==1 else f'Decisión (3R)' if r==3 else f'Decisión (5R)',
            annotation_position='right', annotation_font=dict(size=8, color='gray')
        )

    fig.update_layout(
        title=dict(
            text='Distribución y Tendencia del Tiempo de Pelea por División de Peso<br><sup>Las divisiones están ordenadas de menor a mayor duración promedio de combate</sup>',
            x=0, font=dict(size=14, color='gray')
        ),
        xaxis=dict(
            title='Divisiones de Peso',
            tickangle=45,
            gridcolor='rgba(128,128,128,0.1)'
        ),
        yaxis=dict(
            title='Duración del Combate (Minutos)',
            gridcolor='rgba(128,128,128,0.1)',
            range=[0,27],
            dtick=5,
            zeroline=False
        ),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=520,
        margin=dict(t=80, b=100, l=60, r=80),
    )
    return fig

def generar_heatmap_correlaciones(df_ufc, tipo_odds, label_odds):
    """
    Calcula la correlación de Pearson y aísla EXCLUSIVAMENTE el cruce de los atributos
    físicos/rendimiento contra el tipo de apuesta especificado al comenzar.
    Muestra una sola columna limpia (Rectangular) donde VERDE = Favoritismo.
    """
    columna_odds_real = None
    posibles_odds_cols = [tipo_odds, f"R_{tipo_odds}", f"B_{tipo_odds}", f"avg_{tipo_odds}", f"R_{tipo_odds.lower()}", f"B_{tipo_odds.lower()}"]
    
    for col in posibles_odds_cols:
        if col in df_ufc.columns:
            columna_odds_real = col
            break
            
    if not columna_odds_real:
        return go.Figure()

    atributos = {
        'dec_odds': 'Odds Decisión',
        'sub_odds': 'Odds Sumisión',
        'ko_odds': 'Odds KO/TKO',
        'avg_SIG_STR_landed': 'Strikes Conectados',
        'avg_SIG_STR_pct': 'Precisión Strikes',
        'avg_TD_landed': 'Takedowns Conectados',
        'avg_TD_pct': 'Precisión Takedowns',
        'avg_SUB_ATT': 'Intentos Sumisión',
        'avg_REV': 'Reversiones',
        'avg_PASS': 'Pases de Guardia',
        'total_fight_time_secs': 'Tiempo de Pelea',
        'current_win_streak': 'Racha de Peleas Ganadas',
        'current_loss_streak': 'Racha de Peleas Perdidas',
        'Pound-for-Pound_rank': 'Rango Libra por Libra',
        'age': 'Edad',
        'Height_cms': 'Altura',
        'Weight_lbs': 'Peso',
        'Reach_cms': 'Alcance'
    }

    columnas_disponibles = [columna_odds_real]
    labels_atributos_finales = []

    for col_base, label_base in atributos.items():
        if col_base == tipo_odds:
            continue
            
        if col_base == 'total_fight_time_secs':
            if col_base in df_ufc.columns:
                columnas_disponibles.append(col_base)
                labels_atributos_finales.append('Tiempo de Pelea (min)')
        else:
            posibles_cols = [col_base, f'R_{col_base}', f'B_{col_base}', f'R_{col_base.lower()}', f'B_{col_base.lower()}']
            for col in posibles_cols:
                if col in df_ufc.columns and col not in columnas_disponibles:
                    columnas_disponibles.append(col)
                    labels_atributos_finales.append(label_base)
                    break

    df_heat = df_ufc[columnas_disponibles].dropna()
    if df_heat.empty or len(columnas_disponibles) < 2:
        return go.Figure()

    if 'total_fight_time_secs' in df_heat.columns:
        df_heat['total_fight_time_secs'] = df_heat['total_fight_time_secs'] / 60

    matriz_completa = df_heat.corr().round(2)
    
    todos_los_labels = [label_odds] + labels_atributos_finales
    matriz_completa.index = todos_los_labels
    matriz_completa.columns = todos_los_labels

    matriz_exclusiva = matriz_completa.loc[labels_atributos_finales, [label_odds]].copy()

    matriz_exclusiva[label_odds] = matriz_exclusiva[label_odds] * -1

    fig2 = go.Figure(go.Heatmap(
        z             = matriz_exclusiva.values,
        x             = matriz_exclusiva.columns.tolist(),
        y             = matriz_exclusiva.index.tolist(),
        colorscale    = [[0.0, '#D85A30'], [0.5, '#F1EFE8'], [1.0, '#1D9E75']], 
        zmid          = 0,
        zmin          = -1,
        zmax          = 1,
        text          = matriz_exclusiva.values.round(2),
        texttemplate  = '%{text}',
        textfont      = dict(size=11, weight='bold'),
        hovertemplate = 'Atributo: <b>%{y}</b><br>Impacto en ' + label_odds + ': <b>%{z:.2f}</b><extra></extra>',
        colorbar      = dict(
            title      = 'Impacto de<br>Favoritismo',
            tickvals   = [-1, -0.5, 0, 0.5, 1],
            ticktext   = ['Underdog (-1)', '-0.5', 'Neutro (0)', '0.5', 'Favorito (1)'],
            thickness  = 14,
        ),
    ))

    fig2.update_layout(
        title=dict(
            text=f'Análisis de Influencia Directa sobre: {label_odds}<br><sup><b>VERDE</b> = Atributos que te hacen favorito en este mercado · <b>NARANJA</b> = Atributos que aumentan el pago (Underdog)</sup>',
            x=0, font=dict(size=13, color='gray'),
        ),
        paper_bgcolor = 'rgba(0,0,0,0)',
        plot_bgcolor  = 'rgba(0,0,0,0)',
        height        = 580,
        margin        = dict(t=80, b=40, l=180, r=40),
        xaxis         = dict(tickfont=dict(size=11, weight='bold'), zeroline=False, side='top'), # La apuesta queda fija arriba
        yaxis         = dict(tickfont=dict(size=11), zeroline=False),
    )
    return fig2

def generar_chart_finalizaciones_divisiones(df_ufc):
    """
    Genera un gráfico de barras verticales apiladas al 100% buscando dinámicamente
    las divisiones de peso y mapeando la columna 'finish' para el método de victoria.
    """
    col_weight = None
    posibles_weight = ['weight_class', 'WeightClass', 'weightclass', 'division', 'w_class', 'WEIGHT_CLASS']
    for col in posibles_weight:
        if col in df_ufc.columns:
            col_weight = col
            break
            
    col_method = None
    posibles_method = ['finish', 'method', 'Method', 'win_by', 'FINISH']
    for col in posibles_method:
        if col in df_ufc.columns:
            col_method = col
            break

    if not col_weight or not col_method:
        return go.Figure()

    df_clean = df_ufc.dropna(subset=[col_weight, col_method]).copy()

    def clasificar_metodo(m):
        m = str(m).upper()
        if 'KO' in m or 'TKO' in m: 
            return 'ko'
        elif 'SUB' in m or 'SUM' in m or 'TAP' in m: 
            return 'sub'
        elif 'DEC' in m or 'JUDG' in m or 'PTS' in m: 
            return 'dec'
        return 'otros'

    df_clean['metodo_cat'] = df_clean[col_method].apply(clasificar_metodo)

    conteos_div = df_clean[col_weight].value_counts()
    div_validas = conteos_div[conteos_div > 15].index
    df_clean = df_clean[df_clean[col_weight].isin(div_validas)]

    df_counts = df_clean.groupby([col_weight, 'metodo_cat']).size().unstack(fill_value=0)
    
    for c in ['ko', 'sub', 'dec']:
        if c not in df_counts.columns:
            df_counts[c] = 0
            
    totales = df_counts.sum(axis=1)
    
    div_stats = pd.DataFrame()
    div_stats['ko_pct']  = df_counts['ko'] / totales
    div_stats['sub_pct'] = df_counts['sub'] / totales
    div_stats['dec_pct'] = df_counts['dec'] / totales
    
    div_stats = div_stats.sort_values(by='ko_pct', ascending=False)
    xtick_labels = div_stats.index.tolist()

    fig = go.Figure()

    bloques = [
        {'col': 'ko_pct',  'name': 'KO/TKO',  'color': '#378ADD'},
        {'col': 'sub_pct', 'name': 'Sumisión', 'color': '#1D9E75'},
        {'col': 'dec_pct', 'name': 'Decisión', 'color': '#888780'}
    ]

    for b in bloques:
        textos_barra = [
            f"{val:.0%}" if val > 0.05 else "" 
            for val in div_stats[b['col']]
        ]

        fig.add_trace(go.Bar(
            x             = xtick_labels,
            y             = div_stats[b['col']],
            name          = b['name'],
            marker_color  = b['color'],
            text          = textos_barra,
            textposition  = 'inside',         
            textfont      = dict(size=10, color='white', weight='bold'),
            insidetextanchor = 'middle',
            hovertemplate = 'División: <b>%{x}</b><br>' + b['name'] + ': <b>%{y:.1%}</b><extra></extra>'
        ))

    fig.update_layout(
        title=dict(
            text='Manera de finalización por división de peso',
            x=0, font=dict(size=14, color='gray')
        ),
        barmode          = 'stack', 
        paper_bgcolor    = 'rgba(0,0,0,0)',
        plot_bgcolor     = 'rgba(0,0,0,0)',
        height           = 520,
        margin           = dict(t=80, b=120, l=60, r=40),
        legend           = dict(orientation='h', yanchor='bottom', y=-0.25, xanchor='center', x=0.5),
        
        yaxis=dict(
            title      = '% de peleas',
            tickformat = '.0%',
            autorange  = True,             
            zeroline   = False,
            gridcolor  = 'rgba(128,128,128,0.1)'
        ),
        xaxis=dict(
            tickangle = -20,
            tickfont  = dict(size=10),
            zeroline  = False
        )
    )

    return fig

def generar_analisis_pca(df_ufc):
    """
    Realiza un Análisis de Componentes Principales (PCA) sobre las características 
    de rendimiento de los peleadores para reducir la dimensionalidad y encontrar patrones.
    """
    columnas_base = [
        'avg_SIG_STR_landed', 'avg_SIG_STR_pct', 
        'avg_TD_landed', 'avg_TD_pct', 
        'avg_SUB_ATT', 'avg_REV', 'avg_PASS',
        'age', 'Height_cms', 'Reach_cms', 'Weight_lbs','odds', 
        'current_win_streak', 'current_lose_streak',
        #'win_by_KO/TKO', 'win_by_Submission', 'wins',
        #'win_by_Decision_Majority', 'win_by_Decision_Split', 'win_by_Decision_Unanimous',
    ]
    
    columnas_analisis = []
    mapeo_labels = {}
    
    for col in columnas_base:
        for posible in [col, f'R_{col}', f'B_{col}']:
            if posible in df_ufc.columns:
                columnas_analisis.append(posible)
                mapeo_labels[posible] = col.replace('avg_', '').replace('_cms', '')
                break

    if len(columnas_analisis) < 3:
        return None, None

    df_pca_input = df_ufc[columnas_analisis].dropna().copy()
    
    if len(df_pca_input) < 10:
        return None, None

    scaler = StandardScaler()
    datos_escalados = scaler.fit_transform(df_pca_input)

    n_components = min(3, datos_escalados.shape[1])
    pca = PCA(n_components=n_components)
    componentes_principales = pca.fit_transform(datos_escalados)

    df_resultado = pd.DataFrame(
        data = componentes_principales, 
        columns = [f'PC{i+1}' for i in range(n_components)]
    )

    varianza_exp = pca.explained_variance_ratio_
    varianza_acum = varianza_exp.cumsum()

    fig_varianza = go.Figure()
    fig_varianza.add_trace(go.Bar(
        x=[f'PC{i+1}' for i in range(n_components)], y=varianza_exp,
        name='Varianza Individual', marker_color='#378ADD', text=[f'{v:.1%}' for v in varianza_exp], textposition='auto'
    ))
    fig_varianza.add_trace(go.Scatter(
        x=[f'PC{i+1}' for i in range(n_components)], y=varianza_acum,
        name='Varianza Acumulada', line=dict(color='#D85A30', width=3), mode='lines+markers'
    ))
    fig_varianza.update_layout(
        title='Codo de Varianza Explicada por Componente',
        xaxis_title='Componentes Principales', yaxis_title='% Varianza Explicada',
        yaxis=dict(tickformat='.0%'), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=350, margin=dict(t=50, b=40, l=50, r=20), legend=dict(orientation='h', y=-0.2)
    )

    cargas = pca.components_
    fig_biplot = go.Figure()

    for i, col in enumerate(columnas_analisis):
        fig_biplot.add_trace(go.Scatter(
            x=[0, cargas[0, i]], y=[0, cargas[1, i]],
            mode='lines+markers+text',
            name=mapeo_labels[col],
            text=["", mapeo_labels[col]],
            textposition="top center",
            marker=dict(size=6),
            line=dict(width=2.5),
            hovertemplate=f'<b>{mapeo_labels[col]}</b><br>Peso PC1: {cargas[0, i]:.2f}<br>Peso PC2: {cargas[1, i]:.2f}<extra></extra>'
        ))

    fig_biplot.update_layout(
        title='Mapa de Impacto: ¿Qué mide realmente cada Componente Principal? (PC1 x PC2)',
        xaxis=dict(title='Componente Principal 1 (PC1)', zeroline=True, zerolinewidth=1.5, zerolinecolor='gray'),
        yaxis=dict(title='Componente Principal 2 (PC2)', zeroline=True, zerolinewidth=1.5, zerolinecolor='gray'),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=500, margin=dict(t=60, b=40, l=50, r=40)
    )

    return fig_varianza, fig_biplot

# 2. Definición del Diálogo
@st.dialog("Elige el peleador")
def elegir_peleador_dialog(slot):
    st.write(f"Configurando **{slot.replace('_', ' ').title()}**")
    
    genero = st.selectbox(
        "Género del peleador",
        options=["Male", "Female"],
        format_func=lambda x: "Hombre" if x == "Male" else "Mujer"
    )
    
    df_filtrado_genero = fighters_info[fighters_info['gender'] == genero]
    divisiones_disponibles = sorted(df_filtrado_genero['weight_class'].unique())
    
    division = st.selectbox(
        "División / Categoría de peso",
        options=divisiones_disponibles
    )
    
    df_filtrado_final = df_filtrado_genero[df_filtrado_genero['weight_class'] == division]
    nombres_disponibles = sorted(df_filtrado_final['fighter'].unique())
    
    nombre_seleccionado = st.selectbox(
        "Selecciona el Peleador",
        options=nombres_disponibles
    )
    
    if st.button("Confirmar Selección", type="primary", use_container_width=True):
        st.session_state[slot] = {
            "nombre": nombre_seleccionado,
            "division": division,
            "genero": "Hombre" if genero == "Male" else "Mujer"
        }
        st.rerun()

cols_finish = [
    'R_fighter', 'B_fighter', 'Winner', 'finish', 'finish_details', 'finish_round', 'finish_round_time'
    ]

@st.dialog("📋 Historial de Combates Detallado", width="large")
def mostrar_modal_historial(nombre_peleador, df_filtrado):
    """
    Función interna que renderiza la tabla dentro de la ventana emergente.
    """
    st.markdown(f"### Registros oficiales de **{nombre_peleador}**")
    
    if not df_filtrado.empty:
        if 'date' in df_filtrado.columns:
            df_filtrado = df_filtrado.sort_values(by='date', ascending=False)
            
        st.dataframe(
            df_filtrado, 
            use_container_width=True,
            hide_index=True 
        )
        st.caption(f"Mostrando {len(df_filtrado)} combates encontrados en el histórico de la UFC.")
    else:
        st.info(f"No se encontraron registros tabulares detallados para {nombre_peleador}.")

col1, col2 = st.columns(2, width="stretch")

with col1:
    st.header("Peleador 1")
    if "peleador_1" not in st.session_state:
        st.info("Ninguno seleccionado")
        if st.button("Seleccionar Peleador 1", key="btn_p1"):
            elegir_peleador_dialog("peleador_1")
    else:
        p1 = st.session_state["peleador_1"]
        st.success(f"**{p1['nombre']}**")
        st.caption(f"{p1['division']} ({p1['genero']})")
        if st.button("Cambiar Peleador 1", key="btn_p1_c"):
            elegir_peleador_dialog("peleador_1")

with col2:
    st.header("Peleador 2")
    if "peleador_2" not in st.session_state:
        st.info("Ninguno seleccionado")
        if st.button("Seleccionar Peleador 2", key="btn_p2"):
            elegir_peleador_dialog("peleador_2")
    else:
        p2 = st.session_state["peleador_2"]
        st.success(f"**{p2['nombre']}**")
        st.caption(f"{p2['division']} ({p2['genero']})")
        if st.button("Cambiar Peleador 2", key="btn_p2_c"):
            elegir_peleador_dialog("peleador_2")

st.divider()

if "peleador_1" in st.session_state and "peleador_2" in st.session_state:
    
    n1 = st.session_state["peleador_1"]["nombre"]
    n2 = st.session_state["peleador_2"]["nombre"]

    perf_p1 = df_peleador_perfil[df_peleador_perfil['fighter'] == n1]
    perf_p2 = df_peleador_perfil[df_peleador_perfil['fighter'] == n2]
    
    acc_p1 = df_peleador_acciones.set_index('fighter').loc[n1]
    acc_p2 = df_peleador_acciones.set_index('fighter').loc[n2]

    mask_tabla_1 = (df_ufc['R_fighter'] == n1) | (df_ufc['B_fighter'] == n1)
    df_mostrar_1 = df_ufc[mask_tabla_1][cols_finish].copy()

    mask_tabla_2 = (df_ufc['R_fighter'] == n2) | (df_ufc['B_fighter'] == n2)
    df_mostrar_2 = df_ufc[mask_tabla_2][cols_finish].copy()

    col1, col2, col3 = st.columns([1, 1, 3], width="stretch", border=True)

    with col1:
        img, info = st.columns([1, 2])
        with img:
            #st.image("perfil_vacio.png"#, caption="Foto del peleador"
            #         )
            st.write("Imagen referencial del peleador")
        with info:
            st.markdown(f"**Nombre**: {n1}")
            st.markdown(f"**Edad**: {int(perf_p1['age'].values)}")
            st.markdown(f"**Estatura**: {int(perf_p1['Height_cms'].values)} cm")
            st.markdown(f"**Peso**: {int(perf_p1['Weight_lbs'].values)} lbs")
            st.markdown(f"**Alcance**: {int(perf_p1['Reach_cms'].values)} cm")

        fig_radar = generar_radar_chart([n1], df_peleador_acciones)
        st.plotly_chart(fig_radar, use_container_width=True)

        if st.button(f"Ver historial de {n1}", use_container_width=True, key="btn_h1"):
            mostrar_modal_historial(n1, df_mostrar_1)

    with col2:
        img, info = st.columns([1, 2])
        with img:
            st.image("perfil_vacio.png"#, caption="Foto del peleador"
                     )
        with info:
            st.markdown(f"**Nombre**: {n2}")
            st.markdown(f"**Edad**: {int(perf_p2['age'].values)}")
            st.markdown(f"**Estatura**: {int(perf_p2['Height_cms'].values)} cm")
            st.markdown(f"**Peso**: {int(perf_p2['Weight_lbs'].values)} lbs")
            st.markdown(f"**Alcance**: {int(perf_p2['Reach_cms'].values)} cm")

        fig_radar = generar_radar_chart([n2], df_peleador_acciones)
        st.plotly_chart(fig_radar, use_container_width=True)

        if st.button(f"Ver historial de {n2}", use_container_width=True, key="btn_h2"):
            mostrar_modal_historial(n2, df_mostrar_2)

    with col3:
        radar, donas = st.columns(2)
        with radar:
            fig_radar = generar_radar_chart([n1, n2], df_peleador_acciones)
            st.plotly_chart(fig_radar, use_container_width=True)

            fig_record = generar_chart_record_barras([n1, n2], df_peleador_perfil)
            st.plotly_chart(fig_record, use_container_width=True)

        with donas:
            fig_donas = generar_donas_finalizaciones([n1, n2], df_ufc)
            st.plotly_chart(fig_donas, use_container_width=True)

            fig_derrotas = generar_donas_derrotas([n1, n2], df_ufc)
            st.plotly_chart(fig_derrotas, use_container_width=True)

    tab_rendimiento, tab_ROI ,tab_tiempo = st.tabs([
            "Rendimiento por Round",
            "Retorno de Inversión Historico",
            "Distribución de Tiempo"
        ])

    with tab_rendimiento:
        metrica_seleccionada = st.selectbox(
            "Métrica de rendimiento a evaluar",
            options=['sig_str', 'td_landed', 'sig_str_pct', 'sub_att'],
            format_func=lambda x: {
                'sig_str': 'Strikes Significativos / min',
                'td_landed': 'Takedowns Conectados',
                'sig_str_pct': 'Porcentaje de Precisión (%)',
                'sub_att': 'Intentos de Sumisión'
            }[x]
        )

        fig_scatter = generar_scatter_rendimiento([n1, n2], df_ufc, metrica=metrica_seleccionada)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with tab_ROI:
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            tipo_seleccionado = st.selectbox(
                "Mercado de cuotas (Odds)",
                options=['odds', 'dec_odds', 'sub_odds', 'ko_odds'],
                format_func=lambda x: {
                    'odds': 'Odds Generales (Ganador)',
                    'dec_odds': 'Método: Decisión',
                    'sub_odds': 'Método: Sumisión',
                    'ko_odds': 'Método: KO/TKO'
                }[x]
            )
        with col_opt2:
            monto_inversion = st.number_input("Inversión base por pelea ($)", min_value=10, max_value=1000, value=100, step=10)
        
        fig_roi = generar_chart_roi([n1, n2], df_ufc, tipo_odds=tipo_seleccionado, inversion=monto_inversion)
        st.plotly_chart(fig_roi, use_container_width=True)
            
    with tab_tiempo:
        fig_campana = generar_chart_distribucion_tiempo([n1, n2], df_ufc)
        st.plotly_chart(fig_campana, use_container_width=True)

    tab_tiempo_div, tab_final_div ,tab_corr_ap = st.tabs([
            "Tiempo por División",
            "Finalización por División",
            "Correlación por Apuestas"
        ])
    
    with tab_tiempo_div:
        fig_box = generar_boxplot_tiempo_divisiones(df_ufc)
        
        if fig_box.data:
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.warning("No hay suficientes registros de tiempo para procesar el análisis por divisiones.")

    with tab_final_div:
        fig_fin_div = generar_chart_finalizaciones_divisiones(df_ufc)
        
        if fig_fin_div.data:
            st.plotly_chart(fig_fin_div, use_container_width=True)
        else:
            st.warning("No se encontraron los datos de 'weight_class' o 'method' para procesar el gráfico de finalizaciones.")

    with tab_corr_ap:
        label_actual = {
            'odds': 'Odds Generales'
        }[tipo_seleccionado]

        fig_heat_corr = generar_heatmap_correlaciones(df_ufc, tipo_odds=tipo_seleccionado, label_odds=label_actual)
        
        if fig_heat_corr.data:
            st.plotly_chart(fig_heat_corr, use_container_width=True)
        else:
            st.info("No hay suficientes cruces de variables para calcular la matriz de Pearson.")

    st.divider()
    st.subheader("🧬 Mapa de ADN de Combate: Análisis Multivariado PCA")
    st.markdown(
        """
        El **PCA (Análisis de Componentes Principales)** fusiona todas las métricas físicas y de rendimiento 
        para descubrir los verdaderos perfiles de los peleadores. En lugar de ver números sueltos, 
        este modelo matemático agrupa las estadísticas en ejes de comportamiento estilístico.
        """
    )

    fig_var, fig_bi = generar_analisis_pca(df_ufc)

    if fig_var is not None and fig_bi is not None:
        col_v, col_b = st.columns(2)

        with col_v:
            st.markdown("### **Confiabilidad Matemática del Modelo**")
            st.plotly_chart(fig_var, use_container_width=True)
            st.caption(
                "**Nota técnica:** La línea roja muestra que al usar solo los 3 primeros ejes, "
                "logramos capturar casi el 60% de toda la información y variabilidad histórica de la UFC. "
                "Esto valida que el mapa de la derecha es altamente representativo de la realidad."
            )

        with col_b:
            st.markdown("### **Mapa de Estilos: ¿Qué define a cada eje?**")
            st.plotly_chart(fig_bi, use_container_width=True)
            st.caption(
                "**Cómo leer el mapa:** "
                "El **Eje Horizontal (PC1)** representa la **Morfología**: hacia la derecha están los peleadores altos y con mucho alcance (`Reach`, `Height`). "
                "El **Eje Vertical (PC2)** representa la **Estrategia de Suelo**: hacia arriba están los expertos en lucha y derribos (`TD_landed`, `TD_pct`, `SUB_ATT`). "
            )
    else:
        st.info(
            "No hay suficientes registros numéricos sin valores nulos "
            "para calcular la matriz de covarianza del PCA."
        )
        
else:
    st.warning("Por favor, selecciona ambos peleadores en la parte superior para ver la comparación.")