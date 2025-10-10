import os
from typing import Dict, List, Optional, Tuple
from functools import lru_cache

import geopandas as gpd
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import unidecode
from dash import Dash, Input, Output, dcc, html
import dash_bootstrap_components as dbc

# Server / App
app = Dash(__name__, external_stylesheets=[
    dbc.themes.FLATLY,
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"
])
server = app.server
app.config.suppress_callback_exceptions = True

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
SHAPE_PATH = os.path.join(os.path.dirname(__file__), 'coordenadas', 'COLOMBIA', 'COLOMBIA.shp')

def _list_data_files(data_dir: str) -> List[str]:
    """Return sorted list of Excel files in data directory."""
    if not os.path.isdir(data_dir):
        return []
    files: List[str] = []
    for fn in os.listdir(data_dir):
        if fn.lower().endswith(('.xls', '.xlsx')):
            files.append(os.path.join(data_dir, fn))
    files.sort()
    return files


def _read_excel(path: str) -> Tuple[Optional[pd.DataFrame], Optional[Exception]]:
    """Try to read an Excel file using appropriate engine fallbacks."""
    ext = os.path.splitext(path)[1].lower()
    attempts: List[Dict[str, str]] = []
    if ext == '.xls':
        attempts.append({'engine': 'xlrd'})
    elif ext in {'.xlsx', '.xlsm'}:
        attempts.append({'engine': 'openpyxl'})
    attempts.append({})  # pandas default as last resort

    last_exc: Optional[Exception] = None
    for opts in attempts:
        try:
            df = pd.read_excel(path, **opts)
            return df, None
        except Exception as exc:  # pragma: no cover - logging only
            last_exc = exc
    return None, last_exc


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe with normalized ASCII column names."""
    df = df.copy()
    df.columns = [unidecode.unidecode(str(col)).strip() for col in df.columns]
    return df


def _prepare_dataset() -> Tuple[pd.DataFrame, List[str]]:
    """Load all data files, concatenating them and returning load warnings."""
    files = _list_data_files(DATA_DIR)
    warnings: List[str] = []

    if not files:
        fallback = os.path.join(os.path.dirname(__file__), '..', 'Geo', 'Mortalidad_Materna.xlsx')
        if os.path.exists(fallback):
            files = [fallback]
        else:
            warnings.append('No se encontraron archivos en la carpeta data.')
            return pd.DataFrame(columns=['Departamento_residencia', 'FEC_DEF', 'ANO', 'Municipio_residencia']), warnings

    frames: List[pd.DataFrame] = []
    for path in files:
        df, err = _read_excel(path)
        if df is None:
            warnings.append(f"No se pudo leer {os.path.basename(path)}: {err}")
            continue
        df = _normalize_columns(df)
        df['source_file'] = os.path.basename(path)
        frames.append(df)

    if not frames:
        warnings.append('Ningún archivo se pudo cargar correctamente.')
        return pd.DataFrame(columns=['Departamento_residencia', 'FEC_DEF', 'ANO', 'Municipio_residencia']), warnings

    combined = pd.concat(frames, ignore_index=True)

    # Ensure core columns exist
    for col in ['Departamento_residencia', 'Municipio_residencia', 'FEC_DEF']:
        if col not in combined.columns:
            combined[col] = pd.NA

    if 'ANO' not in combined.columns or not combined['ANO'].notna().any():
        combined['ANO'] = pd.to_datetime(combined['FEC_DEF'], errors='coerce').dt.year
    else:
        combined['ANO'] = pd.to_numeric(combined['ANO'], errors='coerce').astype('Int64')

    # Extract month from FEC_DEF for monthly analysis
    combined['MES'] = pd.to_datetime(combined['FEC_DEF'], errors='coerce').dt.month
    combined['MES_NOMBRE'] = pd.to_datetime(combined['FEC_DEF'], errors='coerce').dt.month_name(locale='es_ES')
    # Fallback if locale not available
    if combined['MES_NOMBRE'].isna().all():
        meses = {1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio',
                 7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'}
        combined['MES_NOMBRE'] = combined['MES'].map(meses)

    combined['Departamento_residencia'] = (
        combined['Departamento_residencia']
        .astype(str)
        .str.strip()
        .str.upper()
        .map(unidecode.unidecode)
        .replace({'<NA>': pd.NA})
    )

    return combined, warnings


_df, _load_warnings = _prepare_dataset()

# Load shapefile
gdf_map = gpd.read_file(SHAPE_PATH)
# Normalize department name column if exists
if 'DPTO_CNMBR' in gdf_map.columns:
    gdf_map['DPTO_CNMBR'] = gdf_map['DPTO_CNMBR'].astype(str).str.strip().str.upper().map(unidecode.unidecode)

# Prepare case points: compute department centroids safely by projecting to a metric CRS first
# then transform centroids back to EPSG:4326 to get lon/lat.
try:
    gdf_map = gdf_map.to_crs(epsg=4326)
except Exception:
    # if shapefile has no CRS, proceed but centroids may be unreliable
    pass

# compute centroids in projected CRS to avoid geodesic centroid warnings
gdf_proj = gdf_map.to_crs(epsg=3857)
if 'DPTO_CNMBR' in gdf_proj.columns:
    centroids = gdf_proj[['DPTO_CNMBR', 'geometry']].copy()
    centroids['centroid'] = centroids.geometry.centroid
else:
    centroids = gdf_proj.copy()
    centroids['DPTO_CNMBR'] = centroids.index.astype(str)
    centroids['centroid'] = centroids.geometry.centroid

# transform centroids back to geographic coordinates for map display
centroids = centroids.set_geometry('centroid').to_crs(epsg=4326)
centroids['lon'] = centroids.geometry.x
centroids['lat'] = centroids.geometry.y
centroids = centroids.reset_index(drop=True)

# Map department name in df to centroids
_cases = _df.copy()
_cases['Departamento_residencia'] = _cases['Departamento_residencia'].fillna('SIN_DEPARTAMENTO')
_cases = _cases.merge(centroids[['DPTO_CNMBR', 'lon', 'lat']], left_on='Departamento_residencia', right_on='DPTO_CNMBR', how='left')

# For cases without match, leave lon/lat NaN

# Prepare GeoJSON from shapefile for choropleth
gdf_map_geo = gdf_map.copy()
gdf_map_geo = gdf_map_geo.to_crs(epsg=4326)

# Precalcular datos para optimización
_yearly_counts = _cases.groupby('ANO').size().to_dict()
_dept_yearly_counts = _cases.groupby(['Departamento_residencia', 'ANO']).size().to_dict()

# App layout
years = sorted(_cases['ANO'].dropna().unique().tolist()) if not _cases.empty else []
departments = sorted(_cases['Departamento_residencia'].dropna().unique().tolist()) if not _cases.empty else []

year_options = [{'label': 'Todos', 'value': 'Todos'}]
year_options.extend({'label': str(y), 'value': int(y) if pd.notna(y) else y} for y in years)
default_year = year_options[-1]['value'] if year_options else None

dept_options = [{'label': 'Todos', 'value': 'Todos'}]
dept_options.extend({'label': d, 'value': d} for d in departments if d)

if not dept_options:
    dept_options = [{'label': 'Todos', 'value': 'Todos'}]

notes_children: List = []
if _load_warnings:
    notes_children = [
        html.H6('Avisos de carga', className='mb-2'),
        html.Ul([html.Li(msg) for msg in _load_warnings], className='mb-0')
    ]
elif not _cases.empty:
    resumen = f"Archivos combinados: {_cases['source_file'].nunique()} | Registros: {len(_cases)}"
    notes_children = [html.Small(resumen)]
else:
    notes_children = [html.Small('No hay datos disponibles para mostrar.')] 

# Agregar siempre referencia a la fuente de datos (Sivigila)
notes_children.append(
    html.Div([
        html.Small("Fuente de datos: "),
        html.A("Portal de microdatos de Sivigila", href="https://portalsivigila.ins.gov.co/Paginas/Buscador.aspx", target="_blank", rel="noopener noreferrer")
    ], className='mt-1')
)

app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("Dashboard de Mortalidad Materna", className='text-center mb-0'),
            html.P("Colombia - Análisis de Datos", className='text-center text-muted')
        ], width=12)
    ], className='my-3 py-3 bg-light rounded'),

    # Filters
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("🔍 Filtros", className='card-title mb-3'),
                    html.Label("Año:", className='fw-bold'),
                    dcc.Dropdown(
                        id='year-filter', 
                        options=year_options, 
                        value=default_year, 
                        clearable=False, 
                        placeholder='Selecciona un año',
                        className='mb-3'
                    ),
                    html.Label("Departamento:", className='fw-bold'),
                    dcc.Dropdown(
                        id='dept-filter', 
                        options=dept_options, 
                        value='Todos', 
                        clearable=False,
                        className='mb-3'
                    ),
                    html.Label("Tipo de Mapa:", className='fw-bold'),
                    dcc.Dropdown(
                        id='map-style', 
                        options=[
                            {'label': 'Mapa Coroplético', 'value': 'choropleth'},
                            {'label': 'Mapa de Puntos', 'value': 'scatter'},
                            {'label': 'Mapa de Calor', 'value': 'density'}
                        ], 
                        value='choropleth', 
                        clearable=False,
                        className='mb-2'
                    ),
                ])
            ], className='shadow-sm')
        ], width=12, lg=3),

        # KPIs
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-notes-medical fa-2x text-danger mb-2"),
                                html.H6("Total de Casos", className='text-muted mb-1'),
                                html.H2(id='kpi-total', className='mb-0 text-primary fw-bold')
                            ], className='text-center')
                        ])
                    ], className='shadow-sm h-100')
                ], width=12, md=6, lg=3, className='mb-3'),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-calendar-alt fa-2x text-info mb-2"),
                                html.H6("Promedio Mensual", className='text-muted mb-1'),
                                html.H2(id='kpi-avg', className='mb-0 text-primary fw-bold')
                            ], className='text-center')
                        ])
                    ], className='shadow-sm h-100')
                ], width=12, md=6, lg=3, className='mb-3'),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-map-marker-alt fa-2x text-success mb-2"),
                                html.H6("Departamento Principal", className='text-muted mb-1'),
                                html.H5(id='kpi-top-dept', className='mb-0 text-primary fw-bold', style={'fontSize': '1.2rem'})
                            ], className='text-center')
                        ])
                    ], className='shadow-sm h-100')
                ], width=12, md=6, lg=3, className='mb-3'),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-chart-line fa-2x text-warning mb-2"),
                                html.H6("Tendencia Anual", className='text-muted mb-1'),
                                html.H2(id='kpi-trend', className='mb-0 fw-bold')
                            ], className='text-center')
                        ])
                    ], className='shadow-sm h-100')
                ], width=12, md=6, lg=3, className='mb-3'),
            ])
        ], width=12, lg=9)
    ], className='mb-4'),

    # Map and Time Series
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Distribución Geográfica de Casos", className='mb-0')),
                dbc.CardBody([
                    dcc.Graph(id='map', config={'displayModeBar': False}, style={'height': '500px'})
                ])
            ], className='shadow-sm')
        ], width=12, lg=8, className='mb-4'),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Tendencia Histórica", className='mb-0')),
                dbc.CardBody([
                    dcc.Graph(id='time-series', config={'displayModeBar': False}, style={'height': '500px'})
                ])
            ], className='shadow-sm')
        ], width=12, lg=4, className='mb-4'),
    ]),

    # Additional Charts
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Top 10 Departamentos", className='mb-0')),
                dbc.CardBody([
                    dcc.Graph(id='top-depts', config={'displayModeBar': False}, style={'height': '400px'})
                ])
            ], className='shadow-sm')
        ], width=12, lg=6, className='mb-4'),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Distribución Mensual", className='mb-0')),
                dbc.CardBody([
                    dcc.Graph(id='monthly-dist', config={'displayModeBar': False}, style={'height': '400px'})
                ])
            ], className='shadow-sm')
        ], width=12, lg=6, className='mb-4'),
    ]),

    # Footer Notes
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div(id='notes', children=notes_children)
                ])
            ], className='shadow-sm bg-light')
        ], width=12)
    ], className='mb-3')
], fluid=True, className='px-4')

# Callbacks

@lru_cache(maxsize=128)
def _filter_data(year, dept):
    """Cache filtered data to avoid redundant processing."""
    if year == 'Todos':
        dff = _cases.copy()
    else:
        dff = _cases[_cases['ANO'] == year].copy()
    if dept and dept != 'Todos':
        dff = dff[dff['Departamento_residencia'] == dept]
    return dff.to_json(date_format='iso', orient='split')

@app.callback(
    Output('map', 'figure'),
    Output('kpi-total', 'children'),
    Output('kpi-avg', 'children'),
    Output('kpi-top-dept', 'children'),
    Output('kpi-trend', 'children'),
    Output('time-series', 'figure'),
    Output('top-depts', 'figure'),
    Output('monthly-dist', 'figure'),
    Input('year-filter', 'value'),
    Input('dept-filter', 'value'),
    Input('map-style', 'value')
)
def update(year, dept, map_style):
    # Usar datos cacheados
    dff_json = _filter_data(year, dept)
    dff = pd.read_json(dff_json, orient='split')
    
    total = len(dff)

    # KPI: Promedio mensual
    avg_monthly = total / 12 if total > 0 else 0

    # KPI: Departamento con más casos
    if not dff.empty and dept == 'Todos':
        top_dept_series = dff['Departamento_residencia'].value_counts()
        top_dept = top_dept_series.index[0] if len(top_dept_series) > 0 else 'N/A'
    else:
        top_dept = dept if dept != 'Todos' else 'N/A'

    # KPI: Tendencia (comparación con año anterior)
    trend_text = 'N/A'
    trend_color = 'text-secondary'
    if year and year != 'Todos' and year > _cases['ANO'].min():
        prev_year_data = _cases[_cases['ANO'] == year - 1]
        if dept and dept != 'Todos':
            prev_year_data = prev_year_data[prev_year_data['Departamento_residencia'] == dept]
        prev_total = len(prev_year_data)
        if prev_total > 0:
            change = total - prev_total
            pct_change = (change / prev_total) * 100
            if change > 0:
                trend_text = f"+{change} ({pct_change:+.1f}%)"
                trend_color = 'text-danger'
            elif change < 0:
                trend_text = f"{change} ({pct_change:.1f}%)"
                trend_color = 'text-success'
            else:
                trend_text = "Sin cambio"
                trend_color = 'text-secondary'

    # Map - Crear figura según tipo de mapa seleccionado
    if map_style == 'choropleth':
        # Mapa coroplético usando shapefile - optimizado
        dept_counts = dff.groupby('Departamento_residencia', observed=True).size().reset_index(name='Casos')
        gdf_merged = gdf_map_geo.merge(dept_counts, left_on='DPTO_CNMBR', right_on='Departamento_residencia', how='left')
        gdf_merged['Casos'] = gdf_merged['Casos'].fillna(0).astype(int)
        
        fig_map = px.choropleth_map(
            gdf_merged,
            geojson=gdf_merged.geometry,
            locations=gdf_merged.index,
            color='Casos',
            color_continuous_scale='Reds',
            hover_name='DPTO_CNMBR',
            hover_data={'Casos': True},
            center={'lat': 4.5709, 'lon': -74.2973},
            zoom=4.5,
            height=500
        )
        fig_map.update_geos(fitbounds="locations", visible=False)
        fig_map.update_layout(margin={'r':0,'t':0,'l':0,'b':0}, uirevision='constant')
        
    elif map_style == 'scatter':
        # Puntos sobre shapefile - optimizado
        fig_map = px.choropleth_map(
            gdf_map_geo,
            geojson=gdf_map_geo.geometry,
            locations=gdf_map_geo.index,
            color_discrete_sequence=['#ffebee'],
            hover_name='DPTO_CNMBR',
            center={'lat': 4.5709, 'lon': -74.2973},
            zoom=4.5,
            height=500
        )
        
        # Agregar puntos solo si hay datos con coordenadas válidas
        if not dff.empty:
            dff_valid = dff.dropna(subset=['lon', 'lat'])
            if not dff_valid.empty:
                scatter_trace = px.scatter_map(
                    dff_valid,
                    lon='lon',
                    lat='lat',
                    hover_data={'Municipio_residencia': True, 'FEC_DEF': True, 'lon': False, 'lat': False},
                    color_discrete_sequence=['#c62828'],
                    size_max=10
                ).data[0]
                fig_map.add_trace(scatter_trace)
        
        fig_map.update_geos(fitbounds="locations", visible=False)
        fig_map.update_layout(margin={'r':0,'t':0,'l':0,'b':0}, uirevision='constant')
        
    else:  # density
        # Mapa de densidad - optimizado
        dept_counts = dff.groupby('Departamento_residencia', observed=True).size().reset_index(name='Casos')
        gdf_merged = gdf_map_geo.merge(dept_counts, left_on='DPTO_CNMBR', right_on='Departamento_residencia', how='left')
        gdf_merged['Casos'] = gdf_merged['Casos'].fillna(0).astype(int)
        
        fig_map = px.choropleth_map(
            gdf_merged,
            geojson=gdf_merged.geometry,
            locations=gdf_merged.index,
            color='Casos',
            color_continuous_scale=['#ffebee', '#ffcdd2', '#ef9a9a', '#e57373', '#ef5350', '#f44336', '#e53935', '#d32f2f', '#c62828', '#b71c1c'],
            hover_name='DPTO_CNMBR',
            hover_data={'Casos': True},
            center={'lat': 4.5709, 'lon': -74.2973},
            zoom=4.5,
            height=500
        )
        fig_map.update_geos(fitbounds="locations", visible=False)
        fig_map.update_layout(margin={'r':0,'t':0,'l':0,'b':0}, uirevision='constant')

    # Center map if dept selected
    if dept and dept != 'Todos':
        row = centroids[centroids['DPTO_CNMBR'] == dept]
        if not row.empty:
            fig_map.update_layout(
                map_center={'lon': float(row['lon'].iloc[0]), 'lat': float(row['lat'].iloc[0])}, 
                map_zoom=6
            )

    # Handle empty data
    if dff.empty or (map_style in ['scatter', 'density'] and 'lon' in dff.columns and dff['lon'].isnull().all()):
        if map_style != 'choropleth':
            fig_map.add_annotation(
                text='No hay datos disponibles para los filtros seleccionados', 
                xref='paper', yref='paper', x=0.5, y=0.5, 
                showarrow=False, font=dict(color='#c62828', size=14)
            )

    # Time Series Chart - Tendencia histórica (optimizado)
    if dept == 'Todos':
        time_data = _cases.groupby('ANO', observed=True).size().reset_index(name='Casos')
    else:
        time_data = _cases[_cases['Departamento_residencia'] == dept].groupby('ANO', observed=True).size().reset_index(name='Casos')
    
    fig_time = go.Figure()
    fig_time.add_trace(go.Scatter(
        x=time_data['ANO'],
        y=time_data['Casos'],
        mode='lines+markers',
        line=dict(color='#c62828', width=3),
        marker=dict(size=8, color='#e53935'),
        fill='tozeroy',
        fillcolor='rgba(198, 40, 40, 0.2)',
        hovertemplate='<b>%{x}</b><br>Casos: %{y}<extra></extra>'
    ))
    fig_time.update_layout(
        xaxis_title='Año',
        yaxis_title='Número de Casos',
        hovermode='x unified',
        margin={'r':10,'t':10,'l':10,'b':10},
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12),
        uirevision='constant'
    )
    fig_time.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#ecf0f1')
    fig_time.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#ecf0f1')

    # Top 10 Departamentos (optimizado)
    if dept == 'Todos':
        dept_counts = dff.groupby('Departamento_residencia', observed=True).size().reset_index(name='Casos')
        dept_counts = dept_counts.nlargest(10, 'Casos')
        
        fig_top_depts = px.bar(
            dept_counts,
            x='Casos',
            y='Departamento_residencia',
            orientation='h',
            color='Casos',
            color_continuous_scale='Reds',
            text='Casos'
        )
        fig_top_depts.update_traces(texttemplate='%{text}', textposition='outside')
        fig_top_depts.update_layout(
            yaxis={'categoryorder':'total ascending'},
            xaxis_title='Número de Casos',
            yaxis_title='',
            showlegend=False,
            margin={'r':10,'t':10,'l':10,'b':10},
            plot_bgcolor='white',
            paper_bgcolor='white',
            uirevision='constant'
        )
        fig_top_depts.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#ecf0f1')
    else:
        # Si hay un departamento seleccionado, mostrar top municipios
        muni_counts = dff.groupby('Municipio_residencia', observed=True).size().reset_index(name='Casos')
        muni_counts = muni_counts.nlargest(10, 'Casos')
        
        fig_top_depts = px.bar(
            muni_counts,
            x='Casos',
            y='Municipio_residencia',
            orientation='h',
            color='Casos',
            color_continuous_scale='Reds',
            text='Casos'
        )
        fig_top_depts.update_traces(texttemplate='%{text}', textposition='outside')
        fig_top_depts.update_layout(
            yaxis={'categoryorder':'total ascending'},
            xaxis_title='Número de Casos',
            yaxis_title='',
            title_text='Top 10 Municipios' if dept != 'Todos' else '',
            showlegend=False,
            margin={'r':10,'t':30,'l':10,'b':10},
            plot_bgcolor='white',
            paper_bgcolor='white',
            uirevision='constant'
        )
        fig_top_depts.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#ecf0f1')

    # Distribución Mensual (optimizado)
    if 'MES' in dff.columns:
        monthly_counts = dff.groupby('MES', observed=True).size().reset_index(name='Casos')
        meses_nombres = {1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'May', 6: 'Jun',
                        7: 'Jul', 8: 'Ago', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'}
        monthly_counts['Mes_Nombre'] = monthly_counts['MES'].map(meses_nombres)
        
        fig_monthly = px.bar(
            monthly_counts,
            x='Mes_Nombre',
            y='Casos',
            color='Casos',
            color_continuous_scale='Reds',
            text='Casos'
        )
        fig_monthly.update_traces(texttemplate='%{text}', textposition='outside')
        fig_monthly.update_layout(
            xaxis_title='Mes',
            yaxis_title='Número de Casos',
            showlegend=False,
            margin={'r':10,'t':10,'l':10,'b':10},
            plot_bgcolor='white',
            paper_bgcolor='white',
            uirevision='constant'
        )
        fig_monthly.update_xaxes(showgrid=False)
        fig_monthly.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#ecf0f1')
    else:
        fig_monthly = go.Figure()
        fig_monthly.update_layout(
            annotations=[dict(text='Datos mensuales no disponibles', x=0.5, y=0.5, 
                            showarrow=False, xref='paper', yref='paper')],
            uirevision='constant'
        )

    return (
        fig_map, 
        f"{total}", 
        f"{avg_monthly:.1f}",
        top_dept,
        html.Span(trend_text, className=trend_color),
        fig_time,
        fig_top_depts,
        fig_monthly
    )

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    app.run(host='0.0.0.0', port=port, debug=False)
