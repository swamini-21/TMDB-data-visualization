# ---------- Imports and Setup ----------
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import networkx as nx
from pyvis.network import Network
import tempfile, os, json, ast
import streamlit.components.v1 as components
from collections import Counter

st.set_page_config(layout="wide")

# Custom CSS to shrink sidebar width
st.markdown("""
    <style>
    /* Shrink sidebar */
    section[data-testid="stSidebar"] {
        width: 150px !important;
    }
    /* Shrink sidebar content container */
    div[data-testid="stSidebar"] > div:first-child {
        width: 200px !important;
    }
    /* Compact sidebar widgets */
    div[data-testid="stSidebar"] label, 
    div[data-testid="stSidebar"] .stSelectbox,
    div[data-testid="stSidebar"] .stSlider {
        margin-bottom: 0.3rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎬 Dataflix: Interactive Movie Trends Dashboard")

# ---------- Load Data ----------
def load_clean_data():
    df = pd.read_csv('clean.csv')
    df['decade'] = (df['year'] // 10) * 10
    return df

# Load and prepare data
df = load_clean_data()
df_expl = df.copy()
df_expl = df_expl.assign(genres=df['genres'].str.split(r' \| ')).explode('genres')
genre_counts = df_expl.genres.value_counts()
unique_genres = sorted(genre_counts.index)

# CPI
cpi = pd.read_csv('cpi.csv')
cpi.rename(columns={'Year': 'year', 'CPI': 'cpi'}, inplace=True)
base_cpi = cpi.loc[cpi['year'] == 2023, 'cpi'].iloc[0]
cpi['adj_factor'] = base_cpi / cpi['cpi']

infl = df[(df['budget'] > 0) & (df['revenue'] > 0)].copy()
infl = infl.merge(cpi[['year', 'adj_factor']], on='year', how='left')
infl['budget_adj'] = infl['budget'] * infl['adj_factor']
infl['revenue_adj'] = infl['revenue'] * infl['adj_factor']
infl['roi_adj'] = infl['revenue_adj'] / infl['budget_adj']
infl['decade'] = (infl['year'] // 10) * 10

# ---------- Sidebar Filters ----------
st.sidebar.header("Filters")
top_n = st.sidebar.selectbox("Top-N Genres", [2, 3, 4, 5])
rank_feature = st.sidebar.selectbox("Rank Genres by", ['count', 'popularity', 'vote_average', 'revenue'])
selected_genre = st.sidebar.selectbox("Genre", ["All"] + unique_genres)
selected_decade = st.sidebar.selectbox("Decade", ["All"] + sorted(df['decade'].dropna().unique()))
top_actors_n = st.sidebar.slider("Select Top-N Actors", min_value=5, max_value=50, value=30, step=5)

# ---------- Filter Helper ----------
def apply_filters(data):
    if selected_genre != "All":
        data = data[data['genres'].str.contains(selected_genre, na=False)]
    if selected_decade != "All":
        data = data[data['decade'] == int(selected_decade)]
    return data

df_filtered = apply_filters(df)
df_expl_filtered = apply_filters(df_expl)
infl_filtered = apply_filters(infl)

# ---------- Section 1: High-Level Trends ----------
st.markdown("## 📈 High-Level Trends")

df_counts = df_expl.groupby(['decade', 'genres']).size().reset_index(name='count')
df_expl_agg = df_expl.groupby(['decade', 'genres'], as_index=False).agg({
    'popularity': 'mean', 'vote_average': 'mean', 'revenue': 'sum'
})
df_counts_aggregated = df_counts.merge(df_expl_agg, on=['decade', 'genres'], how='left')
df_counts_aggregated['rank'] = df_counts_aggregated.groupby('decade')[rank_feature].rank(method='first', ascending=False)
df_top_n = df_counts_aggregated[df_counts_aggregated['rank'] <= top_n]

fig_bump = px.line(df_top_n, x='decade', y='rank', color='genres', markers=True, line_shape='spline', 
          hover_data=['count'], title=f'Top {top_n} Genres per Decade - by {rank_feature.capitalize()}')
fig_bump.update_traces(textposition='top center', marker=dict(size=6))
fig_bump.update_yaxes(autorange='reversed', dtick=1)
fig_bump.update_xaxes(dtick=10)

movies_per_decade = df_filtered['decade'].value_counts().sort_index().reset_index()
movies_per_decade.columns = ['Decade', 'Number of Movies']
fig_line = px.line(movies_per_decade, x='Decade',markers=True, y='Number of Movies', title="Movies Released per Decade")
fig_line.update_traces(line_color='green')

col1, col2 = st.columns(2)
col1.plotly_chart(fig_bump, use_container_width=True)
col2.plotly_chart(fig_line, use_container_width=True)

# ---------- Section 3: Runtime + Treemap ----------
st.markdown("## 🕒 Runtime & Genre Popularity")

fig_violin = px.violin(df_filtered[df_filtered['runtime'] > 0], x='decade', y='runtime',
                       box=True, points='all', labels={'decade': 'Decade', 'runtime': 'Runtime (min)'},
                       color_discrete_sequence=px.colors.qualitative.Safe, title='Runtime Distribution by Decade')
fig_violin.update_traces(
        selector={'type':'violin'},
        line_color='#1B4F72',
        line_width=0.25,
        points='all',        
        jitter=0.3,          
        marker=dict(symbol="circle",color="white", line=dict(width=0.5, color='#2E86C1'),opacity=1) 
    )

fig_tree = px.treemap(df_counts_aggregated, path=['decade', 'genres'], values='popularity',
                      color='popularity', color_continuous_scale='Sunset',
                      title='Popular Genre per Decade (Global View)')

# col1, col2 = st.columns(2)
st.plotly_chart(fig_violin, use_container_width=True)
st.plotly_chart(fig_tree, use_container_width=True)

#-------------Heatmap---------------

working_df = df.copy()
working_df['release_date'] = pd.to_datetime(working_df['release_date'], errors='coerce')
working_df = working_df.dropna(subset=['release_date'])  # remove rows with invalid dates
working_df['release_month'] = working_df['release_date'].dt.month
working_df['release_year'] = working_df['release_date'].dt.year

# Create pivot table
heatmap_data = working_df.pivot_table(
    index='release_month',
    columns='release_year',
    values='id',
    aggfunc='count',
    fill_value=0
)

# Create heatmap
fig = px.imshow(
    heatmap_data,
    labels=dict(x="Year", y="Month", color="# of Releases"),
    x=heatmap_data.columns,
    y=heatmap_data.index,
    color_continuous_scale='YlOrRd',
    aspect="auto"
)
fig.update_layout(
    title="Movie Releases by Month and Year",
    xaxis_title="Year",
    yaxis_title="Month",
    height=600
)

st.plotly_chart(fig)

# ---------- Section 2: Financials ----------
st.markdown("## 💰 Financial Insights")

fig_budget = px.scatter(
    infl_filtered, x='budget_adj', y='revenue_adj',
    hover_data=['title', 'year', 'genres'],
    log_x=True, log_y=True, opacity=0.5,
    title="Inflation-Adjusted Budget vs Revenue"
)
fig_budget.update_traces(marker=dict(size=5, symbol='circle',color='white',line=dict(width=0.75,color='Blue')))

df_scatter_size = infl_filtered.dropna(subset=['revenue_adj', 'popularity', 'vote_average'])
fig_scatter_size = px.scatter(
    df_scatter_size, x='popularity', y='vote_average',
    color='vote_count', size='revenue_adj',
    labels={
        'popularity': 'Popularity Score',
        'vote_average': 'Average Rating',
        'vote_count': 'Vote Count',
        'revenue_adj': 'Adjusted Revenue'
    },
    size_max=30, hover_data=['title', 'year'],color_continuous_scale='Sunset',
    title='Popularity vs Vote Average (Color=Vote Count, Size=Revenue)'
)

col1, col2 = st.columns(2)
col1.plotly_chart(fig_budget, use_container_width=True)
col2.plotly_chart(fig_scatter_size, use_container_width=True)

# ---------- Section 4: Network Graph ----------
st.markdown("## 👥 Genre-Actor Network Graph")

def parse_cast(x):
    if pd.isna(x): return []
    try: return [a.strip().strip("'").strip('"') for a in x.strip("[]").split("|") if a.strip()]
    except: return []

df['cast_names'] = df['cast'].apply(parse_cast)

if selected_genre == "All":
    sub = df.copy()
    edges = [(actor, genre) for _, row in sub.iterrows()
             for genre in row['genres'].split(" | ")
             for actor in row['cast_names']]
else:
    sub = df[df['genres'].str.contains(selected_genre, na=False)].copy()
    edges = [(actor, selected_genre) for cl in sub['cast_names'] for actor in cl]

actor_counts = Counter([a for cl in sub['cast_names'] for a in cl])
top_actors = {a for a, _ in actor_counts.most_common(top_actors_n)}
filtered_edges = [edge for edge in edges if edge[0] in top_actors]

G = nx.Graph()
G.add_edges_from(filtered_edges)

net = Network(height='700px', width='100%', bgcolor='#222222', font_color='white')
net.barnes_hut()

for node in G.nodes():
    is_genre = node in unique_genres
    size = 30 if is_genre else 15
    color = '#ff6f61' if is_genre else '#97C2FC'
    net.add_node(node, label=node, title=node, size=size, color=color, font={'size': 18, 'color': 'white'})

for u, v in G.edges(): net.add_edge(u, v, color='#cccccc')

net.set_options("""
{
  "physics": {
    "forceAtlas2Based": {
      "gravitationalConstant": -80,
      "springLength": 120,
      "springConstant": 0.03
    },
    "minVelocity": 0.75,
    "solver": "forceAtlas2Based"
  },
  "nodes": { "borderWidth": 1, "shadow": true },
  "edges": { "smooth": { "enabled": true } },
  "interaction": { "hover": true, "zoomView": true }
}
""")

with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html') as tmp_file:
    net.write_html(tmp_file.name)
    tmp_path = tmp_file.name

components.html(open(tmp_path, 'r', encoding='utf-8').read(), height=750)
os.unlink(tmp_path)

# ---------- Section 5: Word Cloud ----------
import streamlit as st
import pandas as pd
import json
import ast
import streamlit.components.v1 as components

st.markdown("## 🧠 Keyword Word Cloud by Genre")

st.subheader("🔤 Genre-Filtered Word Cloud (with Transitions)")

# Build keyword frequency mapping
df_expl = (df_expl.assign(keywords=df['keywords'].str.split(r' \| ')).explode('keywords')  )
mapping = {}
for genre, grp in df_expl.groupby('genres'):
    vc = grp['keywords'].value_counts().head(100)
    mapping[genre] = {kw: int(cnt) for kw, cnt in vc.items()}

data_json = json.dumps(mapping)
options_html = ''.join(f'<option value="{g}">{g}</option>' for g in mapping.keys())

# HTML template with animation transitions
html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Genre-Filtered Word Cloud</title>
  <script src="https://d3js.org/d3.v5.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/d3-cloud/1.2.5/d3.layout.cloud.min.js"></script>
  <style>
    body {{ font-family: sans-serif; margin: 0; text-align: center; }}
    #wcSVG {{ width: 900px; height: 600px; margin: auto; display: block; }}
    select {{ font-size: 16px; padding: 6px; margin-bottom: 15px; }}
  </style>
</head>
<body>
  <div>
    <label for="genreSelect"><strong>Select Genre:</strong></label>
    <select id="genreSelect">
      {options_html}
    </select>
  </div>
  <svg id="wcSVG" width="900" height="600"></svg>
  <script>
    const data = {data_json};
    const genres = Object.keys(data);
    const sel = d3.select('#genreSelect');
    sel.selectAll('option')
       .data(genres).enter()
       .append('option')
         .attr('value', d=>d)
         .text(d=>d);

    const svg = d3.select('#wcSVG');
    const width = +svg.attr('width');
    const height = +svg.attr('height');
    const group = svg.append('g')
                     .attr('transform', 'translate(' + width/2 + ',' + height/2 + ')');
    const color = d3.scaleOrdinal(d3.schemeDark2);

    const layout = d3.layout.cloud()
       .size([width, height])
       .padding(2)
       .rotate(() => Math.random() < 0.3 ? 90 : 0)
       .font('Impact')
       .fontSize(d => d.size)
       .on('end', draw);

    function draw(words) {{
      const t = group.selectAll('text').data(words, d => d.text);

      t.exit()
        .transition().duration(600)
        .style('opacity', 0)
        .remove();

      t.transition().duration(600)
        .attr('transform', d => 'translate(' + d.x + ',' + d.y + ')rotate(' + d.rotate + ')')
        .style('font-size', d => d.size + 'px');

      t.enter().append('text')
        .attr('text-anchor', 'middle')
        .attr('font-family', 'Impact')
        .style('fill', (d, i) => color(i))
        .attr('opacity', 0)
        .text(d => d.text)
        .transition().duration(600)
        .attr('opacity', 1)
        .attr('transform', d => 'translate(' + d.x + ',' + d.y + ')rotate(' + d.rotate + ')')
        .style('font-size', d => d.size + 'px');
    }}

    function update(genre) {{
      const freqs = data[genre];
      const entries = Object.entries(freqs);
      if (entries.length === 0) return draw([]);
      const counts = entries.map(([_, c]) => c);
      const sizeScale = d3.scaleSqrt()
                          .domain([0, d3.max(counts)])
                          .range([20, 80]);
      const words = entries.map(([t, c]) => {{ return {{ text: t, size: sizeScale(c) }}; }});
      layout.words(words).start();
    }}

    sel.on('change', () => update(sel.property('value')));
    update(genres[0]);
  </script>
</body>
</html>
"""

# Display word cloud in Streamlit
components.html(html_template, height=700)


# ---------- Section 6: Choropleth ----------
st.markdown("## 🌎 Global Movie Production")

df_expl_countries = df_filtered.assign(country=df_filtered['production_countries'].str.split(r' \| ')).explode('country')

choropleth_data = df_expl_countries
if selected_decade != "All":
    choropleth_data = choropleth_data[choropleth_data['decade'] == int(selected_decade)]

dfc_counts = (
    choropleth_data.groupby('country').size().reset_index(name='count')
)

fig_choropleth = px.choropleth(
    dfc_counts, locations='country', locationmode='country names',
    color='count', color_continuous_scale='Viridis',
    range_color=[0, dfc_counts['count'].quantile(0.95) if not dfc_counts.empty else 1],
    projection='natural earth', scope='world',
    title=f"Movies Produced per Country ({selected_decade if selected_decade != 'All' else 'All Decades'})"
)

fig_choropleth.update_geos(
    showland=True, landcolor='whitesmoke',
    showcountries=True, countrycolor='lightgray',
    showcoastlines=True, coastlinecolor='gray',
    showocean=True, oceancolor='lightblue',
    showlakes=True, lakecolor='lightblue',
    projection_scale=0.95,
)
fig_choropleth.update_layout(height=800, margin=dict(l=0, r=0, t=50, b=0), template='plotly_white')
st.plotly_chart(fig_choropleth, use_container_width=True)

