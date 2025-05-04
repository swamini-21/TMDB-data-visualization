
# 📊 DataFlix: Uncovering Patterns in the Film Industry

**Authors**: Hritik Munde, Shamika Karnik, Swamini Sontakke  
**Course**: Data Visualization  
**Semester**: Spring 2025

---

## 📝 Project Overview

DataFlix is an interactive visualization dashboard designed to explore and analyze trends in the global film industry using data from The Movie Database (TMDB). Spanning over a century of cinema (1916–2017), the project extracts insights from movie metadata including genres, budgets, revenues, cast, and country of production. The visualizations aim to reveal patterns in film production, genre evolution, financial performance, and more.

Our primary goals:
- Track how genres rise and fall over time
- Understand relationships between budget and revenue
- Explore actor-genre networks
- Visualize global expansion of film production
- Provide a visually appealing, interactive dashboard using Plotly and Streamlit

---

## 🧱 Project Structure

```
.
├── Dashboardv1.ipynb               # Main dashboard and visualizations
├── DataCleaningV2.ipynb            # Data cleaning and CPI adjustment
├── clean_new.csv                   # Cleaned movie data
├── cpi.csv                         # CPI data used for inflation adjustment
├── data.zip/
│   ├── tmdb_5000_movies.csv        # Raw TMDB movies data
│   └── tmdb_5000_credits.csv       # Raw TMDB credits data
├── *.png                           # Visualizations for report
```

---

## 🧹 Data Cleaning and Preprocessing

Found in: `DataCleaningV2.ipynb`

1. **Merging Raw Datasets**:  
   `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` were merged on `movie_id`.

2. **Feature Extraction**:  
   - Extracted year from `release_date` → `release_year`
   - Computed `decade` for decade-wise trends
   - Parsed nested JSON columns like `genres`, `cast`, `crew`, and `production_countries`
   - Extracted top 10 cast and directors for each movie

3. **CPI Adjustment**:
   - Loaded CPI data from `cpi.csv`
   - Merged CPI values based on `release_year`
   - Computed `adj_factor = CPI_2023 / CPI_release_year`
   - Adjusted all monetary fields (`budget`, `revenue`) to 2023 USD

4. **ROI Calculation**:
```python
df['roi_adj'] = (df['revenue_adj'] - df['budget_adj']) / df['budget_adj']
```

5. **Filtering**:
   - Removed entries with zero runtime, budget, or revenue
   - Applied IQR method to remove extreme outliers in financial fields

---

## 📈 Visualizations

All visualizations are built using **Plotly** and implemented in `Dashboardv1.ipynb`.

### 1. 🌍 Animated Choropleth Map
**Purpose**: Show global movie production per country per decade  
📷 `choropleth-movies-by-country-by-decade.png`

---

### 2. 🧬 Runtime Violin Plot + Beeswarm
**Purpose**: Display runtime distribution by decade  
📷 `runtime-distribution-by-decade.png`

---

### 3. 🔤 Word Cloud by Genre
**Purpose**: Identify common keywords per genre  
📷 `wordcloud.png`

---

### 4. 🌳 Treemap of Genre Distribution
**Purpose**: Show dominant genres by decade  
📷 `treemap-of-movie-counts-by-decade.png`

---

### 5. 🌟 Popularity vs. Vote Average
**Purpose**: Compare audience popularity with average IMDb score  
📷 `popularity-v-vote-avg.png`

---

### 6. 💵 Budget vs. Revenue (Inflation Adjusted)
**Purpose**: Analyze correlation between investment and return  
📷 `inflation-adjusted-budget-revenue.png`

---

### 7. 👥 Genre-Actor Network Graph
**Purpose**: Show actors frequently appearing in a given genre  
📷 `network.png`

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/your-username/dataflix.git
cd dataflix
pip install -r requirements.txt
streamlit run Dashboardv1.ipynb
```

---

## ✅ Dependencies

- Python 3.8+
- pandas
- numpy
- plotly
- ast
- pyvis
- ipywidgets
- streamlit

---

## 📌 Credits

- TMDB Dataset: [Kaggle TMDB 5000 Movies Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- CPI Data: Bureau of Labor Statistics

---

## 📎 License

MIT License
