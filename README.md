# Hackathon
# Stories Coffee — Menu Profit Optimisation

> **Data Science consulting project for Stories, one of Lebanon's fastest-growing coffee chains (25 branches).**  
> We used machine learning and statistical analysis to identify which menu items are driving profit — and which are silently eroding it.

---

## The Business Problem

Stories' menu has grown organically across 25 branches with no systematic framework to evaluate product profitability at scale. Management could not answer three critical questions with confidence:

1. **Which products should we keep, monitor, or cut?**
2. **How do our branches compare in menu efficiency?**
3. **Where is our biggest margin opportunity?**

This project delivers a fully reproducible, data-driven answer to all three — and packages the result in a no-code Streamlit dashboard that any branch manager can use with next month's data export.

---

## Repository Structure

```
stories-profit-optimisation/
│
├── Hackathon.ipynb            # Full analysis notebook (cleaning → ML → recommendations)
├── stories_dashboard.py       # Streamlit dashboard — upload CSVs and get instant insights
│
├── data/                      # (not committed — see Data Format below)
│   ├── REP_S_00134_SMRY.csv   # Monthly branch sales
│   ├── rep_s_00014_SMRY.csv   # Product profitability
│   ├── rep_s_00191_SMRY-3.csv # Sales by items by group
│   └── rep_s_00673_SMRY.csv   # Profit by category
│
└── README.md
```

---

## Approach & Methodology

The project is structured in three stages:

### Stage 1 — Data Cleaning
Raw exports from the Stories POS system have no clean column headers, multi-row metadata blocks, page-break artifacts, and split table structures (Jan–Sep on one page, Oct–Dec on another). Custom Python parsers reconstruct four clean DataFrames:

| DataFrame | Contents |
|-----------|----------|
| `df_sales` | Monthly sales per branch for 2025–2026 |
| `df_product` | Price, cost, profit per product per branch |
| `df_sales_group` | Quantity and revenue by item group and division |
| `df_category` | Beverage vs. food profit by branch |

### Stage 2 — Exploratory Analysis
- **Monthly trend charts** — identify seasonality and branch-level growth trajectories
- **Branch heatmaps** — spot which branches underperform in which months
- **Category scatter charts** — compare beverage vs. food margin across all branches
- **Revenue-by-division breakdowns** — understand which product groups drive volume

### Stage 3 — Machine Learning

#### a. Pareto (80/20) Analysis
For each branch, products are ranked by descending profit and cumulative share is computed. The **% of the menu needed to generate 80% of profit** quantifies menu leanness.

- **Saida** is the leanest: only **12.6%** of its menu drives 80% of profit
- **Raouche** requires **23.1%** — nearly double
- No branch exceeds the 40% "danger zone," but all have meaningful room to trim

#### b. K-Means Product Clustering (k=4)
Every product across all branches is clustered using four features:
`Qty`, `Total Profit`, `Profit Margin %`, `Cost Ratio %`

The optimal k=4 is confirmed by an elbow plot. Clusters are auto-labeled by centroid profit value:

| Tier | Count | Avg Profit | Action |
|------|-------|-----------|--------|
|  Stars | 265 | ~862K | Protect and promote |
|  Workhorses | 32 | ~243K | Maintain volume |
|  Marginal | 12,000+ | Low | Review pricing/placement |
|  Loss-Makers | 2 | Negative | Remove immediately |

Every branch carries **85–90% Marginal products** — a massive pruning opportunity.

#### c. Menu Efficiency Score (Gini Coefficient)
The Gini coefficient of each branch's profit distribution is combined with the % of loss-making products into a composite **Bloat Score** (0–100, higher = more pruning needed):

```
Bloat Score = (1 - Gini) × 50 + Loss_Pct × 0.5
```

- **Amioun**: Bloat Score 52.4 — highest pruning urgency
- **Event Starco**: Bloat Score 35.5 — healthiest menu structure

#### d. Branch-Level Recommendations
Each product at each branch receives one of three action flags:

| Flag | Criteria | Meaning |
|------|----------|---------|
|  KEEP | Stars/Workhorses, positive profit | Core menu — protect |
|  MONITOR | Marginal tier | Review pricing or placement |
|  REMOVE | Loss-Makers or negative profit | Eliminate to stop margin drag |

**Results across all 25 branches:**
- **KEEP**: 277 products
- **MONITOR**: 11,046 products
- **REMOVE**: 1,199 products

---

##  How to Run

### Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit
```

### Run the Notebook

```bash
jupyter notebook Hackathon.ipynb
```

Place the four CSV files in the same directory as the notebook (or update the paths at the top of the notebook). All cleaning, analysis, and ML will run end-to-end.

### Run the Dashboard

```bash
streamlit run stories_dashboard.py
```

The dashboard will open in your browser. Use the file upload widgets to load your CSV exports — **no code changes required**. The app accepts the standard Stories POS export format and produces all charts and recommendation tables automatically.

---

##  Data Format

The dashboard accepts four CSV files in the standard Stories POS export format:

| File | Description |
|------|-------------|
| `REP_S_00134_SMRY.csv` | Monthly sales summary by branch |
| `rep_s_00014_SMRY.csv` | Product profitability (price, cost, profit per SKU) |
| `rep_s_00191_SMRY-3.csv` | Sales by items, grouped by division and category |
| `rep_s_00673_SMRY.csv` | Theoretical profit by category (beverages vs. food) |

**No manual transformation needed.** The parsers handle multi-row headers, page-break artifacts, and split table structures automatically.

---

##  Key Findings

1. **Menus are dramatically oversized.** Fewer than 3% of products per branch are high-profit Stars, yet 85–90% of every menu is Marginal — items with thin or near-zero contribution.

2. **1,199 products are actively losing money.** These should be removed immediately. Amioun has the highest bloat score and is the most urgent candidate.

3. **The 80/20 rule holds — but varies widely.** Saida needs only 12.6% of its menu to generate 80% of profit; Raouche needs 23.1%. Best practices from lean branches can be replicated chain-wide.

4. **Beverages consistently outperform food on margin.** Food items carry higher costs and lower margins — a pricing or supplier opportunity exists at food-heavy branches.

5. **Branch performance is highly heterogeneous.** High-revenue branches do not always have the best margins. Several mid-tier branches show above-average efficiency, suggesting transferable operating practices.

---

##  Tech Stack

| Tool | Purpose |
|------|---------|
| `pandas` / `numpy` | Data cleaning and transformation |
| `matplotlib` / `seaborn` | Static visualisations |
| `scikit-learn` | K-Means clustering, StandardScaler, cross-validation |
| `streamlit` | Interactive, no-code dashboard |

---

##  Reproducibility

The analysis is fully reproducible:
- **No hardcoded paths** — file locations are configurable at the top of the notebook
- **No manual data transformation** — all cleaning is done programmatically
- **Future-proof dashboard** — upload next month's exports and get updated results instantly
- **Documented parsers** — all data-cleaning logic is commented and modular

---
