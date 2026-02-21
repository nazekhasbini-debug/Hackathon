"""
═══════════════════════════════════════════════════════════════════════════
 Stories Coffee — Profit & Menu Optimisation Dashboard
 Version 1.0 | Future-Proof Design (no hardcoded paths)
═══════════════════════════════════════════════════════════════════════════

HOW TO RUN (first time):
  1. Install dependencies:
       pip install streamlit pandas numpy plotly scikit-learn openpyxl

  2. Launch the app:
       streamlit run stories_dashboard.py

HOW TO USE EACH MONTH:
  - Upload your 4 CSV exports in the left sidebar (same file formats, any date).
  - The app auto-detects branches, months, and products — zero code changes needed.

REQUIRED FILES (upload in sidebar):
  ┌─────────────────────────────────────────────────────────────────────┐
  │ File 1 – Monthly Sales   : REP_S_00134_SMRY.csv                    │
  │ File 2 – Product Profit  : rep_s_00014_SMRY.csv                    │
  │ File 3 – Sales by Group  : rep_s_00191_SMRY-3.csv                  │
  │ File 4 – Category Profit : rep_s_00673_SMRY.csv                    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import io
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stories Coffee Dashboard",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — clean, professional look
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    h1 { color: #3b2314; }
    h2, h3 { color: #5c3d2e; }
    .metric-card {
        background: #fdf6f0;
        border-left: 4px solid #c8824a;
        padding: 0.9rem 1.2rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #fdf6f0;
        border-radius: 6px 6px 0 0;
        padding: 8px 18px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] { background-color: #c8824a; color: white; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# HELPER UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def to_num(val) -> float:
    """Convert a messy string value (e.g. '1,234,567.89') to float."""
    try:
        return float(str(val).replace(",", "").strip())
    except Exception:
        return np.nan


def fmt_m(val: float) -> str:
    if pd.isna(val):
        return "N/A"
    return f"{val / 1_000_000:.1f}M"


# ──────────────────────────────────────────────────────────────────────────────
# DATA CLEANING — one function per CSV export
# All functions accept a file-like object so they work with Streamlit uploaders.
# ──────────────────────────────────────────────────────────────────────────────

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
JAN_SEP = MONTHS[:9]
OCT_DEC = MONTHS[9:] + ["Total"]


@st.cache_data(show_spinner=False)
def clean_monthly_sales(raw_bytes: bytes) -> pd.DataFrame:
    """
    Cleans REP_S_00134_SMRY.csv — Comparative Monthly Sales.

    Raw format quirks handled:
    • Rows 0-3 are metadata / title rows with no column headers.
    • Month columns span Jan-Sep on one page, Oct-Dec+Total on another.
    • Page-break rows repeat the month header mid-file.
    • Year value appears only on the first row for each year block (forward-filled).
    • Commas in numbers (e.g. '3,355,705.33') are stripped.

    Returns: DataFrame[Branch, Year, January…December, Total]
    """
    raw = pd.read_csv(io.BytesIO(raw_bytes), header=None, dtype=str)

    records_jan_sep, records_oct_dec = [], []
    current_section = None   # 'jan_sep' | 'oct_dec'
    jan_start_col   = 3
    current_year    = None

    for _, row in raw.iterrows():
        r = [str(v).strip() if pd.notna(v) else "" for v in row]

        # Detect section header rows
        if r[3] == "January" and r[4] == "February":
            current_section = "jan_sep"
            jan_start_col   = 3
            continue
        if r[2] == "January" and r[3] == "February":
            current_section = "jan_sep"
            jan_start_col   = 2
            continue
        if r[2] == "October" and r[3] == "November":
            current_section = "oct_dec"
            continue
        if current_section is None:
            continue

        # Forward-fill year
        if r[0] not in ("", "nan"):
            try:
                current_year = int(float(r[0]))
            except ValueError:
                pass

        branch = r[1] if r[1] not in ("", "nan") else None
        if not branch or "Total" in branch:
            continue

        if current_section == "jan_sep":
            rec = {"Year": current_year, "Branch": branch}
            for i, m in enumerate(JAN_SEP):
                rec[m] = to_num(row.iloc[jan_start_col + i])
            records_jan_sep.append(rec)

        elif current_section == "oct_dec":
            rec = {"Year": current_year, "Branch": branch}
            for i, m in enumerate(OCT_DEC):
                rec[m] = to_num(row.iloc[2 + i])
            records_oct_dec.append(rec)

    df_jan = pd.DataFrame(records_jan_sep)
    df_oct = pd.DataFrame(records_oct_dec)

    df = df_jan.merge(df_oct, on=["Branch", "Year"], how="outer")
    month_cols = MONTHS + ["Total"]
    df = df[["Branch", "Year"] + [c for c in month_cols if c in df.columns]]
    df = df.sort_values(["Year", "Branch"]).reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def clean_product_profitability(raw_bytes: bytes) -> pd.DataFrame:
    """
    Cleans rep_s_00014_SMRY.csv — Theoretical Profit by Product.

    Raw format quirks handled:
    • Branch headers are standalone rows (col0 starts with 'Stories', rest NaN).
    • Title / date / subtotal rows are skipped via SKIP_PREFIXES.
    • Columns: Product | Qty | Total Price | (blank) | Total Cost | Cost% | Total Profit | (blank) | Profit%

    Returns: DataFrame[Branch, Product, Qty, Total_Price, Total_Cost,
                        Total_Cost_Pct, Total_Profit, Total_Profit_Pct]
    """
    raw = pd.read_csv(io.BytesIO(raw_bytes), header=None, dtype=str)

    SKIP_PREFIXES = (
        "Theoretical", "Stories,",
        "22-Jan", "22-jan",
        "Product Desc",
        "Total By",
    )

    records = []
    current_branch = None

    for _, row in raw.iterrows():
        r = [
            str(v).strip() if pd.notna(v) and str(v).strip() not in ("nan", "") else ""
            for v in row
        ]
        col0 = r[0]

        if col0.startswith("Stories") and all(v == "" for v in r[1:]):
            current_branch = col0
            continue

        if any(col0.startswith(p) for p in SKIP_PREFIXES):
            continue
        if current_branch is None:
            continue

        qty_raw = r[1] if len(r) > 1 else ""
        if qty_raw == "":
            continue  # section label row

        def n(v):
            try:
                return float(v.replace(",", ""))
            except Exception:
                return np.nan

        records.append({
            "Branch":           current_branch,
            "Product":          col0,
            "Qty":              n(r[1]),
            "Total_Price":      n(r[2]),
            "Total_Cost":       n(r[4]) if len(r) > 4 else np.nan,
            "Total_Cost_Pct":   n(r[5]) if len(r) > 5 else np.nan,
            "Total_Profit":     n(r[6]) if len(r) > 6 else np.nan,
            "Total_Profit_Pct": n(r[8]) if len(r) > 8 else np.nan,
        })

    num_cols = ["Qty", "Total_Price", "Total_Cost", "Total_Profit"]
    df = pd.DataFrame(records).dropna(subset=num_cols, how="all").reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def clean_sales_group(raw_bytes: bytes) -> pd.DataFrame:
    """
    Cleans rep_s_00191_SMRY-3.csv — Sales by Items by Group.

    Raw format quirks handled:
    • Branch / Division / Group headers are labeled rows (prefix 'Branch:', etc.).
    • Page-break / metadata rows are skipped.
    • Data rows identified by numeric Qty in col2.

    Returns: DataFrame[Branch, Division, Group, Item, Qty, Total_Amount]
    """
    raw = pd.read_csv(io.BytesIO(raw_bytes), header=None, dtype=str)

    records = []
    current_branch = current_division = current_group = None

    def n2(v):
        try:
            return float(str(v).replace(",", "").strip())
        except Exception:
            return np.nan

    def cl(v):
        return str(v).strip() if pd.notna(v) and str(v).strip() not in ("nan", "") else ""

    for _, row in raw.iterrows():
        r = [cl(v) for v in row]
        col0 = r[0]

        if col0 in ("", "Stories", "Sales by Items By Group", "Description"):
            continue
        if col0[:2].isdigit() and "-Jan-" in col0:
            continue
        if col0.startswith("Total by"):
            continue

        if col0.startswith("Branch:"):
            current_branch   = col0.replace("Branch:", "").strip()
            current_division = None
            current_group    = None
            continue
        if col0.startswith("Division:"):
            current_division = col0.replace("Division:", "").strip()
            current_group    = None
            continue
        if col0.startswith("Group:"):
            current_group = col0.replace("Group:", "").strip()
            continue
        if current_branch is None:
            continue

        qty_val = n2(r[2]) if len(r) > 2 else np.nan
        if np.isnan(qty_val):
            continue

        records.append({
            "Branch":       current_branch,
            "Division":     current_division,
            "Group":        current_group,
            "Item":         col0,
            "Qty":          qty_val,
            "Total_Amount": n2(r[3]) if len(r) > 3 else np.nan,
        })

    df = (
        pd.DataFrame(records)
        .dropna(subset=["Qty", "Total_Amount"], how="all")
        .reset_index(drop=True)
    )
    return df


@st.cache_data(show_spinner=False)
def clean_category(raw_bytes: bytes) -> pd.DataFrame:
    """
    Cleans rep_s_00673_SMRY.csv — Theoretical Profit by Category.

    Raw format quirks handled:
    • Same branch-header pattern as Product Profitability file.
    • Columns: Category | Qty | Total Price | (blank) | Total Cost | Cost% | Total Profit | (blank) | Profit%

    Returns: DataFrame[Branch, Category, Qty, Total_Price, Total_Cost,
                        Total_Cost_Pct, Total_Profit, Total_Profit_Pct]
    """
    raw = pd.read_csv(io.BytesIO(raw_bytes), header=None, dtype=str)

    records = []
    current_branch = None

    def nc(v):
        try:
            return float(str(v).replace(",", "").strip())
        except Exception:
            return np.nan

    def clc(v):
        return str(v).strip() if pd.notna(v) and str(v).strip() not in ("nan", "") else ""

    for _, row in raw.iterrows():
        r = [clc(v) for v in row]
        col0 = r[0]

        if col0 == "":
            continue
        if col0 in ("Stories", "Theoretical Profit By Category", "Category"):
            continue
        if col0[:2].isdigit() and "-Jan-" in col0:
            continue
        if col0.startswith("Total By"):
            continue

        if col0.startswith("Stories") and all(v == "" for v in r[1:]):
            current_branch = col0
            continue
        if current_branch is None:
            continue

        qty_val = nc(r[1]) if len(r) > 1 else np.nan
        if np.isnan(qty_val):
            continue

        records.append({
            "Branch":           current_branch,
            "Category":         col0,
            "Qty":              qty_val,
            "Total_Price":      nc(r[2]) if len(r) > 2 else np.nan,
            "Total_Cost":       nc(r[4]) if len(r) > 4 else np.nan,
            "Total_Cost_Pct":   nc(r[5]) if len(r) > 5 else np.nan,
            "Total_Profit":     nc(r[6]) if len(r) > 6 else np.nan,
            "Total_Profit_Pct": nc(r[8]) if len(r) > 8 else np.nan,
        })

    df = (
        pd.DataFrame(records)
        .dropna(subset=["Qty", "Total_Cost"], how="all")
        .reset_index(drop=True)
    )
    return df


# ──────────────────────────────────────────────────────────────────────────────
# ANALYSIS HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def gini_coefficient(arr: np.ndarray) -> float:
    """Gini coefficient of profit distribution (handles negatives)."""
    arr = np.array(arr, dtype=float)
    arr = arr - arr.min() + 1e-9
    arr = np.sort(arr)
    n   = len(arr)
    idx = np.arange(1, n + 1)
    return float((2 * (idx * arr).sum()) / (n * arr.sum()) - (n + 1) / n)


@st.cache_data(show_spinner=False)
def compute_pareto(_df_product: pd.DataFrame) -> pd.DataFrame:
    records = []
    for branch, grp in _df_product.groupby("Branch"):
        pos = grp[grp["Total_Profit"] > 0].sort_values("Total_Profit", ascending=False).reset_index(drop=True)
        if pos.empty:
            continue
        pos["CumPct"] = pos["Total_Profit"].cumsum() / pos["Total_Profit"].sum() * 100
        n_80    = int((pos["CumPct"] >= 80).idxmax()) + 1
        n_total = len(grp)
        records.append({
            "Branch":               branch,
            "Total_Products":       n_total,
            "Profitable":           len(pos),
            "Loss_Making":          int((grp["Total_Profit"] < 0).sum()),
            "Zero_Profit":          int((grp["Total_Profit"] == 0).sum()),
            "Products_for_80pct":   n_80,
            "Menu_Pct_for_80pct":   round(n_80 / n_total * 100, 1),
        })
    return pd.DataFrame(records).sort_values("Menu_Pct_for_80pct").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def compute_kmeans(_df_product: pd.DataFrame):
    """Run K-Means (k=4) on product features; return labeled DataFrame + efficiency scores."""
    FEATURES = ["Qty", "Total_Profit", "Total_Profit_Pct", "Total_Cost_Pct"]
    df_km = _df_product[["Branch", "Product"] + FEATURES].copy()
    df_km[FEATURES] = df_km[FEATURES].fillna(0)

    scaler = StandardScaler()
    X      = scaler.fit_transform(df_km[FEATURES])

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_km["Cluster"] = kmeans.fit_predict(X)

    centers = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=FEATURES,
    )
    sorted_idx   = centers.sort_values("Total_Profit", ascending=False).index.tolist()
    label_list   = ["Stars", "Workhorses", "Marginal", "Loss-Makers"]
    cluster_map  = {cid: lbl for cid, lbl in zip(sorted_idx, label_list)}
    df_km["Tier"] = df_km["Cluster"].map(cluster_map)

    # Efficiency / Bloat Score per branch
    eff_records = []
    for branch, grp in _df_product.groupby("Branch"):
        profit_vals = grp["Total_Profit"].fillna(0).values
        g           = round(gini_coefficient(profit_vals), 3)
        loss_pct    = round(float((profit_vals < 0).mean() * 100), 1)
        bloat       = round((1 - g) * 50 + loss_pct * 0.5, 1)
        eff_records.append({
            "Branch":      branch,
            "N_Products":  len(profit_vals),
            "Gini":        g,
            "Loss_Pct":    loss_pct,
            "Bloat_Score": bloat,
        })
    df_eff = pd.DataFrame(eff_records).sort_values("Bloat_Score", ascending=False).reset_index(drop=True)

    return df_km, df_eff


@st.cache_data(show_spinner=False)
def compute_recommendations(_df_km: pd.DataFrame, _df_product: pd.DataFrame):
    def action_flag(tier, profit):
        if profit < 0:
            return "REMOVE"
        return "KEEP" if tier in ("Stars", "Workhorses") else "MONITOR"

    df = _df_km.copy()
    df["Action"] = df.apply(lambda r: action_flag(r["Tier"], r["Total_Profit"]), axis=1)
    extra = _df_product[["Branch", "Product", "Total_Price"]].drop_duplicates(subset=["Branch", "Product"])
    df = df.merge(extra, on=["Branch", "Product"], how="left")
    df = df.rename(columns={
        "Action":          "Recommendation",
        "Total_Price":     "Revenue",
        "Total_Profit":    "Profit",
        "Total_Profit_Pct":"Margin_%",
    })
    return df[["Branch", "Product", "Qty", "Revenue", "Profit", "Margin_%", "Tier", "Recommendation"]]


def short_name(branch: str) -> str:
    return branch.replace("Stories ", "").replace("Stories", "").strip()


TIER_COLORS = {
    "Stars":       "#1a9850",
    "Workhorses":  "#91cf60",
    "Marginal":    "#fee08b",
    "Loss-Makers": "#d73027",
}
ACTION_COLORS = {"KEEP": "#1a9850", "MONITOR": "#fee08b", "REMOVE": "#d73027"}


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR — file uploads + instructions
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://via.placeholder.com/220x60/3b2314/ffffff?text=☕+Stories+Coffee", use_column_width=True)
    st.title("Data Upload")

    st.markdown(
        """
        Upload your **4 CSV exports** below.  
        The app works with any export date — no code changes needed.
        """
    )

    st.markdown("---")
    st.markdown("**① Monthly Sales** `REP_S_00134_SMRY.csv`")
    f_sales = st.file_uploader("Monthly Sales CSV", type=["csv"], label_visibility="collapsed", key="f_sales")

    st.markdown("**② Product Profitability** `rep_s_00014_SMRY.csv`")
    f_product = st.file_uploader("Product Profitability CSV", type=["csv"], label_visibility="collapsed", key="f_product")

    st.markdown("**③ Sales by Group** `rep_s_00191_SMRY-3.csv`")
    f_group = st.file_uploader("Sales by Group CSV", type=["csv"], label_visibility="collapsed", key="f_group")

    st.markdown("**④ Category Profit** `rep_s_00673_SMRY.csv`")
    f_category = st.file_uploader("Category Profit CSV", type=["csv"], label_visibility="collapsed", key="f_category")

    st.markdown("---")
    st.caption(
        "💡 **Tip:** Files are processed in memory — nothing is saved to disk. "
        "Re-upload next month's exports to refresh all charts instantly."
    )


# ──────────────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────────────

st.title("☕ Stories Coffee — Profit & Menu Dashboard")
st.caption("Upload your CSV exports in the sidebar to get started.")

files_ready = {
    "sales":    f_sales    is not None,
    "product":  f_product  is not None,
    "group":    f_group    is not None,
    "category": f_category is not None,
}

if not any(files_ready.values()):
    st.info(
        "👈 **Upload your CSV files in the sidebar** to load the dashboard.\n\n"
        "Each section becomes available as soon as its required file is uploaded — "
        "you don't need all 4 to start exploring."
    )
    st.stop()


# ──────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────────────────────────

df_sales = df_product = df_group = df_category = None

if files_ready["sales"]:
    with st.spinner("Cleaning Monthly Sales data…"):
        df_sales = clean_monthly_sales(f_sales.read())

if files_ready["product"]:
    with st.spinner("Cleaning Product Profitability data…"):
        df_product = clean_product_profitability(f_product.read())

if files_ready["group"]:
    with st.spinner("Cleaning Sales by Group data…"):
        df_group = clean_sales_group(f_group.read())

if files_ready["category"]:
    with st.spinner("Cleaning Category Profit data…"):
        df_category = clean_category(f_category.read())


# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL KPI ROW (requires sales + product)
# ──────────────────────────────────────────────────────────────────────────────

# Year selector works as soon as sales file is uploaded
selected_year = None
if df_sales is not None:
    years_available = sorted(df_sales["Year"].dropna().unique().tolist(), reverse=True)
    selected_year   = st.selectbox("📅 Select Year for Sales Charts", years_available, index=0)

if df_sales is not None and df_product is not None and selected_year is not None:
    sales_year = df_sales[df_sales["Year"] == selected_year]
    total_sales    = sales_year["Total"].sum() if "Total" in sales_year.columns else sales_year[MONTHS].sum().sum()
    total_profit   = df_product["Total_Profit"].sum()
    total_branches = df_product["Branch"].nunique()
    total_products = df_product["Product"].nunique()
    loss_products  = int((df_product["Total_Profit"] < 0).sum())

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Sales", f"{total_sales/1e6:.1f}M", help=f"Sum across all branches for {selected_year}")
    k2.metric("Total Profit", f"{total_profit/1e6:.1f}M", help="Sum across all products & branches")
    k3.metric("Branches", total_branches)
    k4.metric("Unique Products", total_products)
    k5.metric("Loss-Making SKUs", loss_products, delta=f"-{loss_products} to review", delta_color="inverse")

    st.markdown("---")


# ──────────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────────

tab_labels = ["📊 Sales Overview", "💰 Product Profitability", "🏪 Sales by Group", "📁 Category", "🔬 Menu Optimisation"]
tabs = st.tabs(tab_labels)


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — SALES OVERVIEW
# ════════════════════════════════════════════════════════════════════════════

with tabs[0]:
    if df_sales is None:
        st.warning("⬅ Upload the **Monthly Sales CSV** to see this section.")
        st.stop()

    st.subheader(f"Sales Overview — {selected_year}")

    sales_y = df_sales[df_sales["Year"] == selected_year].copy()
    avail_months = [m for m in MONTHS if m in sales_y.columns and sales_y[m].sum() > 0]

    # ── Branch bar chart ──
    branch_totals = sales_y[["Branch", "Total"]].sort_values("Total")
    branch_totals["Short"] = branch_totals["Branch"].apply(short_name)

    fig_bar = px.bar(
        branch_totals,
        x="Total", y="Short",
        orientation="h",
        labels={"Total": "Total Sales", "Short": ""},
        title=f"Total Annual Sales by Branch — {selected_year}",
        color="Total",
        color_continuous_scale="Blues",
        text=branch_totals["Total"].apply(fmt_m),
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(coloraxis_showscale=False, height=520, xaxis_tickformat=".0s")
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Monthly trend ──
    if avail_months:
        monthly_totals = sales_y[avail_months].sum()
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=avail_months, y=monthly_totals.values,
            mode="lines+markers",
            line=dict(color="#c8824a", width=3),
            marker=dict(size=8),
            fill="tozeroy", fillcolor="rgba(200,130,74,0.1)",
            name="Total Sales",
        ))
        fig_line.update_layout(
            title=f"Monthly Sales Trend — All Branches ({selected_year})",
            xaxis_title="Month",
            yaxis_title="Sales",
            yaxis_tickformat=".2s",
            height=360,
        )
        st.plotly_chart(fig_line, use_container_width=True)

        # ── Heatmap ──
        heat_data = sales_y.set_index("Branch")[avail_months].copy()
        heat_data.index = heat_data.index.map(short_name)
        heat_vals = heat_data / 1e6

        fig_heat = px.imshow(
            heat_vals,
            labels=dict(color="Sales (M)"),
            color_continuous_scale="YlOrRd",
            title=f"Monthly Sales Heatmap — {selected_year} (M)",
            text_auto=".0f",
            aspect="auto",
        )
        fig_heat.update_layout(height=600)
        st.plotly_chart(fig_heat, use_container_width=True)

    # ── Year comparison (if multiple years) ──
    if len(df_sales["Year"].unique()) > 1 and "Total" in df_sales.columns:
        st.subheader("Year-over-Year Comparison")
        yoy = (
            df_sales.groupby("Year")["Total"].sum().reset_index()
            .sort_values("Year")
        )
        yoy["Year"] = yoy["Year"].astype(str)
        fig_yoy = px.bar(
            yoy, x="Year", y="Total",
            title="Total Network Sales by Year",
            labels={"Total": "Total Sales"},
            color="Year",
            color_discrete_sequence=px.colors.qualitative.Set2,
            text=yoy["Total"].apply(fmt_m),
        )
        fig_yoy.update_traces(textposition="outside")
        fig_yoy.update_layout(yaxis_tickformat=".2s", height=360, showlegend=False)
        st.plotly_chart(fig_yoy, use_container_width=True)

    # ── Raw data expander ──
    with st.expander("🔍 View cleaned Monthly Sales table"):
        st.dataframe(sales_y, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — PRODUCT PROFITABILITY
# ════════════════════════════════════════════════════════════════════════════

with tabs[1]:
    if df_product is None:
        st.warning("⬅ Upload the **Product Profitability CSV** to see this section.")
        st.stop()

    st.subheader("Product Profitability")

    # ── Branch filter ──
    branches = sorted(df_product["Branch"].unique().tolist())
    sel_branches = st.multiselect(
        "Filter by Branch (leave empty for all)",
        options=branches,
        default=[],
        key="prod_branch_filter",
    )
    df_p = df_product[df_product["Branch"].isin(sel_branches)] if sel_branches else df_product

    col1, col2 = st.columns(2)

    # Top 15 products by profit
    with col1:
        top15 = df_p.groupby("Product")["Total_Profit"].sum().sort_values(ascending=False).head(15).reset_index()
        top15["Short"] = top15["Product"].str[:35]
        fig_top = px.bar(
            top15[::-1].reset_index(drop=True),
            x="Total_Profit", y="Short",
            orientation="h",
            title="Top 15 Products by Total Profit",
            labels={"Total_Profit": "Total Profit", "Short": ""},
            color="Total_Profit",
            color_continuous_scale="Greens",
            text=top15["Total_Profit"][::-1].reset_index(drop=True).apply(fmt_m),
        )
        fig_top.update_layout(coloraxis_showscale=False, height=480, xaxis_tickformat=".2s")
        fig_top.update_traces(textposition="outside")
        st.plotly_chart(fig_top, use_container_width=True)

    # Top 15 loss-making products
    with col2:
        loss15 = df_p[df_p["Total_Profit"] < 0].groupby("Product")["Total_Profit"].sum().sort_values().head(15).reset_index()
        loss15["Short"] = loss15["Product"].str[:35]
        fig_loss = px.bar(
            loss15[::-1].reset_index(drop=True),
            x="Total_Profit", y="Short",
            orientation="h",
            title="Top 15 Loss-Making Products",
            labels={"Total_Profit": "Total Profit", "Short": ""},
            color="Total_Profit",
            color_continuous_scale="Reds_r",
            text=loss15["Total_Profit"][::-1].reset_index(drop=True).apply(fmt_m),
        )
        fig_loss.update_layout(coloraxis_showscale=False, height=480, xaxis_tickformat=".2s")
        fig_loss.update_traces(textposition="outside")
        st.plotly_chart(fig_loss, use_container_width=True)

    # Loss-making products per branch
    neg_per_branch = (
        df_p[df_p["Total_Profit"] < 0]
        .groupby("Branch")["Product"]
        .count()
        .sort_values()
        .reset_index()
    )
    neg_per_branch["Short"] = neg_per_branch["Branch"].apply(short_name)
    fig_neg = px.bar(
        neg_per_branch,
        x="Product", y="Short",
        orientation="h",
        title="Number of Loss-Making Products per Branch",
        labels={"Product": "# Loss-Making Products", "Short": ""},
        color="Product",
        color_continuous_scale="Reds",
        text="Product",
    )
    fig_neg.update_layout(coloraxis_showscale=False, height=520)
    fig_neg.update_traces(textposition="outside")
    st.plotly_chart(fig_neg, use_container_width=True)

    # Cost vs Profit scatter
    branch_pnl = df_p.groupby("Branch")[["Total_Cost", "Total_Profit"]].sum().reset_index()
    branch_pnl["Short"] = branch_pnl["Branch"].apply(short_name)
    fig_scatter = px.scatter(
        branch_pnl,
        x="Total_Cost", y="Total_Profit",
        text="Short",
        title="Cost vs Profit by Branch (product roll-up)",
        labels={
            "Total_Cost":   "Total Cost",
            "Total_Profit": "Total Profit",
        },
        color="Total_Profit",
        color_continuous_scale="RdYlGn",
        size_max=18,
    )
    fig_scatter.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
    fig_scatter.update_traces(textposition="top center")
    fig_scatter.update_layout(height=480, xaxis_tickformat=".2s", yaxis_tickformat=".2s")
    st.plotly_chart(fig_scatter, use_container_width=True)

    with st.expander("🔍 View Product Profitability table"):
        cols = ["Branch", "Product", "Qty", "Total_Price", "Total_Cost", "Total_Profit", "Total_Profit_Pct"]
        st.dataframe(df_p[[c for c in cols if c in df_p.columns]], use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — SALES BY GROUP
# ════════════════════════════════════════════════════════════════════════════

with tabs[2]:
    if df_group is None:
        st.warning("⬅ Upload the **Sales by Group CSV** to see this section.")
        st.stop()

    st.subheader("Sales by Group & Division")

    col1, col2 = st.columns([3, 2])

    with col1:
        top_groups = (
            df_group.groupby("Group")["Total_Amount"]
            .sum().sort_values(ascending=False).head(15).reset_index()
        )
        top_groups["Short"] = top_groups["Group"].str[:30]
        fig_grp = px.bar(
            top_groups[::-1].reset_index(drop=True),
            x="Total_Amount", y="Short",
            orientation="h",
            title="Top 15 Groups by Total Revenue",
            labels={"Total_Amount": "Revenue", "Short": ""},
            color="Total_Amount",
            color_continuous_scale="Purples",
            text=top_groups["Total_Amount"][::-1].reset_index(drop=True).apply(fmt_m),
        )
        fig_grp.update_layout(coloraxis_showscale=False, height=500, xaxis_tickformat=".2s")
        fig_grp.update_traces(textposition="outside")
        st.plotly_chart(fig_grp, use_container_width=True)

    with col2:
        div_rev = df_group.groupby("Division")["Total_Amount"].sum().sort_values(ascending=False).reset_index()
        top8    = div_rev.head(8)
        other   = div_rev.iloc[8:]["Total_Amount"].sum()
        if other > 0:
            top8 = pd.concat([top8, pd.DataFrame([{"Division": "Other", "Total_Amount": other}])], ignore_index=True)

        fig_pie = px.pie(
            top8,
            values="Total_Amount",
            names="Division",
            title="Revenue Share by Division",
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.35,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    # Stacked bar: top 6 divisions per branch
    div_revenue = df_group.groupby("Division")["Total_Amount"].sum().sort_values(ascending=False)
    top6_div    = div_revenue.head(6).index.tolist()
    pivot_div   = (
        df_group[df_group["Division"].isin(top6_div)]
        .groupby(["Branch", "Division"])["Total_Amount"].sum()
        .unstack(fill_value=0)
    )
    pivot_div.index = pivot_div.index.map(short_name)
    fig_stk = px.bar(
        pivot_div.reset_index().melt(id_vars="Branch", value_name="Revenue", var_name="Division"),
        x="Branch", y="Revenue",
        color="Division",
        barmode="stack",
        title="Revenue by Division per Branch (Top 6 Divisions)",
        labels={"Revenue": "Revenue", "Branch": ""},
        color_discrete_sequence=px.colors.qualitative.Plotly,
    )
    fig_stk.update_layout(height=500, xaxis_tickangle=-45, yaxis_tickformat=".2s")
    st.plotly_chart(fig_stk, use_container_width=True)

    with st.expander("🔍 View Sales by Group table"):
        st.dataframe(df_group, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — CATEGORY ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

with tabs[3]:
    if df_category is None:
        st.warning("⬅ Upload the **Category Profit CSV** to see this section.")
        st.stop()

    st.subheader("Beverages vs Food — Category Analysis")

    categories = df_category["Category"].unique().tolist()
    if len(categories) < 2:
        st.info("Only one category found in this file.")
    else:
        pivot_cat = (
            df_category.pivot_table(
                index="Branch", columns="Category",
                values="Total_Profit", aggfunc="sum",
            )
            .reset_index()
        )
        pivot_cat["Short"] = pivot_cat["Branch"].apply(short_name)
        pivot_cat = pivot_cat.sort_values(categories[0], ascending=False)

        melted = pivot_cat.melt(
            id_vars=["Short"],
            value_vars=categories,
            var_name="Category",
            value_name="Total_Profit",
        )
        fig_cat = px.bar(
            melted,
            x="Short", y="Total_Profit",
            color="Category",
            barmode="group",
            title="Theoretical Profit: Beverages vs Food by Branch",
            labels={"Total_Profit": "Profit", "Short": ""},
            color_discrete_map={"BEVERAGES": "#4a90d9", "FOOD": "#e87040"},
        )
        fig_cat.update_layout(height=480, xaxis_tickangle=-45, yaxis_tickformat=".2s")
        st.plotly_chart(fig_cat, use_container_width=True)

        # Margin scatter per branch & category
        margin_df = df_category.copy()
        margin_df["Short"] = margin_df["Branch"].apply(short_name)
        avg_margin = df_category["Total_Profit_Pct"].mean()

        fig_mg = px.scatter(
            margin_df,
            x="Short", y="Total_Profit_Pct",
            color="Category",
            title="Profit Margin % by Branch and Category",
            labels={"Total_Profit_Pct": "Profit Margin %", "Short": ""},
            color_discrete_map={"BEVERAGES": "#4a90d9", "FOOD": "#e87040"},
            size_max=10,
        )
        fig_mg.add_hline(
            y=avg_margin,
            line_dash="dash", line_color="grey",
            annotation_text=f"Overall avg {avg_margin:.1f}%",
        )
        fig_mg.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig_mg, use_container_width=True)

    with st.expander("🔍 View Category data table"):
        st.dataframe(df_category, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — MENU OPTIMISATION
# ════════════════════════════════════════════════════════════════════════════

with tabs[4]:
    if df_product is None:
        st.warning("⬅ Upload the **Product Profitability CSV** to see this section.")
        st.stop()

    st.subheader("Menu Optimisation")
    st.caption(
        "This section uses the Pareto (80/20) rule, K-Means clustering, and Gini-based Bloat Score "
        "to flag which products to **Keep**, **Monitor**, or **Remove** at each branch."
    )

    with st.spinner("Running analysis (Pareto + K-Means + Gini)…"):
        df_pareto          = compute_pareto(df_product)
        df_km, df_eff      = compute_kmeans(df_product)
        df_reco            = compute_recommendations(df_km, df_product)

    # ── Section A: Pareto ──────────────────────────────────────────────────
    st.markdown("### A · Pareto Analysis (80/20 Rule)")
    st.caption("% of the menu needed to generate 80% of branch profit — lower is leaner.")

    df_pareto["Short"] = df_pareto["Branch"].apply(short_name)
    colors_pareto = ["#d73027" if v > 40 else "#fee08b" if v > 25 else "#1a9850"
                     for v in df_pareto["Menu_Pct_for_80pct"]]

    col1, col2 = st.columns(2)
    with col1:
        fig_pareto = px.bar(
            df_pareto,
            x="Menu_Pct_for_80pct", y="Short",
            orientation="h",
            title="% of Menu Needed for 80% of Profit",
            labels={"Menu_Pct_for_80pct": "% of Menu", "Short": ""},
            color="Menu_Pct_for_80pct",
            color_continuous_scale=[[0, "#1a9850"], [0.5, "#fee08b"], [1, "#d73027"]],
            text=df_pareto["Menu_Pct_for_80pct"].apply(lambda v: f"{v}%"),
        )
        fig_pareto.add_vline(x=20, line_dash="dash", line_color="black", annotation_text="20% ideal")
        fig_pareto.add_vline(x=40, line_dash="dash", line_color="red",   annotation_text="40% bloated")
        fig_pareto.update_layout(coloraxis_showscale=False, height=520)
        fig_pareto.update_traces(textposition="outside")
        st.plotly_chart(fig_pareto, use_container_width=True)

    with col2:
        pareto_stk = df_pareto.set_index("Short").copy()
        pareto_stk["Remaining_Profitable"] = pareto_stk["Profitable"] - pareto_stk["Products_for_80pct"]
        fig_stk_pareto = go.Figure()
        fig_stk_pareto.add_trace(go.Bar(name="Drive 80% profit", x=pareto_stk.index, y=pareto_stk["Products_for_80pct"], marker_color="#1a9850"))
        fig_stk_pareto.add_trace(go.Bar(name="Remaining profitable", x=pareto_stk.index, y=pareto_stk["Remaining_Profitable"], marker_color="#a6d96a"))
        fig_stk_pareto.add_trace(go.Bar(name="Zero profit", x=pareto_stk.index, y=pareto_stk["Zero_Profit"], marker_color="#ffffbf"))
        fig_stk_pareto.add_trace(go.Bar(name="Loss-making", x=pareto_stk.index, y=pareto_stk["Loss_Making"], marker_color="#d73027"))
        fig_stk_pareto.update_layout(
            barmode="stack",
            title="Menu Composition per Branch",
            xaxis_tickangle=-45,
            yaxis_title="Number of Products",
            height=520,
        )
        st.plotly_chart(fig_stk_pareto, use_container_width=True)

    # ── Section B: K-Means clustering ──────────────────────────────────────
    st.markdown("### B · Product Tier Clustering (K-Means, k=4)")
    tier_order = ["Stars", "Workhorses", "Marginal", "Loss-Makers"]

    tier_dist = (
        df_km.groupby(["Branch", "Tier"])["Product"]
        .count().unstack(fill_value=0)
        .reindex(columns=tier_order, fill_value=0)
    )
    tier_dist.index = tier_dist.index.map(short_name)
    tier_dist_pct = tier_dist.div(tier_dist.sum(axis=1), axis=0) * 100

    col1, col2 = st.columns(2)
    with col1:
        fig_tier_abs = go.Figure()
        for tier in tier_order:
            fig_tier_abs.add_trace(go.Bar(
                name=tier, x=tier_dist.index, y=tier_dist[tier],
                marker_color=TIER_COLORS[tier],
            ))
        fig_tier_abs.update_layout(
            barmode="stack",
            title="Product Tier Count per Branch",
            xaxis_tickangle=-45,
            yaxis_title="Number of Products",
            height=480,
        )
        st.plotly_chart(fig_tier_abs, use_container_width=True)

    with col2:
        fig_tier_pct = go.Figure()
        for tier in tier_order:
            fig_tier_pct.add_trace(go.Bar(
                name=tier, x=tier_dist_pct.index, y=tier_dist_pct[tier],
                marker_color=TIER_COLORS[tier],
            ))
        fig_tier_pct.update_layout(
            barmode="stack",
            title="Product Tier Composition (%) per Branch",
            xaxis_tickangle=-45,
            yaxis_title="% of Menu",
            height=480,
        )
        st.plotly_chart(fig_tier_pct, use_container_width=True)

    # Cluster summary table
    summary_km = (
        df_km.groupby("Tier")
        .agg(Count=("Product", "count"),
             Avg_Profit=("Total_Profit", "mean"),
             Avg_Margin=("Total_Profit_Pct", "mean"),
             Avg_Qty=("Qty", "mean"))
        .reindex(tier_order)
        .round(1)
    )
    st.markdown("**Cluster Summary**")
    st.dataframe(summary_km, use_container_width=True)

    # ── Section C: Bloat Score ─────────────────────────────────────────────
    st.markdown("### C · Menu Efficiency (Bloat Score)")
    st.caption(
        "Bloat Score = (1 − Gini) × 50 + Loss% × 0.5 — higher means more pruning needed. "
        "Gini measures profit concentration: higher Gini = fewer products dominate."
    )

    df_eff["Short"] = df_eff["Branch"].apply(short_name)
    col1, col2 = st.columns(2)

    with col1:
        bloat_colors = ["#d73027" if v > 35 else "#fee08b" if v > 25 else "#1a9850"
                        for v in df_eff["Bloat_Score"]]
        fig_bloat = px.bar(
            df_eff[::-1].reset_index(drop=True),
            x="Bloat_Score", y="Short",
            orientation="h",
            title="Menu Bloat Score per Branch (↑ = more pruning needed)",
            labels={"Bloat_Score": "Bloat Score", "Short": ""},
            color="Bloat_Score",
            color_continuous_scale=[[0, "#1a9850"], [0.5, "#fee08b"], [1, "#d73027"]],
            text="Bloat_Score",
        )
        fig_bloat.add_vline(x=35, line_dash="dash", line_color="red",    annotation_text="High (>35)")
        fig_bloat.add_vline(x=25, line_dash="dash", line_color="orange", annotation_text="Moderate (>25)")
        fig_bloat.update_layout(coloraxis_showscale=False, height=520)
        fig_bloat.update_traces(textposition="outside")
        st.plotly_chart(fig_bloat, use_container_width=True)

    with col2:
        gini_med = df_eff["Gini"].median()
        loss_med = df_eff["Loss_Pct"].median()
        scatter_colors = ["#d73027" if b > 35 else "#fee08b" if b > 25 else "#1a9850"
                          for b in df_eff["Bloat_Score"]]
        fig_quad = px.scatter(
            df_eff,
            x="Gini", y="Loss_Pct",
            text="Short",
            title="Gini vs Loss-Making %<br><sup>Top-right = highest pruning urgency</sup>",
            labels={"Gini": "Gini Coefficient", "Loss_Pct": "% Loss-Making Products"},
            color=scatter_colors,
        )
        fig_quad.add_vline(x=gini_med, line_dash="dash", line_color="grey")
        fig_quad.add_hline(y=loss_med, line_dash="dash", line_color="grey")
        fig_quad.update_traces(textposition="top center", marker_size=10)
        fig_quad.update_layout(height=520, showlegend=False)
        st.plotly_chart(fig_quad, use_container_width=True)

    # ── Section D: Recommendations ─────────────────────────────────────────
    st.markdown("### D · Branch-Level Recommendations")

    action_pivot = (
        df_reco.groupby(["Branch", "Recommendation"])
        .size().unstack(fill_value=0)
    )
    for col in ["KEEP", "MONITOR", "REMOVE"]:
        if col not in action_pivot.columns:
            action_pivot[col] = 0
    action_pivot = action_pivot[["KEEP", "MONITOR", "REMOVE"]]
    action_pct   = action_pivot.div(action_pivot.sum(axis=1), axis=0) * 100
    action_pivot.index = action_pivot.index.map(short_name)
    action_pct.index   = action_pct.index.map(short_name)

    col1, col2 = st.columns(2)
    for ax, data, title, ylabel in [
        (col1, action_pivot, "Recommended Actions (count)", "Number of Products"),
        (col2, action_pct,   "Recommended Actions (%)",     "% of Menu"),
    ]:
        fig_reco = go.Figure()
        for action in ["KEEP", "MONITOR", "REMOVE"]:
            fig_reco.add_trace(go.Bar(
                name=action, x=data.index, y=data[action],
                marker_color=ACTION_COLORS[action],
            ))
        fig_reco.update_layout(
            barmode="stack", title=title,
            xaxis_tickangle=-45, yaxis_title=ylabel, height=460,
        )
        ax.plotly_chart(fig_reco, use_container_width=True)

    # Summary table with bloat score
    st.markdown("**Branch Optimisation Summary**")
    summary_tbl = action_pivot.copy()
    summary_tbl["Total"]    = summary_tbl.sum(axis=1)
    summary_tbl["Remove_%"] = (summary_tbl["REMOVE"] / summary_tbl["Total"] * 100).round(1)
    eff_idx = df_eff.set_index("Short")
    summary_tbl["Gini"]       = eff_idx["Gini"]
    summary_tbl["Bloat_Score"]= eff_idx["Bloat_Score"]
    summary_tbl["Urgency"]    = summary_tbl["Bloat_Score"].apply(
        lambda v: "🔴 HIGH" if v > 35 else "🟡 MODERATE" if v > 25 else "🟢 LOW"
    )
    st.dataframe(summary_tbl.sort_values("Bloat_Score", ascending=False), use_container_width=True)

    # Per-branch removal detail
    st.markdown("**Top 5 Products to Remove per Branch**")
    remove_df = df_reco[df_reco["Recommendation"] == "REMOVE"].copy()
    branch_filter = st.selectbox(
        "Select branch",
        options=sorted(remove_df["Branch"].unique()),
        key="removal_branch",
    )
    branch_removes = remove_df[remove_df["Branch"] == branch_filter].sort_values("Profit").head(10)
    st.dataframe(
        branch_removes[["Product", "Qty", "Revenue", "Profit", "Margin_%", "Tier"]].reset_index(drop=True),
        use_container_width=True,
    )

    with st.expander("🔍 View full recommendations table"):
        st.dataframe(df_reco, use_container_width=True)

    # ── CSV export ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**⬇ Export Results**")
    col_dl1, col_dl2, col_dl3 = st.columns(3)

    with col_dl1:
        csv_reco = df_reco.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Recommendations CSV",
            csv_reco, "recommendations.csv", "text/csv",
        )
    with col_dl2:
        csv_eff = df_eff.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Bloat Scores CSV",
            csv_eff, "bloat_scores.csv", "text/csv",
        )
    with col_dl3:
        csv_pareto = df_pareto.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Pareto Summary CSV",
            csv_pareto, "pareto_summary.csv", "text/csv",
        )


# ──────────────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Stories Coffee Profit Dashboard · Built with Streamlit & Plotly · "
    "All data processed in-memory — no external servers."
)
