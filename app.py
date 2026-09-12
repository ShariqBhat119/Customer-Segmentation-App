"""
Customer Segmentation Explorer
==============================
An interactive Streamlit rebuild of the "Customer Segmentation" capstone
notebook. Upload the Online Retail II workbook and walk through EDA,
cleaning, RFM feature engineering, and four clustering approaches
(KMeans, DBSCAN, Gaussian Mixture, Hierarchical/Agglomerative).

Run with:  streamlit run app.py
"""

import datetime as dt
import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.figure_factory import create_dendrogram
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Customer Segmentation Explorer",
    page_icon="\U0001F6D2",
    layout="wide",
)

CLUSTER_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                   "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes, sheet_name) -> pd.DataFrame:
    """Load one sheet (or all sheets concatenated) from the uploaded workbook."""
    buf = io.BytesIO(file_bytes)
    if sheet_name == "__ALL__":
        sheets = pd.read_excel(buf, sheet_name=None)
        df = pd.concat(sheets.values(), ignore_index=True)
    else:
        df = pd.read_excel(buf, sheet_name=sheet_name)
    return df


@st.cache_data(show_spinner=False)
def get_sheet_names(file_bytes: bytes):
    xl = pd.ExcelFile(io.BytesIO(file_bytes))
    return xl.sheet_names


# ---------------------------------------------------------------------
# Cleaning (regex-based invoice / stock-code validation, as in the notebook)
# ---------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def clean_data(df: pd.DataFrame):
    """Replicates the notebook's cleaning pipeline and returns the cleaned
    dataframe plus a small dict of diagnostic counts for display."""
    diagnostics = {}
    working = df.copy()
    working["Invoice"] = working["Invoice"].astype("str")
    working["StockCode"] = working["StockCode"].astype("str")

    diagnostics["raw_rows"] = len(working)
    diagnostics["null_customer_id"] = int(working["Customer ID"].isna().sum())
    diagnostics["negative_quantity"] = int((working["Quantity"] < 0).sum())

    non_six_digit_invoice = (~working["Invoice"].str.match(r"^\d{6}$")).sum()
    diagnostics["non_six_digit_invoice"] = int(non_six_digit_invoice)

    invoice_prefixes = (
        working["Invoice"].str.replace(r"[0-9]", "", regex=True).replace("", np.nan).dropna().unique()
    )
    diagnostics["invoice_letter_prefixes"] = sorted(invoice_prefixes.tolist())

    valid_stockcode = (
        working["StockCode"].str.match(r"^\d{5}$")
        | working["StockCode"].str.match(r"^\d{5}[a-zA-Z]+$")
    )
    diagnostics["invalid_stockcode_rows"] = int((~valid_stockcode).sum())

    cleaned = working.copy()

    # Keep only 6-digit invoices (drops cancellations "C..." and other codes)
    cleaned = cleaned[cleaned["Invoice"].str.match(r"^\d{6}$") == True]  # noqa: E712

    # Keep only 5-digit stock codes (optionally with trailing letters)
    mask = (
        (cleaned["StockCode"].str.match(r"^\d{5}$") == True)  # noqa: E712
        | (cleaned["StockCode"].str.match(r"^\d{5}[a-zA-Z]+$") == True)  # noqa: E712
    )
    cleaned = cleaned[mask]

    cleaned = cleaned.dropna(subset=["Customer ID"])

    zero_price_rows = int((cleaned["Price"] == 0).sum())
    diagnostics["zero_price_rows"] = zero_price_rows
    cleaned = cleaned[cleaned["Price"] > 0.0]

    diagnostics["cleaned_rows"] = len(cleaned)
    diagnostics["pct_retained"] = round(len(cleaned) / len(df) * 100, 2) if len(df) else 0.0

    cleaned["SalesLineTotal"] = cleaned["Quantity"] * cleaned["Price"]

    return cleaned, diagnostics


# ---------------------------------------------------------------------
# RFM feature engineering (KMeans / DBSCAN path — raw-scale RFM + IQR trim)
# ---------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def build_rfm_raw(cleaned: pd.DataFrame) -> pd.DataFrame:
    aggregated = cleaned.groupby(by="Customer ID", as_index=False).agg(
        MonetaryValue=("SalesLineTotal", "sum"),
        Frequency=("Invoice", "nunique"),
        LastInvoiceDate=("InvoiceDate", "max"),
    )
    max_invoice_date = aggregated["LastInvoiceDate"].max()
    aggregated["Recency"] = (max_invoice_date - aggregated["LastInvoiceDate"]).dt.days
    return aggregated


def iqr_outlier_mask(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return (series > (q3 + multiplier * iqr)) | (series < (q1 - multiplier * iqr))


@st.cache_data(show_spinner=False)
def remove_rfm_outliers(aggregated: pd.DataFrame, multiplier: float = 1.5):
    monetary_mask = iqr_outlier_mask(aggregated["MonetaryValue"], multiplier)
    frequency_mask = iqr_outlier_mask(aggregated["Frequency"], multiplier)
    outlier_mask = monetary_mask | frequency_mask
    non_outliers = aggregated[~outlier_mask].copy()
    outliers = aggregated[outlier_mask].copy()
    return non_outliers, outliers


# ---------------------------------------------------------------------
# RFM feature engineering (GMM / Hierarchical path — log-transformed RFM)
# ---------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def build_rfm_log(df_raw: pd.DataFrame):
    df_clean = df_raw.dropna(subset=["Customer ID"]).copy()
    df_clean = df_clean[(df_clean["Quantity"] > 0) & (df_clean["Price"] > 0)]
    df_clean["TotalPrice"] = df_clean["Quantity"] * df_clean["Price"]

    snapshot_date = df_clean["InvoiceDate"].max() + dt.timedelta(days=1)

    rfm = df_clean.groupby("Customer ID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=("TotalPrice", "sum"),
    ).reset_index()

    rfm_log = np.log1p(rfm[["Recency", "Frequency", "Monetary"]])
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)

    return rfm, rfm_scaled


# ---------------------------------------------------------------------
# Clustering helpers
# ---------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def kmeans_scan(X, k_min: int, k_max: int):
    inertia, sil_scores = [], []
    k_values = list(range(k_min, k_max + 1))
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, max_iter=1000, n_init=10)
        labels = km.fit_predict(X)
        sil_scores.append(silhouette_score(X, labels))
        inertia.append(km.inertia_)
    return k_values, inertia, sil_scores


@st.cache_data(show_spinner=False)
def run_kmeans(X, k: int):
    km = KMeans(n_clusters=k, random_state=42, max_iter=1000, n_init=10)
    labels = km.fit_predict(X)
    return labels


@st.cache_data(show_spinner=False)
def k_distance_values(X, min_samples: int):
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors.fit(X)
    distances, _ = neighbors.kneighbors(X)
    distances = np.sort(distances[:, -1])
    return distances


@st.cache_data(show_spinner=False)
def run_dbscan(X, eps: float, min_samples: int):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    return labels


@st.cache_data(show_spinner=False)
def run_gmm(X, n_components: int, covariance_type: str = "full", reg_covar: float = 1e-6):
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=42,
        reg_covar=reg_covar,
        n_init=5,
    )
    labels = gmm.fit_predict(X)
    probs = gmm.predict_proba(X)
    return labels, probs, gmm


@st.cache_data(show_spinner=False)
def run_hierarchical(X, n_clusters: int):
    model = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage="ward")
    labels = model.fit_predict(X)
    return labels


def cluster_summary_table(df: pd.DataFrame, cluster_col: str, value_cols, id_col: str):
    agg = {c: "mean" for c in value_cols}
    agg[id_col] = "count"
    summary = df.groupby(cluster_col).agg(agg).rename(columns={id_col: "Count"}).reset_index()
    total = summary["Count"].sum()
    summary["Pct"] = (summary["Count"] / total * 100).round(1)
    return summary


def safe_silhouette(X, labels):
    unique = set(labels)
    unique.discard(-1)
    if len(unique) < 2:
        return None
    mask = labels != -1 if -1 in labels else np.ones(len(labels), dtype=bool)
    if mask.sum() < 2:
        return None
    return silhouette_score(X[mask], np.asarray(labels)[mask])


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------

def scatter_3d(df, x, y, z, color, title, color_discrete=True):
    if color_discrete:
        fig = px.scatter_3d(
            df, x=x, y=y, z=z, color=df[color].astype(str),
            title=title, opacity=0.7,
            color_discrete_sequence=CLUSTER_COLORS,
        )
    else:
        fig = px.scatter_3d(df, x=x, y=y, z=z, color=color, title=title, opacity=0.7)
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(height=650, legend_title_text="Cluster")
    return fig


def violin_by_cluster(df, cluster_col, value_col, title):
    fig = px.violin(
        df, x=df[cluster_col].astype(str), y=value_col, color=df[cluster_col].astype(str),
        box=True, points=False, title=title,
        color_discrete_sequence=CLUSTER_COLORS,
    )
    fig.update_layout(showlegend=False, xaxis_title="Cluster", height=350)
    return fig


# ---------------------------------------------------------------------
# Sidebar — data upload
# ---------------------------------------------------------------------

st.title("\U0001F6D2 Customer Segmentation Explorer")
st.caption(
    "Interactive rebuild of the RFM customer-segmentation capstone notebook — "
    "EDA, cleaning, feature engineering, and four clustering methods."
)

with st.sidebar:
    st.header("1. Data")
    uploaded = st.file_uploader("Upload Online Retail II workbook (.xlsx)", type=["xlsx"])

    if uploaded is None:
        st.info("Upload the retail workbook to get started.")
        st.stop()

    file_bytes = uploaded.getvalue()
    sheet_names = get_sheet_names(file_bytes)
    sheet_options = sheet_names + (["__ALL__"] if len(sheet_names) > 1 else [])
    sheet_choice = st.selectbox(
        "Sheet",
        sheet_options,
        format_func=lambda s: "All sheets (concatenated)" if s == "__ALL__" else s,
    )

df_raw = load_data(file_bytes, sheet_choice)

with st.sidebar:
    st.success(f"Loaded {len(df_raw):,} rows.")
    required_cols = {"Invoice", "StockCode", "Quantity", "InvoiceDate", "Price", "Customer ID"}
    missing = required_cols - set(df_raw.columns)
    if missing:
        st.error(f"Missing expected column(s): {', '.join(sorted(missing))}")
        st.stop()


tab_eda, tab_clean, tab_rfm, tab_kmeans, tab_dbscan, tab_gmm, tab_hier, tab_compare = st.tabs(
    ["EDA", "Cleaning", "RFM Features", "KMeans", "DBSCAN", "Gaussian Mixture",
     "Hierarchical", "Compare Models"]
)

# ---------------------------------------------------------------------
# Tab: EDA
# ---------------------------------------------------------------------

with tab_eda:
    st.subheader("Raw data preview")
    st.dataframe(df_raw.head(10), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Column info**")
        info_df = pd.DataFrame({
            "dtype": df_raw.dtypes.astype(str),
            "non_null": df_raw.notna().sum(),
            "nulls": df_raw.isna().sum(),
        })
        st.dataframe(info_df, use_container_width=True)
    with col2:
        st.markdown("**Numeric summary**")
        st.dataframe(df_raw.describe(), use_container_width=True)

    st.markdown("**Object-column summary**")
    st.dataframe(df_raw.describe(include="O"), use_container_width=True)

    st.markdown("**Rows with a null Customer ID**")
    st.dataframe(df_raw[df_raw["Customer ID"].isna()].head(10), use_container_width=True)

    st.markdown("**Rows with negative Quantity (likely cancellations/returns)**")
    st.dataframe(df_raw[df_raw["Quantity"] < 0].head(10), use_container_width=True)


# ---------------------------------------------------------------------
# Tab: Cleaning
# ---------------------------------------------------------------------

with tab_clean:
    st.subheader("Regex-based cleaning")
    st.markdown(
        "Mirrors the notebook's validation rules: invoices must be exactly "
        "6 digits (this drops cancellations, which start with `C`, and other "
        "non-standard codes), and stock codes must be 5 digits optionally "
        "followed by letters."
    )

    cleaned_df, diag = clean_data(df_raw)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Raw rows", f"{diag['raw_rows']:,}")
    m2.metric("Null Customer ID", f"{diag['null_customer_id']:,}")
    m3.metric("Non-6-digit invoices", f"{diag['non_six_digit_invoice']:,}")
    m4.metric("Invalid stock codes", f"{diag['invalid_stockcode_rows']:,}")

    st.markdown(f"**Invoice letter prefixes found:** `{diag['invoice_letter_prefixes']}`")

    st.markdown("**Rows with Invoice starting with `A` (e.g. adjustment entries)**")
    a_rows = df_raw.assign(Invoice=df_raw["Invoice"].astype(str))
    st.dataframe(a_rows[a_rows["Invoice"].str.startswith("A")].head(10), use_container_width=True)

    m5, m6, m7 = st.columns(3)
    m5.metric("Zero-price rows dropped", f"{diag['zero_price_rows']:,}")
    m6.metric("Cleaned rows", f"{diag['cleaned_rows']:,}")
    m7.metric("Retained", f"{diag['pct_retained']}%")

    st.markdown("**Cleaned data preview** (adds `SalesLineTotal = Quantity × Price`)")
    st.dataframe(cleaned_df.head(10), use_container_width=True)


# ---------------------------------------------------------------------
# Tab: RFM Features
# ---------------------------------------------------------------------

with tab_rfm:
    st.subheader("RFM feature engineering")
    aggregated_df = build_rfm_raw(cleaned_df)
    st.dataframe(aggregated_df.head(10), use_container_width=True)

    st.markdown("**Distributions**")
    dist_cols = st.columns(3)
    for c, color in zip(["MonetaryValue", "Frequency", "Recency"], ["skyblue", "lightgreen", "salmon"]):
        with dist_cols[["MonetaryValue", "Frequency", "Recency"].index(c)]:
            fig = px.histogram(aggregated_df, x=c, nbins=20, title=f"{c} distribution")
            fig.update_traces(marker_color=color)
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Boxplots (raw scale — note the outliers)**")
    box_cols = st.columns(3)
    for i, c in enumerate(["MonetaryValue", "Frequency", "Recency"]):
        with box_cols[i]:
            fig = px.box(aggregated_df, y=c, title=f"{c} boxplot")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("**Outlier removal (IQR method on Monetary & Frequency)**")
    iqr_mult = st.slider("IQR multiplier", 1.0, 3.0, 1.5, 0.1, key="iqr_mult")
    non_outliers_df, outliers_df = remove_rfm_outliers(aggregated_df, iqr_mult)

    o1, o2, o3 = st.columns(3)
    o1.metric("Total customers", f"{len(aggregated_df):,}")
    o2.metric("Outliers removed", f"{len(outliers_df):,}")
    o3.metric("Remaining", f"{len(non_outliers_df):,}")

    st.markdown("**Boxplots after outlier removal**")
    box_cols2 = st.columns(3)
    for i, c in enumerate(["MonetaryValue", "Frequency", "Recency"]):
        with box_cols2[i]:
            fig = px.box(non_outliers_df, y=c, title=f"{c} boxplot (cleaned)")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

    fig3d = px.scatter_3d(
        non_outliers_df, x="MonetaryValue", y="Frequency", z="Recency",
        title="3D RFM scatter (non-outliers)", opacity=0.6,
    )
    fig3d.update_traces(marker=dict(size=3))
    fig3d.update_layout(height=600)
    st.plotly_chart(fig3d, use_container_width=True)

    # Scale for downstream clustering (KMeans / DBSCAN path)
    scaler_raw = StandardScaler()
    scaled_data = scaler_raw.fit_transform(non_outliers_df[["MonetaryValue", "Frequency", "Recency"]])
    scaled_data_df = pd.DataFrame(
        scaled_data, index=non_outliers_df.index, columns=["MonetaryValue", "Frequency", "Recency"]
    )

    st.session_state["non_outliers_df"] = non_outliers_df
    st.session_state["scaled_data"] = scaled_data
    st.session_state["scaled_data_df"] = scaled_data_df

    # GMM / hierarchical path (log-transformed, full customer base)
    rfm_log_df, rfm_scaled = build_rfm_log(df_raw)
    st.session_state["rfm_log_df"] = rfm_log_df
    st.session_state["rfm_scaled"] = rfm_scaled

    st.markdown("---")
    st.caption(
        "Two feature sets are prepared for downstream use: a **raw-scale, "
        "outlier-trimmed** RFM table (used by KMeans/DBSCAN below, matching "
        "the notebook), and a **log-transformed, full-population** RFM table "
        "(used by GMM/Hierarchical, also matching the notebook)."
    )


# ---------------------------------------------------------------------
# Tab: KMeans
# ---------------------------------------------------------------------

with tab_kmeans:
    st.subheader("KMeans clustering")
    if "scaled_data_df" not in st.session_state:
        st.warning("Visit the **RFM Features** tab first to prepare the data.")
    else:
        scaled_data_df = st.session_state["scaled_data_df"]
        non_outliers_df = st.session_state["non_outliers_df"].copy()

        c1, c2 = st.columns(2)
        k_min, k_max = c1.slider("Elbow/silhouette scan range", 2, 15, (2, 12))
        if st.checkbox("Run elbow & silhouette scan", value=True, key="km_scan"):
            k_values, inertia, sil_scores = kmeans_scan(scaled_data_df.values, k_min, k_max)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=k_values, y=inertia, mode="lines+markers", name="Inertia"))
            fig.update_layout(title="KMeans inertia vs. k", xaxis_title="k", yaxis_title="Inertia", height=350)
            st.plotly_chart(fig, use_container_width=True)

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=k_values, y=sil_scores, mode="lines+markers",
                                       line_color="orange", name="Silhouette"))
            fig2.update_layout(title="Silhouette score vs. k", xaxis_title="k",
                                yaxis_title="Silhouette score", height=350)
            st.plotly_chart(fig2, use_container_width=True)

        k = st.slider("Number of clusters (k) for the final model", 2, 12, 4, key="km_k")
        labels = run_kmeans(scaled_data_df.values, k)
        non_outliers_df["Cluster"] = labels
        sil = silhouette_score(scaled_data_df.values, labels)
        st.metric("Silhouette score", f"{sil:.4f}")

        st.plotly_chart(
            scatter_3d(non_outliers_df, "MonetaryValue", "Frequency", "Recency",
                       "Cluster", "3D customer clusters (KMeans)"),
            use_container_width=True,
        )

        st.markdown("**Cluster profiles**")
        summary = cluster_summary_table(
            non_outliers_df, "Cluster", ["MonetaryValue", "Frequency", "Recency"], "Customer ID"
        )
        st.dataframe(summary, use_container_width=True)

        st.markdown("**Feature distribution by cluster**")
        vcols = st.columns(3)
        for i, c in enumerate(["MonetaryValue", "Frequency", "Recency"]):
            with vcols[i]:
                st.plotly_chart(
                    violin_by_cluster(non_outliers_df, "Cluster", c, f"{c} by cluster"),
                    use_container_width=True,
                )

        st.session_state["kmeans_labels"] = labels
        st.session_state["kmeans_sil"] = sil


# ---------------------------------------------------------------------
# Tab: DBSCAN
# ---------------------------------------------------------------------

with tab_dbscan:
    st.subheader("DBSCAN clustering")
    if "scaled_data_df" not in st.session_state:
        st.warning("Visit the **RFM Features** tab first to prepare the data.")
    else:
        scaled_data = st.session_state["scaled_data"]
        scaled_data_df = st.session_state["scaled_data_df"]
        non_outliers_df = st.session_state["non_outliers_df"].copy()

        min_samples = st.slider("min_samples", 5, 100, 50, key="db_minsamp")

        if st.checkbox("Show k-distance graph (for choosing eps)", value=True, key="db_kdist"):
            distances = k_distance_values(scaled_data, min_samples)
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=distances, mode="lines"))
            fig.update_layout(
                title=f"K-distance graph (k={min_samples}) — look for the elbow",
                xaxis_title="Points sorted by distance",
                yaxis_title=f"Distance to {min_samples}th nearest neighbor",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

        eps_value = st.slider("eps", 0.05, 2.0, 0.3, 0.05, key="db_eps")
        labels = run_dbscan(scaled_data, eps_value, min_samples)
        non_outliers_df["Cluster"] = labels

        unique_labels, counts = np.unique(labels, return_counts=True)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

        d1, d2 = st.columns(2)
        d1.metric("Clusters found", n_clusters)
        d2.metric("Noise points", int(counts[unique_labels == -1][0]) if -1 in unique_labels else 0)

        dist_table = pd.DataFrame({"Cluster": unique_labels, "Count": counts})
        dist_table["Cluster"] = dist_table["Cluster"].apply(lambda x: "Noise (-1)" if x == -1 else x)
        st.dataframe(dist_table, use_container_width=True)

        sil = safe_silhouette(scaled_data, labels)
        if sil is not None:
            st.metric("Silhouette score (excluding noise)", f"{sil:.4f}")
        else:
            st.info("Not enough non-noise clusters to compute a silhouette score. Try adjusting eps/min_samples.")

        st.plotly_chart(
            scatter_3d(non_outliers_df, "MonetaryValue", "Frequency", "Recency",
                       "Cluster", "3D customer clusters (DBSCAN, -1 = noise)"),
            use_container_width=True,
        )

        st.markdown("**Feature distribution by cluster**")
        vcols = st.columns(3)
        for i, c in enumerate(["MonetaryValue", "Frequency", "Recency"]):
            with vcols[i]:
                st.plotly_chart(
                    violin_by_cluster(non_outliers_df, "Cluster", c, f"{c} by cluster"),
                    use_container_width=True,
                )

        st.session_state["dbscan_labels"] = labels
        st.session_state["dbscan_sil"] = sil


# ---------------------------------------------------------------------
# Tab: Gaussian Mixture
# ---------------------------------------------------------------------

def gmm_ellipse_points(mean, cov, n_std=2.0, n_points=100):
    """Return x, y points tracing an n_std confidence ellipse for a 2D Gaussian."""
    U, s, _ = np.linalg.svd(cov)
    theta = np.linspace(0, 2 * np.pi, n_points)
    circle = np.stack([np.cos(theta), np.sin(theta)])
    ellipse = U @ np.diag(n_std * np.sqrt(s)) @ circle
    ellipse = ellipse + mean[:, None]
    return ellipse[0], ellipse[1]


with tab_gmm:
    st.subheader("Gaussian Mixture Model clustering")
    if "rfm_scaled" not in st.session_state:
        st.warning("Visit the **RFM Features** tab first to prepare the data.")
    else:
        rfm_scaled = st.session_state["rfm_scaled"]
        rfm_log_df = st.session_state["rfm_log_df"].copy()

        c1, c2, c3 = st.columns(3)
        n_components = c1.slider("Number of components", 2, 10, 4, key="gmm_k")
        covariance_type = c2.selectbox("Covariance type", ["full", "tied", "diag", "spherical"], key="gmm_cov")
        reg_covar = c3.select_slider(
            "Covariance regularization", options=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2], value=1e-6,
            format_func=lambda v: f"{v:.0e}", key="gmm_reg",
        )

        labels, probs, gmm_model = run_gmm(rfm_scaled, n_components, covariance_type, reg_covar)
        rfm_log_df["Cluster"] = labels
        rfm_log_df["MaxProb"] = probs.max(axis=1)

        sil = silhouette_score(rfm_scaled, labels)
        st.metric("Silhouette score", f"{sil:.4f}")

        st.plotly_chart(
            scatter_3d(
                pd.DataFrame(rfm_scaled, columns=["Log_Recency", "Log_Frequency", "Log_Monetary"])
                .assign(Cluster=labels),
                "Log_Monetary", "Log_Frequency", "Log_Recency",
                "Cluster", "3D customer clusters (GMM, scaled log-RFM space)",
            ),
            use_container_width=True,
        )

        st.markdown("**PCA projection with GMM confidence ellipses**")
        pca = PCA(n_components=2)
        rfm_pca = pca.fit_transform(rfm_scaled)
        gmm_2d = GaussianMixture(
            n_components=n_components, covariance_type=covariance_type,
            random_state=42, reg_covar=reg_covar, n_init=5,
        )
        labels_2d = gmm_2d.fit_predict(rfm_pca)

        fig = go.Figure()
        pca_df = pd.DataFrame(rfm_pca, columns=["PC1", "PC2"])
        pca_df["Cluster"] = labels_2d.astype(str)
        for i, cluster_id in enumerate(sorted(pca_df["Cluster"].unique(), key=int)):
            sub = pca_df[pca_df["Cluster"] == cluster_id]
            fig.add_trace(go.Scatter(
                x=sub["PC1"], y=sub["PC2"], mode="markers", name=f"Cluster {cluster_id}",
                marker=dict(size=5, opacity=0.6, color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)]),
            ))

        def component_covariance_2d(idx: int) -> np.ndarray:
            """Return a proper 2x2 covariance matrix for component idx,
            regardless of the GaussianMixture covariance_type used."""
            raw = gmm_2d.covariances_
            if covariance_type == "full":
                return raw[idx]
            if covariance_type == "tied":
                return raw  # shared (2, 2) matrix
            if covariance_type == "diag":
                return np.diag(raw[idx])
            # spherical: single variance scalar per component
            return np.eye(2) * raw[idx]

        for i, mean in enumerate(gmm_2d.means_):
            cov = component_covariance_2d(i)
            for n_std in (1, 2):
                ex, ey = gmm_ellipse_points(mean, cov, n_std=n_std)
                fig.add_trace(go.Scatter(
                    x=ex, y=ey, mode="lines", line=dict(color="red", width=1),
                    showlegend=False, opacity=0.4,
                ))
            fig.add_trace(go.Scatter(
                x=[mean[0]], y=[mean[1]], mode="markers",
                marker=dict(symbol="x", size=12, color="red", line=dict(width=2)),
                showlegend=False,
            ))

        fig.update_layout(
            title=f"GMM clusters on PCA-reduced RFM "
                  f"(PC1 {pca.explained_variance_ratio_[0]*100:.1f}% / PC2 {pca.explained_variance_ratio_[1]*100:.1f}% variance)",
            xaxis_title="PCA Component 1", yaxis_title="PCA Component 2", height=600,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Cluster profiles** (original Recency / Frequency / Monetary units)")
        summary = cluster_summary_table(
            rfm_log_df, "Cluster", ["Recency", "Frequency", "Monetary"], "Customer ID"
        )
        st.dataframe(summary, use_container_width=True)

        st.markdown("**Pairwise feature relationships**")
        pair_df = pd.DataFrame(rfm_scaled, columns=["Log_Recency", "Log_Frequency", "Log_Monetary"])
        pair_df["Cluster"] = labels.astype(str)
        fig_matrix = px.scatter_matrix(
            pair_df, dimensions=["Log_Recency", "Log_Frequency", "Log_Monetary"],
            color="Cluster", opacity=0.6, color_discrete_sequence=CLUSTER_COLORS,
        )
        fig_matrix.update_layout(height=700)
        st.plotly_chart(fig_matrix, use_container_width=True)

        st.markdown("**Membership confidence**")
        st.caption(
            "GMM is a soft-clustering method — MaxProb is each customer's probability "
            "of belonging to their assigned cluster. Low values indicate boundary cases."
        )
        st.dataframe(
            rfm_log_df[["Customer ID", "Recency", "Frequency", "Monetary", "Cluster", "MaxProb"]]
            .sort_values("MaxProb").head(15),
            use_container_width=True,
        )

        st.session_state["gmm_labels"] = labels
        st.session_state["gmm_sil"] = sil
        st.session_state["rfm_log_df"] = rfm_log_df


# ---------------------------------------------------------------------
# Tab: Hierarchical
# ---------------------------------------------------------------------

with tab_hier:
    st.subheader("Hierarchical (Agglomerative) clustering")
    if "rfm_scaled" not in st.session_state:
        st.warning("Visit the **RFM Features** tab first to prepare the data.")
    else:
        rfm_scaled = st.session_state["rfm_scaled"]
        rfm_log_df = st.session_state["rfm_log_df"].copy()

        st.markdown("**Dendrogram (Ward linkage, last 30 merges)**")
        dendro_fig = create_dendrogram(
            rfm_scaled, orientation="bottom", linkagefun=lambda x: linkage(x, method="ward"),
        )
        dendro_fig.update_layout(
            title="Hierarchical Clustering Dendrogram (Ward linkage)",
            xaxis_title="Sample index", yaxis_title="Euclidean distance", height=450,
        )
        st.plotly_chart(dendro_fig, use_container_width=True)
        st.caption(
            "Full dendrogram shown (Plotly renders all leaves rather than a "
            "truncated `lastp` view). Look for the tallest vertical gap to "
            "choose a sensible cut height / number of clusters."
        )

        n_clusters = st.slider("Number of clusters", 2, 10, 4, key="hier_k")
        labels = run_hierarchical(rfm_scaled, n_clusters)
        rfm_log_df["Hierarchical_Cluster"] = labels

        sil = silhouette_score(rfm_scaled, labels)
        st.metric("Silhouette score", f"{sil:.4f}")

        st.plotly_chart(
            scatter_3d(
                pd.DataFrame(rfm_scaled, columns=["Log_Recency", "Log_Frequency", "Log_Monetary"])
                .assign(Cluster=labels),
                "Log_Monetary", "Log_Frequency", "Log_Recency",
                "Cluster", "3D customer clusters (Hierarchical, scaled log-RFM space)",
            ),
            use_container_width=True,
        )

        st.markdown("**Cluster profiles**")
        summary = cluster_summary_table(
            rfm_log_df, "Hierarchical_Cluster", ["Recency", "Frequency", "Monetary"], "Customer ID"
        )
        st.dataframe(summary, use_container_width=True)

        st.session_state["hier_labels"] = labels
        st.session_state["hier_sil"] = sil
        st.session_state["rfm_log_df"] = rfm_log_df


# ---------------------------------------------------------------------
# Tab: Compare Models
# ---------------------------------------------------------------------

with tab_compare:
    st.subheader("Model comparison")
    st.caption(
        "Silhouette scores from each method run in this session. Note KMeans/DBSCAN "
        "score on the outlier-trimmed raw-scale RFM features, while GMM/Hierarchical "
        "score on the full-population log-scaled features — scores are informative "
        "within a method, not strictly comparable across the two feature sets."
    )

    rows = []
    if "kmeans_sil" in st.session_state:
        rows.append({"Model": "KMeans", "Silhouette Score": st.session_state["kmeans_sil"],
                     "Feature set": "Raw RFM, outliers trimmed"})
    if "dbscan_sil" in st.session_state and st.session_state["dbscan_sil"] is not None:
        rows.append({"Model": "DBSCAN", "Silhouette Score": st.session_state["dbscan_sil"],
                     "Feature set": "Raw RFM, outliers trimmed"})
    if "gmm_sil" in st.session_state:
        rows.append({"Model": "Gaussian Mixture", "Silhouette Score": st.session_state["gmm_sil"],
                     "Feature set": "Log RFM, full population"})
    if "hier_sil" in st.session_state:
        rows.append({"Model": "Hierarchical", "Silhouette Score": st.session_state["hier_sil"],
                     "Feature set": "Log RFM, full population"})

    if not rows:
        st.info("Run at least one clustering method in the tabs above to populate this comparison.")
    else:
        comp_df = pd.DataFrame(rows).sort_values("Silhouette Score", ascending=False)
        st.dataframe(comp_df, use_container_width=True)
        fig = px.bar(comp_df, x="Model", y="Silhouette Score", color="Model",
                     color_discrete_sequence=CLUSTER_COLORS, title="Silhouette score by method")
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption(
    "Built from the Customer Segmentation capstone notebook — RFM feature "
    "engineering with KMeans, DBSCAN, Gaussian Mixture, and Hierarchical clustering."
)
