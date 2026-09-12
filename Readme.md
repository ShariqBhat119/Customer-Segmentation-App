
# Customer Segmentation Explorer

An interactive Streamlit rebuild of the RFM customer-segmentation capstone
notebook. Upload the *Online Retail II* workbook and walk through EDA,
cleaning, RFM feature engineering, and four clustering approaches.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`)
and upload your `online_retail_II.xlsx` file (or any workbook with the same
columns: `Invoice`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`,
`Price`, `Customer ID`, `Country`).

## What's inside

| Tab | What it does |
|---|---|
| **EDA** | Raw preview, column info, summary stats, null/negative-quantity checks |
| **Cleaning** | Regex-based invoice (6-digit) and stock-code (5-digit ± letters) validation, matching the notebook's rules; shows before/after row counts |
| **RFM Features** | Builds Recency/Frequency/Monetary per customer, plots distributions, and lets you tune the IQR outlier-removal multiplier interactively |
| **KMeans** | Elbow + silhouette scan across a range of *k*, then a final model with adjustable *k*, 3D cluster scatter, and violin plots per feature |
| **DBSCAN** | k-distance graph to help pick `eps`, adjustable `eps`/`min_samples`, cluster/noise counts, 3D scatter |
| **Gaussian Mixture** | Adjustable number of components, covariance type, and regularization; 3D scatter, PCA projection with confidence ellipses, pairwise scatter matrix, and per-customer membership confidence |
| **Hierarchical** | Ward-linkage dendrogram, adjustable cluster count, 3D scatter, cluster profile table |
| **Compare Models** | Side-by-side silhouette scores for whichever methods you've run |

## Notes on fidelity to the original notebook

- **KMeans / DBSCAN** use the *raw-scale* RFM features with IQR-based outlier
  trimming (the notebook's first pipeline).
- **Gaussian Mixture / Hierarchical** use *log-transformed* RFM features
  computed from the full customer base, no outlier trimming (the notebook's
  second pipeline). This matches the notebook exactly — the two pipelines
  aren't meant to be perfectly comparable to each other.
- The notebook's DBSCAN cell referenced an undefined `min_samples` variable
  in one place and an undefined `kmeans_cluster_labels` in its comparison
  cell — both are fixed here so the app runs cleanly end-to-end.
- 3D plots use Plotly instead of Matplotlib so they're interactive
  (rotate/zoom) in the browser.
