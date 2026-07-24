import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import mygene
import gzip
import requests
import time
import re
import warnings

# Transcriptomics quality checks

def transcri_qc(transcri_data_c):
    if transcri_data_c.shape[0]<transcri_data_c.shape[1]:
        warnings.warn("Number of rows smaller than number of columns. Please ensure that the genes are correctly positioned in the rows and samples in the columns")
        
    # ── 1. Basic overview ─────────────────────────────────────────────────────────
    print("Shape of the dataset: ",transcri_data_c.shape)
    print(f"Value range: {transcri_data_c.min().min():.2f} to {transcri_data_c.max().max():.2f}")
    print(f"Number of zeros: {(transcri_data_c == 0).sum().sum()}")
    print(f"Number of NaNs: {transcri_data_c.isnull().sum().sum()}")

    # ── 2. Per-sample distribution (boxplot) ─────────────────────────────────────
    # Values should be positive and NOT centred on 0
    # All samples should have similar median and spread
    transcri_data_c.plot(kind='box', figsize=(16, 5), legend=False,
            title='Per-sample log2 expression distributions')
    plt.ylabel('log2(RSEM_UQ + 1)')
    plt.tight_layout(); plt.show()

    # ── 3. Per-sample median — flag outliers ─────────────────────────────────────
    sample_medians = transcri_data_c.median()
    sample_medians.plot(kind='bar', figsize=(14, 4),
                        title='Median expression per sample')
    plt.axhline(sample_medians.mean(), color='red', linestyle='--',
                label='Mean of medians')
    plt.ylabel('Median log2 expression')
    plt.legend(); plt.tight_layout(); plt.show()
    # Flag samples deviating > 2 SD from the mean median

    # ── 4. Density plots — check for bimodal or outlier samples ──────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    for col in transcri_data_c.columns:
        transcri_data_c[col].plot(kind='kde', ax=ax, alpha=0.3, linewidth=0.8)
    ax.set_xlabel('log2(RSEM_UQ + 1)')
    ax.set_title('Density curves — all samples')
    plt.tight_layout(); plt.show()

    # ── 5. Fraction of genes with zero / very low expression ─────────────────────
    zero_frac = (transcri_data_c == 0).mean(axis=1)
    print(f"Genes with 0 in >50% of samples: {(zero_frac > 0.5).sum()}")
    # Filter these out before downstream analysis

    # ── 6. Sample-sample Pearson correlation heatmap ─────────────────────────────
    # Use expressed genes only (avoid noise from zeros driving correlations)
    expressed = transcri_data_c   # log2 > 1 threshold: [transcri_data_c.mean(axis=1) > 1]
    corr = expressed.corr()
    corr_filled = corr.fillna(0)  # treat undefined correlation as "no relationship"
    sns.clustermap(corr_filled, cmap='RdYlBu_r', vmin=0.7, vmax=1.0, figsize=(10, 10))
    plt.suptitle('Sample correlation (expressed genes only)', y=1.02)
    plt.show()

    # ── 7. PCA ────────────────────────────────────────────────────────────────────
    X = StandardScaler().fit_transform(expressed.T)
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X)

    plt.figure(figsize=(7, 6))
    plt.scatter(pcs[:, 0], pcs[:, 1], s=40, alpha=0.7)
    for i, s in enumerate(transcri_data_c.columns):
        plt.annotate(s, (pcs[i, 0], pcs[i, 1]), fontsize=6)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.title('PCA of tumor samples'); plt.tight_layout(); plt.show()

    # ── 8. Summary stats ──────────────────────────────────────────────────────────
    print("Global median:", transcri_data_c.stack().median().round(3))
    print("Global IQR:", stats.iqr(transcri_data_c.stack().dropna()).round(3))
    print("Samples with median < 1 (suspicious):", (sample_medians < 1).sum())
    return None

#Proteomics quality checks

def proteomics_qc(prot_data_c, color_pc=False, color_code=None):

    if prot_data_c.shape[0]<prot_data_c.shape[1]:
        warnings.warn("Number of rows smaller than number of columns. Please ensure that the genes are correctly positioned in the rows and samples in the columns")

    # ── 1. Basic overview ─────────────────────────────────────────────────────────
    print("Shape of the dataset: ", prot_data_c.shape)               # (genes x samples)
    print(f"Value range: {prot_data_c.min().min():.2f} to {prot_data_c.max().max():.2f}")
    print("Data type: ", prot_data_c.dtypes.value_counts())
    print(f"Number of NaNs: {prot_data_c.isnull().sum().sum()}") # total missing values

    # ── 2. Missing value rate per sample ─────────────────────────────────────────
    missing_per_sample = prot_data_c.isnull().mean()
    missing_per_sample.plot(kind='bar', figsize=(14, 4), 
                            title='Missing value fraction per sample')
    plt.ylabel('Fraction missing'); plt.tight_layout(); plt.show()
    # Flag samples with >30% missing as potentially poor quality

    # ── 3. Value distribution per sample (should be centred near 0) ───────────────
    prot_data_c.plot(kind='box', figsize=(16, 5), legend=False,
            title='Per-sample protein abundance distributions')
    plt.axhline(0, color='red', linestyle='--', linewidth=0.8)
    plt.ylabel('log2 ratio vs reference'); plt.tight_layout(); plt.show()

    # ── 4. Density / violin plot ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    for col in prot_data_c.columns:
        prot_data_c[col].dropna().plot(kind='kde', ax=ax, alpha=0.3, linewidth=0.8)
    ax.axvline(0, color='red', linestyle='--')
    ax.set_xlabel('log2 ratio'); ax.set_title('Density of all samples')
    plt.tight_layout(); plt.show()

    # ── 5. Sample-sample Pearson correlation heatmap ─────────────────────────────
    corr = prot_data_c.corr()
    corr_filled = corr.fillna(0)  # treat undefined correlation as "no relationship"
    sns.clustermap(corr_filled, cmap='RdYlBu_r', vmin=0.5, vmax=1.0,
                   figsize=(10, 10), annot=False)
    plt.title('Sample correlation matrix'); plt.show()
    # Expect high inter-sample correlations (>0.8); outliers stand out as low-correlation rows/columns

    # ── 6. PCA for global structure / batch effects ───────────────────────────────
    df_complete = prot_data_c.dropna()   # use only fully-observed proteins
    X = StandardScaler().fit_transform(df_complete.T)   # samples x proteins
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X)

    plt.figure(figsize=(7, 6))
    if color_pc:
        plt.scatter(pcs[:, 0], pcs[:, 1], s=40, alpha=0.7, c=color_code)
    else:
        plt.scatter(pcs[:, 0], pcs[:, 1], s=40, alpha=0.7)
    for i, s in enumerate(prot_data_c.columns):
        plt.annotate(s, (pcs[i, 0], pcs[i, 1]), fontsize=6)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.title('PCA of tumor samples'); plt.tight_layout(); plt.show()

    # ── 7. Missing value heatmap (gene x sample) ─────────────────────────────────
    missing_matrix = prot_data_c.isnull().astype(int)
    # plot top 50 most-missing genes for visibility
    top_missing = missing_matrix.sum(axis=1).nlargest(50).index
    sns.heatmap(missing_matrix.loc[top_missing], cmap='Greys',
                yticklabels=True, xticklabels=False)
    plt.title('Missing value pattern (top 50 genes)'); plt.tight_layout(); plt.show()

    # ── 8. Summary statistics ─────────────────────────────────────────────────────
    print("Global median:", prot_data_c.stack().median().round(3))  # should be ~0
    print("Global IQR:", stats.iqr(prot_data_c.stack().dropna()))
    print("Proteins with >50% missing:", (prot_data_c.isnull().mean(axis=1) > 0.5).sum())
    print("Samples with >30% missing:", (prot_data_c.isnull().mean(axis=0) > 0.3).sum())
    return None


#Ensembl to HGNC conversion
def check_gtf_version(gtf_path: str):
    """Print genome assembly and Ensembl release from GTF header."""
    import gzip
    open_func = gzip.open if gtf_path.endswith(".gz") else open
    with open_func(gtf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                print(line.strip())  # header lines contain assembly + release info
            else:
                break  # stop at first data line
                
def strip_version(ensembl_id: str) -> str:
    """Remove version suffix: ENSG00000141510.18 -> ENSG00000141510"""
    return ensembl_id.split(".")[0]


def load_gtf(gtf_path: str) -> pd.DataFrame:
    """
    Parse Ensembl GTF file into a lookup DataFrame.
    Returns columns: ensembl_id, hgnc_symbol, biotype
    Call once and reuse — no need to reload on every conversion.
    """
    print(f"Loading GTF from {gtf_path}...")
    rows = []
    open_func = gzip.open if gtf_path.endswith(".gz") else open

    with open_func(gtf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            if fields[2] != "gene":  # gene-level entries only
                continue

            attrs = {}
            for attr in fields[8].strip().split(";"):
                attr = attr.strip()
                if " " in attr:
                    key, val = attr.split(" ", 1)
                    attrs[key] = val.strip('"')

            rows.append({
                "ensembl_id":  attrs.get("gene_id", "").split(".")[0],
                "hgnc_symbol": attrs.get("gene_name"),
                "biotype":     attrs.get("gene_biotype"),
            })

    df = pd.DataFrame(rows).drop_duplicates(subset="ensembl_id")
    print(f"GTF loaded: {len(df)} genes, assembly info above.")
    return df


def _ensembl_rest_fallback(ensembl_ids: list[str], gtf_lookup: dict = None) -> dict:
    """
    Falls back first to local GTF lookup if available,
    then to Ensembl REST API with retries for anything still unmapped.
    """
    lookup = {}
    still_unmapped = []

    # ── Step 1: try GTF first ─────────────────────────────────────────────────
    if gtf_lookup:
        for eid in ensembl_ids:
            info = gtf_lookup.get(eid)
            if info and pd.notna(info.get("hgnc_symbol")):
                lookup[eid] = {
                    "hgnc_symbol": info["hgnc_symbol"],
                    "entrez_id":   None,
                    "biotype":     info.get("biotype"),
                }
            else:
                still_unmapped.append(eid)
        print(f"  GTF resolved {len(lookup)}/{len(ensembl_ids)} fallback IDs. "
              f"{len(still_unmapped)} still unmapped → trying REST API...")
    else:
        still_unmapped = ensembl_ids

    # ── Step 2: REST API with retries for anything GTF couldn't resolve ───────
    if still_unmapped:
        batch_size  = 200
        base_url    = "https://rest.ensembl.org/lookup/id"
        headers     = {"Content-Type": "application/json", "Accept": "application/json"}
        max_retries = 3

        for i in range(0, len(still_unmapped), batch_size):
            batch = still_unmapped[i : i + batch_size]

            for attempt in range(max_retries):
                try:
                    r = requests.post(base_url, headers=headers,
                                      json={"ids": batch}, timeout=60)
                    r.raise_for_status()
                    for eid, data in r.json().items():
                        if data and data.get("display_name"):
                            lookup[eid] = {
                                "hgnc_symbol": data["display_name"],
                                "entrez_id":   None,
                                "biotype":     data.get("biotype"),
                            }
                    break  # success

                except requests.RequestException as e:
                    wait = 2 ** attempt
                    if attempt < max_retries - 1:
                        print(f"  Batch {i}: attempt {attempt+1} failed ({e}). "
                              f"Retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        print(f"  Batch {i}: all {max_retries} attempts failed. Skipping.")
            time.sleep(0.5)

    return lookup


def get_ensembl_biotypes(ensembl_ids: list[str], gtf_lookup: dict = None) -> dict:
    """
    Get biotype for each Ensembl ID.
    Uses local GTF if available, falls back to REST API.
    """
    biotype_lookup = {}
    still_needed   = []

    # ── Try GTF first ─────────────────────────────────────────────────────────
    if gtf_lookup:
        for eid in ensembl_ids:
            info = gtf_lookup.get(eid)
            if info and info.get("biotype"):
                biotype_lookup[eid] = info["biotype"]
            else:
                still_needed.append(eid)
        print(f"  GTF resolved {len(biotype_lookup)} biotypes. "
              f"{len(still_needed)} still needed → trying REST API...")
    else:
        still_needed = ensembl_ids

    # ── REST fallback for remaining ───────────────────────────────────────────
    if still_needed:
        batch_size  = 200
        base_url    = "https://rest.ensembl.org/lookup/id"
        headers     = {"Content-Type": "application/json", "Accept": "application/json"}
        max_retries = 3

        for i in range(0, len(still_needed), batch_size):
            batch = still_needed[i : i + batch_size]

            for attempt in range(max_retries):
                try:
                    r = requests.post(base_url, headers=headers,
                                      json={"ids": batch}, timeout=60)
                    r.raise_for_status()
                    for eid, data in r.json().items():
                        if data:
                            biotype_lookup[eid] = data.get("biotype")
                    break

                except requests.RequestException as e:
                    wait = 2 ** attempt
                    if attempt < max_retries - 1:
                        print(f"  Biotype batch {i}: attempt {attempt+1} failed. "
                              f"Retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        print(f"  Biotype batch {i}: all attempts failed. Skipping.")
            time.sleep(0.5)

    return biotype_lookup


def convert_ensembl_to_hgnc(
    ensembl_ids: list[str],
    get_biotype: bool = True,
    gtf_path:    str  = None,   # ← pass your GTF path here
) -> pd.DataFrame:
    
    """
    Convert Ensembl IDs to HGNC symbols.
    Priority: mygene → local GTF → Ensembl REST API
    Biotypes:           local GTF → Ensembl REST API
    """
    
    # ── Load GTF once if provided ─────────────────────────────────────────────
    gtf_lookup = None
    if gtf_path:
        check_gtf_version(gtf_path)  # print assembly info for traceability
        gtf_df     = load_gtf(gtf_path)
        gtf_lookup = gtf_df.set_index("ensembl_id").to_dict(orient="index")

    # ── Strip versions ────────────────────────────────────────────────────────
    id_map    = {eid: strip_version(eid) for eid in ensembl_ids}
    clean_ids = list(set(id_map.values()))

    # ── mygene (primary) ──────────────────────────────────────────────────────
    print(f"Querying mygene for {len(clean_ids)} Ensembl IDs...")
    mg      = mygene.MyGeneInfo()
    results = mg.querymany(
        clean_ids,
        scopes="ensembl.gene",
        fields=["symbol", "entrezgene", "ensembl.gene"],
        species="human",
        returnall=True,
    )
    mygene_lookup = {}
    for hit in results["out"]:
        eid    = hit.get("query")
        symbol = hit.get("symbol")
        entrez = hit.get("entrezgene")
        if symbol and eid:
            mygene_lookup[eid] = {
                "hgnc_symbol": symbol,
                "entrez_id":   str(entrez) if entrez else None,
            }

    # ── GTF + REST fallback for unmapped ──────────────────────────────────────
    unmapped      = [eid for eid in clean_ids if eid not in mygene_lookup]
    ensembl_lookup = {}
    if unmapped:
        print(f"Falling back for {len(unmapped)} unmapped IDs...")
        ensembl_lookup = _ensembl_rest_fallback(unmapped, gtf_lookup=gtf_lookup)

    # ── Biotypes ──────────────────────────────────────────────────────────────
    biotype_lookup = {}
    if get_biotype:
        print("Fetching biotype information...")
        biotype_lookup = get_ensembl_biotypes(clean_ids, gtf_lookup=gtf_lookup)

    # ── Build final DataFrame ─────────────────────────────────────────────────
    rows = []
    for original_id in ensembl_ids:
        clean = id_map[original_id]
        info  = mygene_lookup.get(clean) or ensembl_lookup.get(clean, {})
        rows.append({
            "original_id": original_id,
            "ensembl_id":  clean,
            "hgnc_symbol": info.get("hgnc_symbol"),
            "entrez_id":   info.get("entrez_id"),
            "biotype":     biotype_lookup.get(clean),
        })

    df     = pd.DataFrame(rows)
    mapped = df["hgnc_symbol"].notna().sum()
    print(f"\nMapping complete: {mapped}/{len(df)} genes mapped ({mapped/len(df)*100:.1f}%)")
    return df

def keep_highest_expressed_isoform(data: pd.DataFrame) -> pd.DataFrame:
    """
    A function to remove duplicates (multiple isoforms of a given gene) by keeping only the most expressed row.
    """
    data = data.drop("biotype", axis=1, errors="ignore")
    
    # Compute mean across samples
    data = data.copy()
    data["_mean"] = data.mean(axis=1)
    
    # Reset index to make gene names a regular column – positional indexing safe use
    data = data.reset_index()
    
    # Within each group of duplicates, keep the row with highest mean
    data = data.loc[data.groupby("idx")["_mean"].idxmax()]
    
    # Restore gene names as index and drop helper column
    data = data.set_index("idx").drop(columns="_mean") 
    
    return data