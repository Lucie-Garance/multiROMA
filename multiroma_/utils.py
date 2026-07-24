import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import anndata as ad
import mudata
mudata.set_options(pull_on_update=False)
import muon as mu

# printing class 
class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def integrate_projection_results(global_adata, cell_type, gene_set_results, ct_label):
    import pandas as pd

    # Boolean mask for cells of the specified cell type.
    subset_mask = global_adata.obs[ct_label] == cell_type
    # Get the indices (order preserved) for these cells.
    subset_indices = global_adata.obs.index[subset_mask]
    if not 'pyroma_scores' in global_adata.uns:
        global_adata.uns['pyroma_scores'] = pd.DataFrame(index=global_adata.obs.index)

    for pathway, result in gene_set_results.items():
        # Get computed projection values for this pathway
        projections = result.svd.components_[0]
        if len(projections) != len(subset_indices):
            raise ValueError(
                f"Length mismatch for cell type '{cell_type}' and pathway '{pathway}': "
                f"found {len(subset_indices)} cells in adata vs. {len(projections)} projection values."
            )
        # Create a new Series (one value per cell in global_adata) initialized to 0.0
        new_col = pd.Series(0.0, index=global_adata.obs.index)
        # Fill in the computed projection values for the subset cells (order preserved)
        new_col.loc[subset_indices] = projections
        
        # Create the new column name as "{cell_type}_{pathway}"
        col_name = f"{cell_type}|{pathway}"
        global_adata.uns['pyroma_scores'][col_name] = new_col
    
    print(f"Projection columns for cell type '{cell_type}' integrated successfully.")


def multiindex_df_to_mudata(df, feature_level=0, name_level=1):
    """
    Convert a MultiIndex column DataFrame (level 0 = modality/feature type,
    level 1 = feature name) back to a MuData object.
    """
    
    if type(df)!=pd.DataFrame:
        raise ValueError("Input has to be a dataframe")
    if type(feature_level)!= int:
        raise ValueError("Feature level should be an integer: it represents the indexing level of the feature")
    if type(name_level)!= int:
        raise ValueError("Name level should be an integer: it represents the indexing level of the names")
        
    mod_dict = {}

    # get unique modalities from level 0, preserving order of first appearance
    modalities = df.columns.get_level_values(feature_level).unique()

    for mod_name in modalities:
        # subset columns belonging to this modality
        sub_df = df.loc[:, df.columns.get_level_values(feature_level) == mod_name]

        # drop the modality level, keep only the feature-name level as columns
        sub_df.columns = sub_df.columns.get_level_values(name_level)

        X = sub_df.values

        var = pd.DataFrame(index=sub_df.columns)
        var.index.name = None

        obs = pd.DataFrame({"celltype": ["no_data" for _ in sub_df.index]}, index=sub_df.index)

        adata = sc.AnnData(X=X, obs=obs, var=var)
        mod_dict[mod_name] = adata

    mdata = mudata.MuData(mod_dict)
    return mdata

def multiindex_df_to_mudata_extended(df, 
                                     feature_level=0, 
                                     name_level=1, 
                                     init_annot=None,
                                     prefixes=["rs", "pp"]):
    """
    Convert a MultiIndex column DataFrame (level 0 = modality/feature type,
    level 1 = feature name) back to a MuData object.
    """
    
    if type(df)!=pd.DataFrame:
        raise ValueError("Input has to be a dataframe")
    if type(feature_level)!= int:
        raise ValueError("Feature level should be an integer: it represents the indexing level of the feature")
    if type(name_level)!= int:
        raise ValueError("Name level should be an integer: it represents the indexing level of the names")
        
    mod_dict = {}

    # get unique modalities from level 0, preserving order of first appearance
    modalities = df.columns.get_level_values(feature_level).unique()
    c=0

    for mod_name in modalities:
        prefix=prefixes[c]
        c+=1
        
        # subset columns belonging to this modality
        sub_df = df.loc[:, df.columns.get_level_values(feature_level) == mod_name]

        # drop the modality level, keep only the feature-name level as columns
        sub_df.columns = sub_df.columns.get_level_values(name_level)
        X = sub_df.values
        
        if init_annot==None:
            var = pd.DataFrame(index=sub_df.columns)
            var.index.name = None
            obs = pd.DataFrame({"celltype": ["no_data" for _ in sub_df.index]}, index=sub_df.index)
            adata = sc.AnnData(X=X, obs=obs, var=var)
            
        else:
            common_index = sub_df.index
            pref_var=init_annot[mod_name].var.rename(index=lambda c: f"{prefix}_{c}")
            #import pdb
            #pdb.set_trace()
            var = pref_var.loc[sub_df.columns.to_list()]
            obs = init_annot[mod_name].obs.loc[common_index]
            uns = init_annot[mod_name].uns
            obsm = init_annot[mod_name].obsm
            adata = sc.AnnData(X=X, obs=obs, var=var, uns=uns, obsm=obsm)

        mod_dict[mod_name] = adata
        
    mdata = mudata.MuData(mod_dict)
    return mdata

def mudata_to_multiindex_df(mdata, feature_level_name="modality", name_level_name="feature"):
    """
    Convert a MuData object back to a single DataFrame with MultiIndex columns
    (level 0 = modality, level 1 = feature name).

    Inverse of multiindex_df_to_mudata.

    Parameters
    ----------
    mdata : mu.MuData
    feature_level_name : str
        Name to give the level-0 (modality) level of the resulting MultiIndex.
    name_level_name : str
        Name to give the level-1 (feature name) level of the resulting MultiIndex.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with MultiIndex columns (modality, feature), and rows
        indexed by sample/obs names.
    """

    if type(mdata)!=mu.MuData:
        raise ValueError("Input has to be a mu.MuData object")
    if type(feature_level_name)!= str:
        raise ValueError("Feature level name should be an str: it is the name of the feature level")
    if type(name_level_name)!= str:
        raise ValueError("Name level name should be an str: it is the name of the names level")
        
    dfs = []

    for mod_name, adata in mdata.mod.items():
        X = adata.X
        if sp.issparse(X):
            X = X.toarray()

        columns = pd.MultiIndex.from_arrays(
            [[mod_name] * adata.n_vars, adata.var_names.values],
            names=[feature_level_name, name_level_name]
        )

        sub_df = pd.DataFrame(X, index=adata.obs_names, columns=columns)
        dfs.append(sub_df)

    df = pd.concat(dfs, axis=1)
    return df

def _columns_agree(series_list):
    """
    Check whether a list of pandas Series (all aligned to the same index)
    agree everywhere more than one of them is non-null.
 
    Returns True if there is no conflicting value between any pair of
    series wherever both have data, meaning they can be safely merged
    into a single shared column.
    """
    base = series_list[0]
    for other in series_list[1:]:
        overlap = base.notna() & other.notna()
        if overlap.any():
            a = base[overlap].astype(object)
            b = other[overlap].astype(object)
            if not (a == b).all():
                return False
        # accumulate values seen so far so 3+ modality comparisons are
        # checked against the union of what's been merged, not just the
        # first modality's values
        base = base.combine_first(other)
    return True

def mudata_to_flat_anndata_v2(mdata, feature_col="modality"):
    """
    Flatten a MuData object into a single AnnData, with modality stored
    as a column in .var (instead of column MultiIndex or separate .mod entries).

    Preserves per-modality obs and var annotations:
    - obs: each modality's own .obs columns are merged into the global obs,
      prefixed with the modality name to avoid collisions. Samples not present
      in a given modality get NaN for that modality's columns.
    - var: each modality's own .var columns are concatenated (stacked) into
      the final var, alongside feature_col. Columns not present in a given
      modality's var are filled with NaN for those rows.

    Assumes all modalities share the same, identically-ordered obs
    (use check_shared_ordered_obs() beforehand to confirm).

    Parameters
    ----------
    mdata : mu.MuData
    feature_col : str
        Name of the .var column that will store the modality/feature type.

    Returns
    -------
    adata : sc.AnnData
        Single AnnData with X = horizontal concatenation of all modalities' X,
        var[feature_col] indicating modality of origin, and obs/var extended
        with each modality's own annotations.
    """
    
    if type(mdata)!=mu.MuData:
        raise ValueError("Input has to be a mu.MuData object")
    if type(feature_col)!= str:
        raise ValueError("Feature col should be an str: it is the name of the feature column ")
        
    blocks = []
    var_blocks = []
    any_sparse = False

    # --- start from global obs (shared annotations across modalities) ---
    obs = mdata.obs.copy() if mdata.obs.shape[1] > 0 else pd.DataFrame(index=mdata.obs_names)

    for mod_name, mod_adata in mdata.mod.items():
        X = mod_adata.X
        if sp.issparse(X):
            any_sparse = True
        blocks.append(X)

        # --- build this modality's var block, preserving its own columns ---
        mod_var = mod_adata.var.copy()
        mod_var[feature_col] = mod_name
        var_blocks.append(mod_var)

        # --- merge this modality's obs columns into the global obs ---
        if mod_adata.obs.shape[1] > 0:
            mod_obs = mod_adata.obs.copy()
            # prefix columns to avoid collisions between modalities
            mod_obs.columns = [f"{mod_name}:{c}" for c in mod_obs.columns]
            # reindex to the full obs index -> NaN for samples missing from this modality
            mod_obs = mod_obs.reindex(mdata.obs_names)
            obs = obs.join(mod_obs, how="left")

    # --- concatenate X blocks ---
    if any_sparse:
        blocks = [b if sp.issparse(b) else sp.csr_matrix(b) for b in blocks]
        X_full = sp.hstack(blocks).tocsr()
    else:
        X_full = np.hstack(blocks)

    # --- concatenate var blocks (fills NaN for columns missing in a given modality) ---
    var = pd.concat(var_blocks, axis=0)
    var.index.name = None

    # move feature_col to the front for readability (optional)
    cols = [feature_col] + [c for c in var.columns if c != feature_col]
    var = var[cols]

    adata = sc.AnnData(X=X_full, obs=obs, var=var)
    return adata

def mudata_to_flat_anndata_v3(mdata, feature_col="modality"):
    """
    Flatten a MuData object into a single AnnData, with modality stored
    as a column in .var (instead of column MultiIndex or separate .mod entries).
 
    Preserves per-modality obs and var annotations:
    - obs: each modality's own .obs columns are merged into the global obs.
      If a column name is shared by more than one modality AND the modalities
      agree everywhere they overlap (no conflicting values), it is merged
      into a single shared column (no prefix). Otherwise -- i.e. the column
      is modality-specific, or shared modalities disagree -- it is kept
      per-modality, prefixed with the modality name to avoid collisions.
      Samples not present in a given modality get NaN for that modality's
      (prefixed) columns.
    - var: each modality's own .var columns are concatenated (stacked) into
      the final var, alongside feature_col. Columns sharing the same name
      across modalities are naturally aligned/merged by the row-stacking
      concat (no separate handling needed). Columns not present in a given
      modality's var are filled with NaN for those rows.
 
    Assumes all modalities share the same, identically-ordered obs
    (use check_shared_ordered_obs() beforehand to confirm).
 
    Parameters
    ----------
    mdata : mu.MuData
    feature_col : str
        Name of the .var column that will store the modality/feature type.
 
    Returns
    -------
    adata : sc.AnnData
        Single AnnData with X = horizontal concatenation of all modalities' X,
        var[feature_col] indicating modality of origin, and obs/var extended
        with each modality's own annotations.
    """
 
    if type(mdata) != mu.MuData:
        raise ValueError("Input has to be a mu.MuData object")
    if type(feature_col) != str:
        raise ValueError("Feature col should be an str: it is the name of the feature column ")
 
    blocks = []
    var_blocks = []
    any_sparse = False
 
    # --- start from global obs (shared annotations across modalities) ---
    obs = mdata.obs.copy() if mdata.obs.shape[1] > 0 else pd.DataFrame(index=mdata.obs_names)
 
    # --- collect each modality's obs, reindexed to the full obs index, unprefixed ---
    per_mod_obs = {}
    for mod_name, mod_adata in mdata.mod.items():
        if mod_adata.obs.shape[1] > 0:
            per_mod_obs[mod_name] = mod_adata.obs.reindex(mdata.obs_names)
 
    # --- find column names shared by more than one modality ---
    col_to_mods = {}
    for mod_name, mod_obs in per_mod_obs.items():
        for c in mod_obs.columns:
            col_to_mods.setdefault(c, []).append(mod_name)
 
    # --- for shared columns that agree everywhere they overlap, merge into one column ---
    merged_cols = {}
    for col, mods_with_col in col_to_mods.items():
        if len(mods_with_col) < 2:
            continue  # present in only one modality -> handled via prefix below
        series_list = [per_mod_obs[m][col] for m in mods_with_col]
        if _columns_agree(series_list):
            merged = series_list[0]
            for s in series_list[1:]:
                merged = merged.combine_first(s)
            merged_cols[col] = merged
 
    for col, merged_series in merged_cols.items():
        obs[col] = merged_series
 
    for mod_name, mod_adata in mdata.mod.items():
        X = mod_adata.X
        if sp.issparse(X):
            any_sparse = True
        blocks.append(X)
 
        # --- build this modality's var block, preserving its own columns ---
        mod_var = mod_adata.var.copy()
        mod_var[feature_col] = mod_name
        var_blocks.append(mod_var)
 
        # --- merge this modality's remaining (non-shared / conflicting) obs columns ---
        if mod_name in per_mod_obs:
            mod_obs = per_mod_obs[mod_name]
            cols_to_prefix = [c for c in mod_obs.columns if c not in merged_cols]
            if cols_to_prefix:
                mod_obs_sub = mod_obs[cols_to_prefix].copy()
                mod_obs_sub.columns = [f"{mod_name}:{c}" for c in cols_to_prefix]
                obs = obs.join(mod_obs_sub, how="left")
 
    # --- concatenate X blocks ---
    if any_sparse:
        blocks = [b if sp.issparse(b) else sp.csr_matrix(b) for b in blocks]
        X_full = sp.hstack(blocks).tocsr()
    else:
        X_full = np.hstack(blocks)
 
    # --- concatenate var blocks (fills NaN for columns missing in a given modality) ---
    var = pd.concat(var_blocks, axis=0)
    var.index.name = None
    # move feature_col to the front for readability (optional)
    cols = [feature_col] + [c for c in var.columns if c != feature_col]
    var = var[cols]
 
    adata = sc.AnnData(X=X_full, obs=obs, var=var)
    return adata

def mudata_to_flat_anndata(mdata, feature_col="modality"):
    """
    Flatten a MuData object into a single AnnData, with modality stored
    as a column in .var (instead of column MultiIndex or separate .mod entries).
    
    Assumes all modalities share the same, identically-ordered obs
    (use check_shared_ordered_obs() beforehand to confirm).

    Parameters
    ----------
    mdata : mu.MuData
    feature_col : str
        Name of the .var column that will store the modality/feature type.

    Returns
    -------
    adata : sc.AnnData
        Single AnnData with X = horizontal concatenation of all modalities' X,
        and var[feature_col] indicating which modality each feature came from.
    """

    if type(mdata)!=mu.MuData:
        raise ValueError("Input has to be a mu.MuData object")
    if type(feature_col)!= str:
        raise ValueError("Feature col should be an str: it is the name of the feature column ")
        
    blocks = []
    var_names = []
    modality_labels = []
    any_sparse = False

    for mod_name, mod_adata in mdata.mod.items():
        X = mod_adata.X
        if sp.issparse(X):
            any_sparse = True
        blocks.append(X)
        var_names.extend(list(mod_adata.var_names))
        modality_labels.extend([mod_name] * mod_adata.n_vars)

    if any_sparse:
        blocks = [b if sp.issparse(b) else sp.csr_matrix(b) for b in blocks]
        X_full = sp.hstack(blocks).tocsr()
    else:
        X_full = np.hstack(blocks)

    var = pd.DataFrame({feature_col: modality_labels}, index=var_names)
    var.index.name = None
    
    obs = mdata.obs.copy() if mdata.obs.shape[1] > 0 else pd.DataFrame(index=mdata.obs_names)
    

    adata = sc.AnnData(X=X_full, obs=obs, var=var)
    return adata

def flat_anndata_to_mudata(adata, feature_col="modality"):
    """
    Un-flatten a single AnnData (with modality stored in .var[feature_col])
    back into a MuData object with one AnnData per modality.

    Inverse of mudata_to_flat_anndata.

    Parameters
    ----------
    adata : sc.AnnData
        Flattened AnnData, with var[feature_col] indicating each feature's
        modality of origin.
    feature_col : str
        Name of the .var column holding the modality label.

    Returns
    -------
    mdata : mu.MuData
        Reconstructed MuData object, with one AnnData per modality,
        var[feature_col] dropped from each modality's own .var.
    """
    
    if type(adata)!=sc.AnnData:
        raise ValueError("Input has to be a sc.AnnData object")
    if type(feature_col)!= str:
        raise ValueError("Feature col should be an str: it is the name of the feature column to extract the dataframe level 0 index ")
        
    mod_dict = {}

    modalities = adata.var[feature_col].unique()

    for mod_name in modalities:
        mask = (adata.var[feature_col] == mod_name).values
        X_sub = adata.X[:, mask]

        var_sub = adata.var.loc[mask].drop(columns=[feature_col])
        obs_sub = pd.DataFrame(index=adata.obs_names)

        mod_dict[mod_name] = sc.AnnData(X=X_sub, obs=obs_sub, var=var_sub)

    mdata = mu.MuData(mod_dict)
    mdata.obs = adata.obs.copy()  # restore any shared/global obs annotations

    return mdata

def flat_anndata_to_mudata_v2(adata, feature_col="modality"):
    """
    Revert mudata_to_flat_anndata_v2(): reconstruct a MuData object from a
    single flattened AnnData.

    Reverses both preservation steps performed by the forward function:
    - var: splits var rows by `feature_col`, restoring each modality's own
      .var columns (columns that are entirely NaN within a modality's block
      are dropped, since those were NaN-filled placeholders introduced by
      the stacking/concat step for columns that belong to other modalities).
    - obs: obs columns prefixed with "<modality>:" are peeled off, the
      prefix is stripped, and they are assigned back to that modality's
      .obs. Any remaining (non-prefixed) obs columns are treated as the
      original shared/global obs and are restored onto mdata.obs.

    Assumes the flattened AnnData was produced by mudata_to_flat_anndata_v2
    (in particular, that all modalities shared the same, identically
    ordered obs, and that global obs column names never collide with the
    "<modality>:" prefix pattern).

    Parameters
    ----------
    adata : ad.AnnData
        Flattened AnnData produced by mudata_to_flat_anndata_v2.
    feature_col : str
        Name of the .var column that stores the modality/feature type.

    Returns
    -------
    mdata : mu.MuData
        Reconstructed MuData object with per-modality obs/var restored,
        plus the shared global obs.
    """

    if type(adata) != ad.AnnData:
        raise ValueError("Input has to be an ad.AnnData object")
    if type(feature_col) != str:
        raise ValueError(
            "feature_col should be a str: it is the name of the feature column"
        )
    if feature_col not in adata.var.columns:
        raise ValueError(f"'{feature_col}' not found in adata.var columns")

    # preserve order of first appearance of each modality
    mod_names = list(pd.unique(adata.var[feature_col]))

    mod_adatas = {}
    all_prefixed_obs_cols = set()

    for mod_name in mod_names:
        mask = (adata.var[feature_col] == mod_name).values

        # --- split X back out for this modality ---
        X_sub = adata.X[:, mask]
        if sp.issparse(X_sub):
            X_sub = X_sub.tocsr()

        # --- recover this modality's own var columns ---
        var_sub = adata.var.loc[mask].drop(columns=[feature_col]).copy()
        all_nan_cols = [c for c in var_sub.columns if var_sub[c].isna().all()]
        var_sub = var_sub.drop(columns=all_nan_cols)

        # --- recover this modality's own obs columns ---
        prefix = f"{mod_name}:"
        mod_obs_cols = [c for c in adata.obs.columns if c.startswith(prefix)]
        all_prefixed_obs_cols.update(mod_obs_cols)
        mod_obs = adata.obs[mod_obs_cols].copy()
        mod_obs.columns = [c[len(prefix):] for c in mod_obs.columns]

        mod_adatas[mod_name] = sc.AnnData(X=X_sub, obs=mod_obs, var=var_sub)

    # --- anything not claimed by a modality prefix is the shared/global obs ---
    global_obs_cols = [c for c in adata.obs.columns if c not in all_prefixed_obs_cols]
    global_obs = adata.obs[global_obs_cols].copy()

    mdata = mu.MuData(mod_adatas)

    # MuData recomputes .obs from the modalities on construction, so restore
    # the shared/global obs columns afterwards.
    for c in global_obs.columns:
        mdata.obs[c] = global_obs[c]

    return mdata

def anndata_to_multiindex_df(adata, feature_col="modality"):
    """
    Convert AnnData to a pandas DataFrame with column MultiIndex
    where level 0 = feature type and level 1 = feature name.
    """

    if type(adata)!=sc.AnnData:
        raise ValueError("Input has to be a sc.AnnData object")
    if type(feature_col)!= str:
        raise ValueError("Feature col should be an str: it is the name of the feature column to extract the dataframe level 0 index ")
        
    # build MultiIndex
    columns = pd.MultiIndex.from_arrays(
        [adata.var[feature_col].values, adata.var_names.values],
        names=[feature_col, "feature"]
    )

    X = adata.X

    # sparse case (preferred for large data)
    if sp.issparse(X):
        df = pd.DataFrame.sparse.from_spmatrix(
            X,
            index=adata.obs_names,
            columns=columns
        )
    else:
        df = pd.DataFrame(
            X,
            index=adata.obs_names,
            columns=columns
        )

    return df

def multiindex_df_to_anndata(df, feature_level=0, name_level=1):
    """
    Convert MultiIndex column dataframe back to AnnData.
    """

    if type(df)!=pd.DataFrame:
        raise ValueError("Input has to be a dataframe")
    if type(feature_level)!= int:
        raise ValueError("Feature level should be an integer: it represents the indexing level of the feature")
    if type(name_level)!= int:
        raise ValueError("Name level should be an integer: it represents the indexing level of the names")

    # flatten matrix
    X = df.values

    # rebuild var dataframe
    var = pd.DataFrame({
        "feature type": df.columns.get_level_values(feature_level),
        "gene_ids": df.columns.get_level_values(name_level)
    }, index=df.columns.get_level_values(name_level))

    var.index.name = None

    # build obs dataframe
    obs = pd.DataFrame({"celltype": ["no_data" for _ in df.index]})

    adata = sc.AnnData(X=X, obs=obs, var=var)

    return adata