## Choosing genes for active pathways
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scanpy as sc
from utils import mudata_to_flat_anndata_v3

def hgnc_list_to_expression_df(adata, hgnc_genes, gene_col="gene_name", sample_col="orig.ident"):
    """
    Given a list of HGNC gene symbols, extract expression values from an AnnData
    object (matching against messy/clone-style names in adata.var[gene_col]),
    and return a samples x genes DataFrame labeled with the clean HGNC symbols.

    Parameters
    ----------
    adata : sc.AnnData
    hgnc_genes : list of str
        Genes of interest, in HGNC symbol form.
    gene_col : str
        Column in adata.var holding the gene names as they appear in the dataset.
    sample_col : str
        Column in adata.obs holding sample IDs.

    Returns
    -------
    expr_df : pd.DataFrame
        Index = sample IDs, columns = HGNC gene symbols (only those successfully matched).
    report : dict
        Diagnostic info: which genes were matched directly, via alias, or not found at all.
    """
    var_names_in_data = adata.var[gene_col].astype(str)
    hgnc_targets = list(dict.fromkeys(hgnc_genes))  # dedupe, preserve order
    hgnc_set = set(hgnc_targets)

    # --- Step 1: direct matches ---
    direct_matches = {g: g for g in hgnc_targets if g in set(var_names_in_data)}
    unmatched = [g for g in hgnc_targets if g not in direct_matches]

    # --- Step 2: alias-based matches for the leftovers ---
    alias_matches = {}
    still_missing = []

    if unmatched:
        mg = mygene.MyGeneInfo()
        results = mg.querymany(
            unmatched, scopes="symbol", fields="symbol,alias,other_names",
            species="human", verbose=False
        )

        var_names_set = set(var_names_in_data)
        for r in results:
            target = r.get("query")
            if target in direct_matches:
                continue  # shouldn't happen, but just in case

            aliases = r.get("alias", [])
            if isinstance(aliases, str):
                aliases = [aliases]
            other_names = r.get("other_names", [])
            if isinstance(other_names, str):
                other_names = [other_names]

            candidates = set(aliases) | set(other_names)
            found = candidates & var_names_set

            if found:
                # if multiple aliases match, just take the first (report will flag if >1)
                chosen = sorted(found)[0]
                alias_matches[target] = chosen
                if len(found) > 1:
                    print(f"Note: '{target}' matched multiple aliases in data {found}, using '{chosen}'")
            else:
                still_missing.append(target)

    # --- Step 3: build final mapping HGNC -> actual name in adata ---
    final_mapping = {**direct_matches, **alias_matches}  # target_hgnc -> name_in_data

    report = {
        "direct_matches": list(direct_matches.keys()),
        "alias_matches": alias_matches,
        "not_found": still_missing,
    }

    if not final_mapping:
        raise ValueError("None of the requested HGNC genes could be matched in adata.")

    # --- Step 4: subset adata using the actual in-data names ---
    names_to_keep = list(final_mapping.values())
    mask = var_names_in_data.isin(names_to_keep).values
    adata_sub = adata[:, mask].copy()

    # --- Step 5: build expression matrix ---
    X = adata_sub.X
    if sp.issparse(X):
        X = X.toarray()

    # map each column back to its HGNC label (in the order columns appear post-subset)
    sub_var_names = adata_sub.var[gene_col].astype(str).values
    name_to_hgnc = {v: k for k, v in final_mapping.items()}
    column_labels = [name_to_hgnc[n] for n in sub_var_names]

    expr_df = pd.DataFrame(
        X,
        index=adata_sub.obs.index,
        columns=column_labels
    )

    # reorder columns to match the original hgnc_genes order (only those found)
    ordered_cols = [g for g in hgnc_targets if g in expr_df.columns]
    expr_df = expr_df[ordered_cols]

    return expr_df, report

def gene_selection(active_modules, top_gene=3, thr_z=5, gap_thr=1,  min_genes_per_pathway=1, verbose=False): 
    """
    Function that performs the gene selection
    --------------
    Inputs:
        active_modules: the list of active gene sets
        top_gene: number of best genes
        thr_z: threshold for the corrected z-score of the genes
        gap_thr: threshold for gap in the corrected z-score (keeping only the genes well separated from the set)
        min_genes_per_pathway: minimal number of genes kept per pathway
        
    """
    genes_of_interest=[]
    for geneset in active_modules.index.tolist():
    
        idx= np.load(f'results_batch_aware_pyroma_no_celltype/{geneset}/subsetlist.npy')
        list_out=np.load(f'results_batch_aware_pyroma_no_celltype/{geneset}/outliers', allow_pickle=True)
        idx_no_out=np.delete(idx, list_out, axis=0)
    
        df = pd.DataFrame(
            np.load(f'results_batch_aware_pyroma_no_celltype/{geneset}/projections_1.npy'),
            index=idx_no_out,
            columns=[f'{geneset}']
        )
        
        ## Gene selection
        scores= df[f"{geneset}"]

        median = scores.median()
        q1, q3 = scores.quantile([0.25, 0.75])
        iqr = q3 - q1

        robust_z = (scores - median) / iqr
    
        sorted_z = robust_z.sort_values(ascending=False)
        top_genes = sorted_z[sorted_z>thr_z].head(top_gene)
        if len(top_genes)< min_genes_per_pathway:
            top_genes=sorted_z.head(min_genes_per_pathway)
        gaps_top_genes = top_genes.diff(-1).abs()

        # find the first place where the gap is NOT > 1 (i.e., where clustering starts)
        below_threshold = gaps_top_genes[gaps_top_genes <= gap_thr]

        if len(below_threshold) == 0:
            # all 5 are well separated from each other
            selected = top_genes
        else:
            # keep only the leading, well-separated block, up to (not including) the first small gap
            first_break_pos = gaps_top_genes.index.get_loc(below_threshold.index[0])
            selected = top_genes.iloc[:first_break_pos + 1]

        if verbose:
            print(selected)
        genes_of_interest+=list(selected.index)

    return list(set(genes_of_interest))

def state_order_generator(expr_df, metadata, discrete_variables, is_consequence, is_contextual):

    """
    Function creating the state order file of MIIC
    """

    groups={"metadata" : metadata.keys()}

    if discrete_variables=="metadata" and metadata is not None:
        discrete_variables=list(metadata.keys())
    else:
        discrete_variables=[]
    
    # other genes of interest to include in the analysis

    selected_genes = list(expr_df.columns)
    all_variables_selected = selected_genes + list(metadata.keys())

    # create a dataframe to store the order of the states for each variable
    # this is a configuration file that will be used by MIIC to order the states of each variable
    state_order_df = pd.DataFrame(columns=["var_names",
                                               "var_type",
                                                "levels_increasing_order",
                                                "group",
                                                "is_contextual",
                                                "is_consequence"])

    state_order_df["var_names"] = all_variables_selected
    state_order_df["var_type"] = [ 0 if var in discrete_variables else 1 for var in all_variables_selected]
    state_order_df["levels_increasing_order"] = [ metadata[var] if var in metadata else "" for var in all_variables_selected]

    for key,val in groups.items():
        state_order_df.loc[state_order_df["var_names"].isin(val), "group"] = key
    state_order_df["group"] = state_order_df["group"].fillna("gene")

    state_order_df["is_contextual"] = [ 1 if gene in is_contextual else 0 for gene in all_variables_selected]
    state_order_df["is_consequence"] = [ 1 if gene in is_consequence else 0 for gene in all_variables_selected]

    return state_order_df

def MIIC_input_generator(mudata, 
                         metadata_of_interest=["subtype", "celltype_major"], 
                         binarise=["subtype"], 
                         order={"subtype": None, "celltype_major":None},
                         filter_option={"celltype_major": "Cancer Epithelial"}, 
                         is_consequence=[],
                         is_contextual=[],
                         discrete_variables="metadata",
                         gene_selection=False, 
                         verbose=False,
                         save=False):

    """
    Function to create MIIC input files from roma results. 
    ---------------------
    inputs:
        mudata: the mudata object output from multiROMA
        metadata of interest: metadata to include as covariates, or used for filtering
        binarise: covariates to binarise
        order: the order of the str values (for example, "False,True", or "wt,cancer" – this allows MIIC to give a sign to the correlation
        filter_option: the metadata used for filtering the cells if necessary – dictionary with the variable as key and the value to filter as value
        is_consequence: list of variables that can only be consequence of the other in the network (i.e. the growth of the cell, or the result of a given test)
        is_contextual: list of variables that are associated with experimental context, and will not be a consequence of the other variables (i.e. having glucose in the input medium)
        discrete_variables: variables that are not continuous – by default all metadata
        gene_selection: do we use the top genes or the pathways scores
        save: whether the final files should be saved.

    ----------------------
    
    Typical call for the simulation dataset : 
    from multiroma_V3.pyroma_.MIIC_input_generator import MIIC_input_generator
                    state_order_df, expr_df_filtered = MIIC_input_generator(roma.mudata, 
                         metadata_of_interest=["cell_type"], #["subtype", "celltype_major"], 
                         binarise=[], #["subtype"], 
                         order={"cell_type":None}, #{"subtype": None, "celltype_major":None},
                         filter_option= {"cell_type": "CellType_1"}, #{"celltype_major": "Cancer Epithelial"}, 
                         is_consequence=[],
                         is_contextual=[]`,
                         discrete_variables="metadata",
                         gene_selection=False, 
                         verbose=False,
                         save=False)

                        state_order_df
    
    """
    
    active_modules = mudata.uns["ROMA_active_modules"]
    adata=mudata_to_flat_anndata_v3(mudata)
    
    metadata={}
    metadata_order={}
    for key in metadata_of_interest:
        if key in binarise:
            df_val=pd.Series(adata.obs[key])
            types=pd.get_dummies(df_val)
            for col in types.columns():
                metadata[col]=types[col].astype(str)
                metadata_order[col]="False,True"
        else:
            metadata[key]=pd.Series(adata.obs[key])
            metadata_order[key]=order[key]
    
    if gene_selection:
        
        unique_genes_of_interest=gene_selection(active_modules)

        expr_df, report = hgnc_list_to_expression_df(adata, unique_genes_of_interest)
        
        if verbose:
            print("Nb genes of interest: ", len(unique_genes_of_interest))
            print(f"Matched directly: {len(report['direct_matches'])}")
            print(f"Matched via alias: {len(report['alias_matches'])}")
            print(f"Not found: {report['not_found']}")

    else:
        sample_scores= {active_path: mudata.uns["ROMA"][f"{active_path}"].svd.components_ for active_path in active_modules.index.tolist()}
        expr_df=pd.DataFrame({key: arr[0] for key, arr in sample_scores.items()})

    for key, vals in metadata.items():
        expr_df.index=vals.index
        expr_df[key]=list(vals)

    #filtering for a given cell type
    for name, filtering in filter_option.items():
        expr_df_filtered = expr_df[expr_df[name]==filtering]
        expr_df_filtered = expr_df_filtered.drop(name, axis=1)
        del metadata_order[name]

    state_order_df=state_order_generator(expr_df_filtered, metadata_order, discrete_variables, is_consequence, is_contextual)

    #Saving
    if save:
        expr_df_filtered.to_csv(f"MIIC_input_scBRCA_{name}_gene_selection_{gene_selection}.csv", index=False)
        state_order_df.to_csv(f"MIIC_state_order_scBRCA_{name}_gene_selection_{gene_selection}.tsv", index=False ,sep="\t")

    return state_order_df, expr_df_filtered