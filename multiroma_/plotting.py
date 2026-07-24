import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import sklearn
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

def Gene_score_projections(roma, geneset, top=None, save_file=False, file_name="gene_projection.png"):
    
    df= roma.mudata.uns["ROMA"][f"{geneset}"].projections_1.copy() * roma.correct_pc_sign
    df.index=df.index.droplevel(0)
    df=df.sort_values(key=abs, ascending=True)
    
    if top!=None:
        df=df.tail(top) #head
    
    fig, ax= plt.subplots(figsize=(8, len(df)*0.3))

    ax.hlines(y=range(len(df)),
              xmin=0,
              xmax=df, #[f"{geneset}"]
              color="grey",
              linewidth=0.8,
              linestyle="--",
              alpha=0.5)

    ax.scatter(x=df, #[f"{geneset}"]
               y=range(len(df)),
               s=40,
               zorder=2)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df.index, fontsize=8)
    ax.set_xlabel("Score per gene")
    ax.set_title(geneset)
    ax.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()

    if save_file:
        plt.savefig(file_name, bbox_inches="tight", dpi=150)

    plt.show()
    return None
    

def PC_samples_projection_GSVD(mudata, save_file=False, file_name="sample_projection.png"):

    if 'ROMA' not in mudata.uns:
        raise ValueError("No ROMA results found. Run compute() first.")
        
    plt.figure(1, figsize=(15, 6))
    
    pathways=list(mudata.uns['ROMA'].keys())
    pathways.sort()
    m1,m2 = list(mudata.mod)#list(adata.var["modality"].unique())
    x_ticks=[]
    x_labels=[]
    number=0
    
    for p in pathways:
        geneset=p
        number+=1
        df = pd.DataFrame(
                mudata.uns['ROMA'][f'{geneset}'].projections_1,
                #index=roma.adata.uns['ROMA'][f'{geneset}'].subsetlist,
                columns=[f'{geneset}']
            )

        proj_M1=df.T[m1].to_numpy().T 
        proj_M2=df.T[m2].to_numpy().T

        # PC1 axis line (extend in both directions)
        t = np.linspace(proj_M1[:, 0].min(), proj_M2[:, 0].max(), 100)
        plt.plot([number for _ in range(len(t))], t, 'k--')

        #Indicating the zero
        plt.plot(np.linspace(0,50,100), [0 for _ in range(len(t))], "--", color="tab:gray")

        # Projected points on PC1
        plt.scatter([number for _ in range(len(proj_M1))], proj_M1[:, 0], 
               color='red', zorder=3, s=10)
        # Projected points on PC1
        plt.scatter([number for _ in range(len(proj_M2))], proj_M2[:, 0], 
               color='orange', zorder=3, s=10)

        # Compute medians
        median_M1 = np.median(proj_M1)
        median_M2 = np.median(proj_M2)
        mean_of_medians = np.mean([median_M1, median_M2])
        median_of_proj = np.median(df)

        # Plot median of proj_A in blue
        plt.scatter(number, median_M1, color='blue', zorder=5, s=40,
                marker='D')

        # Plot median of proj_B in green
        plt.scatter(number, median_M2, color='green', zorder=5, s=40,
                marker='D')

        # Plot mean of medians in black
        plt.scatter(number, mean_of_medians, color='black', zorder=5, s=40,
                marker='*')

        # Plot mean of medians of projection in 
        plt.scatter(number, median_of_proj, color='tab:grey', zorder=5, s=40,
                marker='*')
        
        #Collect tick positions and labels
        x_ticks.append(number)
        x_labels.append(p)

    plt.xticks(
        ticks=x_ticks,
        labels=x_labels,
        rotation=90,
        ha="right",
        fontsize=8)
 
    plt.xlabel("Pahtways")
    plt.ylabel("PC1")
    legend_elements = [
        Line2D([0], [0], color='k', linestyle='--', label='PC1 projection line'),
        Line2D([0], [0], color='tab:gray', linestyle='--', label='level zero'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=6, label=f'Projections on PC1 – {m1}'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=6, label=f'Projections on PC1 – {m2}'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='blue', markersize=8, label=f'Median projection {m1}'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='green', markersize=8, label=f'Median projection {m2}'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='black', markersize=10, label='Mean of medians'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='tab:grey', markersize=10, label='Median of both projection'),
    ]

    plt.legend(handles=legend_elements)
    plt.tight_layout()

    if save_file:
        plt.savefig(file_name, bbox_inches='tight')
        
    plt.show()

    return None

def PC_samples_projection_MFA(mudata, save_file=False, file_name="projection.png"):

    if 'ROMA' not in mudata.uns:
        raise ValueError("No ROMA results found. Run compute() first.")
        
    plt.figure(1, figsize=(15, 6))
    
    pathways=list(mudata.uns['ROMA'].keys())
    pathways.sort()

    x_ticks=[]
    x_labels=[]
    number=0
    
    for p in pathways:
        geneset=p
        number+=1
        df = pd.DataFrame(
                mudata.uns['ROMA'][f'{geneset}'].projections_1,
                #index=roma.adata.uns['ROMA'][f'{geneset}'].subsetlist,
                columns=[f'{geneset}']
            )

        proj=df.to_numpy()

        # PC1 axis line (extend in both directions)
        t = np.linspace(proj[:, 0].min(), proj[:, 0].max(), 100)
        plt.plot([number for _ in range(len(t))], t, 'k--')

        #Indicating the zero
        plt.plot(np.linspace(0,50,100), [0 for _ in range(len(t))], "--", color="tab:gray")

        # Projected points on PC1
        plt.scatter([number for _ in range(len(proj))], proj[:, 0], 
               color='orange', zorder=3, s=10)
        
        # Compute medians
        median = np.median(proj)

        # Plot median of proj in blue
        plt.scatter(number, median, color='black', zorder=5, s=40,
                marker='*')

        #Collect tick positions and labels
        x_ticks.append(number)
        x_labels.append(p)

    plt.xticks(
        ticks=x_ticks,
        labels=x_labels,
        rotation=90,
        ha="right",
        fontsize=8)

    plt.xlabel("Pathway")
    plt.ylabel("PC1")

    legend_elements = [
        Line2D([0], [0], color='k', linestyle='--', label='PC1 projection line'),
        Line2D([0], [0], color='tab:gray', linestyle='--', label='level zero'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=6, label=f'Projections on PC1'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='black', markersize=8, label=f'Median projection'),
    ]

    plt.legend(handles=legend_elements)
    plt.tight_layout()

    if save_file:
        plt.savefig(file_name, bbox_inches='tight')
        
    plt.show()

    return None


def plotting_L1_distribution(null_distributions):
    
    fig_name="L1_distrib"
    plt.figure(1)
    for key, value in null_distributions.items():
        samples = value[0]
        plt.hist(samples, bins=20, density=True, alpha=0.3, label=f"size {key}")

    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.title("L1 distribution per gene set size")
    plt.legend()
    plt.show()
    
    return None

def plotting_L1_distribution_per_set(mudata, null_distributions, geneset):
    
    fig_name="L1_distrib"
    subsetlist=mudata.uns["ROMA"][f"{geneset}"].subsetlist
    outliers=mudata.uns["ROMA"][f"{geneset}"].outliers
    genesetsize=sum(1 for i in range(len(subsetlist)) if i not in outliers)
    observed_score=mudata.uns["ROMA"][f"{geneset}"].test_l1
    
    key = mudata.uns["ROMA"][f"{geneset}"].nullgenesetsize
    value= null_distributions[key]
    
    plt.figure(1)
    
    samples = value[0]
    plt.hist(samples, bins=20, density=True, alpha=0.3, label=f"size {key}")
    plt.axvline(observed_score, color='red', linestyle='--', linewidth=1.5, zorder=4)
    plt.scatter(observed_score, 0, color='red', marker='v', s=120, zorder=5,
            label=f"Observed score ({observed_score:.2f})")

    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.title(f"L1 distribution – {geneset}")
    plt.legend()
    plt.show()
    
    return None


def plotting_clustermap(mudata, ground_truth, clustering_labels=None, save_file=False, file_name="clustermap.png", verbose=False):
    
    if 'ROMA' not in mudata.uns:
        raise ValueError("No ROMA results found. Run compute() first.")
    if 'ROMA_active_modules' not in mudata.uns:
        raise ValueError("Issue with retrieving ROMA results. Please verify that compute() ran without error.")

    # ground_truth_name=list(ground_truth.columns())[0]
    ground_truth_name=list(ground_truth.keys())[0]
    ground_truth=ground_truth[ground_truth_name]
        
    sample_scores={"multiROMA":{active_path : mudata.uns['ROMA'][f"{active_path}"].svd.components_
            for active_path in mudata.uns['ROMA_active_modules'].index.tolist()}}

    key="multiROMA"

    data = pd.DataFrame(
        {key: arr[0] for key, arr in sample_scores[f'{key}'].items()},
        index=ground_truth.index
        ).T

    """
    g0=sns.clustermap(data, cmap=sns.color_palette("coolwarm", as_cmap=True), row_cluster=True) 
    plt.savefig(f"heatmap_{key}.png", bbox_inches='tight')
    plt.close()
    """

    #Clustering
    if clustering_labels==None:
        hierarchical_clustering = sklearn.cluster.AgglomerativeClustering(n_clusters=5,metric='euclidean').fit(data.T)
        sample_labels=pd.DataFrame(hierarchical_clustering.labels_, index=ground_truth.index, columns=["Clustering_labels"])
    else:
        samples_lables=pd.DataFrame(clustering_labels, index=ground_truth.index, columns=["Clustering_labels"])
    
    # --- Color mappings for the bands ---
    gt_levels = list(ground_truth.unique())  # or .cat.categories if it's already categorical
    gt_palette = sns.color_palette('Set1', len(gt_levels))
    gt_colors = dict(zip(gt_levels, gt_palette))
    #ct_colors = {"CellType_1": '#4CAF9A', "CellType_2": '#F4845F', "CellType_3": '#B77103', "CellType_4": '#C90076', "CellType_5": '#2081CA'}  # HEX colors 
    cah_palette      = sns.color_palette('Set2', int(sample_labels["Clustering_labels"].max()) + 1)
    cah_colors       = {i: cah_palette[i] for i in range(int(sample_labels["Clustering_labels"].max()) + 1)}
    
    # Build the col_colors DataFrame (one row per band)
    col_colors = pd.DataFrame({
        f'{ground_truth_name}': ground_truth.astype(str).map(gt_colors.get),
        'Clustering': sample_labels["Clustering_labels"].map(cah_colors.get),
        }, index=ground_truth.index)

    n_samples  = data.shape[1]
    n_pathways = data.shape[0]
    large      = n_samples > 500   # ← flag for large datasets
    
    # --- Draw the clustermap ---
    g = sns.clustermap(
        data.astype(float),
        col_colors=col_colors,                                              # <-- the color bands on top
        row_cluster=True,                                                  # set True if you want row dendrogram too
        col_cluster=True,                                                   # CAH dendrogram on columns
        cmap=sns.color_palette("coolwarm", as_cmap=True),                   # red/white/blue diverging colormap
        figsize=(max(14, n_samples * 0.08), max(5, n_pathways * 0.4)),
        dendrogram_ratio=(.1, .2),
        cbar_pos=(1.10, 0.25, 0.0015, 0.50),
    linewidths=0 if large else 0.1,
    )

    g.ax_heatmap.set_xticklabels(
        g.ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=9
    )
    g.ax_heatmap.set_yticklabels(
        g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=9
    )

    # Move y-axis label to the LEFT side to avoid overlap with legend
    #g.ax_heatmap.yaxis.set_label_position('left')
    #g.ax_heatmap.yaxis.tick_left()

    # --- Legends ---
    gt_patches = [
        mpatches.Patch(color=color, label=label)
        for label, color in gt_colors.items()
    ]
    cah_patches = [
        mpatches.Patch(color=color, label=f'Cluster {label}')
        for label, color in cah_colors.items()
    ]

    # First legend: PAM50 conditions
    legend1 = g.ax_heatmap.legend(
        handles=gt_patches,
        bbox_to_anchor=(1.33, 1),      # ← push further right
        loc='upper left',
        title=f'{ground_truth_name}',
        frameon=False,
    )

    # Second legend: CAH clusters (manually added since ax.legend() replaces the first)
    g.ax_heatmap.add_artist(legend1)   # ← preserve the first legend!
    legend2 = g.ax_heatmap.legend(
        handles=cah_patches,
        bbox_to_anchor=(1.33, 0.15),    # ← placed below the first legend
        loc='upper left',
        title='Clustering',
        frameon=False,
    )

    #g.fig.suptitle('Shifted gene sets', y=1.02, fontweight='bold')
    if save_file:
        plt.savefig(file_name, bbox_inches='tight')
        
    plt.show()

    mutual_info_score=sklearn.metrics.adjusted_mutual_info_score(ground_truth,sample_labels["Clustering_labels"])
    if verbose:
        print("\n Mutual information score: ",mutual_info_score, "\n")

    return mutual_info_score