import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

from sklearn.cluster import KMeans

def cluster_data(pca_results, n_clusters=5):
    """Cluster PCA results into n clusters."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(pca_results)
    return kmeans.labels_  # Cluster assignments


def perform_pca(embeddings):
    """Perform PCA on embeddings."""
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(embeddings)
    return pca_results


def visualize_with_seaborn(pca_results, labels, colors):
    """Visualize PCA results with Seaborn, colored by field."""
    df = pd.DataFrame(pca_results, columns=["PCA1", "PCA2"])
    df["Sub-Area"] = labels
    df["Color"] = colors

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 10))
    plot = sns.scatterplot(
        data=df,
        x="PCA1",
        y="PCA2",
        hue="Sub-Area",
        palette=df["Color"].to_list(),
        s=100,
        legend=False,
    )

    for line in range(0, df.shape[0]):
        plot.text(
            df.PCA1[line] + 0.02,
            df.PCA2[line],
            df["Sub-Area"][line],
            horizontalalignment="left",
            size="small",
            color="black",
            weight="semibold",
        )

    plt.title("PCA of Sub-Areas by Field")
    plt.xlabel("PCA Dimension 1")
    plt.ylabel("PCA Dimension 2")
    plt.show()


def visualize_with_plotly(pca_results, labels, colors):
    """Visualize PCA results with Plotly, showing labels on hover with specific colors for each point."""
    df = pd.DataFrame(pca_results, columns=["PCA1", "PCA2"])
    df["Sub-Area"] = labels
    df["Color"] = colors  # This assumes colors are specific color codes

    # Create a Plotly figure
    fig = go.Figure()
    # Adjust this value to increase or decrease point size
    point_size = 12

    # Add each point to the figure individually to control its color
    for idx, row in df.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row["PCA1"]],
                y=[row["PCA2"]],
                marker=dict(color=row["Color"], size=point_size),
                mode="markers",
                hoverinfo="text",
                hovertext=row["Sub-Area"],
                name=row["Sub-Area"],
            )
        )

    # Update layout to add titles and adjust hover mode
    fig.update_layout(
        title="PCA of Sub-Areas",
        xaxis_title="PCA Dimension 1",
        yaxis_title="PCA Dimension 2",
        hovermode="closest",
    )

    # Show the figure
    fig.show()


def visualize_clusters_with_plotly(pca_results, labels, cluster_labels):
    """Visualize PCA results with clusters shown in different colors."""
    df = pd.DataFrame(pca_results, columns=['PCA1', 'PCA2'])
    df['Sub-Area'] = labels
    df['Cluster'] = cluster_labels  # Assign cluster to each point

    fig = go.Figure()

    # Use a unique color for each cluster
    colors = px.colors.qualitative.Plotly

    for cluster_id in sorted(df['Cluster'].unique()):
        cluster_df = df[df['Cluster'] == cluster_id]
        fig.add_trace(go.Scatter(x=cluster_df['PCA1'], y=cluster_df['PCA2'],
                                 marker=dict(color=colors[cluster_id % len(colors)], size=12),
                                 mode='markers',
                                 hoverinfo='text',
                                 hovertext=cluster_df['Sub-Area'],
                                 name=f'Cluster {cluster_id}'))

    fig.update_layout(title='PCA of Sub-Areas with Clusters',
                      xaxis_title='PCA Dimension 1',
                      yaxis_title='PCA Dimension 2',
                      hovermode='closest')

    fig.show()


import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Assuming 'pca_results' and 'labels' are available
# 'sub_areas' should be modified to include PCA results or be matched with 'labels' for coloring

def visualize_with_field_density_contours(pca_results, labels, sub_areas, field_colors):
    """Visualize PCA results with density contours for each field."""
    # Prepare DataFrame
    df = pd.DataFrame(pca_results, columns=['PCA1', 'PCA2'])
    df['Sub-Area'] = labels
    
    # Initialize a figure
    fig = go.Figure()

    # Generate a plot for each field
    for field, sub_area_list in sub_areas.items():
        # Filter points belonging to the current field
        field_points = df[df['Sub-Area'].isin(sub_area_list)]
        
        # Check if there are points to plot
        if not field_points.empty:
            # Calculate density
            x = field_points['PCA1']
            y = field_points['PCA2']
            z = np.histogram2d(x, y, bins=[30, 30], density=True)[0]
            
            # # Create a contour plot for the field
            # fig.add_trace(go.Contour(
            #     x=(np.linspace(x.min(), x.max(), 30)),
            #     y=(np.linspace(y.min(), y.max(), 30)),
            #     z=z, 
            #     colorscale=[(0, field_colors[field]), (1, field_colors[field])],
            #     showscale=False,
            #     name=field,
            #     hoverinfo='skip'  # Optional: Adjust based on whether you want hover info for contours
            # ))

            # Add scatter plot for points
            fig.add_trace(go.Scatter(
                x=field_points['PCA1'], y=field_points['PCA2'],
                mode='markers',
                marker=dict(color=field_colors[field], size=5, line=dict(width=1)),
                name=field,
                text=field_points['Sub-Area'],  # Display sub-area on hover
                hoverinfo='text'
            ))

    # Update layout
    fig.update_layout(
        title='PCA Results with Field-Specific Density Contours',
        xaxis_title='PCA Dimension 1',
        yaxis_title='PCA Dimension 2',
        hovermode='closest'
    )
    
    fig.show()
