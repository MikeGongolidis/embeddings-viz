from fetch_embeddings import get_embeddings
from visualize_embeddings import perform_pca, visualize_with_plotly, cluster_data, visualize_clusters_with_plotly, visualize_with_field_density_contours
from keywords import sub_areas, field_colors
import numpy as np

def main():
    print("Fetching embeddings for the sub-areas...")
    model = 'text-embedding-3-large'
    
    # Flatten sub_areas and prepare colors
    flat_sub_areas = [item for sublist in sub_areas.values() for item in sublist]
    colors = [color for field, sublist in sub_areas.items() for color in [field_colors[field]]*len(sublist)]
    
    embeddings = get_embeddings(flat_sub_areas, model)
    embeddings_array = np.array(embeddings)
    print("Embeddings fetched successfully.\n")

    print("Performing PCA on the embeddings...")
    pca_results = perform_pca(embeddings_array)
    print("PCA completed.\n")

    print("Visualizing the PCA results with Plotly...")
    visualize_with_plotly(pca_results, flat_sub_areas, colors)
    print("Visualization complete. Check the generated plot.")

    # # Example of integrating clustering and visualization in your workflow
    # n_clusters = 14  # Set the number of clusters you want

    # # Assuming embeddings_array is your PCA results
    # cluster_labels = cluster_data(pca_results, n_clusters=n_clusters)

    # # Then visualize
    # visualize_clusters_with_plotly(pca_results, flat_sub_areas, cluster_labels)
    
    #visualize_with_field_density_contours(pca_results, flat_sub_areas, colors)


if __name__ == "__main__":
    main()
