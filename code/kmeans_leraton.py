import numpy as np
import matplotlib.pyplot as plt
import imageio


def euclidian_distance(point1: np.array = None, point2: np.array = None) -> float:
    """Compute the euclidian distance of 2 points in n-th dimension

    Args:
        point1 (np.array): First point. Defaults to None.
        point2 (np.array): Second point. Defaults to None.

    Returns:
        float: Euclidian distance
    """
    return np.linalg.norm(point1 - point2)

def K_means(
    points: np.array = None,
    nb_clust: int = 0,
    max_it: int = np.inf,
    stop_val: float = 0.5,
    ) -> tuple:
    """K-means method

    Args:
        points (np.array): The points to cluster. Defaults to None.
        nb_clust (int): Number of cluster also referd as K. Defaults to 0.
        max_it (int, optional): Max number of iteration. Defaults to np.inf.
        stop_val (float, optional): Min centers movement to stop the iterative evolution of the centers. Defaults to 0.5.
        show_evolution (bool, optional): use matplotlib to display the evolution of the identification (only in 2D and 3D). Defaults to False.

    Returns:
        tuple:
            - centers, a np.array with the coordonates of the center.
            - clusters_points a np.array with each cluster containing the values of the point of the cluster
            - cluster_index a np.array with each cluster containing the index of each point from points
    """

    num_it = 0
    dim = len(points[0])
    centers = np.zeros((nb_clust, dim), dtype=np.float64)
    distances = np.zeros((len(points), nb_clust))
    for i in range(nb_clust):
        centers[i] = points[np.random.randint(np.shape(points)[0])]

    new_centers = centers.copy()
    stop = False

    while (not stop) and num_it < max_it:
        num_it += 1
        clusters = [[] for i in range(nb_clust)]

        # computation of the distances

        for i in range(len(points)):
            for j in range(nb_clust):
                distances[i][j] = euclidian_distance(centers[j], points[i])

            # stockage de l'index des points du cluster dans le tableau cluster

            index = np.where(distances[i] == min(distances[i]))[0][0]
            clusters[index].append(i)

        # computation of the new centers

        stop_dist = 0
        for j in range(nb_clust):
            if np.shape(clusters[j])[0] != 0:
                new_centers[j] = (
                    sum([np.array(points[i], dtype=np.float64) for i in clusters[j]])
                    / np.shape(clusters[j])[0]
                )
                stop_dist += euclidian_distance(new_centers[j], centers[j])
            else:
                new_centers[j] = points[
                    np.where(
                        np.transpose(distances)[j] == min(np.transpose(distances)[j])
                    )[0][0]
                ]

        # update of the center or stop

        #print(stop_dist, num_it)
        if stop_dist > stop_val:
            centers = new_centers.copy()
        else:
            stop = True

    cluster_index = clusters
    # centers = new_centers.copy()
    clusters_points = [[points[j] for j in cluster_index[i]] for i in range(nb_clust)]

    return centers, clusters_points, cluster_index
