import argparse

import numpy as np
import pandas as pd

from tqdm import tqdm


def compute_precision_at_k(ranked_targets: np.ndarray,
                           k: int) -> float:
    """
    Computes the precision at k.
    Args:
        ranked_targets: A boolean array of retrieved targets, True if relevant and False otherwise.
        k: The number of examples to consider

    Returns: The precision at k
    """
    assert k >= 1
    assert ranked_targets.size >= k, ValueError('Relevance score length < k')
    return np.mean(ranked_targets[:k])


def compute_average_precision(ranked_targets: np.ndarray,                                     # Computes the average precision.
                              gtp: int) -> float:
    """
    Args:
        ranked_targets: A boolean array of retrieved targets, True if relevant and False otherwise.
        gtp: ground truth positives.

    Returns:
        The average precision.
    """
    assert gtp >= 1
    # compute precision at rank only for positive targets
    out = [compute_precision_at_k(ranked_targets, k + 1) for k in range(ranked_targets.size) if ranked_targets[k]]
    if len(out) == 0:
        # no relevant targets in top1000 results
        return 0.0
    else:
        return np.sum(out) / gtp


def calculate_map(ranked_retrieval_results: np.ndarray                       # Calculates the mean average precision.
                  query_labels: np.ndarray,
                  gallery_labels: np.ndarray) -> float:
    """
    Args:
        ranked_retrieval_results: A 2D array of ranked retrieval results (shape: n_queries x 1000), because we use
                                top1000 retrieval results.
        query_labels: A 1D array of query class labels (shape: n_queries).
        gallery_labels: A 1D array of gallery class labels (shape: n_gallery_items).
    Returns:
        The mean average precision.
    """
    assert ranked_retrieval_results.ndim == 2
    assert ranked_retrieval_results.shape[1] == 1000

    class_average_precisions = []

    class_ids, class_counts = np.unique(gallery_labels, return_counts=True)
    class_id2quantity_dict = dict(zip(class_ids, class_counts))
    for gallery_indices, query_class_id in tqdm(
                            zip(ranked_retrieval_results, query_labels),
                            total=len(query_labels)):
        # Checking that no image is repeated in the retrival results
        assert len(np.unique(gallery_indices)) == len(gallery_indices), \
                    ValueError('Repeated images in retrieval results')

        current_retrieval = gallery_labels[gallery_indices] == query_class_id
        gpt = class_id2quantity_dict[query_class_id]

        class_average_precisions.append(
            compute_average_precision(current_retrieval, gpt)
        )

    mean_average_precision = np.mean(class_average_precisions)
    return mean_average_precision






resnet_path = 'C:/Users/Bhupesh Hada/Documents/MS in ESDS/CSE573_Computer_vision_and_Image_processing/Academic Project/cv_project/resent18_model_results/class_rank_resnet18.csv'
seller_path = 'C:/Users/Bhupesh Hada/Documents/MS in ESDS/CSE573_Computer_vision_and_Image_processing/Academic Project/Academic Project/MCS2023_development_test_data/development_test_data/gallery.csv'
user_path = 'C:/Users/Bhupesh Hada/Documents/MS in ESDS/CSE573_Computer_vision_and_Image_processing/Academic Project/Academic Project/MCS2023_development_test_data/development_test_data/queries.csv'
convnext = 'C:/Users/Bhupesh Hada/Documents/MS in ESDS/CSE573_Computer_vision_and_Image_processing/Academic Project/cv_project/convnext_xxlarge_result/class_rank.csv'



query_df = pd.read_csv(user_path)
query_labels = query_df['product_id'].values

gallery_df = pd.read_csv(seller_path)
gallery_labels = gallery_df['product_id'].values

ranked_retrieval_result = pd.read_csv(resnet_path)
ranked_retrieval_result = ranked_retrieval_result.iloc[:,1:]
ranked_retrieval_results = ranked_retrieval_result.to_numpy()

print(calculate_map(ranked_retrieval_results, query_labels, gallery_labels))


ranked_retrieval_result = pd.read_csv(convnext)
ranked_retrieval_result = ranked_retrieval_result.iloc[:,1:]
ranked_retrieval_results = ranked_retrieval_result.to_numpy()

calculate_map(ranked_retrieval_results, query_labels, gallery_labels)


