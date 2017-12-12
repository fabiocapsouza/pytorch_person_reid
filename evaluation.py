import torch
import numpy as np


# (x - y)^2 = x^2 - 2*x*y + y^2
def pairwise_dist(mat):
    # get the product x * y
    # here, y = x.t()
    r = torch.mm(mat, mat.t())
    # get the diagonal elements
    diag = r.diag().unsqueeze(0)
    diag = diag.expand_as(r)
    # compute the distance matrix
    D = diag + diag.t() - 2*r + 1e-7
    del r, diag
    return D


def pairwise_squared_euclidian_distance(x, y, normalize=False):
    # (x - y)^2 = x^2 + y^2 - 2*x*y
    N = x.size(0)
    M = y.size(0)

    if normalize:
        x /= x.norm(2)
        y /= y.norm(2)
    # Get the squared norm of the elements of x as a column vector and expand to shape (N, M)
    x2 = torch.mm(x, x.t())
    xdiag = x2.diag().unsqueeze(1).expand(N, M)
    # Get the squared norm of the elements of y as a row vector and expand to shape (N, M)
    y2 = torch.mm(y, y.t())
    ydiag = y2.diag().unsqueeze(0).expand(N, M)
    # Compute the distances (add 1e-7 for numerical stability)
    D = xdiag + ydiag - 2*torch.mm(x, y.t()) + 1e-7
    del x2, y2, xdiag, ydiag
    return D


def generate_split_cuhk03(y_val, cam_ids):
    unique_cams = np.unique(cam_ids)
    unique_pids = np.unique(y_val)
    
    indices_by_cam = {cam_id: [] for cam_id in unique_cams}

    for pid in unique_pids:
        indices_of_pid = np.argwhere(y_val == pid).squeeze()
        cam_ids_of_pid = cam_ids[indices_of_pid]
        for cam_id in unique_cams:
            indices_filtered_for_cam = indices_of_pid[cam_ids_of_pid == cam_id]
            if len(indices_filtered_for_cam) > 0:
                random_idx = np.random.choice(indices_filtered_for_cam)
                indices_by_cam[cam_id].append(random_idx)


    rand = np.random.rand()
    query_cam = 0 if rand < 0.5 else 1
    gallery_cam = 1 if rand < 0.5 else 0
                
    return {'query': np.array(indices_by_cam[query_cam]),
            'gallery': np.array(indices_by_cam[gallery_cam])}


def get_topk_results(dists, k, indices):

    query_idx = torch.from_numpy(indices['query']).type(torch.LongTensor)
    gall_idx = torch.from_numpy(indices['gallery']).type(torch.LongTensor)

    if dists.is_cuda:
        query_idx = query_idx.cuda()
        gall_idx = gall_idx.cuda()

    dists_subset = torch.index_select(dists, dim=0, index=query_idx)
    dists_subset = torch.index_select(dists_subset, dim=1, index=gall_idx)

    _, selected = torch.topk(dists_subset, k=k, dim=1, largest=False)

    return selected


def generate_query(y_query, cam_ids):
    ''' For each Person ID, selects one image per camera (if available) to be used as 
        a Query image. The image is selected by its index. This is the query protocol
        used by Market-1501. For Market-1501, it should have 3368 images.

        # Parameters:
            - y_query: array containing the person ID (labels) of every image in the Query set.
            - cam_ids: array containing the camera ID for every image in the Query set.

        # Returns:
            - indices_by_cam (dict): dictionary of cam_id -> [indices to be used as Query] '''

    unique_cams = np.unique(cam_ids)
    unique_pids = np.unique(y_query)

    indices_by_cam = {cam_id: [] for cam_id in unique_cams}

    for pid in unique_pids:
        indices_of_pid = np.argwhere(y_query == pid).squeeze()
        cam_ids_of_pid = cam_ids[indices_of_pid]
        for cam_id in unique_cams:
            indices_filtered_for_cam = indices_of_pid[cam_ids_of_pid == cam_id]
            if len(indices_filtered_for_cam) > 0:
                random_idx = int(np.random.choice(indices_filtered_for_cam))
                indices_by_cam[cam_id].append(random_idx)
                
    return indices_by_cam


def get_gallery_for_cross_camera_search(gal_cam_ids, cam_id):
    ''' Return the gallery indices for Cross Camera Search (elements of the Gallery set 
        that are not from the same camera `cam_id`). '''
    return np.argwhere(gal_cam_ids != cam_id).squeeze()


def find_best_matches():
    """
    For each camera:
    1- Find the gallery indexes for the camera ID.
    2- Select query and gallery features from `all_features`.
    3- Calculate the similarity matrix between query and gallery features.
    4- Get the best matches by sorting the similarity matrix in dim/axis=1
    """
    pass

def calculate_mAP(query_indices_by_cam, all_features, y_gallery, gal_cam_ids):
    """
    For each camera:
    1- Find the gallery indexes for the camera ID.
    2- Select query and gallery features from `all_features`.
    3- Calculate the similarity matrix between query and gallery features.
    4- Get the best matches by sorting the similarity matrix in dim/axis=1
    5- Calculate the Mean Average Precision for all Queries.
    """

    maps_per_cam = []
    # Para fazer função, usa: y_querygal, cam_ids, gal_cam_ids, allfeatures

    for cam, indices in query_indices_by_cam.items():
        #print('Índices de query para cam {}:\n{}'.format(cam, indices[:10]))
        gal_indices = get_gallery_for_cross_camera_search(gal_cam_ids, cam)
        #print('Índices da galeria para essa câmera:\n{} {}'.format(gal_indices[:10], gal_indices[-10:]))

        qfeatures = all_features[torch.LongTensor(indices)]
        gfeatures = all_features[torch.LongTensor(gal_indices)]
        distances = pairwise_squared_euclidian_distance(qfeatures, gfeatures).numpy()

        #pdb.set_trace()
        # match_indexes = índices dos resultados ordenados em ordem crescente.
        # estes índices devem ser convertidos para índices da galeria via indexação de `gal_indices`
        match_indexes = np.argsort(distances, axis=1)
        predicted = gal_indices[match_indexes]

        y_query = y_gallery[indices]
        actual = []

        #pdb.set_trace()
        for qpid in y_query:
            act = np.argwhere(np.logical_and(gal_cam_ids != cam, y_gallery == qpid)).squeeze(axis=1)
            if len(act) == 0:
                print('act com len == 0 para cam {}, pid {}'.format(cam, qpid))
            actual.append(act)

        map_result = mapk(actual, predicted, k=-1)
        #print('Câmera {}, mAP: {}\n'.format(cam, mapk_result))
        maps_per_cam.append(map_result)

    return np.mean(maps_per_cam)


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """

    if len(predicted)>k:
        predicted = predicted[:k]

    if k == -1:
        k = np.Inf

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

        if num_hits == len(actual):
            break

    if actual is None or len(actual) == 0:
        return 0.0

    return score / min(len(actual), k)


def apk2(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    precision = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            precision = num_hits / (i+1.0)
            score += precision * (1/len(actual))
        if num_hits == len(actual):
            break

    if actual is None or len(actual) == 0:
        return 0.0

    return score



def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        If higher than zero, set the number of predicted elements to be considered.
        if -1, calculate mAP over the entire predicted array

    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    # if k == 'length':
    #     return np.mean([apk(a,p,len(p)) for a,p in zip(actual, predicted)])
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])