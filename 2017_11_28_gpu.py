import numpy as np
from numba import cuda
import accelerate.cuda.blas as cublas
import accelerate.cuda.sorting as sorting
import numba
from numba import float32, guvectorize
import math
from timeit import default_timer as timer


@guvectorize([(float32[:, :], float32[:])], '(m,k)->(m)', nopython=True)
def _create_dot_product(a, dots_a):
    for i in np.arange(a.shape[0]):
        dots_a[i] = np.dot(a[i, :], a[i, :])


@cuda.jit
def _sums_of_dots_gpu(dots_a, dots_b, s_o_dots):
    a_index = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x

    if a_index < dots_a.shape[0]:
        for b_index in range(dots_b.shape[0]):
            s_o_dots[a_index, b_index] = dots_a[a_index] + dots_b[b_index]


def _calculate_distances_on_gpu(a, b, distances_on_gpu, verbose=False):
    m = a.shape[0]
    n = b.shape[0]
    k = a.shape[1]

    dot_time, dots_a, dots_b = inner_product(a, b)
    print_execution_time(dot_time, 'Making dot products:', verbose)  # why print this and not profile instead?

    s_o_dot_time = sum_of_dot_products_matrix(distances_on_gpu, dots_a, dots_b)
    print_execution_time(s_o_dot_time, 'Summing dot products on GPU:', verbose)

    gemm_time = cross_dot_products_and_sum(a, b, distances_on_gpu, k, m, n)
    print_execution_time(gemm_time, 'cuBLAS gemm:', verbose)


def print_execution_time(variable, string, verbose):
    if verbose:
        print("{}:".format(string), "%.3f" % variable, "s")

def cross_dot_products_and_sum(a, b, distances_on_gpu, k, m, n):
    """
    calculate the -2<a,b> cross dot products matrix and do the sum (||a||^2 + ||b||^2) -2<a,b>
    :param a:
    :param b:
    :param distances_on_gpu:
    :param k:
    :param m:
    :param n:
    :return:
    """
    s3 = timer()
    da = cuda.to_device(np.asfortranarray(a))
    db = cuda.to_device(np.asfortranarray(b))
    blas = cublas.Blas()
    blas.gemm('N', 'T', m, n, k, -2.0, da, db, 1.0, distances_on_gpu)  # TODO: replace magic number with variable
    numba.cuda.synchronize()
    e3 = timer()
    time_gemm = e3 - s3
    return time_gemm


def sum_of_dot_products_matrix(distances_on_gpu, dots_a, dots_b):

    """
    calculates the ||a||^2 + ||b||^2 sum of dot products matrix (a.a + b.b) on the gpu and save on the gpu_temp matrix
    :param distances_on_gpu:
    :param dots_a:
    :param dots_b:
    :return:
    """
    s2 = timer()
    ddots_a = cuda.to_device(np.asfortranarray(dots_a))
    ddots_b = cuda.to_device(np.asfortranarray(dots_b))
    threads_per_block = 32
    blocks_per_grid = math.ceil(ddots_a.shape[0] / threads_per_block)
    _sums_of_dots_gpu[blocks_per_grid, threads_per_block](ddots_a, ddots_b, distances_on_gpu)
    numba.cuda.synchronize()
    e2 = timer()
    s_o_dot_time = e2 - s2
    return s_o_dot_time


def inner_product(a, b):
    """
    calculates the inner product of each row for both matrices
    :param a:
    :param b:
    :return:
    """
    s1 = timer()
    dots_a = _create_dot_product(a)
    dots_b = _create_dot_product(b)
    e1 = timer()
    dot_time = e1 - s1
    return dot_time, dots_a, dots_b


def _segment_sort_transposed_distances_get_knns(num_of_neighbours, distances_on_gpu, n_sorts,
                                                verbose=False):

    m = distances_on_gpu.shape[0]  # all spikes
    n = distances_on_gpu.shape[1]  # part of spikes in iteration

    selected_sorted_distances = np.empty((n, num_of_neighbours))
    selected_sorted_indices = np.empty((n, num_of_neighbours))

    s = timer()  # TODO: make a timer decorator and extract everything in between as functions
    p = np.append(np.arange(0, n, int(n / n_sorts)), n)
    for i in np.arange(1, p.shape[0]):
        delta_n = p[i] - p[i - 1]
        keys = get_something_keys(delta_n, distances_on_gpu, i, m, num_of_neighbours, p, selected_sorted_distances)
        values = get_something_values(delta_n, m, num_of_neighbours)
        get_sorted_segments(delta_n, i, keys, m, p, selected_sorted_indices, values)
        if verbose:
            print('     Sorted ' + str(i) + ' of ' + str(p.shape[0] - 1) + ' segments of this iteration')
    e = timer()
    sort_time = e - s
    print("SORTING TIME:", "%.3f" % sort_time, "s")

    return selected_sorted_indices, selected_sorted_distances


def get_something_keys(delta_n, distances_on_gpu, i, m, num_of_neighbours, p, selected_sorted_distances):
    keys = np.ascontiguousarray(distances_on_gpu.copy_to_host()[:, p[i - 1]:p[i]].transpose().reshape(m * delta_n))
    keys = np.reshape(keys, (delta_n, m))[:, :num_of_neighbours]
    selected_sorted_distances[p[i - 1]:p[i], :] = keys[:, :]
    return keys


def get_something_values(delta_n, m, num_of_neighbours):
    values = np.ascontiguousarray(np.tile(np.arange(m), (delta_n, 1)).reshape(m * delta_n))
    values = np.reshape(values, (delta_n, m))[:, :num_of_neighbours]
    return values


def get_sorted_segments(delta_n, i, keys, m, p, selected_sorted_indices, values):
    segments = np.ascontiguousarray(np.arange(m, m * delta_n, m))
    sorting.segmented_sort(keys=keys, vals=values, segments=segments)
    selected_sorted_indices[p[i - 1]:p[i], :] = values[:, :]


def calculate_knn_distances(template_features_sparse_clean, perplexity=100, mem_usage=0.9, verbose=True):
    start = timer()
    n_neighbours = perplexity * 3 + 1  # TODO: replace magic number with variable
    n_template_features = template_features_sparse_clean.shape[0]

    closest_indices = np.empty((n_template_features, n_neighbours))
    closest_distances = np.empty((n_template_features, n_neighbours))

    gpu_mem = cuda.current_context().get_memory_info()
    available_gpu_mem = 0.5 * gpu_mem[0] # TODO: replace magic number with variable

    n = int(np.ceil(available_gpu_mem / (4 * n_template_features)))  # TODO: replace magic number with variable

    n_iters = int(np.ceil(n_template_features / n))

    some_descriptive_name = np.arange(0, n_iters * (n - 1), n)
    second_matrix_idx = [(i, i + n) for i in some_descriptive_name]
    second_matrix_idx[-1] = (second_matrix_idx[-1][0], n_template_features)

    first_matrix = np.array(template_features_sparse_clean, dtype=np.float32)

    for i in np.arange(n_iters):
        second_matrix = np.array(template_features_sparse_clean[second_matrix_idx[i][0]:second_matrix_idx[i][1], :],
                                 dtype=np.float32)

        s = timer()
        if i != 0:
            del distances_on_gpu
        cuda.current_context().deallocations.clear()  # for numba version 0.30
        # cuda.current_context().trashing.clear()  # for numba version 0.25

        if verbose:
            print('LOADING UP THE GPU')

        distances_on_gpu, load_time = load_matrix_to_gpu(n_template_features, s, second_matrix)

        if verbose:
            print("Loading matrix time:", "%.3f" % load_time, "s")
            print('ITERATION NUMBER: ' + str(i + 1))

        _calculate_distances_on_gpu(a=first_matrix, b=second_matrix, distances_on_gpu=distances_on_gpu, verbose=verbose)

        n_sorts = calculate_n_sorting_segments(n, n_template_features)

        if verbose:
            print('     Number of sorting segments = ' + str(n_sorts + 1))

        temp_indices, temp_distances = \
            _segment_sort_transposed_distances_get_knns(num_of_neighbours=n_neighbours,
                                                        distances_on_gpu=distances_on_gpu,
                                                        n_sorts=n_sorts, verbose=verbose)

        closest_indices[second_matrix_idx[i][0]: second_matrix_idx[i][1], :] = np.ascontiguousarray(temp_indices)
        closest_distances[second_matrix_idx[i][0]: second_matrix_idx[i][1], :] = np.ascontiguousarray(temp_distances)

        if verbose:
            print('FINISHED CALCULATING {} OF {}'.format(str(i + 1), str(n_iters)))

        end = timer()
        full_time = end - start
        if verbose:
            print("Spend Time:", "%.3f" % full_time, "s")

    return closest_indices, np.sqrt(np.abs(closest_distances))

def calculate_n_sorting_segments(n, n_template_features):
    """

    :param n:
    :param n_template_features:
    :return:
    """
    N_BYTES_REQUIRED = 16 # 4 is the bytes per float32, for 2 arrays loaded to gpu, plus overhead

    gpu_mem = cuda.current_context().get_memory_info()
    available_gpu_mem = 0.5 * gpu_mem[0] # TODO: replace magic number with variable
    n_sorts = int(np.ceil((N_BYTES_REQUIRED * n * n_template_features) / available_gpu_mem))
    return n_sorts


def load_matrix_to_gpu(n_template_features, s, second_matrix):
    temp = np.array(np.zeros((n_template_features, second_matrix.shape[0]), dtype=np.float32))
    distances_on_gpu = cuda.to_device(np.asfortranarray(temp))
    e = timer()
    load_time = e - s
    return distances_on_gpu, load_time


