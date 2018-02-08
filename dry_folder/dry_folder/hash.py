import hashlib

STRONG = True

try:
    weak_hash = hashlib.md5
    strong_hash = hashlib.sha1
except AttributeError:
    weak_hash = None  # md5 may not be available
    strong_hash = hashlib.sha1


def read_by_chunks(file_object, chunk_size=1024):
    while True:
        data = file_object.read(chunk_size)
        if not data:
            break
        else:
            yield data


def hash_file(file_path, hash_algorithm, block_size=2**16):
    """

    :param str file_path:
    :param hashlib.hash hash_algorithm:
    :param int block_size:
    :return:
    :rtype: str
    """
    h = hash_algorithm()
    with open(file_path, 'rb') as infile:
        for chunk in read_by_chunks(infile, block_size):
            h.update(chunk)
    hash_sum = h.hexdigest()
    return hash_sum


def rehash(fp, weak_sums):
    """
    Gets weak hash and if colides with hash in weak_sums, rehashes with strong hash
    finally, returns hash

    :param str fp:
    :param set weak_sums:
    :return:
    """
    weak_sum = None
    if weak_hash is None:
        hash_sum = hash_file(fp, strong_hash)
    else:
        weak_sum = hash_file(fp, weak_hash)
        if STRONG:
            hash_sum = hash_file(fp, strong_hash)
        else:
            hash_sum = weak_sum
    return hash_sum, weak_sum

