import os


def get_total_file_size(src_folder):  # TODO move to folder operations
    """
    Gets the total file size of a folder (du -s)

    :param src_folder:
    :return:
    """
    total_file_size = 0
    for root, dirs, files in os.walk(src_folder):
        for f in files:
            total_file_size += os.path.getsize(os.path.join(root, f))
    return total_file_size


def append_or_add_set_to_dict(k, val, dic):
    """
    if k is in dic.keys(), appends val to the set of dic[k]
    otherwise, creates the set

    .. warning:: Operates in place for efficiency

    :param k:
    :param val:
    :param dict dic:
    :return: dictionary
    """
    if k in dic.keys():
        dic[k].add(val)
    else:
        dic[k] = {val}


def flatten_dict_to_set(dic):
    """

    .. warning:: Assumes a dic of lists

    :param dic:
    :return:
    """
    out = set()
    for lst in dic.values():
        out.update(lst)
    return out


def get_set_element(st, idx=0):
    return list(st)[idx]


def get_ordered_dict_entry_by_idx(dic, idx):
    k = list(dic.keys())[idx]
    return dic[k]