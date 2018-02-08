import os
from collections import OrderedDict

from dry_folder.hash import rehash
from dry_folder.path_node import PathNode
from dry_folder.utils import append_or_add_set_to_dict, get_total_file_size, flatten_dict_to_set


def nodes_list_to_dict(nodes_list):
    nodes_dic = OrderedDict()
    for nd in nodes_list:
        if nd.is_root and nd.is_duplicate:  # a tree root
            append_or_add_set_to_dict(nd.checksum, nd, nodes_dic)  # FIXME: use ordered dict
    return nodes_dic


def file_path_dict_to_nodes_list(dic):
    nodes_list = []
    for k, paths in dic.items():
        for path in paths:
            parent_node_path = os.path.dirname(path)
            if not get_node(nodes_list, parent_node_path):
                nodes_list.append(PathNode(_id=parent_node_path))
            nodes_list.append(PathNode(_id=path, checksum=k, duplicate=True,
                                       parent=get_node(nodes_list, parent_node_path)))
    return nodes_list


def get_node(nodes_list, node_id):
    node = [nd for nd in nodes_list if nd.id == node_id]
    if node:
        return node[0]
    else:
        return None


def trim_singletons(duplicate_files_paths):
    out = OrderedDict()
    for k, val in duplicate_files_paths.items():
        if len(val) > 1:  # not a singleton
            out[k] = val
    return out


class DuplicateFilesFinder(object):

    def __init__(self, src_dir, minimum_file_size):
        self.src_dir = src_dir
        self.minimum_file_size = minimum_file_size
        self.duplicate_files = OrderedDict()
        self.nodes_list = []
        self.nodes_dic = OrderedDict()

    def get_total_file_size(self):
        return get_total_file_size(self.src_dir)

    def get_nodes_dict(self):
        self.nodes_dic = nodes_list_to_dict(self.nodes_list)  # TODO: move to class

    def find_duplicates(self):
        weak_sums = set()
        duplicate_files_paths = OrderedDict()

        file_size_cumsum = 0
        for root, dirs, files in os.walk(self.src_dir):
            file_paths = [os.path.join(root, f) for f in files]
            for fp in file_paths:
                file_size = os.path.getsize(fp)
                file_size_cumsum += file_size
                yield file_size_cumsum
                if file_size < self.minimum_file_size:  # Skip small files (e.g. empty files)
                    continue
                hash_sum, weak_sum = rehash(fp, weak_sums)
                if weak_sum is not None:
                    weak_sums.add(weak_sum)
                append_or_add_set_to_dict(hash_sum, fp, duplicate_files_paths)
        self.duplicate_files = trim_singletons(duplicate_files_paths)

    def find_duplicate_folders(self, remove_subfiles=False):  # FIXME: deal with orphan duplicate files
        src_folder = self.src_dir

        duplicate_file_paths_set = flatten_dict_to_set(self.duplicate_files)
        nodes_list = file_path_dict_to_nodes_list(self.duplicate_files)

        duplicate_dirs_paths_set = set()

        # TODO: use nodes back navigation instead ? or add pbar
        for root, dirs, files in os.walk(src_folder, topdown=False):  # restart bottom up
            file_paths_set = set([os.path.join(root, f) for f in files])
            dir_paths_set = set([os.path.join(root, d) for d in dirs])

            all_files_duplicate = file_paths_set.issubset(duplicate_file_paths_set)
            all_sub_folders_duplicate = dir_paths_set.issubset(duplicate_dirs_paths_set)
            if all_files_duplicate and all_sub_folders_duplicate:
                duplicate_dirs_paths_set.add(root)  # Then the whole directory is duplicate (also if empty)
                root_node = get_node(nodes_list, root)
                if not root_node:
                    root_node = PathNode(_id=root, duplicate=True)
                    nodes_list.append(root_node)
                else:
                    root_node.is_duplicate = True
                for nd in nodes_list:
                    if nd.id in dir_paths_set:  # TODO: check is_duplicate attribute ?
                        nd.parent = root_node
                if remove_subfiles:
                    for k, val in self.duplicate_files.keys():  # OPTIMISE: costly (wrong data structure)
                        self.duplicate_files[k] = list(set(val).difference(file_paths_set))
        self.nodes_list = nodes_list

