import argparse
import os
import sys

sys.path.append(os.path.abspath('.'))

from dry_folder.file_handler import DuplicateFilesFinder
from dry_folder.curses_ui import CursesUi

STRONG = True  # TODO: replace by args.strong_algo


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('source_folder', type=str)  # TODO: deal with multiple source folders
    parser.add_argument('-s', '--minimum-byte-size', dest='minimum_byte_size', type=int, default=2,
                        help='The minimum size in bytes a file has to be to qualify for the search.')
    parser.add_argument('-x', dest='same_filesystem', action='store_true',
                        help='Do not cross filesystem boundaries, i.e. only count files and '
                             'directories on the same filesystem as the directory being scanned.')  # TODO: use
    # TODO: add option matching_names Path.Node comparison = name if is_file
    # TODO: add args.strong (to add sha1 option)
    # TODO: add imperfect comparison (> x% shared files)
    # FIXME: with min size: should only affect print (folder with empty files should still be possible duplicate)
    args = parser.parse_args()
    if not os.path.isdir(args.source_folder):
        print("Source folder {} does not exist".format(args.source_folder))
        sys.exit('1')
    return args


def main():
    args = get_args()
    d = DuplicateFilesFinder(args.source_folder, args.minimum_byte_size)

    curses_ui(args, d)


def curses_ui(args, duplicate_files_finder):
    ui = CursesUi(duplicate_files_finder)
    try:
        ui.run()
    except KeyboardInterrupt:
        pass
    ui.quit()


if __name__ == '__main__':
    # if not use_curses:
    #  with tqdm(total=self.get_total_file_size(), desc='Performing file checksums', unit='B', unit_scale=True) as pbar:
    main()
    # if not use_curses:
    #     print_nodes(nodes_dic)
