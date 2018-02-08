import urwid

from dry_folder.path_node import PathNode
from dry_folder.ui_theme import palette


def make_btn(title, click_callback, callback_arg=None):
    button = urwid.Button(title)
    if callback_arg is not None:
        urwid.connect_signal(button, 'click', click_callback, callback_arg)
    else:
        urwid.connect_signal(button, 'click', click_callback)
    return button


def make_title(title_text):
    return urwid.AttrMap(urwid.Text(title_text), 'title')


class CustomPBar(urwid.ProgressBar):
    """ProgressBar with extended inline text"""
    def __init__(self, normal, complete, done=100, satt=None):
        super(CustomPBar, self).__init__(normal, complete, 0, done, satt)

    def get_text(self):
        """Return extended progress bar text"""
        percent = int(self.current * 100 / self.done)
        return 'Processing {0:.2f} of {1:.2f} ({2}%)'.format(self.current, self.done, percent)


class CursesUi(object):
    def __init__(self, file_finder):
        """

        :param dry_folder.file_handler.DuplicateFileFinder file_finder:
        """
        self.title = u'Dry folder'
        self.duplicate_files_finder = file_finder
        self.file_path_window = self.make_file_path_window()
        self.main_window = self.make_main_window()

    def make_main_window(self):
        header = urwid.AttrMap(urwid.Text(self.title, align='center', wrap='space', layout=None), 'title')
        background = urwid.AttrMap(urwid.SolidFill(u' '), 'bg')
        pbar = CustomPBar('', '', done=self.duplicate_files_finder.get_total_file_size())
        self.pbar = pbar
        footer = urwid.Columns([pbar, urwid.Text('Navigation: Enter, down, right; Q=quit')])
        frame = urwid.Frame(background, header=header, footer=footer)

        main_window = urwid.Overlay(self.file_path_window, frame,  # TODO: remove magic numbers
                                    align='center', width=('relative', 80),
                                    valign='middle', height=('relative', 70),
                                    min_width=20, min_height=9)
        return main_window

    def run(self):
        self.loop = urwid.MainLoop(self.main_window, palette=palette, unhandled_input=self.handle_input)
        self.loop.run()

    def refresh(self):
        self.loop.draw_screen()

    def quit(self):
        raise urwid.ExitMainLoop()

    def handle_input(self, key):
        if key in ('q', 'Q'):
            self.quit()
        if key in ('s', 'S'):  # TODO: find wayt to start automatically
            self.get_duplicate_nodes()

    def make_file_path_window(self):
        main_nodes = list(self.duplicate_files_finder.nodes_dic.values())
        if self.duplicate_files_finder.nodes_dic:
            file_path_window = urwid.Padding(self.make_file_menu(u'Redundant folders', main_nodes),
                                             left=2, right=2)
        else:
            blank = urwid.ListBox(urwid.SimpleFocusListWalker([]))
            file_path_window = urwid.Padding(blank,
                                             left=2, right=2)
        return file_path_window

    def get_duplicate_nodes(self):
        for progress in self.duplicate_files_finder.find_duplicates():
            self.pbar.set_completion(progress)
            self.pbar.render((10, ))
            self.refresh()
        self.duplicate_files_finder.find_duplicate_folders()  # TODO: add pbar
        self.duplicate_files_finder.get_nodes_dict()
        self.file_path_window.original_widget = self.make_file_path_window()
        self.refresh()

    def make_file_menu(self, title, nodes):

        def node_chosen(_, nd):
            if not nd.is_leaf:
                self.file_path_window.original_widget = urwid.Padding(self.make_file_menu(nd.file_path, nd.children),
                                                                      left=2, right=2)

        def go_back(_, nodes):  # FIXME: buggy
            nd = nodes[0].parent  # All siblings share parent
            try:
                _title = nd.parent.name
                self.file_path_window.original_widget = urwid.Padding(self.make_file_menu(_title, nd.siblings),
                                                                      left=2, right=2)
            except AttributeError:
                self.file_path_window.original_widget = self.make_file_path_window()

        title_widget = make_title(title)
        body = [title_widget,
                urwid.Divider()  # Blank line
                ]
        for i, nd in enumerate(nodes):
            if isinstance(nd, PathNode):
                if i == 0:
                    back_btn = make_btn('..', go_back, nodes)
                    body.insert(2, back_btn)
                dir_symbol = '+' if not nd.is_leaf else ''
                btn_title = '{} {}'.format(dir_symbol, nd.name)
                button = make_btn(btn_title, node_chosen, nd)  # TODO: distinguish from copy
                body.append(urwid.AttrMap(button, None, focus_map='reversed'))
            elif isinstance(nd, (list, set)):  # The root
                for j, node in enumerate(nd):
                    button = make_btn(node.name, node_chosen, node)
                    body.append(urwid.AttrMap(button, None, focus_map='reversed'))
                body.append(urwid.Divider())
        return urwid.ListBox(urwid.SimpleFocusListWalker(body))

