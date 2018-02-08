# import collections
import os

from anytree import AnyNode, RenderTree

from dry_folder.hash import strong_hash
from dry_folder.output import CustomAsciiStyle

# Row = collections.namedtuple("Row", ("pre", "fill", "node"))


# class MaxDepthRenderTree(RenderTree):
#     # def __init__(self, node, style=ContStyle(), childiter=list):
#     #     if not isinstance(style, AbstractStyle):
#     #         style = style()
#     #     self.node = node
#     #     self.style = style
#     #     self.childiter = childiter
#
#     def __iter__(self):
#         return self.__next(self.node, tuple())
#
#     @staticmethod
#     def __item(node, continues, style):
#         if not continues:
#             return Row(u'', u'', node)
#         else:
#             items = [style.vertical if cont else style.empty for cont in continues]
#             indent = ''.join(items[:-1])
#             branch = style.cont if continues[-1] else style.end
#             pre = indent + branch
#             fill = ''.join(items)
#             return Row(pre, fill, node)
#
#     def __next(self, node, continues):
#         yield MaxDepthRenderTree.__item(node, continues, self.style)
#         children = node.children
#         if children:
#             lastidx = len(children) - 1
#             for idx, child in enumerate(self.childiter(children)):
#                 for grandchild in self.__next(child, continues + (idx != lastidx, )):
#                     # if grandchild in self.node.root.children:
#                     yield grandchild
#                     # else:
#                     #     continue
#
#     def by_attr(self, attrname="name"):
#         """Return rendered tree with node attribute `attrname`."""
#         def get():
#             for param in self:
#                 # if not param:
#                 #     continue
#                 pre, fill, node = param
#                 attr = getattr(node, attrname, "")
#                 if isinstance(attr, (list, tuple)):
#                     lines = attr
#                 else:
#                     lines = str(attr).split("\n")
#                 yield u"%s%s" % (pre, lines[0])
#                 for line in lines[1:]:
#                     yield u"%s%s" % (fill, line)
#         return "\n".join(get())


class PathNode(AnyNode):
    def __init__(self, _id, checksum='', duplicate=False, parent=None):
        super(PathNode, self).__init__()
        self.id = _id
        self._checksum = checksum
        self.parent = parent
        self.is_duplicate = duplicate

    def __str__(self):
        return RenderTree(self, style=CustomAsciiStyle(prefix='    ')).by_attr('file_path')

    @property
    def checksum(self):
        if self._checksum == '':
            h = strong_hash()
            for child_checksum in sorted([child.checksum for child in self.children]):
                h.update(child_checksum.encode('ascii'))
            # self._checksum = h.hexdigest()
            return h.hexdigest()
        return self._checksum

    @property
    def name(self):
        return os.path.basename(self.file_path)

    @property
    def file_path(self):  # TODO: replace
        return self.id

    @property
    def is_file(self):
        return os.path.isfile(self.id)

    @property
    def is_dir(self):
        return os.path.isdir(self.id)

    @property
    def is_mount(self):
        return os.path.ismount(self.id)

    @property
    def size(self):
        return os.path.getsize(self.id)


