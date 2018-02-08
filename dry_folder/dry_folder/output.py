from anytree import AbstractStyle


class CustomAsciiStyle(AbstractStyle):

    def __init__(self, prefix='\t\t'):
        """
        Custom Ascii style.

        Node('/root')
        |-- Node('/root/sub0')
        |   |-- Node('/root/sub0/sub0B')
        |   +-- Node('/root/sub0/sub0A')
        +-- Node('/root/sub1')
        """
        seps = [u'|   ', u'|-- ', u'+-- ']
        seps = ['{}{}'.format(prefix, sep) for sep in seps]
        super(CustomAsciiStyle, self).__init__(*seps)


def print_nodes(nodes_dic):
    for nodes_group in nodes_dic.values():
        if len(nodes_group) == 1:
            nd = nodes_group.pop()
            print("Orphan duplicate folder at {}\n\t{}".format(nd.id, nd))
        else:
            print('Folders or files {} with checksum "{}" are identical'.format([nd.id for nd in nodes_group],
                                                                                [nd.checksum for nd in nodes_group]))
            for nd in nodes_group:
                print(" {}".format(nd))

# TODO sort by folder and allow browsing

