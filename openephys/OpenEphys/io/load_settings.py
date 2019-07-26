__author__ = 'blota'

"""
Load data from the settings.xml file
"""

import xml.etree.ElementTree
import pandas as pd


def read_settings(xml_file):
    """Read the settings.xml file

    For now just stupidly read and put part that seemed interesting in dataframes
    :param xml_file:str
    :return:
    """
    root = xml.etree.ElementTree.parse(xml_file).getroot()

    for child in root.getchildren():
        if child.tag == 'INFO':
            info = pd.Series(name='xml_info',
                             data=[c.text for c in child.getchildren()],
                             index=[c.tag for c in child.getchildren()])
        elif child.tag == 'SIGNALCHAIN':
            signal_chain = {}
            channel_inf = {}  # should be a single one per file? Just in the FGPA node?
            channel_prop = {}
            other_stuff = []
            for what in child.getchildren():
                if what.tag == 'PROCESSOR':
                    attr = what.attrib
                    name = attr.get('NodeId')
                    signal_chain[name] = pd.Series(name=name,
                                                   data=attr.values(),
                                                   index=attr.keys())
                    series_list = []
                    for ch in what.getchildren():
                        if ch.tag == 'CHANNEL_INFO':
                            channel_inf[name] = pd.DataFrame([pd.Series(c.attrib) for c in ch.getchildren()])
                        elif ch.tag == 'CHANNEL':
                            chname = ch.get('name')
                            s = pd.Series(name=chname)
                            for leaf in ch.getchildren():
                                assert len(leaf.getchildren()) == 0
                                lname = leaf.tag.lower()
                                for k, v in leaf.attrib.items():
                                    s['_'.join((lname, k))] = v
                                s['number'] = ch.get('number')
                                s['name'] = ch.get('name')
                            series_list.append(s)
                    channel_prop[name] = pd.DataFrame(series_list)
                else:
                    other_stuff.append([what.tag, what.attrib, what.getchildren()])
    assert len(channel_inf) == 1  # it should be the case if I understood what it is
    channel_inf = channel_inf.values()[0]
    return info, other_stuff, pd.DataFrame(signal_chain.values()), channel_inf, channel_prop
