"""Read a CSV file and return a pandas dataframe

Also contains type attribute that I use in my protocol csv
"""
import os
import pandas as pd

j_ = os.path.join


def read_csv(fname, type_dict=dict(), index=None):
    """Read a CSV file with one header line and return a pandas dataframe

    Force type if `type_dict` is provided (`type_dict` has the header values as key and type as value)
    Set the index of the dataframe to `index` if not None"""

    with open(fname) as f:
        header = f.readline().strip().split(',')
        lines = [l.strip().split(',') for l in f]

    temp_dict = {}
    for i, h in enumerate(header):
        if h in type_dict:
            dt = type_dict[h]
            if isinstance(dt, str) and dt.startswith('date:'):
                s = pd.Series([l[i].strip('"') for l in lines])
                temp_dict[h] = pd.to_datetime(s, format=dt.split(':')[1])
            else:
                try:
                    temp_dict[h] = pd.Series([l[i].strip('"') for l in lines],
                                             dtype=dt)
                except ValueError:
                    print('Got %s for %s, cannot convert to %s' % ([l[i].strip('"') for l in lines], h, dt))
        else:
            temp_dict[h] = pd.Series([l[i].strip('"') for l in lines])
    df = pd.DataFrame(temp_dict)

    if index is not None:
        return df.set_index(index)
    return df
