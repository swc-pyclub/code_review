import click
import os, time
from .__init__ import read_python_file, moveinsubfolder
from .convert2dat import convertfromprofile
from .concatenate_dat_files import concatenate_dat_files
from .create_phy_prm_files import create_prm_for_folder
from .delete_temp_files import delete_temp_files
import pandas as pd


@click.group()
def cli():
    pass


@cli.command()
@click.option('--recursive', is_flag=True, help='Do all directory below target dir (default False)')
@click.argument('profile', type=click.Path(exists=True))  # , help='Profile file')
@click.option('--target_path', default=None, type=click.Path(exists=True, file_okay=False,
                                               dir_okay=True),
              help='Where to write dat files, will overwrite value found in profile file')
@click.option('--source_path', default=None, help='Where are the OE files,' +
              'will overwrite value found in profile file',
              type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--log/--no-log', default=True, help='Create a log file in the target dir')
@click.option('--create_csv', is_flag=True, default=True,
              help='Create a csv in target path to list written dat files')
@click.option('--pyfunc', default=None, help='Path to a file with a function equals to the file name. '
                                             'This function will be called on the data before writing')
@click.option('--w_wo_func', is_flag=True, default=False,
              help='If pyfunc is not None will save a file with and without applying func')
@click.option('--prefix', default='', help='Prefix to add before converted file name')
@click.option('--replace_folder_name', '-r', nargs=2, multiple=True, type=str,
              help='"in_folder" "out_folder" pair. Will replace any occurence of "in_folder"' +
                   'in source path by "out_folder" in the target path')


def convert2dat(profile, log, recursive, create_csv, replace_folder_name, prefix,
                pyfunc, w_wo_func, target_path, source_path):
    """Convert OpenEphys files to dat.

    profile is the path to a python-like file containing the profile information:



    process (int):

              -  OE process to use for convert2dat

    target_path  (str):

              -  where to write dat files

    source_path  (str):

              -  where are the OE files

    num_chans (None or int):

              -  used only if num_shanks is an integer, total number of channels

    ref  (None, 'CMR' or 'CAR'):

             -  reference the data before converting

    detekt_path (str):

              -  path for dat files used to detect spikes. Is usually target_path
    if you don't move files around. Will be hard coded in the prm file (should I change that??)

    num_shanks (dict or int):

              -  nums_shanks is an int or a dict. If it is an int, the `num_chans`
    channels are split in `num_shanks` equal parts and writen in different files (useful to split
    shanks or tetrodes). If it is a dict, then keys are a identifier, that will be append to the file
    name and values are the ORDERED list of channels to write in this file (see template profile for ex).

    probe_files (dict):

              -  dictionary with  keys equal to shank IDs and values to probe files
    """

    prof = read_python_file(profile)
    if source_path is not None:
        prof['source_path'] = source_path
    if target_path is not None:
        prof['target_path'] = target_path

    if not log:
        echo = click.echo
    else:
        path2log = os.path.join(prof['target_path'],
                                'convert2dat_%s_log.txt' % time.strftime('%y%m%d_%H%M%S'))

        def echo(txt):
            click.echo(txt)
            with open(path2log, 'a') as F:
                F.write(txt + '\n')

        echo('Logging in %s' % path2log)

    if pyfunc is not None:
        p = read_python_file(pyfunc)
        fname = os.path.split(pyfunc)[-1]
        pyfunc = p[fname]
        if w_wo_func:
            def nof(x): return x

            pyfunc = {fname: pyfunc, '': nof}

    echo('Starting convert2dat')
    echo('')
    echo('Parameters:')
    echo('\n'.join('%s: %s' % (k, v) for k, v in prof.items()))
    echo('')
    convertfromprofile(profile_dict=prof, log_func=echo, recursive=recursive, funcpy=pyfunc,
                       create_csv=create_csv, replace_folder_name=replace_folder_name, prefix=prefix)


@cli.command()
@click.argument('csv', type=click.Path(exists=True))  # , help='Profile file')
@click.option('--log/--no-log', default=True)
@click.option('--root_col', default='Home directory',
              help='Name of the column where to find the path to the file (default "Home directory")')
@click.option('--filename_col', default='File name',
              help='Name of the column where to find the name of the dat file (default "File name")')
@click.option('--sep', default=',',
              help='Separator used in the csv file (default to coma)')
@click.option('--delimiter', default=None,
              help='Filed delimiter used in the csv file (default to None)')
@click.option('--col_to_group', '-c', nargs=2, multiple=True, type=str,
              help='"Column_name" "alias" pair. Only files with exact same value in' +
                   'this/these column(s) will be concatenated together. The value of the column(s)' +
                   ' will be appended to the file name prefixed by "alias"')
@click.option('--order', '-o', default='Ordering key',
              help='Column that will be used to order the files (default "Ordering key")')
@click.option('--numsample', default='Recording length',
              help='Column with the number of samples per dat file (default "Recording length")')
@click.option('--numchan', default='Num channels',
              help='Column with the number of channels per dat file (default "Num channels")')
@click.option('--prefix', '-p', default=None,
              help='Columns that will be used to prefix the file name')
@click.option('--subfolder', default=None,
              help='Name of the subfolder in which creating the concatenated file. Subfolder should exist')
def concatenatedat(csv, log, root_col, filename_col, sep, delimiter, order, col_to_group, prefix,
                   numchan, numsample, subfolder):
    """Given a csv, concatenate dat files.

    The csv must have a single header line and then one line per file.
    A template can be found in OpenEphys/oe_clustering

    All the files that have the same values for "col_to_group" are concatenated together
    A csv file with the same name as the output dat file is also created. If numchan and
    numsamples columns exists, then this csv will contain the info about the border of
    each file in the concatenated dat file.

    root_col can be:
     1) the name of a column in which the root path is used. Only the first cell will be used.
     2) an absolute path starting with "/" that will be used for all conversions

    Example:

        Given a csv like that

    File name, Exp,  Shank,  Home directory,Ordering key, Pos,  Mouse
    file1.dat, exp1, tet1,    /tmp,            1,         1,      M1
    file2.dat, exp1, tet2,    /tmp,            2,         1,      M1
    file3.dat, exp1, tet1,    /tmp,            4,         1,      M1
    file4.dat, exp1, tet2,    /tmp,            5,         1,      M1
    file5.dat, exp1, tet1,    /tmp,            6,         2,      M1
    file6.dat, exp2, tet1,    /tmp,            3,         1,      M1


        oe_clustering concatenatedat myfile.csv -c Shank shk -c Pos pos

        # will execute:

        cat /tmp/file1.dat /tmp/file6.dat /tmp/file3.dat > /tmp/concatenated_pos_1-shk_tet1.dat

        cat /tmp/file2.dat /tmp/file4.dat > /tmp/concatenated_pos_1-shk_tet2.dat

        cat /tmp/file5.dat > /tmp/concatenated_pos_2-shk_tet1.dat

        # note the inversion between file6 and file3 following the ordering key

        oe_clustering concatenatedat myfile.csv -c Shank shk -c "Pos" pos -c "Exp" exp

        # will move file6 that was from another exp

        cat /tmp/file1.dat /tmp/file3.dat > /tmp/concatenated_pos_1-exp_exp1-shk_tet1.dat

        cat /tmp/file2.dat /tmp/file4.dat > /tmp/concatenated_pos_1-exp_exp1-shk_tet2.dat

        cat /tmp/file6.dat > /tmp/concatenated_pos_1-exp_exp2-shk_tet1.dat

        cat /tmp/file5.dat > /tmp/concatenated_pos_2-exp_exp1-shk_tet1.dat

        oe_clustering concatenatedat myfile.csv -p Mouse -c Shank myshk

        # will add a prefix to output names

        cat /tmp/file1.dat /tmp/file6.dat /tmp/file3.dat /tmp/file5.dat > /tmp/M1_concatenated_myshk_tet1.dat

        cat /tmp/file2.dat /tmp/file4.dat > /tmp/M1_concatenated_myshk_tet2.dat


    """
    if not len(col_to_group):
        raise click.BadArgumentUsage('Need a column to group')

    if not log:
        echo = click.echo
    elif isinstance(log, str):
        path2log = log

        def echo(txt):
            click.echo(txt)
            with open(path2log, 'a') as F:
                F.write(txt + '\n')
    else:
        root_path, csv_name = os.path.split(os.path.abspath(csv))
        path2log = os.path.join(root_path,
                                'concatdatfile_%s_output.csv' % time.strftime('%y%m%d_%H%M%S'))

        def echo(txt):
            click.echo(txt)
            with open(path2log, 'a') as F:
                F.write(txt + '\n')

    csv_df = pd.read_csv(csv, delimiter=delimiter, sep=sep, skipinitialspace=True)

    not_found = 0
    for col in [filename_col, order] + [c[0] for c in col_to_group]:
        if col not in csv_df.columns:
            echo('%s not found in header' % col)
            not_found += 1
    # For root_col, check if 1) it is in the header or 2) it's an abs path
    if (root_col not in csv_df.columns) and (not root_col.startswith('/')):
        echo("%s not found in header and is not an absolute path." % root_col)
        not_found += 1

    if not_found:
        echo('%i column(s) not found' % not_found)
        echo('Header was: %s' % csv_df.columns)
        raise click.Abort()

    echo('Starting concatenation')
    for gp_name, gp_df in csv_df.groupby([c[0] for c in col_to_group]):
        concatenate_dat_files(gp_df, col_to_group, order, root_col,
                              filename_col, numsample, numchan,
                              log_func=echo, prefix=prefix, subfolder=subfolder)
    echo('Done!')
    return


@cli.command()
@click.argument('folder', type=click.Path(exists=True))  # , help='Profile file')
@click.option('--log/--no-log', default=True)
@click.option('--do_only', default=None, type=str,
              help='Move only files containing `do_only` in their name')
@click.option('--skip_lonely/--no-skip_lonely', default=True,
              help='Do not move files that are already alone (with that ext) in their folder (default True)')
@click.option('--ext', default='.dat', type=str,
              help='Move only files with that file extension (default ".dat")')
@click.option('--recursive', is_flag=True,
              help='Do all directory below target dir (default False)')
def move_in_subfolder(folder, log, recursive, ext, do_only, skip_lonely):
    """Move files in individual folders."""
    if log:
        path2log = os.path.join(folder,
                                'movedat_%s.txt' % time.strftime('%y%m%d_%H%M%S'))

        def echo(txt):
            click.echo(txt)
            with open(path2log, 'a') as F:
                F.write(txt + '\n')
    else:
        echo = click.echo
    moveinsubfolder(folder, ext, do_only=do_only, log_func=echo, recursive=recursive,
                    skip_lonely=skip_lonely)
    return


@cli.command()
@click.argument('profile', type=click.Path(exists=True))  # , help='Profile file')
@click.option('--do_only', default=None, type=str,
              help='Do only files containing `do_only` in their name')
@click.option('--ext', default='.dat', type=str,
              help='Create param for files with that extension (default .dat)')
@click.option('--recursive', is_flag=True,
              help='Do all directory below target dir (default False)')
@click.option('--skip_warning', is_flag=True,
              help='Do not print warning when multiple prm are created in the same directory (default False)')
@click.option('--overwrite', is_flag=True,
              help='Replace existing prm files (skip otherwise)')
def create_param_file(profile, recursive, ext, do_only, skip_warning, overwrite):
    """Create a .prm file to use with klusta or phy based on template"""
    prof = read_python_file(profile)
    path2log = os.path.join(prof['detekt_path'],
                            'create_param_file_%s.txt' % time.strftime('%y%m%d_%H%M%S'))

    def echo(txt):
        click.echo(txt)
        with open(path2log, 'a') as F:
            F.write(txt + '\n')

    if prof['param_template'] is None:
        from . import param_template as params_module
        origin_params = dict([(k, getattr(params_module, k)) for k in ['prb_file', 'spikedetekt',
                                                                       'klustakwik2', 'traces']])
    else:
        origin_params = read_python_file(prof['param_template'])

    echo('Starting prm file creation\n')

    echo('Template params are:')
    for k, v in origin_params.items():
        echo(k + ':    ' + str(v))
    echo('')
    create_prm_for_folder(prof['detekt_path'], prof, origin_params=origin_params, ext=ext, log_func=echo,
                          skip_warning=skip_warning, do_only=do_only, overwrite=overwrite, recursive=recursive)


@cli.command()
@click.argument('folder', type=click.Path(exists=True))  # , help='Profile file')
@click.option('--recursive', is_flag=True,
              help='Do all directory below target dir (default False)')
@click.option('--detect/--no-detect', default=True,
              help='Run the spike detection (default True)')
@click.option('--cluster/--no-cluster', default=True,
              help='Run the automatic clustering (default True)')
@click.option('--overwrite/--no-overwrite', default=True,
              help='Replace kwik files that already have been processed (default True)')
@click.option('--exclude', default=None, type=str,
              help='Do not process files that have `exclude` in their name (default None)')
def run_phy(folder, recursive, overwrite, detect, cluster, exclude):
    """Call phy on a prm or kwik file"""
    import phy

    path2log = os.path.join(folder,
                            'run_phy_%s.txt' % time.strftime('%y%m%d_%H%M%S'))

    def echo(txt):
        click.echo(txt)
        with open(path2log, 'a') as F:
            F.write(txt + '\n')

    phycmd = 'klusta'
    if detect and cluster:
        options = []
        ext = '.prm'
    elif detect:
        options = ['--detect-only']
        ext = '.prm'
    elif cluster:
        options = ['--cluster-only']
        ext = '.kwik'
    else:
        click.Abort('detect and/or cluster must be true')

    def do(path, opt):
        options = opt[:] # make a copy
        good_file = [f for f in os.listdir(path) if f.endswith(ext)]
        if len(good_file) > 1:
            click.Abort('More than one %s file in %s ... I don\'t like that. Check it' % (ext, path))
        if not good_file:
            return

        file_name = good_file[0]
        echo(file_name)
        if (exclude is not None) and (exclude in file_name):
            echo('Excluding %s'%file_name)
            return


        options.append('--output-dir %s' % path)

        if overwrite:
            options.append('--overwrite')

        bash_command = '%s %s '%(phycmd, os.path.join(path, file_name))
        bash_command += ' '.join(options)
        echo('Executing %s' % bash_command)
        echo('\n')
        os.system(bash_command)

    if recursive:
        for dirpath, dirnames, filenames in os.walk(folder):
            indent = dirpath[len(folder):].count(os.sep)
            echo('  ' * indent + 'Descending into %s' % dirpath)
            do(dirpath, options)
    else:
        do(folder, options)

@cli.command()
@click.argument('folder', type=click.Path(exists=True))  # , help='Profile file')
@click.option('--recursive', is_flag=True,
              help='Do all directory below target dir (default False)')
@click.option('--ask_conf/--no-ask_conf', default=True,
              help='Ask for confirmation before removing a folder/file (default True)')
@click.option('--kwik_temp/--no-kwik_temp', default=True,
              help='Remove kwik temp directory: .spikdetekd, .phy and .klustakwik2 (default True)')
def clean_folder(folder, recursive, kwik_temp, ask_conf):
    """Clean a folder by deleting useless files"""
    if recursive:
        for dirpath, dirnames, filenames in os.walk(folder):
            indent = dirpath[len(folder):].count(os.sep)
            click.echo('  ' * indent + 'Descending into %s' % dirpath)
            if kwik_temp:
                delete_temp_files(dirpath, delete_without_asking=not ask_conf, confirm_func=click.confirm,
                                  log_func=click.echo)
    else:
        if kwik_temp:
            delete_temp_files(folder, delete_without_asking=not ask_conf, confirm_func=click.confirm,
                              log_func=click.echo)
