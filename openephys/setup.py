from setuptools import setup, find_packages

setup(
    name='OpenEphys',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'Click',
        'pandas',
        'numpy',
    ],
    entry_points='''
        [console_scripts]
            oe_clustering = OpenEphys.oe_clustering.cli:cli
    ''',
)
