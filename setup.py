from setuptools import setup, find_packages


# Specify version
VERSION = '1.0.0.dev3'


# Preprocess
print('Running setup.py for roma-v' + VERSION + ' ...')
print('-' * 79)


# Run setup
def readme():
  with open('README.md', 'r') as f:
    return f.read()

# Submodules will be included as package data, specified in MANIFEST.in
setup(
  name='pictor',
  packages=find_packages(),
  include_package_data=True,
  version=VERSION,
  description='A general data viewer for scientific study',
  long_description=readme(),
  long_description_content_type='text/markdown',
  author='William Ro',
  author_email='luo.wei@zju.edu.cn',
  url='https://github.com/WilliamRo/pictor',
  download_url='https://github.com/WilliamRo/pictor/tarball/v' + VERSION,
  license='Apache-2.0',
  keywords=['plot'],
  classifiers=[
    "Development Status :: 2 - Pre-Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: Apache Software License",
    "Natural Language :: English",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 2",
    "Programming Language :: Python :: 3",
    "Topic :: Utilities",
  ],
)
