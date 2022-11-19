from setuptools import setup, find_packages


def find_version():
    version_file = 'adabins/__init__.py'
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']

 

setup(
    name='adabins',
    version=find_version(),
    description='A library for deep learning person re-ID in PyTorch',
    author='Kaiyang Zhou',
    license='MIT',
    url='https://github.com/KaiyangZhou/deep-person-reid',
    packages=find_packages(),
    keywords=['Person Re-Identification', 'Deep Learning', 'Computer Vision'],
    
)
