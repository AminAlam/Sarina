import os
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from distutils.core import Extension

MINIMAL_DESCRIPTION = '''Sarina is an ASCII art generator written in Python3 and C++. It transforms an input image and a text file containing words and their weights into a unique ASCII art representation. The algorithm behind Sarina is randomized, ensuring that every output is distinct, even for identical inputs.'''

def get_requires():
    """Read requirements.txt."""
    requirements = open('requirements.txt', "r").read()
    return list(filter(lambda x: x != "", requirements.split()))

def read_description():
    """Read README.md and CHANGELOG.md."""
    try:
        with open("README.md") as r:
            description = "\n"
            description += r.read()
        return description
    except Exception:
        return MINIMAL_DESCRIPTION
    


setup(
    name="sarina-cli",
    version="0.0.11",
    author='Amin Alam',
    description='ASCII Art Generator',
    long_description=read_description(),
    long_description_content_type='text/markdown',
    install_requires=get_requires(),
    python_requires='>=3.5',
    license='MIT',
    include_package_data=True,
    url='https://github.com/AminAlam/Sarina',
    keywords="ASCII-Art Word-Cloud-Generator",
    entry_points={
        'console_scripts': [
            'sarina=sarina.__main__:main',
        ]
    },
    # specify the name of the compiled extension as cpp_backend
    ext_modules=[Extension('sarina.lib.cpp_backend', 
                sources=[os.path.join('sarina', 'lib', 'cpp_backend.cpp')])],   
    packages=['sarina']
    )

