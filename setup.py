from setuptools import find_packages, setup

classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Operating System :: POSIX :: Linux',
    'Programming Language :: Python :: 3',
    'Topic :: Scientific/Engineering',


]

setup(
    name='postfoam',
    packages=find_packages(include=['postfoam']),
    version='0.1.5',
    description='A python library for postporcessing openfoam simulations.',
    #long_description=open('README.md').read(),
    author='Ioannis Kyritsopoulos',
    author_email='yiagoskyrits@gmail.com',
    keywords=['OpenFOAM', 'postprocessing'],
    classifiers=classifiers,
    install_requires=['numpy','matplotlib','scipy'],
    license='MIT',
)

'''To publish use'''
# python3 setup.py sdist
# twine upload --skip-existing dist/*