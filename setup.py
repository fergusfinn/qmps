from setuptools import setup

setup(name='qmps',
      version='0.1',
      description='mps on quantum computers',
      author='Fergus Barratt',
      author_email='fergus.barratt@kcl.ac.uk',
      url='https://github.com/fergusbarratt/quantum-matrix-product-states.git/',
      license='GPL',
      packages=['qmps'],
      install_requires=[
          'numpy', 
          'scipy',
          'matplotlib', 
          'tqdm',
          'scikit-optimize']
      ,
      zip_safe=False
      )
