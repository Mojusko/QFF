from setuptools import setup

setup(name='QFF',
      version='0.0.1',
      description='Fourier Features Embedding for Python',
      url='',
      author='Mojmir Mutny',
      author_email='mojmir.mutny@inf.ethz.ch',
      license='MIT Licence',
      modules=['embedding.py'],
	    zip_safe=False,
      install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
          'sklearn',
          'tensorflow',
          'torch',
          'torchvision',
          'mpmath',
      ])
