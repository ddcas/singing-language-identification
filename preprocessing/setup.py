from setuptools import setup, find_packages

setup(name='preprocessing_spark',
      version='0.1',
      packages=find_packages(),
      description='',
      author='',
      author_email='',
      license='MIT',
      install_requires=[
          'librosa == 0.6.3',
          'scipy == 1.2.1',
          'numpy == 1.16.2',
          'torchvision == 0.2.2',
          'Soundfile == 0.10.2',
          'tensorflow == 1.14'
      ],
      zip_safe=False)
