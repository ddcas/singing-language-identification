from setuptools import setup, find_packages

REQUIRED_PACKAGES = [
	'tensorflow == 1.14',
	'torchvision == 0.2.2',
	'matplotlib == 2.2.4',
      'stacklogging == 0.1.2',
      'Soundfile == 0.10.2'
]
setup(name='trainer',
      version='0.1',
      packages=find_packages(),
      author='',
      author_email='',
      license='MIT',
      install_requires=REQUIRED_PACKAGES,
      include_package_data=True,
      zip_safe=False,
      description='My pytorch trainer application package.'
)
