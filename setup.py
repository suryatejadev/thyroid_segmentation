from setuptools import setup
from setuptools import find_packages

setup(name='tnseg',
      version='0.1',
      description='Thyorid Nodule Segmentation',
      url='http://github.com/suryatejadev/thyroid-segmentation',
      author='Surya Teja Devarakonda, Santhosh Vangapelli',
      author_email='suryatejadev@cs.umass.edu, svangapelli@cs.umass.edu',
      license='MIT',
      packages=['tnseg', 'tnseg.models'],
      zip_safe=False)
