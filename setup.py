from distutils.core import setup
import setuptools

setup(
    name = 'pecok',
    packages = ['pecok'],
	install_requires=[
          'numpy', 'scipy', 'scikit-learn'
      ],
    version = '1.0.1',
    description = 'Implementation of Pecok',
    author='Martin Royer',
    author_email='martinpierreroyer@gmail.com',
	url="https://github.com/martinroyer/pecok",
	license="MIT/X",
	classifiers=[
		  "Development Status :: 4 - Beta",
          "Intended Audience :: Science/Research",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
          "Programming Language :: Python",
          "Programming Language :: Python :: 3",
          "Topic :: Scientific/Engineering",
          ]
)

