# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='autoID',
    version='0.1.0',
    description='automatic dragonfly identification from wings',
    long_description=readme,
    author='William R. Kuhn',
    author_email='willkuhn@crossveins.com',
    url='https://github.com/willkuhn/odomatic-wings',
    license=license,
    packages=find_packages()
)
