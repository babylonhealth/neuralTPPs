from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

setup(
    name='tpp',
    version='0.0.1',
    description='Playing around with TPPs.',
    long_description=readme,
    author='Babylon ML team',
    author_email='loss.goes.down@babylonhealth.com',
    url='https://github.com/babylonhealth/neural-tpps',
    install_requires=['tqdm'],
    dependency_links=[],
    packages=find_packages(exclude=('tests', 'docs'))
)
