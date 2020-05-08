from distutils.core import setup


def readme():
    """Import the README.md Markdown file and try to convert it to RST format."""
    try:
        import pypandoc
        return pypandoc.convert('README.md', 'rst')
    except(IOError, ImportError):
        with open('README.md') as readme_file:
            return readme_file.read()


setup(
    name='article-analysis',
    version='0.1',
    description='Analysis of the Data Science articles',
    long_description=readme(),
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    url='https://github.com/dudikbender/article-popularity-nlp',
    author='David Bender',
    author_email='bender2242@gmail.com',
    license='MIT',
    packages=['article-analysis'],
)