from setuptools import setup, find_packages

setup(
    name='min_llm_server_client',
    version='0.3.10',
    author='Afshin Sadeghi',
    author_email='sadeghi.afshin@gmail.com',
    description='A minimal API server and client for local LLM inference.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'flask',
        'transformers',
        'sentencepiece'
    ],
    python_requires='>=3.6',
)
