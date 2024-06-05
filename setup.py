from setuptools import setup, find_packages

setup(
    name='storch',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        # List your dependencies here, e.g., 'torch', 'numpy'
    ],
    include_package_data=True,
    description='Pytorch starter project containing training code and utils like configuration and logging.',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/storch',  # Replace with your actual GitHub URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
