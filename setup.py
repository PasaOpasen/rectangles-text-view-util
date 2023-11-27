import setuptools


def parse_requirements(requirements: str):
    with open(requirements) as f:
        return [
            l.strip('\n') for l in f if l.strip('\n') and not l.startswith('#')
        ]


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ordered_rectangles", 
    version="0.0.2",
    author="Demetry Pascal",
    author_email="qtckpuhdsa@gmail.com",
    maintainer='Demetry Pascal',
    description="Environment to python dictionary parser util",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PasaOpasen/rectangles-text-view-util",
    license='MIT',
    keywords=['rectangles'],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    install_requires=parse_requirements('./requirements.txt')
    
)
