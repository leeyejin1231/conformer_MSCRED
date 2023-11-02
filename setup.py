from setuptools import setup, find_packages

with open('requirements.txt', "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="conformer-MSCRED",
    version="0.0.1",
    author="jaeyeonkim",
    author_email="0310kjy@gmail.com",
    description="conformer-mscred",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/leeyejin1231/conformer_MSCRED",
    packages=find_packages(),
    install_requires=requirements,
)