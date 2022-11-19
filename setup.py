from setuptools import find_packages, setup


def find_version():
    version_file = "adabins/__init__.py"
    with open(version_file, "r") as f:
        exec(compile(f.read(), version_file, "exec"))
    return locals()["__version__"]


setup(
    name="adabins",
    version=find_version(),
    description="A library monocular depth estimation",
    author="Vladimir Yugay",
    license="MIT",
    url="https://github.com/VladimirYugay/AdaBins",
    packages=find_packages(),
    keywords=["Deep Learning", "Computer Vision", "Depth Estimation"],
)
