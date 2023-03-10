from importlib.machinery import SourceFileLoader
import os
from pathlib import Path
import site

site.ENABLE_USER_SITE = True  # workaround https://github.com/pypa/pip/issues/7953

# The following import has to stay after imports from `setuptools`:
# - https://stackoverflow.com/questions/21594925/
#     error-each-element-of-ext-modules-option-must-be-an-extension-instance-or-2-t
from Cython.Build import cythonize  # noqa: E402
import numpy as np  # noqa: E402
from setuptools import find_packages, setup  # noqa: E402

project_root = Path(__file__).parent
code_root = project_root / "medvedi"
os.chdir(str(project_root))
version = SourceFileLoader("version", str(code_root / "metadata.py")).load_module()

with open(project_root / "README.md", encoding="utf-8") as f:
    long_description = f.read()


ext_modules = cythonize(
    [
        str(path)
        # fmt: off
        for path in (
            code_root / "accelerators.pyx",
            code_root / "io.pyx",
            code_root / "native" / "mi_heap_destroy_stl_allocator.pyx",
        )
        # fmt: on
    ],
)
for ext_mod in ext_modules:
    ext_mod.library_dirs = [str(code_root)]


setup(
    name=version.__package__.replace(".", "-"),
    description=version.__description__,
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=version.__version__,
    license="Apache 2.0",
    author="Athenian",
    author_email="vadim@athenian.co",
    url="https://github.com/athenianco/medvedi",
    download_url="https://github.com/athenianco/medvedi",
    packages=find_packages(exclude=["*tests"]),
    ext_modules=ext_modules,
    include_dirs=[np.get_include(), str(code_root)],
    keywords=[],
    install_requires=["numpy>=1.23,<1.24"],
    extras_require={
        "arrow": ["pyarrow"],
    },
    tests_require=[],  # see requirements-test.txt
    package_data={
        "": ["*.md"],
        "medvedi": ["../requirements.txt", "libmimalloc.so*", "mimalloc.h", "*.pyx"],
        "medvedi.native": ["*.pyx", "*.pxd", "*.h"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
