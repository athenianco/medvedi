# MEDVeDI [![Build status](https://github.com/athenianco/medvedi/actions/workflows/push.yml/badge.svg)](https://github.com/athenianco/medvedi/actions/workflows/push.yml) [![codecov](https://codecov.io/gh/athenianco/medvedi/branch/master/graph/badge.svg?token=rkTwlIlCaI)](https://codecov.io/gh/athenianco/medvedi) [![Latest Version](https://img.shields.io/pypi/v/medvedi.svg)](https://pypi.python.org/pypi/medvedi) [![Python Versions](https://img.shields.io/pypi/pyversions/medvedi.svg)](https://pypi.python.org/pypi/medvedi) [![License](https://img.shields.io/pypi/l/medvedi.svg)](https://github.com/athenianco/medvedi/blob/main/LICENSE)

![logo](docs/logo.png)

Memory Efficient Deconstructed Vectorized Dataframe Interface.

Design goals:

- Favor performance over nice syntax features. Sacrifice fool-proof for efficient zero-copy operations.
- Ensure ideal micro-performance and optimize for moderate data sizes (megabytes).
- The use-case is API server code that you write once and execute many times.
- Try to stay compatible with the Pandas interface. There is no `Series`, however.
- Rely on numpy.
- Friends with Arrow.
- Frequently release GIL and depend on native extensions doing unsafe things.
- Test only CPython and Linux.
- Support only x86-64 CPUs with AVX2.
- Support only Python 3.10+.
- 100% test coverage.

Otherwise, you should be way better with regular Pandas.

Medvedi used to be heavily used in production at [Athenian](https://athenian.co).
