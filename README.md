# pysurvival

A package for Python that wraps some of the survival functions in R
and makes them callable from Python. Most notably, and most useful,
it wraps the _Cox model_ and provides it as a Python class.

The package is developed for Python3.3 and Python2.7. Other versions
may work but I do not make any guarantees.

## Installation instructions

I recommend using __pip__:

    pip install -e .

This should install any requirements automatically as well.
See _requirements.txt_ if you wish to install using your distribution's
normal package manager.



## If you have trouble with Rpy2

pysurvival depends on numpy and rpy2 in python, and in R on survival (also on R of course!).

To get rpy2 to work properly can be tricky. If rpy2 gives you trouble, see if the guide below helps.

To get rpy to find libR.so and other files, the following must be done:

Download and unpack R,
compile as shared library:
./configure --enable-R-shlib
make
make install

Now, with R installed as shared library, you might have to force feed it into your path.
To work around this problem, I had to feed ldconfig with the /usr/lib64/R/lib path by creating a file R-x86_64.conf in /etc/ld.so.conf.d

But make sure it points to your newly compiled and installed R, it must be built as shared library so libR.so and other files can be found.

Now you can install rpy, if shit happens, try forcing the r-home variable
python setup.py build --r-home /usr/local/lib64/R install

Make sure you've got Numpy installed, then the tests should complete successfully!
python -m 'rpy2.tests'
