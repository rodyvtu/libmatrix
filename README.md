Libmatrix
=========

Trilinos-based matrix backend. In development.

## Installation ##

 1. Install Trilinos following [gjvz.nl](http://gjvz.nl/trilinos.html)
 2. Trilinos will be located in ``$HOME/Computer/trilinos``, define a
    variable that contains this path to tell the compiler where Trilinos
    is. This can be done by adding the following line to
    ``.bashrc``:

        TRILINOSPATH="$HOME/Computer/trilinos"

        export CPATH="$TRILINOSPATH/include:$CPATH"
        export C_INCLUDE_PATH="$TRILINOSPATH/include:$C_INCLUDE_PATH"
        export CPLUS_INCLUDE_PATH="$TRILINOSPATH/include:$CPLUS_INCLUDE_PATH"
        export LIBRARY_PATH="$TRILINOSPATH/lib:$LIBRARY_PATH"
        export LD_LIBRARY_PATH="$TRILINOSPATH/lib:$LD_LIBRARY_PATH"

 3. Restart the terminal to make it load the ``TRILINOSPATH`` variable
 4. Navigate to the libmatrix folder where this ``README.md`` file is
    located.
 5. Type ``make`` to start compiling libmatrix
 6. Install ``mpi4py`` using ``sudo pip install mpi4py``

     >  If you don't have ``pip`` installed, you can install it on Debian
     >  based systems using ``sudo apt-get install python-pip``
