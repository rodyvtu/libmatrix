# Start of the makefile

libmatrix.mpi: libmatrix.cpp
	mpic++ $< -o $@ \
	  -DDEBUG \
	  -ltpetra \
	  -lteuchoscore \
	  -lteuchoscomm \
	  -lteuchosnumerics \
	  -lteuchosparameterlist \
	  -lteuchosremainder \
	  -lkokkos \
	  -lkokkosdisttsqr \
	  -lkokkoslinalg \
	  -lkokkosnodeapi \
	  -lkokkosnodetsqr

clean:
	rm -f libmatrix.mpi *.o

test: libmatrix.mpi
	python test

.PHONY: clean test
	
# vim:noexpandtab
