
LIBRARIES := \
	tpetra \
	belos belostpetra \
	teuchoscore teuchoscomm teuchosnumerics teuchosparameterlist teuchosremainder \
	kokkos kokkosdisttsqr kokkoslinalg kokkosnodeapi kokkosnodetsqr

TOKENS := \
	params_new params_set<scalar_t> params_set<number_t> params_print \
	matrix_new matrix_add_block matrix_complete matrix_norm matrix_apply matrix_solve \
	vector_new vector_add_block vector_getdata vector_dot vector_norm \
	map_new graph_new precon_new export_new release toggle_stdout

null :=
space := $(null) #
comma := ,
tokens_sep := $(subst $(space),$(comma) ,$(TOKENS))
macros := \
	-DTOKENS='"$(tokens_sep)"' \
	-DTOKENFUNCS='$(foreach token,$(tokens_sep),&LibMatrix::$(token))'

libmatrix.mpi: libmatrix.cpp
	mpic++ $< -o $@ -std=c++11 $(macros) $(foreach library,$(LIBRARIES),-l$(library))

clean:
	rm -f libmatrix.mpi *.pyc

test: libmatrix.mpi
	python test

.PHONY: clean test
	
# vim:noexpandtab
