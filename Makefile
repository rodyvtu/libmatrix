
LIBRARIES := \
	tpetra \
	belos belostpetra \
	teuchoscore teuchoscomm teuchosnumerics teuchosparameterlist teuchosremainder \
	kokkos kokkosdisttsqr kokkoslinalg kokkosnodeapi kokkosnodetsqr

FUNCTIONS := \
	params_new params_set<scalar_t> params_set<number_t> params_print \
	matrix_new_static matrix_new_dynamic matrix_add_block matrix_complete matrix_norm matrix_apply \
	vector_new vector_fill vector_add_block vector_getdata vector_dot vector_norm vector_complete vector_or vector_copy vector_axpy vector_as_setzero_operator vector_nan_from_supp \
	linearproblem_new linearproblem_set_hermitian linearproblem_add_left_precon linearproblem_add_right_precon linearproblem_solve \
	map_new graph_new precon_new export_new release set_verbosity

null :=
space := $(null) #
comma := ,
macros := \
	-DFUNCNAMES='$(subst $(space),$(comma) ,$(foreach token,$(FUNCTIONS),"$(token)"))' \
	-DFUNCS='$(subst $(space),$(comma) ,$(foreach token,$(FUNCTIONS),&LibMatrix::$(token)))'

libmatrix.mpi: libmatrix.cpp
	mpic++ $< -o $@ -std=c++11 $(macros) $(foreach library,$(LIBRARIES),-l$(library))

clean:
	rm -f libmatrix.mpi *.pyc

test: libmatrix.mpi
	python test

.PHONY: clean test
	
# vim:noexpandtab
