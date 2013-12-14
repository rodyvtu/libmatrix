FUNCTIONS := \
	params_new params_set<scalar_t> params_set<number_t> params_print \
	matrix_new_static matrix_new_dynamic matrix_add_block matrix_complete matrix_norm matrix_toarray matrix_add \
	vector_new vector_fill vector_add_block vector_toarray vector_dot vector_norm vector_complete vector_or vector_copy vector_update vector_nan_from_supp vector_imul \
	linearproblem_new linearproblem_set_hermitian linearproblem_set_precon linearproblem_solve \
	map_new graph_new precon_new export_new release set_verbosity \
	operator_apply matrix_constrained

null :=
space := $(null) #
comma := ,
macros := \
	-DFUNCNAMES='$(subst $(space),$(comma) ,$(foreach token,$(FUNCTIONS),"$(token)"))' \
	-DFUNCS='$(subst $(space),$(comma) ,$(foreach token,$(FUNCTIONS),&LibMatrix::$(token)))'

USER_CXX_FLAGS=-std=c++11 $(macros)

include $(TRILINOSPATH)/include/Makefile.export.Trilinos

# Make sure to use same compilers and flags as Trilinos
CXX=$(Trilinos_CXX_COMPILER)
CXX_FLAGS=$(Trilinos_CXX_COMPILER_FLAGS) $(USER_CXX_FLAGS)

INCLUDE_DIRS=$(Trilinos_INCLUDE_DIRS) $(Trilinos_TPL_INCLUDE_DIRS)
LIBRARY_DIRS=$(Trilinos_LIBRARY_DIRS) $(Trilinos_TPL_LIBRARY_DIRS)
LIBRARIES=$(Trilinos_LIBRARIES) $(Trilinos_TPL_LIBRARIES)

LINK_FLAGS=$(Trilinos_EXTRA_LD_FLAGS)

default: print_info libmatrix.mpi

# Echo trilinos build info just for fun
print_info:
	@echo "\nFound Trilinos!  Here are the details: "
	@echo "   Trilinos_VERSION = $(Trilinos_VERSION)"
	@echo "   Trilinos_PACKAGE_LIST = $(Trilinos_PACKAGE_LIST)"
	@echo "   Trilinos_LIBRARIES = $(Trilinos_LIBRARIES)"
	@echo "   Trilinos_INCLUDE_DIRS = $(Trilinos_INCLUDE_DIRS)"
	@echo "   Trilinos_LIBRARY_DIRS = $(Trilinos_LIBRARY_DIRS)"
	@echo "   Trilinos_TPL_LIST = $(Trilinos_TPL_LIST)"
	@echo "   Trilinos_TPL_INCLUDE_DIRS = $(Trilinos_TPL_INCLUDE_DIRS)"
	@echo "   Trilinos_TPL_LIBRARIES = $(Trilinos_TPL_LIBRARIES)"
	@echo "   Trilinos_TPL_LIBRARY_DIRS = $(Trilinos_TPL_LIBRARY_DIRS)"
	@echo "   Trilinos_BUILD_SHARED_LIBS = $(Trilinos_BUILD_SHARED_LIBS)"
	@echo "End of Trilinos details\n"

libmatrix.mpi: libmatrix.cpp
	$(CXX) $(CXX_FLAGS) $< -o $@ $(LINK_FLAGS) $(INCLUDE_DIRS) $(LIBRARY_DIRS) $(LIBRARIES)

clean:
	rm -f libmatrix.mpi *.pyc

test: libmatrix.mpi
	python test

.PHONY: default clean test
	
# vim:noexpandtab
