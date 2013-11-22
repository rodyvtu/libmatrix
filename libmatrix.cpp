#include <Tpetra_DefaultPlatform.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_Version.hpp>
#include <Tpetra_CrsMatrix.hpp>

#include <BelosTpetraAdapter.hpp>
#include <BelosSolverFactory.hpp>

#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_DefaultMpiComm.hpp>
#include <Teuchos_oblackholestream.hpp>

#include <Ifpack2_Factory.hpp>

#include <mpi.h>

  
#define ASSERT( cond ) if ( ! ( cond ) ) { printf( "assertion failed in %s (" __FILE__ ":%d), " #cond, __FUNCTION__, __LINE__ ); MPI::Finalize(); exit( EXIT_FAILURE ); }

Teuchos::oblackholestream blackHole;

auto supportedPreconNames = Teuchos::tuple(
  std::string( "ILUT" ),
  std::string( "RILUK" ),
  std::string( "RELAXATION" ),
  std::string( "CHEBYSHEV" ),
  std::string( "DIAGONAL" ),
  std::string( "SCHWARZ" ),
  std::string( "KRYLOV" )
);

class DescribableParams : public Teuchos::Describable, public Teuchos::ParameterList {};

typedef double scalar_t;
typedef int number_t;
typedef int handle_t;
typedef int local_t;
typedef long global_t;
typedef uint8_t token_t;
typedef bool bool_t;
typedef Kokkos::DefaultNode::DefaultNodeType node_t;
typedef Tpetra::Map<local_t, global_t, node_t> map_t;
typedef Tpetra::Vector<scalar_t, local_t, global_t, node_t> vector_t;
typedef Tpetra::Operator<scalar_t, local_t, global_t, node_t> operator_t;
typedef Tpetra::MultiVector<scalar_t, local_t, global_t, node_t> multivector_t;
typedef Tpetra::CrsMatrix<scalar_t, local_t, global_t, node_t> matrix_t;
typedef Tpetra::CrsGraph<local_t, global_t, node_t> graph_t;
typedef Belos::SolverManager<scalar_t, multivector_t, operator_t> solvermanager_t;
typedef Belos::LinearProblem<scalar_t, multivector_t, operator_t> linearproblem_t;
typedef Belos::SolverFactory<scalar_t, multivector_t, operator_t> solverfactory_t;
typedef DescribableParams params_t;
typedef Ifpack2::Preconditioner<scalar_t, local_t, global_t, node_t> precon_t;
typedef Ifpack2::Factory preconfactory_t;
typedef Teuchos::ScalarTraits<scalar_t>::magnitudeType magnitude_t;
typedef Tpetra::Export<local_t, global_t, node_t> export_t;

const global_t indexbase = 0;

/* MISC HELPER FUNCTIONS */


/*
bool verify( bool good, const char *msg, MPI::Intercomm intercomm ) {
  int status = good ? 0 : strlen( msg );
  intercomm.Gather( (void *)(&status), 1, MPI::INT, NULL, 1, MPI::INT, 0 );
  if ( ! good ) {
    intercomm.Send( msg, status, MPI::CHAR, 0, 10 );
  }
  return good;
}
*/


inline std::ostream& out( MPI::Intercomm intercomm ) {
  return
  #ifdef DEBUG
    std::cout << '[' << intercomm.Get_rank() << '/' << intercomm.Get_size() << "] ";
  #else
    blackHole;
  #endif
}


/* OBJECT ARRAY */


Teuchos::Array<Teuchos::RCP<Teuchos::Describable> > OBJECTS;

inline void set_object( handle_t handle, Teuchos::RCP<Teuchos::Describable> object ) {
  if ( handle == OBJECTS.size() ) {
    OBJECTS.push_back( object );
  }
  else {
    ASSERT( OBJECTS[handle].is_null() );
    OBJECTS[handle] = object;
  }
  #ifdef DEBUG
    std::cout << "SET #" << handle << " " << Teuchos::describe( *object, Teuchos::VERB_LOW ) << std::endl;
  #endif
}

template <class T>
inline Teuchos::RCP<T> get_object( handle_t handle ) {
  return Teuchos::rcp_dynamic_cast<T>( OBJECTS[handle], true );
}

inline void release_object( handle_t handle ) {
  OBJECTS[handle] = Teuchos::RCP<Teuchos::Describable>();
  #ifdef DEBUG
    std::cout << "DEL #" << handle << std::endl;
  #endif
}


/* TEMPLATED MPI */


template <class T>
inline void Bcast( MPI::Intercomm intercomm, T *data, const int n=1 ) {
  intercomm.Bcast( (void *)data, n * sizeof(T), MPI::BYTE, 0 );
}

template <class T>
inline void Scatter( MPI::Intercomm intercomm, T *data, const int n=1 ) {
  intercomm.Scatter( NULL, 0, MPI::BYTE, (void *)data, n * sizeof(T), MPI::BYTE, 0 );
}

template <class T>
inline void Scatterv( MPI::Intercomm intercomm, T *data, const int n=1 ) {
  intercomm.Scatterv( NULL, NULL, NULL, MPI::BYTE, (void *)data, n * sizeof(T), MPI::BYTE, 0 );
}

template <class T>
inline void Recv( MPI::Intercomm intercomm, T *data, const int n=1, const int tag=0 ) {
  intercomm.Recv( (void *)data, n * sizeof(T), MPI::BYTE, tag, 0 );
}

template <class T>
inline void Gatherv( MPI::Intercomm intercomm, T *data, const int n=1 ) {
  intercomm.Gatherv( (void *)data, n * sizeof(T), MPI::BYTE, NULL, NULL, NULL, MPI::BYTE, 0 );
}

template <class T>
inline void Gather( MPI::Intercomm intercomm, T *data, const int n=1 ) {
  intercomm.Gather( (void *)data, n * sizeof(T), MPI::BYTE, NULL, 0, MPI::BYTE, 0 );
}


/* LIBMATRIX API */


/* PARAMS_NEW: create new parameter list
   
    -> broadcast 1 HANDLE params_handle
*/
void params_new( MPI::Intercomm intercomm ) {

  struct { handle_t params; } handle;
  Bcast( intercomm, &handle );

  Teuchos::RCP<params_t> params = Teuchos::rcp( new params_t );

  out(intercomm) << "creating parameter list #" << handle.params << std::endl;

  set_object( handle.params, params );

}


/* PARAMS_SET: set new integer in parameter list
   
    -> broadcast HANDLE params_handle 
    -> broadcast SIZE length_of_key
    -> broadcast CHAR key[length_of_key]
    -> broadcast TEMLATE_ARG value
*/
template <class T>
void params_set ( MPI::Intercomm intercomm ) {

  struct { handle_t params; } handle;
  Bcast( intercomm, &handle );

  size_t nchar;
  Bcast( intercomm, &nchar );

  std::string key( nchar, 0 );
  Bcast( intercomm, const_cast<char*>(key.data()), nchar );

  T value;
  Bcast( intercomm, &value );

  Teuchos::RCP<params_t> params = get_object<params_t>( handle.params );
  params->set( key, value );

  out(intercomm) << "added key=\"" << key << "\" with value=" << value << std::endl;
}


/* PARAMS_PRINT: print the params list (c-sided)
   
    -> broadcast 1 HANDLE params_handle 
*/
void params_print ( MPI::Intercomm intercomm ) {

  struct { handle_t params; } handle;
  Bcast( intercomm, &handle );

  Teuchos::RCP<params_t> params = get_object<params_t>( handle.params );
  params->print( out(intercomm) );
}


/* RELEASE: release object
   
     -> broadcast HANDLE handle
*/
void release( MPI::Intercomm intercomm ) {

  handle_t handle;
  Bcast( intercomm, &handle );
  release_object( handle );
}


/* MAP_NEW: create new map
   
     -> broadcast HANDLE handle.map
     -> broadcast SIZE map_size
     -> scatter SIZE number_of_items[map_size]
     -> scatterv GLOBAL items[number_of_items]
*/
void map_new( MPI::Intercomm intercomm ) {

  struct { handle_t map; } handle;
  Bcast( intercomm, &handle );

  size_t size, ndofs;
  Bcast( intercomm, &size );
  Scatter( intercomm, &ndofs );

  out(intercomm) << "creating map #" << handle.map << " with " << ndofs << '/' << size << " items" << std::endl;

  Teuchos::Array<global_t> elementList( ndofs );
  Scatterv( intercomm, elementList.getRawPtr(), ndofs );

  Teuchos::RCP<node_t> node = Kokkos::DefaultNode::getDefaultNode ();
  Teuchos::RCP<const Teuchos::Comm<int> > comm = Tpetra::DefaultPlatform::getDefaultPlatform().getComm();
  Teuchos::RCP<map_t> map = Teuchos::rcp( new map_t( size, elementList, indexbase, comm, node ) );

  set_object( handle.map, map );
}


/* VECTOR_NEW: create new vector
   
     -> broadcast HANDLE handle.{vector,map}
*/
void vector_new( MPI::Intercomm intercomm ) {

  struct { handle_t vector, map; } handle;
  Bcast( intercomm, &handle );

  Teuchos::RCP<const map_t> map = get_object<const map_t>( handle.map );

  out(intercomm) << "creating vector #" << handle.vector << " from map #" << handle.map << std::endl;

  Teuchos::RCP<vector_t> vector = Teuchos::rcp( new vector_t( map ) );

  set_object( handle.vector, vector );
}


/* VECTOR_ADD_BLOCK: add items to vector
   
     -> broadcast SIZE rank
   if rank == myrank
     -> recv HANDLE handle.vector
     -> recv SIZE number_of_items
     -> recv GLOBAL indices[number_of_items]
     -> recv SCALAR values[number_of_items]
   endif
*/
void vector_add_block( MPI::Intercomm intercomm ) {

  size_t rank;
  Bcast( intercomm, &rank );

  if ( rank != intercomm.Get_rank() ) {
    return;
  }

  struct { handle_t vector; } handle;
  Recv( intercomm, &handle );

  size_t nitems;
  Recv( intercomm, &nitems );

  out(intercomm) << "ivec = " << handle.vector << ", nitems = " << nitems << std::endl;

  Teuchos::ArrayRCP<local_t> idx( nitems );
  Teuchos::ArrayRCP<scalar_t> data( nitems );

  Recv( intercomm, idx.getRawPtr(), nitems );
  Recv( intercomm, data.getRawPtr(), nitems );

  Teuchos::RCP<vector_t> vec = get_object<vector_t>( handle.vector );

  for ( int i = 0; i < nitems; i++ ) {
    out(intercomm) << idx[i] << " : " << data[i] << std::endl;
    vec->sumIntoLocalValue( idx[i], data[i] );
  }

}


/* VECTOR_GETDATA: collect vector over the intercom
  
     -> broadcast HANDLE handle.vector
    <-  gatherv SCALAR values[vector.size]
*/
void vector_getdata( MPI::Intercomm intercomm ) {

  struct { handle_t vector; } handle;
  Bcast( intercomm, &handle );

  Teuchos::RCP<vector_t> vec = get_object<vector_t>( handle.vector );

  Teuchos::ArrayRCP<const scalar_t> data = vec->getData();

  Gatherv( intercomm, data.get(), data.size() );
}


/* VECTOR_DOT: compute frobenius norm
   
     -> broadcast HANDLE handle.{vector1,vector2}
    <-  gather SCALAR norm
*/
void vector_dot( MPI::Intercomm intercomm ) {

  struct { handle_t vector1, vector2; } handle;
  Bcast( intercomm, &handle );
  Teuchos::RCP<vector_t> vector1 = get_object<vector_t>( handle.vector1 );
  Teuchos::RCP<vector_t> vector2 = get_object<vector_t>( handle.vector2 );

  scalar_t dot = vector1->dot( *vector2 );

  Gather( intercomm, &dot );
}


/* VECTOR_NORM: compute frobenius norm
   
     -> broadcast HANDLE handle.vector
    <-  gather SCALAR norm
*/
void vector_norm( MPI::Intercomm intercomm ) {

  struct { handle_t vector; } handle;
  Bcast( intercomm, &handle );
  Teuchos::RCP<vector_t> vector = get_object<vector_t>( handle.vector );

  scalar_t norm = vector->norm2();

  Gather( intercomm, &norm );
}


/* GRAPH_NEW: create new graph
   
     -> broadcast HANDLE handle.{graph,rowmap,colmap,domainmap,rangemap}
     -> scatterv SIZE offsets[nrows+1]
     -> scatterv LOCAL columns[offsets[-1]]
*/
void graph_new( MPI::Intercomm intercomm ) {

  struct { handle_t graph, rowmap, colmap; } handle;
  Bcast( intercomm, &handle );

  Teuchos::RCP<const map_t> rowmap = get_object<const map_t>( handle.rowmap );
  Teuchos::RCP<const map_t> colmap = get_object<const map_t>( handle.colmap );

  size_t nrows = rowmap->getNodeNumElements();

  out(intercomm) << "creating graph #" << handle.graph << " from rowmap #" << handle.rowmap << ", colmap #" << handle.colmap << " with " << nrows << " rows" << std::endl;

  Teuchos::ArrayRCP<size_t> offsets( nrows+1 );
  Scatterv( intercomm, offsets.getRawPtr(), nrows+1 );

  int nindices = offsets[nrows];
  Teuchos::ArrayRCP<local_t> indices( nindices );
  Scatterv( intercomm, indices.getRawPtr(), nindices );

  Teuchos::RCP<graph_t> graph = Teuchos::rcp( new graph_t( rowmap, colmap, offsets, indices ) );
  graph->fillComplete();

  set_object( handle.graph, graph );
}


/* MATRIX_NEW: create new matrix
   
     -> broadcast HANDLE handle.{matrix,graph}
*/
void matrix_new( MPI::Intercomm intercomm ) {

  struct { handle_t matrix, graph; } handle;
  Bcast( intercomm, &handle );

  Teuchos::RCP<const graph_t> graph = get_object<const graph_t>( handle.graph );

  out(intercomm) << "creating matrix #" << handle.matrix << " from graph #" << handle.graph << std::endl;

  Teuchos::RCP<matrix_t> matrix = Teuchos::rcp( new matrix_t( graph ) );

  set_object( handle.matrix, matrix );
}


/* MATRIX_ADD_BLOCK: add items to matrix
   
     -> broadcast SIZE rank
   if rank == myrank
     -> recv HANDLE handle.vector
     -> recv SIZE number_of_{rows,cols}
     -> recv GLOBAL indices[number_of_rows]
     -> recv GLOBAL indices[number_of_cols]
     -> recv SCALAR values[number_of_(rows*cols)]
   endif
*/
void matrix_add_block( MPI::Intercomm intercomm ) {

  size_t rank;
  Bcast( intercomm, &rank );

  if ( rank != intercomm.Get_rank() ) {
    return;
  }

  struct { handle_t matrix; } handle;
  Recv( intercomm, &handle );

  size_t nitems[2];
  Recv( intercomm, nitems, 2 );

  out(intercomm) << "imat = " << handle.matrix << ", nitems = " << nitems[0] << "," << nitems[1] << std::endl;

  Teuchos::ArrayRCP<local_t> rowidx( nitems[0] );
  Teuchos::ArrayRCP<local_t> colidx( nitems[1] );
  Teuchos::ArrayRCP<scalar_t> data( nitems[0]*nitems[1] );

  Recv( intercomm, rowidx.getRawPtr(), nitems[0] );
  Recv( intercomm, colidx.getRawPtr(), nitems[1] );
  Recv( intercomm, data.getRawPtr(), nitems[0]*nitems[1] );

  Teuchos::RCP<matrix_t> mat = get_object<matrix_t>( handle.matrix );

  const Teuchos::ArrayView<const local_t> colidx_view = colidx.view( 0, nitems[1] );
  for ( int i = 0; i < nitems[0]; i++ ) {
    mat->sumIntoLocalValues( rowidx[i], colidx_view, data.view(i*nitems[0],nitems[1]) );
  }

}


/* MATRIX_COMPLETE: set matrix to fill-complete
   
     -> broadcast HANDLE handle.{matrix,exporter}
*/
void matrix_complete( MPI::Intercomm intercomm ) {

  struct { handle_t matrix, exporter; } handle;
  Bcast( intercomm, &handle );

  Teuchos::RCP<const matrix_t> matrix = get_object<const matrix_t>( handle.matrix );
  Teuchos::RCP<const export_t> exporter = get_object<const export_t>( handle.exporter );
  Teuchos::RCP<const map_t> domainmap = exporter->getTargetMap();
  Teuchos::RCP<const map_t> rangemap = exporter->getTargetMap();

  out(intercomm) << "completing matrix #" << handle.matrix << std::endl;

  Teuchos::RCP<matrix_t> completed_matrix = Tpetra::exportAndFillCompleteCrsMatrix( matrix, *exporter, domainmap, rangemap );

  release_object( handle.matrix );
  set_object( handle.matrix, completed_matrix );
}


/* MATRIX_NORM: compute frobenius norm
   
     -> broadcast HANDLE handle.matrix
    <-  gather SCALAR norm
*/
void matrix_norm( MPI::Intercomm intercomm ) {

  struct { handle_t matrix; } handle;
  Bcast( intercomm, &handle );
  Teuchos::RCP<matrix_t> matrix = get_object<matrix_t>( handle.matrix );

  scalar_t norm = matrix->getFrobeniusNorm();

  Gather( intercomm, &norm );
}


/* MATRIX_APPLY: matrix vector multiplication
   
     -> broadcast HANDLE handle.{matrix,out,vector}
*/
void matrix_apply( MPI::Intercomm intercomm ) {

  struct { handle_t matrix, rhs, lhs; } handle;
  Bcast( intercomm, &handle );

  Teuchos::RCP<matrix_t> matrix = get_object<matrix_t>( handle.matrix );
  Teuchos::RCP<vector_t> rhs = get_object<vector_t>( handle.rhs );
  Teuchos::RCP<vector_t> lhs = get_object<vector_t>( handle.lhs );

  matrix->apply( *rhs, *lhs );
}


/* MATRIX_SOLVE: solve linear system
   
     -> broadcast HANDLE handle.{matrix,precon,rhs,lhs,solvertype,solverparams}
     -> broadcast BOOL symmetric
*/
void matrix_solve( MPI::Intercomm intercomm ) {

  struct { handle_t matrix, precon, rhs, lhs, solvertype, solverparams; } handle;
  Bcast( intercomm, &handle );

  bool_t symmetric;
  Bcast( intercomm, &symmetric );

  Teuchos::RCP<matrix_t> matrix = get_object<matrix_t>( handle.matrix );
  Teuchos::RCP<operator_t> precon = get_object<operator_t>( handle.precon );
  Teuchos::RCP<vector_t> rhs = get_object<vector_t>( handle.rhs );
  Teuchos::RCP<vector_t> lhs = get_object<vector_t>( handle.lhs );
  Teuchos::RCP<params_t> solverparams = get_object<params_t>( handle.solverparams );

  solverfactory_t factory;
  Teuchos::RCP<solvermanager_t> solver = factory.create( factory.supportedSolverNames()[handle.solvertype], solverparams );
  Teuchos::RCP<linearproblem_t> problem = Teuchos::rcp( new linearproblem_t( matrix, lhs, rhs ) );

  if ( symmetric ) {
    problem->setHermitian();
  }
  problem->setRightPrec( precon );

  // from the docs: Many of Belos' solvers require that this method has been
  // called on the linear problem, before they can solve it.
  problem->setProblem();

  // Tell the solver what problem you want to solve.
  solver->setProblem( problem );

  // Attempt to solve the linear system.  result == Belos::Converged
  // means that it was solved to the desired tolerance.  This call
  // overwrites X with the computed approximate solution.
  Belos::ReturnType result = solver->solve();

  // Ask the solver how many iterations the last solve() took.
  const int numIters = solver->getNumIters();

  out(intercomm) << "solver finished in " << numIters << " iterations with result " << result << std::endl;
}


/* PRECON_NEW: create new preconditioner
   
     -> broadcast HANDLE handle.{precon,matrix,precontype,preconparams}
*/
void precon_new( MPI::Intercomm intercomm ) {

  struct { handle_t precon, matrix, precontype, preconparams; } handle;
  Bcast( intercomm, &handle );

  Teuchos::RCP<const matrix_t> matrix = get_object<const matrix_t>( handle.matrix );
  Teuchos::RCP<const params_t> preconparams = get_object<const params_t>( handle.preconparams );
 
  preconfactory_t factory;
  Teuchos::RCP<precon_t> precon = factory.create( supportedPreconNames[handle.precontype], matrix );

  precon->setParameters( *preconparams );
  precon->initialize();
  precon->compute();

  magnitude_t condest = precon->computeCondEst( Ifpack2::Cheap );
  out(intercomm) << "Ifpack2 preconditioner's estimated condition number: " << condest << std::endl;

  set_object( handle.precon, precon );
}


/* EXPORT_NEW: create new exporter
   
     -> broadcast HANDLE handle.{exporter,srcmap,dstmap}
*/
void export_new( MPI::Intercomm intercomm ) {

  struct { handle_t exporter, srcmap, dstmap; } handle;
  Bcast( intercomm, &handle );

  Teuchos::RCP<const map_t> srcmap = get_object<const map_t>( handle.srcmap );
  Teuchos::RCP<const map_t> dstmap = get_object<const map_t>( handle.dstmap );

  Teuchos::RCP<export_t> exporter = Teuchos::rcp( new export_t( srcmap, dstmap ) );

  set_object( handle.exporter, exporter );
}


/* MPI SETUP */


typedef void ( *funcptr )( MPI::Intercomm );
#define TOKENS release, \
  vector_new, vector_add_block, vector_getdata, vector_norm, vector_dot, \
  matrix_new, matrix_add_block, matrix_complete, matrix_norm, matrix_apply, matrix_solve, \
  params_new, params_set<number_t>, params_set<scalar_t>, params_print, \
  map_new, graph_new, precon_new, export_new

funcptr FTABLE[] = { TOKENS };
#define NTOKENS ( sizeof(FTABLE) / sizeof(funcptr) )
#define STR(...) XSTR((__VA_ARGS__))
#define XSTR(s) #s


void eventloop( char *progname ) {

  int argc = 1;
  char **argv = &progname;

  MPI::Init( argc, argv ); 
  MPI::COMM_WORLD.Set_errhandler( MPI::ERRORS_THROW_EXCEPTIONS );
  MPI::Intercomm intercomm = MPI::Comm::Get_parent();

  token_t c;
  for ( ;; ) {
    out(intercomm) << "waiting\n";
    Bcast( intercomm, &c );
    out(intercomm) << "received " << (int)c << '\n';
    if ( c >= NTOKENS ) {
      out(intercomm) << "quit\n";
      break;
    }
    FTABLE[c]( intercomm );
  }

  intercomm.Disconnect();
  MPI::Finalize(); 
}


int main( int argc, char *argv[] ) {

  if ( argc == 2 && std::strcmp( argv[1], "info" ) == 0 ) {

    std::string tokens( STR(TOKENS) ); // wrapped in parentheses
    std::cout << "tokens: " << tokens.substr(1,tokens.length()-2) << std::endl;

    std::cout << "solvers";
    {char sep[] = ": ";
    for ( auto name : solverfactory_t().supportedSolverNames() ) {
      std::cout << sep << name;
      sep[0] = ',';
    }}
    std::cout << std::endl;

    std::cout << "precons";
    {char sep[] = ": ";
    for ( auto name : supportedPreconNames ) {
      std::cout << sep << name;
      sep[0] = ',';
    }}
    std::cout << std::endl;

    std::cout << "token_t: uint" << (sizeof(token_t) << 3) << std::endl;
    std::cout << "local_t: int" << (sizeof(local_t) << 3) << std::endl;
    std::cout << "global_t: int" << (sizeof(global_t) << 3) << std::endl;
    std::cout << "size_t: uint" << (sizeof(size_t) << 3) << std::endl;
    std::cout << "handle_t: int" << (sizeof(handle_t) << 3) << std::endl;
    std::cout << "number_t: int" << (sizeof(number_t) << 3) << std::endl;
    std::cout << "bool_t: uint" << (sizeof(bool_t) << 3) << std::endl;
    std::cout << "scalar_t: float" << (sizeof(scalar_t) << 3) << std::endl;
  }
  else if ( argc == 2 && std::strcmp( argv[1], "eventloop" ) == 0 ) {
    eventloop( argv[0] );
  }
  else {
    std::cout << "syntax: " << argv[0] << " info|eventloop" << std::endl;
    return 1;
  }
  return 0;
}


// vim:foldmethod=syntax
