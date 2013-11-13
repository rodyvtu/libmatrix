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

#include <mpi.h>

  
#define ASSERT( cond ) if ( ! ( cond ) ) { printf( "assertion failed in %s (" __FILE__ ":%d), " #cond, __FUNCTION__, __LINE__ ); MPI::Finalize(); exit( EXIT_FAILURE ); }

Teuchos::oblackholestream blackHole;

typedef double scalar_t;
typedef int handle_t;
typedef int local_t;
typedef long global_t;

// matching mpi types
const MPI::Datatype MPI_SCALAR = MPI::DOUBLE;
const MPI::Datatype MPI_HANDLE = MPI::INT;
const MPI::Datatype MPI_SIZE = sizeof(size_t) == sizeof(int) ? MPI::INT : sizeof(size_t) == sizeof(long) ? MPI::LONG : 0;
const MPI::Datatype MPI_LOCAL = MPI::INT;
const MPI::Datatype MPI_GLOBAL = MPI::LONG;

typedef Kokkos::DefaultNode::DefaultNodeType node_t;
typedef Tpetra::Map<local_t, global_t, node_t> map_t;
typedef Tpetra::Vector<scalar_t, local_t, global_t, node_t> vector_t;
typedef Tpetra::Operator<scalar_t, local_t, global_t, node_t> operator_t;
typedef Tpetra::MultiVector<scalar_t, local_t, global_t, node_t> multivector_t;
typedef Tpetra::CrsMatrix<scalar_t, local_t, global_t, node_t> matrix_t;
typedef Tpetra::CrsGraph<local_t, global_t, node_t> graph_t;
typedef Belos::SolverManager<scalar_t, multivector_t, operator_t> solvermanager_t;
typedef Belos::LinearProblem<scalar_t, multivector_t, operator_t> linearproblem_t;
typedef Teuchos::ParameterList params_t;

const global_t indexBase = 0;

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


/* LIBMATRIX API */


/* RELEASE: release object
   
     -> broadcast HANDLE handle
*/
void release( MPI::Intercomm intercomm ) {

  handle_t handle;
  intercomm.Bcast( (void *)(&handle), 1, MPI_HANDLE, 0 );
  release_object( handle );
}


/* NEW_MAP: create new map
   
     -> broadcast HANDLE handle.map
     -> broadcast SIZE map_size
     -> scatter SIZE number_of_items[map_size]
     -> scatterv GLOBAL items[number_of_items]
*/
void new_map( MPI::Intercomm intercomm ) {

  struct { handle_t map; } handle;
  intercomm.Bcast( (void *)(&handle), 1, MPI_HANDLE, 0 );

  size_t size, ndofs;
  intercomm.Bcast( (void *)(&size), 1, MPI_SIZE, 0 );
  intercomm.Scatter( NULL, 1, MPI_SIZE, (void *)(&ndofs), 1, MPI_SIZE, 0 );

  out(intercomm) << "creating map #" << handle.map << " with " << ndofs << '/' << size << " items" << std::endl;

  Teuchos::Array<global_t> elementList( ndofs );
  intercomm.Scatterv( NULL, NULL, NULL, MPI_GLOBAL, (void *)elementList.getRawPtr(), ndofs, MPI_GLOBAL, 0 );

  Teuchos::RCP<node_t> node = Kokkos::DefaultNode::getDefaultNode ();
  Teuchos::RCP<const Teuchos::Comm<int> > comm = Tpetra::DefaultPlatform::getDefaultPlatform().getComm();
  Teuchos::RCP<map_t> map = Teuchos::rcp( new map_t( size, elementList, indexBase, comm, node ) );

  set_object( handle.map, map );
}


/* NEW_VECTOR: create new vector
   
     -> broadcast HANDLE handle.{vector,map}
*/
void new_vector( MPI::Intercomm intercomm ) {

  struct { handle_t vector, map; } handle;
  intercomm.Bcast( (void *)(&handle), 2, MPI_HANDLE, 0 );

  Teuchos::RCP<const map_t> map = get_object<const map_t>( handle.map );

  out(intercomm) << "creating vector #" << handle.vector << " from map #" << handle.map << std::endl;

  Teuchos::RCP<vector_t> vector = Teuchos::rcp( new vector_t( map ) );

  set_object( handle.vector, vector );
}


/* ADD_EVEC: add items to vector
   
     -> broadcast SIZE rank
   if rank == myrank
     -> recv HANDLE handle.vector
     -> recv SIZE number_of_items
     -> recv GLOBAL indices[number_of_items]
     -> recv SCALAR values[number_of_items]
   endif
*/
void add_evec( MPI::Intercomm intercomm ) {

  size_t rank;
  intercomm.Bcast( (void *)(&rank), 1, MPI_SIZE, 0 );

  if ( rank != intercomm.Get_rank() ) {
    return;
  }

  struct { handle_t vector; } handle;
  intercomm.Recv( (void *)(&handle), 1, MPI_HANDLE, 0, 0 );

  size_t nitems;
  intercomm.Recv( (void *)(&nitems), 1, MPI_SIZE, 0, 0 );

  out(intercomm) << "ivec = " << handle.vector << ", nitems = " << nitems << std::endl;

  Teuchos::ArrayRCP<global_t> idx( nitems );
  Teuchos::ArrayRCP<scalar_t> data( nitems );

  intercomm.Recv( (void *)idx.getRawPtr(), nitems, MPI_GLOBAL, 0, 0 );
  intercomm.Recv( (void *)data.getRawPtr(), nitems, MPI_SCALAR, 0, 0 );

  Teuchos::RCP<vector_t> vec = get_object<vector_t>( handle.vector );

  for ( int i = 0; i < nitems; i++ ) {
    out(intercomm) << idx[i] << " : " << data[i] << std::endl;
    vec->sumIntoGlobalValue( idx[i], data[i] );
  }

}


/* GET_VECTOR: collect vector over the intercom
  
     -> broadcast HANDLE handle.vector
    <-  gatherv SCALAR values[vector.size]
*/
void get_vector( MPI::Intercomm intercomm ) {

  struct { handle_t vector; } handle;
  intercomm.Bcast( (void *)(&handle), 1, MPI_HANDLE, 0 );

  Teuchos::RCP<vector_t> vec = get_object<vector_t>( handle.vector );

  Teuchos::ArrayRCP<const scalar_t> data = vec->getData();

  intercomm.Gatherv( (void *)(data.get()), data.size(), MPI_SCALAR, NULL, NULL, NULL, MPI_SCALAR, 0 );
}


/* VECTOR_DOT: compute frobenius norm
   
     -> broadcast HANDLE handle.{vector1,vector2}
    <-  gather SCALAR norm
*/
void vector_dot( MPI::Intercomm intercomm ) {

  struct { handle_t vector1, vector2; } handle;
  intercomm.Bcast( (void *)(&handle), 2, MPI_HANDLE, 0 );
  Teuchos::RCP<vector_t> vector1 = get_object<vector_t>( handle.vector1 );
  Teuchos::RCP<vector_t> vector2 = get_object<vector_t>( handle.vector2 );

  scalar_t dot = vector1->dot( *vector2 );

  intercomm.Gather( (void *)(&dot), 1, MPI_SCALAR, NULL, 1, MPI_SCALAR, 0 );
}


/* VECTOR_NORM: compute frobenius norm
   
     -> broadcast HANDLE handle.vector
    <-  gather SCALAR norm
*/
void vector_norm( MPI::Intercomm intercomm ) {

  struct { handle_t vector; } handle;
  intercomm.Bcast( (void *)(&handle), 1, MPI_HANDLE, 0 );
  Teuchos::RCP<vector_t> vector = get_object<vector_t>( handle.vector );

  scalar_t norm = vector->norm2();

  intercomm.Gather( (void *)(&norm), 1, MPI_SCALAR, NULL, 1, MPI_SCALAR, 0 );
}


/* NEW_GRAPH: create new graph
   
     -> broadcast HANDLE handle.{graph,rowmap,colmap,domainmap,rangemap}
     -> scatterv SIZE offsets[nrows+1]
     -> scatterv LOCAL columns[offsets[-1]]
*/
void new_graph( MPI::Intercomm intercomm ) {

  struct { handle_t graph, rowmap, colmap, domainmap, rangemap; } handle;
  intercomm.Bcast( (void *)(&handle), 5, MPI_HANDLE, 0 );

  Teuchos::RCP<const map_t> rowmap = get_object<const map_t>( handle.rowmap );
  Teuchos::RCP<const map_t> colmap = get_object<const map_t>( handle.colmap );
  Teuchos::RCP<const map_t> dommap = get_object<const map_t>( handle.domainmap );
  Teuchos::RCP<const map_t> rngmap = get_object<const map_t>( handle.rangemap );

  size_t nrows = rowmap->getNodeNumElements();

  out(intercomm) << "creating graph #" << handle.graph << " from rowmap #" << handle.rowmap << ", colmap #" << handle.colmap << ", domainmap #" << handle.domainmap << ", rangemap #" << handle.rangemap << " with " << nrows << " rows" << std::endl;

  Teuchos::ArrayRCP<size_t> offsets( nrows+1 );
  intercomm.Scatterv( NULL, NULL, NULL, MPI_SIZE, (void *)offsets.getRawPtr(), nrows+1, MPI_SIZE, 0 );

  int nindices = offsets[nrows];
  Teuchos::ArrayRCP<local_t> indices( nindices );
  intercomm.Scatterv( NULL, NULL, NULL, MPI_LOCAL, (void *)indices.getRawPtr(), nindices, MPI_LOCAL, 0 );

  Teuchos::RCP<graph_t> graph = Teuchos::rcp( new graph_t( rowmap, colmap, offsets, indices ) );
  graph->fillComplete( dommap, rngmap );

  set_object( handle.graph, graph );
}


/* NEW_MATRIX: create new matrix
   
     -> broadcast HANDLE handle.{matrix,graph}
*/
void new_matrix( MPI::Intercomm intercomm ) {

  struct { handle_t matrix, graph; } handle;
  intercomm.Bcast( (void *)(&handle), 2, MPI_HANDLE, 0 );

  Teuchos::RCP<const graph_t> graph = get_object<const graph_t>( handle.graph );

  out(intercomm) << "creating matrix #" << handle.matrix << " from graph #" << handle.graph << std::endl;

  Teuchos::RCP<matrix_t> matrix = Teuchos::rcp( new matrix_t( graph ) );

  set_object( handle.matrix, matrix );
}


/* ADD_EMAT: add items to matrix
   
     -> broadcast SIZE rank
   if rank == myrank
     -> recv HANDLE handle.vector
     -> recv SIZE number_of_{rows,cols}
     -> recv GLOBAL indices[number_of_rows]
     -> recv GLOBAL indices[number_of_cols]
     -> recv SCALAR values[number_of_(rows*cols)]
   endif
*/
void add_emat( MPI::Intercomm intercomm ) {

  size_t rank;
  intercomm.Bcast( (void *)(&rank), 1, MPI_SIZE, 0 );

  if ( rank != intercomm.Get_rank() ) {
    return;
  }

  struct { handle_t matrix; } handle;
  intercomm.Recv( (void *)(&handle), 1, MPI_HANDLE, 0, 0 );

  size_t nitems[2];
  intercomm.Recv( (void *)nitems, 2, MPI_SIZE, 0, 0 );

  out(intercomm) << "imat = " << handle.matrix << ", nitems = " << nitems[0] << "," << nitems[1] << std::endl;

  Teuchos::ArrayRCP<global_t> rowidx( nitems[0] );
  Teuchos::ArrayRCP<global_t> colidx( nitems[1] );
  Teuchos::ArrayRCP<scalar_t> data( nitems[0]*nitems[1] );

  intercomm.Recv( (void *)rowidx.getRawPtr(), nitems[0], MPI_GLOBAL, 0, 0 );
  intercomm.Recv( (void *)colidx.getRawPtr(), nitems[1], MPI_GLOBAL, 0, 0 );
  intercomm.Recv( (void *)data.getRawPtr(), nitems[0]*nitems[1], MPI_SCALAR, 0, 0 );

  Teuchos::RCP<matrix_t> mat = get_object<matrix_t>( handle.matrix );

  const Teuchos::ArrayView<const global_t> colidx_view = colidx.view( 0, nitems[1] );
  for ( int i = 0; i < nitems[0]; i++ ) {
    mat->sumIntoGlobalValues( rowidx[i], colidx_view, data.view(i*nitems[0],nitems[1]) );
  }

}


/* FILL_COMPLETE: set matrix to fill-complete
   
     -> broadcast HANDLE handle.matrix
*/
void fill_complete( MPI::Intercomm intercomm ) {

  struct { handle_t matrix; } handle;
  intercomm.Bcast( (void *)(&handle), 1, MPI_HANDLE, 0 );
  Teuchos::RCP<matrix_t> matrix = get_object<matrix_t>( handle.matrix );

  out(intercomm) << "completing matrix #" << handle.matrix << std::endl;

  matrix->fillComplete( matrix->getDomainMap(), matrix->getRangeMap() );
}


/* MATRIX_NORM: compute frobenius norm
   
     -> broadcast HANDLE handle.matrix
    <-  gather SCALAR norm
*/
void matrix_norm( MPI::Intercomm intercomm ) {

  struct { handle_t matrix; } handle;
  intercomm.Bcast( (void *)(&handle), 1, MPI_HANDLE, 0 );
  Teuchos::RCP<matrix_t> matrix = get_object<matrix_t>( handle.matrix );

  scalar_t norm = matrix->getFrobeniusNorm();

  intercomm.Gather( (void *)(&norm), 1, MPI_SCALAR, NULL, 1, MPI_SCALAR, 0 );
}


/* MATVEC: matrix vector multiplication
   
     -> broadcast HANDLE handle.{matrix,out,vector}
*/
void matvec( MPI::Intercomm intercomm ) {

  struct { handle_t matrix, rhs, lhs; } handle;
  intercomm.Bcast( (void *)(&handle), 3, MPI_HANDLE, 0 );

  Teuchos::RCP<matrix_t> matrix = get_object<matrix_t>( handle.matrix );
  Teuchos::RCP<vector_t> rhs = get_object<vector_t>( handle.rhs );
  Teuchos::RCP<vector_t> lhs = get_object<vector_t>( handle.lhs );

  matrix->apply( *rhs, *lhs );
}


/* SOLVE: solve linear system
   
     -> broadcast HANDLE handle.{matrix,rhs,lhs}
*/
void solve( MPI::Intercomm intercomm ) {

  struct { handle_t matrix, rhs, lhs; } handle;
  intercomm.Bcast( (void *)(&handle), 3, MPI_HANDLE, 0 );

  Teuchos::RCP<matrix_t> matrix = get_object<matrix_t>( handle.matrix );
  Teuchos::RCP<vector_t> rhs = get_object<vector_t>( handle.rhs );
  Teuchos::RCP<vector_t> lhs = get_object<vector_t>( handle.lhs );

  Belos::SolverFactory<scalar_t, multivector_t, operator_t> factory;

  Teuchos::RCP<params_t> solverParams = Teuchos::rcp( new params_t );
  solverParams->set( "Num Blocks", 40 );
  solverParams->set( "Maximum Iterations", 400 );
  solverParams->set( "Convergence Tolerance", 1.0e-8 );

  Teuchos::RCP<solvermanager_t> solver = factory.create( "GMRES", solverParams );
  Teuchos::RCP<linearproblem_t> problem = Teuchos::rcp( new linearproblem_t( matrix, lhs, rhs ) );

  //problem->setRightPrec (M);

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


/* MPI SETUP */


typedef void ( *funcptr )( MPI::Intercomm );
#define TOKENS release, new_vector, add_evec, get_vector, new_map, new_graph, new_matrix, add_emat, fill_complete, matrix_norm, matvec, vector_norm, vector_dot, solve
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

  unsigned char c;
  for ( ;; ) {
    out(intercomm) << "waiting\n";
    intercomm.Bcast( (void *)(&c), 1, MPI::CHAR, 0 );
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
    std::cout << "token: enum" << STR(TOKENS) << std::endl;
    std::cout << "local: int" << (sizeof(local_t) << 3) << std::endl;
    std::cout << "global: int" << (sizeof(global_t) << 3) << std::endl;
    std::cout << "size: int" << (sizeof(size_t) << 3) << std::endl;
    std::cout << "handle: int" << (sizeof(handle_t) << 3) << std::endl;
    std::cout << "scalar: float" << (sizeof(scalar_t) << 3) << std::endl;
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
