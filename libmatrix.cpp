#include <Tpetra_DefaultPlatform.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_Version.hpp>
#include <Tpetra_CrsMatrix.hpp>

#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_DefaultMpiComm.hpp>
#include <Teuchos_oblackholestream.hpp>

#include <mpi.h>


Teuchos::oblackholestream blackHole;

typedef double scalar_t;
typedef int handle_t;
typedef int local_t;
typedef long global_t;

// matching mpi types
const MPI::Datatype MPI_SCALAR = MPI::DOUBLE;
const MPI::Datatype MPI_HANDLE = MPI::INT;
const MPI::Datatype MPI_SIZE = MPI::INT;
const MPI::Datatype MPI_LOCAL = MPI::INT;
const MPI::Datatype MPI_GLOBAL = MPI::LONG;

typedef Kokkos::DefaultNode::DefaultNodeType node_t;
typedef Tpetra::Map<local_t, global_t, node_t> map_t;
typedef Tpetra::Vector<scalar_t, local_t, global_t, node_t> vector_t;
typedef Tpetra::CrsMatrix<scalar_t, local_t, global_t> matrix_t;
typedef Tpetra::CrsGraph<local_t, global_t, node_t> graph_t;

const global_t indexBase = 0;

Teuchos::Array<Teuchos::RCP<Teuchos::Describable> > OBJECTS;


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


/* LIBMATRIX API */


/* NEW_MAP: create new map
   
     -> broadcast 1 SIZE map_size
     -> scatter map_size SIZE number_of_items
     -> scatterv number_of_items GLOBAL items
    <-  gather 1 HANDLE map_handle
*/
void new_map( MPI::Intercomm intercomm ) {

  handle_t imap = OBJECTS.size();

  size_t size, ndofs;
  intercomm.Bcast( (void *)(&size), 1, MPI_SIZE, 0 );
  intercomm.Scatter( NULL, 1, MPI_SIZE, (void *)(&ndofs), 1, MPI_SIZE, 0 );

  out(intercomm) << "creating map #" << imap << " with " << ndofs << '/' << size << " items" << std::endl;

  Teuchos::Array<global_t> elementList( ndofs );
  intercomm.Scatterv( NULL, NULL, NULL, MPI_GLOBAL, (void *)elementList.getRawPtr(), ndofs, MPI_GLOBAL, 0 );

  Teuchos::RCP<node_t> node = Kokkos::DefaultNode::getDefaultNode ();
  Teuchos::RCP<const Teuchos::Comm<int> > comm = Tpetra::DefaultPlatform::getDefaultPlatform().getComm();

  OBJECTS.push_back( Teuchos::rcp( new map_t( size, elementList, indexBase, comm, node ) ) );

  intercomm.Gather( (void *)(&imap), 1, MPI_HANDLE, NULL, 1, MPI_HANDLE, 0 );
}


/* NEW_VECTOR: create new vector
   
     -> broadcast 1 HANDLE map_handle
    <-  gather 1 HANDLE vector_handle
*/
void new_vector( MPI::Intercomm intercomm ) {

  handle_t ivec = OBJECTS.size();

  handle_t imap;
  intercomm.Bcast( (void *)(&imap), 1, MPI_HANDLE, 0 );
  Teuchos::RCP<const map_t> map = Teuchos::rcp_dynamic_cast<const map_t>( OBJECTS[imap], true );

  out(intercomm) << "creating vector #" << ivec << " from map #" << imap << std::endl;

  OBJECTS.push_back( Teuchos::rcp( new vector_t( map ) ) );

  intercomm.Gather( (void *)(&ivec), 1, MPI_HANDLE, NULL, 1, MPI_HANDLE, 0 );
}


/* ADD_EVEC: add items to vector
   
     -> broadcast 1 SIZE rank
   if rank == myrank
     -> recv 1 HANDLE vector_handle
     -> recv 1 SIZE number_of_items
     -> recv number_of_items GLOBAL indices
     -> recv number_of_items SCALAR values
   endif
*/
void add_evec( MPI::Intercomm intercomm ) {

  size_t rank;
  intercomm.Bcast( (void *)(&rank), 1, MPI_SIZE, 0 );

  if ( rank != intercomm.Get_rank() ) {
    return;
  }

  handle_t ivec;
  intercomm.Recv( (void *)(&ivec), 1, MPI_HANDLE, 0, 0 );

  size_t nitems;
  intercomm.Recv( (void *)(&nitems), 1, MPI_SIZE, 0, 0 );

  out(intercomm) << "ivec = " << ivec << ", nitems = " << nitems << std::endl;

  Teuchos::ArrayRCP<global_t> idx( nitems );
  Teuchos::ArrayRCP<scalar_t> data( nitems );

  intercomm.Recv( (void *)idx.getRawPtr(), nitems, MPI_GLOBAL, 0, 0 );
  intercomm.Recv( (void *)data.getRawPtr(), nitems, MPI_SCALAR, 0, 0 );

  Teuchos::RCP<vector_t> vec = Teuchos::rcp_dynamic_cast<vector_t>( OBJECTS[ivec], true );

  for ( int i = 0; i < nitems; i++ ) {
    out(intercomm) << idx[i] << " : " << data[i] << std::endl;
    vec->sumIntoGlobalValue( idx[i], data[i] );
  }

}


/* GET_VECTOR: collect vector over the intercom
  
     -> broadcast 1 HANDLE vector_handle
    <-  gatherv vector.size SCALAR values
*/
void get_vector( MPI::Intercomm intercomm ) {

  handle_t ivec;
  intercomm.Bcast( (void *)(&ivec), 1, MPI_HANDLE, 0 );

  Teuchos::RCP<vector_t> vec = Teuchos::rcp_dynamic_cast<vector_t>( OBJECTS[ivec], true );

  Teuchos::ArrayRCP<const scalar_t> data = vec->getData();

  intercomm.Gatherv( (void *)(data.get()), data.size(), MPI_SCALAR, NULL, NULL, NULL, MPI_SCALAR, 0 );
}


/* VECTOR_DOT: compute frobenius norm
   
     -> broadcast 1 HANDLE vector1_handle
     -> broadcast 1 HANDLE vector2_handle
    <-  gather 1 SCALAR norm
*/
void vector_dot( MPI::Intercomm intercomm ) {

  handle_t ivec;
  intercomm.Bcast( (void *)(&ivec), 1, MPI_HANDLE, 0 );
  Teuchos::RCP<vector_t> vector1 = Teuchos::rcp_dynamic_cast<vector_t>( OBJECTS[ivec], true );
  intercomm.Bcast( (void *)(&ivec), 1, MPI_HANDLE, 0 );
  Teuchos::RCP<vector_t> vector2 = Teuchos::rcp_dynamic_cast<vector_t>( OBJECTS[ivec], true );

  scalar_t dot = vector1->dot( *vector2 );

  intercomm.Gather( (void *)(&dot), 1, MPI_SCALAR, NULL, 1, MPI_SCALAR, 0 );
}


/* VECTOR_NORM: compute frobenius norm
   
     -> broadcast 1 HANDLE vector_handle
    <-  gather 1 SCALAR norm
*/
void vector_norm( MPI::Intercomm intercomm ) {

  handle_t ivec;
  intercomm.Bcast( (void *)(&ivec), 1, MPI_HANDLE, 0 );
  Teuchos::RCP<vector_t> vector = Teuchos::rcp_dynamic_cast<vector_t>( OBJECTS[ivec], true );

  scalar_t norm = vector->norm2();

  intercomm.Gather( (void *)(&norm), 1, MPI_SCALAR, NULL, 1, MPI_SCALAR, 0 );
}


/* NEW_GRAPH: create new graph
   
     -> broadcast 4 HANDLE (rowmap,colmap,dommap,rngmap)_handle
     -> scatterv nrows+1 SIZE offsets
     -> scatterv offsets[-1] LOCAL columns (concatenated)
    <-  gather 1 HANDLE graph_handle
*/
void new_graph( MPI::Intercomm intercomm ) {

  handle_t igraph = OBJECTS.size();

  handle_t imap[4];
  intercomm.Bcast( (void *)imap, 4, MPI_HANDLE, 0 );

  Teuchos::RCP<const map_t> rowmap = Teuchos::rcp_dynamic_cast<const map_t>( OBJECTS[imap[0]], true );
  Teuchos::RCP<const map_t> colmap = Teuchos::rcp_dynamic_cast<const map_t>( OBJECTS[imap[1]], true );
  Teuchos::RCP<const map_t> dommap = Teuchos::rcp_dynamic_cast<const map_t>( OBJECTS[imap[2]], true );
  Teuchos::RCP<const map_t> rngmap = Teuchos::rcp_dynamic_cast<const map_t>( OBJECTS[imap[3]], true );

  size_t nrows = rowmap->getNodeNumElements();

  out(intercomm) << "creating graph #" << igraph << " from rowmap #" << imap[0] << ", colmap #" << imap[1] << ", dommap #" << imap[2] << ", rngmap #" << imap[3] << " with " << nrows << " rows" << std::endl;

  Teuchos::ArrayRCP<size_t> offsets( nrows+1 );
  intercomm.Scatterv( NULL, NULL, NULL, MPI_SIZE, (void *)offsets.getRawPtr(), nrows+1, MPI_SIZE, 0 );

  int nindices = offsets[nrows];
  Teuchos::ArrayRCP<local_t> indices( nindices );
  intercomm.Scatterv( NULL, NULL, NULL, MPI_LOCAL, (void *)indices.getRawPtr(), nindices, MPI_LOCAL, 0 );

  Teuchos::RCP<graph_t> graph = Teuchos::rcp( new graph_t( rowmap, colmap, offsets, indices ) );
  graph->fillComplete( dommap, rngmap );

  OBJECTS.push_back( graph );

  intercomm.Gather( (void *)(&igraph), 1, MPI_HANDLE, NULL, 1, MPI_HANDLE, 0 );
}


/* NEW_MATRIX: create new matrix
   
     -> broadcast (HANDLE) graph id
    <-  gather (HANDLE) matrix id
*/
void new_matrix( MPI::Intercomm intercomm ) {

  handle_t imat = OBJECTS.size();

  handle_t igraph;
  intercomm.Bcast( (void *)(&igraph), 1, MPI_HANDLE, 0 );
  Teuchos::RCP<const graph_t> graph = Teuchos::rcp_dynamic_cast<const graph_t>( OBJECTS[igraph], true );

  out(intercomm) << "creating matrix #" << imat << " from graph #" << igraph << std::endl;

  OBJECTS.push_back( Teuchos::rcp( new matrix_t( graph ) ) );

  intercomm.Gather( (void *)(&imat), 1, MPI_HANDLE, NULL, 1, MPI_HANDLE, 0 );
}


/* ADD_EMAT: add items to matrix
   
     -> broadcast 1 SIZE rank
   if rank == myrank
     -> recv 1 HANDLE vector_handle
     -> recv 2 SIZE number_of_(rows,cols)
     -> recv number_of_rows GLOBAL indices
     -> recv number_of_cols GLOBAL indices
     -> recv number_of_(rows*cols) SCALAR values
   endif
*/
void add_emat( MPI::Intercomm intercomm ) {

  size_t rank;
  intercomm.Bcast( (void *)(&rank), 1, MPI_SIZE, 0 );

  if ( rank != intercomm.Get_rank() ) {
    return;
  }

  handle_t imat;
  intercomm.Recv( (void *)(&imat), 1, MPI_HANDLE, 0, 0 );

  size_t nitems[2];
  intercomm.Recv( (void *)nitems, 2, MPI_SIZE, 0, 0 );

  out(intercomm) << "imat = " << imat << ", nitems = " << nitems[0] << "," << nitems[1] << std::endl;

  Teuchos::ArrayRCP<global_t> rowidx( nitems[0] );
  Teuchos::ArrayRCP<global_t> colidx( nitems[1] );
  Teuchos::ArrayRCP<scalar_t> data( nitems[0]*nitems[1] );

  intercomm.Recv( (void *)rowidx.getRawPtr(), nitems[0], MPI_GLOBAL, 0, 0 );
  intercomm.Recv( (void *)colidx.getRawPtr(), nitems[1], MPI_GLOBAL, 0, 0 );
  intercomm.Recv( (void *)data.getRawPtr(), nitems[0]*nitems[1], MPI_SCALAR, 0, 0 );

  Teuchos::RCP<matrix_t> mat = Teuchos::rcp_dynamic_cast<matrix_t>( OBJECTS[imat], true );

  const Teuchos::ArrayView<const global_t> colidx_view = colidx.view( 0, nitems[1] );
  for ( int i = 0; i < nitems[0]; i++ ) {
    mat->sumIntoGlobalValues( rowidx[i], colidx_view, data.view(i*nitems[0],nitems[1]) );
  }

}


/* FILL_COMPLETE: set matrix to fill-complete
   
     -> broadcast 1 HANDLE matrix_handle
*/
void fill_complete( MPI::Intercomm intercomm ) {

  handle_t imat;
  intercomm.Bcast( (void *)(&imat), 1, MPI_HANDLE, 0 );
  Teuchos::RCP<matrix_t> matrix = Teuchos::rcp_dynamic_cast<matrix_t>( OBJECTS[imat], true );

  out(intercomm) << "completing matrix #" << imat << std::endl;

  matrix->fillComplete( matrix->getDomainMap(), matrix->getRangeMap() );
}


/* MATRIX_NORM: compute frobenius norm
   
     -> broadcast 1 HANDLE matrix_handle
    <-  gather 1 SCALAR norm
*/
void matrix_norm( MPI::Intercomm intercomm ) {

  handle_t imat;
  intercomm.Bcast( (void *)(&imat), 1, MPI_HANDLE, 0 );
  Teuchos::RCP<matrix_t> matrix = Teuchos::rcp_dynamic_cast<matrix_t>( OBJECTS[imat], true );

  scalar_t norm = matrix->getFrobeniusNorm();

  intercomm.Gather( (void *)(&norm), 1, MPI_SCALAR, NULL, 1, MPI_SCALAR, 0 );
}


/* MATVEC: matrix vector multiplication
   
     -> broadcast 3 HANDLE (matrix,out,vector)_handle
*/
void matvec( MPI::Intercomm intercomm ) {

  handle_t handles[3];
  intercomm.Bcast( (void *)handles, 3, MPI_HANDLE, 0 );

  Teuchos::RCP<matrix_t> matrix = Teuchos::rcp_dynamic_cast<matrix_t>( OBJECTS[handles[0]], true );
  Teuchos::RCP<vector_t> vector = Teuchos::rcp_dynamic_cast<vector_t>( OBJECTS[handles[1]], true );
  Teuchos::RCP<vector_t> out = Teuchos::rcp_dynamic_cast<vector_t>( OBJECTS[handles[2]], true );

  matrix->apply( *vector, *out );
}


/* MPI SETUP CODE */


typedef void ( *funcptr )( MPI::Intercomm );
#define TOKENS new_vector, add_evec, get_vector, new_map, new_graph, new_matrix, add_emat, fill_complete, matrix_norm, matvec, vector_norm, vector_dot
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
