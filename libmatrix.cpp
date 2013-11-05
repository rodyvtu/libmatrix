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

typedef double scalar_type;
typedef int local_ordinal_type;
typedef long global_ordinal_type;
typedef Kokkos::DefaultNode::DefaultNodeType node_type;
typedef Tpetra::Map<local_ordinal_type, global_ordinal_type, node_type> map_type;
typedef Tpetra::Vector<scalar_type, local_ordinal_type, global_ordinal_type, node_type> vector_type;
typedef Tpetra::CrsMatrix<scalar_type, local_ordinal_type, global_ordinal_type> matrix_type;
typedef Tpetra::CrsGraph<local_ordinal_type, global_ordinal_type, node_type> graph_type;


const global_ordinal_type indexBase = 0;

Teuchos::Array< Teuchos::RCP<vector_type> > VECTORS;
Teuchos::Array< Teuchos::RCP<matrix_type> > MATRICES;
Teuchos::Array< Teuchos::RCP<const map_type> > MAPS;
Teuchos::Array< Teuchos::RCP<const graph_type> > GRAPHS;


/*-------------------------*
 |                         |
 |  MISC HELPER FUNCTIONS  |
 |                         |
 *-------------------------*/


bool verify( bool good, const char *msg, MPI::Intercomm intercomm ) {
  int status = good ? 0 : strlen( msg );
  intercomm.Gather( (void *)(&status), 1, MPI::INT, NULL, 1, MPI::INT, 0 );
  if ( ! good ) {
    intercomm.Send( msg, status, MPI::CHAR, 0, 10 );
  }
  return good;
}


inline std::ostream& out( MPI::Intercomm intercomm ) {
  return
  #ifdef DEBUG
    std::cout << '[' << intercomm.Get_rank() << '/' << intercomm.Get_size() << "] ";
  #else
    blackHole;
  #endif
}


/*-------------------------*
 |                         |
 |      LIBMATRIX API      |
 |                         |
 *-------------------------*/


/* NEW_MAP: create new map
   
     -> broadcast map size
     -> scatter number of items
     -> scatterv items
    <-  gather map id
*/
void new_map( MPI::Intercomm intercomm ) {

  int imap = MAPS.size();

  int size, ndofs;
  intercomm.Bcast( (void *)(&size), 1, MPI::INT, 0 );
  intercomm.Scatter( NULL, 1, MPI::INT, (void *)(&ndofs), 1, MPI::INT, 0 );

  out(intercomm) << "creating map #" << imap << " with " << ndofs << '/' << size << " items" << std::endl;

  Teuchos::Array<global_ordinal_type> elementList( ndofs );
  intercomm.Scatterv( NULL, NULL, NULL, MPI::LONG, (void *)elementList.getRawPtr(), ndofs, MPI::LONG, 0 );

  Teuchos::RCP<node_type> node = Kokkos::DefaultNode::getDefaultNode ();
  Teuchos::RCP<const Teuchos::Comm<int> > comm = Tpetra::DefaultPlatform::getDefaultPlatform().getComm();

  MAPS.push_back( Teuchos::rcp( new map_type( size, elementList, indexBase, comm, node ) ) );

  intercomm.Gather( (void *)(&imap), 1, MPI::INT, NULL, 1, MPI::INT, 0 );
}


/* NEW_VECTOR: create new vector
   
     -> broadcast map id
    <-  gather vector id
*/
void new_vector( MPI::Intercomm intercomm ) {

  int ivec = VECTORS.size();

  int imap;
  intercomm.Bcast( (void *)(&imap), 1, MPI::INT, 0 );
  Teuchos::RCP<const map_type> map = MAPS[imap];

  out(intercomm) << "creating vector #" << ivec << " from map #" << imap << std::endl;

  VECTORS.push_back( Teuchos::rcp( new vector_type( map ) ) );

  intercomm.Gather( (void *)(&ivec), 1, MPI::INT, NULL, 1, MPI::INT, 0 );
}


/* NEW_GRAPH: create new graph
   
     -> broadcast map id
     -> scatterv ncolumns per row
     -> scatterv columns (concatenated)
    <-  gather graph id
*/
void new_graph( MPI::Intercomm intercomm ) {

  int igraph = GRAPHS.size();

  int imap;
  intercomm.Bcast( (void *)(&imap), 1, MPI::INT, 0 );
  Teuchos::RCP<const map_type> map = MAPS[imap];

  int nrows = map->getNodeNumElements();
  out(intercomm) << "creating graph #" << igraph << " from map #" << imap << " with " << nrows << " rows" << std::endl;

  Teuchos::ArrayRCP<const size_t> numcols( nrows );
  const size_t *numcols_ptr = numcols.getRawPtr();

  intercomm.Scatterv( NULL, NULL, NULL, MPI::INT, (void *)numcols_ptr, nrows, MPI::INT, 0 );

  int nitems = 0;
  for ( int irow = 0; irow < nrows; irow++ ) {
    nitems += numcols_ptr[ irow ]; // TODO check if ArrayRCP supports summation
  }

  Teuchos::ArrayRCP<global_ordinal_type> items( nitems );
  intercomm.Scatterv( NULL, NULL, NULL, MPI::LONG, (void *)items.getRawPtr(), nitems, MPI::LONG, 0 );

  Teuchos::RCP<graph_type> graph = Teuchos::rcp( new graph_type( map, numcols ) );
  size_t offset = 0;
  for ( int irow = 0; irow < nrows; irow++ ) {
    size_t size = numcols_ptr[ irow ];
    graph->insertGlobalIndices( irow, items.view(offset,size) );
    offset += size;
  }
  graph->fillComplete();

  GRAPHS.push_back( graph );

  intercomm.Gather( (void *)(&igraph), 1, MPI::INT, NULL, 1, MPI::INT, 0 );
}


/* ADD_EVEC: add items to vector
   
     -> broadcast rank
   if rank == myrank
     -> recv vector id
     -> recv number of items
     -> recv indices
     -> recv values
   endif
*/
void add_evec( MPI::Intercomm intercomm ) {

  int rank;
  intercomm.Bcast( (void *)(&rank), 1, MPI::INT, 0 );

  if ( rank != intercomm.Get_rank() ) {
    return;
  }

  int ivec, nitems;
  intercomm.Recv( (void *)(&ivec), 1, MPI::INT, 0, 0 );
  intercomm.Recv( (void *)(&nitems), 1, MPI::INT, 0, 0 );

  out(intercomm) << "ivec = " << ivec << ", nitems = " << nitems << std::endl;

  Teuchos::ArrayRCP<global_ordinal_type> idx( nitems );
  Teuchos::ArrayRCP<scalar_type> data( nitems );

  intercomm.Recv( (void *)idx.getRawPtr(), nitems, MPI::LONG, 0, 0 );
  intercomm.Recv( (void *)data.getRawPtr(), nitems, MPI::DOUBLE, 0, 0 );

  Teuchos::RCP<vector_type> vec = VECTORS[ivec];

  for ( int i = 0; i < nitems; i++ ) {
    out(intercomm) << idx[i] << " : " << data[i] << std::endl;
    vec->sumIntoGlobalValue( idx[i], data[i] );
  }

}


/* GET_VECTOR: collect vector over the intercom
  
     -> broadcast vector id
    <-  gatherv values
*/
void get_vector( MPI::Intercomm intercomm ) {

  int ivec;
  intercomm.Bcast( (void *)(&ivec), 1, MPI::INT, 0 );
  Teuchos::ArrayRCP<const scalar_type> data = VECTORS[ivec]->getData();

  intercomm.Gatherv( (void *)(data.get()), data.size(), MPI::DOUBLE, NULL, NULL, NULL, MPI::DOUBLE, 0 );
}


/* NEW_MATRIX: create new matrix
   
     -> broadcast graph id
    <-  gather matrix id
*/
void new_matrix( MPI::Intercomm intercomm ) {

  int imat = MATRICES.size();

  int igraph;
  intercomm.Bcast( (void *)(&igraph), 1, MPI::INT, 0 );
  Teuchos::RCP<const graph_type> graph = GRAPHS[igraph];

  out(intercomm) << "creating matrix #" << imat << " from graph #" << igraph << std::endl;

  MATRICES.push_back( Teuchos::rcp( new matrix_type( graph ) ) );

  intercomm.Gather( (void *)(&imat), 1, MPI::INT, NULL, 1, MPI::INT, 0 );
}


/*-------------------------*
 |                         |
 |     MPI SETUP CODE      |
 |                         |
 *-------------------------*/


typedef void ( *funcptr )( MPI::Intercomm );
#define TOKENS new_matrix, new_vector, add_evec, get_vector, new_map, new_graph
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

  if ( argc == 2 && std::strcmp( argv[1], "getfuncs" ) == 0 ) {
    std::cout << STR(TOKENS) << '\n';
  }
  else if ( argc == 2 && std::strcmp( argv[1], "eventloop" ) == 0 ) {
    eventloop( argv[0] );
  }
  else {
    std::cout << "syntax: " << argv[0] << " getfuncs|eventloop" << std::endl;
    return 1;
  }
  return 0;
}


// vim:foldmethod=syntax
