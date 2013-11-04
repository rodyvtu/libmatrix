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


const global_ordinal_type indexBase = 0;

Teuchos::RCP<vector_type> VECTORS[256];
int nvectors = 0;

Teuchos::RCP<matrix_type> MATRICES[256];
int nmatrices = 0;

Teuchos::RCP<const map_type> MAPS[256];
int nmaps = 0;


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

  int imap = nmaps;
  nmaps++;

  int size, ndofs;
  intercomm.Bcast( (void *)(&size), 1, MPI::INT, 0 );
  intercomm.Scatter( NULL, 1, MPI::INT, (void *)(&ndofs), 1, MPI::INT, 0 );

  out(intercomm) << "creating map #" << imap << " with " << ndofs << '/' << size << " items" << std::endl;

  Teuchos::Array<global_ordinal_type> elementList( ndofs );
  intercomm.Scatterv( NULL, NULL, NULL, MPI::LONG, (void *)elementList.getRawPtr(), ndofs, MPI::LONG, 0 );

  Teuchos::RCP<node_type> node = Kokkos::DefaultNode::getDefaultNode ();
  Teuchos::RCP<const Teuchos::Comm<int> > comm = Tpetra::DefaultPlatform::getDefaultPlatform().getComm();

  MAPS[imap] = Teuchos::rcp( new map_type( size, elementList, indexBase, comm, node ) );

  intercomm.Gather( (void *)(&imap), 1, MPI::INT, NULL, 1, MPI::INT, 0 );
}


/* NEW_VECTOR: create new vector
   
     -> broadcast map id
    <-  gather vector id
*/
void new_vector( MPI::Intercomm intercomm ) {

  int ivec = nvectors;
  nvectors++;

  int imap;
  intercomm.Bcast( (void *)(&imap), 1, MPI::INT, 0 );

  out(intercomm) << "creating vector #" << ivec << " from map #" << imap << std::endl;

  VECTORS[ivec] = Teuchos::rcp( new vector_type( MAPS[imap] ) );

  intercomm.Gather( (void *)(&ivec), 1, MPI::INT, NULL, 1, MPI::INT, 0 );
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

  Teuchos::Array<global_ordinal_type> idx( nitems );
  Teuchos::Array<scalar_type> data( nitems );

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
   
     -> broadcast map id
    <-  gather matrix id
*/
void new_matrix( MPI::Intercomm intercomm ) {

  int imat = nmatrices;
  nmatrices++;

  int imap;
  intercomm.Bcast( (void *)(&imap), 1, MPI::INT, 0 );

  out(intercomm) << "creating matrix #" << imat << " from map #" << imap << std::endl;

  MATRICES[imat] = Teuchos::rcp( new matrix_type( MAPS[imap], 0 ) );

  intercomm.Gather( (void *)(&imat), 1, MPI::INT, NULL, 1, MPI::INT, 0 );
}


/*-------------------------*
 |                         |
 |     MPI SETUP CODE      |
 |                         |
 *-------------------------*/


typedef void ( *funcptr )( MPI::Intercomm );
#define TOKENS new_matrix, new_vector, add_evec, get_vector, new_map
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
