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


class ObjectArray {

public:

  inline void set_object( handle_t handle, Teuchos::RCP<Teuchos::Describable> object ) {
    if ( handle == objects.size() ) {
      objects.push_back( object );
    }
    else {
      ASSERT( objects[handle].is_null() );
      objects[handle] = object;
    }
  }
  
  template <class T>
  inline Teuchos::RCP<T> get_object( handle_t handle ) {
    return Teuchos::rcp_dynamic_cast<T>( objects[handle], true );
  }
  
  inline void release_object( handle_t handle ) {
    objects[handle] = Teuchos::RCP<Teuchos::Describable>();
  }

private:

  Teuchos::Array<Teuchos::RCP<Teuchos::Describable> > objects;

};

class Intercomm {

public:
  
  Intercomm( char *progname ) {
    int argc = 1;
    char **argv = &progname;
    MPI::Init( argc, argv ); 
    MPI::COMM_WORLD.Set_errhandler( MPI::ERRORS_THROW_EXCEPTIONS );
    comm = MPI::Comm::Get_parent();
    myrank = comm.Get_rank();
    nprocs = comm.Get_size();
  }

  ~Intercomm() {
    comm.Disconnect();
    MPI::Finalize(); 
  }

  template <class T>
  inline void bcast( T *data, const int n=1 ) {
    comm.Bcast( (void *)data, n * sizeof(T), MPI::BYTE, 0 );
  }
  
  template <class T>
  inline void scatter( T *data, const int n=1 ) {
    comm.Scatter( NULL, 0, MPI::BYTE, (void *)data, n * sizeof(T), MPI::BYTE, 0 );
  }
  
  template <class T>
  inline void scatterv( T *data, const int n=1 ) {
    comm.Scatterv( NULL, NULL, NULL, MPI::BYTE, (void *)data, n * sizeof(T), MPI::BYTE, 0 );
  }
  
  template <class T>
  inline void recv( T *data, const int n=1, const int tag=0 ) {
    comm.Recv( (void *)data, n * sizeof(T), MPI::BYTE, tag, 0 );
  }
  
  template <class T>
  inline void gatherv( T *data, const int n=1 ) {
    comm.Gatherv( (void *)data, n * sizeof(T), MPI::BYTE, NULL, NULL, NULL, MPI::BYTE, 0 );
  }
  
  template <class T>
  inline void gather( T *data, const int n=1 ) {
    comm.Gather( (void *)data, n * sizeof(T), MPI::BYTE, NULL, 0, MPI::BYTE, 0 );
  }

public:

  int myrank, nprocs;

private:

  MPI::Intercomm comm;

};

class LibMatrix : public Intercomm, public ObjectArray {

public:

  LibMatrix( char *progname ) : Intercomm( progname ) {}

  typedef void (LibMatrix::*funcptr)();

  void params_new() /* create new parameter list
     
      -> broadcast 1 HANDLE params_handle
  */{
  
    struct { handle_t params; } handle;
    bcast( &handle );
  
    Teuchos::RCP<params_t> params = Teuchos::rcp( new params_t );
  
    out() << "creating parameter list #" << handle.params << std::endl;
  
    set_object( handle.params, params );
  
  }
  
  template <class T>
  void params_set() /* set new integer in parameter list
     
      -> broadcast HANDLE params_handle 
      -> broadcast SIZE length_of_key
      -> broadcast CHAR key[length_of_key]
      -> broadcast TEMLATE_ARG value
  */{
  
    struct { handle_t params; } handle;
    bcast( &handle );
  
    size_t nchar;
    bcast( &nchar );
  
    std::string key( nchar, 0 );
    bcast( const_cast<char*>(key.data()), nchar );
  
    T value;
    bcast( &value );
  
    Teuchos::RCP<params_t> params = get_object<params_t>( handle.params );
    params->set( key, value );
  
    out() << "added key=\"" << key << "\" with value=" << value << std::endl;
  }
  
  void params_print() /* print the params list (c-sided)
     
      -> broadcast 1 HANDLE params_handle 
  */{
  
    struct { handle_t params; } handle;
    bcast( &handle );
  
    Teuchos::RCP<params_t> params = get_object<params_t>( handle.params );
    params->print( out() );
  }
  
  void release() /* release object
     
       -> broadcast HANDLE handle
  */{
  
    handle_t handle;
    bcast( &handle );
    release_object( handle );
  }
  
  void map_new() /* create new map
     
       -> broadcast HANDLE handle.map
       -> broadcast SIZE map_size
       -> scatter SIZE number_of_items[map_size]
       -> scatterv GLOBAL items[number_of_items]
  */{
  
    struct { handle_t map; } handle;
    bcast( &handle );
  
    size_t size, ndofs;
    bcast( &size );
    scatter( &ndofs );
  
    out() << "creating map #" << handle.map << " with " << ndofs << '/' << size << " items" << std::endl;
  
    Teuchos::Array<global_t> elementList( ndofs );
    scatterv( elementList.getRawPtr(), ndofs );
  
    Teuchos::RCP<node_t> node = Kokkos::DefaultNode::getDefaultNode ();
    Teuchos::RCP<const Teuchos::Comm<int> > comm = Tpetra::DefaultPlatform::getDefaultPlatform().getComm();
    Teuchos::RCP<map_t> map = Teuchos::rcp( new map_t( size, elementList, indexbase, comm, node ) );
  
    set_object( handle.map, map );
  }
  
  void vector_new() /* create new vector
     
       -> broadcast HANDLE handle.{vector,map}
  */{
  
    struct { handle_t vector, map; } handle;
    bcast( &handle );
  
    Teuchos::RCP<const map_t> map = get_object<const map_t>( handle.map );
  
    out() << "creating vector #" << handle.vector << " from map #" << handle.map << std::endl;
  
    Teuchos::RCP<vector_t> vector = Teuchos::rcp( new vector_t( map ) );
  
    set_object( handle.vector, vector );
  }
  
  void vector_add_block() /* add items to vector
     
       -> broadcast SIZE rank
     if rank == myrank
       -> recv HANDLE handle.vector
       -> recv SIZE number_of_items
       -> recv GLOBAL indices[number_of_items]
       -> recv SCALAR values[number_of_items]
     endif
  */{
  
    size_t rank;
    bcast( &rank );
  
    if ( rank != myrank ) {
      return;
    }
  
    struct { handle_t vector; } handle;
    recv( &handle );
  
    size_t nitems;
    recv( &nitems );
  
    out() << "ivec = " << handle.vector << ", nitems = " << nitems << std::endl;
  
    Teuchos::ArrayRCP<local_t> idx( nitems );
    Teuchos::ArrayRCP<scalar_t> data( nitems );
  
    recv( idx.getRawPtr(), nitems );
    recv( data.getRawPtr(), nitems );
  
    Teuchos::RCP<vector_t> vec = get_object<vector_t>( handle.vector );
  
    for ( int i = 0; i < nitems; i++ ) {
      out() << idx[i] << " : " << data[i] << std::endl;
      vec->sumIntoLocalValue( idx[i], data[i] );
    }
  
  }
  
  void vector_getdata() /* collect vector over the intercom
    
       -> broadcast HANDLE handle.vector
      <-  gatherv SCALAR values[vector.size]
  */{
  
    struct { handle_t vector; } handle;
    bcast( &handle );
  
    Teuchos::RCP<vector_t> vec = get_object<vector_t>( handle.vector );
  
    Teuchos::ArrayRCP<const scalar_t> data = vec->getData();
  
    gatherv( data.get(), data.size() );
  }
  
  void vector_dot() /* compute frobenius norm
     
       -> broadcast HANDLE handle.{vector1,vector2}
      <-  gather SCALAR norm
  */{
  
    struct { handle_t vector1, vector2; } handle;
    bcast( &handle );
    Teuchos::RCP<vector_t> vector1 = get_object<vector_t>( handle.vector1 );
    Teuchos::RCP<vector_t> vector2 = get_object<vector_t>( handle.vector2 );
  
    scalar_t dot = vector1->dot( *vector2 );
  
    gather( &dot );
  }
  
  void vector_norm() /* compute frobenius norm
     
       -> broadcast HANDLE handle.vector
      <-  gather SCALAR norm
  */{
  
    struct { handle_t vector; } handle;
    bcast( &handle );
    Teuchos::RCP<vector_t> vector = get_object<vector_t>( handle.vector );
  
    scalar_t norm = vector->norm2();
  
    gather( &norm );
  }
  
  void graph_new() /* create new graph
     
       -> broadcast HANDLE handle.{graph,rowmap,colmap,domainmap,rangemap}
       -> scatterv SIZE offsets[nrows+1]
       -> scatterv LOCAL columns[offsets[-1]]
  */{
  
    struct { handle_t graph, rowmap, colmap; } handle;
    bcast( &handle );
  
    Teuchos::RCP<const map_t> rowmap = get_object<const map_t>( handle.rowmap );
    Teuchos::RCP<const map_t> colmap = get_object<const map_t>( handle.colmap );
  
    size_t nrows = rowmap->getNodeNumElements();
  
    out() << "creating graph #" << handle.graph << " from rowmap #" << handle.rowmap << ", colmap #" << handle.colmap << " with " << nrows << " rows" << std::endl;
  
    Teuchos::ArrayRCP<size_t> offsets( nrows+1 );
    scatterv( offsets.getRawPtr(), nrows+1 );
  
    int nindices = offsets[nrows];
    Teuchos::ArrayRCP<local_t> indices( nindices );
    scatterv( indices.getRawPtr(), nindices );
  
    Teuchos::RCP<graph_t> graph = Teuchos::rcp( new graph_t( rowmap, colmap, offsets, indices ) );
    graph->fillComplete();
  
    set_object( handle.graph, graph );
  }
  
  void matrix_new() /* create new matrix
     
       -> broadcast HANDLE handle.{matrix,graph}
  */{
  
    struct { handle_t matrix, graph; } handle;
    bcast( &handle );
  
    Teuchos::RCP<const graph_t> graph = get_object<const graph_t>( handle.graph );
  
    out() << "creating matrix #" << handle.matrix << " from graph #" << handle.graph << std::endl;
  
    Teuchos::RCP<matrix_t> matrix = Teuchos::rcp( new matrix_t( graph ) );
  
    set_object( handle.matrix, matrix );
  }
  
  void matrix_add_block() /* add items to matrix
     
       -> broadcast SIZE rank
     if rank == myrank
       -> recv HANDLE handle.vector
       -> recv SIZE number_of_{rows,cols}
       -> recv GLOBAL indices[number_of_rows]
       -> recv GLOBAL indices[number_of_cols]
       -> recv SCALAR values[number_of_(rows*cols)]
     endif
  */{
  
    size_t rank;
    bcast( &rank );
  
    if ( rank != myrank ) {
      return;
    }
  
    struct { handle_t matrix; } handle;
    recv( &handle );
  
    size_t nitems[2];
    recv( nitems, 2 );
  
    out() << "imat = " << handle.matrix << ", nitems = " << nitems[0] << "," << nitems[1] << std::endl;
  
    Teuchos::ArrayRCP<local_t> rowidx( nitems[0] );
    Teuchos::ArrayRCP<local_t> colidx( nitems[1] );
    Teuchos::ArrayRCP<scalar_t> data( nitems[0]*nitems[1] );
  
    recv( rowidx.getRawPtr(), nitems[0] );
    recv( colidx.getRawPtr(), nitems[1] );
    recv( data.getRawPtr(), nitems[0]*nitems[1] );
  
    Teuchos::RCP<matrix_t> mat = get_object<matrix_t>( handle.matrix );
  
    const Teuchos::ArrayView<const local_t> colidx_view = colidx.view( 0, nitems[1] );
    for ( int i = 0; i < nitems[0]; i++ ) {
      mat->sumIntoLocalValues( rowidx[i], colidx_view, data.view(i*nitems[0],nitems[1]) );
    }
  
  }
  
  void matrix_complete() /* set matrix to fill-complete
     
       -> broadcast HANDLE handle.{matrix,exporter}
  */{
  
    struct { handle_t matrix, exporter; } handle;
    bcast( &handle );
  
    Teuchos::RCP<const matrix_t> matrix = get_object<const matrix_t>( handle.matrix );
    Teuchos::RCP<const export_t> exporter = get_object<const export_t>( handle.exporter );
    Teuchos::RCP<const map_t> domainmap = exporter->getTargetMap();
    Teuchos::RCP<const map_t> rangemap = exporter->getTargetMap();
  
    out() << "completing matrix #" << handle.matrix << std::endl;
  
    Teuchos::RCP<matrix_t> completed_matrix = Tpetra::exportAndFillCompleteCrsMatrix( matrix, *exporter, domainmap, rangemap );
  
    release_object( handle.matrix );
    set_object( handle.matrix, completed_matrix );
  }
  
  void matrix_norm() /* compute frobenius norm
     
       -> broadcast HANDLE handle.matrix
      <-  gather SCALAR norm
  */{
  
    struct { handle_t matrix; } handle;
    bcast( &handle );
    Teuchos::RCP<matrix_t> matrix = get_object<matrix_t>( handle.matrix );
  
    scalar_t norm = matrix->getFrobeniusNorm();
  
    gather( &norm );
  }
  
  void matrix_apply() /* matrix vector multiplication
     
       -> broadcast HANDLE handle.{matrix,out,vector}
  */{
  
    struct { handle_t matrix, rhs, lhs; } handle;
    bcast( &handle );
  
    Teuchos::RCP<matrix_t> matrix = get_object<matrix_t>( handle.matrix );
    Teuchos::RCP<vector_t> rhs = get_object<vector_t>( handle.rhs );
    Teuchos::RCP<vector_t> lhs = get_object<vector_t>( handle.lhs );
  
    matrix->apply( *rhs, *lhs );
  }
  
  void matrix_solve() /* solve linear system
     
       -> broadcast HANDLE handle.{matrix,precon,rhs,lhs,solvertype,solverparams}
       -> broadcast BOOL symmetric
  */{
  
    struct { handle_t matrix, precon, rhs, lhs, solvertype, solverparams; } handle;
    bcast( &handle );
  
    bool_t symmetric;
    bcast( &symmetric );
  
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
  
    out() << "solver finished in " << numIters << " iterations with result " << result << std::endl;
  }
  
  void precon_new() /* create new preconditioner
     
       -> broadcast HANDLE handle.{precon,matrix,precontype,preconparams}
  */{
  
    struct { handle_t precon, matrix, precontype, preconparams; } handle;
    bcast( &handle );
  
    Teuchos::RCP<const matrix_t> matrix = get_object<const matrix_t>( handle.matrix );
    Teuchos::RCP<const params_t> preconparams = get_object<const params_t>( handle.preconparams );
   
    preconfactory_t factory;
    Teuchos::RCP<precon_t> precon = factory.create( supportedPreconNames[handle.precontype], matrix );
  
    precon->setParameters( *preconparams );
    precon->initialize();
    precon->compute();
  
    magnitude_t condest = precon->computeCondEst( Ifpack2::Cheap );
    out() << "Ifpack2 preconditioner's estimated condition number: " << condest << std::endl;
  
    set_object( handle.precon, precon );
  }
  
  void export_new() /* create new exporter
     
       -> broadcast HANDLE handle.{exporter,srcmap,dstmap}
  */{
  
    struct { handle_t exporter, srcmap, dstmap; } handle;
    bcast( &handle );
  
    Teuchos::RCP<const map_t> srcmap = get_object<const map_t>( handle.srcmap );
    Teuchos::RCP<const map_t> dstmap = get_object<const map_t>( handle.dstmap );
  
    Teuchos::RCP<export_t> exporter = Teuchos::rcp( new export_t( srcmap, dstmap ) );
  
    set_object( handle.exporter, exporter );
  }

  void toggle_stdout() /* switch std output on/off
  */{
  
    if ( verbose ) {
      out() << "output is OFF" << std::endl;
      verbose = false;
    }
    else {
      verbose = true;
      out() << "output is ON" << std::endl;
    }
  }

  inline std::ostream& out() {
    return verbose ? std::cout << '[' << myrank << '/' << nprocs << "] " : blackHole;
  }

private:

  bool verbose = false;

};

void eventloop( char *progname ) {

  LibMatrix::funcptr FTABLE[] = { FUNCS };
  const size_t NFUNCS = sizeof(FTABLE) / sizeof(LibMatrix::funcptr);

  LibMatrix intercomm( progname );
  token_t c;
  for ( ;; ) {
    intercomm.out() << "waiting" << std::endl;
    intercomm.bcast( &c );
    intercomm.out() << "received " << c << std::endl;
    if ( c >= NFUNCS ) {
      intercomm.out() << "quit" << std::endl;
      break;
    }
    (intercomm.*FTABLE[c])();
  }
}

int main( int argc, char *argv[] ) {

  if ( argc == 2 && std::strcmp( argv[1], "info" ) == 0 ) {

    std::cout << "tokens: " << FUNCNAMES << std::endl;

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
