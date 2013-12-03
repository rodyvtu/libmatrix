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

#define ASSERT( cond ) if ( ! (cond) ) throw( "assertion failed: " #cond );

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

template <class T>
void print_all( const char name[], T iterable ) {
  std::cout << name;
  char sep[] = ": ";
  for ( auto item : iterable ) {
    std::cout << sep << item;
    sep[0] = ',';
  }
  std::cout << std::endl;
}

template <class T, class V>
inline bool contains( T iterable, V value ) {
  for ( auto item : iterable ) {
    if ( item == value ) {
      return true;
    }
  }
  return false;
}

typedef double scalar_t;
typedef int number_t;
typedef int handle_t;
typedef int local_t;
typedef long global_t;
typedef uint8_t token_t;
typedef bool bool_t;

typedef Kokkos::DefaultNode::DefaultNodeType node_t;
typedef Tpetra::Map<local_t,global_t,node_t> map_t;
typedef Tpetra::Vector<scalar_t,local_t,global_t,node_t> vector_t;
typedef Tpetra::Operator<scalar_t,local_t,global_t,node_t> operator_t;
typedef Tpetra::MultiVector<scalar_t,local_t,global_t,node_t> multivector_t;
typedef Tpetra::CrsMatrix<scalar_t,local_t,global_t,node_t> crsmatrix_t;
typedef Tpetra::CrsGraph<local_t,global_t,node_t> crsgraph_t;
typedef Tpetra::RowGraph<local_t,global_t,node_t> rowgraph_t;
typedef Belos::SolverManager<scalar_t,multivector_t,operator_t> solvermanager_t;
typedef Belos::SolverFactory<scalar_t,multivector_t,operator_t> solverfactory_t;
typedef Ifpack2::Preconditioner<scalar_t,local_t,global_t,node_t> precon_t;
typedef Ifpack2::Factory preconfactory_t;
typedef Teuchos::ScalarTraits<scalar_t>::magnitudeType magnitude_t;
typedef Tpetra::Export<local_t,global_t,node_t> export_t;

class params_t : public Teuchos::Describable, public Teuchos::ParameterList {};

class linearproblem_t : public Teuchos::Describable, public Belos::LinearProblem<scalar_t,multivector_t,operator_t> {
public:
  linearproblem_t( const Teuchos::RCP<const operator_t> &A, const Teuchos::RCP<multivector_t> &X, const Teuchos::RCP<const multivector_t> &B) :
    Belos::LinearProblem<scalar_t,multivector_t,operator_t>( A, X, B ) {}
};


const global_t indexbase = 0;

class SetZero : public operator_t {

public:

  SetZero( Teuchos::RCP<vector_t> selection ) : selection( selection ) {}

  const Teuchos::RCP<const map_t> &getDomainMap() const {
    return selection->getMap();
  }

  const Teuchos::RCP<const map_t> &getRangeMap() const {
    return selection->getMap();
  }

  void apply(
      const multivector_t &x,
      multivector_t &y,
      Teuchos::ETransp mode=Teuchos::NO_TRANS,
      scalar_t alpha=Teuchos::ScalarTraits<scalar_t>::one(),
      scalar_t beta=Teuchos::ScalarTraits<scalar_t>::zero() ) const {
    // y = alpha C A C x + beta y
    y.elementWiseMultiply( alpha, *selection, x, beta );

  }

  bool hasTransposeApply() const {
    return true;
  }

private:
  
  const Teuchos::RCP<vector_t> selection;
};


class ObjectArray {

public:

  inline void set( handle_t handle, Teuchos::RCP<Teuchos::Describable> object, std::ostream &out ) {
    if ( out != blackHole ) {
      out << "set #" << handle << ": " << Teuchos::describe( *object, Teuchos::VERB_LOW ) << std::endl;
    }
    if ( handle == objects.size() ) {
      objects.push_back( object );
    }
    else {
      ASSERT( objects[handle].is_null() );
      objects[handle] = object;
    }
  }

  template <class T>
  inline Teuchos::RCP<T> get( handle_t handle, std::ostream &out ) {
    Teuchos::RCP<Teuchos::Describable> object = objects[handle];
    if ( out != blackHole ) {
      out << "get #" << handle << ": " << Teuchos::describe( *object, Teuchos::VERB_LOW ) << std::endl;
    }
    return Teuchos::rcp_dynamic_cast<T>( object, true );
  }

  inline void release( handle_t handle ) {
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

  void abort( int errorcode=1 ) {
    comm.Abort( errorcode );
  }

public:

  int myrank, nprocs;

private:

  MPI::Intercomm comm;

};

class LibMatrix : public Intercomm {

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
  
    objects.set( handle.params, params, out() );
  
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
  
    Teuchos::RCP<params_t> params = objects.get<params_t>( handle.params, out() );
    params->set( key, value );
  
    out() << "added key=\"" << key << "\" with value=" << value << std::endl;
  }
  
  void params_print() /* print the params list (c-sided)
     
      -> broadcast 1 HANDLE params_handle 
  */{
  
    struct { handle_t params; } handle;
    bcast( &handle );
  
    Teuchos::RCP<params_t> params = objects.get<params_t>( handle.params, out() );
    params->print( out() );
  }
  
  void release() /* release object
     
       -> broadcast HANDLE handle
  */{
  
    handle_t handle;
    bcast( &handle );
    objects.release( handle );
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
  
    objects.set( handle.map, map, out() );
  }
  
  void vector_new() /* create new vector
     
       -> broadcast HANDLE handle.{vector,map}
  */{
  
    struct { handle_t vector, map; } handle;
    bcast( &handle );
  
    Teuchos::RCP<const map_t> map = objects.get<const map_t>( handle.map, out() );
  
    out() << "creating vector #" << handle.vector << " from map #" << handle.map << std::endl;
  
    Teuchos::RCP<vector_t> vector = Teuchos::rcp( new vector_t( map ) );
  
    objects.set( handle.vector, vector, out() );
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
  
    Teuchos::RCP<vector_t> vec = objects.get<vector_t>( handle.vector, out() );
  
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
  
    Teuchos::RCP<vector_t> vec = objects.get<vector_t>( handle.vector, out() );
  
    Teuchos::ArrayRCP<const scalar_t> data = vec->getData();
  
    gatherv( data.get(), data.size() );
  }

  void vector_fill() /* fill vector with scalar value
    
       -> broadcast HANDLE handle.vector
       -> broadcast SCALAR value
  */{
  
    struct { handle_t vector; } handle;
    bcast( &handle );

    scalar_t value;
    bcast( &value );
  
    Teuchos::RCP<vector_t> vec = objects.get<vector_t>( handle.vector, out() );
    vec->putScalar( value );
  }
  
  void vector_dot() /* compute frobenius norm
     
       -> broadcast HANDLE handle.{vector1,vector2}
      <-  gather SCALAR norm
  */{
  
    struct { handle_t vector1, vector2; } handle;
    bcast( &handle );
    Teuchos::RCP<vector_t> vector1 = objects.get<vector_t>( handle.vector1, out() );
    Teuchos::RCP<vector_t> vector2 = objects.get<vector_t>( handle.vector2, out() );
  
    scalar_t dot = vector1->dot( *vector2 );
  
    gather( &dot );
  }
  
  void vector_norm() /* compute frobenius norm
     
       -> broadcast HANDLE handle.vector
      <-  gather SCALAR norm
  */{
  
    struct { handle_t vector; } handle;
    bcast( &handle );
    Teuchos::RCP<vector_t> vector = objects.get<vector_t>( handle.vector, out() );
  
    scalar_t norm = vector->norm2();
  
    gather( &norm );
  }

  void vector_complete() /* export vector

       -> broadcast HANDLE handle.{vector,exporter}
  */{

    struct { handle_t vector, exporter; } handle;
    bcast( &handle );
    Teuchos::RCP<vector_t> vector = objects.get<vector_t>( handle.vector, out() );
    Teuchos::RCP<export_t> exporter = objects.get<export_t>( handle.exporter, out() );
    Teuchos::RCP<const map_t> map = exporter->getTargetMap();

    Teuchos::RCP<vector_t> completed_vector = Teuchos::rcp( new vector_t( map ) );
    completed_vector->doExport( *vector, *exporter, Tpetra::ADD );

    objects.release( handle.vector );
    objects.set( handle.vector, completed_vector, out() );
  }

  void vector_or() /* logical OR vectors

       -> broadcast HANDLE handle.{self,other}
  */{

    struct { handle_t self, other; } handle;
    bcast( &handle );
    Teuchos::RCP<vector_t> self = objects.get<vector_t>( handle.self, out() );
    Teuchos::RCP<const vector_t> other = objects.get<vector_t>( handle.other, out() );
    ASSERT( self->getMap() == other->getMap() );
    Teuchos::ArrayRCP<scalar_t> _self = self->getDataNonConst();
    Teuchos::ArrayRCP<const scalar_t> _other = other->getData();
    for ( int i = 0; i < _self.size(); i++ ) {
      if ( std::isnan( _self[i] ) ) {
        _self[i] = _other[i];
      }
    }
  }

  void vector_axpy() /* logical OR vectors

       -> broadcast HANDLE handle.{self,other}
       -> broadcast SCALAR factor
  */{

    struct { handle_t self, other; } handle;
    bcast( &handle );
    scalar_t factor;
    bcast( &factor );
    Teuchos::RCP<vector_t> self = objects.get<vector_t>( handle.self, out() );
    Teuchos::RCP<const vector_t> other = objects.get<vector_t>( handle.other, out() );
    ASSERT( self->getMap() == other->getMap() );
    Teuchos::ArrayRCP<scalar_t> _self = self->getDataNonConst();
    Teuchos::ArrayRCP<const scalar_t> _other = other->getData();
    for ( int i = 0; i < _self.size(); i++ ) {
      _self[i] += factor * _other[i];
    }
  }

  void vector_copy() /* logical OR vectors

       -> broadcast HANDLE handle.{copy,orig}
  */{

    struct { handle_t copy, orig; } handle;
    bcast( &handle );
    Teuchos::RCP<const vector_t> orig = objects.get<const vector_t>( handle.orig, out() );
    Teuchos::RCP<vector_t> copy = Teuchos::rcp<vector_t>( new vector_t( *orig ) );
    objects.set( handle.copy, copy, out() );
  }
  
  void graph_new() /* create new graph
     
       -> broadcast HANDLE handle.{graph,rowmap,colmap,domainmap,rangemap}
       -> scatterv SIZE offsets[nrows+1]
       -> scatterv LOCAL columns[offsets[-1]]
  */{
  
    struct { handle_t graph, rowmap, colmap; } handle;
    bcast( &handle );
  
    Teuchos::RCP<const map_t> rowmap = objects.get<const map_t>( handle.rowmap, out() );
    Teuchos::RCP<const map_t> colmap = objects.get<const map_t>( handle.colmap, out() );
  
    size_t nrows = rowmap->getNodeNumElements();
  
    out() << "creating graph #" << handle.graph << " from rowmap #" << handle.rowmap << ", colmap #" << handle.colmap << " with " << nrows << " rows" << std::endl;
  
    Teuchos::ArrayRCP<size_t> offsets( nrows+1 );
    scatterv( offsets.getRawPtr(), nrows+1 );
  
    int nindices = offsets[nrows];
    Teuchos::ArrayRCP<local_t> indices( nindices );
    scatterv( indices.getRawPtr(), nindices );
  
    Teuchos::RCP<crsgraph_t> graph = Teuchos::rcp( new crsgraph_t( rowmap, colmap, offsets, indices ) );
    graph->fillComplete();
  
    objects.set( handle.graph, graph, out() );
  }
  
  void matrix_new_static() /* create new matrix
     
       -> broadcast HANDLE handle.{matrix,graph}
  */{
  
    struct { handle_t matrix, graph; } handle;
    bcast( &handle );
  
    Teuchos::RCP<const crsgraph_t> graph = objects.get<const crsgraph_t>( handle.graph, out() );
  
    out() << "creating matrix #" << handle.matrix << " from graph #" << handle.graph << std::endl;
  
    Teuchos::RCP<crsmatrix_t> matrix = Teuchos::rcp( new crsmatrix_t( graph ) );
  
    objects.set( handle.matrix, matrix, out() );
  }

  void matrix_new_dynamic() /* create new matrix

       -> broadcast HANDLE handle.{matrix,rowmap,colmap}
  */{

    struct { handle_t matrix, rowmap, colmap; } handle;
    bcast( &handle );

    out() << "creating matrix #" << handle.matrix << " from rowmap #" << handle.rowmap << " and colmap #" << handle.colmap << std::endl;

    Teuchos::RCP<const map_t> rowmap = objects.get<const map_t>( handle.rowmap, out() );
    Teuchos::RCP<const map_t> colmap = objects.get<const map_t>( handle.colmap, out() );

    Teuchos::RCP<crsmatrix_t> matrix = Teuchos::rcp( new crsmatrix_t( rowmap, colmap, 0 ) );

    objects.set( handle.matrix, matrix, out() );
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
    Teuchos::ArrayRCP<scalar_t> data( nitems[0] * nitems[1] );
  
    recv( rowidx.getRawPtr(), nitems[0] );
    recv( colidx.getRawPtr(), nitems[1] );
    recv( data.getRawPtr(), nitems[0] * nitems[1] );
  
    Teuchos::RCP<crsmatrix_t> mat = objects.get<crsmatrix_t>( handle.matrix, out() );
    Teuchos::RCP<const crsgraph_t> graph = mat->getCrsGraph();
  
    Teuchos::ArrayView<const local_t> current_icols;
    Teuchos::ArrayRCP<local_t> this_colidx( nitems[1] );
    Teuchos::ArrayRCP<scalar_t> this_data( nitems[1] );
    for ( local_t irow = 0; irow < nitems[0]; irow++ ) {
      int nnew = 0;
      graph->getLocalRowView( rowidx[irow], current_icols );
      for ( int icol = 0; icol < nitems[1]; icol++ ) {
        int i = contains( current_icols, colidx[icol] ) ? icol - nnew : nitems[1] - (++nnew);
        this_colidx[i] = colidx[icol];
        this_data[i] = data[irow*nitems[1]+icol];
      }
      if ( nnew > 0 ) {
        out() << "inserting " << nnew << " new items in row " << irow << std::endl;
        mat->insertLocalValues( rowidx[irow], this_colidx.view(nitems[1]-nnew,nnew), this_data.view(nitems[1]-nnew,nnew) );
      }
      if ( nnew < nitems[1] ) {
        out() << "adding " << nnew << " existing items in row " << irow << std::endl;
        mat->sumIntoLocalValues( rowidx[irow], this_colidx.view(0,nitems[1]-nnew), this_data.view(0,nitems[1]-nnew) );
      }
    }
  }
  
  void matrix_complete() /* export matrix and fill-complete
     
       -> broadcast HANDLE handle.{matrix,exporter}
  */{
  
    struct { handle_t matrix, exporter; } handle;
    bcast( &handle );
  
    Teuchos::RCP<const crsmatrix_t> matrix = objects.get<const crsmatrix_t>( handle.matrix, out() );
    Teuchos::RCP<const export_t> exporter = objects.get<const export_t>( handle.exporter, out() );
    Teuchos::RCP<const map_t> domainmap = exporter->getTargetMap();
    Teuchos::RCP<const map_t> rangemap = exporter->getTargetMap();
  
    out() << "completing matrix #" << handle.matrix << std::endl;
  
    Teuchos::RCP<crsmatrix_t> completed_matrix = Tpetra::exportAndFillCompleteCrsMatrix( matrix, *exporter, domainmap, rangemap );
    // defaults to "ADD" combine mode (reverseMode=false in Tpetra_CrsMatrix_def.hpp)
  
    objects.release( handle.matrix );
    objects.set( handle.matrix, completed_matrix, out() );
  }
  
  void matrix_norm() /* compute frobenius norm
     
       -> broadcast HANDLE handle.matrix
      <-  gather SCALAR norm
  */{
  
    struct { handle_t matrix; } handle;
    bcast( &handle );
    Teuchos::RCP<crsmatrix_t> matrix = objects.get<crsmatrix_t>( handle.matrix, out() );
  
    scalar_t norm = matrix->getFrobeniusNorm();
  
    gather( &norm );
  }
  
  void matrix_apply() /* matrix vector multiplication
     
       -> broadcast HANDLE handle.{matrix,out,vector}
  */{
  
    struct { handle_t matrix, rhs, lhs; } handle;
    bcast( &handle );
  
    Teuchos::RCP<crsmatrix_t> matrix = objects.get<crsmatrix_t>( handle.matrix, out() );
    Teuchos::RCP<vector_t> rhs = objects.get<vector_t>( handle.rhs, out() );
    Teuchos::RCP<vector_t> lhs = objects.get<vector_t>( handle.lhs, out() );
  
    matrix->apply( *rhs, *lhs );
  }
  
  void vector_as_setzero_operator() /* create setzero from nan values
     
       -> broadcast HANDLE handle.{setzero,vector}
  */{
  
    struct { handle_t setzero, vector; } handle;
    bcast( &handle );
  
    Teuchos::RCP<const vector_t> vector = objects.get<const vector_t>( handle.vector, out() );
    Teuchos::RCP<vector_t> selection = Teuchos::rcp( new vector_t( vector->getMap() ) );

    Teuchos::ArrayRCP<const scalar_t> _vector = vector->getData();
    Teuchos::ArrayRCP<scalar_t> _selection = selection->getDataNonConst();
    for ( int i = 0; i < _vector.size(); i++ ) {
      _selection[i] = std::isnan( _vector[i] );
      out() << "i=" << i << " selection=" << _selection[i] << std::endl;
    }
    Teuchos::RCP<SetZero> setzero = Teuchos::rcp( new SetZero(selection) );
    objects.set( handle.setzero, setzero, out() );
  }

  void precon_new() /* create new preconditioner
     
       -> broadcast HANDLE handle.{precon,matrix,precontype,preconparams}
  */{
  
    struct { handle_t precon, matrix, precontype, preconparams; } handle;
    bcast( &handle );
  
    Teuchos::RCP<const crsmatrix_t> matrix = objects.get<const crsmatrix_t>( handle.matrix, out() );
    Teuchos::RCP<const params_t> preconparams = objects.get<const params_t>( handle.preconparams, out() );
   
    preconfactory_t factory;
    Teuchos::RCP<precon_t> precon = factory.create( supportedPreconNames[handle.precontype], matrix );
  
    precon->setParameters( *preconparams );
    precon->initialize();
    precon->compute();
  
    magnitude_t condest = precon->computeCondEst( Ifpack2::Cheap );
    out() << "Ifpack2 preconditioner's estimated condition number: " << condest << std::endl;
  
    objects.set( handle.precon, precon, out() );
  }
  
  void export_new() /* create new exporter
     
       -> broadcast HANDLE handle.{exporter,srcmap,dstmap}
  */{
  
    struct { handle_t exporter, srcmap, dstmap; } handle;
    bcast( &handle );
  
    Teuchos::RCP<const map_t> srcmap = objects.get<const map_t>( handle.srcmap, out() );
    Teuchos::RCP<const map_t> dstmap = objects.get<const map_t>( handle.dstmap, out() );
  
    Teuchos::RCP<export_t> exporter = Teuchos::rcp( new export_t( srcmap, dstmap ) );
  
    objects.set( handle.exporter, exporter, out() );
  }

  void linearproblem_new() /* create new linear problem
     
       -> broadcast HANDLE handle.{linprob,matrix,lhs0}
  */{
  
    struct { handle_t linprob, matrix, lhs, rhs; } handle;
    bcast( &handle );
  
    Teuchos::RCP<const operator_t> matrix = objects.get<const operator_t>( handle.matrix, out() );
    Teuchos::RCP<vector_t> lhs = objects.get<vector_t>( handle.lhs, out() );
    Teuchos::RCP<const vector_t> rhs = objects.get<const vector_t>( handle.rhs, out() );

    Teuchos::RCP<linearproblem_t> linprob = Teuchos::rcp( new linearproblem_t( matrix, lhs, rhs ) );
  
    objects.set( handle.linprob, linprob, out() );
  }

  void linearproblem_set_hermitian() /* tell that operator is hermitian
     
       -> broadcast HANDLE handle.linprob
  */{
  
    struct { handle_t linprob; } handle;
    bcast( &handle );
  
    Teuchos::RCP<linearproblem_t> linprob = objects.get<linearproblem_t>( handle.linprob, out() );
    linprob->setHermitian();
  }

  void linearproblem_set_precon() /* set left preconditioner
     
       -> broadcast HANDLE handle.{linprob,prec}
       -> broadcast BOOL right
  */{
  
    struct { handle_t linprob, precon; } handle;
    bcast( &handle );

    bool_t right;
    bcast( &right );
  
    Teuchos::RCP<linearproblem_t> linprob = objects.get<linearproblem_t>( handle.linprob, out() );
    Teuchos::RCP<const operator_t> precon = objects.get<const operator_t>( handle.precon, out() );
    ((*linprob).*( right ? &linearproblem_t::setRightPrec : &linearproblem_t::setLeftPrec ))( precon );
  }

  void linearproblem_solve() /* solve system
     
       -> broadcast HANDLE handle.{linprob,solverparams,solvertype}
  */{
  
    struct { handle_t linprob, solverparams, solvertype; } handle;
    bcast( &handle );
  
    Teuchos::RCP<linearproblem_t> linprob = objects.get<linearproblem_t>( handle.linprob, out() );
    Teuchos::RCP<params_t> solverparams = objects.get<params_t>( handle.solverparams, out() );

    solverfactory_t factory;
    Teuchos::RCP<solvermanager_t> solver = factory.create( factory.supportedSolverNames()[handle.solvertype], solverparams );
  
    // called on the linear problem, before they can solve it.
    linprob->setProblem();
  
    // Tell the solver what problem you want to solve.
    solver->setProblem( linprob );
  
    // Attempt to solve the linear system.  result == Belos::Converged
    // means that it was solved to the desired tolerance.  This call
    // overwrites X with the computed approximate solution.
    Belos::ReturnType result = solver->solve();

    Teuchos::RCP<const crsmatrix_t> crsmatrix = Teuchos::rcp_dynamic_cast<const crsmatrix_t>( linprob->getOperator() );
    if ( ! crsmatrix.is_null() ) {
      Teuchos::RCP<const rowgraph_t> graph = crsmatrix->getGraph();
      Teuchos::RCP<multivector_t> mlhs = linprob->getLHS();
      for ( int ivec = 0; ivec < mlhs->getNumVectors(); ivec++ ) {
        Teuchos::ArrayRCP<scalar_t> _lhs = mlhs->getDataNonConst( ivec );
        for ( int irow = 0; irow < _lhs.size(); irow++ ) {
          if ( graph->getNumEntriesInLocalRow( irow ) == 0 ) {
            out() << "row " << irow << " is empty" << std::endl;
            _lhs[irow] = NAN;
          }
        }
      }
    }
  
    // Ask the solver how many iterations the last solve() took.
    const int numIters = solver->getNumIters();
  
    out() << "solver finished in " << numIters << " iterations with result " << result << std::endl;
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
  ObjectArray objects;

};

int main( int argc, char *argv[] ) {

  const LibMatrix::funcptr FTABLE[] = { FUNCS };
  const size_t NFUNCS = sizeof(FTABLE) / sizeof(LibMatrix::funcptr);
  const std::string _funcnames[] = { FUNCNAMES };
  const std::vector<std::string> funcnames( _funcnames, _funcnames + NFUNCS );

  if ( argc == 2 && std::strcmp( argv[1], "info" ) == 0 ) {
    print_all( "functions", funcnames );
    print_all( "solvers", solverfactory_t().supportedSolverNames() );
    print_all( "precons", supportedPreconNames );
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
    LibMatrix intercomm( argv[0] );
    token_t c;
    for ( ;; ) {
      intercomm.bcast( &c );
      if ( c >= NFUNCS ) {
        intercomm.out() << "quit" << std::endl;
        break;
      }
      intercomm.out() << "enter " << funcnames[c] << std::endl;
      try {
        (intercomm.*FTABLE[c])();
      }
      catch ( const char *s ) {
        std::cerr << "error in " << funcnames[c] << ": " << s << std::endl;
        intercomm.abort();
        break;
      }
      intercomm.out() << "leave " << funcnames[c] << std::endl;
    }
    intercomm.out() << "EXIT" << std::endl;
  }
  else {
    std::cout << "syntax: " << argv[0] << " info|eventloop" << std::endl;
    return 1;
  }
  return 0;
}


// vim:foldmethod=syntax
