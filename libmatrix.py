import numpy, os
from mpi import InterComm


_info  = dict( line.rstrip().split( ': ', 1 ) for line in os.popen( './libmatrix.mpi info' ) )
_names = _info.pop('tokens')[1:-1].split(', ')

local_t  = numpy.dtype( _info.pop( 'local_t'  ) )
global_t = numpy.dtype( _info.pop( 'global_t' ) )
handle_t = numpy.dtype( _info.pop( 'handle_t' ) )
size_t   = numpy.dtype( _info.pop( 'size_t'   ) )
scalar_t = numpy.dtype( _info.pop( 'scalar_t' ) )
number_t = numpy.dtype( _info.pop( 'number_t' ) )
token_t  = numpy.dtype( _info.pop( 'token_t'  ) )
char_t   = numpy.dtype( 'str' )

def bcast_token_template( template_var_arg ):
  def bcast_token( func ):
    int_token = _names.index( func.func_name + '<number_t>' )
    float_token = _names.index( func.func_name + '<scalar_t>' )
    def wrapped( self, *args ):
      assert self.isconnected(), 'connection is closed'
      template_arg = args[template_var_arg]
      if isinstance( template_arg, int ):
        token = int_token
        dtype = number_t
      elif isinstance( template_arg, float ):
        token = float_token
        dtype = scalar_t
      else:
        raise Exception, 'invalid argument for function %r: %r' % ( func.func_name, template_arg )
      template_arg = numpy.asarray( template_arg, dtype=dtype )
      self.bcast( token, token_t )
      args = args[:template_var_arg] + (template_arg,) + args[template_var_arg+1:]
      return func( self, *args )
    return wrapped    
  return bcast_token 

def bcast_token( func ):
  token = _names.index( func.func_name )
  def wrapped( self, *args, **kwargs ):
    assert self.isconnected(), 'connection is closed'
    self.bcast( token, token_t )
    return func( self, *args, **kwargs )
  return wrapped

assert not _info, 'leftover info: %s' % _info
del _info


class LibMatrix( InterComm ):
  'interface to all libmatrix functions'

  def __init__( self, nprocs ):
    InterComm.__init__( self, 'libmatrix.mpi', args=['eventloop'], maxprocs=nprocs )
    assert self.size == nprocs
    self.nhandles = 0
    self.released = []

  def claim_handle( self ):
    if self.released:
      handle = self.released.pop()
    else:
      handle = self.nhandles
      self.nhandles += 1
    return handle

  def release_handle( self, handle ):
    assert handle < self.nhandles and handle not in self.released
    self.released.append( handle )

  @bcast_token
  def release( self, handle ):
    self.bcast( handle, handle_t )

  @bcast_token
  def map_new( self, globs ):
    map_handle = self.claim_handle()
    lengths = map( len, globs )
    size = sum( lengths ) # TODO check meaning of size in map constructor
    self.bcast( map_handle, handle_t )
    self.bcast( size, size_t )
    self.scatter( lengths, size_t )
    self.scatterv( globs, global_t )
    return map_handle

  @bcast_token
  def params_new( self ):
    params_handle = self.claim_handle()
    self.bcast( params_handle, handle_t )
    return params_handle

  @bcast_token_template( 2 )
  def params_set( self, handle, key, value ):
    self.bcast( handle, handle_t )
    self.bcast( len(key), size_t )
    self.bcast( key, char_t )
    self.bcast( value )

  @bcast_token
  def params_print( self, handle ):
    self.bcast( handle, handle_t )

  @bcast_token
  def vector_new( self, map_handle ):
    vec_handle = self.claim_handle()
    self.bcast( [ vec_handle, map_handle ], handle_t )
    return vec_handle

  @bcast_token
  def vector_add_block( self, handle, rank, idx, data ):
    n = len(idx)
    assert len(data) == n
    self.bcast( rank, size_t )
    self.send( rank, handle, handle_t )
    self.send( rank, n, size_t )
    self.send( rank, idx, global_t )
    self.send( rank, data, scalar_t )

  @bcast_token
  def vector_getdata( self, vec_handle, size, globs ):
    self.bcast( vec_handle, handle_t )
    array = numpy.zeros( size ) # TODO fix length
    lengths = map( len, globs )
    local_arrays = self.gatherv( lengths, scalar_t )
    for idx, local_array in zip( globs, local_arrays ):
      array[idx] += local_array
    return array

  @bcast_token
  def vector_norm( self, handle ):
    self.bcast( handle, handle_t )
    return self.gather_equal( scalar_t )

  @bcast_token
  def vector_dot( self, handle1, handle2 ):
    self.bcast( [ handle1, handle2 ], handle_t )
    return self.gather_equal( scalar_t )

  @bcast_token
  def matrix_new( self, graph_handle ):
    matrix_handle = self.claim_handle()
    self.bcast( [ matrix_handle, graph_handle ], handle_t )
    return matrix_handle

  @bcast_token
  def matrix_add_block( self, handle, rank, rowidx, colidx, data ):
    data = numpy.asarray(data)
    shape = len(rowidx), len(colidx)
    assert data.shape == shape
    self.bcast( rank, size_t )
    self.send( rank, handle, handle_t )
    self.send( rank, shape, size_t )
    print rowidx, colidx
    self.send( rank, rowidx, global_t )
    self.send( rank, colidx, global_t )
    self.send( rank, data.ravel(), scalar_t )

  @bcast_token
  def matrix_fillcomplete( self, handle ):
    self.bcast( handle, handle_t )

  @bcast_token
  def matrix_norm( self, handle ):
    self.bcast( handle, handle_t )
    return self.gather_equal( scalar_t )

  @bcast_token
  def matrix_apply( self, matrix_handle, vec_handle, out_handle ):
    self.bcast( [ matrix_handle, vec_handle, out_handle ], handle_t )

  @bcast_token
  def graph_new( self, rowmap_handle, colmap_handle, dommap_handle, rngmap_handle, rows ):
    graph_handle = self.claim_handle()
    self.bcast( [ graph_handle, rowmap_handle, colmap_handle, dommap_handle, rngmap_handle ], handle_t )
    offsets = [ numpy.cumsum( [0] + map( len, row ) ) for row in rows ]
    self.scatterv( offsets, size_t )
    cols = [ numpy.concatenate( row ) for row in rows ]
    self.scatterv( cols, local_t )
    return graph_handle

  @bcast_token
  def matrix_solve( self, matrix_handle, rhs_handle, lhs_handle ):
    self.bcast( [ matrix_handle, rhs_handle, lhs_handle ], handle_t )

  def __del__( self ):
    self.bcast( -1, token_t )
    InterComm.__del__( self )


#--- user facing objects ---


class Object( object ):

  def __init__( self, comm, handle ):
    self.comm = comm
    self.handle = handle

  def __del__ ( self ):
    self.comm.release( self.handle )


class ParameterList( Object ):

  def __init__( self, comm ):
    Object.__init__( self, comm, comm.params_new() )

  def cprint ( self ):
    self.comm.params_print( self.handle )

  def set ( self, key, value ):
    assert isinstance(key,str), 'Expected first argument to be a string'
    assert isinstance(value,float) or isinstance(value,int), 'Current implementation supports int and float only'
    self.comm.params_set( self.handle, key, value )


class Map( Object ):

  def __init__( self, comm, globs ):
    assert len(globs) == comm.size
    self.globs = [ numpy.asarray(glob,dtype=int) for glob in globs ]
    Object.__init__( self, comm, comm.map_new( globs ) )


class Vector( Object ):

  def __init__( self, comm, size, mp ):
    self.size = size
    assert isinstance( mp, Map )
    self.mp = mp
    Object.__init__( self, comm, comm.vector_new( mp.handle ) )

  def add( self, rank, idx, data ):
    self.comm.vector_add_block( self.handle, rank, idx, data )

  def toarray( self ):
    return self.comm.vector_getdata( self.handle, self.size, self.mp.globs )

  def norm( self ):
    return self.comm.vector_norm( self.handle )

  def dot( self, other ):
    assert isinstance( other, Vector )
    assert self.size == other.size
    return self.comm.vector_dot( self.handle, other.handle )


class Matrix( Object ):

  def __init__( self, comm, shape, graph ):
    self.shape = shape
    assert isinstance( graph, Graph )
    self.graph = graph
    Object.__init__( self, comm, comm.matrix_new( graph.handle ) )

  def add( self, rank, rowidx, colidx, data ):
    self.comm.matrix_add_block( self.handle, rank, rowidx, colidx, data )

  def complete( self ):
    self.comm.matrix_fillcomplete( self.handle )

  def norm( self ):
    return self.comm.matrix_norm( self.handle )

  def matvec( self, vec ):
    assert isinstance( vec, Vector )
    assert vec.mp == self.graph.domainmap
    assert self.shape[1] == vec.size
    out = Vector( self.comm, self.shape[0], self.graph.rangemap )
    self.comm.matrix_apply( self.handle, vec.handle, out.handle )
    return out

  def solve( self, rhs ):
    assert isinstance( rhs, Vector )
    assert self.shape[0] == rhs.size
    lhs = Vector( self.comm, self.shape[1], self.graph.domainmap )
    self.comm.matrix_solve( self.handle, rhs.handle, lhs.handle )
    return lhs


class Graph( Object ):

  def __init__( self, comm, rowmap, columnmap, domainmap, rangemap, rows ):
    assert isinstance( rowmap, Map )
    assert isinstance( columnmap, Map )
    assert isinstance( domainmap, Map )
    assert isinstance( rangemap, Map )
    self.rowmap = rowmap
    self.columnmap = columnmap
    self.domainmap = domainmap
    self.rangemap = rangemap
    Object.__init__( self, comm, comm.graph_new( rowmap.handle, columnmap.handle, domainmap.handle, rangemap.handle, rows ) )
