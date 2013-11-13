import numpy, os
from mpi import InterComm, dtypes


_info = dict( line.rstrip().split( ': ', 1 ) for line in os.popen( './libmatrix.mpi info' ) )

def bcast_token( func, names=_info.pop('token')[5:-1].split(', ') ):
  token = chr( names.index( func.func_name ) )
  def wrapped( self, *args, **kwargs ):
    assert self.isconnected(), 'connection is closed'
    self.bcast( token, TOKEN )
    return func( self, *args, **kwargs )
  return wrapped

LOCAL  = dtypes[ _info.pop('local')  ]
GLOBAL = dtypes[ _info.pop('global') ]
HANDLE = dtypes[ _info.pop('handle') ]
SIZE   = dtypes[ _info.pop('size')   ]
SCALAR = dtypes[ _info.pop('scalar') ]
TOKEN  = dtypes[ 'char' ]

assert not _info
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
    self.bcast( handle, HANDLE )

  @bcast_token
  def map_new( self, globs ):
    map_handle = self.claim_handle()
    lengths = map( len, globs )
    size = sum( lengths ) # TODO check meaning of size in map constructor
    self.bcast( map_handle, HANDLE )
    self.bcast( size, SIZE )
    self.scatter( lengths, SIZE )
    self.scatterv( globs, GLOBAL )
    return map_handle

  @bcast_token
  def vector_new( self, map_handle ):
    vec_handle = self.claim_handle()
    self.bcast( [ vec_handle, map_handle ], HANDLE )
    return vec_handle

  @bcast_token
  def vector_add_block( self, handle, rank, idx, data ):
    n = len(idx)
    assert len(data) == n
    self.bcast( rank, SIZE )
    self.send( rank, handle, HANDLE )
    self.send( rank, n, SIZE )
    self.send( rank, idx, GLOBAL )
    self.send( rank, data, SCALAR )

  @bcast_token
  def vector_getdata( self, vec_handle, size, globs ):
    self.bcast( vec_handle, HANDLE )
    array = numpy.zeros( size ) # TODO fix length
    lengths = map( len, globs )
    local_arrays = self.gatherv( lengths, SCALAR )
    for idx, local_array in zip( globs, local_arrays ):
      array[idx] += local_array
    return array

  @bcast_token
  def vector_norm( self, handle ):
    self.bcast( handle, HANDLE )
    return self.gather_equal( SCALAR )

  @bcast_token
  def vector_dot( self, handle1, handle2 ):
    self.bcast( [ handle1, handle2 ], HANDLE )
    return self.gather_equal( SCALAR )

  @bcast_token
  def matrix_new( self, graph_handle ):
    matrix_handle = self.claim_handle()
    self.bcast( [ matrix_handle, graph_handle ], HANDLE )
    return matrix_handle

  @bcast_token
  def matrix_add_block( self, handle, rank, rowidx, colidx, data ):
    data = numpy.asarray(data)
    shape = len(rowidx), len(colidx)
    assert data.shape == shape
    self.bcast( rank, SIZE )
    self.send( rank, handle, HANDLE )
    self.send( rank, shape, SIZE )
    print rowidx, colidx
    self.send( rank, rowidx, GLOBAL )
    self.send( rank, colidx, GLOBAL )
    self.send( rank, data.ravel(), SCALAR )

  @bcast_token
  def matrix_fillcomplete( self, handle ):
    self.bcast( handle, HANDLE )

  @bcast_token
  def matrix_norm( self, handle ):
    self.bcast( handle, HANDLE )
    return self.gather_equal( SCALAR )

  @bcast_token
  def matrix_apply( self, matrix_handle, vec_handle, out_handle ):
    self.bcast( [ matrix_handle, vec_handle, out_handle ], HANDLE )

  @bcast_token
  def graph_new( self, rowmap_handle, colmap_handle, dommap_handle, rngmap_handle, rows ):
    graph_handle = self.claim_handle()
    self.bcast( [ graph_handle, rowmap_handle, colmap_handle, dommap_handle, rngmap_handle ], HANDLE )
    offsets = [ numpy.cumsum( [0] + map( len, row ) ) for row in rows ]
    self.scatterv( offsets, SIZE )
    cols = [ numpy.concatenate( row ) for row in rows ]
    self.scatterv( cols, LOCAL )
    return graph_handle

  @bcast_token
  def matrix_solve( self, matrix_handle, rhs_handle, lhs_handle ):
    self.bcast( [ matrix_handle, rhs_handle, lhs_handle ], HANDLE )

  def __del__( self ):
    self.bcast( '\xff', TOKEN )
    InterComm.__del__( self )


#--- user facing objects ---


class Map( object ):

  def __init__( self, comm, globs ):
    self.comm = comm
    assert len(globs) == comm.size
    self.globs = [ numpy.asarray(glob,dtype=int) for glob in globs ]
    self.handle = comm.map_new( globs )

  def __del__( self ):
    self.comm.release( self.handle )


class Vector( object ):

  def __init__( self, comm, size, mp ):
    self.comm = comm
    self.size = size
    assert isinstance( mp, Map )
    self.mp = mp
    self.handle = comm.vector_new( mp.handle )

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

  def __del__( self ):
    self.comm.release( self.handle )


class Matrix( object ):

  def __init__( self, comm, shape, graph ):
    self.comm = comm
    self.shape = shape
    assert isinstance( graph, Graph )
    self.graph = graph
    self.handle = comm.matrix_new( graph.handle )

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

  def __del__( self ):
    self.comm.release( self.handle )


class Graph( object ):

  def __init__( self, comm, rowmap, columnmap, domainmap, rangemap, rows ):
    self.comm = comm
    assert isinstance( rowmap, Map )
    assert isinstance( columnmap, Map )
    assert isinstance( domainmap, Map )
    assert isinstance( rangemap, Map )
    self.rowmap = rowmap
    self.columnmap = columnmap
    self.domainmap = domainmap
    self.rangemap = rangemap
    self.handle = comm.graph_new( rowmap.handle, columnmap.handle, domainmap.handle, rangemap.handle, rows )

  def __del__( self ):
    self.comm.release( self.handle )
