import numpy, subprocess
from mpi import InterComm

exe, pyext = __file__.rsplit( '.', 1 )
exe += '.mpi'
info, dummy = subprocess.Popen( [ exe, 'info' ], stdout=subprocess.PIPE ).communicate()

_info  = dict( line.split( ': ', 1 ) for line in info.splitlines() )
_functions = _info.pop('functions').split(', ')
_solvers = _info.pop('solvers').split(', ')
_precons = _info.pop('precons').split(', ')

local_t  = numpy.dtype( _info.pop( 'local_t'  ) )
global_t = numpy.dtype( _info.pop( 'global_t' ) )
handle_t = numpy.dtype( _info.pop( 'handle_t' ) )
size_t   = numpy.dtype( _info.pop( 'size_t'   ) )
scalar_t = numpy.dtype( _info.pop( 'scalar_t' ) )
number_t = numpy.dtype( _info.pop( 'number_t' ) )
bool_t   = numpy.dtype( _info.pop( 'bool_t'   ) )
token_t  = numpy.dtype( _info.pop( 'token_t'  ) )
char_t   = numpy.dtype( 'str' )

def cacheprop( f ):
  cache = []
  def wrapped( self ):
    if not cache:
      cache.append( f(self) )
    return cache[0]
  return property( wrapped )

def where( array ):
  return numpy.nonzero( array )[0]

def first( array ):
  return where( array )[0]

def bcast_token_template( template_var_arg ):
  def bcast_token( func ):
    int_token = _functions.index( func.func_name + '<number_t>' )
    float_token = _functions.index( func.func_name + '<scalar_t>' )
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
  token = _functions.index( func.func_name )
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
    InterComm.__init__( self, exe, args=['eventloop'], maxprocs=nprocs )
    assert self.nprocs == nprocs
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
  def map_new( self, local2global ):
    map_handle = self.claim_handle()
    lengths = map( len, local2global )
    size = sum( lengths ) # TODO check meaning of size in map constructor
    self.bcast( map_handle, handle_t )
    self.bcast( size, size_t )
    self.scatter( lengths, size_t )
    self.scatterv( local2global, global_t )
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
  def export_new( self, srcmap_handle, dstmap_handle ):
    export_handle = self.claim_handle()
    self.bcast( [ export_handle, srcmap_handle, dstmap_handle ], handle_t )
    return export_handle

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
    self.send( rank, idx, local_t )
    self.send( rank, data, scalar_t )

  @bcast_token
  def vector_getdata( self, vec_handle, size, local2global ):
    self.bcast( vec_handle, handle_t )
    array = numpy.zeros( size ) # TODO fix length
    lengths = map( len, local2global )
    local_arrays = self.gatherv( lengths, scalar_t )
    for idx, local_array in zip( local2global, local_arrays ):
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
  def vector_complete( self, vector_handle, export_handle ):
    self.bcast( [ vector_handle, export_handle ], handle_t )

  @bcast_token
  def matrix_new_static( self, graph_handle ):
    matrix_handle = self.claim_handle()
    self.bcast( [ matrix_handle, graph_handle ], handle_t )
    return matrix_handle

  @bcast_token
  def matrix_new_dynamic( self, rowmap_handle, colmap_handle ):
    matrix_handle = self.claim_handle()
    self.bcast( [ matrix_handle, rowmap_handle, colmap_handle ], handle_t )
    return matrix_handle

  @bcast_token
  def precon_new( self, matrix_handle, precontype_handle, preconparams_handle ):
    precon_handle = self.claim_handle()
    self.bcast( [ precon_handle, matrix_handle, precontype_handle, preconparams_handle ], handle_t )
    return precon_handle

  @bcast_token
  def matrix_add_block( self, handle, rank, rowidx, colidx, data ):
    data = numpy.asarray(data)
    shape = len(rowidx), len(colidx)
    assert data.shape == shape
    self.bcast( rank, size_t )
    self.send( rank, handle, handle_t )
    self.send( rank, shape, size_t )
    self.send( rank, rowidx, local_t )
    self.send( rank, colidx, local_t )
    self.send( rank, data.ravel(), scalar_t )

  @bcast_token
  def matrix_complete( self, matrix_handle, export_handle ):
    self.bcast( [ matrix_handle, export_handle ], handle_t )

  @bcast_token
  def matrix_norm( self, handle ):
    self.bcast( handle, handle_t )
    return self.gather_equal( scalar_t )

  @bcast_token
  def matrix_apply( self, matrix_handle, vec_handle, out_handle ):
    self.bcast( [ matrix_handle, vec_handle, out_handle ], handle_t )

  @bcast_token
  def graph_new( self, rowmap_handle, colmap_handle, rows ):
    graph_handle = self.claim_handle()
    self.bcast( [ graph_handle, rowmap_handle, colmap_handle ], handle_t )
    offsets = [ numpy.cumsum( [0] + map( len, row ) ) for row in rows ]
    self.scatterv( offsets, size_t )
    cols = [ numpy.concatenate( row ) for row in rows ]
    self.scatterv( cols, local_t )
    return graph_handle

  @bcast_token
  def matrix_solve( self, matrix_handle, precon_handle, rhs_handle, lhs_handle, solvertype_handle, symmetric, solverparams_handle ):
    self.bcast( [ matrix_handle, precon_handle, rhs_handle, lhs_handle, solvertype_handle, solverparams_handle ], handle_t )
    self.bcast( symmetric, bool_t )

  @bcast_token
  def toggle_stdout( self ):
    pass

  def __del__( self ):
    self.bcast( -1, token_t )
    InterComm.__del__( self )


#--- user facing objects ---


class Object( object ):

  def __init__( self, comm, handle ):
    self.comm = comm
    self.handle = handle

  def __del__ ( self ):
    if hasattr( self, 'comm' ):
      self.comm.release( self.handle )


class ParameterList( Object ):

  def __init__( self, comm ):
    self.items = {}
    Object.__init__( self, comm, comm.params_new() )

  def cprint ( self ):
    self.comm.params_print( self.handle )

  def __setitem__( self, key, value ):
    assert isinstance( key, str ), 'Expected first argument to be a string'
    self.comm.params_set( self.handle, key, value )
    self.items[ key ] = value

  def __str__( self ):
    return 'ParameterList( %s )' % ', '.join( '%s=%s' % item for item in self.items.items() )


class Map( Object ):

  def __init__( self, comm, used, size=None ):
    if isinstance( used, numpy.ndarray ) and used.dtype == bool:
      assert used.ndim == 2
      assert used.shape[0] == comm.nprocs
      if size is not None:
        assert used.shape[1] == size
      else:
        size = used.shape[1]
    else:
      indices = used
      assert len(indices) == comm.nprocs
      assert size is not None
      used = numpy.zeros( [ comm.nprocs, size ], dtype=bool )
      for iproc, idx in enumerate( indices ):
        used[ iproc, idx ] = True

    self.owned = self.__distribute( used )
    self.local2global = []
    self.is1to1 = True
    for x, y in zip( self.owned, used & ~self.owned ):
      x = where( x )
      y = where( y )
      if y.size:
        x = numpy.concatenate([ x, y ])
        self.is1to1 = False
      self.local2global.append( x )

    self.size = size
    Object.__init__( self, comm, comm.map_new( self.local2global ) )

  @staticmethod
  def __distribute( used ):
    owned = numpy.empty_like( used )
    touched = numpy.zeros_like( used[0] )
    for i in used.sum( axis=1 ).argsort():
      owned[i] = used[i] & ~touched
      touched |= used[i]
    assert touched.all()
    return owned

  @cacheprop
  def global2local( self ):
    global2local = numpy.empty( [ self.comm.nprocs, self.size ], dtype=local_t )
    global2local[:] = -1
    arange = numpy.arange( self.size, dtype=local_t )
    for g2l, l2g in zip( global2local, self.local2global ):
      g2l[l2g] = arange
    return global2local

  @cacheprop
  def ownedmap( self ):
    if self.is1to1:
      return self
    ownedmap = Map( self.comm, self.owned )
    assert ownedmap.is1to1
    return ownedmap

  @cacheprop
  def export( self ):
    return Export( self.comm, self, self.ownedmap )


class Vector( Object ):

  def __init__( self, comm, map ):
    self.size = map.size
    assert isinstance( map, Map )
    self.map = map
    Object.__init__( self, comm, comm.vector_new( map.handle ) )

  def add( self, rank, idx, data ):
    idx, = idx
    self.comm.vector_add_block( self.handle, rank, idx, data )

  def add_global( self, idx, data ):
    idx, = idx
    local = self.map.global2local[:,idx]
    rank = first( ( local != -1 ).all( axis=1 ) )
    self.add( rank, [local[rank]], data )

  def toarray( self ):
    return self.comm.vector_getdata( self.handle, self.size, self.map.local2global )

  def norm( self ):
    return self.comm.vector_norm( self.handle )

  def dot( self, other ):
    assert isinstance( other, Vector )
    assert self.size == other.size
    return self.comm.vector_dot( self.handle, other.handle )

  def complete( self, export ):
    assert isinstance( export, Export )
    assert export.srcmap == self.map
    self.comm.vector_complete( self.handle, export.handle )
    self.map = export.dstmap


class Operator( Object ):

  def __init__( self, comm, handle, shape ):
    self.shape = tuple(shape)
    Object.__init__( self, comm, handle )


class Precon( Operator ):

  def __init__( self, comm, matrix, precontype, preconparams=None ):
    assert isinstance( matrix, Operator )
    if not preconparams:
      preconparams = ParameterList( comm )
    assert isinstance( preconparams, ParameterList )
    myhandle = comm.precon_new( matrix.handle, _precons.index(precontype), preconparams.handle )
    Operator.__init__( self, comm, myhandle, reversed(matrix.shape) )


class Matrix( Operator ):

  def __init__( self, comm, init ):
    if isinstance( init, Graph ):
      self.rowmap = init.rowmap
      self.colmap = init.colmap
      matrix_handle = comm.matrix_new_static( init.handle )
    else:
      if isinstance( init, Map ):
        self.rowmap = self.colmap = init
      else:
        self.rowmap, self.colmap = init
        assert isinstance( self.rowmap, Map )
        assert isinstance( self.colmap, Map )
      matrix_handle = comm.matrix_new_dynamic( self.rowmap.handle, self.colmap.handle )
    self.shape = self.rowmap.size, self.colmap.size
    Operator.__init__( self, comm, matrix_handle, self.shape )

  def add( self, rank, idx, data ):
    rowidx, colidx = idx
    self.comm.matrix_add_block( self.handle, rank, rowidx, colidx, data )

  def add_global( self, idx, data ):
    rowidx, colidx = idx
    rowlocal = self.rowmap.global2local[:,rowidx]
    collocal = self.colmap.global2local[:,colidx]
    rank = first( ( (rowlocal!=-1) & (collocal!=-1) ).all( axis=1 ) )
    self.add( rank, ( rowlocal[rank], collocal[rank] ), data )

  def complete( self, exporter ):
    assert isinstance( exporter, Export )
    assert self.rowmap == exporter.srcmap
    self.comm.matrix_complete( self.handle, exporter.handle )
    self.domainmap = self.rangemap = exporter.dstmap

  def norm( self ):
    return self.comm.matrix_norm( self.handle )

  def matvec( self, vec ):
    assert isinstance( vec, Vector )
    assert vec.map == self.domainmap
    assert self.shape[1] == vec.size
    out = Vector( self.comm, self.rangemap )
    self.comm.matrix_apply( self.handle, vec.handle, out.handle )
    return out

  def solve( self, precon, rhs, name='GMRES', symmetric=False, params=None ):
    assert isinstance( precon, Operator )
    assert isinstance( rhs, Vector )
    if not params:
      params = ParameterList( self.comm )
    assert isinstance( params, ParameterList )
    assert self.shape[0] == rhs.size
    lhs = Vector( self.comm, self.domainmap )
    self.comm.matrix_solve( self.handle, precon.handle, rhs.handle, lhs.handle, _solvers.index( name ), symmetric, params.handle )
    return lhs


class Export( Object ):

  def __init__( self, comm, srcmap, dstmap ):
    assert isinstance( srcmap, Map )
    assert isinstance( dstmap, Map )
    self.srcmap = srcmap
    self.dstmap = dstmap
    Object.__init__( self, comm, comm.export_new( srcmap.handle, dstmap.handle ) )


class Graph( Object ):

  def __init__( self, comm, rowmap, colmap, rows ):
    assert isinstance( rowmap, Map )
    assert isinstance( colmap, Map )
    self.rowmap = rowmap
    self.colmap = colmap
    Object.__init__( self, comm, comm.graph_new( rowmap.handle, colmap.handle, rows ) )
