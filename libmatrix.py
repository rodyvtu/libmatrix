import numpy, subprocess
from mpi import InterComm

exe, pyext = __file__.rsplit( '.', 1 )
exe += '.mpi'
info, dummy = subprocess.Popen( exe, stdout=subprocess.PIPE ).communicate()

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
  name = f.func_name
  def wrapped( self ):
    try:
      value = self.__dict__[ name ]
    except KeyError:
      value = f( self )
      self.__dict__[ name ] = value
    return value
  return property( wrapped )

def where( array ):
  return numpy.nonzero( array )[0]

def first( array ):
  return where( array )[0]

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

  @bcast_token
  def params_update( self, handle, xml ):
    self.bcast( handle, handle_t )
    self.bcast( len(xml), size_t )
    self.bcast( xml, char_t )

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
  def vector_copy( self, orig_handle ):
    copy_handle = self.claim_handle()
    self.bcast( [ copy_handle, orig_handle ], handle_t )
    return copy_handle

  @bcast_token
  def vector_fill( self, vec_handle, value ):
    self.bcast( vec_handle, handle_t )
    self.bcast( value, scalar_t )

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
  def vector_toarray( self, vec_handle ):
    self.bcast( vec_handle, handle_t )
    nitems_perproc = self.gather( size_t )
    indices = self.gatherv( nitems_perproc, global_t )
    values = self.gatherv( nitems_perproc, scalar_t )
    nitems = sum( nitems_perproc )
    array = numpy.empty( nitems, dtype=scalar_t )
    map( array.__setitem__, indices, values )
    return array

  @bcast_token
  def vector_norm( self, handle ):
    self.bcast( handle, handle_t )
    return self.gather_equal( scalar_t )

  @bcast_token
  def vector_sum( self, handle ):
    self.bcast( handle, handle_t )
    return self.gather( scalar_t ).sum()

  @bcast_token
  def vector_dot( self, handle1, handle2 ):
    self.bcast( [ handle1, handle2 ], handle_t )
    return self.gather_equal( scalar_t )

  @bcast_token
  def vector_complete( self, builder_handle, export_handle ):
    vector_handle = self.claim_handle()
    self.bcast( [ vector_handle, builder_handle, export_handle ], handle_t )
    return vector_handle

  @bcast_token
  def vector_or( self, self_handle, other_handle ):
    self.bcast( [ self_handle, other_handle ], handle_t )

  @bcast_token
  def vector_and( self, self_handle, other_handle ):
    self.bcast( [ self_handle, other_handle ], handle_t )

  @bcast_token
  def vector_imul( self, self_handle, other_handle ):
    self.bcast( [ self_handle, other_handle ], handle_t )

  @bcast_token
  def vector_update( self, self_handle, other_handle, alpha, beta ):
    self.bcast( [ self_handle, other_handle ], handle_t )
    self.bcast( [ alpha, beta ], scalar_t )

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
  def matrix_add( self, mat1_handle, mat2_handle, alpha, beta ):
    sum_handle = self.claim_handle()
    self.bcast( [ sum_handle, mat1_handle, mat2_handle ], handle_t )
    self.bcast( [ alpha, beta ], scalar_t )
    return sum_handle

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
  def matrix_complete( self, builder_handle, export_handle ):
    matrix_handle = self.claim_handle()
    self.bcast( [ matrix_handle, builder_handle, export_handle ], handle_t )
    return matrix_handle

  @bcast_token
  def matrix_constrained( self, matrix_handle, vector_handle ):
    conmat_handle = self.claim_handle()
    self.bcast( [ conmat_handle, matrix_handle, vector_handle ], handle_t )
    return conmat_handle

  @bcast_token
  def matrix_norm( self, handle ):
    self.bcast( handle, handle_t )
    return self.gather_equal( scalar_t )

  @bcast_token
  def operator_apply( self, operator_handle, vec_handle, out_handle ):
    self.bcast( [ operator_handle, vec_handle, out_handle ], handle_t )

  @bcast_token
  def matrix_toarray( self, matrix_handle ):
    self.bcast( matrix_handle, handle_t )
    ncols_perproc = self.gather( size_t )
    nrows_perproc = self.gather( size_t )
    irows = self.gatherv( nrows_perproc, global_t )
    nitems = self.gatherv( nrows_perproc, size_t )
    nentries = map( sum, nitems )
    icols = self.gatherv( nentries, global_t )
    values = self.gatherv( nentries, scalar_t )
    shape = sum( nrows_perproc ), sum( ncols_perproc )
    array = numpy.zeros( shape, dtype=scalar_t )
    for _nitems, _irows, _icols, _values in zip( nitems, irows, icols, values ):
      for i, n, d in zip( _irows, _nitems, numpy.cumsum( _nitems ) ):
        s = slice(d-n,d)
        j = _icols[s]
        v = _values[s]
        array[i,j] = v
    return array

  @bcast_token
  def vector_nan_from_supp( self, vector_handle, matrix_handle ):
    self.bcast( [ vector_handle, matrix_handle ], handle_t )

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
  def linearproblem_new( self, matrix_handle, lhs_handle, rhs_handle ):
    linprob_handle = self.claim_handle()
    self.bcast( [ linprob_handle, matrix_handle, lhs_handle, rhs_handle ], handle_t )
    return linprob_handle

  @bcast_token
  def linearproblem_set_hermitian( self, linprob_handle ):
    self.bcast( linprob_handle, handle_t )

  @bcast_token
  def linearproblem_set_precon( self, linprob_handle, precon_handle, right ):
    self.bcast( [ linprob_handle, precon_handle ], handle_t )
    self.bcast( right, bool_t )

  @bcast_token
  def linearproblem_solve( self, linprob_handle, solverparams_handle, solvername_handle ):
    self.bcast( [ linprob_handle, solverparams_handle, solvername_handle ], handle_t )

  @bcast_token
  def set_verbosity( self, level ):
    self.bcast( level, number_t )

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

  #<ParameterList name="ANONYMOUS">
  #  <Parameter docString="" id="0" isDefault="false" isUsed="true" name="foo" type="double" value="1.00000000000000000e+00"/>
  #  <Parameter docString="" id="1" isDefault="false" isUsed="true" name="bar" type="int" value="2"/>
  #  <Validators/>
  #</ParameterList>

  @staticmethod
  def toxml( items ):
    s = '<ParameterList>\n'
    for key, value in items.items():
      if isinstance( value, int ):
        dtype = 'int'
      elif isinstance( value, float ):
        dtype = 'double'
      else:
        raise Exception, 'invalid value %r' % value
      s += '<Parameter isUsed="true" name="%s" type="%s" value="%s"/>\n' % ( key, dtype, value )
    s += '</ParameterList>'
    return s

  def cprint ( self ):
    self.comm.params_print( self.handle )

  def __setitem__( self, key, value ):
    assert isinstance( key, str ), 'Expected first argument to be a string'
    self.items[ key ] = value
    xml = self.toxml({ key: value })
    self.comm.params_update( self.handle, xml )

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
      used = used.copy()
    else:
      indices = used
      assert len(indices) == comm.nprocs
      assert size is not None
      used = numpy.zeros( [ comm.nprocs, size ], dtype=bool )
      for iproc, idx in enumerate( indices ):
        used[ iproc, idx ] = True

    self.used = used
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
    return Export( self, self.ownedmap )


class Vector( Object ):

  def __init__( self, map, handle=None ):
    assert isinstance( map, Map )
    if handle is None:
      handle = map.comm.vector_new( map.handle )
    self.map = map
    self.shape = map.size,
    Object.__init__( self, map.comm, handle )

  def toarray( self ):
    assert self.map.is1to1
    array = self.comm.vector_toarray( self.handle )
    assert array.shape == self.shape
    return array

  def norm( self ):
    return self.comm.vector_norm( self.handle )

  def sum( self ):
    return self.comm.vector_sum( self.handle )

  def dot( self, other ):
    assert isinstance( other, Vector )
    assert self.shape == other.shape
    return self.comm.vector_dot( self.handle, other.handle )

  def fill( self, value ):
    self.comm.vector_fill( self.handle, value )

  def copy( self ):
    handle = self.comm.vector_copy( self.handle )
    return Vector( self.map, handle )

  def __neg__( self ):
    neg = Vector( self.map )
    self.comm.vector_update( neg.handle, self.handle, -1, 0 )
    return neg

  def __or__( self, other ):
    return self.copy().__ior__( other )

  def __ior__( self, other ):
    other = self.asme( other )
    self.comm.vector_or( self.handle, other.handle )
    return self

  def __and__( self, other ):
    return self.copy().__iand__( other )

  def __iand__( self, other ):
    other = self.asme( other )
    self.comm.vector_and( self.handle, other.handle )
    return self

  def __sub__( self, other ):
    return self.copy().__isub__( other )

  def __rsub__( self, other ):
    if not other:
      return -self
    return self.asme( other, copy=True ).__isub__( self )
    
  def __isub__( self, other ):
    other = self.asme( other )
    self.comm.vector_update( self.handle, other.handle, -1, 1 )
    return self

  def __add__( self, other ):
    return self.copy().__iadd__( other )

  __radd__ = __add__

  def __iadd__( self, other ):
    other = self.asme( other )
    self.comm.vector_update( self.handle, other.handle, 1, 1 )
    return self

  def __mul__( self, other ):
    return self.copy().__imul__( other )

  __rmul__ = __mul__

  def __imul__( self, other ):
    other = self.asme( other )
    self.comm.vector_imul( self.handle, other.handle )
    return self

  def asme( self, other, copy=False ):
    if isinstance( other, (int,float,numpy.ndarray) ):
      value = other
      other = Vector( self.map )
      if value:
        other.fill( value )
    elif copy:
      other = other.copy()
    assert isinstance( other, Vector )
    assert other.map == self.map
    return other

  def nan_from_supp( self, matrix ):
    self.comm.vector_nan_from_supp( self.handle, matrix.handle )

class VectorBuilder( Object ):

  def __init__( self, init ):
    if isinstance( init, Map ):
      self.shape = init.size,
      self.map = init
      myhandle = init.comm.vector_new( self.map.handle )
    elif isinstance( init, Vector ):
      self.shape = init.shape
      self.map = init.map
      myhandle = init.comm.vector_copy( init.handle )
    else:
      raise Exception, 'cannot construct vector from %r' % init
    Object.__init__( self, init.comm, myhandle )

  def add( self, rank, idx, data ):
    idx, = idx
    self.comm.vector_add_block( self.handle, rank, idx, data )

  def add_global( self, idx, data ):
    idx, = idx
    local = self.map.global2local[:,idx]
    rank = first( ( local != -1 ).all( axis=1 ) )
    self.add( rank, [local[rank]], data )

  def complete( self, export=None ):
    if export is None:
      export = self.map.export
    assert isinstance( export, Export )
    assert self.map == export.srcmap
    handle = self.comm.vector_complete( self.handle, export.handle )
    return Vector( export.dstmap, handle )


class Operator( Object ):

  def __init__( self, handle, domainmap, rangemap ):
    assert domainmap.comm == rangemap.comm
    assert isinstance( domainmap, Map )
    assert isinstance( rangemap, Map )
    self.domainmap = domainmap
    self.rangemap = rangemap
    self.shape = rangemap.size, domainmap.size
    Object.__init__( self, domainmap.comm, handle )

  def apply( self, vec ):
    assert isinstance( vec, Vector )
    assert vec.map == self.domainmap
    out = Vector( self.rangemap )
    self.comm.operator_apply( self.handle, vec.handle, out.handle )
    return out

  def toarray( self ):

    array = numpy.empty( self.shape, dtype=float )
    for i in range( self.shape[1] ):
      e = Vector( self.domainmap )
      e.add_global( [[i]], [1] )
      array[:,i] = self.apply( e ).toarray()
    return array

  def constrained( self, selection ):
    handle = self.comm.matrix_constrained( self.handle, selection.handle )
    return Operator( handle, self.domainmap, self.rangemap )

  def linearproblem( self, rhs=0, lhs=None, constrain=None ):
    if constrain:
      assert isinstance( constrain, Vector )
      assert constrain.map == self.domainmap
      matrix = self.constrained( constrain )
      rhs = constrain | ( rhs - self.apply( constrain | 0 ) )
    else:
      if not rhs:
        return Vector( self.domainmap )
      matrix = self
    return LinearProblem( matrix, rhs, lhs )

  def solve( self, rhs=0, lhs=None, constrain=None, precon=None, name=None, symmetric=False, params=None ):
    linprob = self.linearproblem( rhs, lhs, constrain )
    if symmetric:
      linprob.set_hermitian()
    if precon:
      linprob.set_precon( precon )
    if not name:
      name = 'CG' if symmetric else 'GMRES'
    return linprob.solve( name, params )


class Precon( Operator ):

  def __init__( self, matrix, precontype, preconparams=None ):
    assert isinstance( matrix, Operator )
    comm = matrix.comm
    if not preconparams:
      preconparams = ParameterList( comm )
    else:
      assert isinstance( preconparams, ParameterList )
      assert preconparams.comm is comm
    myhandle = comm.precon_new( matrix.handle, _precons.index(precontype), preconparams.handle )
    Operator.__init__( self, myhandle, matrix.rangemap, matrix.domainmap )


class Matrix( Operator ):

  def __add__( self, other ):
    assert isinstance( self, Matrix )
    assert isinstance( other, Matrix )
    assert self.shape == other.shape
    handle = self.comm.matrix_add( self.handle, other.handle, 1, 1 )
    return Matrix( handle, self.domainmap, self.rangemap )

  def __sub__( self, other ):
    assert isinstance( self, Matrix )
    assert isinstance( other, Matrix )
    assert self.shape == other.shape
    handle = self.comm.matrix_add( self.handle, other.handle, 1, -1 )
    return Operator( handle, self.domainmap, self.rangemap )

  def norm( self ):
    return self.comm.matrix_norm( self.handle )

  def toarray( self ):
    array = self.comm.matrix_toarray( self.handle )
    assert array.shape == self.shape
    return array


class MatrixBuilder( Object ):

  def __init__( self, init ):
    if isinstance( init, Graph ):
      self.rowmap = init.rowmap
      self.colmap = init.colmap
      comm = init.comm
      matrix_handle = comm.matrix_new_static( init.handle )
    else:
      if isinstance( init, Map ):
        self.rowmap = self.colmap = init
        comm = self.rowmap.comm
      else:
        self.rowmap, self.colmap = init
        assert isinstance( self.rowmap, Map )
        assert isinstance( self.colmap, Map )
        comm = self.rowmap.comm
        assert self.colmap.comm is comm
      matrix_handle = comm.matrix_new_dynamic( self.rowmap.handle, self.colmap.handle )
    self.shape = self.rowmap.size, self.colmap.size
    Object.__init__( self, comm, matrix_handle )

  def add( self, rank, idx, data ):
    rowidx, colidx = idx
    self.comm.matrix_add_block( self.handle, rank, rowidx, colidx, data )

  def add_global( self, idx, data ):
    rowidx, colidx = idx
    rowlocal = self.rowmap.global2local[:,rowidx]
    collocal = self.colmap.global2local[:,colidx]
    rank = first( (rowlocal!=-1).all(axis=1) & (collocal!=-1).all(axis=1) )
    self.add( rank, ( rowlocal[rank], collocal[rank] ), data )

  def complete( self, export=None ):
    if export is None:
      export = self.rowmap.export
    assert isinstance( export, Export )
    assert self.rowmap == export.srcmap
    handle = self.comm.matrix_complete( self.handle, export.handle )
    return Matrix( handle, export.dstmap, export.dstmap )


class LinearProblem( Object ):

  def __init__( self, matrix, rhs, lhs=None ):
    assert isinstance( matrix, Operator )
    ndofs = matrix.shape[0]
    assert matrix.shape[1] == ndofs
    comm = matrix.comm
    if not lhs:
      lhs = Vector( matrix.domainmap )
    assert isinstance( lhs, Vector )
    assert lhs.map == matrix.domainmap
    assert lhs.shape[0] == ndofs
    assert lhs.comm == comm
    assert isinstance( rhs, Vector )
    assert rhs.map == matrix.rangemap
    assert rhs.shape[0] == ndofs
    assert rhs.comm == comm
    myhandle = comm.linearproblem_new( matrix.handle, lhs.handle, rhs.handle )
    Object.__init__( self, comm, myhandle )
    self.matrix = matrix
    self.lhs = lhs
    self.rhs = rhs

  def set_hermitian( self ):
    self.comm.linearproblem_set_hermitian( self.handle )

  def set_precon( self, precon, right=False ):
    assert isinstance( precon, Operator )
    assert precon.shape == self.matrix.shape
    self.comm.linearproblem_set_precon( self.handle, precon.handle, right )

  def solve( self, name='GMRES', params=None ):
    if not params:
      params = ParameterList( self.comm )
    assert isinstance( params, ParameterList )
    assert params.comm is self.comm
    solver_handle = _solvers.index( name )
    self.comm.linearproblem_solve( self.handle, params.handle, solver_handle )
    return self.lhs

  def res( self ):
    return ( self.rhs - self.matrix.apply( self.lhs ) ).norm()


class Export( Object ):

  def __init__( self, srcmap, dstmap ):
    assert isinstance( srcmap, Map )
    assert isinstance( dstmap, Map )
    comm = srcmap.comm
    assert dstmap.comm is comm
    self.srcmap = srcmap
    self.dstmap = dstmap
    Object.__init__( self, comm, comm.export_new( srcmap.handle, dstmap.handle ) )


class Graph( Object ):

  def __init__( self, rowmap, colmap, rows ):
    assert isinstance( rowmap, Map )
    assert isinstance( colmap, Map )
    comm = rowmap.comm
    assert colmap.comm is comm
    self.rowmap = rowmap
    self.colmap = colmap
    Object.__init__( self, comm, comm.graph_new( rowmap.handle, colmap.handle, rows ) )


class ScalarBuilder( object ):

  def __init__( self ):
    self.value = 0.

  def add_global( self, index, value ):
    assert not index
    self.value += value

  def complete( self ):
    return self.value

def ArrayBuilder( shape ):
  if len( shape ) == 2:
    return MatrixBuilder( shape )
  if len( shape ) == 1:
    return VectorBuilder( shape[0] )
  assert not shape
  return ScalarBuilder()


# vim:shiftwidth=2:foldmethod=indent:foldnestmax=2
