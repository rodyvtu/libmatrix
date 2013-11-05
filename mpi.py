import numpy
from mpi4py import MPI


dtypes = {
  'int32'   : ( numpy.int32,     MPI.INT    ),
  'int64'   : ( numpy.int64,     MPI.LONG   ),
  'float64' : ( numpy.float64,   MPI.DOUBLE ),
  'char'    : ( numpy.character, MPI.CHAR   ),
}


class InterComm( object ):
  'generic MPI communicator wrapper'

  def __init__( self, cmd, **kwargs ):
    self.__comm = MPI.COMM_SELF.Spawn( cmd, **kwargs )
    self.size = self.__comm.remote_size

  def isconnected( self ):
    return bool( self.__comm )

  def bcast( self, data, (npytype,mpitype) ):
    data = numpy.asarray( data, npytype )
    self.__comm.Bcast( [ data, mpitype ], root=MPI.ROOT )
  
  def gather( self, (npytype,mpitype) ):
    array = numpy.empty( self.size, dtype=npytype )
    self.__comm.Gather( None, [ array, mpitype ], root=MPI.ROOT )
    return array
  
  def gather_equal( self, dtype ):
    array = self.gather( dtype )
    assert numpy.all( array[1:] == array[0] )
    return array[0]
  
  def scatter( self, array, (npytype,mpitype) ):
    array = numpy.asarray( array, dtype=npytype )
    self.__comm.Scatter( [ array, mpitype ], None, root=MPI.ROOT )
  
  def scatterv( self, arrays, (npytype,mpitype) ):
    arrays = [ numpy.asarray( array, dtype=npytype ) for array in arrays ]
    lengths = map( len, arrays )
    offsets = numpy.concatenate( [ [0], numpy.cumsum( lengths[:-1] ) ] ) # first offset = 0
    concatarray = numpy.concatenate( arrays )
    self.__comm.Scatterv( [ concatarray, lengths, offsets, mpitype ], None, root=MPI.ROOT )
  
  def gatherv( self, lengths, (npytype,mpitype) ):
    array = numpy.empty( 20, dtype=npytype )
    offsets = numpy.concatenate( [ [0], numpy.cumsum( lengths[:-1] ) ] ) # first offset = 0
    self.__comm.Gatherv( None, [ array, lengths, offsets, mpitype ], root=MPI.ROOT )
    return [ array[i:i+n] for i, n in zip( offsets, lengths ) ]
  
  def send( self, rank, array, (npytype,mpitype) ):
    array = numpy.asarray( array, npytype )
    self.__comm.Send( [ array, mpitype ], rank, tag=0 )

  def verify( self ):
    raise NotImplementedError
    strlen = self.gather( int )
    good = (strlen == 0)
    if good.all():
      return 0
    msgs = [ 'In libmatrix: %d errors occurred:' % (~good).sum() ]
    for i, s in enumerate( strlen ):
      if s:
        x = numpy.empty( s, dtype='c' )
        self.__comm.Recv( [x,MPI.CHAR], i, tag=10 )
        msgs.append( '[%d] %s' % ( i, x.tostring() ) )
    raise Exception( '\n  '.join( msgs ) )

  def disconnect( self ):
    if self.isconnected():
      self.__comm.Disconnect()

  def __del__( self ):
    self.disconnect()
