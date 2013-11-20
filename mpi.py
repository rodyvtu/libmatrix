import numpy
from mpi4py import MPI


class InterComm( object ):
  'generic MPI communicator wrapper'

  def __init__( self, cmd, **kwargs ):
    self.__comm = MPI.COMM_SELF.Spawn( cmd, **kwargs )
    self.size = self.__comm.remote_size

  def isconnected( self ):
    return bool( self.__comm )

  def bcast( self, data, dtype ):
    data = numpy.asarray( data, dtype )
    self.__comm.Bcast( [ data, MPI.BYTE ], root=MPI.ROOT )
  
  def gather( self, dtype ):
    array = numpy.empty( self.size, dtype=dtype )
    self.__comm.Gather( None, [ array, MPI.BYTE ], root=MPI.ROOT )
    return array
  
  def gather_equal( self, dtype ):
    array = self.gather( dtype )
    assert numpy.all( array[1:] == array[0] )
    return array[0]
  
  def scatter( self, array, dtype ):
    array = numpy.asarray( array, dtype=dtype )
    self.__comm.Scatter( [ array, MPI.BYTE ], None, root=MPI.ROOT )
  
  def scatterv( self, arrays, dtype ):
    arrays = [ numpy.asarray( array, dtype=dtype ) for array in arrays ]
    nbytes = [ arr.nbytes for arr in arrays ]
    offsets = numpy.concatenate( [ [0], numpy.cumsum( nbytes[:-1] ) ] ) # first offset = 0
    data = numpy.concatenate( arrays )
    self.__comm.Scatterv( [ data, nbytes, offsets, MPI.BYTE ], None, root=MPI.ROOT )
  
  def gatherv( self, lengths, dtype ):
    data = numpy.empty( sum(lengths), dtype=dtype )
    arrays = [ data[i-n:i] for i, n in zip( numpy.cumsum(lengths), lengths ) ]
    nbytes = [ arr.nbytes for arr in arrays ]
    offsets = numpy.concatenate( [ [0], numpy.cumsum( nbytes[:-1] ) ] ) # first offset = 0
    self.__comm.Gatherv( None, [ data, nbytes, offsets, MPI.BYTE ], root=MPI.ROOT )
    return arrays
  
  def send( self, rank, array, dtype ):
    array = numpy.asarray( array, dtype )
    self.__comm.Send( [ array, MPI.BYTE ], rank, tag=0 )

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
