from seren.analysis.facade import get_sim_no, Singletons
import commandsource as Commands
import numpy
from scipy import interpolate


def L1errornorm(x=None, y=None, xmin=None, xmax=None, sim = "current", snap = "current"):
    '''Computes the L1 error norm from the simulation data relative to the analytical solution'''
    
    #get the simulation number from the buffer
    simno = get_sim_no(snap)
    
    #istantiate and setup the command object to retrieve analytical solution
    command1 = Commands.AnalyticalPlotCommand(x, y, snap, simno)
    adata = command1.prepareData(Singletons.globallimits)

    #istantiate and setup the 2nd command object to retrieve particle data
    command2 = Commands.ParticlePlotCommand(x, y, snap, simno)
    pdata = command2.prepareData(Singletons.globallimits)

    #cut arrays if limits are provided
    if xmin != None and xmax != None:
        aindex = numpy.logical_and( adata.x_data > xmin, adata.x_data < xmax)
        adata.x_data = adata.x_data[aindex]
        adata.y_data = adata.y_data[aindex]
        pindex = numpy.logical_and( pdata.x_data > adata.x_data.min(),
                                    pdata.x_data < adata.x_data.max() )
        pdata.x_data = pdata.x_data[pindex]
        pdata.y_data = pdata.y_data[pindex]

    
        
    #prepare interpolation function from analytical data
    #f = interpolate.interp1d(adata.x_data[::-1], adata.y_data[::-1], kind = 'linear', axis=0, bounds_error = False)
    f = interpolate.interp1d(adata.x_data[::], adata.y_data[::], kind = 'linear', axis=0, bounds_error = False)

    #compute error norm of particle data relative to analytical data
    L1 = numpy.linalg.norm((pdata.y_data - f(pdata.x_data)), ord=1)/pdata.x_data.size
    return L1