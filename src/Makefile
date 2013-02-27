# =============================================================================
# MAKEFILE
# =============================================================================


ifneq ($(PYTHON),)
ifeq ($(NUMPY),)
NUMPY = $(shell $(PYTHON) -c "import numpy; print numpy.get_include()")
endif
ifeq ($(PYLIB),)
PYLIB = $(shell $(PYTHON) -c "import distutils.sysconfig; print distutils.sysconfig.get_python_inc()")
endif
endif

#PYLIB = /sw/include/python2.7
#NUMPY = /sw/lib/python2.7/site-packages/numpy/core/include

#location of the python files produced by swig
PYFOLDER = ../analysis/swig_generated

GTEST = /Users/david/astro/code/gtest-1.6.0/include


# Dimensionality of the code
# ----------------------------------------------------------------------------
ifeq ($(NDIM),1)
CFLAGS += -DNDIM=1 -DVDIM=1 -DFIXED_DIMENSIONS
else ifeq ($(NDIM),2)
CFLAGS += -DNDIM=2 -DVDIM=2 -DFIXED_DIMENSIONS
else ifeq ($(NDIM),3)
CFLAGS += -DNDIM=3 -DVDIM=3 -DFIXED_DIMENSIONS
else ifeq ($(NDIM),0)
CFLAGS += -DNDIM=3 -DVDIM=3
else
ERROR += "Invalid value for NDIM : "$(NDIM)"\n"
endif


# Precision options
# ----------------------------------------------------------------------------
ifeq ($(PRECISION),SINGLE)
CFLAGS += -DSINGLE_PRECISION
else ifeq ($(PRECISION),DOUBLE)
CFLAGS += -DDOUBLE_PRECISION
endif


# Debug flags
# ----------------------------------------------------------------------------
ifeq ($(DEBUG),1)
CFLAGS += -DDEBUG1
else ifeq ($(DEBUG),2)
CFLAGS += -DDEBUG1 -DDEBUG2
endif


# Include expensive verification code
# ----------------------------------------------------------------------------
ifeq ($(VERIFY_ALL),1)
CFLAGS += -DVERIFY_ALL
endif



# Object files to be compiled
# ----------------------------------------------------------------------------
#SWIG_HEADERS = Parameters.i SimUnits.i Sph.i SphSnapshot.i SphSimulation.i
SWIG_HEADERS = SphSimulation.i
WRAP_OBJ = SphSimulation_wrap.o
#WRAP_OBJ = Parameters_wrap.o SimUnits_wrap.o Sph_wrap.o SphSnapshot_wrap.o SphSimulation_wrap.o
OBJ = Parameters.o SimUnits.o SphSnapshot.o SphSimulation.o
OBJ += SphSimulationIC.o SphSimulationIO.o SphSimulationTimesteps.o
OBJ += SphAnalysis.o
OBJ += M4Kernel.o QuinticKernel.o
OBJ += Sph.o GradhSph.o
OBJ += EnergyPEC.o
OBJ += SphIntegration.o SphLeapfrogKDK.o
#OBJ += SphNeighbourSearch.o 
OBJ += BruteForceSearch.o GridSearch.o
OBJ += AdiabaticEOS.o IsothermalEOS.o
OBJ += SimGhostParticles.o
OBJ += toymain.o
OBJ += Exception.o

TEST_OBJ = TestScaling.o Parameters.o SimUnits.o Exception.o

.SUFFIXES: .cpp .i .o


%_wrap.cxx: %.i
	swig -c++ -python -outdir $(PYFOLDER) $(CFLAGS) $<

%.o: %.cxx
	$(CC) $(OPT) $(CFLAGS) -c $< -I$(PYLIB) -I$(PYLIB)/config -I$(NUMPY)

%.o: %.cpp
	$(CC) $(OPT) $(CFLAGS) -c $< -I$(GTEST)


UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
SHARED_OPTIONS = -bundle -flat_namespace -undefined suppress
else ifeq ($(UNAME_S),Linux)
SHARED_OPTIONS = -shared
endif


# =============================================================================
all : $(TEST_OBJ) $(WRAP_OBJ) $(OBJ)
#	@echo -e $(PYLIB)
	$(CC) $(CFLAGS) $(OPT) $(SHARED_OPTIONS) $(WRAP_OBJ) $(OBJ) -o _SphSim.so
	$(F2PY) --quiet -c shocktub.pyf shocktub.f
#uses the interface file rather than directly parsing the fortran file
#if it's needed, interface file can be generated by running
#	f2py -m shocktub -h shocktube.pyf shocktub.f 
	$(CC) $(CFLAGS) $(OPT) -o ../bin/toymain $(OBJ)
	$(CC) $(CFLAGS) $(OPT) -o ../bin/testing $(TEST_OBJ) /Users/david/astro/code/gtest-1.6.0/make/gtest_main.a
	mv shocktub.so ../analysis
	mv _SphSim.so ../analysis/swig_generated

# =============================================================================
clean ::
	\rm -f *_wrap.cxx
	\rm -f *.o
	\rm -f *.so
	\rm -f ../analysis/*.so
	\rm -f ../analysis/swig_generated/*.so
	\rm -f *.pyc
	\rm -f ../analysis/swig_generated/SphSim.py