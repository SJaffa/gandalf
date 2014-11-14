//=================================================================================================
//  MpiControl.h
//  Main MPI class for controlling the distribution of work amongst all MPI
//  tasks for the current simulation, including load balancing and moving
//  and copying particles between nodes.
//
//  This file is part of GANDALF :
//  Graphical Astrophysics code for N-body Dynamics And Lagrangian Fluids
//  https://github.com/gandalfcode/gandalf
//  Contact : gandalfcode@gmail.com
//
//  Copyright (C) 2013  D. A. Hubber, G. Rosotti
//
//  GANDALF is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 2 of the License, or
//  (at your option) any later version.
//
//  GANDALF is distributed in the hope that it will be useful, but
//  WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
//  General Public License (http://www.gnu.org/licenses) for more details.
//=================================================================================================


#ifndef _MPI_CONTROL_H_
#define _MPI_CONTROL_H_


#include <string>
#include "Precision.h"
#include "MpiNode.h"
#include "Sph.h"
#include "Nbody.h"
#include "MpiTree.h"
#include "SphParticle.h"
#include "DomainBox.h"
#include "Diagnostics.h"
#if defined MPI_PARALLEL
#include "mpi.h"
#endif
using namespace std;


static const int tag_srpart = 1;
static const int tag_league = 2;
static const int tag_bal = 3;


//=================================================================================================
//  Class MpiControl
/// \brief   Main MPI control class for managing MPI simulations.
/// \details Main MPI control class for managing MPI simulations.
/// \author  D. A. Hubber, G. Rosotti
/// \date    09/10/2013
//=================================================================================================
template <int ndim>
class MpiControl
{
 protected:

  MPI_Datatype box_type;                 ///< Datatype for the box
  MPI_Datatype diagnostics_type;         ///< Datatype for diagnostic info
  MPI_Datatype ExportParticleType;       ///< Datatype for the information to export
  MPI_Datatype ExportBackParticleType;   ///< Datatype for the information to get back
                                         ///< from exported particles

  // Buffers needed to send and receive particles
  int tot_particles_to_receive;
  std::vector<int> num_particles_export_per_node;
  std::vector<int> displacements_send;
  std::vector<int> num_particles_to_be_received;
  std::vector<int> receive_displs;
  std::vector<int> Nbytes_exported_from_proc;
  std::vector<int> Nbytes_to_each_proc;
  std::vector<Box<ndim> > boxes_buffer;      ///< Buffer needed by the UpdateAllBoundingBoxes routine
  std::vector<int> my_matches;               ///< List of the matches of this node.
                                             ///< For each turn, gives the node we will play with
  SphNeighbourSearch<ndim>* neibsearch;      ///< Neighbour search class

  void CreateLeagueCalendar();


 public:

  // Constructor and destructor
  //-----------------------------------------------------------------------------------------------
  MpiControl();
  ~MpiControl();


  // Other functions
  //-----------------------------------------------------------------------------------------------
  void AllocateMemory(int);
  void DeallocateMemory(void);
  void SetNeibSearch(SphNeighbourSearch<ndim>* _neibsearch) {neibsearch=_neibsearch;}
  void CollateDiagnosticsData(Diagnostics<ndim> &);
  void UpdateAllBoundingBoxes(int, Sph<ndim> *, SphKernel<ndim> *);
  void CommunicatePrunedTrees() {neibsearch->CommunicatePrunedTrees(my_matches,rank);};

  //void ExportMpiGhostParticles(FLOAT, DomainBox<ndim>, Sph<ndim> *);
  virtual void ExportParticlesBeforeForceLoop (Sph<ndim>* sph) = 0;
  virtual void GetExportedParticlesAccelerations (Sph<ndim>* sph) = 0;
  virtual void CreateInitialDomainDecomposition(Sph<ndim> *, Nbody<ndim> *, Parameters*,
                                                DomainBox<ndim>, bool&) = 0;
  virtual void LoadBalancing(Sph<ndim> *, Nbody<ndim> *) = 0;


  // MPI control variables
  //-----------------------------------------------------------------------------------------------
  bool allocated_mpi;                       ///< Flag if memory has been allocated.
  int balance_level;                        ///< MPI tree level to do load balancing
  int rank;                                 ///< MPI rank of process
  int Nmpi;                                 ///< No. of MPI processes
  int Nloadbalance;                         ///< No. of steps between load-balancing
  char hostname[MPI_MAX_PROCESSOR_NAME];    ///< ..
  DomainBox<ndim> mpibox;                   ///< ..
  MpiNode<ndim> *mpinode;                   ///< Data for all MPI nodes
  CodeTiming *timing;                       ///< Simulation timing object (pointer)

};



//=================================================================================================
//  Class MpiControlType
/// \brief   ..
/// \details ..
/// \author  D. A. Hubber, G. Rosotti
/// \date    09/10/2013
//=================================================================================================
template <int ndim, template<int> class ParticleType>
class MpiControlType : public MpiControl<ndim>
{
  using MpiControl<ndim>::balance_level;
  using MpiControl<ndim>::Nmpi;
  using MpiControl<ndim>::mpinode;
  using MpiControl<ndim>::mpibox;
  using MpiControl<ndim>::rank;
  using MpiControl<ndim>::Nloadbalance;
  using MpiControl<ndim>::timing;
  using MpiControl<ndim>::my_matches;
  using MpiControl<ndim>::num_particles_export_per_node;
  using MpiControl<ndim>::displacements_send;
  using MpiControl<ndim>::num_particles_to_be_received;
  using MpiControl<ndim>::receive_displs;
  using MpiControl<ndim>::tot_particles_to_receive;
  using MpiControl<ndim>::Nbytes_exported_from_proc;
  using MpiControl<ndim>::Nbytes_to_each_proc;
  using MpiControl<ndim>::ExportParticleType;
  using MpiControl<ndim>::ExportBackParticleType;
  using MpiControl<ndim>::neibsearch;


  // Buffers needed to send and receive particles
  std::vector<std::vector<ParticleType<ndim>* > > particles_to_export_per_node;
  std::vector<ParticleType<ndim> > particles_to_export;
  std::vector<ParticleType<ndim> > particles_receive;
  std::vector<ParticleType<ndim> > sendbuffer;         ///< Used by the SendParticles routine
  MpiTree<ndim,ParticleType> *mpitree;                 ///< Main MPI load balancing tree
  BruteForceSearch<ndim,ParticleType>* bruteforce;
  MPI_Datatype particle_type;                          ///< Datatype for the particles

  void SendParticles(int Node, int Nparticles, int* list, ParticleType<ndim>*);
  void ReceiveParticles(int Node, int& Nparticles, ParticleType<ndim>** array);


public:

  MpiControlType ();
  ~MpiControlType () {delete bruteforce;};

  virtual void CreateInitialDomainDecomposition(Sph<ndim> *, Nbody<ndim> *, Parameters*,
                                                DomainBox<ndim>, bool&);
  virtual void LoadBalancing(Sph<ndim> *, Nbody<ndim> *);
  int SendReceiveGhosts(const FLOAT, Sph<ndim> *,ParticleType<ndim>**);
  int UpdateGhostParticles(ParticleType<ndim>** array);
  virtual void ExportParticlesBeforeForceLoop (Sph<ndim>* sph);
  virtual void GetExportedParticlesAccelerations (Sph<ndim>* sph);
};

#endif
