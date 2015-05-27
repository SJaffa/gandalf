//=================================================================================================
//  Sinks.cpp
//  All routines for creating new sinks and accreting gas and updating all
//  sink particle propterties.
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


#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <math.h>
#include "Precision.h"
#include "NbodyParticle.h"
#include "StarParticle.h"
#include "Parameters.h"
#include "Sph.h"
#include "Nbody.h"
#include "Sinks.h"
#include "Debug.h"
#include "Exception.h"
#include "InlineFuncs.h"
using namespace std;



//=================================================================================================
//  Sinks::Sinks()
/// Sinks class constructor
//=================================================================================================
template <int ndim>
Sinks<ndim>::Sinks()
{
  allocated_memory = false;
  Nsink = 0;
  Nsinkmax = 0;
}



//=================================================================================================
//  Sinks::~Sinks()
/// Sinks class destructor
//=================================================================================================
template <int ndim>
Sinks<ndim>::~Sinks()
{
  DeallocateMemory();
}



//=================================================================================================
//  Sinks::AllocateMemory
/// Allocate all memory required for storing sink particle data.
//=================================================================================================
template <int ndim>
void Sinks<ndim>::AllocateMemory(int N)
{
  if (N > Nsinkmax) {
    if (allocated_memory) DeallocateMemory();
    Nsinkmax = N;
    sink = new SinkParticle<ndim>[Nsinkmax];
    allocated_memory = true;
  }

  return;
}



//=================================================================================================
//  Sinks::DeallocateMemory
/// Deallocate all sink particle arrays
//=================================================================================================
template <int ndim>
void Sinks<ndim>::DeallocateMemory(void)
{
  if (allocated_memory) delete[] sink;
  allocated_memory = false;

  return;
}



//=================================================================================================
//  Sinks::SearchForNewSinkParticles
/// Searches through all SPH particles for new sink particle candidates, and
/// if a particle satisfies all tests, then a sink is created.
//=================================================================================================
template <int ndim>
void Sinks<ndim>::SearchForNewSinkParticles
 (int n,                               ///< [in] Current integer time
  FLOAT t,                             ///< [in] Current time
  Sph<ndim> *sph,                      ///< [inout] Object containing SPH ptcls
  Nbody<ndim> *nbody)                  ///< [inout] Object containing star ptcls
{
  bool sink_flag;                      // Flag if particle is to become a sink
  int i;                               // Particle counter
  int isink;                           // i.d. of SPH particle to form sink from
  int k;                               // Dimension counter
  int s;                               // Sink counter
  FLOAT dr[ndim];                      // Relative position vector
  FLOAT drsqd;                         // Distance squared
  FLOAT rho_max = (FLOAT) 0.0;         // Maximum density of sink candidates

  debug2("[Sinks::SearchForNewSinkParticles]");
  timing->StartTimingSection("SEARCH_NEW_SINKS");


  // Continuous loop to search for new sinks.  If a new sink is found, then repeat
  // entire process to search for other sinks on current timestep.
  // If no sinks are found, then exit and return to main program.
  //===============================================================================================
  do {
    isink = -1;
		rho_max = (FLOAT) 0.0;

    // Loop over all SPH particles finding the particle with the highest
    // density that obeys all of the formation criteria, if any do.
    //---------------------------------------------------------------------------------------------
    for (i=0; i<sph->Nhydro; i++) {
      sink_flag = true;
      SphParticle<ndim>& part = sph->GetSphParticlePointer(i);

      // Make sure we don't include dead particles
      if (part.itype == dead) continue;

      // Only consider SPH particles located at a local potential minimum
      if (!part.potmin) continue;

      // If density of SPH particle is too low, skip to next particle
      if (part.rho < rho_sink) continue;

      // Make sure candidate particle is at the end of its current timestep
      if (n%part.nstep != 0) continue;

      // If SPH particle neighbours a nearby sink, skip to next particle
      for (s=0; s<Nsink; s++) {
        for (k=0; k<ndim; k++) dr[k] = part.r[k] - sink[s].star->r[k];
        drsqd = DotProduct(dr,dr,ndim);
        if (drsqd < pow(sink_radius*part.h + sink[s].radius,2)) sink_flag = false;
      }
      if (!sink_flag) continue;


      // If candidate particle has passed all the tests, then check if it is
      // the most dense candidate.  If yes, record the particle id and density
      if (sink_flag && part.rho > rho_max) {
        isink = i;
        rho_max = part.rho;
      }

    }
    //---------------------------------------------------------------------------------------------

#if defined MPI_PARALLEL
    // We need to know what the other processors have found
		vector<FLOAT> rho_maxs(mpicontrol->Nmpi);
		MPI_Allgather(&rho_max,1,GANDALF_MPI_FLOAT,&rho_maxs[0],1,GANDALF_MPI_FLOAT,MPI_COMM_WORLD);
		  const int proc_max = std::max_element(rho_maxs.begin(),rho_maxs.end()) - rho_maxs.begin();
		const FLOAT global_rho_max = rho_maxs[proc_max];
		if (global_rho_max > 0.0) {
			if (rho_max < global_rho_max)
				// A sink is being created, but not on this processor - mark it
				isink=-2;
		}	
#endif


    // If all conditions have been met, then create a new sink particle.
    // Also, set minimum sink smoothing lengtha
    if (isink >= 0) {
      SphParticle<ndim>& part_sink = sph->GetSphParticlePointer(isink);
      sph->hmin_sink = min(sph->hmin_sink,part_sink.h);
      CreateNewSinkParticle(part_sink,isink,t,sph,nbody);

    }
#if defined MPI_PARALLEL
    if (isink != -1) {
			// The owner of the new sink broadcasts it to everyone
			MPI_Bcast(&sink[Nsink],sizeof(SinkParticle<ndim>),MPI_BYTE,proc_max,MPI_COMM_WORLD);
			MPI_Bcast(&nbody->stardata[nbody->Nstar],sizeof(StarParticle<ndim>),MPI_BYTE,proc_max,MPI_COMM_WORLD);
			if (isink == -2) {
					sink[Nsink].star = &(nbody->stardata[nbody->Nstar]);
					nbody->nbodydata[nbody->Nnbody] = &(nbody->stardata[nbody->Nstar]);
			}
    }
#endif

  // Calculate total mass inside sink (direct sum for now since this is not computed that often).
		if (isink != -1) {
			sink[Nsink].mmax = (FLOAT) 0.0;
			for (i=0; i<sph->Nhydro; i++) {
				SphParticle<ndim>& part = sph->GetSphParticlePointer(i);
				if (part.itype == dead) continue;
				for (k=0; k<ndim; k++) dr[k] = sink[Nsink].star->r[k] - part.r[k];
				drsqd = DotProduct(dr,dr,ndim);
				if (drsqd < pow(sink[Nsink].radius,2)) sink[Nsink].mmax += part.m;
			}
#if defined MPI_PARALLEL
  	MPI_Allreduce(MPI_IN_PLACE,&(sink[Nsink].mmax),1,GANDALF_MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
#endif
		}

		if (isink >=0) {
			SphParticle<ndim>& part_sink = sph->GetSphParticlePointer(isink);
			cout << "--------------------------------------------------------------------" << endl;
			cout << "Created new sink particle : " << isink << "     Nsink : " << Nsink+1  << endl;
			cout << "radius : " << sink[Nsink].radius << "    " << part_sink.h << endl;
			cout << "m : " << sink[Nsink].star->m << "    mmax : " << sink[Nsink].mmax << endl;
			cout << "r : " << sink[Nsink].star->r[0] << "   " << sink[Nsink].star->r[0] << endl;
			cout << "--------------------------------------------------------------------" << endl;
		}

  if (isink != -1) {
			nbody->Nstar++;
		  nbody->Nnbody++;
			Nsink++;
  }

  } while (isink != -1);
  //===============================================================================================


  timing->EndTimingSection("SEARCH_NEW_SINKS");

  return;
}



//=================================================================================================
//  Sinks::CreateNewSinkParticle
/// Create a new sink particle from specified SPH particle 'isink' and then
/// removes particle from main arrays.
//=================================================================================================
template <int ndim>
void Sinks<ndim>::CreateNewSinkParticle
 (SphParticle<ndim>& part_sink,        ///< [inout] SPH particle to be turned in a sink particle
  int isink,                           ///< [in]    i.d. of the above SPH particle
  FLOAT t,                             ///< [in]    Current time
  Sph<ndim> *sph,                      ///< [inout] Object containing SPH ptcls
  Nbody<ndim> *nbody)                  ///< [inout] Object containing star ptcls
{
  int i;                               // SPH particle counter
  int k;                               // Dimension counter
  FLOAT dr[ndim];                      // Relative position vector
  FLOAT drsqd;                         // Distance squared

  debug2("[Sinks::CreateNewSinkParticle]");

  // If we've reached the maximum number of sinks, then throw exception
  if (Nsink == Nsinkmax || nbody->Nstar == nbody->Nstarmax) {
    cout << "Run out of memory : " << Nsink << "    " << Nsinkmax << endl;
    exit(0);
  }

  // First create new star and set N-body pointer to star
  sink[Nsink].star = &(nbody->stardata[nbody->Nstar]);
  nbody->nbodydata[nbody->Nnbody] = &(nbody->stardata[nbody->Nstar]);

  // Calculate new sink radius depending on chosen sink parameter
  if (sink_radius_mode == "fixed") {
    sink[Nsink].radius = sink_radius;
  }
  else if (sink_radius_mode == "hmult") {
    sink[Nsink].radius = sink_radius*part_sink.h;
  }
  else {
    sink[Nsink].radius = sph->kernp->kernrange*part_sink.h;
  }


  // Calculate all other sink properties based on radius and SPH particle properties
  sink[Nsink].star->h            = sph->kernp->invkernrange*sink[Nsink].radius;
  sink[Nsink].star->invh         = (FLOAT) 1.0/part_sink.h;
  sink[Nsink].star->radius       = sink[Nsink].radius;
  //sink[Nsink].star->hfactor      = pow(sink[Nsink].star->invh,ndim);
  sink[Nsink].star->m            = part_sink.m;
  sink[Nsink].star->gpot         = part_sink.gpot;
  sink[Nsink].star->gpe_internal = (FLOAT) 0.0;
  sink[Nsink].star->dt           = part_sink.dt;
  sink[Nsink].star->tlast        = t;
  sink[Nsink].star->nstep        = part_sink.nstep;
  sink[Nsink].star->nlast        = part_sink.nlast;
  sink[Nsink].star->level        = part_sink.level;
  sink[Nsink].star->active       = part_sink.active;
  sink[Nsink].star->Ncomp        = 1;
  for (k=0; k<ndim; k++) sink[Nsink].star->r[k] = part_sink.r[k];
  for (k=0; k<ndim; k++) sink[Nsink].star->v[k] = part_sink.v[k];
  for (k=0; k<ndim; k++) sink[Nsink].star->a[k] = part_sink.a[k];
  for (k=0; k<ndim; k++) sink[Nsink].fhydro[k] = part_sink.m*(part_sink.a[k] - part_sink.agrav[k]);
  for (k=0; k<ndim; k++) sink[Nsink].star->adot[k]  = (FLOAT) 0.0;
  for (k=0; k<ndim; k++) sink[Nsink].star->a2dot[k] = (FLOAT) 0.0;
  for (k=0; k<ndim; k++) sink[Nsink].star->a3dot[k] = (FLOAT) 0.0;
  for (k=0; k<ndim; k++) sink[Nsink].star->r0[k]    = part_sink.r0[k];
  for (k=0; k<ndim; k++) sink[Nsink].star->v0[k]    = part_sink.v0[k];
  for (k=0; k<ndim; k++) sink[Nsink].star->a0[k]    = part_sink.a0[k];
  for (k=0; k<ndim; k++) sink[Nsink].star->adot0[k] = (FLOAT) 0.0; //part_sink.adot0[k];
  for (k=0; k<3; k++) sink[Nsink].angmom[k]         = (FLOAT) 0.0;

  // Remove SPH particle from main arrays
  part_sink.m      = (FLOAT) 0.0;
  part_sink.active = false;
  part_sink.itype  = dead;

  return;
}



//=================================================================================================
//  Sinks::AcceteMassToSinks
/// Identify all SPH particles inside sinks and accrete some fraction (or all)
/// of the gas mass to the sinks if selected accretion criteria are satisfied.
//=================================================================================================
template <int ndim>
void Sinks<ndim>::AccreteMassToSinks
 (Sph<ndim> *sph,                      ///< [inout] Object containing SPH ptcls
  Nbody<ndim> *nbody,                  ///< [inout] Object containing star ptcls
  int n,                               ///< [in] Integer timestep
  DOUBLE timestep)                     ///< [in] Minimum timestep level
{
  int i,j,k;                           // Particle and dimension counters
  int Nlist = 0;                       // Max. no of gas particles inside sink
  int Nlisttot = 0;                    // Total number of gas ptcls inside sinks
  int Nneib;                           // No. of particles inside sink
  int s;                               // Sink counter
  int saux;                            // Aux. sink i.d.
  int *ilist;                          // List of particle ids
  FLOAT asqd;                          // Acceleration squared
  FLOAT dr[ndim];                      // Relative position vector
  FLOAT drmag;                         // Distance
  FLOAT drsqd;                         // Distance squared
  FLOAT dt;                            // Sink/star timestep
  FLOAT dv[ndim];                      // Relative velocity vector
  FLOAT dvtang[ndim];                  // Relative tangential velocity vector
  FLOAT efrac;                         // Energy fraction
  FLOAT macc;                          // Accreted mass
  FLOAT macc_temp;                     // Temp. accreted mass variable
  FLOAT mold;                          // Old mass
  FLOAT mtemp;                         // Aux. mass variable
  FLOAT rsqdmin;                       // Distance (sqd) to closest sink
  FLOAT rold[ndim];                    // Old sink position
  FLOAT vold[ndim];                    // Old sink velocity
  FLOAT wnorm;                         // Kernel normalisation factor
  FLOAT *rsqdlist;                     // Array of particle-sink distances

  debug2("[Sinks::AccreteMassToSinks]");
  timing->StartTimingSection("SINK_ACCRETE_MASS");

  // Allocate local memory and initialise values
  for (i=0; i<sph->Ntot; i++) sph->GetSphParticlePointer(i).sinkid = -1;
  for (s=0; s<Nsinkmax; s++) sink[s].Ngas = 0;

  Box<ndim> mydomain = mpicontrol->MyDomain();

  // Determine which sink each SPH particle accretes to.  If none, flag -1
  // (note we should really use the tree to compute this)
  //-----------------------------------------------------------------------------------------------
  for (int ipart=0; ipart<sph->Nhydro + sph->Nmpighost; ipart++) {

    // The loop is over real and MPI ghosts, so need to modify the index accordingly
    if (ipart > sph->Nhydro)
		i = ipart + sph->NPeriodicGhost;
    else
		i = ipart;


    SphParticle<ndim>& part = sph->GetSphParticlePointer(i);
    if (part.itype == dead) continue;

    saux = -1;
    rsqdmin = big_number;

    // Loop over all sinks and mark id of closest sink
    for (s=0; s<Nsink; s++) {
      for (k=0; k<ndim; k++) dr[k] = part.r[k] - sink[s].star->r[k];
      drsqd = DotProduct(dr,dr,ndim);
      if (drsqd <= powf(sink[s].radius + sph->kernrange*part.h,2) && drsqd < rsqdmin) {
#if defined MPI_PARALLEL
        // If the sink is NOT local, then we should not accrete locally this SPH particle
		if (!ParticleInBox(*(sink[s].star), mydomain))
          saux = -1;
        else
#endif
          saux = s;
        rsqdmin = drsqd; //*part.m;
      }

      // If particle is close enough to sink, then record timestep level
      if (drsqd <= powf(sink[s].radius + sph->kernrange*part.h,2)) {
        part.levelneib = max(part.levelneib,sink[s].star->level);
      }
    }

    part.sinkid = saux;
    if (saux != -1) {
      sink[saux].Ngas++;
      Nlisttot++;
      Nlist = max(Nlist,sink[saux].Ngas);
    }
  }
  //-----------------------------------------------------------------------------------------------

  
#if defined MPI_PARALLEL
  // In MPI case, we need to know if other processors found something to accrete
  MPI_Allreduce(MPI_IN_PLACE,&Nlist,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
  // Declare the list that will hold the indices of the ghosts we accreted (to communicate it to their owners)
  list<int> ghosts_accreted;

#endif
  // If there are no particles inside any sink, return to main loop.
  if (Nlist == 0) return;


  // Calculate the accretion timescale and the total mass accreted from all
  // particles for each sink.
  //===============================================================================================
/*#pragma omp parallel for schedule(dynamic,1) default(none)\
  shared(cout,n,Nlist,Nlisttot,sph,timestep) private(asqd,dr,drmag,drsqd,dt) \
  private(dv,dvtang,efrac,i,ilist,ilist2,j,k,macc,macc_temp,mold,mtemp)\
  private(Nneib,rold,rsqdlist,s,vold,wnorm)*/
  for (s=0; s<Nsink; s++) {

#if defined MPI_PARALLEL
    // Only accrete from local sinks
    if (!ParticleInBox(*(sink[s].star), mydomain)) continue;
#endif

    /*cout << "Accreting?? : " << s << "   " << Nsink << "   " << sink[s].Ngas << "    " << n
         << "    " << sink[s].star->nlast << endl;
    cout << "r0 : " << sink[s].star->r[0] << "    " << sink[s].star->r[1] << "    " << sink[s].star->r[2] << endl;
    cout << "v0 : " << sink[s].star->v[0] << "    " << sink[s].star->v[1] << "    " << sink[s].star->v[2] << endl;
    cout << "a0 : " << sink[s].star->a[0] << "    " << sink[s].star->a[1] << "    " << sink[s].star->a[2] << endl;
*/
    // Skip sink if it contains no gas, or unless it's at the beginning of its current step.
    //if (sink[s].Ngas == 0 || !sink[s].star->active) continue;
    //if (sink[s].Ngas == 0 || n%sink[s].star->nstep != 0) continue;
    //if (sink[s].Ngas == 0 || n%sink[s].star->nstep != sink[s].star->nstep/2) continue;
    if (sink[s].Ngas == 0 || sink[s].star->nlast != n) continue;


    // Allocate local array for sink
    ilist = new int[Nlisttot];
    rsqdlist = new FLOAT[Nlisttot];

    // Initialise all variables for current sink
    Nneib = 0;
    wnorm = (FLOAT) 0.0;
    sink[s].menc     = (FLOAT) 0.0;
    sink[s].trad     = (FLOAT) 0.0;
    sink[s].tvisc    = (FLOAT) 1.0;
    sink[s].ketot    = (FLOAT) 0.0;
    sink[s].rotketot = (FLOAT) 0.0;
    sink[s].gpetot   = (FLOAT) 0.0;


    // Calculate distances (squared) from sink to all neighbouring particles
    for (int ipart=0; ipart<sph->Nhydro+sph->Nmpighost; ipart++) {

			// The loop is over real and MPI ghosts, so need to modify the index accordingly
			if (ipart > sph->Nhydro)
				i = ipart + sph->NPeriodicGhost;
			else
				i = ipart;

      SphParticle<ndim>& part = sph->GetSphParticlePointer(i);
      if (part.itype == dead) continue;
      if (part.sinkid == s) {
        for (k=0; k<ndim; k++) dr[k] = part.r[k] - sink[s].star->r[k];
        drsqd = DotProduct(dr,dr,ndim);
        if (drsqd > sink[s].radius*sink[s].radius) continue;
        ilist[Nneib]    = i;
        rsqdlist[Nneib] = drsqd;  //*part.m;
        Nneib++;
        assert(drsqd <= sink[s].radius*sink[s].radius);
        assert(part.m > (FLOAT) 0.0);
      }
    }

    // Double-check that numbers add up here
    //if (Nneib != sink[s].Ngas) {
    //  cout << "Error with neibs : " << Nneib << "   " << sink[s].Ngas << endl;
    //}
    //if (Nneib > Nlist) cout << "ERROR!!" << Nneib << "   " << Nlist << endl;

    // Sort particle ids by increasing distance from the sink
    InsertionSortIds(Nneib,ilist,rsqdlist);

    // Calculate all important quantities (e.g. energy contributions) due to
    // all particles inside the sink
    //---------------------------------------------------------------------------------------------
    for (j=0; j<Nneib; j++) {
      i = ilist[j];

      SphParticle<ndim>& part = sph->GetSphParticlePointer(i);
      if (part.itype == dead) continue;

      for (k=0; k<ndim; k++) dr[k] = part.r[k] - sink[s].star->r[k];
      drsqd = DotProduct(dr,dr,ndim);
      drmag = sqrt(drsqd) + small_number;
      for (k=0; k<ndim; k++) dr[k] /= drmag;

      sink[s].menc += part.m;
      wnorm += part.m*sph->kernp->w0(drmag*sink[s].star->invh)*
        pow(sink[s].star->invh,ndim)*part.invrho;

      // Sum total grav. potential energy of all particles inside sink
      sink[s].gpetot += (FLOAT) 0.5*part.m*(sink[s].star->m + sink[s].menc)*
        sink[s].star->invh*sph->kernp->wpot(drmag*sink[s].star->invh);

      // Compute rotational component of kinetic energy
      for (k=0; k<ndim; k++) dv[k] = part.v[k] - sink[s].star->v[k];
      for (k=0; k<ndim; k++) dvtang[k] = dv[k] - DotProduct(dv,dr,ndim)*dr[k];

      // Compute total and rotational kinetic energies
      sink[s].ketot += part.m*DotProduct(dv,dv,ndim)*
        sph->kernp->w0(drmag*sink[s].star->invh)*pow(sink[s].star->invh,ndim)*part.invrho;
      sink[s].rotketot += part.m*DotProduct(dvtang,dvtang,ndim)*
        sph->kernp->w0(drmag*sink[s].star->invh)*pow(sink[s].star->invh,ndim)*part.invrho;

      // Add contributions to average timescales from particles
      sink[s].tvisc *= pow(sqrt(drmag)/part.sound/part.sound,part.m);
      sink[s].trad += fabs((FLOAT) 4.0*pi*drsqd*part.m*DotProduct(dv,dr,ndim)*
                           sph->kernp->w0(drmag*sink[s].star->invh)*pow(sink[s].star->invh,ndim));

      //cout << "smooth : " << j << "    " << sink[s].star->invh << "    "
        //   <<  sink[s].ketot << "   " << sink[s].rotketot << "    " << sink[s].gpetot << endl;
    }
    //---------------------------------------------------------------------------------------------


    // Normalise SPH sums correctly
    sink[s].ketot *= (FLOAT) 0.5*sink[s].menc/wnorm;
    sink[s].rotketot *= (FLOAT) 0.5*sink[s].menc/wnorm;


    // Calculate the sink accretion timescale and the total amount of mass accreted by sink s
    // this timestep.  If the contained mass is greater than the maximum allowed, accrete the
    // excess mass.  Otherwise, accrete a small amount based on freefall/viscous timescale.
    //---------------------------------------------------------------------------------------------
    if (smooth_accretion == 1) {
      efrac = min((FLOAT) 2.0*sink[s].rotketot/sink[s].gpetot,(FLOAT) 1.0);
      sink[s].tvisc = (sqrt(sink[s].star->m + sink[s].menc)*
                            pow(sink[s].tvisc,(FLOAT) 1.0/sink[s].menc))/alpha_ss;
      sink[s].trad  = sink[s].menc / sink[s].trad;
      sink[s].trot  = twopi*sqrt(pow(sink[s].radius,3)/(sink[s].menc + sink[s].star->m));

      // Finally calculate accretion timescale and mass accreted
      // If there's too much mass inside the sink, artificially increase accretion rate to
      // restore equilibrium (between mass entering sink and that being accreted) quicker.
      sink[s].taccrete = pow(sink[s].trad, (FLOAT) 1.0 - efrac)*pow(sink[s].tvisc,efrac);
      if (sink[s].mmax > small_number && sink[s].menc > sink[s].mmax) {
        sink[s].taccrete *= pow(sink[s].mmax/sink[s].menc,2);
      }
      dt = (FLOAT) sink[s].star->nstep*timestep;
      macc = sink[s].menc*max((FLOAT) 1.0 - exp(-dt/sink[s].taccrete), (FLOAT) 0.0);

      /*cout << "efrac : " << efrac << "    taccrete : " << sink[s].taccrete << "   "
           << sink[s].tvisc << "    " << sink[s].trad << "    " << sink[s].trot << endl;
      cout << "energy : " << sink[s].ketot << "   " << sink[s].rotketot << "    " << sink[s].gpetot << endl;
      cout << "macc : " << macc << "     macc/mmean : " << macc/sph->mmean
           << "    " << sink[s].menc << "    mmax : " << sink[s].mmax
           << "    mmax/mmean : " << sink[s].mmax/sph->mmean << "     dmdt : " << macc/dt << endl;
      */
    }
    else {
      macc = sink[s].menc;
    }


    // Now accrete SPH particles to sink
    //---------------------------------------------------------------------------------------------
    macc_temp = macc;
    for (k=0; k<ndim; k++) rold[k] = sink[s].star->r[k];
    for (k=0; k<ndim; k++) vold[k] = sink[s].star->v[k];
    mold = sink[s].star->m;

    for (k=0; k<ndim; k++) sink[s].star->r[k] *= sink[s].star->m;
    for (k=0; k<ndim; k++) sink[s].star->v[k] *= sink[s].star->m;
    for (k=0; k<ndim; k++) sink[s].star->a[k] *= sink[s].star->m;
    //for (k=0; k<ndim; k++) sink[s].star->adot[k] *= sink[s].star->m;


    // Loop over all neighbouring particles
    //---------------------------------------------------------------------------------------------
    for (j=0; j<Nneib; j++) {
      i = ilist[j];

      SphParticle<ndim>& part = sph->GetSphParticlePointer(i);
      if (part.itype == dead) continue;

      mtemp = min(part.m,macc_temp);
      dt = part.dt;

      //cout << "Removing particle 1?? : " << j << "    " << Nneib << "    " << i << "    " << part.m
      //     << "    " << sph->mmean << "    " << mtemp << "     macc : " << macc_temp
      //     << "      dt : " << part.dt << endl;

      // Special conditions for total particle accretion
      if (smooth_accretion == 0 || part.m - mtemp < smooth_accrete_frac*sph->mmean ||
          dt < smooth_accrete_dt*sink[s].trot) {
        mtemp = part.m;
      }
      macc_temp -= mtemp;

      // Now accrete COM quantities to sink particle
      sink[s].star->m += mtemp;
      for (k=0; k<ndim; k++) sink[s].star->r[k] += mtemp*part.r[k];
      for (k=0; k<ndim; k++) sink[s].star->v[k] += mtemp*part.v[k];
      for (k=0; k<ndim; k++) sink[s].star->a[k] += mtemp*part.a[k];
      //for (k=0; k<ndim; k++) sink[s].star->adot[k] += mtemp*part.adot[k];
      for (k=0; k<ndim; k++) sink[s].fhydro[k] += mtemp*(part.a[k] - part.agrav[k]);
      sink[s].utot += mtemp*part.u;

      // If we've reached/exceeded the mass limit, do not include more ptcls
      if (macc_temp < small_number) break;
    }
    //---------------------------------------------------------------------------------------------


    // Normalise COM quantities
    for (k=0; k<ndim; k++) sink[s].star->r[k] /= sink[s].star->m;
    for (k=0; k<ndim; k++) sink[s].star->v[k] /= sink[s].star->m;
    for (k=0; k<ndim; k++) sink[s].star->a[k] /= sink[s].star->m;
    //for (k=0; k<ndim; k++) sink[s].star->adot[k] /= sink[s].star->m;

    //if (n%sink[s].star->nstep == 0) {
    for (k=0; k<ndim; k++) sink[s].star->r0[k] = sink[s].star->r[k];
    for (k=0; k<ndim; k++) sink[s].star->v0[k] = sink[s].star->v[k];
    for (k=0; k<ndim; k++) sink[s].star->a0[k] = sink[s].star->a[k];
    //for (k=0; k<ndim; k++) sink[s].star->adot0[k] = sink[s].star->adot[k];
      //}

    // Calculate angular momentum of old COM around new COM
    for (k=0; k<ndim; k++) dr[k] = rold[k] - sink[s].star->r[k];
    for (k=0; k<ndim; k++) dv[k] = vold[k] - sink[s].star->v[k];
    if (ndim == 3) {
      sink[s].angmom[0] += mold*(dr[1]*dv[2] - dr[2]*dv[1]);
      sink[s].angmom[1] += mold*(dr[2]*dv[0] - dr[0]*dv[2]);
      sink[s].angmom[2] += mold*(dr[0]*dv[1] - dr[1]*dv[0]);
    }
    else if (ndim == 2) {
      sink[s].angmom[2] += mold*(dr[0]*dv[1] - dr[1]*dv[0]);
    }

    /*cout << "New rcom : " << sink[s].star->r[0] << "    " << sink[s].star->r[1] << "    " << sink[s].star->r[2] << endl;
    cout << "New vcom : " << sink[s].star->v[0] << "    " << sink[s].star->v[1] << "    " << sink[s].star->v[2] << endl;
    cout << "New acom : " << sink[s].star->a[0] << "    " << sink[s].star->a[1] << "    " << sink[s].star->a[2] << endl;
*/
    // Now add angular momentum contribution of individual SPH particles
    //---------------------------------------------------------------------------------------------
    for (j=0; j<Nneib; j++) {
      i = ilist[j];

      SphParticle<ndim>& part = sph->GetSphParticlePointer(i);
      if (part.itype == dead) continue;

      mtemp = min(part.m,macc);
      dt = part.dt;

      //cout << "Removing particle 2?? : " << j << "    " << Nneib << "    " << i << "    " << part.m
      //     << "    " << sph->mmean << "    " << mtemp << "     macc : " << macc
      //     << "      dt : " << part.dt << endl;

      // Special conditions for total particle accretion
      if (smooth_accretion == 0 || part.m - mtemp < smooth_accrete_frac*sph->mmean ||
          dt < smooth_accrete_dt*sink[s].trot) {
        mtemp       = part.m;
        part.m      = (FLOAT) 0.0;
        part.itype  = dead;
        part.active = false;
      }
      else part.m -= mtemp;
#if defined MPI_PARALLEL
		 if (i > sph->Nhydro) {
			 // We are accreting a MPI ghost, so we need to record that to transmit it to the owner
#pragma omp critical (ghost_accreted)
			 ghosts_accreted.push_back(i);
		 }
#endif
      macc -= mtemp;


      // Calculate angular momentum of old COM around new COM
      for (k=0; k<ndim; k++) dr[k] = part.r[k] - sink[s].star->r[k];
      for (k=0; k<ndim; k++) dv[k] = part.v[k] - sink[s].star->v[k];
      if (ndim == 3) {
        sink[s].angmom[0] += mtemp*(dr[1]*dv[2] - dr[2]*dv[1]);
        sink[s].angmom[1] += mtemp*(dr[2]*dv[0] - dr[0]*dv[2]);
        sink[s].angmom[2] += mtemp*(dr[0]*dv[1] - dr[1]*dv[0]);
      }
      else if (ndim == 2) {
        sink[s].angmom[2] += mtemp*(dr[0]*dv[1] - dr[1]*dv[0]);
      }

      // If we've reached/exceeded the mass limit, do not include more ptcls
      if (macc < small_number) break;
    }
    //---------------------------------------------------------------------------------------------


    // Calculate internal sink timestep here
    asqd = DotProduct(sink[s].star->a,sink[s].star->a,ndim);
    sink[s].star->dt_internal = (FLOAT) 0.4*sqrt(sink[s].radius/(sqrt(asqd) + small_number));


    // Free local thread memory
    delete[] rsqdlist;
    delete[] ilist;

  }
  //===============================================================================================

#if defined MPI_PARALLEL
  mpicontrol->UpdateMpiGhostParents(ghosts_accreted,sph);
#endif

  timing->EndTimingSection("SINK_ACCRETE_MASS");

  return;
}



// Create template class instances of the main SphSimulation object for
// each dimension used (1, 2 and 3)
template class Sinks<1>;
template class Sinks<2>;
template class Sinks<3>;
