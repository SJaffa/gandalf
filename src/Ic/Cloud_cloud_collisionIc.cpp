#include <fstream>
#include <sstream>
#include "Precision.h"
#include "Debug.h"
#include "Ic.h"
using namespace std;

//=================================================================================================
//  TurbulentCoreIc::TurbulentCoreIc
/// Set-up SILCC-type simulation initial conditions.
//=================================================================================================
template <int ndim>
Cloud_cloud_collisionIc<ndim>::Cloud_cloud_collisionIc(Simulation<ndim>* _sim, FLOAT _invndim) :
  Ic<ndim>(_sim, _invndim)
{
  // Some sanity checking to ensure correct dimensionality is used
  if (simparams->intparams["ndim"] != 3) {
    ExceptionHandler::getIstance().raise("Turbulent core sim only runs in 3D");
  }
  if (simparams->intparams["dimensionless"] != 0) {
    ExceptionHandler::getIstance().raise("dimensionless units not permitted");
  }
#if !defined(FFTW_TURBULENCE)
  ExceptionHandler::getIstance().raise("FFTW turbulence flag not set");
#endif
}

//=================================================================================================
//  Ic::Cloud_cloud_collision
/// Set-up two turbulent clouds for collision
//=================================================================================================
template <int ndim>
void Cloud_cloud_collisionIc<ndim>::Generate(void)
{
  // Only compile for 3-dimensional case
  //-----------------------------------------------------------------------------------------------
  if (ndim == 3) {

  int i;                               // Particle counter
  int k;                               // Dimension counter
  int Nsphere;                         // Actual number of particles in sphere
  int Npart;                           // Total number of particles
  FLOAT gpecloud1;                     // Total grav. potential energy of cloud 1
  FLOAT gpecloud2;                     // Total grav. potential energy of cloud 2
  FLOAT keturb;                        // Total turbulent kinetic energy of entire cloud
  FLOAT mp1;                           // Mass of one particle in cloud 1
  FLOAT mp2;                           // Mass of one particle in cloud 2
  FLOAT rcentre[ndim];                 // Position of sphere centre
  FLOAT rho1;                          // Fluid density of cloud 1
  FLOAT rho2;                          // Fluid density of cloud 2
  FLOAT xmin;                          // Minimum coordinate value
  FLOAT vfactor;                       // Velocity scaling factor (to scale correct alpha_turb)
  FLOAT v0;                            // Initial relative velocity
  FLOAT *r1;                           // Positions of all particles in cloud 1
  FLOAT *v1;                           // Velocities of all particles in cloud 1
  FLOAT *r2;                           // Positions of all particles in cloud 2
  FLOAT *v2;                           // Velocities of all particles in cloud 2
  FLOAT dxgrid;                        // Grid spacing
  FLOAT rmax[ndim];                    // Maximum size of bounding box
  FLOAT rmin[ndim];                    // Minimum size of bounding box
  DOUBLE *vfield;                      // Table with turbulent velocity field from
  FLOAT xoffset;                       // Initial x-separation of cloud COMs
  FLOAT vxcloud1;                      // Initial x velocity of cloud 1
  FLOAT vxcloud2;                      // Initial x velocity of cloud 2

  // Create local copies of initial conditions parameters
  int field_type   = simparams->intparams["field_type"];
  int gridsize     = simparams->intparams["gridsize"];
  int Npart1        = simparams->intparams["Npart1"];
  int Npart2        = simparams->intparams["Npart2"];
  FLOAT alpha_turb1 = simparams->floatparams["alpha_turb1"];
  FLOAT alpha_turb2 = simparams->floatparams["alpha_turb2"];
  FLOAT gammaone   = simparams->floatparams["gamma_eos"] - 1.0;
  FLOAT mcloud1     = simparams->floatparams["mcloud1"];
  FLOAT mcloud2     = simparams->floatparams["mcloud2"];
  FLOAT mu_bar     = simparams->floatparams["mu_bar"];
  FLOAT power_turb = simparams->floatparams["power_turb"];
  FLOAT radius1     = simparams->floatparams["radius1"];
  FLOAT radius2     = simparams->floatparams["radius2"];
  FLOAT radgrad     = simparams->floatparams["radgrad"];
  FLOAT temp0      = simparams->floatparams["temp0"];
  FLOAT impact_param  = simparams->floatparams["impact_param"];
  FLOAT deltax0       = simparams->floatparams["deltax0"];
  FLOAT relative_vel  = simparams->floatparams["relative_vel"];
  string particle_dist = simparams->stringparams["particle_distribution"];

#if !defined(FFTW_TURBULENCE)
  string message = "FFTW turbulence flag not set";
  ExceptionHandler::getIstance().raise(message);
#endif

    debug2("[Cloud_cloud_collisionIc::Generate]");

  // Convert any parameters to code units
  mcloud1 /= simunits.m.outscale;
  mcloud2 /= simunits.m.outscale;
  radius1 /= simunits.r.outscale;
  radius2 /= simunits.r.outscale;
  temp0  /= simunits.temp.outscale;
  impact_param /= simunits.r.outscale;
  relative_vel /= simunits.v.outscale;

  xoffset=deltax0;

  // compute initial velocity from relative_vel and initial separation

  v0=sqrt(relative_vel*relative_vel+mcloud1*mcloud2/
	  ((mcloud1+mcloud2)*
	   sqrt(xoffset*xoffset+impact_param*impact_param)));

  vxcloud1=-(mcloud2/(mcloud1+mcloud2)*v0);
  vxcloud2=mcloud1/(mcloud1+mcloud2)*v0;

  //cout << "relative_vel, v0, vxcloud1, vxcloud2: " << relative_vel << " "
    //   << v0 << " " << vxcloud1 << " " << vxcloud2 << endl;


  // Calculate gravitational potential energy of uniform density spherical cloud
  gpecloud1 = (FLOAT) 0.6*mcloud1*mcloud1/radius1;
  gpecloud2 = (FLOAT) 0.6*mcloud2*mcloud2/radius2;

  Npart=Npart1+Npart2;

  r1 = new FLOAT[ndim*Npart1];
  v1 = new FLOAT[ndim*Npart1];
  r2 = new FLOAT[ndim*Npart2];
  v2 = new FLOAT[ndim*Npart2];

  // CLOUD 1:
  // Add a sphere of random particles with origin 'rcentre' and radius 'radius'
  for (k=0; k<ndim; k++) rcentre[k] = (FLOAT) 0.0;

  // Create the sphere depending on the choice of initial particle distribution
  if (particle_dist == "random") {
          Ic<ndim>::AddralphaSphere(Npart1, r1, rcentre, radius1, radgrad, sim->randnumb);
  }
  else if (particle_dist == "cubic_lattice" || particle_dist == "hexagonal_lattice") {
    Nsphere = Ic<ndim>::AddLatticeSphere(Npart1, rcentre, radius1, particle_dist, r1, sim->randnumb);
    assert(Nsphere <= Npart1);
    if (Nsphere != Npart1)
      cout << "Warning! Unable to converge to required "
           << "no. of ptcls due to lattice symmetry" << endl;
    Npart1 = Nsphere;
  }
  else {
    string message = "Invalid particle distribution option";
    ExceptionHandler::getIstance().raise(message);
  }

  // CLOUD 2:
  // Add a sphere of random particles with origin 'rcentre' and radius 'radius'
  for (k=0; k<ndim; k++) rcentre[k] = (FLOAT) 0.0;

  // Create the sphere depending on the choice of initial particle distribution
  if (particle_dist == "random") {
    Ic<ndim>::AddralphaSphere(Npart2, r2, rcentre, radius2, radgrad, sim->randnumb);;
    //    for (i=0; i<Npart2; i++){
    //cout << r2[i] << endl;
    //}
  }
  else if (particle_dist == "cubic_lattice" || particle_dist == "hexagonal_lattice") {
    Nsphere = Ic<ndim>::AddLatticeSphere(Npart2, rcentre, radius2, particle_dist, r2, sim->randnumb);
    assert(Nsphere <= Npart2);
    if (Nsphere != Npart2)
      cout << "Warning! Unable to converge to required "
           << "no. of ptcls due to lattice symmetry" << endl;
    Npart2 = Nsphere;
  }
  else {
    string message = "Invalid particle distribution option";
    ExceptionHandler::getIstance().raise(message);
  }

  Npart=Npart1+Npart2;

  //  cout << Npart << endl;

  // Allocate local and main particle memory
  hydro->Nhydro = Npart;
  sim->AllocateParticleMemory();
  mp1 = mcloud1 / (FLOAT) Npart1;
  rho1 = (FLOAT) 3.0*mcloud1 / ((FLOAT) 4.0*pi*pow(radius1,3));


  // Record particle properties in main memory
  for (i=0; i<Npart1; i++) {
    Particle<ndim>& part = hydro->GetParticlePointer(i);
    for (k=0; k<ndim; k++) part.r[k] = r1[ndim*i + k];
    for (k=0; k<ndim; k++) part.v[k] = 0.0;
    part.m = mp1;
    part.h = hydro->h_fac*powf(mp1/rho1,invndim);
    part.u = temp0/gammaone/mu_bar;
    part.ptype = gas_type;
  }

  sim->initial_h_provided = true;


  // Generate turbulent velocity field for given power spectrum slope
  vfield = new DOUBLE[ndim*gridsize*gridsize*gridsize];

  // Calculate bounding box of SPH smoothing kernels
  for (k=0; k<ndim; k++) rmin[k] = big_number;
  for (k=0; k<ndim; k++) rmax[k] = -big_number;
  for (i=0; i<Npart1; i++) {
    Particle<ndim>& part = hydro->GetParticlePointer(i);
    for (k=0; k<ndim; k++) rmin[k] = min(rmin[k], part.r[k] - hydro->kernrange*part.h);
    for (k=0; k<ndim; k++) rmax[k] = max(rmax[k], part.r[k] + hydro->kernrange*part.h);
  }

  xmin = (FLOAT) 9.9e20;
  dxgrid = (FLOAT) 0.0;
  for (k=0; k<ndim; k++) {
    xmin = min(xmin, rmin[k]);
    dxgrid = max(dxgrid, (rmax[k] - rmin[k])/(FLOAT) (gridsize - 1));
    //xmin = min(xmin,rmin[k]);
  }
  dxgrid = max(dxgrid, (FLOAT) 2.0*fabs(xmin)/(FLOAT) (gridsize - 1));

  // Generate gridded velocity field
	Ic<ndim>::GenerateTurbulentVelocityField(field_type, gridsize, power_turb, vfield);

  // Now interpolate generated field onto particle positions
	Ic<ndim>::InterpolateVelocityField(Npart1, gridsize, xmin, dxgrid, r1, vfield, v1);

  // Finally, copy velocities to main SPH particle array
  for (i=0; i<Npart1; i++) {
    Particle<ndim>& part = hydro->GetParticlePointer(i);
    for (k=0; k<ndim; k++) part.v[k] = v1[ndim*i + k];
  }

  // Change to COM frame of reference
  //  sim->SetComFrame();

  // Calculate total kinetic energy of turbulent velocity field
  keturb = (FLOAT) 0.0;
  for (i=0; i<Npart1; i++) {
    Particle<ndim>& part = hydro->GetParticlePointer(i);
    keturb += part.m*DotProduct(part.v, part.v, ndim);
  }
  keturb *= (FLOAT) 0.5;

  vfactor = sqrt(alpha_turb1*gpecloud1/keturb);
  cout << "Scaling factor : " << vfactor << endl;

  // Now rescale velocities to give required turbulent energy in cloud
  for (i=0; i<Npart1; i++) {
    Particle<ndim>& part = hydro->GetParticlePointer(i);
    for (k=0; k<ndim; k++) part.v[k] *= vfactor;
  }

  // Finally, move cloud to required position and bulk velocity
  for (i=0; i<Npart1; i++) {
    Particle<ndim>& part = hydro->GetParticlePointer(i);
    part.r[0]+=xoffset/2.;
    part.r[1]+=impact_param/2.;
    part.v[0]+=vxcloud1;

  }



 
  mp2 = mcloud2 / (FLOAT) Npart2;
  rho2 = (FLOAT) 3.0*mcloud2 / ((FLOAT) 4.0*pi*pow(radius2,3));


  // Record particle properties in main memory
  for (i=Npart1; i<Npart; i++) {
    Particle<ndim>& part = hydro->GetParticlePointer(i);
    for (k=0; k<ndim; k++) {

      part.r[k] = r2[ndim*(i - Npart1) + k];
      //     cout << "particle " << i << " " << k << " " << part.r[k] << " " << r2[ndim*(i - Npart1) + k] << endl;
    }
    for (k=0; k<ndim; k++) part.v[k] = 0.0;
    part.m = mp2;
    part.h = hydro->h_fac*powf(mp2/rho2,invndim);
    part.u = temp0/gammaone/mu_bar;
    part.ptype = gas_type;
  }

  sim->initial_h_provided = true;


  // Generate turbulent velocity field for given power spectrum slope
  vfield = new DOUBLE[ndim*gridsize*gridsize*gridsize];

  // Calculate bounding box of SPH smoothing kernels
  for (k=0; k<ndim; k++) rmin[k] = big_number;
  for (k=0; k<ndim; k++) rmax[k] = -big_number;
  for (i=Npart1; i<Npart; i++) {
    Particle<ndim>& part = hydro->GetParticlePointer(i);
    for (k=0; k<ndim; k++) {

      rmin[k] = min(rmin[k], part.r[k] - hydro->kernrange*part.h);
      //      cout << "rmin: " << rmin[k] << " " << part.r[k] << endl;
    }
    for (k=0; k<ndim; k++) rmax[k] = max(rmax[k], part.r[k] + hydro->kernrange*part.h);
  }

  xmin = (FLOAT) 9.9e20;
  dxgrid = (FLOAT) 0.0;
  for (k=0; k<ndim; k++) {
    xmin = min(xmin, rmin[k]);
    dxgrid = max(dxgrid, (rmax[k] - rmin[k])/(FLOAT) (gridsize - 1));
    //xmin = min(xmin,rmin[k]);
  }
  dxgrid = max(dxgrid, (FLOAT) 2.0*fabs(xmin)/(FLOAT) (gridsize - 1));

  // Generate gridded velocity field
    Ic<ndim>::GenerateTurbulentVelocityField(field_type, gridsize, power_turb, vfield);
 
  //  cout << mcloud1 << " " << mcloud2 << " " << endl;
  //cout << radius1 << " " << radius2 << " " << endl;

  //cout << Npart2 << " " << gridsize << " " << xmin << endl;

  // Now interpolate generated field onto particle positions
	Ic<ndim>::InterpolateVelocityField(Npart2, gridsize, xmin, dxgrid, r2, vfield, v2);

  // Finally, copy velocities to main SPH particle array
  for (i=Npart1; i<Npart; i++) {
    Particle<ndim>& part = hydro->GetParticlePointer(i);
    for (k=0; k<ndim; k++) part.v[k] = v2[ndim*(i - Npart1) + k];
  }



  // Calculate total kinetic energy of turbulent velocity field
  keturb = (FLOAT) 0.0;
  for (i=Npart1; i<Npart; i++) {
    Particle<ndim>& part = hydro->GetParticlePointer(i);
    keturb += part.m*DotProduct(part.v, part.v, ndim);
  }
  keturb *= (FLOAT) 0.5;

  vfactor = sqrt(alpha_turb2*gpecloud2/keturb);
  cout << "Scaling factor : " << vfactor << endl;

  // Now rescale velocities to give required turbulent energy in cloud
  for (i=Npart1; i<Npart; i++) {
    Particle<ndim>& part = hydro->GetParticlePointer(i);
    for (k=0; k<ndim; k++) part.v[k] *= vfactor;
  }

  // Finally, move cloud to required position and bulk velocity
  for (i=Npart1; i<Npart; i++) {
    Particle<ndim>& part = hydro->GetParticlePointer(i);
    part.r[0]-=xoffset/2.;
    part.r[1]-=impact_param/2.;
    part.v[0]+=vxcloud2;

  }

  // Change to COM frame of reference
  sim->SetComFrame();

  // Sanity check

  ofstream sanity;
  sanity.open("cc_sanity_check.dat");
  for (i=0; i<Npart; i++) {
    Particle<ndim>& part = hydro->GetParticlePointer(i);
    sanity << part.r[0] << " " << part.r[1] << " " << part.r[2] << " " << part.v[0] << " " << part.v[1] << " " << part.v[2] << endl;

  }
 

  sanity.close();

  delete[] v1;
  delete[] v2;
  delete[] r1;
  delete[] r2;
  }

  return;
}

template class Cloud_cloud_collisionIc<1>;
template class Cloud_cloud_collisionIc<2>;
template class Cloud_cloud_collisionIc<3>;

