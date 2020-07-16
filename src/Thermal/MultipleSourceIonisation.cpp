#include "Precision.h"
#include "Particle.h"
#include "Debug.h"
#include "SphNeighbourSearch.h"
#include "Radiation.h"
#include "Sinks.h"
#include <sys/stat.h>
#include <iostream>
#include <fstream>
using namespace std;

#ifdef _OPENMP
#include "omp.h"
//==============================================================================
//  OmpNestLockGuard
/// A scoped guard for an OpenMP lock, adapted from the book "Pattern-Oriented
/// Software Architecture".
//==============================================================================
class OmpLockGuard {
public:
  /// Acquire the lock and store a pointer to it
  OmpLockGuard(omp_lock_t *lock) : _lock (lock), _owner (false) {
    acquire();
  }
  /// Set the lock explicitly
  void acquire() {
    omp_set_lock(_lock);
    _owner = true;
  }
  /// Release the lock explicitly (owner thread only!)
  void release() {
    if (_owner) {
      _owner = false;
      omp_unset_lock(_lock);
    }
  }
  ~OmpLockGuard() {
    release();
  }
private:
  omp_lock_t *_lock;
  bool _owner;

  // Disallow copies or assignment
  OmpLockGuard(const OmpLockGuard &);
  void operator=(const OmpLockGuard &);
};
#endif

//=============================================================================
//      MultipleSourceIonisation::MultipleSourceIonisation
//      MultipleSourceIonisation class constructor
//=============================================================================

template <int ndim, template<int> class ParticleType>
MultipleSourceIonisation<ndim,ParticleType>::MultipleSourceIonisation
(SphTree<ndim, ParticleType> *sphneibaux ,
 float mu_baraux,
 float mu_ionaux,
 float temp0aux,
 float temp_ionaux,
 double Ndotminaux,
 float gamma_eosaux,
 float scaleaux,
 float tempscaleaux,
 double rad_contaux)
  
{
  sphneib = sphneibaux;
  mu_bar=mu_baraux;
  mu_ion=mu_ionaux;
  temp0=temp0aux;
  temp_ion=temp_ionaux;
  Ndotmin=Ndotminaux;
  gamma_eos=gamma_eosaux;
  scale=scaleaux;
  tempscale=tempscaleaux;
  rad_cont=rad_contaux;
}

//=============================================================================
//      MultipleSourceIonisation::~MultipleSourceIonisation
//      MultipleSourceIonisation class destructor
//=============================================================================

template <int ndim, template<int> class ParticleType>
MultipleSourceIonisation<ndim,ParticleType>::~MultipleSourceIonisation()
{
}



//=============================================================================
//      MultipleSourceIonisation::UpdateRadiationFieldMMS
//      Calculates the internal energy of particles due to ionising radiation.
//=============================================================================

template <int ndim, template<int> class ParticleType>
void MultipleSourceIonisation<ndim, ParticleType>::UpdateRadiationField
(int Nhydro,
 int Nstar,
 int aux,
 SphParticle<ndim>  * sphgen,
 NbodyParticle<ndim> ** ndata,
 SinkParticle<ndim> * sphaux)
{
  ParticleType<ndim>* sphdata = static_cast<ParticleType<ndim>* > (sphgen);

  ionisation_intergration(Nstar,Nhydro,ndata,sphdata,scale,tempscale,sphneib,
                          temp0,mu_bar,mu_ion,temp_ion,Ndotmin,gamma_eos-1);

  return;
}



//=============================================================================
//      MultipleSourceIonisation::probs
//      Fraction of ionizing flux coming from the test source
//=============================================================================

template <int ndim, template<int> class ParticleType>
double MultipleSourceIonisation<ndim, ParticleType>::probs
(int Nsource,
 IonPar<ndim> *ionisedsph,
 int testpart,
 int testsource)
{
  IonPar<ndim>& test = ionisedsph[testpart] ;

  //Loop over sources to find total photon flux at current location
  double sum = 0;
  double phot = 0 ;
  for (int p=0; p<Nsource; p++)                                             
    {
      IonPar<ndim>& next = ionisedsph[test.neigh[p]] ;

      if (next.ionised[p]==1)  {                        
        sum += next.photons[p];

	if (p == testsource)
	  phot = next.photons[p] ;
      }
    }
  
  //Scale so total fraction of used photons is one
  if (sum > 0)
    return phot / sum ;
  else
    return 0 ;
}

template<class FLOAT>
FLOAT distance_sqd(FLOAT *r1, FLOAT *r2, int ndim) {
  if (ndim == 0)
    return 0 ;
  return (r1[0] - r2[0])*(r1[0] - r2[0]) + distance_sqd(r1+1, r2+1, ndim-1);
}


//=============================================================================
//      MultipleSourceIonisation::photons
//      Compute the photons leaving particle testpart from source pp.
//      This is done by taking the photons from the previous particle in the
//      chain and subtracting off the number required to ionize the particle.
//      If the remaining photons is positive, the particle will be flagged as
//      ionized by this source. Otherwise, the number of photons will be set to
//      zero and the ionized flag for the source will be set to false (0).
//=============================================================================
template <int ndim, template<int> class ParticleType>
double MultipleSourceIonisation<ndim, ParticleType>::photons
(IonPar<ndim> *ionisedsph,
 const vector<int>& sinkid,
 int Ntot,
 int pp,
 int testpart,
 int Nsource,
 int &change)
{
  IonPar<ndim>& test = ionisedsph[testpart] ;
  IonPar<ndim>& source = ionisedsph[sinkid[pp]] ;

  //If the particle has not been checked
  if(test.checked[pp]==0)
    {
#ifdef _OPENMP
      // We need to make sure no other thread tries to access the particle
      // we are updating, so lock it. The lock will be released automatically
      // once the function terminates  
      OmpLockGuard lock_guard(&test.lock);

      // We check again because its possible that we've followed another thread
      // into this if statement and then waited while it updated this particle
      if (test.checked[pp] != 0)
	return test.photons[pp];
#endif


      // The ionizing flux is the total flux if test is a sink
      if (test.sink==1) {
	test.checked[pp] = 1 ;
	return test.photons[pp] ;
      }

      // Compute the amount photons between us and our neighbour
      // If we don't have a neighbour for this source, we will assume
      // zero photons.
      double my_photons ;
      if (test.neigh[pp]!=Ntot)
	{
	  my_photons =
	    photons(ionisedsph,sinkid,Ntot,pp,test.neigh[pp],Nsource,change) ;

	  // Work out the amount absorbed - we can skip this if there are no
	  // photons already
	  if (my_photons > 0) {
	    IonPar<ndim>& next = ionisedsph[test.neigh[pp]] ;
	    double d1=sqrt(distance_sqd(test.r, source.r, ndim));              
	    double d2=sqrt(distance_sqd(next.r, source.r, ndim));
	  
	    double rho_bar ;
	    if (next.sink==0)
	      rho_bar = 0.5*(test.rho+next.rho) ;
	    else
	      rho_bar = test.rho ;

	    //Call the probs function to work out ionisation fraction of wach source    
	    double prob = probs(Nsource,ionisedsph,testpart, pp);
	    double absorbed=rho_bar*rho_bar*(d1*d1*d1 - d2*d2*d2)/3 * prob ;
	    
	    my_photons -= absorbed;
	  }
	  
	  // Limit to zero
	  my_photons = max(my_photons, 0.0) ;
	}
      else
	{
	  my_photons = 0 ; 
	}
    

      // Check whether we've changed the status of this particle
      bool ionised = my_photons > 0 ;
      change += test.ionised[pp] != ionised ;
      test.ionised[pp] = ionised ;

      // Store the flux and report that we've done this particle
      test.photons[pp]= my_photons ;
      test.checked[pp] = 1 ;

      return test.photons[pp];
    }
  else
    {
      return test.photons[pp];
    }


}

//=============================================================================
//      MultipleSourceIonisation::photoncount
//      Compute the photons leaving particle testpart from each source an use
//      that to flag whether the particle should be ionised.
//=============================================================================

template <int ndim, template<int> class ParticleType>
void MultipleSourceIonisation<ndim, ParticleType>::photoncount
(IonPar<ndim> *ionisedsph,
 const vector<int>& sinkid,
 int Ntot,
 int Nsource,
 int testpart,
 int &change)
{
  // Determines if a particle is ionised by seeing if enough
  // gets to the particle from any of the ionizing sources.

  // 1. Get the photons from each source
  for (int p=0; p<Nsource; p++)                            
    photons(ionisedsph,sinkid,Ntot,p,testpart,Nsource,change);
  
  // 2. Set to ionized if we're ionized by any of the sources
  ionisedsph[testpart].fionised=0; 
  for (int p=0; p<Nsource; p++)
    if (ionisedsph[testpart].ionised[p]==1)
      ionisedsph[testpart].fionised=1; 
}

//////////////////////////
//Main contole roughtine//
//////////////////////////
template <int ndim, template<int> class ParticleType>
void MultipleSourceIonisation<ndim, ParticleType>::ionisation_intergration
(int Nstar,                             //Number of stars
 int Nhydro,                            //Number of SPH particles
 NbodyParticle<ndim> ** ndata,          //Source Data
 SphParticle<ndim> * sphgen,            //SPH particle data
 double scale,                          //Scaling
 double tempscale,                      //Temperature scaling
 SphTree<ndim, ParticleType>* sphneib,  //Neighbour Search Roughtine
 double tn,                             //Neutral gas temperature
 double mu_bar,                         //Average neutral gas mass
 double mu_ion,                         //Average ionised gas mass
 double ti,                             //Ionised gas temperature
 double Ndotmin,                        //Minimum Ionising output
 double gammam1)                        //gamma-1
{

  //Casts particle arrays
  ParticleType<ndim>* sphdata = static_cast<ParticleType<ndim>* > (sphgen);

  struct timeval start, end;
  gettimeofday(&start, NULL);

  int debug=2; //Debug mode controller
  float delta=0;

  //Check that the stellar.dat file is present
  struct stat buf;
  stat("stellar.dat", &buf);
  if(S_ISREG(buf.st_mode) == 0)
    {
      cout<<"Stellar.dat is not present in run directory, "
	  << "ionisation will not be included"<<endl;
      return;
    }

  //Checks if there are currently any sinks in gandalf and if not exits
  if (Nstar==0)
    {
      cout<<"No stars"<<endl;
      return;
    }

  // Determine the number and ids of stars that are active sources
  int Nsource=0;
  vector<int> star_id(Nstar) ;

  //Deturmines which sinks are active sources based on user choices
  for(int i=0;i<Nstar;i++)
    {
      if(ndata[i]->NLyC>=Ndotmin)
        { 
          star_id[Nsource]=i;
	  Nsource++ ;
        }
    }

  //Checks if the sinks are of large enough size
  if (Nsource==0)
    {
      cout<<"No stars of suitable mass"<<endl;
      return;
    }

  cout<<"# of sources followed is "<<Nsource<<". ";

  if (debug==1){
    gettimeofday(&end, NULL);

    delta = (((end.tv_sec  - start.tv_sec) * 1000000 +
              end.tv_usec - start.tv_usec) / 1.e6)-delta;
    cout<<delta<<"s to ";
    cout<<"Starting"<<endl; //Debug message
  }

  // Make a local copy of the particle array containing the data needed for
  // the ionization calculation. We will add the sinks to this array too.

  int Ntot = Nhydro+Nsource;   
  
  //Create sink id table.
  vector<int> sinkid(Nsource) ;

  IonPar<ndim> *ionisedsph=new IonPar<ndim>[Ntot];

  //Add ionisedsph particle data
#pragma omp parallel for
  for (int i=0; i<Nhydro; i++) 
    {
      ionisedsph[i] = sphdata[i] ;
      ionisedsph[i].t=tn;                      //Neutral gas temp
      ionisedsph[i].neigh=new int[Nsource];
      ionisedsph[i].photons=new double[Nsource];
      ionisedsph[i].checked=new int[Nsource];
      ionisedsph[i].ionised=new int[Nsource];

      for(int j=0; j<Nsource; j++)
        {
          ionisedsph[i].neigh[j]=Ntot;
	  ionisedsph[i].ionised[j]=0;
        }
#ifdef _OPENMP
      omp_init_lock(&ionisedsph[i].lock) ;
#endif
    }
  //Add sink propertys to sink particles
#pragma omp parallel for
  for (int i=0; i < Nsource; i++)
    {
      ionisedsph[Nhydro+i] = *ndata[star_id[i]] ; 
      sinkid[i]=Nhydro+i;
      ionisedsph[Nhydro+i].neigh=new int[Nsource];
      ionisedsph[Nhydro+i].photons=new double[Nsource];
      ionisedsph[Nhydro+i].checked=new int[Nsource];
      ionisedsph[Nhydro+i].ionised=new int[Nsource];

      for(int j=0; j<Nsource; j++)
        {
          ionisedsph[Nhydro+i].neigh[j]=Ntot;
          ionisedsph[Nhydro+i].photons[j]=0;
          ionisedsph[Nhydro+i].ionised[j]=0;
        }
      // Set the ionising flux for this source.
      // Hard coded constants!
      ionisedsph[Nhydro+i].photons[i] =
	pow(2.4e-24,2.)*ndata[star_id[i]]->NLyC/(4.*pi*2.6e-13)*scale;    

      // Flag the sink as ionized by iteself
      ionisedsph[Nhydro+i].ionised[i]=1; 
      
#ifdef _OPENMP
      omp_init_lock(&ionisedsph[Nhydro+i].lock) ;
#endif
    }


  //Debug message
  if (debug==1){
    gettimeofday(&end, NULL);

    delta = (((end.tv_sec  - start.tv_sec) * 1000000 +
              end.tv_usec - start.tv_usec) / 1.e6)-delta;
    cout<<delta<<"s to ";
    cout<<"Particle arrays created"<<endl;
  }

  //Find the closest source neighbour in chain for each particle
  ////////////////////////////////////////////////////////////////

  // Get the leaf cells in the tree:
  vector<TreeCellBase<ndim> > celllist;
  int Ncells = sphneib->tree->ComputeAllCellList(celllist) ;
  int kernrange = sphneib->kernp->kernrange ;    

#pragma omp parallel
  {
    // Manages the neighbour lists
    NeighbourManager<ndim, IonPar<ndim> > neibmanager =
      sphneib->template create_neighbour_manager<IonPar<ndim> >() ;

    int *activelist = new int[sphneib->tree->MaxNumPartInLeafCell()] ;

    // Begin neighbour find for all particles
#pragma omp for
    for (int cc=0; cc<Ncells; cc++)           
      {
	TreeCellBase<ndim>& cell = celllist[cc];

	// Find the particles in this cell
	int Npartcell =
	  sphneib->tree->ComputeAllParticleList(cell,sphdata,activelist) ;
	if (Npartcell == 0) continue ;

	// Find the neighbour particles for this cell. This list includes all
	// real particles that satisfy |r_i - r_j| < max(h_i, h_j).
	//   We will want the too ghosts at some point, but the rest of the 
	//   code doesn't work with MPI or periodic / reflective boundaries.
	neibmanager.clear() ;
	sphneib->tree->ComputeNeighbourList(cell, neibmanager);
	neibmanager.EndSearch(cell,sphdata);

	// For each particle we want to find the next one in the linked list 
	// pointing towards the sources of ionization.
	for(int ii=0; ii < Npartcell; ii++) {
	  int i = activelist[ii] ;
	  IonPar<ndim> &part = ionisedsph[i] ;
	  
	  // Only do gas particles
	  bool do_part = sphneib->types[part.ptype].hydro_forces;
	  if (not do_part) continue ;
	 	  

	  // Now find the next particle in the direction of each source
	  for (int t=0; t<Nsource; t++) {
	    IonPar<ndim>& source = ionisedsph[sinkid[t]] ;
	    
	    // First checkout whether the source itself is in range
	    double dr_src[ndim] ;
	    for (int k=0; k<ndim; k++)
	      dr_src[k] = part.r[k] - source.r[k];
	    
	    double r_src = sqrt(DotProduct(dr_src, dr_src, ndim)) ;
	    if (r_src <= kernrange*ionisedsph[i].h) {
	      ionisedsph[i].neigh[t] = sinkid[t] ;
	    } else {
	      // Now check each of the neighbours
	      int Nneib = neibmanager.GetNumAllNeib() ;
	      double angle = -1 ;
	      for (int j=0; j < Nneib; j++) {
		IonPar<ndim>& ngb = neibmanager[j] ;

		// Get rid of non-hydro particles
		if (not sphneib->types[part.ptype].hydromask[ngb.ptype])
		  continue ;
		
		// First check that the ngb is within the kernel range
		double dr[ndim] ;
		for (int k=0; k<ndim; k++)
		  dr[k] = part.r[k] - ngb.r[k];
		double r = sqrt(DotProduct(dr, dr, ndim)) ;

		if (r > kernrange*max(ionisedsph[i].h, ngb.h))
		  continue ;
		 
		// Next check if the particle neighbour is nearer to the source
		// than us
		double dr_ngb[ndim] ;
		for (int k=0; k<ndim; k++)
		  dr_ngb[k] = ngb.r[k] - source.r[k];
		double r_ngb = sqrt(DotProduct(dr_ngb, dr_ngb, ndim)) ;

		// Discard ourselves (r = 0)
		if (r_ngb < r_src && r > 0) {
		  // Compute the angle along the line of sight
		  double dot = DotProduct(dr_ngb, dr_src, ndim) ;
		  double angletest= dot/(r_src*r_ngb);
		  
		  // Store the nearest particle to the line of sight
		  if (angletest>angle)
		    {
		      angle=angletest;   
		      part.neigh[t]=neibmanager.GetNeibI(j).first ;
		    }
		}
	      }
	    }
	  }
	}
      }
    delete [] activelist;
  } // End of omp parallel region

  //Debug message
  if (debug==1){
    gettimeofday(&end, NULL);

    delta = (((end.tv_sec  - start.tv_sec) * 1000000 +
              end.tv_usec - start.tv_usec) / 1.e6)-delta;
    cout<<delta<<"s to ";
    cout<<"neigbour step one complete"<<endl;
  }

  //Begin working out if particle are ionised
  ////////////////////////////////////////////

  int change = -1;      // Number of particles with changed ionization state
  int finalcheck ;      // Controls the last iteration of the loop.
  do
    {
      // Reset checked status
#pragma omp parallel for 
      for (int i=0; i<Nhydro; i++)
	for(int p=0; p<Nsource; p++)
	  ionisedsph[i].checked[p]=0;


      // Start the final check if there was no change last time
      finalcheck = (change == 0) ;
      change=0; 

      // Determine if each particle is ionized, counting how many particles
      // change state.
#pragma omp parallel for schedule(dynamic) reduction(+:change)    
      for (int i=0;i<Nhydro;i++)
          photoncount(ionisedsph,sinkid,Ntot,Nsource,i,change);     

      if (debug==1){
        cout<<"The number of adjustments made is "<<change<<endl;     
      }
      

    } while (change!=0 or finalcheck==0)  ;

  
  if (debug==1){
    gettimeofday(&end, NULL);

    delta = (((end.tv_sec  - start.tv_sec) * 1000000 +
              end.tv_usec - start.tv_usec) / 1.e6)-delta;
    cout<<delta<<"s to ";
    cout<<"Iterations complete"<<endl; //Debug message
  }

  //Smooth the temperature
  /////////////////////////////////////
#pragma omp parallel
  {
    // Manages the neighbour lists
    NeighbourManager<ndim, IonPar<ndim> > neibmanager =
      sphneib->template create_neighbour_manager<IonPar<ndim> >() ;

    int *activelist = new int[sphneib->tree->MaxNumPartInLeafCell()] ;

    // Begin neighbour find for all particles
#pragma omp for
    for (int cc=0; cc<Ncells; cc++)           
      {
	TreeCellBase<ndim>& cell = celllist[cc];

	// Find the particles in this cell
	int Npartcell =
	  sphneib->tree->ComputeAllParticleList(cell,sphdata,activelist) ;
	if (Npartcell == 0) continue ;

	// Find the neighbour particles for this cell. This list includes all
	// real particles that satisfy |r_i - r_j| < max(h_i, h_j).
	//   We will want the too ghosts at some point, but the rest of the 
	//   code doesn't work with MPI or periodic / reflective boundaries.
	neibmanager.clear() ;
	sphneib->tree->ComputeNeighbourList(cell, neibmanager);
	neibmanager.EndSearch(cell,ionisedsph);

	// For each particle in the cell compute the smoothed temperature
	for(int ii=0; ii < Npartcell; ii++) {
	  int i = activelist[ii] ;
	  IonPar<ndim> &part = ionisedsph[i] ;

	  // Only do gas particles
	  bool do_part = sphneib->types[part.ptype].hydro_forces;
	  if (not do_part) continue ;

	  // Get the neighbours for this particle.
	  //   Includes particles within max(h_i, h_j) 
	  NeighbourList<IonPar<ndim> > neiblist =
	    neibmanager.GetParticleNeib(part,
					sphneib->types[part.ptype].hydromask,
					false) ;

	  double t_smooth = 0 ;
	  double w_tot = 0 ;
	  int Nneib = neiblist.size() ;
	  for (int j=0; j < Nneib; j++) {
	    IonPar<ndim>& ngb = neiblist[j] ;

	    double rad = sqrt(distance_sqd(part.r, ngb.r, ndim)) ;
	    double h = max(part.h, ngb.h) ;
	    double s = rad/h ;

	    //Work out w for the kernel
	    // Hard coded to use cubic spline to avoid virtual function calls
	    double w = 0 ;
	    if (s < 1) w = 1 - s*s*(1.5 - 0.75*s) ;
	    else if (s < 2) w = 0.25*(2-s)*(2-s)*(2-s) ;
	    
	    w_tot += w ;
	    if (ngb.fionised == 1)
	      t_smooth += ti * w ;
	    else
	      t_smooth += tn * w ;
	  }
	  if (w_tot == 0) {
	    ExceptionHandler::getIstance().raise("wtot=0");
	  }
	  part.t = t_smooth / w_tot ;
	}
      }
    delete [] activelist ;
  } // End of parallel region 

  if (debug==1){
    gettimeofday(&end, NULL);

    delta = (((end.tv_sec  - start.tv_sec) * 1000000 +
              end.tv_usec - start.tv_usec) / 1.e6)-delta;

    cout<<delta<<"s to ";
    cout<<"Temperatures smoothed"<<endl;
  }

#pragma omp parallel for //Initalise openmp
  for (int i=0;i<Nhydro;i++)
    {
      // If the particles temp is less than the neutral temp due to
      // smoothing, set it back to the neutral temp
      if (ionisedsph[i].t<tn)
        ionisedsph[i].t=tn;

      // Interpolate the mean molecular weight to get internal energy
      double f = (ionisedsph[i].t-tn)/(ti-tn);  
      double invmu=f/mu_ion + (1-f)/mu_bar ;
	
      ionisedsph[i].u=ionisedsph[i].t/(tempscale*gammam1*invmu);

      //Set particle ionisation state
      if (ionisedsph[i].fionised==1)
        {
          sphdata[i].ionstate=2;
        }
      else if (ionisedsph[i].t <= 1.01 * tn)
        {
          sphdata[i].ionstate=0;
        }
      else
        {
          sphdata[i].ionstate=1;
        }

      //Write new internal energy to gandalf
      sphdata[i].u=ionisedsph[i].u;     

      //Working out radiation pressure (NOT COMPLEATE)
      /*
      photon_acceleration=3.4455561764e-34*rad_cont*ionisedsph[i].rho/(mu_bar*mu_bar);

      for(int jj=0;jj<Nsource;jj++)
        {
          //Copying ionised state over to holding array
          if(ionisedsph[i].fionised==1)
            {
              theta=atan((ionisedsph[i].r[1]-ionisedsph[sinkid[jj]].r[1])/(ionisedsph[i].r[2]-ionisedsph[sinkid[jj]].r[2]));
              thi=atan((ionisedsph[i].r[1]-ionisedsph[sinkid[jj]].r[1])/(ionisedsph[i].r[0]-ionisedsph[sinkid[jj]].r[0]));
              ionisedsph[i].rad_pre_acc[0]=ionisedsph[i].rad_pre_acc[0]+ionisedsph[i].prob[jj]*photon_acceleration*sin(theta)*cos(thi);
              ionisedsph[i].rad_pre_acc[1]=ionisedsph[i].rad_pre_acc[1]+ionisedsph[i].prob[jj]*photon_acceleration*sin(theta)*sin(thi);
              ionisedsph[i].rad_pre_acc[2]=ionisedsph[i].rad_pre_acc[2]+ionisedsph[i].prob[jj]*photon_acceleration*cos(theta);
            }
        }
      */
      //sphdata[i].rad_pres[0]=0;//ionisedsph[i].rad_pre_acc[0];
      //sphdata[i].rad_pres[1]=0;//ionisedsph[i].rad_pre_acc[1];
      //sphdata[i].rad_pres[2]=0;//ionisedsph[i].rad_pre_acc[2];
    }


  //Memory De-allocation
#pragma omp parallel for 
  for(int i=0;i<Ntot;i++)
    {
      delete [] ionisedsph[i].checked;
      delete [] ionisedsph[i].ionised;
      delete [] ionisedsph[i].neigh;
      delete [] ionisedsph[i].photons;
#ifdef _OPENMP
      omp_destroy_lock(&ionisedsph[i].lock);
#endif
    }

  delete [] ionisedsph;

  gettimeofday(&end, NULL);

  delta = ((end.tv_sec  - start.tv_sec) * 1000000 +
           end.tv_usec - start.tv_usec) / 1.e6;
  cout << "The time taken to calculate ionisation temperatures = "
       << delta << " s" << endl;

  if (debug==1 or debug==2)
    {
      ofstream myfile;
      myfile.open ("timing.dat",ios::app);
      myfile <<delta<<"\n";
    }

}


template class MultipleSourceIonisation<1,GradhSphParticle>;
template class MultipleSourceIonisation<2,GradhSphParticle>;
template class MultipleSourceIonisation<3,GradhSphParticle>;
template class MultipleSourceIonisation<1,SM2012SphParticle>;
template class MultipleSourceIonisation<2,SM2012SphParticle>;
template class MultipleSourceIonisation<3,SM2012SphParticle>;
