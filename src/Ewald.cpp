//=================================================================================================
//  Ewald.cpp
//  Class functions for computing periodic gravity, with either 1D, 2D or 3D periodicity.
//  Based on periodic boundaries derivation by F. Dinnbier.
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


#include <iostream>
#include <string>
#include "Exception.h"
#include "Ewald.h"
#include "Debug.h"
#include "InlineFuncs.h"
using namespace std;



//=================================================================================================
//  Ewald::Ewald
/// Constructor for Ewald periodic gravity object.
//=================================================================================================
template <int ndim>
Ewald<ndim>::Ewald(DomainBox<ndim> &simbox, int _gr_bhewaldseriesn, int _in, int _nEwaldGrid,
                   DOUBLE _ewald_mult, DOUBLE _ixmin, DOUBLE _ixmax, CodeTiming* _timing):
  gr_bhewaldseriesn(_gr_bhewaldseriesn),
  in(_in),
  nEwaldGrid(_nEwaldGrid),
  ewald_mult(_ewald_mult),
  ixmin(_ixmin),
  ixmax(_ixmax),
  lx_per(simbox.boxsize[0]),
  ly_per(simbox.boxsize[1]),
  lz_per(simbox.boxsize[2]),
  timing(_timing)
{
  // Create table for Ewald object

//  const int nEwaldGrid = 16;
//  DOUBLE* ewald_field;
//  DOUBLE* ewald_fields;
//  DOUBLE* ewald_fieldl;
  int ewald_index;
  int i, j, k, jj, ni, nj, nk;
  int hi, hj, hk;
  int es_nrx, es_nry, es_nrz, es_nfx, es_nfy, es_nfz, es_radius2;
  int nterms,nterml;
  int Ncells[3];				// Number of cells in x, y, z
  DOUBLE cr1, cr2, cr3, cf1, cf2, cf3;
  DOUBLE ewc1, ewc2, ewc3, ewc4;
  DOUBLE ratio_p, ratio_pinv, ratio_p1, ratio_p2, ratio_pinv1, ratio_pinv2;
  DOUBLE ewald_alpha, ewald_dzeta;
  DOUBLE x, y, z, xni, yni, zni, rni, rni2;
  DOUBLE linv, linv2, linv_acc;
  DOUBLE rrel[4];
  DOUBLE Lewald[3];				// Size of the Ewald field

  FILE *fo;                                     // file where Ewald field is printed

  debug2("[Ewald::Ewald]");
  timing->StartTimingSection("EWALD",2);

// set ewald_periodicity for given type of boundary conditions
  ewald_periodicity = 0;
  if (simbox.x_boundary_lhs == periodicBoundary && simbox.x_boundary_rhs == periodicBoundary) {
    ewald_periodicity+=1;
  }
  if (simbox.y_boundary_lhs == periodicBoundary && simbox.y_boundary_rhs == periodicBoundary) {
    ewald_periodicity+=2;
  }
  if (simbox.z_boundary_lhs == periodicBoundary && simbox.z_boundary_rhs == periodicBoundary) {
    ewald_periodicity+=4;
  }
  // kill in the case of no periodic boundary conditions
  if ((ewald_periodicity != 1) && (ewald_periodicity != 3) && (ewald_periodicity != 7)) {
    printf("Set normal to the plane in z direction or axis of the cylinder in x direction or periodic BCs");
    exit(0);
  }

// prepare useful constants for generating Ewald field

  es_nrx = gr_bhewaldseriesn;
  es_nry = gr_bhewaldseriesn;
  es_nrz = gr_bhewaldseriesn;
  es_nfx = gr_bhewaldseriesn;
  es_nfy = gr_bhewaldseriesn;
  es_nfz = gr_bhewaldseriesn;
  es_radius2 = gr_bhewaldseriesn*gr_bhewaldseriesn;

// axis ratio of ellipsis
  cr1 = 1.0;
  cr2 = 1.0;
  cr3 = 1.0;
  cf1 = 1.0;
  cf2 = 1.0;
  cf3 = 1.0;

// set sizes and number of cells for the Ewald field
  for (k=0; k<ndim; k++) {
    Lewald[k] = simbox.boxhalf[k];
    Ncells[k] = nEwaldGrid - 1;
  }

  switch (ewald_periodicity) {
    case 1:
      linv = 1.0/(lx_per);
      es_nry = 0;
      es_nrz = 0;
      es_nfy = 0;
      es_nfz = 0;
      Lewald[1] = 4*simbox.boxhalf[0];
      Lewald[2] = Lewald[1];
      Ncells[1] = 4*(nEwaldGrid - 2)+1;
      Ncells[2] = Ncells[1];
      break;
    case 3:
      linv = 1.0/(lx_per);
      ratio_p = (ly_per)/(lx_per);
      es_nry = (int) (1+(gr_bhewaldseriesn)/ratio_p);
      es_nrz = 0;
      es_nfy = (int) (1+(gr_bhewaldseriesn)*ratio_p);
      es_nfz = 0;
      cr2 = pow(ratio_p,2);
      cf2 = 1.0/cr2;
      Lewald[2] = 4*simbox.boxhalf[0];
      Ncells[2] = 4*(nEwaldGrid - 2)+1;
      break;
    case 7:
      linv = 1.0/(lx_per);
      ratio_p1 = (ly_per)/(lx_per);
      ratio_p2 = (lz_per)/(lx_per);
      es_nry = (int) (1+(gr_bhewaldseriesn)/ratio_p1);
      es_nrz = (int) (1+(gr_bhewaldseriesn)/ratio_p2);
      es_nfy = (int) (1+(gr_bhewaldseriesn)*ratio_p1);
      es_nfz = (int) (1+(gr_bhewaldseriesn)*ratio_p2);
      cr2 = pow(ratio_p1,2);
      cr3 = pow(ratio_p2,2);
      cf2 = 1.0/(pow(ratio_p1,2));
      cf3 = 1.0/(pow(ratio_p2,2));
      ratio_pinv1 = 1.0/ratio_p1;
      ratio_pinv2 = 1.0/ratio_p2;
      break;
  }

  if ((ewald_periodicity==1) || (ewald_periodicity==2) || (ewald_periodicity==4)) {
    linv_acc = pi*pow(2.0*linv,2);
  } else if ((ewald_periodicity==3) || (ewald_periodicity==5) || (ewald_periodicity==6)) {
    linv2 = 2.0*linv/(pi*ratio_p);
    accPlane = 2*pi*pow(linv,2)/ratio_p;
    printf("accPlane %lf",accPlane);
  } else if (ewald_periodicity==7) {
    linv2 = linv*ratio_pinv1*ratio_pinv2/(pi);
    linv_acc = 2.0*(pow(linv,2))*ratio_pinv1*ratio_pinv2;
  }

  ratio_pinv = 1.0/ratio_p;
  ewald_alpha = 2.0*linv*(ewald_mult);
  ewald_dzeta = pi*pi*linv*linv/(ewald_alpha*ewald_alpha);

// constants which simplify equations below
  ewc1 = 1.0/sqrt(ewald_dzeta);
  ewc2 = 0.25/ewald_dzeta;

// set gridsize and number of grid points of Ewald field for function CalculatePeriodicCorrection
  for (k=0; k<ndim; k++) {
    dI[k] = (Ncells[k]-1)/Lewald[k];
    Ngrid[k] = Ncells[k]+1;
  }
  one_component = Ngrid[0]*Ngrid[1]*Ngrid[2];
  printf("Ewald test %8d %8d %8d %16.10e %16.10e %16.10e %16.10e %16.10e %16.10e \n",
      Ngrid[0],Ngrid[1],Ngrid[2],Lewald[0],Lewald[1],Lewald[2],dI[0],dI[1],dI[2]);
// generate the Ewald field
// from 0 to one_component-1 is stored potential,
// from one_component*jj to one_component*(jj+1)-1 is stored
// the jj component of acceleration (jj=1 is x, 2 is y, 3 is z)
  ewald_field = new DOUBLE[4*one_component];
  ewald_fields = new DOUBLE[4*one_component];
  ewald_fieldl = new DOUBLE[4*one_component];
//  ewald_coord = new DOUBLE[3*one_component];

  for (k=0; k<Ngrid[2]; k++) {
    for (j=0; j<Ngrid[1]; j++) {
      for (i=0; i<Ngrid[0]; i++) {

 // add parallelization !!!
        nterms=0;
        nterml=0;
        ewald_index = k*Ngrid[1]*Ngrid[0] + j*Ngrid[0] + i;
        x = i*Lewald[0]/(Ncells[0]-1);
        y = j*Lewald[1]/(Ncells[1]-1);
        z = k*Lewald[2]/(Ncells[2]-1);

//  printf("EF %d %d %d %d %16.10lf %16.10lf %16.10lf\n",i,j,k,ewald_index,x,y,z);

        ewald_field[ewald_index] = 0.0;

// short-range contributions

        for (ni=-es_nrx; ni<=es_nrx; ni++) {
          for (nj=-es_nry; nj<=es_nry; nj++) {
            for (nk=-es_nrz; nk<=es_nrz; nk++) {
              if ((cr1*ni*ni+cr2*nj*nj+cr3*nk*nk) <= es_radius2) {
                rrel[1] = x + ni*(lx_per);
                rrel[2] = y + nj*(ly_per);
                rrel[3] = z + nk*(lz_per);
                rni2 = pow(rrel[1],2)+pow(rrel[2],2)+pow(rrel[3],2);
                rni = sqrt(rni2);

                nterms++;

                ewald_field[ewald_index] = ewald_field[ewald_index] + erfc(ewald_alpha*rni)/rni;

                for (jj=1; jj<4; jj++) {
                  ewald_field[ewald_index+jj*one_component] = ewald_field[ewald_index+jj*one_component] + AccShort(ewald_alpha,rrel[jj],rni);
                }
              }
            }
          }
        }

        for (hi=-es_nfx; hi<=es_nfx; hi++) {
          for (hj=-es_nfy; hj<=es_nfy; hj++) {
            for (hk=-es_nfz; hk<=es_nfz; hk++) {
              if ((cf1*hi*hi+cf2*hj*hj+cf3*hk*hk) <= es_radius2) {

                nterml++;
                if (ewald_periodicity==1) {
                  ewald_field[ewald_index] = ewald_field[ewald_index] + PotLong1p2i(hi, ewald_dzeta, linv, x, y, z);
                  ewald_field[ewald_index+one_component] = ewald_field[ewald_index+one_component] +
                      linv_acc*(AccLong1p2iPer(hi, ewald_dzeta, linv, x, y, z));
                  ewald_field[ewald_index + 2*one_component] = ewald_field[ewald_index + 2*one_component] +
                      linv_acc*(AccLong1p2iIso(hi, ewald_dzeta, linv, x, y, z));
                  ewald_field[ewald_index + 3*one_component] = ewald_field[ewald_index + 3*one_component] +
                      linv_acc*(AccLong1p2iIso(hi, ewald_dzeta, linv, x, z, y));
                } else if (ewald_periodicity==3) {
                      ewald_field[ewald_index] = ewald_field[ewald_index] + PotLong2p1i(ewald_dzeta, linv, ratio_pinv, x, y, z, hi, hj, ewc1, ewc2);
                      ewald_field[ewald_index+one_component] = ewald_field[ewald_index+one_component] +
                          hi*(AccLong2p1iPer(ewald_dzeta, linv, ratio_pinv, x, y, z, hi, hj, ewc1, ewc2));
                      ewald_field[ewald_index + 2*one_component] = ewald_field[ewald_index + 2*one_component] +
                          hj*ratio_pinv*(AccLong2p1iPer(ewald_dzeta, linv, ratio_pinv, x, y, z, hi, hj, ewc1, ewc2));
                      ewald_field[ewald_index + 3*one_component] = ewald_field[ewald_index + 3*one_component] +
                          AccLong2p1iIso(ewald_dzeta, linv, ratio_pinv, x, y, z, hi, hj, ewc1, ewc2);
                } else if (ewald_periodicity==7) {
                    if ((pow(hi,2)+pow(hj,2)+pow(hk,2)) > 0) {
                      ewc3 = pow(hi,2)+pow(hj*ratio_pinv1,2)+pow(hk*ratio_pinv2,2);
                      ewc4 = twopi*linv*(hi*x+hj*y*ratio_pinv1+hk*z*ratio_pinv2);
                      ewald_field[ewald_index] = ewald_field[ewald_index] +
                          linv2*exp(-ewald_dzeta*ewc3)*cos(ewc4)/ewc3;
                      ewald_field[ewald_index+one_component] = ewald_field[ewald_index+one_component] +
                          AccLong3pPer(hi, linv_acc, ewald_dzeta, ewc3, ewc4);
                      ewald_field[ewald_index + 2*one_component] = ewald_field[ewald_index + 2*one_component] +
                          ratio_pinv1*(AccLong3pPer(hj, linv_acc, ewald_dzeta, ewc3, ewc4));
                      ewald_field[ewald_index + 3*one_component] = ewald_field[ewald_index + 3*one_component] +
                          ratio_pinv2*(AccLong3pPer(hk, linv_acc, ewald_dzeta, ewc3, ewc4));
                    }
                }
              }
            }
          }
        }
// End of the loop over whole grid

// subtract the 1/r term from potential and 1/r**2 from acceleration
        rni = sqrt(pow(x,2)+pow(y,2)+pow(z,2));
        ewald_field[ewald_index] = ewald_field[ewald_index] - 1.0/rni;
        ewald_field[ewald_index + one_component] = ewald_field[ewald_index + one_component] - x/pow(rni,3);
        ewald_field[ewald_index + 2*one_component] = ewald_field[ewald_index + 2*one_component] - y/pow(rni,3);
        ewald_field[ewald_index + 3*one_component] = ewald_field[ewald_index + 3*one_component] - z/pow(rni,3);
      }
    }
  }

// omit NaN caused by 1/0 at the origin
  ewald_field[0] = 0.0;
  ewald_field[one_component] = 0.0;
  ewald_field[2*one_component] = 0.0;
  ewald_field[3*one_component] = 0.0;

// write the Ewald field into file
  fo=fopen("ewald_field","w");
  for (k=0; k<Ngrid[2]; k++) {
    for (j=0; j<Ngrid[1]; j++) {
      for (i=0; i<Ngrid[0]; i++) {
        ewald_index = k*Ngrid[1]*Ngrid[0] + j*Ngrid[0] + i;
        x = i*Lewald[0]/(Ncells[0]-1);
        y = j*Lewald[1]/(Ncells[1]-1);
        z = k*Lewald[2]/(Ncells[2]-1);
        fprintf(fo,"%16.10e %16.10e %16.10e %10d %16.10e %16.10e %16.10e %16.10e \n",
          x, y, z, ewald_index, ewald_field[ewald_index], ewald_field[ewald_index + one_component],
          ewald_field[ewald_index + 2*one_component], ewald_field[ewald_index + 3*one_component]);
      }
    fprintf(fo,"\n");
    }
    fprintf(fo,"\n");
  }

  fclose(fo);

// determine the constant potC1p2i in the 1p2i case and correct for -1/r term
  if (ewald_periodicity==1) {
    potC1p2i = 2.0*linv*log(2*lx_per) + ewald_field[Ngrid[0]*(Ngrid[1]-2)];
//  } else if (ewald_periodicity==2) {
//    potC1p2i = 2.0*linv*log(2*ly_per) + 
//  } else if (ewald_periodicity==4) {
//    potC1p2i = 2.0*linv*log(2*lz_per) + 
  }

  potC1p2i = -(potC1p2i + 0.5/lx_per);

        printf("potC1p2i %16.10lf %16.10lf %16.10lf %16.10lf %8d\n",potC1p2i,2.0*linv*log(2*lx_per),
           ewald_field[Ngrid[0]*(Ngrid[1]-2)],ewald_field[Ngrid[0]*(Ngrid[1]-1)]+0.5/lx_per,Ngrid[0]*(Ngrid[1]-2));

  timing->EndTimingSection("EWALD");
}



//=================================================================================================
//  Ewald::~Ewald
/// Ewald object destructor.  Deallocates all memory for look-up tables
//=================================================================================================
template <int ndim>
Ewald<ndim>::~Ewald()
{
}



//=================================================================================================
//  Ewald::CalculatePeriodicCorrection
/// ..
//=================================================================================================
template <int ndim>
void Ewald<ndim>::CalculatePeriodicCorrection
 (FLOAT m,                            ///< [in] Mass of particle/cell
  FLOAT dr[ndim],                     ///< [in] Relative vector to closest periodic replicant
  FLOAT acorr[ndim],                  ///< [out] Periodic correction acceleration
  FLOAT &gpotcorr)                    ///< [out] Periodic correction grav. potential
{
  int k;
  int indexBase0, indexBase1, indexBase2, indexBase3;
  int il[3], ip[3];
  double b, c, d;
  double drAbsInv;
  double a[3];
  double grEwald[4];				// array for potential and acceleration
  double wf[8];					// weighting factors used for interpolation
  // find edges of the ewald_field cuboid around dr
  for (k=0; k<ndim; k++) {
    b = dI[k]*abs(dr[k]);
    il[k] = (int) b;
    ip[k] = il[k] + 1;
    a[k] = b - il[k];
  }

// in isolated direction(s) at distance longer than 2*(the size of comp. domain in
// periodic direction) is Ewald field approximated by analytical formula
  if (ewald_periodicity != 7) {
    if (max3(ip[0],ip[1],ip[2]) > (4*nEwaldGrid-7)) {

// compute 1/r term explicitly. Hopefully in the future
// this value will be simply passed to the routine
// the term 1/r or 1/r**2 is subtracted to compensate for the major contribution
      drAbsInv = 1.0/sqrt(pow(dr[0],2) + pow(dr[1],2) + pow(dr[2],2));
      switch (ewald_periodicity) {
        case 1:
          c = pow(dr[1],2)+pow(dr[2],2);
          d = 2.0/(lx_per*c);
          gpotcorr = -m*(log(c)/lx_per+potC1p2i + drAbsInv);
          acorr[0] = -m*dr[0]*pow(drAbsInv,3);
          acorr[1] = m*(dr[1]*d - dr[1]*pow(drAbsInv,3));
          acorr[2] = m*(dr[2]*d - dr[2]*pow(drAbsInv,3));
          break;
        case 3:
          gpotcorr = -m*(abs(dr[2])*accPlane + drAbsInv);
          acorr[0] = -m*dr[0]*pow(drAbsInv,3);
          acorr[1] = -m*dr[1]*pow(drAbsInv,3);
          acorr[2] = m*(accPlane*sgn(dr[2]) - pow(drAbsInv,3)*dr[2]);
          break;

      }
      return;
    }
  }

  // weighting factors

  wf[0]=(1.0 - a[2])*(1.0 - a[1])*(1.0 - a[0]);
  wf[1]=(1.0 - a[2])*(1.0 - a[1])*(      a[0]);
  wf[2]=(1.0 - a[2])*(      a[1])*(1.0 - a[0]);
  wf[3]=(1.0 - a[2])*(      a[1])*(      a[0]);
  wf[4]=(      a[2])*(1.0 - a[1])*(1.0 - a[0]);
  wf[5]=(      a[2])*(1.0 - a[1])*(      a[0]);
  wf[6]=(      a[2])*(      a[1])*(1.0 - a[0]);
  wf[7]=(      a[2])*(      a[1])*(      a[0]);

  indexBase0 = il[0] + il[1]*Ngrid[0] + il[2]*Ngrid[0]*Ngrid[1];
  indexBase1 = indexBase0 + Ngrid[0]*Ngrid[1];
  for (k=0; k<=ndim; k++) {
    indexBase2 = indexBase0 + k*one_component;
    indexBase3 = indexBase1 + k*one_component;
    grEwald[k] = wf[0] * ewald_field[indexBase2] + wf[1] * ewald_field[indexBase2 + 1] +
                 wf[2] * ewald_field[indexBase2 + Ngrid[0]] + wf[3] * ewald_field[indexBase2 + Ngrid[0] + 1] +
                 wf[4] * ewald_field[indexBase3] + wf[5] * ewald_field[indexBase3 + 1] +
                 wf[6] * ewald_field[indexBase3 + Ngrid[0]] + wf[7] * ewald_field[indexBase3 + Ngrid[0] + 1];
  }

  gpotcorr = m*ewald_field[indexBase0];
  for (k=0; k<ndim; k++) {
    acorr[k] = m*grEwald[k+1]*sgn(dr[k]);
  }

  return;
}

//=================================================================================================
//  Ewald::erfcx
/// scaled complementary error function erfcx = exp(x**2)*erfc(x)
//=================================================================================================
template <int ndim>
DOUBLE Ewald<ndim>::erfcx(DOUBLE x)
{
  DOUBLE a;
  a = pow(x,2) + gsl_sf_log_erfc(x);
  return (exp(a));
}



//=================================================================================================
//  Ewald::SimpsonInt
/// Simpson integrator
//=================================================================================================
template <int ndim>
DOUBLE Ewald<ndim>::SimpsonInt
 (DOUBLE (Ewald<ndim>::*f)(DOUBLE, int, DOUBLE, DOUBLE),      ///< ..
  int l,                                         ///< ..
  DOUBLE ewald_dzeta,                            ///< ..
  DOUBLE ewald_eta,                              ///< ..
  DOUBLE xmin,                                   ///< ..
  DOUBLE xmax,                                   ///< ..
  int n)                                         ///< ..
{
  DOUBLE sum = 0.0;                              // ..
  DOUBLE x0prev = xmin;                          // ..
  DOUBLE dx = (xmax - xmin)/(2*n);               // ..
  DOUBLE x0, x1, x2, f0, f1, f2;
  int i;

  for (i=0; i<n; i++)
  {
    x0 = x0prev;
    x1 = x0prev + dx;
    x2 = x0prev + 2*dx;
    f0 = (this->*f)(x0, l, ewald_dzeta, ewald_eta);
    f1 = (this->*f)(x1, l, ewald_dzeta, ewald_eta);
    f2 = (this->*f)(x2, l, ewald_dzeta, ewald_eta);
    x0prev += 2*dx;
    sum += (f0+4*f1+f2);
  }

  return (sum*dx/3);
}



//=================================================================================================
//  Ewald::GravInt2p1i
//  integral for 2p1i relevant to the periodic directions
//=================================================================================================
template <int ndim>
DOUBLE Ewald<ndim>::GravInt2p1i
 (int l1,					///<
  int l2, 					///<
  DOUBLE z, 					///<
  DOUBLE linv, 	      				///<
  DOUBLE ewald_dzeta,				///<
  DOUBLE ratio_pinv, 				///<
  DOUBLE ewc1,  				///<
  DOUBLE ewc2)					///<
{
// a is named ewald_gamma in FLASH
  DOUBLE a = twopi*z*linv;			// ..
  DOUBLE sqrt_term;				// ..
  DOUBLE sqrt_term2; 				// ..
  DOUBLE c;					// ..
  DOUBLE d;					// ..

// distinguish between 0,0 and the other cases
  if ((l1 != 0) or (l2 != 0)) {
    sqrt_term2 = (DOUBLE) pow(l1,2)+pow(l2*ratio_pinv,2);
    sqrt_term = sqrt(sqrt_term2);
    c = exp(-pow(a,2)*ewc2 - ewald_dzeta*sqrt_term2)*erfcx(ewc1*(ewald_dzeta*sqrt_term + 0.5*a)) +
        exp(-a*sqrt_term)*erfc(ewc1*(ewald_dzeta*sqrt_term - 0.5*a));
    d = 1.0/(2.0*sqrt_term);
  return(c*d);
  } else {
    c = -(a*erf(a*0.5*ewc1)+2.0*sqrt(ewald_dzeta*invpi)*exp(-pow(a,2)*ewc2));
    return c;
  }
}



//=================================================================================================
//  Ewald::DerGravInt2p1i
//  integral for 2p1i relevant to the isolated direction
//=================================================================================================
template <int ndim>
DOUBLE Ewald<ndim>::DerGravInt2p1i
 (int l1, 					///<
  int l2, 					///<
  DOUBLE z, 					///<
  DOUBLE linv, 					///<
  DOUBLE ewald_dzeta,				///<
  DOUBLE ratio_pinv, 				///<
  DOUBLE ewc1, 					///<
  DOUBLE ewc2)					///<
{
// a is named ewald_gamma in FLASH
  DOUBLE a = twopi*z*linv;
  DOUBLE sqrt_term2 = (DOUBLE) pow(l1,2)+pow(l2*ratio_pinv,2);
  DOUBLE sqrt_term = sqrt(sqrt_term2);
  DOUBLE c = erfcx(ewc1*(ewald_dzeta*sqrt_term + 0.5*a))*
    exp(-pow(a,2)*ewc2 - ewald_dzeta*sqrt_term2) -
    erfc(ewc1*(ewald_dzeta*sqrt_term - 0.5*a))*exp(-a*sqrt_term);
  return c;
}



//=================================================================================================
//  Ewald::IntFPot
//  integrated function for potential in the 1p2i case (it is also used for acceleration
//  in isolated direstions)
//=================================================================================================
template <int ndim>
DOUBLE Ewald<ndim>::IntFPot
 (DOUBLE x, 					///<
  int l, 					///<
  DOUBLE ewald_dzeta, 				///<
  DOUBLE ewald_eta)				///<
{
  // the function is different for l=0 and the other cases
  if (l == 0) {
    return((gsl_sf_bessel_J0(x*ewald_eta)-1.0)*exp(-ewald_dzeta*pow(x,2))/x);
  }
  else {
    return (gsl_sf_bessel_J0(x*ewald_eta)*exp(-ewald_dzeta*pow(x,2))*x/
             (pow((DOUBLE) l,2)+pow(x,2)));
  }
}



//=================================================================================================
//  Ewald::IntFAcc
//  integrated function for acceleration in the 1p2i case
//=================================================================================================
template <int ndim>
DOUBLE Ewald<ndim>::IntFAcc
 (DOUBLE x, 					///<
  int l, 					///<
  DOUBLE ewald_dzeta, 				///<
  DOUBLE ewald_eta)				///<
{
  return(gsl_sf_bessel_J1(x*ewald_eta)*exp(-ewald_dzeta*pow(x,2))*
         pow(x,2)/(pow((DOUBLE) l,2)+pow(x,2)));
}



//=================================================================================================
//  Ewald::AccShort
//  short-range contributions to acceleration
//=================================================================================================
template <int ndim>
DOUBLE Ewald<ndim>::AccShort
 (DOUBLE ewald_alpha, 				///<
  DOUBLE proj,	 				///<
  DOUBLE r)					///<
{
  DOUBLE r2 = pow(r,2);
  DOUBLE s1 = 2.0*ewald_alpha*proj*exp(-r2*pow(ewald_alpha,2))/(sqrt(pi)*r2);
  DOUBLE s2 = proj*erfc(ewald_alpha*r)/pow(r,3);
  return(s1+s2);
}



//=================================================================================================
//  Ewald::PotLong1p2i
//  long-range contributions to potential in the 1p2i case
//=================================================================================================
template <int ndim>
DOUBLE Ewald<ndim>::PotLong1p2i
  (int l, 					///<
  DOUBLE ewald_dzeta, 				///<
  DOUBLE linv, 					///<
  DOUBLE x, 					///<
  DOUBLE y, 					///<
  DOUBLE z)					///<
{
  typedef DOUBLE (Ewald<ndim>::*fdef)(DOUBLE, int, DOUBLE, DOUBLE);
  fdef f = &Ewald<ndim>::IntFPot;
  DOUBLE ewald_eta = twopi*linv*hypot(y,z);
  DOUBLE a = 2.0*linv*exp(-ewald_dzeta*pow(l,2))*cos(twopi*l*x*linv);
  a *= SimpsonInt(f,l,ewald_dzeta,ewald_eta,ixmin,ixmax,in);
  return(a);
}



//=================================================================================================
//  Ewald::AccLong1p2iPer
//  long-range contributions to acceleration in periodic direction and in the 1p2i case
//=================================================================================================
template <int ndim>
DOUBLE Ewald<ndim>::AccLong1p2iPer
 (int l,
  DOUBLE ewald_dzeta, 				///<
  DOUBLE linv,	 				///<
  DOUBLE x, 					///<
  DOUBLE y, 					///<
  DOUBLE z)					///<
{
  typedef DOUBLE (Ewald<ndim>::*fdef)(DOUBLE, int, DOUBLE, DOUBLE);
  fdef f = &Ewald<ndim>::IntFPot;
  DOUBLE ewald_eta = twopi*linv*hypot(y,z);
  DOUBLE a = l * exp(-ewald_dzeta*pow(l,2))*sin(twopi*linv*l*x);
  a *= SimpsonInt(f,l,ewald_dzeta,ewald_eta,ixmin,ixmax,in);
  return(a);
}



//=================================================================================================
//  Ewald::AccLong1p2iIso
//  long-range contributions to acceleration in isolated directions and in the 1p2i case
//=================================================================================================
template <int ndim>
DOUBLE Ewald<ndim>::AccLong1p2iIso
 (int l, 					///<
  DOUBLE ewald_dzeta, 				///<
  DOUBLE linv,	 				///<
  DOUBLE x, 					///<
  DOUBLE y, 					///<
  DOUBLE z)					///<
{
  typedef DOUBLE (Ewald<ndim>::*fdef)(DOUBLE, int, DOUBLE, DOUBLE);
  fdef f = &Ewald<ndim>::IntFAcc;
  DOUBLE a;
  DOUBLE ewald_eta = twopi*linv*hypot(y,z);
  if (ewald_eta != 0.0) {
    a = y*exp(-ewald_dzeta*pow(l,2))*cos(twopi*linv*l*x)/hypot(y,z);
    a *= SimpsonInt(f,l,ewald_dzeta,ewald_eta,ixmin,ixmax,in);
  } else {
    a = 0.0;
  }
  return(a);
}



//=================================================================================================
//  Ewald::PotLong2p1i
//  long-range contributions to potential in 2p1i case
//=================================================================================================
template <int ndim>
DOUBLE Ewald<ndim>::PotLong2p1i
 (DOUBLE ewald_dzeta,				///<
  DOUBLE linv, 					///<
  DOUBLE ratio_pinv, 				///<
  DOUBLE x,					///<
  DOUBLE y, 					///<
  DOUBLE z, 					///<
  int l1, 					///<
  int l2, 					///<
  DOUBLE ewc1, 					///<
  DOUBLE ewc2)					///<
{
  DOUBLE b = linv*ratio_pinv*cos(linv*twopi*(l1*x+l2*y*ratio_pinv));
  DOUBLE c = GravInt2p1i(l1,l2,z,linv,ewald_dzeta,ratio_pinv,ewc1,ewc2);
  return(b*c);
}



//=================================================================================================
//  Ewald::AccLong2p1iPer
//  long-range contributions to acceleration in periodic directions and in the 2p1i case
//=================================================================================================
template <int ndim>
DOUBLE Ewald<ndim>::AccLong2p1iPer
 (DOUBLE ewald_dzeta, 				///<
  DOUBLE linv, 					///<
  DOUBLE ratio_pinv, 				///<
  DOUBLE x, 					///<
  DOUBLE y,					///<
  DOUBLE z, 					///<
  int l1, 					///<
  int l2, 					///<
  DOUBLE ewc1, 					///<
  DOUBLE ewc2)					///<
{
  DOUBLE a = twopi*ratio_pinv*pow(linv,2)*sin(linv*twopi*(l1*x+l2*y*ratio_pinv));
  DOUBLE b = GravInt2p1i(l1,l2,z,linv,ewald_dzeta,ratio_pinv,ewc1,ewc2);
  return(a*b);
}



//=================================================================================================
//  Ewald::AccLong2p1iIso
//  long-range contributions to acceleration in the isolated direction and in the 2p1i case
//=================================================================================================
template <int ndim>
DOUBLE Ewald<ndim>::AccLong2p1iIso
 (DOUBLE ewald_dzeta, 				///<
  DOUBLE linv, 					///<
  DOUBLE ratio_pinv, 				///<
  DOUBLE x, 					///<
  DOUBLE y,					///<
  DOUBLE z, 					///<
  int l1, 					///<
  int l2, 					///<
  DOUBLE ewc1, 					///<
  DOUBLE ewc2)					///<
{
  DOUBLE a = -pi*pow(linv,2)*ratio_pinv*cos(linv*twopi*(l1*x+l2*y*ratio_pinv));
  DOUBLE b = DerGravInt2p1i(l1,l2,z,linv,ewald_dzeta,ratio_pinv,ewc1,ewc2);
  return(a*b);
}



//=================================================================================================
//  Ewald::AccLong3pPer
//  long-range contributions to acceleration for 3p case
//=================================================================================================
template <int ndim>
DOUBLE Ewald<ndim>::AccLong3pPer
 (int l, 					///<
  DOUBLE linv_acc, 				///<
  DOUBLE ewald_dzeta, 				///<
  DOUBLE ewc3, 					///<
  DOUBLE ewc4)					///<
{
  DOUBLE a = l * linv_acc*exp(-ewald_dzeta*ewc3)*sin(ewc4)/ewc3;
  return a;
}



// Create template class instances of the NbodySimulation object for each dimension (1, 2 and 3)
template class Ewald<1>;
template class Ewald<2>;
template class Ewald<3>;
