//=============================================================================
//  Diagnostics.h
//  ..
//
//  This file is part of GANDALF :
//  Graphical Astrophysics code for N-body Dynamics and Lagrangian Fluids
//  https://github.com/gandalfcode/gandalf
//  Contact : gandalfcode@gmail.com
//
//  Copyright (C) 2013  D. A. Hubber, G Rosotti
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
//=============================================================================


#ifndef _DIAGNOSTICS__H
#define _DIAGNOSTICS__H

#include "Precision.h"


//=============================================================================
//  Structure Diagnostics
/// \brief  Structure containing snapshot of current diagnostic quantities.
/// \author D. A. Hubber, G. Rosotti
/// \date   03/04/2013
//=============================================================================
template <int ndim>
struct Diagnostics {
  DOUBLE Eerror;                    ///< Total energy error
  DOUBLE Etot;                      ///< Total energy
  DOUBLE utot;                      ///< Total thermal energy
  DOUBLE ketot;                     ///< Total kinetic energy
  DOUBLE gpetot;                    ///< Total grav. potential energy
  DOUBLE mtot;                      ///< Total mass in simulation
  DOUBLE mom[ndim];                 ///< Total momentum vector
  DOUBLE angmom[3];                 ///< Total angular momentum vector
  DOUBLE force[ndim];               ///< Net force
  DOUBLE force_grav[ndim];          ///< Net gravitational force
  DOUBLE rcom[ndim];                ///< Position of centre of mass
  DOUBLE vcom[ndim];                ///< Velocity of centre of mass
};

#endif
