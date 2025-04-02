/*
 *  mc_cpp : A Molecular Monte Carlo simulations software.
 *  Copyright (C) 2013  Florent Hedin
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef FFIELD_MDBAS_VECT_H
#define FFIELD_MDBAS_VECT_H

#ifdef VECTORCLASS_EXPERIMENTAL

#include "FField.hpp"

class FField_MDBAS_VECT : public FField
{
public:
  FField_MDBAS_VECT(AtomList& _at_List, PerConditions& _pbc, Ensemble& _ens,
               std::string _cutMode="switch", double _ctoff=12.0, double _cton=10.0, double _dcut=2.0);
  ~FField_MDBAS_VECT();
  
  virtual double getE();
  virtual void getE(double ener[10]);
  
  virtual void askListUpdate(int st);
  
protected:

  virtual double getEtot();
  virtual double getEswitch();
  
  virtual void computeNonBonded_full();
  virtual void computeNonBonded14();
  
  virtual void computeNonBonded_switch();
  virtual void computeNonBonded14_switch();
  
  inline double computeEelec(const double qi, const double qj, const double rt);
  inline double computeEvdw(const double epsi, const double epsj, const double sigi,
                            const double sigj, const double rt);
  
  virtual void computeEbond();
  virtual void computeEang();
  virtual void computeEub();
  virtual void computeEdihe();
  virtual void computeEimpr();

};

#endif

#endif // FFIELD_MDBAS_VECT_H
