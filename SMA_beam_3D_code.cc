/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2000 - 2020 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Wolfgang Bangerth, University of Heidelberg, 2000
 * Modified by Atanu Banerjee, IIT Guwhati, implementing SMA constitutive model
 */



#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/geometry_info.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/sundials/sunlinsol_wrapper.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_dgq.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q_eulerian.h>

#include <fstream>
#include <iostream>

namespace Step8
{
using namespace dealii;

const double Tref = 300.; 		// Kelvin (reference temp of SMA)
const double Tmax = 325.;		// Kelvin (max temp)
const double Pmax = 250./(1e-2*1.5e-3);	// MPa  (max. stress)
const double total_time = 2.;		// total time for simulation
const double time_step = 0.005;
const int n_steps = std::round(total_time / time_step);

struct PointHistory
{
	SymmetricTensor<4,3> continuum_moduli_structural;
	SymmetricTensor<2,3> continuum_moduli_thermal;
	SymmetricTensor<2,3> old_stress;			// stress after load step converged at t_n
	SymmetricTensor<2,3>  stress;				// stress at every NR iteration after material model converged
	SymmetricTensor<2,3>  old_strain;
	SymmetricTensor<2,3>  old_t_strain;
	SymmetricTensor<2,3>  t_strain_r;
	SymmetricTensor<2,3>  old_lambda;
	double old_xi;
	double old_temperature;
	int old_transformation_status;
	int loading_status_0_iter;

};

template <int dim>
SymmetricTensor<4,dim>
get_stress_strain_tensor_plane_strain (const double &nu)
{
	const double E =1.;
	//const double lambda=(E*nu/((1+nu)*(1-2*nu))), mu=E/(2*(1+nu));
	SymmetricTensor<4,dim> tmp;
	for (unsigned int i=0; i<dim; ++i)
		for (unsigned int j=0; j<dim; ++j)
			for (unsigned int k=0; k<dim; ++k)
				for (unsigned int l=0; l<dim; ++l)
					tmp[i][j][k][l] = (((i==k) && (j==l) ? E/(2*(1+nu)) : 0.0) +
							((i==l) && (j==k) ? E/(2*(1+nu)) : 0.0) +
							((i==j) && (k==l) ? (E*nu/((1+nu)*(1-2*nu))) : 0.0));

	//tmp = lambda * outer_product(unit_symmetric_tensor<dim>(), unit_symmetric_tensor<dim>())+ 2.*mu*identity_tensor<dim>();
	return tmp;
}

template <int dim>
SymmetricTensor<2,dim> get_thermal_expansion_coefficient (const double alpha)
{
	SymmetricTensor<2,dim> tmp = alpha * unit_symmetric_tensor<dim>();
	return tmp;
}

template <int dim>
inline SymmetricTensor<2,dim> get_linear_strain (const FEValues<dim> &fe_values,
		const unsigned int   shape_func,
		const unsigned int   q_point)
{
	SymmetricTensor<2, dim> tmp;
	for (unsigned int i=0; i<dim; ++i)
		for (unsigned int j=0; j<dim; ++j)
			tmp[i][j]
				   = (fe_values.shape_grad_component (shape_func,q_point,i)[j] +
						   fe_values.shape_grad_component (shape_func,q_point,j)[i]
				   )/2.;
	return tmp;
}



template <int dim>
inline SymmetricTensor<2,dim> get_linear_strain (const std::vector<Tensor<1,dim> > &grad)
{
	Assert (grad.size() == dim, ExcInternalError());
	SymmetricTensor<2,dim> strain;

	for (unsigned int i=0; i<dim; ++i)
		for (unsigned int j=0; j<dim; ++j)
			strain[i][j] = (grad[i][j] + grad[j][i]) / 2.;
	return strain;
}

template<int dim>
SymmetricTensor<2,3> convert_symmetric_tensor_2d_to_3d(const SymmetricTensor<2,dim>  &symmetric_tensor)
{
	SymmetricTensor<2,3>  symmetric_tensor3d;
	//symmetric_tensor3d =0.;

	for (unsigned int i = 0; i < dim; ++i)
		for (unsigned int j = i; j < dim; ++j)
			symmetric_tensor3d [i][j] = symmetric_tensor [i][j];

	return symmetric_tensor3d;
}

template<int dim>
SymmetricTensor<2,dim> convert_symmetric_tensor_3d_to_2d(SymmetricTensor<2,3>  symmetric_tensor3d)
{
	SymmetricTensor<2,dim>  symmetric_tensor;
	// symmetric_tensor=0.;
	for (unsigned int i = 0; i < dim; ++i)
		for (unsigned int j = i; j < dim; ++j)
			symmetric_tensor [i][j] = symmetric_tensor3d [i][j];

	return symmetric_tensor;
}

template<int dim>
SymmetricTensor<4,dim> convert_symmetric_tensor_3d_to_2d(SymmetricTensor<4,3>  symmetric_tensor3d)
{
	SymmetricTensor<4,dim>  symmetric_tensor;
	symmetric_tensor=0.;
	for (unsigned int i = 0; i < dim; ++i)
		for (unsigned int j = 0; j < dim; ++j)
			for (unsigned int k = 0; k < dim; ++k)
				for (unsigned int l = 0; l < dim; ++l){
					symmetric_tensor[i][j][k][l] = symmetric_tensor3d[i][j][k][l];}

	return symmetric_tensor;
}


template <int dim>
class SMAConstitutiveLaw
{
public:
	SMAConstitutiveLaw (const int &problem_type /* plane stress=0; plane_strain=1; general=2*/);
	SymmetricTensor<4,dim> get_s_inv(const double &xi) const;

	//SymmetricTensor<4,dim> get_delta_s (void) const;
	SymmetricTensor<2,dim> get_alpha(const double &xi) const;

	//SymmetricTensor<2,dim> get_delta_alpha(void) const;
	double get_delf_delxi(const double &xi, const int &transformation) const;

	double get_delphi_delxi(const double &xi, const int &transformation) const;

	SymmetricTensor<2,dim> get_lambda(const SymmetricTensor<2,dim> &tensor, const int &transformation) const;

	double get_phi(const SymmetricTensor<2,dim> &stress, const double &temperature, const double &xi, const int &transformation,
			const SymmetricTensor<2,dim> &t_strain_r) const;

	double get_delta_psi(const SymmetricTensor<2,dim> &stress, const double &temperature, const PointHistory &point_history_at_q) const;

	bool call_convex_cutting(const SymmetricTensor<2,dim> &strain, const double &temperature,
			const PointHistory &point_history_at_q, const unsigned int &nr_counter, SymmetricTensor<2,dim> &stress,
			SymmetricTensor<4,dim> &continuum_moduli_structural, SymmetricTensor<2,dim> &continuum_moduli_thermal,
			SymmetricTensor<2,dim> &lambda, SymmetricTensor<2,dim> &t_strain,
			SymmetricTensor<2,dim> &t_strain_r, double &xi, int &transformation_status, int &loading_status) const;


private:
	const int problem_type, n_stress_strain;
	const double Af_0, As_0, Ms_0, Mf_0, c_A, c_M, H_max, E_M, E_A, nu, alpha_M, alpha_A, T_0, dE, dalpha, dc, tol_l, tol_h;
	const SymmetricTensor<4,dim> c_inv_perunit_E, delta_S;
	const SymmetricTensor<2,dim> delta_Alpha;
	const int hardening_model;
	const double rodelta_s0, rodelta_u0, rodelta_c, rob_M, rob_A, mu_1, mu_2, ac_M, ac_A, n_1, n_2, n_3, n_4, a_1, a_2, a_3, const_Y;
};

template<int dim>
SMAConstitutiveLaw<dim>::SMAConstitutiveLaw(const int &problem_type)
: problem_type(problem_type)
, n_stress_strain(6)
, Af_0(281.6)
, As_0(272.7)
, Ms_0(254.9)
, Mf_0(238.8)
, c_A(8.4e6)
, c_M(8.4e6)
, H_max(0.05)
, E_M(30.e9)
, E_A(72.e9)
, nu(0.42)
, alpha_M(22.e-6)
, alpha_A(22.e-6)
, T_0(Tref)
, dE(E_M-E_A)
, dalpha(alpha_M-alpha_A)
, dc((1./E_M)-(1./E_A))
, tol_l(1e-3)
, tol_h(1e-8)
, c_inv_perunit_E(get_stress_strain_tensor_plane_strain<dim>(nu))		// stiffness tensor per unit E
, delta_S(dc*invert(c_inv_perunit_E))									// delta compliance tensor
, delta_Alpha(dalpha*unit_symmetric_tensor<dim>())					// delta alpha tensor
// Polynomial hardening model
, hardening_model(10)
, rodelta_s0(-c_A*H_max)
, rodelta_u0(rodelta_s0*0.5*(Ms_0+Af_0))
, rodelta_c(0.)
, rob_M(-rodelta_s0*(Ms_0-Mf_0))
, rob_A(-rodelta_s0*(Af_0-As_0))
, mu_1(0.5*rodelta_s0*(Ms_0+Af_0))		// mu_1+-rodelta_u0
, mu_2(0.25*(rob_A-rob_M))
, ac_M(0.)	// not used
, ac_A(0.)	// not used
, n_1(0.)	// not used
, n_2(0.)	// not used
, n_3(0.)	// not used
, n_4(0.)	// not used
, a_1(0.)	// not used
, a_2(0.)	// not used
, a_3(0.)	// not used
, const_Y(0.25*rodelta_s0*(Mf_0+Ms_0-Af_0-As_0))
// Smooth Hadening model Machado
/*, hardening_model(20)
, rodelta_s0(-c_A*H_max)
, rodelta_u0(rodelta_s0*0.5*(Ms_0+Af_0))
, rodelta_c(0.)
, rob_M(-rodelta_s0*(Ms_0-Mf_0))
, rob_A(-rodelta_s0*(Af_0-As_0))
, mu_1(0.)	// not used
, mu_2(0.)	// not used
, ac_M(0.)	// not used
, ac_A(0.)	 // not used
, n_1(1.)
, n_2(1.)
, n_3(1.)
, n_4(1.)
, a_1 (0.)
, a_2 (0.)
, a_3 (0.)
, const_Y(0.5*rodelta_s0*(Ms_0-Af_0))*/
// cosine hardening model
/*, hardening_model(30)
, rodelta_s0(-c_A*H_max)
, rodelta_u0(0.5*rodelta_s0*(Ms_0+Af_0))
, rodelta_c(0.)
, rob_M(0.) 		// not used
, rob_A(0.)  		// not used
, mu_1(0.5*rodelta_s0*(Ms_0+Af_0)) 				// mu_1+rodelta_u0
, mu_2(0.25*rodelta_s0*(Ms_0-Mf_0+As_0-Af_0))
, ac_M(numbers::PI/(Ms_0-Mf_0))
, ac_A(numbers::PI/(Af_0-As_0))
, n_1(0.) 		// not used
, n_2(0.) 		// not used
, n_3(0.) 		// not used
, n_4(0.) 		// not used
, a_1 (0.) 		// not used
, a_2 (0.) 		// not used
, a_3 (0.) 		// not used
, const_Y(0.25*rodelta_s0*(Ms_0+Mf_0-As_0-Af_0))*/
// Lagoudas (2012) smooth hardening model
/*, hardening_model(40)
, rodelta_s0(-c_A*H_max)
, rodelta_u0(rodelta_s0*0.5*(Ms_0+Af_0))
, rodelta_c(0.)
, rob_M(0.) // not used
, rob_A(0.) // not used
, mu_1(0.)	// not used
, mu_2(0.)	// not used
, ac_M(0.)	// not used
, ac_A(0.)	 // not used
, n_1(0.5)
, n_2(0.5)
, n_3(0.5)
, n_4(0.5)
, a_1 (rodelta_s0* (Mf_0-Ms_0))
, a_2 (rodelta_s0* (As_0-Af_0))
, a_3 (-0.25* (1.+ 1./(n_1+1.)- 1./(n_2+1.))+0.25* a_2* (1.+ 1./(n_3+1.)- 1./(n_4+1.)))
, const_Y (0.5* rodelta_s0* (Ms_0-Af_0) -a_3)*/

{}

template<int dim>
SymmetricTensor<4,dim> SMAConstitutiveLaw<dim>::get_s_inv (const double &xi) const
{
	SymmetricTensor<4,dim> tmp;
	const double E = E_A + xi * dE;

	switch(problem_type){
	case 0:		// For plane stress case
	{
		break;
	}
	case 1:		// for Plane strain case
	{
		tmp = E * c_inv_perunit_E;
		break;
	}
	case 2:		// for 3D case
	{
		tmp = E * c_inv_perunit_E;
		break;
	}
	}
	return tmp;
}

/* template<int dim>
 SymmetricTensor<4,dim> SMAConstitutiveLaw<dim>::get_delta_s (void) const
  {
	 SymmetricTensor<4,dim> tmp;
	 const double dS=(1./E_M)-(1./E_A);
	 switch(problem_type)
	 {
	 case 0:		// Plane stress case
	 {
		 break;
	 }

	 case 1:		// Plane strain case
	 {
		 tmp=dS*invert(c_inv_perunit_E);
		 break;
	 }
	 case 2:		// 3D case
	 {
		 tmp=dS*invert(c_inv_perunit_E);
		 break;
	 }
	 }
	 return tmp;
  }*/

template<int dim>
SymmetricTensor<2,dim> SMAConstitutiveLaw<dim>::get_alpha(const double &xi) const
{
	const double alpha=alpha_A + xi * dalpha;
	SymmetricTensor<2,dim> tmp;
	switch(problem_type){

	case 0: // plane stress
	{
		break;
	}
	case 1:		// plane strain
	{
		tmp = alpha * unit_symmetric_tensor<dim>();
		break;
	}
	case 2:		// 3D case
	{
		tmp = alpha * unit_symmetric_tensor<dim>();
		break;
	}
	}
	return tmp;
}

/*
 template<int dim>
SymmetricTensor<2,dim> SMAConstitutiveLaw<dim>::get_delta_alpha(void) const
{
	SymmetricTensor<2,dim> tmp;
		switch(problem_type){

		case 0: // plane stress
		{
			break;
		}
		case 1:		// plane strain
		{
			tmp = dalpha * unit_symmetric_tensor<dim>();
			break;
		}
		case 2:		// 3D case
		{
			tmp = dalpha * unit_symmetric_tensor<dim>();
			break;
		}
		}
	return tmp;
}*/

template<int dim>
double SMAConstitutiveLaw<dim>::get_delf_delxi(const double &xi, const int &transformation) const
{
	double y=0.;
	const double delta = tol_h;
	switch(hardening_model)
	{
	case 10:		// polynomial hardening model
	{
		switch (transformation){
		case 1:
		{
			y = rob_M * xi + mu_1+mu_2;
			break;
		}
		case -1:
		{
			y = rob_A * xi + mu_1-mu_2;
			break;
		}
		}
		break;
	}
	case 20:  //Smooth hardening model
	{
		switch (transformation){
		case 1:
		{
			const double n_1=0.17, n_2=0.27;
			//y=rodelta_u0 + 0.5 * rob_M * (1.+ std::pow(xi, n_1) - std::pow((1-xi), n_2));

			y=rodelta_u0+0.5*rob_M*(1.+ std::pow(std::pow(xi, (1./n_1))*std::pow((delta + xi),(1. - (1./n_1))),n_1)
					- std::pow(std::pow((1.-xi), (1./n_2))*std::pow((delta + 1.-xi),(1. - (1./n_2))),n_2));
			break;
		}
		case -1:
		{
			const double n_3=0.25, n_4=0.35;
			//y=rodelta_u0 + 0.5 * rob_A * (1.+ std::pow(xi, n_3) - std::pow((1-xi), n_4));

			y=rodelta_u0+0.5*rob_A*(1.+ std::pow(std::pow(xi, (1./n_3))*std::pow((delta + xi),(1. - (1./n_3))),n_3)
					- std::pow(std::pow((1.-xi), (1./n_4))*std::pow((delta + 1.-xi),(1. - (1./n_4))),n_4));
			break;
		}
		}
		break;
	}
	case 30:  //Cosine model
	{
		switch (transformation){
		case 1:
		{
			y=-(rodelta_s0/ac_M)*(numbers::PI - std::acos(2*xi-1.)) + mu_1 + mu_2;
			break;
		}
		case -1:
		{
			y=-(rodelta_s0/ac_A)*(numbers::PI - std::acos(2*xi-1.)) + mu_1 - mu_2;
			break;
		}
		}
		break;
	}
	case 40:  //Smooth hardening Lagoudas (2012)
	{
		switch (transformation){
		case 1:
		{
 			y= rodelta_u0+ 0.5*a_1*(1. + std::pow(xi,n_1) - std::pow((1.-xi),n_2)) + a_3;
			//std::cout<< "get_delf_delxi=" << y <<std::endl;

 			break;
		}
		case -1:
		{
 			y=  rodelta_u0+ 0.5*a_2*(1. + std::pow(xi,n_3) - std::pow((1.-xi),n_4)) - a_3;
			break;
		}
		}
		break;
	}
	}

	//std::cout<< "xi = " << xi << " and delf_delxi= " << y << std::endl;
	return y;		// returns rodelta_u0+(del_f/del_xi)
}

template<int dim>
double SMAConstitutiveLaw<dim>::get_delphi_delxi(const double &xi, const int &transformation) const
{
	double y=0.;
	const double delta = tol_h;
	switch(hardening_model)
	{
	case 10:
	{
		switch (transformation){
		case 1:
		{
			y = -rob_M ;
			break;
		}
		case -1:
		{
			y = rob_A ;
			break;
		}
		}
		break;
	}
	case 20:  //Smooth hardening model
	{
		switch (transformation){
		case 1:
		{
			const double n_1=0.17, n_2=0.27;
			//y= - 0.5 * rob_M *( n_1 * (std::pow(xi, n_1-1)) - n_2 * (std::pow(1-xi , n_2-1)));

			y=-0.5*rob_M*(((delta + n_1*xi)*std::pow(std::pow(xi,(1./n_1))*std::pow((delta + xi),((n_1 - 1.)/n_1)),n_1))/(xi*(delta + xi))
					-(std::pow((std::pow((1 - xi),(1/n_2))* std::pow((delta - xi + 1.),((n_2 - 1.)/n_2))),(n_2 - 1.))
							*std::pow((1 - xi),(1./n_2))*(delta + n_2 - n_2*xi))/((xi - 1.)*std::pow((delta - xi + 1.),(1/n_2))));
			break;
		}
		case -1:
		{
			const double n_3=0.25, n_4=0.35;
			//y= 0.5 * rob_A *( n_3 * (std::pow(xi , n_3-1)) - n_4 * (std::pow(1-xi, n_4-1)) );

			y=0.5*rob_A*(((delta + n_3*xi)*std::pow(std::pow(xi,(1./n_3))*std::pow((delta + xi),((n_3 - 1.)/n_3)),n_3))/(xi*(delta + xi))
					-(std::pow((std::pow((1 - xi),(1/n_4))* std::pow((delta - xi + 1.),((n_4 - 1.)/n_4))),(n_4 - 1.))
							*std::pow((1 - xi),(1./n_4))*(delta + n_4 - n_4*xi))/((xi - 1.)*std::pow((delta - xi + 1.),(1/n_4))));

			break;
		}
		}
		break;
	}
	case 30:  //Cosine model
	{
		switch (transformation){
		case 1:
		{
			y= (rodelta_s0/ac_M)/std::sqrt((xi+delta)*(1.-(xi+delta)));
			break;
		}
		case -1:
		{
			y= -(rodelta_s0/ac_A)/std::sqrt((xi+delta)*(1.-(xi+delta)));
			break;
		}
		}
		break;
	}
	case 40:  //Smooth hardening Lagoudas (2012)
	{
		switch (transformation){
		case 1:
		{
 			y= -0.5* a_1* (n_1*std::pow(xi, (n_1-1.)) + n_2*std::pow((1.-xi), (n_2-1.))) ;
			//std::cout<< "get_delphi_delxi=" << y <<std::endl;

			break;
		}
		case -1:
		{
 			y= 0.5* a_2* (n_3*std::pow(xi, (n_3-1.)) + n_4*std::pow((1.-xi), (n_4-1.))) ;
			break;
		}
		}
		break;
	}
	}
	//std::cout<< "denominator = " << std::sqrt((xi+delta)*(1-xi+delta)) << std::endl;
	//std::cout<< "xi = " << xi << " and del2f_delxi2= " << y << std::endl;
	return y;
}

template<int dim>
SymmetricTensor<2,dim> SMAConstitutiveLaw<dim>::get_lambda(const SymmetricTensor<2,dim> &tensor_, const int &transformation) const
{
	SymmetricTensor<2,dim> tmp_tensor;
	switch (transformation){
	case 1:						// forward transformation --> vec = stress vector
	{
		const double coeff = (3./2.) * H_max;
		const SymmetricTensor<2,dim> dev_stress = deviator(tensor_);

		//std::cout<< "trace= " << trace(dev_stress) << std::endl;

		const double dev_stress_norm = dev_stress.norm();
		//std::cout << "sigma_dev_norm " << vec_norm << std::endl;
		if (dev_stress_norm < 1e-6)
		{
			tmp_tensor= coeff * unit_symmetric_tensor<dim>();
		}
		else
		{
			tmp_tensor = coeff * dev_stress / (std::sqrt(3./2.)*dev_stress_norm);
		}
		break;
	}
	case -1:						// reverse transformation --> vec = transformation strain at reversal
	{
		tmp_tensor = H_max * tensor_/(std::sqrt(2./3.)*tensor_.norm());
		break;
	}
	}
	//std::cout << "stress " << tensor_ << std::endl;
	//std::cout << "lambda " << tmp_tensor << std::endl;
	return tmp_tensor;
}

template<int dim>
double SMAConstitutiveLaw<dim>::get_phi(const SymmetricTensor<2,dim> &stress, const double &temperature, const double &xi,
		const int &transformation, const SymmetricTensor<2,dim> &t_strain_r) const
{
	double phi =0.;
	//const SymmetricTensor<4,dim> delta_S = get_delta_s();
	//const SymmetricTensor<2,dim> delta_Alpha = get_delta_alpha();
	const double hardening_fn_value = get_delf_delxi(xi, transformation);

	switch (transformation)
	{
	case 1:
	{
		const SymmetricTensor<2,dim> lambda = get_lambda(stress, transformation);

		//std::cout << "lambda " << lambda[1][1] << std::endl;
		//std::cout << "0.5* delta_s.matrix_scalar_product(stress, stress) " << 0.5* delta_s.matrix_scalar_product(stress, stress) << std::endl;
		//std::cout << "const_Y " << const_Y << std::endl;
		phi = ((stress * lambda) + 0.5* (stress* delta_S* stress) + (stress * delta_Alpha) *(temperature - T_0)
				-rodelta_c*((temperature - T_0) - temperature * std::log(temperature/T_0)) + rodelta_s0* temperature
				- hardening_fn_value) - const_Y;
 		break;
	}
	case -1:
	{
		const SymmetricTensor<2,dim> lambda = get_lambda(t_strain_r, transformation);
		phi = -((stress * lambda) + 0.5* (stress * delta_S* stress) + (stress * delta_Alpha) *(temperature - T_0)
				-rodelta_c*((temperature - T_0) - temperature * std::log(temperature/T_0)) + rodelta_s0* temperature
				- hardening_fn_value) - const_Y;
		break;
	}
	}
	return phi;
}

template<int dim>
double SMAConstitutiveLaw<dim>::get_delta_psi(const SymmetricTensor<2,dim> &stress, const double &temperature,
		const PointHistory &point_history_at_q) const
{
	const SymmetricTensor<2,dim> stress_0=point_history_at_q.old_stress;
	const SymmetricTensor<2,dim> lambda_0=point_history_at_q.old_lambda;
	const double temp_0=point_history_at_q.old_temperature;

	//const FullMatrix<double> delta_S = get_delta_s();
	//const Vector<double> delta_Alpha = get_delta_alpha();

	double tmp = ((stress * lambda_0) + 0.5* (stress * delta_S* stress) + (stress * delta_Alpha) *(temperature - T_0) + rodelta_s0* temperature)
					-((stress_0 * lambda_0) + 0.5* (stress_0* delta_S* stress_0) + (stress_0 * delta_Alpha) *(temp_0 - T_0) + rodelta_s0* temp_0 );
	return tmp;
}

template<int dim>
bool SMAConstitutiveLaw<dim>::call_convex_cutting(const SymmetricTensor<2,dim> &strain, const double &temperature,
		const PointHistory &point_history_at_q, const unsigned int &nr_counter, SymmetricTensor<2,dim> &stress,
		SymmetricTensor<4,dim> &continuum_moduli_structural, SymmetricTensor<2,dim> &continuum_moduli_thermal, SymmetricTensor<2,dim> &lambda, SymmetricTensor<2,dim> &t_strain,
		SymmetricTensor<2,dim> &t_strain_r, double &xi, int &transformation_status, int &loading_status) const
{

	//SymmetricTensor<2,dim> stress_iter=point_history_at_q.old_stress;
	SymmetricTensor<2,dim> lambda_iter=point_history_at_q.old_lambda;
	//const double temp_0=point_history_at_q.old_temperature;
	double xi_iter=point_history_at_q.old_xi;
	SymmetricTensor<2,dim> t_strain_iter=point_history_at_q.old_t_strain;
	const SymmetricTensor<2,dim> t_strain_r0=point_history_at_q.t_strain_r;
	const int loading_status_0_iter=point_history_at_q.loading_status_0_iter;

	SymmetricTensor<4,dim> s_inv_iter = get_s_inv(xi_iter);
	SymmetricTensor<2,dim> alpha_iter = get_alpha(xi_iter);

	SymmetricTensor<2,dim> stress_iter;

	stress_iter = s_inv_iter*(strain - alpha_iter*(temperature -T_0) - t_strain_iter);
	if (nr_counter == 0)
		loading_status = (get_delta_psi(stress_iter, temperature, point_history_at_q) > 0 ? 1 : -1);
	else
		loading_status = loading_status_0_iter;  // updated based on first N-R iteration

	double phi = get_phi(stress_iter, temperature, xi_iter, loading_status, t_strain_r0);
	//std::cout << "load_status is " << loading_status << " and initial phi is " << phi <<std::endl;

	unsigned int matr_iter = 0, total_iter=50;

	while (true)
	{
		if (matr_iter == 0 && phi < 0.)
		{transformation_status = 0;
		break;}
		else if (loading_status== 1 && xi_iter > (1-tol_l))
		{transformation_status = 0;
		break;}
		else if (loading_status== -1 && xi_iter < (tol_l))
		{transformation_status = 0;
		break;}

		// else material iteration starts

		transformation_status = (loading_status == 1 ? 1 : -1);
		const double sgn_xi_dot = (transformation_status == 1 ? 1.0 : -1.0);

		SymmetricTensor<2,dim> del_phi_del_sig = sgn_xi_dot* (delta_S* stress_iter + delta_Alpha* (temperature -T_0) + lambda_iter);
		double del_phi_del_xi = get_delphi_delxi(xi_iter, transformation_status);
		double delta_xi = phi/ (sgn_xi_dot* (del_phi_del_sig* s_inv_iter* del_phi_del_sig) - del_phi_del_xi);
		const double xi_tmp = xi_iter + delta_xi;
		if (transformation_status == 1 && xi_tmp > (1-tol_l))
		{
			delta_xi = (1-0.5*tol_l) - xi_iter;
			xi_iter = (1-0.5*tol_l);
			phi = 0.;					// forcefully quitting the material loop (fine for very small load step)
		}
		else if (transformation_status == -1 && xi_tmp < tol_l)
		{
			delta_xi = (0.5*tol_l) - xi_iter;
			xi_iter = 0.5*tol_l;
			phi = 0.;					// forcefully quitting the material loop (fine for very small load step)
		}
		else
			xi_iter = xi_tmp;

		t_strain_iter += delta_xi* lambda_iter; 	// basically +=delta_t_strain;

		s_inv_iter = get_s_inv(xi_iter);
		alpha_iter = get_alpha(xi_iter);

		stress_iter = s_inv_iter*(strain - alpha_iter*(temperature -T_0) - t_strain_iter);

		if (transformation_status == 1)
			lambda_iter = get_lambda(stress_iter, transformation_status);
		else
			lambda_iter = get_lambda(t_strain_r0, transformation_status);

		phi = get_phi(stress_iter, temperature, xi_iter, transformation_status, t_strain_r0);

		if ( std::fabs(phi) < tol_l || std::fabs(delta_xi) < tol_h || matr_iter > total_iter)
		{
			//calculate tangent moduli
			del_phi_del_sig = sgn_xi_dot* (delta_S* stress_iter + delta_Alpha* (temperature -T_0) + lambda_iter);
			del_phi_del_xi = get_delphi_delxi(xi_iter, transformation_status);
			double del_phi_del_T = sgn_xi_dot * (stress_iter * delta_Alpha + rodelta_c* std::log(temperature/T_0) + rodelta_s0);
			const double a_ =  sgn_xi_dot * (del_phi_del_sig* s_inv_iter* del_phi_del_sig) - del_phi_del_xi;
			SymmetricTensor<2,dim> tmp_tensor = s_inv_iter * del_phi_del_sig;
			continuum_moduli_structural = s_inv_iter - sgn_xi_dot * (1./a_)* outer_product(tmp_tensor, tmp_tensor);
			continuum_moduli_thermal = -(continuum_moduli_structural * alpha_iter) - sgn_xi_dot * (1./a_) * del_phi_del_T* tmp_tensor;

			stress = stress_iter;
			lambda = lambda_iter;
			t_strain = t_strain_iter;
			t_strain_r = (transformation_status == 1 ? t_strain_iter : t_strain_r0);
			xi = xi_iter;
			break;
		}
		else
			matr_iter++;

	}
	/*std::cout << "loading status is " << loading_status << " and final phi = " << phi
			<< " and transformation status is " << transformation_status <<
			" and iteration counter "<< matr_iter << "final xi " << xi_iter << std::endl;*/
	if (transformation_status == 0)
	{
		//std::cout << "currently elastic" << std::endl;
		continuum_moduli_structural = s_inv_iter;
		stress = stress_iter;
		lambda = lambda_iter;
		t_strain_r = t_strain_r0;
		t_strain = t_strain_iter;
		xi = xi_iter;
		// may compute thermal moduli
	}

	const bool converged = ((matr_iter > total_iter) ? false : true);
	return converged;	// if converged
}



template <int dim>
void right_hand_side(const std::vector<Point<dim>> &points, const double &time,
		std::vector<Tensor<1, dim>> &values, Vector<double> &temp)
{
	Assert(values.size() == points.size(),
			ExcDimensionMismatch(values.size(), points.size()));
	Assert(dim >= 2, ExcNotImplemented());

	//Point<dim> point_1, point_2;
	// point_1(0) = 0.5;
	//point_2(0) = -0.5;
	//const double T_max = 325.;
	temp = 0.;
	for (unsigned int point_n = 0; point_n < points.size(); ++point_n)
	{
		values[point_n][0] = 0.0;
		values[point_n][1] = 0.0;

		if (time <= 2.)
		{
			temp[point_n] = Tref;		// same as T_0
		}
		else if (time > 2. && time <= 3.)
			temp[point_n] = Tref + (Tmax - Tref) * (time- 2.)/(3. - 2.);
		else
			temp[point_n] = Tref + (Tmax - Tref) * (4. - time)/(4. - 3.);

		/*
        if (((points[point_n] - point_1).norm_square() < 0.2 * 0.2) ||
            ((points[point_n] - point_2).norm_square() < 0.2 * 0.2))
          values[point_n][0] = 1.0;
        else
          values[point_n][0] = 0.0;

        if (points[point_n].norm_square() < 0.2 * 0.2)
          values[point_n][1] = 1.0;
        else
          values[point_n][1] = 0.0;*/
	}
}

template <int dim>
class NeumannBoundary : public Function<dim>
{
public:
	NeumannBoundary(const double time, const double present_time_step);

	virtual void vector_value(const Point<dim> &p,
			Vector<double> &  values) const override;
	virtual void
	vector_value_list(const std::vector<Point<dim>> &points,
			std::vector<Vector<double>> &  value_list) const override;
private:
	const double present_time, present_time_step;
};


template <int dim>
NeumannBoundary<dim>::NeumannBoundary(const double time, const double present_time_step)
: Function<dim>(dim)
  , present_time(time)
  , present_time_step(present_time_step)
  {}


template <int dim>
inline void NeumannBoundary<dim>::vector_value(const Point<dim> & /*p*/,
		Vector<double> &values) const
{
	AssertDimension(values.size(), dim);
	//const double p0   = 400.e6;				// N /per unit length
	//const double total_time =2.;
	values    = 0;
	if (present_time <= 1.)
		values(1) = Pmax *((present_time - 0.)/(1.-0.));			// total time is 1s
	else if (present_time <= 2.)
		values(1) = Pmax *((2.0 - present_time)/(2. - 1.));
	else
		values(1) = 0.;
}


template <int dim>
void NeumannBoundary<dim>::vector_value_list(
		const std::vector<Point<dim>> &points,
		std::vector<Vector<double>> &value_list) const
{
	const unsigned int n_points = points.size();

	AssertDimension(value_list.size(), n_points);

	for (unsigned int p = 0; p < n_points; ++p)
		NeumannBoundary<dim>::vector_value(points[p], value_list[p]);
}

template <int dim>
class ElasticProblem
{
public:
	ElasticProblem(const unsigned int &study_type);
	void run();

private:
	void create_mesh(/*const unsigned int cycle*/);
	void setup_system();

	void assemble_system();

	void solve();

	void refine_grid();

	void setup_quadrature_point_history();

	void setup_initial_quadrature_point_history();

	void update_quadrature_point_history(const bool &cond, const unsigned int &nr_counter);

	void call_Newton_Raphson_method();

	void output_results(const unsigned int timestep_counter) const;

	void map_scalar_qpoint_to_dof(const Vector<double> &vec, Vector<double> &avg_vec_on_cell_vertices) const;

	void map_vec_qpoint_to_dof(const unsigned int &n_components, const std::vector<Vector<double>> &vec,
			std::vector<Vector<double>> &avg_vec_on_cell_vertices) const;

	void print_conv_header(void);
	void print_conv_footer();

	Triangulation<dim> triangulation;
	DoFHandler<dim>    dof_handler;

	FESystem<dim> fe;

	AffineConstraints<double> hanging_node_constraints;
	const QGauss<dim> quadrature_formula;

	SparsityPattern      sparsity_pattern;
	SparseMatrix<double> system_matrix;
	//FullMatrix<double> system_matrix_full;

	Vector<double> solution, incremental_solution;
	std::vector<PointHistory> quadrature_point_history;			// dim has to be only 3
	//Vector<double> external_force_vector;
	//Vector<double> internal_force_vector;
	Vector<double> residue_vector;
	const unsigned int study_type, n_time_steps;
	double present_time, end_time, time_step;
	//static const double moduli, nu;
	const SMAConstitutiveLaw<3> constitutive_law_sma;				// dim has to be only 3
	const double T_initial;
	//unsigned int n_strain_components;
//};

/*template <int dim>
  const double ElasticProblem<dim> :: moduli = 70.0e9;

  template <int dim>
  const double ElasticProblem<dim> :: nu = 0.3;*/
};

template <int dim>
ElasticProblem<dim>::ElasticProblem(const unsigned int &study_type)
: dof_handler(triangulation)
, fe(FE_Q<dim>(1), dim)
, quadrature_formula(fe.degree + 1)
, study_type(study_type)											/* plane_stress=0; plane_strain=1; 3D general=2*/
, n_time_steps(n_steps)
, present_time(0.)
, end_time(total_time)
, time_step(end_time/n_time_steps)
, constitutive_law_sma(study_type)
, T_initial(Tref)
{
	//n_strain_components = (study_type == 2 ? 6 : 3); 					//plane stress or strain case=3; 3D case=6
}

template <int dim>
void ElasticProblem<dim>::create_mesh(/*const unsigned int cycle*/)
{
	//std::cout << "Cycle " << cycle << ':' << std::endl;

	//      if (cycle == 0){
	//Cantilever beam (R. Mirzaeifar, R. DesRoches, A. Yavari, K. Gall,
	//On superelastic bending of shape memory alloy
	//beams, International Journal of Solids and Structures  (2013) 1664–1680.
	const double length = 0.1;
	const double height = 0.01;
	const double thickness = 0.0015;
	std::vector< unsigned int > repetitions(3, 3); repetitions[1]= 2; repetitions[2]= 1;

	const Point<3> bottom_left(0., 0., 0.);
	const Point<3> top_right(length, height, thickness);

	GridGenerator::subdivided_hyper_rectangle(triangulation, repetitions, bottom_left, top_right);
	// set the right hand side boundary as boundary_id=3
	if (dim==2)
	{
		for(auto &face : triangulation.active_face_iterators())
			if (face->at_boundary())
			{
				if (std::fabs(face->center()[0] - 0.) < 1e-5)
					face->set_boundary_id(1);
				else if (std::fabs(face->center()[1] - 0.) < 1e-5)
					face->set_boundary_id(2);
				else if (std::fabs(face->center()[0] - length) < 1e-5)
					face->set_boundary_id(3);

			}
	}
	else if (dim==3)
	{
		for(auto &face : triangulation.active_face_iterators())
			if (face->at_boundary())
			{
				if (std::fabs(face->center()[0] - 0.) < 1e-5)
					face->set_boundary_id(1);
				else if (std::fabs(face->center()[1] - 0.) < 1e-5)
					face->set_boundary_id(2);
				else if (std::fabs(face->center()[0] - length) < 1e-5)
					face->set_boundary_id(3);
				else if (std::fabs(face->center()[2] - 0.) < 1e-5)
					face->set_boundary_id(4);
			}
	}
	for(unsigned int step =0; step < 4; step++)
	  			  {
	  				  for(auto &cell : triangulation.active_cell_iterators())
	  				  {
	  					  for(unsigned int face_number=0; face_number<GeometryInfo<2>::faces_per_cell; ++face_number)
	  					  {
	  						  if(std::fabs(cell->face(face_number)->center()[0] - (0.)) < 1e-10)
	  						  {
	  							  cell->set_refine_flag();
	  							  break;
	  						  }
	  					  }
	  				  }
	  				  triangulation.execute_coarsening_and_refinement();
	  			  }
//	triangulation.refine_global(1);
}

template <int dim>
void ElasticProblem<dim>::setup_system()
{
	dof_handler.distribute_dofs(fe);
	solution.reinit(dof_handler.n_dofs());
	incremental_solution.reinit(dof_handler.n_dofs());
	//external_force_vector.reinit(dof_handler.n_dofs());
	//internal_force_vector.reinit(dof_handler.n_dofs());
	residue_vector.reinit(dof_handler.n_dofs());
	//system_matrix_full.reinit(dof_handler.n_dofs(),dof_handler.n_dofs());

	hanging_node_constraints.clear();
	DoFTools::make_hanging_node_constraints(dof_handler, hanging_node_constraints);
	/*VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             Functions::ZeroFunction<dim>(dim),
                                             constraints); */
	hanging_node_constraints.close();

	DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
	DoFTools::make_sparsity_pattern(dof_handler,
			dsp,
			hanging_node_constraints,
			/*keep_constrained_dofs = */ false);
	sparsity_pattern.copy_from(dsp);

	system_matrix.reinit(sparsity_pattern);

}

template <int dim>
void ElasticProblem<dim>::assemble_system()
{
	//QGauss<dim> quadrature_formula(fe.degree + 1);
	QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);

	FEValues<dim> fe_values(fe,
			quadrature_formula,
			update_values | update_gradients |
			update_quadrature_points | update_JxW_values);

	FEFaceValues<dim> fe_face_values(fe,
			face_quadrature_formula,
			update_values | update_quadrature_points |
			update_normal_vectors |
			update_JxW_values);

	const unsigned int dofs_per_cell = fe.dofs_per_cell;
	const unsigned int n_q_points    = quadrature_formula.size();
	const unsigned int n_face_q_points = face_quadrature_formula.size();

	FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
	Vector<double>     cell_external_force(dofs_per_cell), cell_internal_force(dofs_per_cell), cell_residue(dofs_per_cell);
	SymmetricTensor<2,dim> stress_at_qpoint;

	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

	std::vector<Tensor<1, dim>> rhs_values(n_q_points);
	Vector<double> body_temp(n_q_points);
	const NeumannBoundary<dim> neumann_boundary(present_time, time_step);
	std::vector<Vector<double>> neumann_values(n_face_q_points, Vector<double>(dim));

	std::vector<std::vector<Tensor<1, dim>>> solution_grads(quadrature_formula.size(),
			std::vector<Tensor<1, dim>>(dim));
	//std::cout<< "residue at the beginning" << residue_vector << std::endl;
	system_matrix  = 0.;
	residue_vector =0.;
	//external_force_vector =0.;
	//internal_force_vector = 0;
	//system_matrix_full =0.;
	for (const auto &cell : dof_handler.active_cell_iterators())
	{
		cell_matrix = 0.;
		cell_external_force = 0.;
		cell_internal_force = 0.;
		cell_residue = 0.;
		PointHistory *local_quadrature_points_history = reinterpret_cast<PointHistory *>(cell->user_pointer());
		Assert (local_quadrature_points_history >= &quadrature_point_history.front(),ExcInternalError());
		Assert (local_quadrature_points_history < &quadrature_point_history.back(),ExcInternalError());

		fe_values.reinit(cell);
		//fe_values.get_function_gradients(solution, solution_grads);

		right_hand_side(fe_values.get_quadrature_points(), present_time, rhs_values, body_temp);

		for (const unsigned int q_point : fe_values.quadrature_point_indices())
		{
			const SymmetricTensor<4,dim> linearised_stress_strain_matrix=convert_symmetric_tensor_3d_to_2d<dim>(local_quadrature_points_history[q_point].continuum_moduli_structural);
			const SymmetricTensor<2,dim> linearised_thermal_moduli = convert_symmetric_tensor_3d_to_2d<dim>(local_quadrature_points_history[q_point].continuum_moduli_thermal);
			const SymmetricTensor<2,dim> stress_at_qpoint = convert_symmetric_tensor_3d_to_2d<dim>(local_quadrature_points_history[q_point].stress);
			const double temp_n = local_quadrature_points_history[q_point].old_temperature;
			// For 3D problem no need to convert, thus use the following only
			//const SymmetricTensor<4,dim> linearised_stress_strain_matrix=local_quadrature_points_history[q_point].continuum_moduli;

			for (const unsigned int i : fe_values.dof_indices())
			{
				const SymmetricTensor<2,dim> eps_phi_i = get_linear_strain (fe_values, i, q_point);
				for (const unsigned int j : fe_values.dof_indices())
				{
					const SymmetricTensor<2,dim> eps_phi_j = get_linear_strain (fe_values, j, q_point);
					cell_matrix(i,j)+= (eps_phi_i * linearised_stress_strain_matrix * eps_phi_j) * fe_values.JxW(q_point);
				}
				cell_internal_force(i) += (stress_at_qpoint * eps_phi_i + linearised_thermal_moduli * eps_phi_i * (body_temp[q_point] - temp_n)) * fe_values.JxW(q_point);
			}
		}

		/*for (const unsigned int q_point : fe_values.quadrature_point_indices())
           {
    	   // for 3D problem no need to convert
    	   SymmetricTensor<2,dim> stress_at_qpoint = convert_symmetric_tensor_3d_to_2d<dim>(local_quadrature_points_history[q_point].stress);
    	   for (const unsigned int i : fe_values.dof_indices())
    	          {
    		      	   const SymmetricTensor<2,dim> eps_phi_i = get_linear_strain (fe_values, i, q_point);
    		      	   cell_internal_force(i) += stress_at_qpoint * eps_phi_i * fe_values.JxW(q_point);
    	          }
    	  // std::cout<< "stress at Gauss point" << stress_at_qpoint << std::endl;
           }*/

		for (const unsigned int i : fe_values.dof_indices())
		{
			const unsigned int component_i = fe.system_to_component_index(i).first;
			for (const unsigned int q_point : fe_values.quadrature_point_indices())
			{
				cell_external_force(i) += ( rhs_values[q_point][component_i]
																* fe_values.shape_value(i,q_point) * fe_values.JxW(q_point));
			}
		}


		for (const auto &face : cell->face_iterators())
			if (face->at_boundary() && (face->boundary_id() == 3))
			{
				fe_face_values.reinit(cell, face);
				for (const unsigned int q_point : fe_face_values.quadrature_point_indices())
				{
					neumann_boundary.vector_value_list(fe_face_values.get_quadrature_points(), neumann_values);
					for (const unsigned int i : fe_values.dof_indices())
					{
						const unsigned int component_i = fe.system_to_component_index(i).first;
						cell_external_force(i) += ( neumann_values[q_point][component_i]
													* fe_face_values.shape_value(i,q_point) * fe_face_values.JxW(q_point));
					}
				}
			}

		cell_residue += cell_external_force;
		cell_residue -= cell_internal_force;
		//*********************************************************************************************************************

		cell->get_dof_indices(local_dof_indices);
		/*  for (unsigned int i=0; i<dofs_per_cell; ++i)
          {
            for (unsigned int j=0; j<dofs_per_cell; ++j)
              {
            	system_matrix.add (local_dof_indices[i],
                                 local_dof_indices[j],
                                 cell_matrix(i,j));
            system_matrix_full(local_dof_indices[i],local_dof_indices[j])+=cell_matrix(i,j);
              }
          }*/

		// To assemble external and internal global force vector
		/*for (unsigned int i =0; i< dofs_per_cell; i++)
        	{
        	external_force_vector(local_dof_indices[i]) +=cell_external_force(i);
        	internal_force_vector(local_dof_indices[i]) +=cell_internal_force(i);
        	}*/
		//hanging_node_constraints.distribute_local_to_global(cell_matrix, cell_external_force, local_dof_indices, system_matrix, external_force);

		hanging_node_constraints.distribute_local_to_global(cell_matrix, cell_residue, local_dof_indices, system_matrix, residue_vector);
	}

	std::map<types::global_dof_index, double> boundary_values;

	if (dim == 2)
	{
		const FEValuesExtractors::Scalar          x_component(0), y_component(1);
		VectorTools::interpolate_boundary_values(dof_handler, 1, Functions::ZeroFunction<dim>(dim), boundary_values, fe.component_mask(x_component));
		VectorTools::interpolate_boundary_values(dof_handler, 1, Functions::ZeroFunction<dim>(dim), boundary_values, fe.component_mask(y_component));
	}
	else if(dim == 3)
	{
		const FEValuesExtractors::Scalar          x_component(0), y_component(1), z_component(2);
		VectorTools::interpolate_boundary_values(dof_handler, 1, Functions::ZeroFunction<dim>(dim), boundary_values, fe.component_mask(x_component));
		VectorTools::interpolate_boundary_values(dof_handler, 1, Functions::ZeroFunction<dim>(dim), boundary_values, fe.component_mask(y_component));
		VectorTools::interpolate_boundary_values(dof_handler, 1, Functions::ZeroFunction<dim>(dim), boundary_values, fe.component_mask(z_component));
	}
	MatrixTools::apply_boundary_values(boundary_values, system_matrix, incremental_solution, residue_vector);
}


template <int dim>
void ElasticProblem<dim>::solve()
{
	/*SolverControl            solver_control(1000, 1e-12);
    SolverCG<Vector<double>> cg(solver_control);

    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);

    cg.solve(system_matrix, increment_solution, residue_vector, preconditioner);*/

	SparseDirectUMFPACK A_direct;
	A_direct.initialize(system_matrix);
	A_direct.vmult(incremental_solution, residue_vector);

	hanging_node_constraints.distribute(incremental_solution);
}



template <int dim>
void ElasticProblem<dim>::refine_grid()
{
			FE_DGQ<dim> history_fe (1);
				DoFHandler<dim> dof_handler_1(triangulation);
				dof_handler_1.distribute_dofs  (history_fe);

				std::vector< std::vector< Vector<double> > >
				             history_field1(dim, std::vector< Vector<double> >(dim)),
							 history_field2 (dim, std::vector< Vector<double> >(dim)),
							 history_field3 (dim, std::vector< Vector<double> >(dim)),
							 history_field4 (dim, std::vector< Vector<double> >(dim)),

				             local_history_stress_values_at_qpoints (dim, std::vector< Vector<double> >(dim)),
							 local_history_strain_values_at_qpoints (dim, std::vector< Vector<double> >(dim)),
							 local_history_t_strain_values_at_qpoints (dim, std::vector< Vector<double> >(dim)),
							 local_history_lambda_values_at_qpoints (dim, std::vector< Vector<double> >(dim)),

				             local_history_fe_values1 (dim, std::vector< Vector<double> >(dim)),
							 local_history_fe_values2 (dim, std::vector< Vector<double> >(dim)),
							 local_history_fe_values3 (dim, std::vector< Vector<double> >(dim)),
							 local_history_fe_values4 (dim, std::vector< Vector<double> >(dim));

				const unsigned int total_quadrature_points = triangulation.n_active_cells() * quadrature_formula.size();
				Vector<double> mvf(total_quadrature_points), temp(total_quadrature_points), transf_status, loading_status;
				Vector<double> avg_mvf_on_cell_vertices, avg_temp_on_cell_vertices, avg_trans_status_on_cell_vertices, avg_loading_status;

				for (unsigned int i=0; i<dim; ++i)
				{
					  for (unsigned int j=0; j<dim; ++j)
					  {
						history_field1[i][j].reinit(dof_handler_1.n_dofs());
						history_field2[i][j].reinit(dof_handler_1.n_dofs());
						history_field3[i][j].reinit(dof_handler_1.n_dofs());
						history_field4[i][j].reinit(dof_handler_1.n_dofs());

						local_history_stress_values_at_qpoints[i][j].reinit(quadrature_formula.size());
						local_history_strain_values_at_qpoints[i][j].reinit(quadrature_formula.size());
						local_history_t_strain_values_at_qpoints[i][j].reinit(quadrature_formula.size());
						local_history_lambda_values_at_qpoints[i][j].reinit(quadrature_formula.size());

						local_history_fe_values1[i][j].reinit(history_fe.n_dofs_per_cell());
						local_history_fe_values2[i][j].reinit(history_fe.n_dofs_per_cell());
						local_history_fe_values3[i][j].reinit(history_fe.n_dofs_per_cell());
						local_history_fe_values4[i][j].reinit(history_fe.n_dofs_per_cell());
					  }
				}
				avg_mvf_on_cell_vertices.reinit(dof_handler_1.n_dofs());
				avg_temp_on_cell_vertices.reinit(dof_handler_1.n_dofs());

				FullMatrix<double> qpoint_to_dof_matrix(history_fe.dofs_per_cell,
						quadrature_formula.size());

				FETools::compute_projection_from_quadrature_points_matrix
										  (history_fe,
										   quadrature_formula, quadrature_formula,
										   qpoint_to_dof_matrix);

				typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
				                                               endc = dof_handler.end(),
				                                               dg_cell = dof_handler_1.begin_active();
				for (; cell!=endc; ++cell, ++dg_cell)
				  {
				    PointHistory *local_quadrature_points_history
				    		= reinterpret_cast<PointHistory *>(cell->user_pointer());

				    Assert (local_quadrature_points_history >= &quadrature_point_history.front(),
				            ExcInternalError());
				    Assert (local_quadrature_points_history < &quadrature_point_history.back(),
				            ExcInternalError());

				    for (unsigned int i=0; i<dim; ++i)
				    {
				      for (unsigned int j=0; j<dim; ++j)
				      {
				        for (unsigned int q=0; q<quadrature_formula.size(); ++q)
				        {
				            local_history_stress_values_at_qpoints[i][j](q)
				            							= local_quadrature_points_history[q].old_stress[i][j];
				        	local_history_strain_values_at_qpoints[i][j](q)
				        								= local_quadrature_points_history[q].old_strain[i][j];
				        	local_history_t_strain_values_at_qpoints[i][j](q)
				        	        					= local_quadrature_points_history[q].old_t_strain[i][j];
				        	local_history_lambda_values_at_qpoints[i][j](q)
				        								= local_quadrature_points_history[q].old_lambda[i][j];

							qpoint_to_dof_matrix.vmult (local_history_fe_values1[i][j],
														local_history_stress_values_at_qpoints[i][j]);
							dg_cell->set_dof_values (local_history_fe_values1[i][j],
																				 history_field1[i][j]);

							qpoint_to_dof_matrix.vmult (local_history_fe_values2[i][j],
														local_history_strain_values_at_qpoints[i][j]);
							dg_cell->set_dof_values (local_history_fe_values2[i][j],
																				 history_field2[i][j]);

							qpoint_to_dof_matrix.vmult (local_history_fe_values3[i][j],
														local_history_t_strain_values_at_qpoints[i][j]);
							dg_cell->set_dof_values (local_history_fe_values3[i][j],
																				 history_field3[i][j]);

							qpoint_to_dof_matrix.vmult (local_history_fe_values4[i][j],
														local_history_lambda_values_at_qpoints[i][j]);
							dg_cell->set_dof_values (local_history_fe_values4[i][j],
													 	 	 	 	 	 	 	 history_field4[i][j]);
						  }
				       }
				    }
				    for (unsigned int q=0; q<quadrature_formula.size(); ++q)
				    {
				    	mvf[q] = local_quadrature_points_history[q].old_xi;
				    	temp[q] = local_quadrature_points_history[q].old_temperature;
				    }
				    map_scalar_qpoint_to_dof(mvf, avg_mvf_on_cell_vertices);
				    map_scalar_qpoint_to_dof(temp, avg_temp_on_cell_vertices);
				  }
				Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

					KellyErrorEstimator<dim>::estimate(dof_handler,
							QGauss<dim - 1>(fe.degree + 1),
							{},
							solution,
							estimated_error_per_cell);

					GridRefinement::refine_and_coarsen_fixed_number(triangulation,
							estimated_error_per_cell,
							0.3,
							0.03);
					triangulation.prepare_coarsening_and_refinement();

//				SolutionTransfer<dim> solution_transfer(dof_handler);
//				Vector<double> previous_solution;
//				previous_solution = solution;
//				triangulation.prepare_coarsening_and_refinement();
//				solution_transfer.prepare_for_coarsening_and_refinement(previous_solution);

				SolutionTransfer<dim, Vector<double> > history_stress_field_transfer(dof_handler_1);
				history_stress_field_transfer.prepare_for_coarsening_and_refinement(history_field1[2]);

				SolutionTransfer<dim, Vector<double> > history_strain_field_transfer(dof_handler_1);
				history_strain_field_transfer.prepare_for_coarsening_and_refinement(history_field2[2]);

				SolutionTransfer<dim, Vector<double> > history_t_strain_field_transfer(dof_handler_1);
				history_t_strain_field_transfer.prepare_for_coarsening_and_refinement(history_field3[2]);

				SolutionTransfer<dim, Vector<double> > history_lambda_field_transfer(dof_handler_1);
				history_lambda_field_transfer.prepare_for_coarsening_and_refinement(history_field4[2]);

				triangulation.execute_coarsening_and_refinement();

				  setup_system();
				  setup_quadrature_point_history ();
//				  solution_transfer.interpolate(previous_solution, solution);

				  hanging_node_constraints.distribute(solution);

				  dof_handler_1.distribute_dofs (history_fe);

				  std::vector< std::vector< Vector<double> > >
				  	  	  	  	  	  	  distributed_history_stress_field (dim, std::vector< Vector<double> >(dim));
				  for (unsigned int i=0; i<dim; ++i)
				  {
					  for (unsigned int j=0; j<dim; ++j)
				      {
				          distributed_history_stress_field[i][j].reinit(dof_handler_1.n_dofs());
				      }
				  }
				      history_stress_field_transfer.interpolate(history_field1[2], distributed_history_stress_field[2]);
				      history_field1 = distributed_history_stress_field;

				   std::vector< std::vector< Vector<double> > >
				   	   	   	   	   	   	  distributed_history_strain_field (dim, std::vector< Vector<double> >(dim));
				   for (unsigned int i=0; i<dim; ++i)
				   {
				       for (unsigned int j=0; j<dim; ++j)
				       {
				           distributed_history_strain_field[i][j].reinit(dof_handler_1.n_dofs());
				       }
				   }
				       history_strain_field_transfer.interpolate(history_field2[2], distributed_history_strain_field[2]);
				       history_field2 = distributed_history_strain_field;

				    std::vector< std::vector< Vector<double> > >
				    					  distributed_history_t_strain_field (dim, std::vector< Vector<double> >(dim));
				    for (unsigned int i=0; i<dim; ++i)
				        {
				        for (unsigned int j=0; j<dim; ++j)
				       		{
				       		distributed_history_t_strain_field[i][j].reinit(dof_handler_1.n_dofs());
				       		}
				        }
				       	history_t_strain_field_transfer.interpolate(history_field3[2], distributed_history_t_strain_field[2]);
				       	history_field3 = distributed_history_t_strain_field;

				       std::vector< std::vector< Vector<double> > >
				       	   	   	   	   	   distributed_history_lambda_field (dim, std::vector< Vector<double> >(dim));
				       for (unsigned int i=0; i<dim; ++i)
				       		{
				       			for (unsigned int j=0; j<dim; ++j)
				       				{
				       				    distributed_history_lambda_field[i][j].reinit(dof_handler_1.n_dofs());
				       				}
				       		}
				       			history_lambda_field_transfer.interpolate(history_field4[2], distributed_history_lambda_field[2]);
				       			history_field4 = distributed_history_lambda_field;

				FullMatrix<double> dof_to_qpoint_matrix (quadrature_formula.size(),
				                                         history_fe.dofs_per_cell);

				FETools::compute_interpolation_to_quadrature_points_matrix
										  (history_fe,
										   quadrature_formula,
										   dof_to_qpoint_matrix);

				cell = dof_handler.begin_active();
				endc = dof_handler.end();
				dg_cell = dof_handler_1.begin_active();

				for (; cell != endc; ++cell, ++dg_cell)
				{
					PointHistory *local_quadrature_points_history
							= reinterpret_cast<PointHistory *>(cell->user_pointer());

				  Assert (local_quadrature_points_history >= &quadrature_point_history.front(),
				          ExcInternalError());
				  Assert (local_quadrature_points_history < &quadrature_point_history.back(),
				          ExcInternalError());

				  for (unsigned int i=0; i<dim; ++i)
				  {
						for (unsigned int j=0; j<dim; ++j)
						{
						  dg_cell->get_dof_values (history_field1[i][j],
												   local_history_fe_values1[i][j]);
						  dof_to_qpoint_matrix.vmult(local_history_stress_values_at_qpoints[i][j],
																				  local_history_fe_values1[i][j]);

						  dg_cell->get_dof_values (history_field2[i][j],
												   local_history_fe_values2[i][j]);
						  dof_to_qpoint_matrix.vmult (local_history_strain_values_at_qpoints[i][j],
																				  local_history_fe_values2[i][j]);

						  dg_cell->get_dof_values (history_field3[i][j],
												   local_history_fe_values3[i][j]);
						  dof_to_qpoint_matrix.vmult (local_history_t_strain_values_at_qpoints[i][j],
																				  local_history_fe_values3[i][j]);

						  dg_cell->get_dof_values (history_field4[i][j],
												   local_history_fe_values4[i][j]);
						  dof_to_qpoint_matrix.vmult (local_history_lambda_values_at_qpoints[i][j],
															  local_history_fe_values4[i][j]);

						  for (unsigned int q=0; q<quadrature_formula.size(); ++q)
							  {
								  local_quadrature_points_history[q].old_stress[i][j]
										  = local_history_stress_values_at_qpoints[i][j](q);
								  local_quadrature_points_history[q].old_strain[i][j]
										  = local_history_strain_values_at_qpoints[i][j](q);
								  local_quadrature_points_history[q].old_t_strain[i][j]
										  = local_history_t_strain_values_at_qpoints[i][j](q);
								local_quadrature_points_history[q].old_lambda[i][j]
										  = local_history_lambda_values_at_qpoints[i][j](q);
							  }
						   }
						}
					  for (unsigned int q=0; q<quadrature_formula.size(); ++q)
					  {
						  local_quadrature_points_history[q].old_xi
														  = mvf[q];
						  local_quadrature_points_history[q].old_temperature
														  = temp[q];
					  }
				}
}

template <int dim>
void ElasticProblem<dim>::setup_quadrature_point_history()
{
	triangulation.clear_user_data();
	{
		std::vector<PointHistory > tmp;
		tmp.swap (quadrature_point_history);
	}
	quadrature_point_history.resize (triangulation.n_active_cells() * quadrature_formula.size());

	unsigned int history_index = 0;
	for (auto &cell : triangulation.active_cell_iterators())
	{
		cell->set_user_pointer (&quadrature_point_history[history_index]);
		history_index += quadrature_formula.size();
	}
	Assert (history_index == quadrature_point_history.size(), ExcInternalError());
}

template <int dim>
void ElasticProblem<dim>::setup_initial_quadrature_point_history()
{
	FEValues<dim> fe_values (fe, quadrature_formula,
			update_values | update_gradients |
			update_quadrature_points);
	const unsigned int n_q_points = quadrature_formula.size();
	//std::vector<std::vector<Tensor<1, dim>>> solution_grads(n_q_points, std::vector<Tensor<1, dim>>(dim));

	const SymmetricTensor<2,3> tmp_tensor=0.*unit_symmetric_tensor<3>();
	const double xi_0=1e-6;
	for(auto &cell : dof_handler.active_cell_iterators())
	{
		PointHistory *local_quadrature_points_history = reinterpret_cast<PointHistory *>(cell->user_pointer());
		Assert (local_quadrature_points_history >= &quadrature_point_history.front(),ExcInternalError());
		Assert (local_quadrature_points_history < &quadrature_point_history.back(),ExcInternalError());

		fe_values.reinit (cell);
		//fe_values.get_function_gradients(solution, solution_grads);

		for (unsigned int q = 0; q < n_q_points; ++q)
		{
			local_quadrature_points_history[q].continuum_moduli_structural = constitutive_law_sma.get_s_inv(xi_0);
			local_quadrature_points_history[q].continuum_moduli_thermal = - (constitutive_law_sma.get_s_inv(xi_0) * constitutive_law_sma.get_alpha(xi_0));
			local_quadrature_points_history[q].old_stress = tmp_tensor;
			local_quadrature_points_history[q].stress = tmp_tensor;
			local_quadrature_points_history[q].old_strain = tmp_tensor;
			local_quadrature_points_history[q].old_t_strain = tmp_tensor;
			local_quadrature_points_history[q].t_strain_r = tmp_tensor;
			local_quadrature_points_history[q].old_xi = xi_0;
			local_quadrature_points_history[q].old_temperature = T_initial;
			local_quadrature_points_history[q].old_transformation_status = 0;
			local_quadrature_points_history[q].old_lambda = constitutive_law_sma.get_lambda(tmp_tensor, 1); //tmp_vector1;
		}
	}
}
template <int dim>
void ElasticProblem<dim>::update_quadrature_point_history(const bool &cond, const unsigned int &nr_counter)
{
	FEValues<dim> fe_values (fe, quadrature_formula,
			update_values | update_gradients |
			update_quadrature_points);
	const unsigned int n_q_points = quadrature_formula.size();

	std::vector<std::vector<Tensor<1, dim>>> solution_grads(n_q_points, std::vector<Tensor<1, dim>>(dim));
	SymmetricTensor<2,dim> strain_tmp;
	SymmetricTensor<2,3> stress_new, strain_new, lambda_new, t_strain_new, t_strain_r_new, continuum_moduli_thermal_new;
	SymmetricTensor<4,3> continuum_moduli_structural_new;

	std::vector<Tensor<1, dim>> rhs_values(n_q_points);
	Vector<double> body_temp(n_q_points);

	double xi_new, temp_new;
	bool outcome = false;

	for(auto &cell : dof_handler.active_cell_iterators())
	{
		PointHistory *local_quadrature_points_history = reinterpret_cast<PointHistory *>(cell->user_pointer());
		Assert (local_quadrature_points_history >= &quadrature_point_history.front(),ExcInternalError());
		Assert (local_quadrature_points_history < &quadrature_point_history.back(),ExcInternalError());

		fe_values.reinit (cell);
		fe_values.get_function_gradients(solution, solution_grads);
		right_hand_side(fe_values.get_quadrature_points(), present_time, rhs_values, body_temp);

		//std::cout << "present time"  << time << " body temp = " << body_temp[0] << std::endl;

		for (unsigned int q_point : fe_values.quadrature_point_indices())
		{
			strain_tmp = get_linear_strain(solution_grads[q_point]);

			stress_new=0.; lambda_new=0.; t_strain_new=0.; t_strain_r_new=0.; xi_new=0.; continuum_moduli_structural_new=0.; continuum_moduli_thermal_new =0.;
			int transformation_status =0, loading_status=0;

			strain_new = convert_symmetric_tensor_2d_to_3d<dim>(strain_tmp);
			temp_new = body_temp[q_point];

			/*for (unsigned int i = 0; i < 2; ++i){
   			       for (unsigned int j = 0; j < 2; ++j)
   			          	  std::cout << strain_new_3d [i][j] << "\t" << std::endl;
   			       	      std::cout << std::endl;}*/
			if (present_time <= time_step)
			{
				continuum_moduli_structural_new =  constitutive_law_sma.get_s_inv(0.);
				stress_new = continuum_moduli_structural_new*strain_new;
				lambda_new =  constitutive_law_sma.get_lambda(stress_new, 1);
			}
			else
			{
			outcome = constitutive_law_sma.call_convex_cutting(strain_new, temp_new, local_quadrature_points_history[q_point],
					nr_counter, stress_new, continuum_moduli_structural_new, continuum_moduli_thermal_new, lambda_new, t_strain_new, t_strain_r_new, xi_new,
					transformation_status, loading_status);
			}
			//std::cout << "Material model converged! " << outcome << std::endl;
			// For elastic material: stress_new =  moduli * strain_new
			//std::cout<< "strain at Gauss point" << strain_new << std::endl;
			//continuum_moduli_new = constitutive_law_sma.get_s_inv(0.);
			//stress_new =  continuum_moduli_new * strain_new;
			//std::cout<< "stress at Gauss point" << stress_new << std::endl;
			//outcome = true;

			if (cond == false)	// N-R iteration not converged
			{
				local_quadrature_points_history[q_point].stress = stress_new;
				local_quadrature_points_history[q_point].continuum_moduli_structural = continuum_moduli_structural_new;
				local_quadrature_points_history[q_point].continuum_moduli_thermal = continuum_moduli_thermal_new;
				local_quadrature_points_history[q_point].old_temperature = temp_new;
				if (nr_counter == 0)
					local_quadrature_points_history[q_point].loading_status_0_iter = loading_status;
			}
			else	// N-R converged for the current load step
			{
				local_quadrature_points_history[q_point].continuum_moduli_structural = continuum_moduli_structural_new;
				local_quadrature_points_history[q_point].continuum_moduli_thermal = continuum_moduli_thermal_new;
				local_quadrature_points_history[q_point].old_stress = stress_new;
				local_quadrature_points_history[q_point].stress = stress_new;
				local_quadrature_points_history[q_point].old_strain = strain_new;
				local_quadrature_points_history[q_point].old_t_strain = t_strain_new;
				local_quadrature_points_history[q_point].t_strain_r = t_strain_r_new;
				local_quadrature_points_history[q_point].old_xi = xi_new;
				/*local_quadrature_points_history[q_point].old_temperature = temp_new;*/
				local_quadrature_points_history[q_point].old_lambda = lambda_new;
				local_quadrature_points_history[q_point].old_transformation_status = transformation_status;
				local_quadrature_points_history[q_point].loading_status_0_iter = loading_status;
			}
		}
	}
	//if (cond == false)
	std::cout << nr_counter <<"		" << outcome << "	" <<std::scientific << strain_new[0][0] << "  	"
			<< stress_new[0][0] << "	"  << temp_new << "	" << xi_new <<"  	"
			<< t_strain_new[0][0] <<"  	"<< continuum_moduli_structural_new[0][0][0][0] <<"  	";
}

template<int dim>
void ElasticProblem<dim>::call_Newton_Raphson_method()
{
	unsigned int nr_counter = 0;
	double residue_norm=0., residue_norm_NR_0 = 0., delta_solution_norm=0.;
	const double tol = 1e-3;
	while(true)
	{
		incremental_solution = 0.;
		assemble_system();

		residue_norm = residue_vector.l2_norm();
		if (nr_counter==0)
		{
			residue_norm_NR_0 = residue_norm;
		}
		//std :: cout << "The norm of residue in " << present_time << " and " <<
		//		nr_counter << " N-R iteration is ---->> " << (residue_norm/residue_norm_NR_0) << std :: endl;

		solve();
		//std::cout<< "incremental soluton	" << incremental_solution << std::endl;
		solution += incremental_solution;


		delta_solution_norm = (incremental_solution.l2_norm()/solution.l2_norm());
		//std::cout << "Change in solution norm " << delta_solution_norm << std::endl;

		if (residue_norm < tol || nr_counter > 50){
			update_quadrature_point_history(true, nr_counter);
			std::cout << std::fixed << std::setprecision(3) << std::setw(7)
															<< std::scientific << residue_norm << "     " << (residue_norm/residue_norm_NR_0)
															<< "	" << delta_solution_norm <<std::endl;
			break;
		}
		else
		{
			update_quadrature_point_history(false, nr_counter);		// update the stress and strain with new solution
			std::cout << std::fixed << std::setprecision(3) << std::setw(7)
															  << std::scientific << residue_norm << "     " << (residue_norm/residue_norm_NR_0)
															  << "	" << delta_solution_norm <<std::endl;
			nr_counter++;
		}
	}
}


template <int dim>
void ElasticProblem<dim>::run()
{
	/*for (unsigned int cycle = 0; cycle < 1; ++cycle)
      {
      std::cout << "Cycle " << cycle << ':' << std::endl;
        if (cycle == 0)
          {
            GridGenerator::hyper_cube(triangulation, -1, 1);
            triangulation.refine_global(4);
          }
        else
          refine_grid();
	   	create_mesh(cycle); */

	create_mesh();
	std::cout << "   Number of active cells:       "
			<< triangulation.n_active_cells() << std::endl;
	setup_quadrature_point_history();

	setup_system();
	std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
                        		<< std::endl;
	solution = 0.;
	setup_initial_quadrature_point_history();	// to make sure the initial stresses are zero at quadrature points

	unsigned int time_step_counter = 1;
	//current_time = 0.; //initiated in the constructor
	while (time_step_counter <= n_time_steps)
	{
		present_time += time_step;
		if (present_time > end_time)
		{
			time_step -= (present_time - end_time);
			present_time = end_time;
		}
		std :: cout << "--------- Current time stamp ---------------------->> "
				<< present_time << std :: endl;
		print_conv_header();
		setup_system();
		// N-R iteration
		call_Newton_Raphson_method();

		if(time_step_counter % 5 == 0)
					{
						refine_grid();
						update_quadrature_point_history(true,0);
					}
		output_results(time_step_counter);
		time_step_counter++;
		print_conv_footer();
	}
}

template <int dim>
void ElasticProblem<dim>::output_results(const unsigned int timestep_counter) const
{
	// calculating strain and stress components for graphical output
	FE_Q<dim> fe_1 (1);
	DoFHandler<dim> dof_handler_1 (triangulation);
	dof_handler_1.distribute_dofs (fe_1);

	AssertThrow(dof_handler_1.n_dofs() == triangulation.n_vertices(),
			ExcDimensionMismatch(dof_handler_1.n_dofs(),triangulation.n_vertices()));

	//___________ separating nodal displacements ________________________
	std::vector< Vector<double> > solution_components (dim), load_components(dim);
	for(unsigned int i = 0; i < dim; ++i)
	{
		solution_components[i].reinit(dof_handler_1.n_dofs());
		load_components[i].reinit(dof_handler_1.n_dofs());
	}
	std::vector<bool> vertex_touched(triangulation.n_vertices(), false);
	const NeumannBoundary<dim> neumann_boundary(present_time, time_step);
	Vector<double> load_vector(dim);
	typename DoFHandler<dim>::active_cell_iterator cell_1 = dof_handler_1.begin_active();
	for (auto &cell : dof_handler.active_cell_iterators())
	{
		neumann_boundary.vector_value(cell->center(), load_vector);					// cell->center gerbage input
		for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
			if (vertex_touched[cell->vertex_index(v)] == false)
			{
				vertex_touched[cell->vertex_index(v)] = true;
				for (unsigned int d = 0; d < dim; ++d){
					solution_components[d](cell_1->vertex_dof_index(v,0)) = solution(cell->vertex_dof_index(v, d));
					load_components[d](cell_1->vertex_dof_index(v,0)) = load_vector(d);
				}
			}
		cell_1++;
	}

	//______________ obtaining nodal stress and strain components____________________
	FEValues<dim> fe_values(fe, quadrature_formula, update_values | update_gradients);

	std::vector<std::vector<Tensor<1, dim>>> solution_grads(quadrature_formula.size(),
			std::vector<Tensor<1, dim>>(dim));
	const unsigned int total_quadrature_points = triangulation.n_active_cells() * quadrature_formula.size();
	std::vector<Vector<double>> strain(total_quadrature_points), t_strain(total_quadrature_points),
			stress(total_quadrature_points);
	Vector<double> mvf(total_quadrature_points), temp(total_quadrature_points);

	const unsigned int n_components = (dim==2 ? 3 : 6);

	for (unsigned int i=0; i < total_quadrature_points; i++ )
	{
		strain[i].reinit(n_components);
		t_strain[i].reinit(n_components);
		stress[i].reinit(n_components);
	}

	unsigned int q_k = 0;
	for (auto &cell : dof_handler.active_cell_iterators())
	{
		PointHistory *local_quadrature_points_history
		= reinterpret_cast<PointHistory *>(cell->user_pointer());
		Assert (local_quadrature_points_history >= &quadrature_point_history.front(),ExcInternalError());
		Assert (local_quadrature_points_history < &quadrature_point_history.back(),ExcInternalError());

		fe_values.reinit(cell);
		//fe_values.get_function_gradients(solution, solution_grads);


		for (unsigned int q_point : fe_values.quadrature_point_indices())
		{
			if (dim == 3)
			{
				strain[q_k][0] = local_quadrature_points_history[q_point].old_strain[0][0]; //get_strain(solution_grads[q_point]);
				strain[q_k][1] = local_quadrature_points_history[q_point].old_strain[1][1];
				strain[q_k][2] = local_quadrature_points_history[q_point].old_strain[2][2];
				strain[q_k][3] = local_quadrature_points_history[q_point].old_strain[1][2];
				strain[q_k][4] = local_quadrature_points_history[q_point].old_strain[0][2];
				strain[q_k][5] = local_quadrature_points_history[q_point].old_strain[0][1];
				t_strain[q_k][0] = local_quadrature_points_history[q_point].old_t_strain[0][0];
				t_strain[q_k][1] = local_quadrature_points_history[q_point].old_t_strain[1][1];
				t_strain[q_k][2] = local_quadrature_points_history[q_point].old_t_strain[2][2];
				t_strain[q_k][3] = local_quadrature_points_history[q_point].old_t_strain[1][2];
				t_strain[q_k][4] = local_quadrature_points_history[q_point].old_t_strain[0][2];
				t_strain[q_k][5] = local_quadrature_points_history[q_point].old_t_strain[0][1];
				stress[q_k][0] = local_quadrature_points_history[q_point].old_stress[0][0];
				stress[q_k][1] = local_quadrature_points_history[q_point].old_stress[1][1];
				stress[q_k][2] = local_quadrature_points_history[q_point].old_stress[2][2];
				stress[q_k][3] = local_quadrature_points_history[q_point].old_stress[1][2];
				stress[q_k][4] = local_quadrature_points_history[q_point].old_stress[0][2];
				stress[q_k][5] = local_quadrature_points_history[q_point].old_stress[0][1];
				mvf[q_k] = local_quadrature_points_history[q_point].old_xi;
				temp[q_k] = local_quadrature_points_history[q_point].old_temperature;
				q_k++;
			}
			else
			{
				strain[q_k][0] = local_quadrature_points_history[q_point].old_strain[0][0]; //get_strain(solution_grads[q_point]);
				strain[q_k][1] = local_quadrature_points_history[q_point].old_strain[1][1];
				strain[q_k][2] = local_quadrature_points_history[q_point].old_strain[0][1];

				t_strain[q_k][0] = local_quadrature_points_history[q_point].old_t_strain[0][0];
				t_strain[q_k][1] = local_quadrature_points_history[q_point].old_t_strain[1][1];
				t_strain[q_k][2] = local_quadrature_points_history[q_point].old_t_strain[0][1];

				stress[q_k][0] = local_quadrature_points_history[q_point].old_stress[0][0];
				stress[q_k][1] = local_quadrature_points_history[q_point].old_stress[1][1];
				stress[q_k][2] = local_quadrature_points_history[q_point].old_stress[0][1];

				mvf[q_k] = local_quadrature_points_history[q_point].old_xi;
				temp[q_k] = local_quadrature_points_history[q_point].old_temperature;
				q_k++;
			}
		}
	}
	std::vector< Vector<double> > avg_strain_on_cell_vertices (n_components),
			avg_t_strain_on_cell_vertices (n_components), avg_stress_on_cell_vertices (n_components);
	Vector<double> avg_mvf_on_cell_vertices, avg_temp_on_cell_vertices;
	for (unsigned int i=0; i < n_components; ++i)
	{
		avg_strain_on_cell_vertices[i].reinit(dof_handler_1.n_dofs());
		avg_t_strain_on_cell_vertices[i].reinit(dof_handler_1.n_dofs());
		avg_stress_on_cell_vertices[i].reinit(dof_handler_1.n_dofs());
	}
	avg_mvf_on_cell_vertices.reinit(dof_handler_1.n_dofs());
	avg_temp_on_cell_vertices.reinit(dof_handler_1.n_dofs());

	map_vec_qpoint_to_dof(n_components, strain, avg_strain_on_cell_vertices);
	map_vec_qpoint_to_dof(n_components, t_strain, avg_t_strain_on_cell_vertices);
	map_vec_qpoint_to_dof(n_components, stress, avg_stress_on_cell_vertices);
	map_scalar_qpoint_to_dof(mvf, avg_mvf_on_cell_vertices);
	map_scalar_qpoint_to_dof(temp, avg_temp_on_cell_vertices);



	// Lets save the strain components
	{
		DataOut<dim>  data_out;
		data_out.attach_dof_handler (dof_handler_1);



		if (dim == 2)
		{
			data_out.add_data_vector (solution_components[0], "x_displacement");
			data_out.add_data_vector (solution_components[1], "y_displacement");

			data_out.add_data_vector (load_components[0], "x_load");
			data_out.add_data_vector (load_components[1], "y_load");

			data_out.add_data_vector (avg_strain_on_cell_vertices[0], "strain_xx");
			data_out.add_data_vector (avg_strain_on_cell_vertices[1], "strain_yy");
			data_out.add_data_vector (avg_strain_on_cell_vertices[2], "strain_xy");

			data_out.add_data_vector (avg_t_strain_on_cell_vertices[0], "t_strain_xx");
			data_out.add_data_vector (avg_t_strain_on_cell_vertices[1], "t_strain_yy");
			data_out.add_data_vector (avg_t_strain_on_cell_vertices[2], "t_strain_xy");

			data_out.add_data_vector (avg_stress_on_cell_vertices[0], "stress_xx");
			data_out.add_data_vector (avg_stress_on_cell_vertices[1], "stress_yy");
			data_out.add_data_vector (avg_stress_on_cell_vertices[2], "stress_xy");
		}
		else
		{
			data_out.add_data_vector (solution_components[0], "x_displacement");
			data_out.add_data_vector (solution_components[1], "y_displacement");
			data_out.add_data_vector (solution_components[2], "z_displacement");

			data_out.add_data_vector (load_components[0], "x_load");
			data_out.add_data_vector (load_components[1], "y_load");
			data_out.add_data_vector (load_components[2], "z_load");

			data_out.add_data_vector (avg_strain_on_cell_vertices[0], "strain_xx");
			data_out.add_data_vector (avg_strain_on_cell_vertices[1], "strain_yy");
			data_out.add_data_vector (avg_strain_on_cell_vertices[2], "strain_zz");
			data_out.add_data_vector (avg_strain_on_cell_vertices[3], "strain_yz");
			data_out.add_data_vector (avg_strain_on_cell_vertices[4], "strain_xz");
			data_out.add_data_vector (avg_strain_on_cell_vertices[5], "strain_xy");

			data_out.add_data_vector (avg_t_strain_on_cell_vertices[0], "t_strain_xx");
			data_out.add_data_vector (avg_t_strain_on_cell_vertices[1], "t_strain_yy");
			data_out.add_data_vector (avg_t_strain_on_cell_vertices[2], "t_strain_zz");
			data_out.add_data_vector (avg_t_strain_on_cell_vertices[3], "t_strain_yz");
			data_out.add_data_vector (avg_t_strain_on_cell_vertices[4], "t_strain_xz");
			data_out.add_data_vector (avg_t_strain_on_cell_vertices[5], "t_strain_xy");

			data_out.add_data_vector (avg_stress_on_cell_vertices[0], "stress_xx");
			data_out.add_data_vector (avg_stress_on_cell_vertices[1], "stress_yy");
			data_out.add_data_vector (avg_stress_on_cell_vertices[2], "stress_zz");
			data_out.add_data_vector (avg_stress_on_cell_vertices[3], "stress_yz");
			data_out.add_data_vector (avg_stress_on_cell_vertices[4], "stress_xz");
			data_out.add_data_vector (avg_stress_on_cell_vertices[5], "stress_xy");

		}
		data_out.add_data_vector (avg_mvf_on_cell_vertices, "mvf");
		data_out.add_data_vector (avg_temp_on_cell_vertices, "temp");

		Vector<double> time;
		time.reinit(dof_handler_1.n_dofs());
		time = present_time;
		data_out.add_data_vector (time, "time");


		Vector<double> soln(solution.size());
		for (unsigned int i = 0; i < soln.size(); ++i)
			soln(i) = solution(i);
		MappingQEulerian<dim> q_mapping(1, dof_handler, soln);
		data_out.build_patches(q_mapping, 1);

//		data_out.build_patches ();

		std::ofstream output("./ResultsAd/beam_r4-" + std::to_string(dim) + "D_"
				+ std::to_string(timestep_counter) + ".vtk");
		data_out.write_vtk(output);
	}
}

template <int dim>
void ElasticProblem<dim>::map_scalar_qpoint_to_dof(const Vector<double> &vec,
		Vector<double> &avg_vec_on_cell_vertices) const
{
	// The input vector vec contains the values of a scalar field at the quadrature points of all cells
	FE_DGQ<dim> history_fe (1);
	DoFHandler<dim> history_dof_handler (triangulation);
	history_dof_handler.distribute_dofs (history_fe);
	Vector<double> vec_field, vec_values_at_qpoints, vec_at_dgcell_vertices;
	vec_field.reinit(history_dof_handler.n_dofs());
	vec_values_at_qpoints.reinit(quadrature_formula.size());
	vec_at_dgcell_vertices.reinit(history_fe.dofs_per_cell);


	FullMatrix<double> qpoint_to_dof_matrix (history_fe.dofs_per_cell,
			quadrature_formula.size());
	FETools::compute_projection_from_quadrature_points_matrix (history_fe, quadrature_formula,
			quadrature_formula, qpoint_to_dof_matrix);

	unsigned int q_k = 0;
	typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
			endc = dof_handler.end(), dg_cell = history_dof_handler.begin_active();
	for (; cell!=endc; ++cell, ++dg_cell)
	{

		for (unsigned int q=0; q<quadrature_formula.size(); ++q)
		{
			vec_values_at_qpoints(q) = vec[q_k];  // particular strain components in all quadrature points in a cell
			q_k++;
		}
		qpoint_to_dof_matrix.vmult (vec_at_dgcell_vertices, vec_values_at_qpoints);
		dg_cell->set_dof_values (vec_at_dgcell_vertices, vec_field);
	}

	// Now we need find strain on cell vertices using nodal averaging

	FE_Q<dim>          fe_1 (1);
	DoFHandler<dim>    dof_handler_1 (triangulation);
	dof_handler_1.distribute_dofs (fe_1);

	AssertThrow(dof_handler_1.n_dofs() == triangulation.n_vertices(),
			ExcDimensionMismatch(dof_handler_1.n_dofs(),triangulation.n_vertices()));

	/*std::vector< Vector<double> > avg_strain_on_cell_vertices (n_strain_components);
      	      for (unsigned int i=0; i < n_strain_components; ++i)
      	          {
      	            avg_strain_on_cell_vertices[i].reinit(dof_handler_1.n_dofs());
      	          }*/


	Vector<double>  counter_on_vertices (dof_handler_1.n_dofs());
	counter_on_vertices = 0;

	cell = dof_handler.begin_active();
	dg_cell = history_dof_handler.begin_active();
	typename DoFHandler<dim>::active_cell_iterator cell_1 = dof_handler_1.begin_active();
	for (; cell!=endc; ++cell, ++dg_cell, ++cell_1)
	{
		dg_cell->get_dof_values (vec_field, vec_at_dgcell_vertices);

		for  (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
		{
			types::global_dof_index dof_1_vertex = cell_1->vertex_dof_index(v, 0);
			counter_on_vertices (dof_1_vertex) += 1;

			avg_vec_on_cell_vertices(dof_1_vertex) += vec_at_dgcell_vertices(v);

		}
	}
	for (unsigned int id=0; id<dof_handler_1.n_dofs(); ++id)
	{
		avg_vec_on_cell_vertices(id) /= counter_on_vertices(id);
	}
}

template <int dim>
void ElasticProblem<dim>::map_vec_qpoint_to_dof(const unsigned int &n_components, const std::vector<Vector<double>> &vec,
		std::vector<Vector<double>> &avg_vec_on_cell_vertices) const
{
	// Lets determine the strain components on the vertices
	Vector<double> vec_at_qpoint(n_components);

	FE_DGQ<dim> history_fe (1);
	DoFHandler<dim> history_dof_handler (triangulation);
	history_dof_handler.distribute_dofs (history_fe);
	std::vector< Vector<double> > vec_field (n_components), vec_values_at_qpoints (n_components),
			vec_at_dgcell_vertices (n_components);

	for (unsigned int i=0; i< n_components; ++i)
	{
		vec_field[i].reinit(history_dof_handler.n_dofs());
		vec_values_at_qpoints[i].reinit(quadrature_formula.size());
		vec_at_dgcell_vertices[i].reinit(history_fe.dofs_per_cell);
	}

	FullMatrix<double> qpoint_to_dof_matrix (history_fe.dofs_per_cell,
			quadrature_formula.size());
	FETools::compute_projection_from_quadrature_points_matrix (history_fe, quadrature_formula,
			quadrature_formula, qpoint_to_dof_matrix);

	unsigned int q_k = 0;
	typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
			endc = dof_handler.end(), dg_cell = history_dof_handler.begin_active();
	for (; cell!=endc; ++cell, ++dg_cell)
	{
		for (unsigned int q=0; q<quadrature_formula.size(); ++q)
		{
			vec_at_qpoint = vec[q_k];

			for (unsigned int i=0; i< n_components; ++i)
			{
				vec_values_at_qpoints[i](q) = vec_at_qpoint[i];  // particular strain components in all quadrature points in a cell
			}
			q_k++;
		}

		for (unsigned int i=0; i< n_components; ++i)
		{
			qpoint_to_dof_matrix.vmult (vec_at_dgcell_vertices[i], vec_values_at_qpoints[i]);
			dg_cell->set_dof_values (vec_at_dgcell_vertices[i], vec_field[i]);
		}
	}

	// Now we need find strain on cell vertices using nodal averaging

	FE_Q<dim>          fe_1 (1);
	DoFHandler<dim>    dof_handler_1 (triangulation);
	dof_handler_1.distribute_dofs (fe_1);

	AssertThrow(dof_handler_1.n_dofs() == triangulation.n_vertices(),
			ExcDimensionMismatch(dof_handler_1.n_dofs(),triangulation.n_vertices()));

	/*std::vector< Vector<double> > avg_strain_on_cell_vertices (n_strain_components);
    	      for (unsigned int i=0; i < n_strain_components; ++i)
    	          {
    	            avg_strain_on_cell_vertices[i].reinit(dof_handler_1.n_dofs());
    	          }*/


	Vector<double>  counter_on_vertices (dof_handler_1.n_dofs());
	counter_on_vertices = 0;

	cell = dof_handler.begin_active();
	dg_cell = history_dof_handler.begin_active();
	typename DoFHandler<dim>::active_cell_iterator cell_1 = dof_handler_1.begin_active();
	for (; cell!=endc; ++cell, ++dg_cell, ++cell_1)
	{
		for (unsigned int i=0; i< n_components; ++i)
		{
			dg_cell->get_dof_values (vec_field[i], vec_at_dgcell_vertices[i]);
		}
		for  (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
		{
			types::global_dof_index dof_1_vertex = cell_1->vertex_dof_index(v, 0);
			counter_on_vertices (dof_1_vertex) += 1;

			for (unsigned int i=0; i< n_components; ++i)
			{
				avg_vec_on_cell_vertices[i](dof_1_vertex) += vec_at_dgcell_vertices[i](v);
			}
		}
	}
	for (unsigned int id=0; id<dof_handler_1.n_dofs(); ++id)
	{
		for (unsigned int i=0; i< n_components; ++i)
		{
			avg_vec_on_cell_vertices[i](id) /= counter_on_vertices(id);
		}
	}
}

template <int dim>
void ElasticProblem<dim> :: print_conv_header()
{
	static const unsigned int l_width = 145;
	for (unsigned int i = 0; i < l_width; ++i)
		std::cout << "-";
	std::cout << std::endl;
	std::cout << "NR_ITER" << "	 ConvexCutAlgo"
			<< "|   STRAIN        STRESS     	Temp		MVF     	t_strain     	MODULI[0][0]"
			<< "	RES_F	  RES_F/RES_F_I  "
			<< "    dNORM_U/NORM_U    "
			<< std::endl;
	for (unsigned int i = 0; i < l_width; ++i)
		std::cout << "-";
	std::cout << std::endl;
}

template <int dim>
void ElasticProblem<dim>::print_conv_footer()
{
	static const unsigned int l_width = 145;
	for (unsigned int i = 0; i < l_width; ++i)
		std::cout << "-";
	std::cout << std::endl;
}

} // namespace Step8


int main()
{
	const unsigned int case_3D = 2;
//	const unsigned int case_plane_strain=1;
	try
	{

//		Step8::ElasticProblem<2> elastic_problem_2d(case_plane_strain);
//		elastic_problem_2d.run();

		Step8::ElasticProblem<3> elastic_problem_3d(case_3D);
		elastic_problem_3d.run();


	}
	catch (std::exception &exc)
	{
		std::cerr << std::endl
				<< std::endl
				<< "----------------------------------------------------"
				<< std::endl;
		std::cerr << "Exception on processing: " << std::endl
				<< exc.what() << std::endl
				<< "Aborting!" << std::endl
				<< "----------------------------------------------------"
				<< std::endl;

		return 1;
	}
	catch (...)
	{
		std::cerr << std::endl
				<< std::endl
				<< "----------------------------------------------------"
				<< std::endl;
		std::cerr << "Unknown exception!" << std::endl
				<< "Aborting!" << std::endl
				<< "----------------------------------------------------"
				<< std::endl;
		return 1;
	}

	return 0;
}
