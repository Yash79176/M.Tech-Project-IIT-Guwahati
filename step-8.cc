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
#include <deal.II/base/parameter_handler.h>

#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/lac/generic_linear_algebra.h>

#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

namespace LA
{
#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) && \
  !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
  using namespace dealii::LinearAlgebraPETSc;
#  define USE_PETSC_LA
#elif defined(DEAL_II_WITH_TRILINOS)
  using namespace dealii::LinearAlgebraTrilinos;
#else
#  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
} // namespace LA

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>
#include <deal.II/lac/sparse_direct.h>
//#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/constraint_matrix.h>


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
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include <fstream>
#include <iostream>

namespace Step8
{
using namespace dealii;

const double Tref = 300.; 			// Kelvin (reference temp of SMA)
const double Tmax = 315.;			// Kelvin (max temp)
const double Pmax = 32.6e6;			// MPa  (max. stress)
const double total_time = 2.;		// total time for simulation
const double time_step = 0.005;
const int n_steps = std::round(total_time / time_step);

struct PointHistory
{
	SymmetricTensor<4,3> continuum_moduli_structural;
	SymmetricTensor<2,3> continuum_moduli_thermal;
	SymmetricTensor<4,3> old_continuum_moduli_structural;
	SymmetricTensor<2,3> old_continuum_moduli_thermal;
	SymmetricTensor<2,3> old_stress;			// stress after load step converged at t_n
	SymmetricTensor<2,3> stress;				// stress at every NR iteration after material model converged
	SymmetricTensor<2,3> old_strain;
	SymmetricTensor<2,3> old_t_strain;
	SymmetricTensor<2,3> t_strain_r;
	SymmetricTensor<2,3> old_lambda;
	double old_xi;
	double old_temperature;
	int old_transformation_status;
	int loading_status_0_iter;

};

template <int dim>
SymmetricTensor<4,dim>
get_stiffness (const int problem_type, const double &nu)
{
	SymmetricTensor<4,dim> tmp;
	const double E =1.;

	if (problem_type == 0){	// plane stress
		  const double const_1=E/(1.-nu*nu), const_2=nu*const_1, const_3=E/(2.*(1.+nu));
//		  std::cout<< "Plane stress case is considered..." << std::endl;
		  tmp[0][0][0][0]= const_1;
		  tmp[0][0][1][1]= const_2;
		  tmp[0][0][2][2]= 0.;
		  tmp[1][1][0][0]= const_2;
		  tmp[1][1][1][1]= const_1;
		  tmp[1][1][2][2]= 0.;
		  tmp[2][2][0][0]= 0.;
		  tmp[2][2][1][1]= 0.;
		  tmp[2][2][2][2]= 0.;

		  tmp[0][1][0][1]= const_3;
		  tmp[1][2][1][2]= 0.;
		  tmp[0][2][0][2]= 0.;
	}
	else
		{// plane strain or general 3D
//		std::cout<< "General case is considered..." << std::endl;
		const double lambda= (E*nu/((1.+nu)*(1.-2.*nu))), mu=E/(2.*(1.+nu));
		//tmp = lambda * outer_product(unit_symmetric_tensor<dim>(), unit_symmetric_tensor<dim>())+ 2.*mu*identity_tensor<dim>();

		//for (unsigned int i=0; i<dim; ++i)
		//	for (unsigned int j=0; j<dim; ++j)
		//		for (unsigned int k=0; k<dim; ++k)
		//			for (unsigned int l=0; l<dim; ++l)
		//				tmp[i][j][k][l] = (((i==k) && (j==l) ? mu : 0.0) +
		//						((i==l) && (j==k) ? mu : 0.0) +
		//						((i==j) && (k==l) ? lambda : 0.0));

	// plane strain and general
	  tmp[0][0][0][0]=lambda + 2*mu;
	  tmp[0][0][1][1]=lambda;
	  tmp[0][0][2][2]=lambda;
	  tmp[1][1][0][0]=lambda;
	  tmp[1][1][1][1]=lambda + 2*mu;
	  tmp[1][1][2][2]=lambda;
	  tmp[2][2][0][0]=lambda;
	  tmp[2][2][1][1]=lambda;
	  tmp[2][2][2][2]= lambda + 2*mu;

	  tmp[0][1][0][1]= mu;
	  tmp[1][2][1][2]= mu;
	  tmp[0][2][0][2]= mu;
		}

	return tmp;
}

template <int dim>
SymmetricTensor<4,dim>
get_compliance (const int problem_type, const double &nu)
{
	SymmetricTensor<4,dim> tmp;
	const double E =1.;
	const double const_1=1/E, const_2=-nu/E, const_3=(1+nu)/(2.*E);

	if (problem_type == 0){		// plane stress
//		std::cout<< "Plane stress case is considered..." << std::endl;
		tmp[0][0][0][0]= const_1;
		tmp[0][0][1][1]= const_2;
		tmp[0][0][2][2]= 0.;
		tmp[1][1][0][0]= const_2;
		tmp[1][1][1][1]= const_1;
		tmp[1][1][2][2]= 0.;
		tmp[2][2][0][0]= 0.;
		tmp[2][2][1][1]= 0.;
		tmp[2][2][2][2]= 0.;

		tmp[0][1][0][1]= const_3;
		tmp[1][2][1][2]= const_3;
		tmp[0][2][0][2]= const_3;
	}
	else
		{
//		std::cout<< "General case is considered..." << std::endl;
		//tmp = lambda * outer_product(unit_symmetric_tensor<dim>(), unit_symmetric_tensor<dim>())+ 2.*mu*identity_tensor<dim>();

		//for (unsigned int i=0; i<dim; ++i)
		//	for (unsigned int j=0; j<dim; ++j)
		//		for (unsigned int k=0; k<dim; ++k)
		//			for (unsigned int l=0; l<dim; ++l)
		//				tmp[i][j][k][l] = (((i==k) && (j==l) ? const_1 : 0.0) + ((i==l) && (j==k) ? const_1 : 0.0) - ((i==j) && (k==l) ? const_2 : 0.0));
		// plane strain
		tmp[0][0][0][0]= const_1;
		tmp[0][0][1][1]= const_2;
		tmp[0][0][2][2]= const_2;
		tmp[1][1][0][0]= const_2;
		tmp[1][1][1][1]= const_1;
		tmp[1][1][2][2]= const_2;
		tmp[2][2][0][0]= const_2;
		tmp[2][2][1][1]= const_2;
		tmp[2][2][2][2]= const_1;

		tmp[0][1][0][1]= const_3;
		tmp[1][2][1][2]= const_3;
		tmp[0][2][0][2]= const_3;
		}

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

/*class ParameterReader : public Subscriptor
  {
  public:
    ParameterReader(ParameterHandler &);
    void read_parameters(const std::string &);

  private:
    void declare_parameters();
    ParameterHandler &prm;
  };

  ParameterReader::ParameterReader(ParameterHandler &paramhandler)
    : prm(paramhandler)
  {}


  void ParameterReader::declare_parameters()
  {
	  prm.enter_subsection("Problem_type");
		{
		  prm.declare_entry("problem_type", "0", Patterns::Integer(), "plane_stress=0, plane_strain=1, 3Dcase=2");
		}
	  prm.leave_subsection();
  }


  void ParameterReader::read_parameters(const std::string &parameter_file)
  {
    declare_parameters();

    prm.parse_input(parameter_file);
  }
*/

template <int dim>
class SMAConstitutiveModel
{
public:
	SMAConstitutiveModel (ParameterHandler &prm/*const int &problem_type plane stress=0; plane_strain=1; general=2*/);

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
	ParameterHandler &prm;

	void set_model_parameters(void);
	const int problem_type, n_stress_strain;
	const double Af_0, As_0, Ms_0, Mf_0, c_A, c_M, H_max, E_M, E_A, nu, alpha_M, alpha_A, T_0, dE, dalpha, dc, tol_l, tol_h;
	const SymmetricTensor<4,dim> stiffness_tensor, delta_compliance;
	const SymmetricTensor<2,dim> delta_Alpha;
	const int hardening_model;
	double rodelta_s0, rodelta_u0, rodelta_c, rob_M, rob_A, mu_1, mu_2, ac_M, ac_A, prm_n_1, prm_n_2, prm_n_3, prm_n_4, prm_a_1, prm_a_2, prm_a_3, const_Y;
};


template<int dim>
SMAConstitutiveModel<dim>::SMAConstitutiveModel(ParameterHandler &param)
: prm(param)
, problem_type(prm.get_integer("problem_type"))
, n_stress_strain(dim == 2 ? 3 : 6)
, Af_0(prm.get_double("Af_0"))
, As_0(prm.get_double("As_0"))
, Ms_0(prm.get_double("Ms_0"))
, Mf_0(prm.get_double("Mf_0"))
, c_A(prm.get_double("c_A"))
, c_M(prm.get_double("c_A"))
, H_max(prm.get_double("Hmax"))
, E_M(prm.get_double("E_M"))
, E_A(prm.get_double("E_A"))
, nu(prm.get_double("nu"))
, alpha_M(prm.get_double("alpha_M"))
, alpha_A(prm.get_double("alpha_A"))
, T_0(prm.get_double("Tref"))
, dE(E_M-E_A)
, dalpha(alpha_M-alpha_A)
, dc((1./E_M)-(1./E_A))
, tol_l(prm.get_double("tol_l"))
, tol_h(prm.get_double("tol_h"))
, stiffness_tensor(get_stiffness<dim>(problem_type, nu))		// stiffness tensor per unit E
, delta_compliance(dc* get_compliance<dim>(problem_type, nu))													//(dc*invert(c_inv_perunit_E))									// delta compliance tensor
, delta_Alpha(dalpha*unit_symmetric_tensor<dim>())					// delta alpha tensor
, hardening_model(prm.get_integer("hardening_model"))		// Polynomial =10; Smooth hardening =20; Cosine =30; Lagoudas(2012) smooth hardening =40;
{
	set_model_parameters();
}

template<int dim>
void SMAConstitutiveModel<dim>::set_model_parameters()
{
	if (hardening_model == 10)			// polynomial hardening model
	{
	rodelta_s0 = -c_A*H_max;
	rodelta_u0 = rodelta_s0*0.5*(Ms_0+Af_0);
	rodelta_c = 0.;
	rob_M = -rodelta_s0*(Ms_0-Mf_0);
	rob_A = -rodelta_s0*(Af_0-As_0);
	mu_1 = 0.5*rodelta_s0*(Ms_0+Af_0);		// mu_1+-rodelta_u0
	mu_2 = 0.25*(rob_A-rob_M);
	const_Y = (0.25*rodelta_s0*(Mf_0+Ms_0-Af_0-As_0));
	ac_M = 0.;	// not used
	ac_A = 0.;	// not used
	prm_n_1 = 0.;	// not used
	prm_n_2 = 0.;	// not used
	prm_n_3 = 0.;	// not used
	prm_n_4 = 0.;	// not used
	prm_a_1 = 0.;	// not used
	prm_a_2 = 0.;	// not used
	prm_a_3 = 0.;	// not used
	}
	else if (hardening_model == 20)					// smooth hardening
	{
	rodelta_s0 = -c_A*H_max;
	rodelta_u0 = rodelta_s0*0.5*(Ms_0+Af_0);
	rodelta_c = 0.;
	rob_M = (-rodelta_s0*(Ms_0-Mf_0));
	rob_A = (-rodelta_s0*(Af_0-As_0));
	mu_1 = 0.;	// not used
	mu_2 = 0.;	// not used
	ac_M = 0.;	// not used
	ac_A = 0.;  // not used
	prm_n_1 = 1.;
	prm_n_2 = 1.;
	prm_n_3 = 1.;
	prm_n_4 = 1.;
	prm_a_1 = 0.;
	prm_a_2 = 0.;
	prm_a_3 = 0.;
	const_Y = (0.5*rodelta_s0*(Ms_0-Af_0));
	}
	else if (hardening_model == 30)				// cosine hardening
	{
	rodelta_s0 = (-c_A*H_max);
	rodelta_u0 = (0.5*rodelta_s0*(Ms_0+Af_0));
	rodelta_c = 0.;
	rob_M = 0.;			// not used
	rob_A = 0.;  		// not used
	mu_1 = (0.5*rodelta_s0*(Ms_0+Af_0)); 				// mu_1+rodelta_u0
	mu_2 = (0.25*rodelta_s0*(Ms_0-Mf_0+As_0-Af_0));
	ac_M = numbers::PI/(Ms_0-Mf_0);
	ac_A = (numbers::PI/(Af_0-As_0));
	prm_n_1 = 0.; 		// not used
	prm_n_2 = 0.; 		// not used
	prm_n_3 = 0.; 		// not used
	prm_n_4 = 0.; 		// not used
	prm_a_1 = 0.; 		// not used
	prm_a_2 = 0.; 		// not used
	prm_a_3 = 0.; 		// not used
	const_Y = (0.25*rodelta_s0*(Ms_0+Mf_0-As_0-Af_0));
	}
	else if (hardening_model == 40)			// smooth hardening modified
	{
	rodelta_s0 = -c_A*H_max;
	rodelta_u0 = (rodelta_s0*0.5*(Ms_0+Af_0));
	rodelta_c = (0.);
	rob_M = (0.);	 // not used
	rob_A = (0.);	 // not used
	mu_1 = (0.);	// not used
	mu_2 = (0.); 	// not used
	ac_M = (0.);	// not used
	ac_A = (0.);	 // not used
	prm_n_1 = (0.5);
	prm_n_2 = (0.5);
	prm_n_3 = (0.5);
	prm_n_4 = (0.5);
	prm_a_1 = (rodelta_s0* (Mf_0-Ms_0));
	prm_a_2 = (rodelta_s0* (As_0-Af_0));
	prm_a_3 = (-0.25* (1.+ 1./(prm_n_1+1.)- 1./(prm_n_2+1.))+0.25* prm_a_2* (1.+ 1./(prm_n_3+1.)- 1./(prm_n_4+1.)));
	const_Y = (0.5* rodelta_s0* (Ms_0-Af_0) -prm_a_3);
	}
}

template<int dim>
SymmetricTensor<4,dim> SMAConstitutiveModel<dim>::get_s_inv (const double &xi) const
{
	SymmetricTensor<4,dim> tmp;
	const double E = E_A + xi * dE;
	tmp = E * stiffness_tensor;

	return tmp;
}

template<int dim>
SymmetricTensor<2,dim> SMAConstitutiveModel<dim>::get_alpha(const double &xi) const
{
	const double alpha=alpha_A + xi * dalpha;
	SymmetricTensor<2,dim> tmp;
	tmp = alpha * unit_symmetric_tensor<dim>();

	return tmp;
}

template<int dim>
double SMAConstitutiveModel<dim>::get_delf_delxi(const double &xi, const int &transformation) const
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
 			y= rodelta_u0 + 0.5*prm_a_1*(1. + std::pow(xi,prm_n_1) - std::pow((1.-xi),prm_n_2)) + prm_a_3;
			//std::cout<< "get_delf_delxi=" << y <<std::endl;

 			break;
		}
		case -1:
		{
 			y= rodelta_u0 + 0.5*prm_a_2*(1. + std::pow(xi,prm_n_3) - std::pow((1.-xi),prm_n_4)) - prm_a_3;
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
double SMAConstitutiveModel<dim>::get_delphi_delxi(const double &xi, const int &transformation) const
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
 			y= -0.5* prm_a_1* (prm_n_1*std::pow(xi, (prm_n_1-1.)) + prm_n_2*std::pow((1.-xi), (prm_n_2-1.))) ;
			//std::cout<< "get_delphi_delxi=" << y <<std::endl;

			break;
		}
		case -1:
		{
 			y= 0.5* prm_a_2* (prm_n_3*std::pow(xi, (prm_n_3-1.)) + prm_n_4*std::pow((1.-xi), (prm_n_4-1.))) ;
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
SymmetricTensor<2,dim> SMAConstitutiveModel<dim>::get_lambda(const SymmetricTensor<2,dim> &tensor_, const int &transformation) const
{
	SymmetricTensor<2,dim> tmp_tensor;
	switch (transformation){
	case 1:						// forward transformation --> vec = stress vector
	{
		const double coeff = (3./2.) * H_max;
		const SymmetricTensor<2,dim> dev_stress = deviator(tensor_);
		//std::cout<< "trace= " << trace(dev_stress) << std::endl;

		const double dev_stress_norm = dev_stress.norm();

		if (dev_stress_norm < 1e-10)
		{
			tmp_tensor= coeff * dev_stress / (1. + std::sqrt(3./2.)* dev_stress_norm);	//coeff * unit_symmetric_tensor<dim>();
//			std::cout << "sigma_dev_norm " << dev_stress_norm << std::endl;
		}
		else
		{
			tmp_tensor = coeff * dev_stress / (std::sqrt(3./2.)* dev_stress_norm);
		}
		break;
	}
	case -1:						// reverse transformation --> vec = transformation strain at reversal
	{
		tmp_tensor = H_max * tensor_/(std::sqrt(2./3.)* tensor_.norm());
		break;
	}
	}
	//std::cout << "stress " << tensor_ << std::endl;
	//std::cout << "lambda " << tmp_tensor << std::endl;
	return tmp_tensor;
}

template<int dim>
double SMAConstitutiveModel<dim>::get_phi(const SymmetricTensor<2,dim> &stress, const double &temperature, const double &xi,
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
		phi = ((stress * lambda) + 0.5* (stress* delta_compliance* stress) + (stress * delta_Alpha) *(temperature - T_0)
				-rodelta_c*((temperature - T_0) - temperature * std::log(temperature/T_0)) + rodelta_s0* temperature
				- hardening_fn_value) - const_Y;
 		break;
	}
	case -1:
	{
		const SymmetricTensor<2,dim> lambda = get_lambda(t_strain_r, transformation);
		phi = -((stress * lambda) + 0.5* (stress * delta_compliance* stress) + (stress * delta_Alpha) *(temperature - T_0)
				-rodelta_c*((temperature - T_0) - temperature * std::log(temperature/T_0)) + rodelta_s0* temperature
				- hardening_fn_value) - const_Y;
		break;
	}
	}
	return phi;
}

template<int dim>
double SMAConstitutiveModel<dim>::get_delta_psi(const SymmetricTensor<2,dim> &stress, const double &temperature,
		const PointHistory &point_history_at_q) const
{
	const SymmetricTensor<2,dim> stress_0=point_history_at_q.old_stress;
	const SymmetricTensor<2,dim> lambda_0=point_history_at_q.old_lambda;
	const double temp_0=point_history_at_q.old_temperature;
	//std::cout << "lambda " << std::scientific << lambda_0 << std::endl;

	//const FullMatrix<double> delta_S = get_delta_s();
	//const Vector<double> delta_Alpha = get_delta_alpha();

	double tmp = ((stress * lambda_0) + 0.5* (stress * delta_compliance* stress) + (stress * delta_Alpha) *(temperature - T_0) + rodelta_s0* temperature)
					-((stress_0 * lambda_0) + 0.5* (stress_0* delta_compliance* stress_0) + (stress_0 * delta_Alpha) *(temp_0 - T_0) + rodelta_s0* temp_0 );
	return tmp;
}

template<int dim>
bool SMAConstitutiveModel<dim>::call_convex_cutting(const SymmetricTensor<2,dim> &strain, const double &temperature,
		const PointHistory &point_history_at_q, const unsigned int &nr_counter, SymmetricTensor<2,dim> &stress,
		SymmetricTensor<4,dim> &continuum_moduli_structural, SymmetricTensor<2,dim> &continuum_moduli_thermal, SymmetricTensor<2,dim> &lambda, SymmetricTensor<2,dim> &t_strain,
		SymmetricTensor<2,dim> &t_strain_r, double &xi, int &transformation_status, int &loading_status) const
{
	SymmetricTensor<2,dim> stress_iter = point_history_at_q.old_stress;				// stress at previous NR iteration
	SymmetricTensor<2,dim> lambda_iter = point_history_at_q.old_lambda;
	//SymmetricTensor<2,dim> strain_n = point_history_at_q.old_strain;
	SymmetricTensor<2,dim> t_strain_iter = point_history_at_q.old_t_strain;
	//const double temp_n = point_history_at_q.old_temperature;
	double xi_iter = point_history_at_q.old_xi;

	const SymmetricTensor<2,dim> t_strain_r0 = point_history_at_q.t_strain_r;
	//const SymmetricTensor<4,dim> tangent_moduli_struct_n = point_history_at_q.old_continuum_moduli_structural;
	//const SymmetricTensor<2,dim> tangent_moduli_ther_n = point_history_at_q.old_continuum_moduli_thermal;
	const int loading_status_0_iter = point_history_at_q.loading_status_0_iter;

	SymmetricTensor<4,dim> s_inv_iter = get_s_inv(xi_iter);
	SymmetricTensor<2,dim> alpha_iter = get_alpha(xi_iter);

	stress_iter = s_inv_iter*(strain - alpha_iter*(temperature -T_0) - t_strain_iter);
	//stress_iter = stress_iter + tangent_moduli_struct_n* (strain -strain_n) + tangent_moduli_ther_n* (temperature - temp_n);
	//std::cout << "moduli " << std::scientific << tangent_moduli_struct_n[0][0][0][0] << " and strain " << (strain[0][0] - strain_n[0][0]) << std::endl;

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

		if (transformation_status==1 && lambda_iter.norm() < tol_h)
			{
			lambda_iter = get_lambda(stress_iter, transformation_status);
			//std::cout << "lambda_norm " << lambda_iter.norm() << std::endl;
			}

		SymmetricTensor<2,dim> del_phi_del_sig = sgn_xi_dot* (delta_compliance* stress_iter + delta_Alpha* (temperature -T_0) + lambda_iter);
		double del_phi_del_xi = get_delphi_delxi(xi_iter, transformation_status);

		//const double denom = (sgn_xi_dot* (del_phi_del_sig* s_inv_iter* del_phi_del_sig) - del_phi_del_xi);
		double delta_xi = phi/ (sgn_xi_dot* (del_phi_del_sig* s_inv_iter* del_phi_del_sig) - del_phi_del_xi);
		const double xi_tmp = xi_iter + delta_xi;
		//std::cout << " phi= " << phi << ", denom= " << denom << ", delta_xi= " << delta_xi << "and xi_tmp= " << xi_tmp << std::endl;
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
			del_phi_del_sig = sgn_xi_dot* (delta_compliance* stress_iter + delta_Alpha* (temperature -T_0) + lambda_iter);
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
	NeumannBoundary(const double &time, const double &Pmax, const double &Tmax);

	virtual void vector_value(const Point<dim> &p, Vector<double> &  values) const override;
	virtual void vector_value_list(const std::vector<Point<dim>> &points, std::vector<Vector<double>> &  value_list) const override;
private:
	const double present_time, Pmax, Tmax;
};


template <int dim>
NeumannBoundary<dim>::NeumannBoundary(const double &time, const double &Pmax, const double &Tmax)
: Function<dim>(dim)
  , present_time(time)
  , Pmax(Pmax)
  , Tmax(Tmax)
  {}


template <int dim>
inline void NeumannBoundary<dim>::vector_value(const Point<dim> & /*p*/, Vector<double> &values) const
{
	AssertDimension(values.size(), dim);
	//const double p0   = 400.e6;				// N /per unit length
	//const double total_time =2.;
	values    = 0;
	if (present_time <= 1.)
		values(0) = Pmax *((present_time - 0.)/(1.-0.));			// total time is 1s
	else if (present_time <= 2.)
		values(0) = Pmax *((2.0 - present_time)/(2. - 1.));
	else
		values(0) = 0.;
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


//**************************************************************************************************************************
//**************************************************************************************************************************

template <int dim>
class FEAnalysis
{
public:
	FEAnalysis(ParameterHandler &prm);
	void run();

	static void declare_parameters(ParameterHandler &prm);

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

//	void map_scalar_qpoint_to_dof(const Vector<double> &vec, Vector<double> &avg_vec_on_cell_vertices) const;
//
//	void map_vec_qpoint_to_dof(const unsigned int &n_components, const std::vector<Vector<double>> &vec,
//			std::vector<Vector<double>> &avg_vec_on_cell_vertices) const;

	void print_conv_header(void);
	void print_conv_footer();

	MPI_Comm mpi_communicator;

	parallel::distributed::Triangulation<dim> triangulation;

	ParameterHandler &prm;

	DoFHandler<dim>    dof_handler;
	FESystem<dim> fe;

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

	/*AffineConstraints<double>*/ConstraintMatrix all_constraints;
	const QGauss<dim> quadrature_formula;

	//SparsityPattern      sparsity_pattern;
	//SparseMatrix<double> system_matrix;
	//FullMatrix<double> system_matrix_full;
	//Vector<double> solution, incremental_solution;

	LA::MPI::SparseMatrix system_matrix;
	LA::MPI::Vector       locally_relevant_solution;
	LA::MPI::Vector       locally_owned_solution, locally_owned_incremental_solution, residue_vector;

	std::vector<PointHistory> quadrature_point_history;			// dim has to be only 3


	//Vector<double> external_force_vector;
	//Vector<double> internal_force_vector;
//	Vector<double> residue_vector;

	const unsigned int study_type;
	double present_time, end_time;
	double time_step;
	//static const double moduli, nu;
	const SMAConstitutiveModel<3> constitutive_law_sma;				// dim has to be only 3
	const double T_initial, P_max, T_max;
	std::string file_name;
	//unsigned int n_strain_components;

    ConditionalOStream pcout;
    TimerOutput        computing_timer;
};

/*template <int dim>
  const double ElasticProblem<dim> :: moduli = 70.0e9;

  template <int dim>
  const double ElasticProblem<dim> :: nu = 0.3;*/

template<int dim>
void FEAnalysis<dim>::declare_parameters(ParameterHandler &prm)
{
	prm.declare_entry("problem_type", "0", Patterns::Integer(0, 2), "plane_stress=0, plane_strain=1, 3Dcase=2");
	prm.declare_entry("Pmax", "32.6e6", Patterns::Double(), " maximum load ");
	prm.declare_entry("Tmax", "315.", Patterns::Double(), "maximum temperature");
	prm.declare_entry("total_time", "2.", Patterns::Double(), " total time of analysis ");
	//prm.declare_entry("n_steps", "100", Patterns::Integer(), "total load steps");
	prm.declare_entry("time_step", "0.005 ", Patterns::Double(), " time step for analysis ");
	prm.declare_entry("Af_0", "288.15", Patterns::Double(), " Reverse transformation finish temperature");
	prm.declare_entry("As_0", "258.15", Patterns::Double(), " Reverse transformation start temperature ");
	prm.declare_entry("Ms_0", "258.15", Patterns::Double(), "Forward transformation start temperature");
	prm.declare_entry("Mf_0", "218.15", Patterns::Double(), "Forward transformation finish temperature");
	prm.declare_entry("c_M", "6e6", Patterns::Double(), " slope of forward transformation zone ");
	prm.declare_entry("c_A", "6e6", Patterns::Double(), " slope of reverse transformation zone ");
	prm.declare_entry("Hmax", "0.047", Patterns::Double(), "Max. transformation strain");
	prm.declare_entry("E_M", "38.2e9", Patterns::Double(), "modulus in martensite phase ");
	prm.declare_entry("E_A", "48.5e9", Patterns::Double(), "modulus in austenite phase ");
	prm.declare_entry("nu", "0.42", Patterns::Double(), "Poisson's ratio ");
	prm.declare_entry("alpha_M", "22e-6", Patterns::Double(), "thermal expansion coeff in martensite phase ");
	prm.declare_entry("alpha_A", "10e-6", Patterns::Double(), "thermal expansion coeff in austenite phase ");
	prm.declare_entry("Tref", "300.", Patterns::Double(), "reference temperature ");

	prm.declare_entry("tol_l", "1e-3", Patterns::Double(), "lower tolerance");
	prm.declare_entry("tol_h", "1e-8", Patterns::Double(), "higher tolerance");
	prm.declare_entry("hardening_model", "10", Patterns::Integer(10, 40), "polynomial=10, smooth=20, cosine=30, smooth2012=40");
	prm.declare_entry("filename", "new_", Patterns::Anything(), " output file name");

}

template <int dim>
FEAnalysis<dim>::FEAnalysis(ParameterHandler &param)
: mpi_communicator(MPI_COMM_WORLD)
, triangulation(mpi_communicator,
        typename Triangulation<dim>::MeshSmoothing(
          Triangulation<dim>::smoothing_on_refinement |
          Triangulation<dim>::smoothing_on_coarsening))
, prm(param)
, dof_handler(triangulation)
, fe(FE_Q<dim>(2), dim)
, quadrature_formula(fe.degree + 1)
, study_type(prm.get_integer("problem_type"))											/* plane_stress=0; plane_strain=1; 3D general=2*/
, present_time(0.)
, end_time(prm.get_double("total_time"))
, time_step(prm.get_double("time_step"))
, constitutive_law_sma(prm)
, T_initial(prm.get_double("Tref"))
, P_max(prm.get_double("Pmax"))
, T_max(prm.get_double("Tmax"))
, file_name(prm.get("filename"))
, pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
, computing_timer(mpi_communicator,
                    pcout,
                    TimerOutput::never,
                    TimerOutput::wall_times)
{
	//n_strain_components = (study_type == 2 ? 6 : 3); 					//plane stress or strain case=3; 3D case=6
}


template <int dim>
void FEAnalysis<dim>::create_mesh(/*const unsigned int cycle*/)
{
GridIn<dim> gridin;
	gridin.attach_triangulation(triangulation);
	std::ifstream f("mj_spring_r1.inp");
	gridin.read_abaqus(f);

//	for(auto &face : triangulation.active_face_iterators())
		    for (typename Triangulation<dim>::active_cell_iterator
		         cell=triangulation.begin_active();
		         cell!=triangulation.end(); ++cell)
		    	for (unsigned int f=0; f <GeometryInfo<dim>::faces_per_cell; ++f)
			  		    if (cell->face(f)->at_boundary())
			  		    {
			  		    	if ((cell->face(f)->center()[2] - 0 < 1e-6)
			  		    		&&(cell->face(f)->center()[1] + 0.0779 > 1e-6 )&& (cell->face(f)->center()[1] + 0.0656054 < 1e-6))
			  		    		{
			  		    		cell->face(f)->set_boundary_id (1);
			  		    			std::cout<<"boundary_id 1 set"<<std::endl;
			  		    		}
			  		    	else if((cell->face(f)->center()[1] - (0.066037)) < 1e-6 && (cell->face(f)->center()[1] - (0.053963) > 1e-6)
			  		    		&&(cell->face(f)->center()[2] - 0 < 1e-6))
			  		    	{
			  		    		cell->face(f)->set_boundary_id (3);
			  		    	  	std::cout<<"boundary_id 3 set"<<std::endl;
			  		    	}
			  		    }
//		    triangulation.refine_global(1);
//	 TimerOutput::Scope t(computing_timer, "create_mesh");
//	//std::cout << "Cycle " << cycle << ':' << std::endl;
//
//	//      if (cycle == 0){
//	//********************** simple square domain
//	GridGenerator::hyper_cube(triangulation, 0, 1);
//	// set the right hand side boundary as boundary_id=3
//	if (dim==2)
//	{
//		//for(auto &face : triangulation.active_face_iterators())
//	    for (typename Triangulation<dim>::active_cell_iterator
//	         cell=triangulation.begin_active();
//	         cell!=triangulation.end(); ++cell)
//        for (unsigned int f=0; f <GeometryInfo<dim>::faces_per_cell; ++f)
//			if (cell->face(f)->at_boundary())
//			{
//				if (std::fabs(cell->face(f)->center()[0] - 0.) < 1e-5)
//					cell->face(f)->set_boundary_id(1);
//				else if (std::fabs(cell->face(f)->center()[1] - 0.) < 1e-5)
//					cell->face(f)->set_boundary_id(2);
//				else if (std::fabs(cell->face(f)->center()[0] - 1.0) < 1e-5)
//					cell->face(f)->set_boundary_id(3);
//
//			}
//	}
//	else if (dim==3)
//	{
//		//for(auto &face : triangulation.active_face_iterators())
//	    for (typename Triangulation<dim>::active_cell_iterator
//	         cell=triangulation.begin_active();
//	         cell!=triangulation.end(); ++cell)
//        for (unsigned int f=0; f <GeometryInfo<dim>::faces_per_cell; ++f)
//			if (cell->face(f)->at_boundary())
//			{
//				if (std::fabs(cell->face(f)->center()[0] - .0) < 1e-5)
//					cell->face(f)->set_boundary_id(1);
//				else if (std::fabs(cell->face(f)->center()[1] - 0.) < 1e-5)
//					cell->face(f)->set_boundary_id(2);
//				else if (std::fabs(cell->face(f)->center()[0] - 1.0) < 1e-5)
//					cell->face(f)->set_boundary_id(3);
//				else if (std::fabs(cell->face(f)->center()[2] - 0.) < 1e-5)
//					cell->face(f)->set_boundary_id(4);
//			}
//	}
//  triangulation.refine_global(2);
	//  }

	//**************** Taking a rectangular domain
	/*std::vector<unsigned int> repetitions(2);
	        	   repetitions[0] = 3;
	        	   repetitions[1] = 2;
	        	   GridGenerator::subdivided_hyper_rectangle(triangulation,
	        	                                             repetitions,
	        	                                             Point<2>(0., 0.),
	        	                                             Point<2>(4.0, 1.0));
	        	   // set the right hand side boundary as boundary_id=1
	        	   for(auto &face : triangulation.active_face_iterators())
	        		   if (face->at_boundary())
	        	   {
	        		   if (std::fabs(face->center()[1] - 0.0) < 1e-5)
	        			   face->set_boundary_id(2);
	        		   else if (std::fabs(face->center()[0] - 4.0) < 1e-5)
	        			   face->set_boundary_id(1);
	        		   else if (std::fabs(face->center()[1] - 1.0) < 1e-5)
	        		   	   face->set_boundary_id(3);
	        	   }
	              triangulation.refine_global(2); */

	// *********************  Plate with a hole problem *********************

	/*const Point<2> hole_origin(0.,0.);
	        	  // plate dimensions a x b, hole radius = r
	        	  const double r=1.0, a=3.0*r, b=2.0*r, alpha= numbers::PI/4.0;

	        	  // defining vertices and cells and material_id
	        	  const std::vector<Point<2>> vertices = {{r, 0.0}, {b, 0.0}, {b,b}, {r * std::cos(alpha), r * std::sin(alpha)}, {0.0, r}, {0.0, b}, {a,0.0}, {a,b}};

	        	  const std::vector<std::array<int, GeometryInfo<dim>::vertices_per_cell>>
	        	  cell_vertices = {{{0, 1, 3, 2}},
	        	                    {{4, 3, 5, 2}},
	        	                    {{1, 6, 2, 7}}
	        	          	  	  };
	        	  const unsigned int n_cells = cell_vertices.size();

	        	  std::vector<CellData<dim>> cells(n_cells, CellData<dim>());
	        	  for (unsigned int i = 0; i < n_cells; ++i)
	        	     {
	        		  for (unsigned int j = 0; j < cell_vertices[i].size(); ++j)
	        			  cells[i].vertices[j] = cell_vertices[i][j];

	        	         cells[i].material_id = 0;
	        	      }

	        	  triangulation.create_triangulation(vertices, cells, SubCellData());

	        	   // we need to set_boundary_id
	        	        for (const auto &face : triangulation.active_face_iterators())
	        	        if (face->at_boundary())
	        	        {
	        	            if (hole_origin.distance(face->center()) < r )		// hole face
	        	                face->set_boundary_id(4);
	        	            else if (std::abs(face->center()[1]) < 1.0e-6)			// lower face
	        	                face->set_boundary_id(2);
	        	            else if (std::abs(face->center()[0] - a) < 1.0e-6)		// right face
	        	                face->set_boundary_id(1);
	        	            else if (std::abs(face->center()[1] - b) < 1.0e-6)		// upper face
	        	                face->set_boundary_id(3);
	        	            else if (std::abs(face->center()[0]) < 1.0e-6)			// left face
	        	                face->set_boundary_id(0);
	        	      }
	        	   // defining manifold id
	        	        const SphericalManifold<2> center_manifold(hole_origin);
	        	        triangulation.set_all_manifold_ids_on_boundary(4, 1);
	        	        triangulation.set_manifold(1, center_manifold);

	        	        triangulation.refine_global(1);

	        	        for(unsigned int step=0; step < 4; step++)
	        	        {
	        	        for(auto &cell : triangulation.active_cell_iterators())
	        	        {
	        	            for(unsigned int v_k = 0; v_k <  GeometryInfo<2>::vertices_per_cell; ++v_k )
	        	            {
	        	                if(std::fabs(hole_origin.distance(cell->vertex(v_k)) - r) < 1e-5)
	        	                    cell->set_refine_flag();
	        	                break;
	        	            }
	        	        }
	        	      triangulation.execute_coarsening_and_refinement();
	        	        }

	            }*/
	//****************************************************************************
	//  else
	//  refine_grid();
}

template <int dim>
void FEAnalysis<dim>::setup_system()
{
	TimerOutput::Scope t(computing_timer, "setup");

	dof_handler.distribute_dofs(fe);
	locally_owned_dofs = dof_handler.locally_owned_dofs();
	DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

	locally_relevant_solution.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
	locally_owned_incremental_solution.reinit(locally_owned_dofs, mpi_communicator);
	locally_owned_solution.reinit(locally_owned_dofs, mpi_communicator);
	residue_vector.reinit(locally_owned_dofs, mpi_communicator);

//	solution.reinit(dof_handler.n_dofs());
//	incremental_solution.reinit(dof_handler.n_dofs());
	//external_force_vector.reinit(dof_handler.n_dofs());
	//internal_force_vector.reinit(dof_handler.n_dofs());
//	residue_vector.reinit(dof_handler.n_dofs());
	//system_matrix_full.reinit(dof_handler.n_dofs(),dof_handler.n_dofs());

	all_constraints.clear();
	all_constraints.reinit(locally_relevant_dofs);
	DoFTools::make_hanging_node_constraints(dof_handler, all_constraints);

//	if (dim == 2)
//	{
//		const FEValuesExtractors::Scalar          x_component(0), y_component(1);
//		VectorTools::interpolate_boundary_values(dof_handler, 1, Functions::ZeroFunction<dim>(dim), all_constraints, fe.component_mask(x_component));
//		VectorTools::interpolate_boundary_values(dof_handler, 2, Functions::ZeroFunction<dim>(dim), all_constraints, fe.component_mask(y_component));
//	}
//	else if(dim == 3)
//	{
//		const FEValuesExtractors::Scalar          x_component(0), y_component(1), z_component(2);
//		VectorTools::interpolate_boundary_values(dof_handler, 1, Functions::ZeroFunction<dim>(dim), all_constraints, fe.component_mask(x_component));
//		VectorTools::interpolate_boundary_values(dof_handler, 2, Functions::ZeroFunction<dim>(dim), all_constraints, fe.component_mask(y_component));
//		VectorTools::interpolate_boundary_values(dof_handler, 4, Functions::ZeroFunction<dim>(dim), all_constraints, fe.component_mask(z_component));
//	}
	if (dim == 2)
			      {
			    	  const FEValuesExtractors::Scalar          x_component(0), y_component(1);
			    	  VectorTools::interpolate_boundary_values(dof_handler, 1, Functions::ZeroFunction<dim>(dim), all_constraints);
			//    	  VectorTools::interpolate_boundary_values(dof_handler, 2, Functions::ZeroFunction<dim>(dim), boundary_values, fe.component_mask(y_component));
			      }
			      else if(dim == 3)
			      {
			    	  const FEValuesExtractors::Scalar          x_component(0), y_component(1), z_component(2);
			    	  VectorTools::interpolate_boundary_values(dof_handler, 1, Functions::ZeroFunction<dim>(dim), all_constraints);
			      }

	all_constraints.close();

	DynamicSparsityPattern dsp(locally_relevant_dofs);
	DoFTools::make_sparsity_pattern(dof_handler, dsp, all_constraints, /*keep_constrained_dofs = */ false);

	SparsityTools::distribute_sparsity_pattern(dsp, dof_handler.n_locally_owned_dofs_per_processor(),
			mpi_communicator, locally_relevant_dofs);

	system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);
}


template <int dim>
void FEAnalysis<dim>::assemble_system()
{
	TimerOutput::Scope t(computing_timer, "assemble_system");
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
	const NeumannBoundary<dim> neumann_boundary(present_time, P_max, T_max);
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
		if (cell->is_locally_owned())
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

		//for (const unsigned int q_point : fe_values.quadrature_point_indices())
		for (unsigned int q_point=0; q_point<n_q_points;++q_point)
		{
			const SymmetricTensor<4,dim> linearised_stress_strain_matrix=convert_symmetric_tensor_3d_to_2d<dim>(local_quadrature_points_history[q_point].continuum_moduli_structural);
			const SymmetricTensor<2,dim> linearised_thermal_moduli = convert_symmetric_tensor_3d_to_2d<dim>(local_quadrature_points_history[q_point].continuum_moduli_thermal);
			const SymmetricTensor<2,dim> stress_at_qpoint = convert_symmetric_tensor_3d_to_2d<dim>(local_quadrature_points_history[q_point].stress);
			const double temp_n = local_quadrature_points_history[q_point].old_temperature;
			// For 3D problem no need to convert, thus use the following only
			//const SymmetricTensor<4,dim> linearised_stress_strain_matrix=local_quadrature_points_history[q_point].continuum_moduli;

			//for (const unsigned int i : fe_values.dof_indices())
	          for (unsigned int i=0; i<dofs_per_cell; ++i)
			{
				const SymmetricTensor<2,dim> eps_phi_i = get_linear_strain (fe_values, i, q_point);
				//for (const unsigned int j : fe_values.dof_indices())
		          for (unsigned int j=0; j<dofs_per_cell; ++j)
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

		//for (const unsigned int i : fe_values.dof_indices())
        for (unsigned int i=0; i<dofs_per_cell; ++i)
		{
			const unsigned int component_i = fe.system_to_component_index(i).first;
			//for (const unsigned int q_point : fe_values.quadrature_point_indices())
	        for (unsigned int q_point=0; q_point<n_q_points;++q_point)
			{
				cell_external_force(i) += ( rhs_values[q_point][component_i]
																* fe_values.shape_value(i,q_point) * fe_values.JxW(q_point));
			}
		}


//		for (const auto &face : cell->face_iterators())
        for (unsigned int f=0; f <GeometryInfo<dim>::faces_per_cell; ++f)
			if (cell->face(f)->at_boundary() && (cell->face(f)->boundary_id() == 3))
			{
				fe_face_values.reinit(cell, f);
				//for (const unsigned int q_point : fe_face_values.quadrature_point_indices())
				for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
				{
					neumann_boundary.vector_value_list(fe_face_values.get_quadrature_points(), neumann_values);
//					for (const unsigned int i : fe_values.dof_indices())
			          for (unsigned int i=0; i<dofs_per_cell; ++i)
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

		all_constraints.distribute_local_to_global(cell_matrix, cell_residue, local_dof_indices, system_matrix, residue_vector);
	}
	//std::map<types::global_dof_index, double> boundary_values;
	//MatrixTools::apply_boundary_values(boundary_values, system_matrix, incremental_solution, residue_vector);
    system_matrix.compress(VectorOperation::add);
    residue_vector.compress(VectorOperation::add);
}


template <int dim>
void FEAnalysis<dim>::solve()
{
	/*SolverControl            solver_control(1000, 1e-12);
    SolverCG<Vector<double>> cg(solver_control);

    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);

    cg.solve(system_matrix, increment_solution, residue_vector, preconditioner);*/
//
//	SparseDirectUMFPACK A_direct;
//	A_direct.initialize(system_matrix);
//	A_direct.vmult(incremental_solution, residue_vector);
//
//	all_constraints.distribute(incremental_solution);
	TimerOutput::Scope t(computing_timer, "solve");
	SolverControl solver_control(dof_handler.n_dofs(), 1e-15);

	#ifdef USE_PETSC_LA
	    LA::SolverCG solver(solver_control, mpi_communicator);
	#else
	    LA::SolverCG solver(solver_control);
	#endif

	    LA::MPI::PreconditionAMG preconditioner;

	    LA::MPI::PreconditionAMG::AdditionalData data;

	#ifdef USE_PETSC_LA
	    data.symmetric_operator = true;
	#else
	    /* Trilinos defaults are good */
	#endif

	    preconditioner.initialize(system_matrix, data);

	    solver.solve(system_matrix,
	                     locally_owned_incremental_solution,
	                     residue_vector,
	                     preconditioner);

	    pcout << "   Solved in " << solver_control.last_step() << " iterations."
	              << std::endl;

	    all_constraints.distribute(locally_owned_incremental_solution);
}



template <int dim>
void FEAnalysis<dim>::refine_grid()
{
	TimerOutput::Scope t(computing_timer, "refine");

	Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

   /* KellyErrorEstimator<dim>::estimate(dof_handler,
                                       QGauss<dim - 1>(fe.degree + 1),
                                       {},
                                       solution,
                                       estimated_error_per_cell);

    GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                    estimated_error_per_cell,
                                                    0.3,
                                                    0.03);*/
    KellyErrorEstimator<dim>::estimate(dof_handler,
    		QGauss<dim - 1>(fe.degree + 1), std::map<types::boundary_id, const Function<dim> *>(),
			locally_relevant_solution, estimated_error_per_cell);

    parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(triangulation,
    		estimated_error_per_cell, 0.3, 0.03);

    triangulation.execute_coarsening_and_refinement();
}


template <int dim>
void FEAnalysis<dim>::setup_quadrature_point_history()
{
	triangulation.clear_user_data();
	{
		std::vector<PointHistory > tmp;
		tmp.swap (quadrature_point_history);
	}
	quadrature_point_history.resize (triangulation.n_locally_owned_active_cells() * quadrature_formula.size());

	unsigned int history_index = 0;
	//for (auto &cell : triangulation.active_cell_iterators())
	for (typename DoFHandler<dim>::active_cell_iterator
			cell = dof_handler.begin_active();
			cell != dof_handler.end(); ++cell)
		if (cell->is_locally_owned())
		{
			cell->set_user_pointer (&quadrature_point_history[history_index]);
			history_index += quadrature_formula.size();
		}
	Assert (history_index == quadrature_point_history.size(), ExcInternalError());
}

template <int dim>
void FEAnalysis<dim>::setup_initial_quadrature_point_history()
{
	FEValues<dim> fe_values (fe, quadrature_formula,
			update_values | update_gradients |
			update_quadrature_points);
	const unsigned int n_q_points = quadrature_formula.size();
	//std::vector<std::vector<Tensor<1, dim>>> solution_grads(n_q_points, std::vector<Tensor<1, dim>>(dim));

	const SymmetricTensor<2,3> tmp_tensor=0.*unit_symmetric_tensor<3>();
	const double xi_0=1e-6;

	const SymmetricTensor<4,3> continuum_moduli_structural_0 = constitutive_law_sma.get_s_inv(xi_0);
	const SymmetricTensor<2,3> continuum_moduli_thermal_0 = - (constitutive_law_sma.get_s_inv(xi_0) * constitutive_law_sma.get_alpha(xi_0));

	SymmetricTensor<2,3> strain_0 = tmp_tensor;
	SymmetricTensor<2,3> stress_0 = continuum_moduli_structural_0* strain_0;

	SymmetricTensor<2,3> t_strain_r0 = 0.* unit_symmetric_tensor<3>();
	SymmetricTensor<2,3> lambda_0 = tmp_tensor;

//	std::cout<< "tmp tensor = " << tmp_tensor << std::endl;

	//for(auto &cell : dof_handler.active_cell_iterators())
    for (typename DoFHandler<dim>::active_cell_iterator
         cell = dof_handler.begin_active();
         cell != dof_handler.end(); ++cell)
		if (cell->is_locally_owned())
		{
		PointHistory *local_quadrature_points_history = reinterpret_cast<PointHistory *>(cell->user_pointer());
		Assert (local_quadrature_points_history >= &quadrature_point_history.front(),ExcInternalError());
		Assert (local_quadrature_points_history < &quadrature_point_history.back(),ExcInternalError());

		fe_values.reinit (cell);
		//fe_values.get_function_gradients(solution, solution_grads);

		for (unsigned int q = 0; q < n_q_points; ++q)
		{

			local_quadrature_points_history[q].continuum_moduli_structural = continuum_moduli_structural_0;
			local_quadrature_points_history[q].continuum_moduli_thermal = continuum_moduli_thermal_0;
			local_quadrature_points_history[q].old_continuum_moduli_structural = continuum_moduli_structural_0;
			local_quadrature_points_history[q].old_continuum_moduli_thermal = continuum_moduli_thermal_0;
			local_quadrature_points_history[q].old_stress = stress_0;
			local_quadrature_points_history[q].stress = stress_0;
			local_quadrature_points_history[q].old_strain = strain_0;
			local_quadrature_points_history[q].old_t_strain = strain_0;
			local_quadrature_points_history[q].t_strain_r = t_strain_r0;
			local_quadrature_points_history[q].old_lambda =  lambda_0;
			local_quadrature_points_history[q].old_xi = xi_0;
			local_quadrature_points_history[q].old_temperature = Tref;
			local_quadrature_points_history[q].old_transformation_status = 100;
			local_quadrature_points_history[q].loading_status_0_iter = 1;

		}
	}
}

template <int dim>
void FEAnalysis<dim>::update_quadrature_point_history(const bool &cond, const unsigned int &nr_counter)
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

	//for(auto &cell : dof_handler.active_cell_iterators())
    for (typename DoFHandler<dim>::active_cell_iterator
         cell = dof_handler.begin_active();
         cell != dof_handler.end(); ++cell)
		if (cell->is_locally_owned())
		{
		PointHistory *local_quadrature_points_history = reinterpret_cast<PointHistory *>(cell->user_pointer());
		Assert (local_quadrature_points_history >= &quadrature_point_history.front(),ExcInternalError());
		Assert (local_quadrature_points_history < &quadrature_point_history.back(),ExcInternalError());

		fe_values.reinit (cell);
		fe_values.get_function_gradients(locally_relevant_solution, solution_grads);
		right_hand_side(fe_values.get_quadrature_points(), present_time, rhs_values, body_temp);

		//std::cout << "present time"  << time << " body temp = " << body_temp[0] << std::endl;

		//for (unsigned int q_point : fe_values.quadrature_point_indices())
		for (unsigned int q_point=0; q_point<quadrature_formula.size(); ++q_point)
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

			outcome = constitutive_law_sma.call_convex_cutting(strain_new, temp_new, local_quadrature_points_history[q_point],
					nr_counter, stress_new, continuum_moduli_structural_new, continuum_moduli_thermal_new, lambda_new, t_strain_new, t_strain_r_new, xi_new,
					transformation_status, loading_status);
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
				//local_quadrature_points_history[q_point].old_strain = strain_new;
				//local_quadrature_points_history[q_point].old_temperature = temp_new;
				local_quadrature_points_history[q_point].continuum_moduli_structural = continuum_moduli_structural_new;
				local_quadrature_points_history[q_point].continuum_moduli_thermal = continuum_moduli_thermal_new;

				if (nr_counter == 0)
					local_quadrature_points_history[q_point].loading_status_0_iter = loading_status;
			}
			else	// N-R converged for the current load step
			{
				local_quadrature_points_history[q_point].continuum_moduli_structural = continuum_moduli_structural_new;
				local_quadrature_points_history[q_point].continuum_moduli_thermal = continuum_moduli_thermal_new;
				local_quadrature_points_history[q_point].old_continuum_moduli_structural = continuum_moduli_structural_new;
				local_quadrature_points_history[q_point].old_continuum_moduli_thermal = continuum_moduli_thermal_new;
				local_quadrature_points_history[q_point].old_stress = stress_new;
				local_quadrature_points_history[q_point].stress = stress_new;
				local_quadrature_points_history[q_point].old_strain = strain_new;
				local_quadrature_points_history[q_point].old_t_strain = t_strain_new;
				local_quadrature_points_history[q_point].t_strain_r = t_strain_r_new;
				local_quadrature_points_history[q_point].old_xi = xi_new;
				local_quadrature_points_history[q_point].old_temperature = temp_new;
				local_quadrature_points_history[q_point].old_lambda = lambda_new;
				local_quadrature_points_history[q_point].old_transformation_status = transformation_status;
				local_quadrature_points_history[q_point].loading_status_0_iter = loading_status;
			}
		}
	}
	//if (cond == false)
	pcout << nr_counter <<"		" << outcome << "	" <<std::scientific << strain_new[0][0] << "  	"
			<< stress_new[0][0] << "	"  << temp_new << "	" << xi_new <<"  	"
			<< t_strain_new[0][0] <<"  	"<< continuum_moduli_structural_new[0][0][0][0] <<"  	";
}


template<int dim>
void FEAnalysis<dim>::call_Newton_Raphson_method()
{
	unsigned int nr_counter = 0;
	double residue_norm=0., residue_norm_NR_0 = 0., delta_solution_norm=0.;
	const double tol = 1e-3;
	while(true)
	{
		locally_owned_incremental_solution = 0.;
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
//		locally_owned_solution.add(1., locally_owned_incremental_solution);
		locally_owned_solution += locally_owned_incremental_solution;
	    locally_relevant_solution = locally_owned_solution;


		delta_solution_norm = (locally_owned_incremental_solution.l2_norm()/locally_owned_solution.l2_norm());
		//std::cout << "Change in solution norm " << delta_solution_norm << std::endl;

		if (residue_norm < tol || nr_counter > 50){
			update_quadrature_point_history(true, nr_counter);
			pcout << std::fixed << std::setprecision(3) << std::setw(7)
															  << std::scientific << residue_norm << "     " << (residue_norm/residue_norm_NR_0)
															  << "	" << delta_solution_norm <<std::endl;
			break;
		}
		else
		{
			update_quadrature_point_history(false, nr_counter);		// update the stress and strain with new solution
			pcout << std::fixed << std::setprecision(3) << std::setw(7)
															  << std::scientific << residue_norm << "     " << (residue_norm/residue_norm_NR_0)
															  << "	" << delta_solution_norm <<std::endl;
			nr_counter++;
		}
	}
}


template <int dim>
void FEAnalysis<dim>::run()
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

	pcout << "Running with "
		  #ifdef USE_PETSC_LA
		            << "PETSc"
		  #else
		            << "Trilinos"
		  #endif
		            << " on " << Utilities::MPI::n_mpi_processes(mpi_communicator)
		            << " MPI rank(s)..." << std::endl;

	create_mesh();
	pcout << "   Number of active cells:     " << triangulation.n_active_cells() << std::endl;

	setup_quadrature_point_history();

	setup_system();
	pcout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
                        		<< std::endl;

	locally_owned_solution = 0.;
	locally_relevant_solution = 0.;
	setup_initial_quadrature_point_history();	// to make sure the initial stresses are zero at quadrature points

	unsigned int time_step_counter = 1;
	const unsigned int n_time_steps = (std::round(end_time / time_step));
	//current_time = 0.; //initiated in the constructor
	while (time_step_counter <= n_time_steps)
	{
		present_time += time_step;
		if (present_time > end_time)
		{
			time_step -= (present_time - end_time);
			present_time = end_time;
		}
		pcout << "--------- Current time stamp ---------------------->> " << present_time << std :: endl;
		print_conv_header();
		// N-R iteration
		call_Newton_Raphson_method();

		output_results(time_step_counter);
		time_step_counter++;
		print_conv_footer();
	}

	computing_timer.print_summary();
	computing_timer.reset();
}

template <int dim>
void FEAnalysis<dim>::output_results(const unsigned int timestep_counter) const
{
//	TimerOutput::Scope t(computing_timer, "output_results");
	// calculating strain and stress components for graphical output
	  pcout << "      Writing graphical output... " << std::flush;

	  DataOut<dim> data_out;
	  data_out.attach_dof_handler(dof_handler);
	  const std::vector<DataComponentInterpretation::DataComponentInterpretation>
	  data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);

	  data_out.add_data_vector(locally_relevant_solution, std::vector<std::string> (dim, "displacement"),
								   DataOut<dim>::type_dof_data, data_component_interpretation);
	  std::string output_dir = "./Results/", filename_base = (file_name + std::to_string(timestep_counter));
	  std::vector<std::string> solution_names;

	  switch (dim)
		{
		case 1:
		  solution_names.push_back ("displacement");
		  break;
		case 2:
		  solution_names.push_back ("x_displacement");
		  solution_names.push_back ("y_displacement");

		  break;
		case 3:
		  solution_names.push_back ("x_displacement");
		  solution_names.push_back ("y_displacement");
		  solution_names.push_back ("z_displacement");
		  break;
		default:
		  AssertThrow (false, ExcNotImplemented());
		}

	  data_out.add_data_vector (locally_relevant_solution, solution_names);

	  Vector<float> subdomain(triangulation.n_active_cells());
	  for (unsigned int i = 0; i < subdomain.size(); ++i)
			subdomain(i) = triangulation.locally_owned_subdomain();
	  data_out.add_data_vector(subdomain, "subdomain");

	  data_out.build_patches();
	  const std::string filename = ( output_dir + filename_base + "-"
			 + Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4));

	  std::ofstream output_vtu((filename + ".vtu").c_str());
	  data_out.write_vtu(output_vtu);
	  pcout << output_dir + filename_base << ".pvtu" << std::endl;


		  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
				{
				  std::vector<std::string> filenames;
				  for (unsigned int i = 0; i < Utilities::MPI::n_mpi_processes(mpi_communicator); ++i)
					filenames.push_back(filename_base + "-" +
										Utilities::int_to_string(i, 4) +
										".vtu");

				  std::ofstream pvtu_master_output((output_dir + filename_base + ".pvtu").c_str());
				  data_out.write_pvtu_record(pvtu_master_output, filenames);

				 // std::ofstream visit_master_output((output_dir + filename_base + ".visit").c_str());
				  //data_out.write_pvtu_record(visit_master_output, filenames);

				  // produce eps files for mesh illustration
				//  std::ofstream output_eps((filename + ".eps").c_str());
				 // GridOut grid_out;
				 // grid_out.write_eps(triangulation, output_eps);
				}


		  // Getting nodal average stress & strain
		  // Extrapolate the stresses from Gauss point to the nodes
			  SymmetricTensor<2, dim> stress_at_qpoint, strain_at_qpoint, t_strain_at_qpoint;
			  double mvf_at_qpoint;

			  FE_DGQ<dim> history_fe (1);
			  DoFHandler<dim> history_dof_handler (triangulation);
			  history_dof_handler.distribute_dofs (history_fe);
			  std::vector< std::vector< Vector<double> > > history_stress_field (dim, std::vector< Vector<double> >(dim)),
								   local_history_stress_values_at_qpoints (dim, std::vector< Vector<double> >(dim)),
								   local_history_stress_fe_values (dim, std::vector< Vector<double> >(dim));

			  std::vector< std::vector< Vector<double> > > history_strain_field (dim, std::vector< Vector<double> >(dim)),
												   local_history_strain_values_at_qpoints (dim, std::vector< Vector<double> >(dim)),
												   local_history_strain_fe_values (dim, std::vector< Vector<double> >(dim));
			  std::vector< std::vector< Vector<double> > > history_t_strain_field (dim, std::vector< Vector<double> >(dim)),
												   local_history_t_strain_values_at_qpoints (dim, std::vector< Vector<double> >(dim)),
												   local_history_t_strain_fe_values (dim, std::vector< Vector<double> >(dim));

			  Vector<double> history_mvf_field, local_history_mvf_values_at_qpoints, local_history_mvf_fe_values;

			  for (unsigned int i=0; i<dim; ++i)
				  for (unsigned int j=0; j<dim; ++j)
				  {
					history_stress_field[i][j].reinit(history_dof_handler.n_dofs());
					local_history_stress_values_at_qpoints[i][j].reinit(quadrature_formula.size());
					local_history_stress_fe_values[i][j].reinit(history_fe.dofs_per_cell);

					history_strain_field[i][j].reinit(history_dof_handler.n_dofs());
					local_history_strain_values_at_qpoints[i][j].reinit(quadrature_formula.size());
					local_history_strain_fe_values[i][j].reinit(history_fe.dofs_per_cell);

					history_t_strain_field[i][j].reinit(history_dof_handler.n_dofs());
					local_history_t_strain_values_at_qpoints[i][j].reinit(quadrature_formula.size());
					local_history_t_strain_fe_values[i][j].reinit(history_fe.dofs_per_cell);
				  }

			  history_mvf_field.reinit(history_dof_handler.n_dofs());
			  local_history_mvf_values_at_qpoints.reinit(quadrature_formula.size());
			  local_history_mvf_fe_values.reinit(history_fe.dofs_per_cell);


			  Vector<double>  VM_stress_field (history_dof_handler.n_dofs()), local_VM_stress_values_at_qpoints (quadrature_formula.size()),
					 local_VM_stress_fe_values (history_fe.dofs_per_cell);

			  FullMatrix<double> qpoint_to_dof_matrix (history_fe.dofs_per_cell, quadrature_formula.size());
			  FETools::compute_projection_from_quadrature_points_matrix( history_fe, quadrature_formula, quadrature_formula, qpoint_to_dof_matrix);



			  typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(), endc = dof_handler.end(),
					  dg_cell = history_dof_handler.begin_active();

				  const FEValuesExtractors::Vector displacement(0);

				  for (; cell!=endc; ++cell, ++dg_cell)
					if (cell->is_locally_owned())
					  {
						PointHistory *local_quadrature_points_history = reinterpret_cast<PointHistory *>(cell->user_pointer());
						Assert (local_quadrature_points_history >=
								&quadrature_point_history.front(),
								ExcInternalError());
						Assert (local_quadrature_points_history <
								&quadrature_point_history.back(),
								ExcInternalError());

						// Then loop over the quadrature points of this cell:
						for (unsigned int q=0; q<quadrature_formula.size(); ++q)
						  {
							stress_at_qpoint = local_quadrature_points_history[q].old_stress;
							strain_at_qpoint = local_quadrature_points_history[q].old_strain;
							t_strain_at_qpoint = local_quadrature_points_history[q].old_t_strain;
							mvf_at_qpoint = local_quadrature_points_history[q].old_xi;

							for (unsigned int i=0; i<dim; ++i)
							  for (unsigned int j=i; j<dim; ++j)
								{
								  local_history_stress_values_at_qpoints[i][j](q) = stress_at_qpoint[i][j];
								  local_history_strain_values_at_qpoints[i][j](q) = strain_at_qpoint[i][j];
								  local_history_t_strain_values_at_qpoints[i][j](q) = t_strain_at_qpoint[i][j];
								}

							local_VM_stress_values_at_qpoints(q) = stress_at_qpoint.norm();
							local_history_mvf_values_at_qpoints(q) = mvf_at_qpoint;
						  }


						for (unsigned int i=0; i<dim; ++i)
						  for (unsigned int j=i; j<dim; ++j)
							{
							  qpoint_to_dof_matrix.vmult (local_history_stress_fe_values[i][j],
														  local_history_stress_values_at_qpoints[i][j]);
							  dg_cell->set_dof_values (local_history_stress_fe_values[i][j],
													   history_stress_field[i][j]);

							  qpoint_to_dof_matrix.vmult (local_history_strain_fe_values[i][j],
									  local_history_strain_values_at_qpoints[i][j]);
							  dg_cell->set_dof_values (local_history_strain_fe_values[i][j],
									  history_strain_field[i][j]);

							  qpoint_to_dof_matrix.vmult (local_history_t_strain_fe_values[i][j],
									  local_history_t_strain_values_at_qpoints[i][j]);
							  dg_cell->set_dof_values (local_history_t_strain_fe_values[i][j],
									  history_t_strain_field[i][j]);
							}

						qpoint_to_dof_matrix.vmult (local_VM_stress_fe_values,
													local_VM_stress_values_at_qpoints);
						dg_cell->set_dof_values (local_VM_stress_fe_values,
												 VM_stress_field);

						qpoint_to_dof_matrix.vmult (local_history_mvf_fe_values,
													local_history_mvf_values_at_qpoints);
						dg_cell->set_dof_values (local_history_mvf_fe_values,
												 history_mvf_field);
					  }

				  // Nodal averaging
				  FE_Q<dim>          fe_1 (1);
				  DoFHandler<dim>    dof_handler_1 (triangulation);
				  dof_handler_1.distribute_dofs (fe_1);

//	                  AssertThrow(dof_handler_1.n_dofs() == triangulation.n_vertices(),
//	                              ExcDimensionMismatch(dof_handler_1.n_dofs(),triangulation.n_vertices()));

				  std::vector< std::vector< Vector<double> > > history_stress_on_vertices (dim, std::vector< Vector<double> >(dim)),
						  history_strain_on_vertices (dim, std::vector< Vector<double> >(dim)),
						  history_t_strain_on_vertices (dim, std::vector< Vector<double> >(dim));

				  for (unsigned int i=0; i<dim; ++i)
					for (unsigned int j=0; j<dim; ++j)
					  {
						history_stress_on_vertices[i][j].reinit(dof_handler_1.n_dofs());
						history_strain_on_vertices[i][j].reinit(dof_handler_1.n_dofs());
						history_t_strain_on_vertices[i][j].reinit(dof_handler_1.n_dofs());
					  }

				  Vector<double>  VM_stress_on_vertices (dof_handler_1.n_dofs()), mvf_on_vertices(dof_handler_1.n_dofs()),
						  counter_on_vertices (dof_handler_1.n_dofs());
				  VM_stress_on_vertices = 0.;
				  mvf_on_vertices =0.;
				  counter_on_vertices = 0;

				  cell = dof_handler.begin_active();
				  dg_cell = history_dof_handler.begin_active();
				  typename DoFHandler<dim>::active_cell_iterator
				  cell_1 = dof_handler_1.begin_active();
				  for (; cell!=endc; ++cell, ++dg_cell, ++cell_1)
					if (cell->is_locally_owned())
					  {
						dg_cell->get_dof_values (VM_stress_field, local_VM_stress_fe_values);
						dg_cell->get_dof_values (history_mvf_field, local_history_mvf_fe_values);

						for (unsigned int i=0; i<dim; ++i)
						  for (unsigned int j=0; j<dim; ++j)
							{
							  dg_cell->get_dof_values (history_stress_field[i][j], local_history_stress_fe_values[i][j]);
							  dg_cell->get_dof_values (history_strain_field[i][j], local_history_strain_fe_values[i][j]);
							  dg_cell->get_dof_values (history_t_strain_field[i][j], local_history_t_strain_fe_values[i][j]);

							}

						for  (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
						  {
							types::global_dof_index dof_1_vertex = cell_1->vertex_dof_index(v, 0);

							// begin check
							//            Point<dim> point1, point2;
							//            point1 = cell_1->vertex(v);
							//            point2 = dg_cell->vertex(v);
							//            AssertThrow(point1.distance(point2) < cell->diameter()*1e-8, ExcInternalError());
							// end check

							counter_on_vertices (dof_1_vertex) += 1;

							VM_stress_on_vertices (dof_1_vertex) += local_VM_stress_fe_values (v);
							mvf_on_vertices(dof_1_vertex) += local_history_mvf_fe_values(v);

							for (unsigned int i=0; i<dim; ++i)
							  for (unsigned int j=0; j<dim; ++j)
								{
								  history_stress_on_vertices[i][j](dof_1_vertex) += local_history_stress_fe_values[i][j](v);
								  history_strain_on_vertices[i][j](dof_1_vertex) += local_history_strain_fe_values[i][j](v);
								  history_t_strain_on_vertices[i][j](dof_1_vertex) += local_history_t_strain_fe_values[i][j](v);
								}
						  }
					  }

				  for (unsigned int id=0; id<dof_handler_1.n_dofs(); ++id)
					{
					  VM_stress_on_vertices(id) /= counter_on_vertices(id);
					  mvf_on_vertices(id) /= counter_on_vertices(id);

					  for (unsigned int i=0; i<dim; ++i)
						for (unsigned int j=0; j<dim; ++j)
						  {
							history_stress_on_vertices[i][j](id) /= counter_on_vertices(id);
							history_strain_on_vertices[i][j](id) /= counter_on_vertices(id);
							history_t_strain_on_vertices[i][j](id) /= counter_on_vertices(id);
						  }
					}

				  {         DataOut<dim>  data_out;
							data_out.attach_dof_handler (dof_handler_1);


							data_out.add_data_vector (history_strain_on_vertices[0][0], "strain_xx_averaged");
							data_out.add_data_vector (history_strain_on_vertices[1][1], "strain_yy_averaged");
							data_out.add_data_vector (history_strain_on_vertices[0][1], "strain_xy_averaged");

							data_out.add_data_vector (history_stress_on_vertices[0][0], "stress_xx_averaged");
							data_out.add_data_vector (history_stress_on_vertices[1][1], "stress_yy_averaged");
							data_out.add_data_vector (history_stress_on_vertices[0][1], "stress_xy_averaged");
							data_out.add_data_vector (VM_stress_on_vertices, "Von_Mises_stress_averaged");

							data_out.add_data_vector (mvf_on_vertices, "mvf");

							data_out.add_data_vector (history_t_strain_on_vertices[0][0], "t_strain_xx_averaged");
							data_out.add_data_vector (history_t_strain_on_vertices[1][1], "t_strain_yy_averaged");
							data_out.add_data_vector (history_t_strain_on_vertices[0][1], "t_strain_xy_averaged");

							if (dim == 3)
							  {
								data_out.add_data_vector (history_strain_on_vertices[0][2], "strain_xz_averaged");
								data_out.add_data_vector (history_strain_on_vertices[1][2], "strain_yz_averaged");
								data_out.add_data_vector (history_strain_on_vertices[2][2], "strain_zz_averaged");

								data_out.add_data_vector (history_stress_on_vertices[0][2], "stress_xz_averaged");
								data_out.add_data_vector (history_stress_on_vertices[1][2], "stress_yz_averaged");
								data_out.add_data_vector (history_stress_on_vertices[2][2], "stress_zz_averaged");

								data_out.add_data_vector (history_t_strain_on_vertices[0][2], "t_strain_xz_averaged");
								data_out.add_data_vector (history_t_strain_on_vertices[1][2], "t_strain_yz_averaged");
								data_out.add_data_vector (history_t_strain_on_vertices[2][2], "t_strain_zz_averaged");
							  }

							data_out.build_patches ();

							const std::string filename_base_stress = ("averaged-stress-strain-" + filename_base);

							const std::string filename =
							  (output_dir + filename_base_stress + "-"
							   + Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4));

							std::ofstream output_vtu((filename + ".vtu").c_str());
							data_out.write_vtu(output_vtu);
							pcout << output_dir + filename_base_stress << ".pvtu" << std::endl;

							if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
							  {
								std::vector<std::string> filenames;
								for (unsigned int i = 0; i < Utilities::MPI::n_mpi_processes(mpi_communicator); ++i)
								  filenames.push_back(filename_base_stress + "-" +
													  Utilities::int_to_string(i, 4) +
													  ".vtu");

								std::ofstream pvtu_master_output((output_dir + filename_base_stress + ".pvtu").c_str());
								data_out.write_pvtu_record(pvtu_master_output, filenames);

//	                                std::ofstream visit_master_output((output_dir + filename_base_stress + ".visit").c_str());
//	                                data_out.write_pvtu_record(visit_master_output, filenames);
							  }
						  }

}

//template <int dim>
//void FEAnalysis<dim>::map_scalar_qpoint_to_dof(const Vector<double> &vec,
//		Vector<double> &avg_vec_on_cell_vertices) const
//{
//	// The input vector vec contains the values of a scalar field at the quadrature points of all cells
//	FE_DGQ<dim> history_fe (1);
//	DoFHandler<dim> history_dof_handler (triangulation);
//	history_dof_handler.distribute_dofs (history_fe);
//	Vector<double> vec_field, vec_values_at_qpoints, vec_at_dgcell_vertices;
//	vec_field.reinit(history_dof_handler.n_dofs());
//	vec_values_at_qpoints.reinit(quadrature_formula.size());
//	vec_at_dgcell_vertices.reinit(history_fe.dofs_per_cell);
//
//
//	FullMatrix<double> qpoint_to_dof_matrix (history_fe.dofs_per_cell,
//			quadrature_formula.size());
//	FETools::compute_projection_from_quadrature_points_matrix (history_fe, quadrature_formula,
//			quadrature_formula, qpoint_to_dof_matrix);
//
//	unsigned int q_k = 0;
//	typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
//			endc = dof_handler.end(), dg_cell = history_dof_handler.begin_active();
//	for (; cell!=endc; ++cell, ++dg_cell)
//	{
//
//		for (unsigned int q=0; q<quadrature_formula.size(); ++q)
//		{
//			vec_values_at_qpoints(q) = vec[q_k];  // particular strain components in all quadrature points in a cell
//			q_k++;
//		}
//		qpoint_to_dof_matrix.vmult (vec_at_dgcell_vertices, vec_values_at_qpoints);
//		dg_cell->set_dof_values (vec_at_dgcell_vertices, vec_field);
//	}
//
//	// Now we need find strain on cell vertices using nodal averaging
//
//	FE_Q<dim>          fe_1 (1);
//	DoFHandler<dim>    dof_handler_1 (triangulation);
//	dof_handler_1.distribute_dofs (fe_1);
//
//	AssertThrow(dof_handler_1.n_dofs() == triangulation.n_vertices(),
//			ExcDimensionMismatch(dof_handler_1.n_dofs(),triangulation.n_vertices()));
//
//	/*std::vector< Vector<double> > avg_strain_on_cell_vertices (n_strain_components);
//      	      for (unsigned int i=0; i < n_strain_components; ++i)
//      	          {
//      	            avg_strain_on_cell_vertices[i].reinit(dof_handler_1.n_dofs());
//      	          }*/
//
//
//	Vector<double>  counter_on_vertices (dof_handler_1.n_dofs());
//	counter_on_vertices = 0;
//
//	cell = dof_handler.begin_active();
//	dg_cell = history_dof_handler.begin_active();
//	typename DoFHandler<dim>::active_cell_iterator cell_1 = dof_handler_1.begin_active();
//	for (; cell!=endc; ++cell, ++dg_cell, ++cell_1)
//	{
//		dg_cell->get_dof_values (vec_field, vec_at_dgcell_vertices);
//
//		for  (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
//		{
//			types::global_dof_index dof_1_vertex = cell_1->vertex_dof_index(v, 0);
//			counter_on_vertices (dof_1_vertex) += 1;
//
//			avg_vec_on_cell_vertices(dof_1_vertex) += vec_at_dgcell_vertices(v);
//
//		}
//	}
//	for (unsigned int id=0; id<dof_handler_1.n_dofs(); ++id)
//	{
//		avg_vec_on_cell_vertices(id) /= counter_on_vertices(id);
//	}
//}
//
//template <int dim>
//void FEAnalysis<dim>::map_vec_qpoint_to_dof(const unsigned int &n_components, const std::vector<Vector<double>> &vec,
//		std::vector<Vector<double>> &avg_vec_on_cell_vertices) const
//{
//	// Lets determine the strain components on the vertices
//	Vector<double> vec_at_qpoint(n_components);
//
//	FE_DGQ<dim> history_fe (1);
//	DoFHandler<dim> history_dof_handler (triangulation);
//	history_dof_handler.distribute_dofs (history_fe);
//	std::vector< Vector<double> > vec_field (n_components), vec_values_at_qpoints (n_components),
//			vec_at_dgcell_vertices (n_components);
//
//	for (unsigned int i=0; i< n_components; ++i)
//	{
//		vec_field[i].reinit(history_dof_handler.n_dofs());
//		vec_values_at_qpoints[i].reinit(quadrature_formula.size());
//		vec_at_dgcell_vertices[i].reinit(history_fe.dofs_per_cell);
//	}
//
//	FullMatrix<double> qpoint_to_dof_matrix (history_fe.dofs_per_cell,
//			quadrature_formula.size());
//	FETools::compute_projection_from_quadrature_points_matrix (history_fe, quadrature_formula,
//			quadrature_formula, qpoint_to_dof_matrix);
//
//	unsigned int q_k = 0;
//	typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
//			endc = dof_handler.end(), dg_cell = history_dof_handler.begin_active();
//	for (; cell!=endc; ++cell, ++dg_cell)
//	{
//		for (unsigned int q=0; q<quadrature_formula.size(); ++q)
//		{
//			vec_at_qpoint = vec[q_k];
//
//			for (unsigned int i=0; i< n_components; ++i)
//			{
//				vec_values_at_qpoints[i](q) = vec_at_qpoint[i];  // particular strain components in all quadrature points in a cell
//			}
//			q_k++;
//		}
//
//		for (unsigned int i=0; i< n_components; ++i)
//		{
//			qpoint_to_dof_matrix.vmult (vec_at_dgcell_vertices[i], vec_values_at_qpoints[i]);
//			dg_cell->set_dof_values (vec_at_dgcell_vertices[i], vec_field[i]);
//		}
//	}

	// Now we need find strain on cell vertices using nodal averaging

//	FE_Q<dim>          fe_1 (1);
//	DoFHandler<dim>    dof_handler_1 (triangulation);
//	dof_handler_1.distribute_dofs (fe_1);
//
//	AssertThrow(dof_handler_1.n_dofs() == triangulation.n_vertices(),
//			ExcDimensionMismatch(dof_handler_1.n_dofs(),triangulation.n_vertices()));
//
//	/*std::vector< Vector<double> > avg_strain_on_cell_vertices (n_strain_components);
//    	      for (unsigned int i=0; i < n_strain_components; ++i)
//    	          {
//    	            avg_strain_on_cell_vertices[i].reinit(dof_handler_1.n_dofs());
//    	          }*/
//
//
//	Vector<double>  counter_on_vertices (dof_handler_1.n_dofs());
//	counter_on_vertices = 0;
//
//	cell = dof_handler.begin_active();
//	dg_cell = history_dof_handler.begin_active();
//	typename DoFHandler<dim>::active_cell_iterator cell_1 = dof_handler_1.begin_active();
//	for (; cell!=endc; ++cell, ++dg_cell, ++cell_1)
//	{
//		for (unsigned int i=0; i< n_components; ++i)
//		{
//			dg_cell->get_dof_values (vec_field[i], vec_at_dgcell_vertices[i]);
//		}
//		for  (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
//		{
//			types::global_dof_index dof_1_vertex = cell_1->vertex_dof_index(v, 0);
//			counter_on_vertices (dof_1_vertex) += 1;
//
//			for (unsigned int i=0; i< n_components; ++i)
//			{
//				avg_vec_on_cell_vertices[i](dof_1_vertex) += vec_at_dgcell_vertices[i](v);
//			}
//		}
//	}
//	for (unsigned int id=0; id<dof_handler_1.n_dofs(); ++id)
//	{
//		for (unsigned int i=0; i< n_components; ++i)
//		{
//			avg_vec_on_cell_vertices[i](id) /= counter_on_vertices(id);
//		}
//	}
//}

template <int dim>
void FEAnalysis<dim> :: print_conv_header()
{
	static const unsigned int l_width = 145;
	for (unsigned int i = 0; i < l_width; ++i)
		pcout << "-";
	pcout << std::endl;
	pcout << "NR_ITER" << "	 ConvexCutAlgo"
			<< "|   STRAIN        STRESS     	Temp		MVF     	t_strain     	MODULI[0][0]"
			<< "	RES_F	  RES_F/RES_F_I  "
			<< "    dNORM_U/NORM_U    "
			<< std::endl;
	for (unsigned int i = 0; i < l_width; ++i)
		pcout << "-";
	pcout << std::endl;
}

template <int dim>
void FEAnalysis<dim>::print_conv_footer()
{
	static const unsigned int l_width = 145;
	for (unsigned int i = 0; i < l_width; ++i)
		pcout << "-";
	pcout << std::endl;
}

} // namespace Step8


int main(int argc, char *argv[])
{
	using namespace dealii;
	using namespace Step8;

	try
	{
		Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

		ParameterHandler prm;
		FEAnalysis<3>::declare_parameters(prm);
		prm.parse_input("inputs.prm");
		//param.read_parameters("inputs.prm");

		Step8::FEAnalysis<3> sma_problem(prm);
		sma_problem.run();

		//Step8::ElasticProblem<3> elastic_problem_3d(case_);
		//elastic_problem_3d.run();
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
