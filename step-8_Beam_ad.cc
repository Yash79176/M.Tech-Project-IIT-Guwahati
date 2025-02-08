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
//#include <deal.II/lac/constraint_matrix.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>
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

#include <fstream>
#include <iostream>

namespace Step8
{
  using namespace dealii;


  struct PointHistory
  {
	  SymmetricTensor<4,3> continuum_moduli;
	  SymmetricTensor<2,3> old_stress;			// stress after load step converged at t_n
	  SymmetricTensor<2,3>  stress;				// stress at every NR iteration after material model converged
	  SymmetricTensor<2,3>  old_strain;
  };

  template <int T>
    SymmetricTensor<4, T>
    get_stress_strain_tensor_plane_strain (const double &E, const double &nu)
    {
	  //const double E =1.;
	  //const double lambda=(E*nu/((1+nu)*(1-2*nu))), mu=E/(2*(1+nu));
      SymmetricTensor<4, T> tmp;
      for (unsigned int i=0; i< T; ++i)
        for (unsigned int j=0; j< T; ++j)
          for (unsigned int k=0; k< T; ++k)
            for (unsigned int l=0; l< T; ++l)
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
    SymmetricTensor<2,dim> convert_symmetric_tensor_3d_to_2d(const SymmetricTensor<2,3>  &symmetric_tensor3d)
        {
     	 SymmetricTensor<2,dim>  symmetric_tensor;
        // symmetric_tensor=0.;
         for (unsigned int i = 0; i < dim; ++i)
        	 for (unsigned int j = i; j < dim; ++j)
        		 symmetric_tensor [i][j] = symmetric_tensor3d [i][j];

     	  return symmetric_tensor;
        }

	template<int dim>
    SymmetricTensor<4,dim> convert_symmetric_tensor_3d_to_2d(const SymmetricTensor<4,3>  &symmetric_tensor3d)
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
  void right_hand_side(const std::vector<Point<dim>> &points,
                       std::vector<Tensor<1, dim>> &  values)
  {
    Assert(values.size() == points.size(),
           ExcDimensionMismatch(values.size(), points.size()));
    Assert(dim >= 2, ExcNotImplemented());

    Point<dim> point_1, point_2;
    point_1(0) = 0.5;
    point_2(0) = -0.5;

    for (unsigned int point_n = 0; point_n < points.size(); ++point_n)
      {
    	values[point_n][0] = 0.0;
    	values[point_n][1] = 0.0;
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
    const double p0   = 250./(1e-2*1.5e-3);				// N /per unit length
//    const double total_time = 2.;
    values    = 0;
//    values(1) = -10;
    if (present_time <= 1.)
    	values(1) = p0*(present_time/(1-0));			// total time is 1s
    else if (present_time <= 2.)
    	values(1) = p0*((2-present_time)/(2. - 1.));
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
     void update_quadrature_point_history(const bool &cond);
     void call_Newton_Raphson_method();
     void map_scalar_qpoint_to_dof(const Vector<double> &vec, Vector<double> &avg_vec_on_cell_vertices) const;
     void map_vec_qpoint_to_dof(const unsigned int &n_components, const std::vector<Vector<double>> &vec, std::vector<Vector<double>> &avg_vec_on_cell_vertices) const;
     void output_results(const unsigned int timestep_counter) const;
     void print_conv_header(void);
     void print_conv_footer();

     Triangulation<dim> triangulation;
     DoFHandler<dim>    dof_handler;

     FESystem<dim> fe;

     AffineConstraints<double>  hanging_node_constraints;
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
     static const double moduli, nu;
     const SymmetricTensor<4, 3> elastic_constitutive_tensor;
   };

  template <int dim>
  const double ElasticProblem<dim> :: moduli = 72.0e9;

  template <int dim>
  const double ElasticProblem<dim> :: nu = 0.42;


  template <int dim>
  ElasticProblem<dim>::ElasticProblem(const unsigned int &study_type)
    : dof_handler(triangulation)
    , fe(FE_Q<dim>(1), dim)
    , quadrature_formula(fe.degree + 1)
  	, study_type(study_type)						/* plane_stress=0; plane_strain=1; 3D general=2*/
  	, n_time_steps(400)
    , present_time(0.)
  	, end_time(2.)
  	, time_step(end_time/n_time_steps)
  	, elastic_constitutive_tensor(get_stress_strain_tensor_plane_strain<3>(moduli, nu))
  {}

  template <int dim>
         void print_mesh_info(const Triangulation<dim> &triangulation,
         	                       const std::string &       filename)
         	  {
         	    std::cout << "Mesh info:" << std::endl
         	              << " dimension: " << dim << std::endl
         	              << " no. of cells: " << triangulation.n_active_cells() << std::endl;
         	    {
         	      std::map<types::boundary_id, unsigned int> boundary_count;
         	      	  for (auto cell : triangulation.active_cell_iterators())
         	      	  {
         	      		  for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
         	      		  {
         	      			  if (cell->face(face)->at_boundary())
         	      			      boundary_count[cell->face(face)->boundary_id()]++;
         	      		  }
         	      	  }
         	      std::cout << " boundary indicators: ";
         	      for (const std::pair<const types::boundary_id, unsigned int> &pair :
         	           boundary_count)
         	        {
         	          std::cout << pair.first << '(' << pair.second << " times) ";
         	        }
         	      std::cout << std::endl;
         	    }
         	    std::ofstream out(filename);
         	    GridOut       grid_out;
         	    grid_out.write_eps(triangulation, out);
         	    std::cout << " written to " << filename << std::endl << std::endl;
         	  }



  template <int dim>
  void ElasticProblem<dim>::create_mesh(/*const unsigned int cycle*/)
  {
  	const double length = 0.1;
  	const double height = 0.01;
  	const double thickness = 0.0015;
  	std::vector< unsigned int > repetitions(3, 10); repetitions[1]= 3;  repetitions[2]= 1;

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
//  	triangulation.execute_coarsening_and_refinement();
//  		        	      print_mesh_info(triangulation, "meshBeam.eps");

//  		triangulation.refine_global(1);
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
    //SymmetricTensor<2,dim> stress_at_qpoint;

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<Tensor<1, dim>> rhs_values(n_q_points);
    const NeumannBoundary<dim> neumann_boundary(present_time, time_step);
    std::vector<Vector<double>> neumann_values(n_face_q_points, Vector<double>(dim));

    std::vector<std::vector<Tensor<1, dim>>> solution_grads(quadrature_formula.size(),
    		std::vector<Tensor<1, dim>>(dim));
    //std::cout<< "residue at the beginning" << residue_vector << std::endl;
    system_matrix  = 0.;
    residue_vector =0.;
    SymmetricTensor<4, dim> linearised_stress_strain_matrix;
    SymmetricTensor<2, dim> stress_at_qpoint;
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
        fe_values.get_function_gradients(solution, solution_grads);

        right_hand_side(fe_values.get_quadrature_points(), rhs_values);

       for (const unsigned int q_point : fe_values.quadrature_point_indices())
       {
    	   const SymmetricTensor<4, 3> moduli_tmp = quadrature_point_history[q_point].continuum_moduli;

    	   linearised_stress_strain_matrix = convert_symmetric_tensor_3d_to_2d<dim>(moduli_tmp);

    	   for (const unsigned int i : fe_values.dof_indices()){
    		   const SymmetricTensor<2,dim> eps_phi_i = get_linear_strain (fe_values, i, q_point);
    	       for (const unsigned int j : fe_values.dof_indices()){
    	       		const SymmetricTensor<2,dim> eps_phi_j = get_linear_strain (fe_values, j, q_point);
    	       		cell_matrix(i,j)+= (eps_phi_i * linearised_stress_strain_matrix * eps_phi_j) * fe_values.JxW(q_point);
    	           	   }
    	   }
       }

       for (const unsigned int q_point : fe_values.quadrature_point_indices())
           {
    	   // for 3D problem no need to convert
    	   const SymmetricTensor<2, 3> stress_tmp = local_quadrature_points_history[q_point].stress;
    	   stress_at_qpoint = convert_symmetric_tensor_3d_to_2d<dim>(stress_tmp);

    	   for (const unsigned int i : fe_values.dof_indices())
    	          {
    		      	   const SymmetricTensor<2,dim> eps_phi_i = get_linear_strain (fe_values, i, q_point);
    		      	   cell_internal_force(i) += stress_at_qpoint * eps_phi_i * fe_values.JxW(q_point);
    	          }
           }

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

        hanging_node_constraints.distribute_local_to_global(cell_matrix, cell_residue,
        		local_dof_indices, system_matrix, residue_vector);
      }

      std::map<types::global_dof_index, double> boundary_values;

//      if (dim == 2)
//      {
//    	  const FEValuesExtractors::Scalar          x_component(0), y_component(1);
//    	  VectorTools::interpolate_boundary_values(dof_handler, 1, Functions::ZeroFunction<dim>(dim), boundary_values);
////    	  VectorTools::interpolate_boundary_values(dof_handler, 2, Functions::ZeroFunction<dim>(dim), boundary_values, fe.component_mask(y_component));
//      }
//      else if(dim == 3)
//      {
//    	  const FEValuesExtractors::Scalar          x_component(0), y_component(1), z_component(2);
//    	  VectorTools::interpolate_boundary_values(dof_handler, 1, Functions::ZeroFunction<dim>(dim), boundary_values);
////    	  VectorTools::interpolate_boundary_values(dof_handler, 2, Functions::ZeroFunction<dim>(dim), boundary_values, fe.component_mask(y_component));
////    	  VectorTools::interpolate_boundary_values(dof_handler, 4, Functions::ZeroFunction<dim>(dim), boundary_values, fe.component_mask(z_component));
//      }

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

	  local_history_stress_values_at_qpoints (dim, std::vector< Vector<double> >(dim)),
	  local_history_strain_values_at_qpoints (dim, std::vector< Vector<double> >(dim)),

	  local_history_fe_values1 (dim, std::vector< Vector<double> >(dim)),
	  local_history_fe_values2 (dim, std::vector< Vector<double> >(dim));

	  for (unsigned int i=0; i<dim; ++i)
	  	  {
	  	  	  for (unsigned int j=0; j<dim; ++j)
	  		  	  {
	  					history_field1[i][j].reinit(dof_handler_1.n_dofs());
	  					local_history_stress_values_at_qpoints[i][j].reinit(quadrature_formula.size());
	  					local_history_fe_values1[i][j].reinit(history_fe.n_dofs_per_cell());

	  					history_field2[i][j].reinit(dof_handler_1.n_dofs());
	  					local_history_strain_values_at_qpoints[i][j].reinit(quadrature_formula.size());
	  					local_history_fe_values2[i][j].reinit(history_fe.n_dofs_per_cell());
	  			  }
	  	   }
	  	FullMatrix<double> qpoint_to_dof_matrix1(history_fe.dofs_per_cell,
	  										quadrature_formula.size());

	  	FETools::compute_projection_from_quadrature_points_matrix
	  										(history_fe,
	  										 quadrature_formula, quadrature_formula,
	  										 qpoint_to_dof_matrix1);

	  	typename DoFHandler<dim>::active_cell_iterator
											 cell = dof_handler.begin_active(),
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
	  						}
	  							qpoint_to_dof_matrix1.vmult (local_history_fe_values1[i][j],
	  								        						local_history_stress_values_at_qpoints[i][j]);
	  							dg_cell->set_dof_values (local_history_fe_values1[i][j],
	  								        						history_field1[i][j]);

	  							qpoint_to_dof_matrix1.vmult (local_history_fe_values2[i][j],
	  								        						local_history_strain_values_at_qpoints[i][j]);
	  							dg_cell->set_dof_values (local_history_fe_values2[i][j],
	  								        						history_field2[i][j]);
	  				}
	  		}
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
	  			SolutionTransfer<dim> solution_transfer(dof_handler);
//	  			Vector<double> previous_solution;
//	  			previous_solution = solution;
//	  			solution_transfer.prepare_for_coarsening_and_refinement(previous_solution);

	  			SolutionTransfer<dim, Vector<double> > history_stress_field_transfer0(dof_handler_1),
	  					                       history_stress_field_transfer1(dof_handler_1),
	  					                       history_stress_field_transfer2(dof_handler_1);
	  			history_stress_field_transfer0.prepare_for_coarsening_and_refinement(history_field1[0]);
	  			      if ( dim > 1)
	  			        {
	  			         history_stress_field_transfer1.prepare_for_coarsening_and_refinement(history_field1[1]);
	  			        }
	  			      if ( dim == 3)
	  			        {
	  			         history_stress_field_transfer2.prepare_for_coarsening_and_refinement(history_field1[2]);
	  			        }

	  			SolutionTransfer<dim, Vector<double> > history_strain_field_transfer0(dof_handler_1),
	  					                       history_strain_field_transfer1(dof_handler_1),
	  					                       history_strain_field_transfer2(dof_handler_1);
	  			history_strain_field_transfer0.prepare_for_coarsening_and_refinement(history_field2[0]);
	  			      if ( dim > 1)
	  			       {
	  			          history_strain_field_transfer1.prepare_for_coarsening_and_refinement(history_field2[1]);
	  			       }
	  			      if ( dim == 3)
	  			        {
	  			          history_strain_field_transfer2.prepare_for_coarsening_and_refinement(history_field2[2]);
	  			        }

	  			triangulation.execute_coarsening_and_refinement();

	  			setup_system();
	  			setup_quadrature_point_history ();

//	  			solution_transfer.interpolate(previous_solution, solution);

	  			hanging_node_constraints.distribute(solution);
	  			dof_handler_1.clear();
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
	  			 history_stress_field_transfer0.interpolate(history_field1[0], distributed_history_stress_field[0]);
	  			      if ( dim > 1)
	  			        {
	  			          history_stress_field_transfer1.interpolate(history_field1[1], distributed_history_stress_field[1]);
	  			        }
	  			      if ( dim == 3)
	  			        {
	  			          history_stress_field_transfer2.interpolate(history_field1[2], distributed_history_stress_field[2]);
	  			        }
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
	  				history_strain_field_transfer0.interpolate(history_field2[0], distributed_history_strain_field[0]);
	  				      if ( dim > 1)
	  				        {
	  				          history_strain_field_transfer1.interpolate(history_field2[1], distributed_history_strain_field[1]);
	  				        }
	  				      if ( dim == 3)
	  				        {
	  				          history_strain_field_transfer2.interpolate(history_field2[2], distributed_history_strain_field[2]);
	  				        }
	  				history_field2 = distributed_history_strain_field;

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

	  				for (unsigned int q=0; q<quadrature_formula.size(); ++q)
	  					{
	  						local_quadrature_points_history[q].old_stress[i][j]
	  							   							= local_history_stress_values_at_qpoints[i][j](q);
	  						local_quadrature_points_history[q].old_strain[i][j]
	  							   							= local_history_strain_values_at_qpoints[i][j](q);
	  					}
	  				}
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
  	  for(auto &cell : dof_handler.active_cell_iterators())
  	  {
  		  PointHistory *local_quadrature_points_history = reinterpret_cast<PointHistory *>(cell->user_pointer());
  		  Assert (local_quadrature_points_history >= &quadrature_point_history.front(),ExcInternalError());
  		  Assert (local_quadrature_points_history < &quadrature_point_history.back(),ExcInternalError());

  		  fe_values.reinit (cell);
  		  //fe_values.get_function_gradients(solution, solution_grads);

  		  for (unsigned int q = 0; q < n_q_points; ++q)
  		    {
  			  local_quadrature_points_history[q].continuum_moduli = elastic_constitutive_tensor;
  			  local_quadrature_points_history[q].old_stress = tmp_tensor;
  			  local_quadrature_points_history[q].stress = tmp_tensor;
  			  local_quadrature_points_history[q].old_strain = tmp_tensor;}
  	  }
    }

  template <int dim>
    void ElasticProblem<dim>::update_quadrature_point_history(const bool &cond)
  {
	  FEValues<dim> fe_values (fe, quadrature_formula,
	                           update_values | update_gradients |
	                           update_quadrature_points);
	  const unsigned int n_q_points = quadrature_formula.size();

	  std::vector<std::vector<Tensor<1, dim>>> solution_grads(n_q_points, std::vector<Tensor<1, dim>>(dim));
	  SymmetricTensor<2, dim> strain_tmp;
	  SymmetricTensor<2, 3> stress_new, strain_new, lambda_new, t_strain_new, t_strain_r_new;
	  SymmetricTensor<4, 3> continuum_moduli_new;

	  for(auto &cell : dof_handler.active_cell_iterators())
	  {
		  PointHistory *local_quadrature_points_history = reinterpret_cast<PointHistory *>(cell->user_pointer());
		  Assert (local_quadrature_points_history >= &quadrature_point_history.front(),ExcInternalError());
		  Assert (local_quadrature_points_history < &quadrature_point_history.back(),ExcInternalError());

		  fe_values.reinit (cell);
		  fe_values.get_function_gradients(solution, solution_grads);

		  for (unsigned int q_point : fe_values.quadrature_point_indices())
		    {
			  strain_tmp = get_linear_strain(solution_grads[q_point]);
			  strain_new = convert_symmetric_tensor_2d_to_3d<dim>(strain_tmp);

			  stress_new = elastic_constitutive_tensor * strain_new;
			  continuum_moduli_new = elastic_constitutive_tensor;

			  if (cond == false)	// N-R iteration not converged
			  {
				  local_quadrature_points_history[q_point].stress = stress_new;
				  local_quadrature_points_history[q_point].continuum_moduli = continuum_moduli_new;
			  }
			  else	// N-R converged for the current load step
			  {
				  local_quadrature_points_history[q_point].continuum_moduli = continuum_moduli_new;
				  local_quadrature_points_history[q_point].old_stress = stress_new;
				  local_quadrature_points_history[q_point].stress = stress_new;
				  local_quadrature_points_history[q_point].old_strain = strain_new;
			  }
		    }
	  }
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

			std::cout << std::fixed << std::setprecision(3) << 	nr_counter  /*std::setw(7)*/ << "		"  << std::scientific << residue_norm << "   " << (residue_norm/residue_norm_NR_0)
												 << "	    " << delta_solution_norm <<std::endl;
			if (residue_norm < tol || nr_counter > 50){
				update_quadrature_point_history(true);
				break;
			}
			else
			{
				update_quadrature_point_history(false);		// update the stress and strain with new solution
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

		 call_Newton_Raphson_method();
		 if(time_step_counter % 5 == 0)
		 	{
		 		refine_grid();
		 		update_quadrature_point_history(true);
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
      for(unsigned int i = 0; i < dim; ++i){
      solution_components[i].reinit(dof_handler_1.n_dofs());
      load_components[i].reinit(dof_handler_1.n_dofs());
      }
      std::vector<bool> vertex_touched(triangulation.n_vertices(), false);
      const NeumannBoundary<dim> neumann_boundary(present_time, time_step);
      Vector<double> load_vector(dim);
      typename DoFHandler<dim>::active_cell_iterator cell_1 = dof_handler_1.begin_active();
      for (auto &cell : dof_handler.active_cell_iterators()){
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
      std::vector<Vector<double>> strain(total_quadrature_points), stress(total_quadrature_points);
      Vector<double> Von_Mises(total_quadrature_points);
      const unsigned int n_components = (dim==2 ? 3 : 6);

      for (unsigned int i=0; i < total_quadrature_points; i++ )
      	{
      	strain[i].reinit(n_components);
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

      		  stress[q_k][0] = local_quadrature_points_history[q_point].old_stress[0][0];
      		  stress[q_k][1] = local_quadrature_points_history[q_point].old_stress[1][1];
      		  stress[q_k][2] = local_quadrature_points_history[q_point].old_stress[2][2];
      		  stress[q_k][3] = local_quadrature_points_history[q_point].old_stress[1][2];
      		  stress[q_k][4] = local_quadrature_points_history[q_point].old_stress[0][2];
      		  stress[q_k][5] = local_quadrature_points_history[q_point].old_stress[0][1];
      		  Von_Mises[q_k] = std::sqrt(1.5) * (deviator(local_quadrature_points_history[q_point].old_stress)).norm();
      		  q_k++;
      			      	  }
      	    	  else
      	    	  {
      		  strain[q_k][0] = local_quadrature_points_history[q_point].old_strain[0][0]; //get_strain(solution_grads[q_point]);
      		  strain[q_k][1] = local_quadrature_points_history[q_point].old_strain[1][1];
      		  strain[q_k][2] = local_quadrature_points_history[q_point].old_strain[0][1];

     		  stress[q_k][0] = local_quadrature_points_history[q_point].old_stress[0][0];
      		  stress[q_k][1] = local_quadrature_points_history[q_point].old_stress[1][1];
      		  stress[q_k][2] = local_quadrature_points_history[q_point].old_stress[0][1];
      		  Von_Mises[q_k] = std::sqrt(1.5) * (deviator(local_quadrature_points_history[q_point].old_stress)).norm();

      		  q_k++;
      	    	  }
      	    }
            }
      std::vector< Vector<double> > avg_strain_on_cell_vertices (n_components), avg_stress_on_cell_vertices (n_components);
      Vector<double> avg_VM_stress_on_cell_vertices;
      for (unsigned int i=0; i < n_components; ++i)
          {
           avg_strain_on_cell_vertices[i].reinit(dof_handler_1.n_dofs());
           avg_stress_on_cell_vertices[i].reinit(dof_handler_1.n_dofs());
           avg_VM_stress_on_cell_vertices.reinit(dof_handler_1.n_dofs());
          }

      map_vec_qpoint_to_dof(n_components, strain, avg_strain_on_cell_vertices);
      map_vec_qpoint_to_dof(n_components, stress, avg_stress_on_cell_vertices);
      map_scalar_qpoint_to_dof(Von_Mises, avg_VM_stress_on_cell_vertices);


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

  		   data_out.add_data_vector (avg_stress_on_cell_vertices[0], "stress_xx");
  		   data_out.add_data_vector (avg_stress_on_cell_vertices[1], "stress_yy");
  		   data_out.add_data_vector (avg_stress_on_cell_vertices[2], "stress_xy");
  		   data_out.add_data_vector (avg_VM_stress_on_cell_vertices, "von_Mises_stress");
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

             data_out.add_data_vector (avg_stress_on_cell_vertices[0], "stress_xx");
             data_out.add_data_vector (avg_stress_on_cell_vertices[1], "stress_yy");
             data_out.add_data_vector (avg_stress_on_cell_vertices[2], "stress_zz");
             data_out.add_data_vector (avg_stress_on_cell_vertices[3], "stress_yz");
             data_out.add_data_vector (avg_stress_on_cell_vertices[4], "stress_xz");
             data_out.add_data_vector (avg_stress_on_cell_vertices[5], "stress_xy");
             data_out.add_data_vector (avg_VM_stress_on_cell_vertices, "von_Mises_stress");

         }

         data_out.build_patches ();

         std::ofstream output("./Results/elastic_beam_ad1_" + std::to_string(dim) + "D_"
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
    std::cout << "NR_ITER" <<  "		RES_F	   RES_F/RES_F_I   dNORM_U/NORM_U  "<< std::endl;
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

//	 Step8::ElasticProblem<3> elastic_problem_2d();
//      elastic_problem_2d.run();

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
