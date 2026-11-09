// pcl::GlobalHypothesesVerification (Aldoma et al., ECCV 2012) from PCL
// 1.14, with ONE line changed. Its addModel filters the model's points
// with the NaN check inverted -- `if (isNormalFinite(n)) continue;`, 1.11
// through master -- keeping only the points whose normal could NOT be
// estimated, so a well-sampled hypothesis is reduced to nothing and every
// hypothesis is kept. addModel is a private, non-virtual member of a
// class PCL instantiates in pcl_recognition, so it cannot be overridden;
// this is the class copied into its own namespace, pcl::pclcompat_hv,
// with the test inverted. Also changed: the OpenMP pragma on the clutter
// cue loop (`default(none)` without the members it reads, which GCC 9+
// rejects) is dropped, so that loop runs on one thread, and the three
// pcl::ScopeTime timers, which print to std::cerr on every verify(),
// are silent. Nothing else.
// PCL's licence follows.
/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2012 Aitor Aldoma, Federico Tombari
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <random>

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>

//includes required by mets.hh
#include <boost/random/linear_congruential.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>

#include <pcl/pcl_macros.h>
#include <pcl/recognition/hv/hypotheses_verification.h>
#include <pcl/recognition/3rdparty/metslib/mets.hh>
#include <pcl/features/normal_3d.h>

#include <memory>
#include <pcl/common/common.h> // for getMinMax3D
#include <pcl/common/time.h>
#include <pcl/point_types.h>

#include <memory>
#include <numeric>

namespace pcl {
namespace pclcompat_hv {

/// Stands in for pcl::ScopeTime, which writes "<title> took <n>ms." to
/// std::cerr on every call.
struct SilentScope {
  explicit SilentScope (const char*) {}
};

  /** \brief A hypothesis verification method proposed in
    * "A Global Hypotheses Verification Method for 3D Object Recognition", A. Aldoma and F. Tombari and L. Di Stefano and Markus Vincze, ECCV 2012
    * \author Aitor Aldoma
    * \ingroup recognition
    */
  template<typename ModelT, typename SceneT>
  class GlobalHypothesesVerification: public HypothesisVerification<ModelT, SceneT>
  {
    private:

      //Helper classes
      struct RecognitionModel
      {
        public:
          std::vector<int> explained_; //indices vector referencing explained_by_RM_
          std::vector<float> explained_distances_; //closest distances to the scene for point i
          std::vector<int> unexplained_in_neighborhood; //indices vector referencing unexplained_by_RM_neighboorhods
          std::vector<float> unexplained_in_neighborhood_weights; //weights for the points not being explained in the neighborhood of a hypothesis
          std::vector<int> outlier_indices_; //outlier indices of this model
          std::vector<int> complete_cloud_occupancy_indices_;
          typename pcl::PointCloud<ModelT>::Ptr cloud_;
          typename pcl::PointCloud<ModelT>::Ptr complete_cloud_;
          int bad_information_;
          float outliers_weight_;
          pcl::PointCloud<pcl::Normal>::Ptr normals_;
          int id_;
      };

      using RecognitionModelPtr = std::shared_ptr<RecognitionModel>;

      using SAOptimizerT = GlobalHypothesesVerification<ModelT, SceneT>;
      class SAModel: public mets::evaluable_solution
      {
        public:
          std::vector<bool> solution_;
          SAOptimizerT * opt_;
          mets::gol_type cost_;

          //Evaluates the current solution
          mets::gol_type cost_function() const override
          {
            return cost_;
          }

          void copy_from(const mets::copyable& o) override
          {
            const auto& s = dynamic_cast<const SAModel&> (o);
            solution_ = s.solution_;
            opt_ = s.opt_;
            cost_ = s.cost_;
          }

          mets::gol_type what_if(int /*index*/, bool /*val*/) const
          {
            /*std::vector<bool> tmp (solution_);
            tmp[index] = val;
            mets::gol_type sol = opt_->evaluateSolution (solution_, index); //evaluate without updating status
            return sol;*/
            return static_cast<mets::gol_type>(0);
          }

          mets::gol_type apply_and_evaluate(int index, bool val)
          {
            solution_[index] = val;
            mets::gol_type sol = opt_->evaluateSolution (solution_, index); //this will update the state of the solution
            cost_ = sol;
            return sol;
          }

          void apply(int /*index*/, bool /*val*/)
          {

          }

          void unapply(int index, bool val)
          {
            solution_[index] = val;
            //update optimizer solution
            cost_ = opt_->evaluateSolution (solution_, index); //this will update the cost function in opt_
          }
          void setSolution(std::vector<bool> & sol)
          {
            solution_ = sol;
          }

          void setOptimizer(SAOptimizerT * opt)
          {
            opt_ = opt;
          }
      };

      /*
       * Represents a move, deactivate a hypothesis
       */

      class move: public mets::move
      {
          int index_;
        public:
          move(int i) :
              index_ (i)
          {
          }

          mets::gol_type evaluate(const mets::feasible_solution& /*cs*/) const override
          {
            return static_cast<mets::gol_type>(0);
          }

          mets::gol_type apply_and_evaluate(mets::feasible_solution& cs)
          {
            auto& model = dynamic_cast<SAModel&> (cs);
            return model.apply_and_evaluate (index_, !model.solution_[index_]);
          }

          void apply(mets::feasible_solution& /*s*/) const override
          {
          }

          void unapply(mets::feasible_solution& s) const
          {
            auto& model = dynamic_cast<SAModel&> (s);
            model.unapply (index_, !model.solution_[index_]);
          }
      };

      class move_manager
      {
        public:
          std::vector<move*> moves_m;
          using iterator = typename std::vector<move *>::iterator;
          iterator begin()
          {
            return moves_m.begin ();
          }
          iterator end()
          {
            return moves_m.end ();
          }

          move_manager(int problem_size)
          {
            for (int ii = 0; ii != problem_size; ++ii)
              moves_m.push_back (new move (ii));
          }

          ~move_manager()
          {
            // delete all moves
            for (auto ii = begin (); ii != end (); ++ii)
              delete (*ii);
          }

          void refresh(mets::feasible_solution& /*s*/)
          {
            std::shuffle (moves_m.begin (), moves_m.end (), std::mt19937(std::random_device()()));
          }

      };

      //inherited class attributes
      using HypothesisVerification<ModelT, SceneT>::mask_;
      using HypothesisVerification<ModelT, SceneT>::scene_cloud_downsampled_;
      using HypothesisVerification<ModelT, SceneT>::scene_downsampled_tree_;
      using HypothesisVerification<ModelT, SceneT>::visible_models_;
      using HypothesisVerification<ModelT, SceneT>::complete_models_;
      using HypothesisVerification<ModelT, SceneT>::resolution_;
      using HypothesisVerification<ModelT, SceneT>::inliers_threshold_;

      //class attributes
      using NormalEstimator_ = pcl::NormalEstimation<SceneT, pcl::Normal>;
      pcl::PointCloud<pcl::Normal>::Ptr scene_normals_;
      pcl::PointCloud<pcl::PointXYZI>::Ptr clusters_cloud_;

      std::vector<int> complete_cloud_occupancy_by_RM_;
      float res_occupancy_grid_;
      float w_occupied_multiple_cm_;

      std::vector<int> explained_by_RM_; //represents the points of scene_cloud_ that are explained by the recognition models
      std::vector<float> explained_by_RM_distance_weighted; //represents the points of scene_cloud_ that are explained by the recognition models
      std::vector<float> unexplained_by_RM_neighboorhods; //represents the points of scene_cloud_ that are not explained by the active hypotheses in the neighboorhod of the recognition models
      std::vector<RecognitionModelPtr> recognition_models_;
      std::vector<std::size_t> indices_;

      float regularizer_;
      float clutter_regularizer_;
      bool detect_clutter_;
      float radius_neighborhood_GO_;
      float radius_normals_;

      float previous_explained_value;
      int previous_duplicity_;
      int previous_duplicity_complete_models_;
      float previous_bad_info_;
      float previous_unexplained_;

      int max_iterations_; //max iterations without improvement
      SAModel best_seen_;
      float initial_temp_;

      int n_cc_;
      std::vector<std::vector<int> > cc_;

      void setPreviousBadInfo(float f)
      {
        previous_bad_info_ = f;
      }

      float getPreviousBadInfo()
      {
        return previous_bad_info_;
      }

      void setPreviousExplainedValue(float v)
      {
        previous_explained_value = v;
      }

      void setPreviousDuplicity(int v)
      {
        previous_duplicity_ = v;
      }

      void setPreviousDuplicityCM(int v)
      {
        previous_duplicity_complete_models_ = v;
      }

      void setPreviousUnexplainedValue(float v)
      {
        previous_unexplained_ = v;
      }

      float getPreviousUnexplainedValue()
      {
        return previous_unexplained_;
      }

      float getExplainedValue()
      {
        return previous_explained_value;
      }

      int getDuplicity()
      {
        return previous_duplicity_;
      }

      int getDuplicityCM()
      {
        return previous_duplicity_complete_models_;
      }

      void updateUnexplainedVector(std::vector<int> & unexplained_, std::vector<float> & unexplained_distances, std::vector<float> & unexplained_by_RM,
          std::vector<int> & explained, std::vector<int> & explained_by_RM, float val)
      {
        {

          float add_to_unexplained = 0.f;

          for (std::size_t i = 0; i < unexplained_.size (); i++)
          {

            bool prev_unexplained = (unexplained_by_RM[unexplained_[i]] > 0) && (explained_by_RM[unexplained_[i]] == 0);
            unexplained_by_RM[unexplained_[i]] += val * unexplained_distances[i];

            if (val < 0) //the hypothesis is being removed
            {
              if (prev_unexplained)
              {
                //decrease by 1
                add_to_unexplained -= unexplained_distances[i];
              }
            } else //the hypothesis is being added and unexplains unexplained_[i], so increase by 1 unless its explained by another hypothesis
            {
              if (explained_by_RM[unexplained_[i]] == 0)
                add_to_unexplained += unexplained_distances[i];
            }
          }

          for (const int &i : explained)
          {
            if (val < 0)
            {
              //the hypothesis is being removed, check that there are no points that become unexplained and have clutter unexplained hypotheses
              if ((explained_by_RM[i] == 0) && (unexplained_by_RM[i] > 0))
              {
                add_to_unexplained += unexplained_by_RM[i]; //the points become unexplained
              }
            } else
            {
              //std::cout << "being added..." << add_to_unexplained << " " << unexplained_by_RM[explained[i]] << std::endl;
              if ((explained_by_RM[i] == 1) && (unexplained_by_RM[i] > 0))
              { //the only hypothesis explaining that point
                add_to_unexplained -= unexplained_by_RM[i]; //the points are not unexplained any longer because this hypothesis explains them
              }
            }
          }

          //std::cout << add_to_unexplained << std::endl;
          previous_unexplained_ += add_to_unexplained;
        }
      }

      void updateExplainedVector(std::vector<int> & vec, std::vector<float> & vec_float, std::vector<int> & explained_,
          std::vector<float> & explained_by_RM_distance_weighted, float sign)
      {
        float add_to_explained = 0.f;
        int add_to_duplicity_ = 0;

        for (std::size_t i = 0; i < vec.size (); i++)
        {
          bool prev_dup = explained_[vec[i]] > 1;

          explained_[vec[i]] += static_cast<int> (sign);
          explained_by_RM_distance_weighted[vec[i]] += vec_float[i] * sign;

          add_to_explained += vec_float[i] * sign;

          if ((explained_[vec[i]] > 1) && prev_dup)
          { //its still a duplicate, we are adding
            add_to_duplicity_ += static_cast<int> (sign); //so, just add or remove one
          } else if ((explained_[vec[i]] == 1) && prev_dup)
          { //if was duplicate before, now its not, remove 2, we are removing the hypothesis
            add_to_duplicity_ -= 2;
          } else if ((explained_[vec[i]] > 1) && !prev_dup)
          { //it was not a duplicate but it is now, add 2, we are adding a conflicting hypothesis for the point
            add_to_duplicity_ += 2;
          }
        }

        //update explained and duplicity values...
        previous_explained_value += add_to_explained;
        previous_duplicity_ += add_to_duplicity_;
      }

      void updateCMDuplicity(std::vector<int> & vec, std::vector<int> & occupancy_vec, float sign) {
        int add_to_duplicity_ = 0;
        for (const int &i : vec)
        {
          bool prev_dup = occupancy_vec[i] > 1;
          occupancy_vec[i] += static_cast<int> (sign);
          if ((occupancy_vec[i] > 1) && prev_dup)
          { //its still a duplicate, we are adding
            add_to_duplicity_ += static_cast<int> (sign); //so, just add or remove one
          } else if ((occupancy_vec[i] == 1) && prev_dup)
          { //if was duplicate before, now its not, remove 2, we are removing the hypothesis
            add_to_duplicity_ -= 2;
          } else if ((occupancy_vec[i] > 1) && !prev_dup)
          { //it was not a duplicate but it is now, add 2, we are adding a conflicting hypothesis for the point
            add_to_duplicity_ += 2;
          }
        }

        previous_duplicity_complete_models_ += add_to_duplicity_;
      }

      float getTotalExplainedInformation(std::vector<int> & explained_, std::vector<float> & explained_by_RM_distance_weighted, int * duplicity_)
      {
        float explained_info = 0;
        int duplicity = 0;

        for (std::size_t i = 0; i < explained_.size (); i++)
        {
          if (explained_[i] > 0)
            explained_info += explained_by_RM_distance_weighted[i];

          if (explained_[i] > 1)
            duplicity += explained_[i];
        }

        *duplicity_ = duplicity;

        return explained_info;
      }

      float getTotalBadInformation(std::vector<RecognitionModelPtr> & recog_models)
      {
        float bad_info = 0;
        for (std::size_t i = 0; i < recog_models.size (); i++)
          bad_info += recog_models[i]->outliers_weight_ * static_cast<float> (recog_models[i]->bad_information_);

        return bad_info;
      }

      float getUnexplainedInformationInNeighborhood(std::vector<float> & unexplained, std::vector<int> & explained)
      {
        float unexplained_sum = 0.f;
        for (std::size_t i = 0; i < unexplained.size (); i++)
        {
          if (unexplained[i] > 0 && explained[i] == 0)
            unexplained_sum += unexplained[i];
        }

        return unexplained_sum;
      }

      //Performs smooth segmentation of the scene cloud and compute the model cues
      void
      initialize();

      mets::gol_type
      evaluateSolution(const std::vector<bool> & active, int changed);

      bool
      addModel(typename pcl::PointCloud<ModelT>::ConstPtr & model, typename pcl::PointCloud<ModelT>::ConstPtr & complete_model, RecognitionModelPtr & recog_model);

      void
      computeClutterCue(RecognitionModelPtr & recog_model);

      void
      SAOptimize(std::vector<int> & cc_indices, std::vector<bool> & sub_solution);

    public:
      GlobalHypothesesVerification() : HypothesisVerification<ModelT, SceneT>()
      {
        resolution_ = 0.005f;
        max_iterations_ = 5000;
        regularizer_ = 1.f;
        radius_normals_ = 0.01f;
        initial_temp_ = 1000;
        detect_clutter_ = true;
        radius_neighborhood_GO_ = 0.03f;
        clutter_regularizer_ = 5.f;
        res_occupancy_grid_ = 0.01f;
        w_occupied_multiple_cm_ = 4.f;
      }

      void
      verify() override;
      
      void setResolutionOccupancyGrid(float r)
      {
        res_occupancy_grid_ = r;
      }

      void setRadiusNormals(float r)
      {
        radius_normals_ = r;
      }

      void setMaxIterations(int i)
      {
        max_iterations_ = i;
      }

      void setInitialTemp(float t)
      {
        initial_temp_ = t;
      }

      void setRegularizer(float r)
      {
        regularizer_ = r;
      }

      void setRadiusClutter(float r)
      {
        radius_neighborhood_GO_ = r;
      }

      void setClutterRegularizer(float cr)
      {
        clutter_regularizer_ = cr;
      }

      void setDetectClutter(bool d)
      {
        detect_clutter_ = d;
      }
  };
}  // namespace pclcompat_hv
}  // namespace pcl

namespace pcl {
namespace pclcompat_hv {

template<typename PointT, typename NormalT>
inline void extractEuclideanClustersSmooth(const typename pcl::PointCloud<PointT> &cloud, const typename pcl::PointCloud<NormalT> &normals, float tolerance,
    const typename pcl::search::Search<PointT>::Ptr &tree, std::vector<pcl::PointIndices> &clusters, double eps_angle, float curvature_threshold,
    unsigned int min_pts_per_cluster, unsigned int max_pts_per_cluster = (std::numeric_limits<int>::max) ())
{

  if (tree->getInputCloud ()->size () != cloud.size ())
  {
    PCL_ERROR("[pcl::extractEuclideanClusters] Tree built for a different point cloud dataset\n");
    return;
  }
  if (cloud.size () != normals.size ())
  {
    PCL_ERROR("[pcl::extractEuclideanClusters] Number of points in the input point cloud different than normals!\n");
    return;
  }

  // Create a bool vector of processed point indices, and initialize it to false
  std::vector<bool> processed (cloud.size (), false);

  pcl::Indices nn_indices;
  std::vector<float> nn_distances;
  // Process all points in the indices vector
  int size = static_cast<int> (cloud.size ());
  for (int i = 0; i < size; ++i)
  {
    if (processed[i])
      continue;

    std::vector<unsigned int> seed_queue;
    int sq_idx = 0;
    seed_queue.push_back (i);

    processed[i] = true;

    while (sq_idx < static_cast<int> (seed_queue.size ()))
    {

      if (normals[seed_queue[sq_idx]].curvature > curvature_threshold)
      {
        sq_idx++;
        continue;
      }

      // Search for sq_idx
      if (!tree->radiusSearch (seed_queue[sq_idx], tolerance, nn_indices, nn_distances))
      {
        sq_idx++;
        continue;
      }

      for (std::size_t j = 1; j < nn_indices.size (); ++j) // nn_indices[0] should be sq_idx
      {
        if (processed[nn_indices[j]]) // Has this point been processed before ?
          continue;

        if (normals[nn_indices[j]].curvature > curvature_threshold)
        {
          continue;
        }

        //processed[nn_indices[j]] = true;
        // [-1;1]

        double dot_p = normals[seed_queue[sq_idx]].normal[0] * normals[nn_indices[j]].normal[0]
            + normals[seed_queue[sq_idx]].normal[1] * normals[nn_indices[j]].normal[1]
            + normals[seed_queue[sq_idx]].normal[2] * normals[nn_indices[j]].normal[2];

        if (std::abs (std::acos (dot_p)) < eps_angle)
        {
          processed[nn_indices[j]] = true;
          seed_queue.push_back (nn_indices[j]);
        }
      }

      sq_idx++;
    }

    // If this queue is satisfactory, add to the clusters
    if (seed_queue.size () >= min_pts_per_cluster && seed_queue.size () <= max_pts_per_cluster)
    {
      pcl::PointIndices r;
      r.indices.resize (seed_queue.size ());
      for (std::size_t j = 0; j < seed_queue.size (); ++j)
        r.indices[j] = seed_queue[j];

      std::sort (r.indices.begin (), r.indices.end ());
      r.indices.erase (std::unique (r.indices.begin (), r.indices.end ()), r.indices.end ());

      r.header = cloud.header;
      clusters.push_back (r); // We could avoid a copy by working directly in the vector
    }
  }
}

}  // namespace pclcompat_hv
}  // namespace pcl

template<typename ModelT, typename SceneT>
mets::gol_type pcl::pclcompat_hv::GlobalHypothesesVerification<ModelT, SceneT>::evaluateSolution(const std::vector<bool> & active, int changed)
{
  float sign = 1.f;
  //update explained_by_RM
  if (active[changed])
  {
    //it has been activated
    updateExplainedVector (recognition_models_[changed]->explained_, recognition_models_[changed]->explained_distances_, explained_by_RM_,
        explained_by_RM_distance_weighted, 1.f);
    updateUnexplainedVector (recognition_models_[changed]->unexplained_in_neighborhood, recognition_models_[changed]->unexplained_in_neighborhood_weights,
        unexplained_by_RM_neighboorhods, recognition_models_[changed]->explained_, explained_by_RM_, 1.f);
    updateCMDuplicity(recognition_models_[changed]->complete_cloud_occupancy_indices_, complete_cloud_occupancy_by_RM_, 1.f);
  } else
  {
    //it has been deactivated
    updateExplainedVector (recognition_models_[changed]->explained_, recognition_models_[changed]->explained_distances_, explained_by_RM_,
        explained_by_RM_distance_weighted, -1.f);
    updateUnexplainedVector (recognition_models_[changed]->unexplained_in_neighborhood, recognition_models_[changed]->unexplained_in_neighborhood_weights,
        unexplained_by_RM_neighboorhods, recognition_models_[changed]->explained_, explained_by_RM_, -1.f);
    updateCMDuplicity(recognition_models_[changed]->complete_cloud_occupancy_indices_, complete_cloud_occupancy_by_RM_, -1.f);
    sign = -1.f;
  }

  int duplicity = getDuplicity ();
  float good_info = getExplainedValue ();

  float unexplained_info = getPreviousUnexplainedValue ();
  float bad_info = static_cast<float> (getPreviousBadInfo ())
      + (recognition_models_[changed]->outliers_weight_ * static_cast<float> (recognition_models_[changed]->bad_information_)) * sign;

  setPreviousBadInfo (bad_info);

  int n_active_hyp = 0;
  for(const bool i : active) {
    if(i)
      n_active_hyp++;
  }

  float duplicity_cm = static_cast<float> (getDuplicityCM ()) * w_occupied_multiple_cm_;
  return static_cast<mets::gol_type> ((good_info - bad_info - static_cast<float> (duplicity) - unexplained_info - duplicity_cm - static_cast<float> (n_active_hyp)) * -1.f); //return the dual to our max problem
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template<typename ModelT, typename SceneT>
void pcl::pclcompat_hv::GlobalHypothesesVerification<ModelT, SceneT>::initialize()
{
  //clear stuff
  recognition_models_.clear ();
  unexplained_by_RM_neighboorhods.clear ();
  explained_by_RM_distance_weighted.clear ();
  explained_by_RM_.clear ();
  mask_.clear ();
  indices_.clear (),
  complete_cloud_occupancy_by_RM_.clear ();

  // initialize mask to false
  mask_.resize (complete_models_.size ());
  for (std::size_t i = 0; i < complete_models_.size (); i++)
    mask_[i] = false;

  indices_.resize (complete_models_.size ());

  NormalEstimator_ n3d;
  scene_normals_.reset (new pcl::PointCloud<pcl::Normal> ());

  typename pcl::search::KdTree<SceneT>::Ptr normals_tree (new pcl::search::KdTree<SceneT>);
  normals_tree->setInputCloud (scene_cloud_downsampled_);

  n3d.setRadiusSearch (radius_normals_);
  n3d.setSearchMethod (normals_tree);
  n3d.setInputCloud (scene_cloud_downsampled_);
  n3d.compute (*scene_normals_);

  //check nans...
  int j = 0;
  for (std::size_t i = 0; i < scene_normals_->size (); ++i)
  {
    if (!std::isfinite ((*scene_normals_)[i].normal_x) || !std::isfinite ((*scene_normals_)[i].normal_y)
        || !std::isfinite ((*scene_normals_)[i].normal_z))
      continue;

    (*scene_normals_)[j] = (*scene_normals_)[i];
    (*scene_cloud_downsampled_)[j] = (*scene_cloud_downsampled_)[i];

    j++;
  }

  scene_normals_->points.resize (j);
  scene_normals_->width = j;
  scene_normals_->height = 1;

  scene_cloud_downsampled_->points.resize (j);
  scene_cloud_downsampled_->width = j;
  scene_cloud_downsampled_->height = 1;

  explained_by_RM_.resize (scene_cloud_downsampled_->size (), 0);
  explained_by_RM_distance_weighted.resize (scene_cloud_downsampled_->size (), 0.f);
  unexplained_by_RM_neighboorhods.resize (scene_cloud_downsampled_->size (), 0.f);

  //compute segmentation of the scene if detect_clutter_
  if (detect_clutter_)
  {
    //initialize kdtree for search
    scene_downsampled_tree_.reset (new pcl::search::KdTree<SceneT>);
    scene_downsampled_tree_->setInputCloud (scene_cloud_downsampled_);

    std::vector<pcl::PointIndices> clusters;
    double eps_angle_threshold = 0.2;
    int min_points = 20;
    float curvature_threshold = 0.045f;

    extractEuclideanClustersSmooth<SceneT, pcl::Normal> (*scene_cloud_downsampled_, *scene_normals_, inliers_threshold_ * 2.f, scene_downsampled_tree_,
        clusters, eps_angle_threshold, curvature_threshold, min_points);

    clusters_cloud_.reset (new pcl::PointCloud<pcl::PointXYZI>);
    clusters_cloud_->points.resize (scene_cloud_downsampled_->size ());
    clusters_cloud_->width = scene_cloud_downsampled_->width;
    clusters_cloud_->height = 1;

    for (std::size_t i = 0; i < scene_cloud_downsampled_->size (); i++)
    {
      pcl::PointXYZI p;
      p.getVector3fMap () = (*scene_cloud_downsampled_)[i].getVector3fMap ();
      p.intensity = 0.f;
      (*clusters_cloud_)[i] = p;
    }

    float intens_incr = 100.f / static_cast<float> (clusters.size ());
    float intens = intens_incr;
    for (const auto &cluster : clusters)
    {
      for (const auto &vertex : cluster.indices)
      {
        (*clusters_cloud_)[vertex].intensity = intens;
      }

      intens += intens_incr;
    }
  }

  //compute cues
  {
    pcl::pclcompat_hv::SilentScope tcues ("Computing cues");
    recognition_models_.resize (complete_models_.size ());
    int valid = 0;
    for (int i = 0; i < static_cast<int> (complete_models_.size ()); i++)
    {
      //create recognition model
      recognition_models_[valid].reset (new RecognitionModel ());
      if(addModel (visible_models_[i], complete_models_[i], recognition_models_[valid])) {
        indices_[valid] = i;
        valid++;
      }
    }

    recognition_models_.resize(valid);
    indices_.resize(valid);
  }

  //compute the bounding boxes for the models
  ModelT min_pt_all, max_pt_all;
  min_pt_all.x = min_pt_all.y = min_pt_all.z = std::numeric_limits<float>::max ();
  max_pt_all.x = max_pt_all.y = max_pt_all.z = (std::numeric_limits<float>::max () - 0.001f) * -1;

  for (std::size_t i = 0; i < recognition_models_.size (); i++)
  {
    ModelT min_pt, max_pt;
    pcl::getMinMax3D (*complete_models_[indices_[i]], min_pt, max_pt);
    if (min_pt.x < min_pt_all.x)
      min_pt_all.x = min_pt.x;

    if (min_pt.y < min_pt_all.y)
      min_pt_all.y = min_pt.y;

    if (min_pt.z < min_pt_all.z)
      min_pt_all.z = min_pt.z;

    if (max_pt.x > max_pt_all.x)
      max_pt_all.x = max_pt.x;

    if (max_pt.y > max_pt_all.y)
      max_pt_all.y = max_pt.y;

    if (max_pt.z > max_pt_all.z)
      max_pt_all.z = max_pt.z;
  }

  int size_x, size_y, size_z;
  size_x = static_cast<int> (std::ceil (std::abs (max_pt_all.x - min_pt_all.x) / res_occupancy_grid_)) + 1;
  size_y = static_cast<int> (std::ceil (std::abs (max_pt_all.y - min_pt_all.y) / res_occupancy_grid_)) + 1;
  size_z = static_cast<int> (std::ceil (std::abs (max_pt_all.z - min_pt_all.z) / res_occupancy_grid_)) + 1;

  complete_cloud_occupancy_by_RM_.resize (size_x * size_y * size_z, 0);

  for (std::size_t i = 0; i < recognition_models_.size (); i++)
  {

    std::map<int, bool> banned;
    std::map<int, bool>::iterator banned_it;

    for (const auto& point: *complete_models_[indices_[i]])
    {
      const int pos_x = static_cast<int> (std::floor ((point.x - min_pt_all.x) / res_occupancy_grid_));
      const int pos_y = static_cast<int> (std::floor ((point.y - min_pt_all.y) / res_occupancy_grid_));
      const int pos_z = static_cast<int> (std::floor ((point.z - min_pt_all.z) / res_occupancy_grid_));

      const int idx = pos_z * size_x * size_y + pos_y * size_x + pos_x;
      banned_it = banned.find (idx);
      if (banned_it == banned.end ())
      {
        complete_cloud_occupancy_by_RM_[idx]++;
        recognition_models_[i]->complete_cloud_occupancy_indices_.push_back (idx);
        banned[idx] = true;
      }
    }
  }

  {
    pcl::pclcompat_hv::SilentScope tcues ("Computing clutter cues");
    for (int j = 0; j < static_cast<int> (recognition_models_.size ()); j++)
      computeClutterCue (recognition_models_[j]);
  }

  cc_.clear ();
  n_cc_ = 1;
  cc_.resize (n_cc_);
  for (std::size_t i = 0; i < recognition_models_.size (); i++)
    cc_[0].push_back (static_cast<int> (i));

}

template<typename ModelT, typename SceneT>
void pcl::pclcompat_hv::GlobalHypothesesVerification<ModelT, SceneT>::SAOptimize(std::vector<int> & cc_indices, std::vector<bool> & initial_solution)
{

  //temporal copy of recogniton_models_
  std::vector<RecognitionModelPtr> recognition_models_copy;
  recognition_models_copy = recognition_models_;

  recognition_models_.clear ();

  for (const int &cc_index : cc_indices)
  {
    recognition_models_.push_back (recognition_models_copy[cc_index]);
  }

  for (std::size_t j = 0; j < recognition_models_.size (); j++)
  {
    RecognitionModelPtr recog_model = recognition_models_[j];
    for (std::size_t i = 0; i < recog_model->explained_.size (); i++)
    {
      explained_by_RM_[recog_model->explained_[i]]++;
      explained_by_RM_distance_weighted[recog_model->explained_[i]] += recog_model->explained_distances_[i];
    }

    if (detect_clutter_)
    {
      for (std::size_t i = 0; i < recog_model->unexplained_in_neighborhood.size (); i++)
      {
        unexplained_by_RM_neighboorhods[recog_model->unexplained_in_neighborhood[i]] += recog_model->unexplained_in_neighborhood_weights[i];
      }
    }
  }

  int occupied_multiple = 0;
  for(const auto& i : complete_cloud_occupancy_by_RM_) {
    if(i > 1) {
      occupied_multiple+=i;
    }
  }

  setPreviousDuplicityCM(occupied_multiple);
  //do optimization
  //Define model SAModel, initial solution is all models activated

  int duplicity;
  float good_information_ = getTotalExplainedInformation (explained_by_RM_, explained_by_RM_distance_weighted, &duplicity);
  float bad_information_ = 0;
  float unexplained_in_neighboorhod = getUnexplainedInformationInNeighborhood (unexplained_by_RM_neighboorhods, explained_by_RM_);

  for (std::size_t i = 0; i < initial_solution.size (); i++)
  {
    if (initial_solution[i])
      bad_information_ += recognition_models_[i]->outliers_weight_ * static_cast<float> (recognition_models_[i]->bad_information_);
  }

  setPreviousExplainedValue (good_information_);
  setPreviousDuplicity (duplicity);
  setPreviousBadInfo (bad_information_);
  setPreviousUnexplainedValue (unexplained_in_neighboorhod);

  SAModel model;
  model.cost_ = static_cast<mets::gol_type> ((good_information_ - bad_information_
                                               - static_cast<float> (duplicity)
                                               - static_cast<float> (occupied_multiple) * w_occupied_multiple_cm_
                                               - static_cast<float> (recognition_models_.size ())
                                               - unexplained_in_neighboorhod) * -1.f);

  model.setSolution (initial_solution);
  model.setOptimizer (this);
  SAModel best (model);

  move_manager neigh (static_cast<int> (cc_indices.size ()));

  mets::best_ever_solution best_recorder (best);
  mets::noimprove_termination_criteria noimprove (max_iterations_);
  mets::linear_cooling linear_cooling;
  mets::simulated_annealing<move_manager> sa (model, best_recorder, neigh, noimprove, linear_cooling, initial_temp_, 1e-7, 2);
  sa.setApplyAndEvaluate(true);

  {
    pcl::pclcompat_hv::SilentScope t ("SA search...");
    sa.search ();
  }

  best_seen_ = static_cast<const SAModel&> (best_recorder.best_seen ());
  for (std::size_t i = 0; i < best_seen_.solution_.size (); i++)
  {
    initial_solution[i] = best_seen_.solution_[i];
  }

  recognition_models_ = recognition_models_copy;

}

///////////////////////////////////////////////////////////////////////////////////////////////////
template<typename ModelT, typename SceneT>
void pcl::pclcompat_hv::GlobalHypothesesVerification<ModelT, SceneT>::verify()
{
  initialize ();

  //for each connected component, find the optimal solution
  for (int c = 0; c < n_cc_; c++)
  {
    //TODO: Check for trivial case...
    //TODO: Check also the number of hypotheses and use exhaustive enumeration if smaller than 10
    std::vector<bool> subsolution (cc_[c].size (), true);
    SAOptimize (cc_[c], subsolution);
    for (std::size_t i = 0; i < subsolution.size (); i++)
    {
      mask_[indices_[cc_[c][i]]] = (subsolution[i]);
    }
  }
}

template<typename ModelT, typename SceneT>
bool pcl::pclcompat_hv::GlobalHypothesesVerification<ModelT, SceneT>::addModel(typename pcl::PointCloud<ModelT>::ConstPtr & model,
    typename pcl::PointCloud<ModelT>::ConstPtr & complete_model, RecognitionModelPtr & recog_model)
{
  //voxelize model cloud
  recog_model->cloud_.reset (new pcl::PointCloud<ModelT> ());
  recog_model->complete_cloud_.reset (new pcl::PointCloud<ModelT> ());

  float size_model = resolution_;
  pcl::VoxelGrid<ModelT> voxel_grid;
  voxel_grid.setInputCloud (model);
  voxel_grid.setLeafSize (size_model, size_model, size_model);
  voxel_grid.filter (*(recog_model->cloud_));

  pcl::VoxelGrid<ModelT> voxel_grid2;
  voxel_grid2.setInputCloud (complete_model);
  voxel_grid2.setLeafSize (size_model, size_model, size_model);
  voxel_grid2.filter (*(recog_model->complete_cloud_));

  {
    //check nans...
    int j = 0;
    for (auto& point: *(recog_model->cloud_))
    {
      if (!isXYZFinite (point))
        continue;

      (*recog_model->cloud_)[j] = point;
      j++;
    }

    recog_model->cloud_->points.resize (j);
    recog_model->cloud_->width = j;
    recog_model->cloud_->height = 1;
  }

  if (recog_model->cloud_->points.empty ())
  {
    PCL_WARN("The model cloud has no points..\n");
    return false;
  }

  //compute normals unless given (now do it always...)
  typename pcl::search::KdTree<ModelT>::Ptr normals_tree (new pcl::search::KdTree<ModelT>);
  pcl::NormalEstimation<ModelT, pcl::Normal> n3d;
  recog_model->normals_.reset (new pcl::PointCloud<pcl::Normal> ());
  normals_tree->setInputCloud (recog_model->cloud_);
  n3d.setRadiusSearch (radius_normals_);
  n3d.setSearchMethod (normals_tree);
  n3d.setInputCloud ((recog_model->cloud_));
  n3d.compute (*(recog_model->normals_));

  //check nans...
  int j = 0;
  for (std::size_t i = 0; i < recog_model->normals_->size (); ++i)
  {
    // pclcompat: PCL tests isNormalFinite here, keeping only the points
    // whose normal could NOT be estimated.
    if (!isNormalFinite((*recog_model->normals_)[i]))
      continue;

    (*recog_model->normals_)[j] = (*recog_model->normals_)[i];
    (*recog_model->cloud_)[j] = (*recog_model->cloud_)[i];
    j++;
  }

  recog_model->normals_->points.resize (j);
  recog_model->normals_->width = j;
  recog_model->normals_->height = 1;

  recog_model->cloud_->points.resize (j);
  recog_model->cloud_->width = j;
  recog_model->cloud_->height = 1;

  std::vector<int> explained_indices;
  std::vector<float> outliers_weight;
  std::vector<float> explained_indices_distances;

  pcl::Indices nn_indices;
  std::vector<float> nn_distances;

  std::map<int, std::shared_ptr<std::vector<std::pair<int, float>>>> model_explains_scene_points; //which point i from the scene is explained by a points j_k with dist d_k from the model

  outliers_weight.resize (recog_model->cloud_->size ());
  recog_model->outlier_indices_.resize (recog_model->cloud_->size ());

  std::size_t o = 0;
  for (std::size_t i = 0; i < recog_model->cloud_->size (); i++)
  {
    if (!scene_downsampled_tree_->radiusSearch ((*recog_model->cloud_)[i], inliers_threshold_, nn_indices, nn_distances, std::numeric_limits<int>::max ()))
    {
      //outlier
      outliers_weight[o] = regularizer_;
      recog_model->outlier_indices_[o] = static_cast<int> (i);
      o++;
    } else
    {
      for (std::size_t k = 0; k < nn_distances.size (); k++)
      {
        std::pair<int, float> pair = std::make_pair (i, nn_distances[k]); //i is a index to a model point and then distance
        auto it = model_explains_scene_points.find (nn_indices[k]);
        if (it == model_explains_scene_points.end ())
        {
          std::shared_ptr<std::vector<std::pair<int, float>>> vec (new std::vector<std::pair<int, float>> ());
          vec->push_back (pair);
          model_explains_scene_points[nn_indices[k]] = vec;
        } else
        {
          it->second->push_back (pair);
        }
      }
    }
  }

  outliers_weight.resize (o);
  recog_model->outlier_indices_.resize (o);

  recog_model->outliers_weight_ = (std::accumulate (outliers_weight.begin (), outliers_weight.end (), 0.f) / static_cast<float> (outliers_weight.size ()));
  if (outliers_weight.empty ())
    recog_model->outliers_weight_ = 1.f;

  pcl::IndicesPtr indices_scene (new pcl::Indices);
  //go through the map and keep the closest model point in case that several model points explain a scene point

  int p = 0;

  for (auto it = model_explains_scene_points.cbegin (); it != model_explains_scene_points.cend (); it++, p++)
  {
    std::size_t closest = 0;
    float min_d = std::numeric_limits<float>::min ();
    for (std::size_t i = 0; i < it->second->size (); i++)
    {
      if (it->second->at (i).second > min_d)
      {
        min_d = it->second->at (i).second;
        closest = i;
      }
    }

    float d = it->second->at (closest).second;
    float d_weight = -(d * d / (inliers_threshold_)) + 1;

    //it->first is index to scene point
    //using normals to weight inliers
    Eigen::Vector3f scene_p_normal = (*scene_normals_)[it->first].getNormalVector3fMap ();
    Eigen::Vector3f model_p_normal =
        (*recog_model->normals_)[it->second->at(closest).first].getNormalVector3fMap();
    float dotp = scene_p_normal.dot (model_p_normal) * 1.f; //[-1,1] from antiparallel trough perpendicular to parallel

    if (dotp < 0.f)
      dotp = 0.f;

    explained_indices.push_back (it->first);
    explained_indices_distances.push_back (d_weight * dotp);

  }

  recog_model->bad_information_ = static_cast<int> (recog_model->outlier_indices_.size ());
  recog_model->explained_ = explained_indices;
  recog_model->explained_distances_ = explained_indices_distances;

  return true;
}

template<typename ModelT, typename SceneT>
void pcl::pclcompat_hv::GlobalHypothesesVerification<ModelT, SceneT>::computeClutterCue(RecognitionModelPtr & recog_model)
{
  if (detect_clutter_)
  {

    float rn_sqr = radius_neighborhood_GO_ * radius_neighborhood_GO_;
    pcl::Indices nn_indices;
    std::vector<float> nn_distances;

    std::vector < std::pair<int, int> > neighborhood_indices; //first is indices to scene point and second is indices to explained_ scene points
    for (pcl::index_t i = 0; i < static_cast<pcl::index_t> (recog_model->explained_.size ()); i++)
    {
      if (scene_downsampled_tree_->radiusSearch ((*scene_cloud_downsampled_)[recog_model->explained_[i]], radius_neighborhood_GO_, nn_indices,
          nn_distances, std::numeric_limits<int>::max ()))
      {
        for (std::size_t k = 0; k < nn_distances.size (); k++)
        {
          if (nn_indices[k] != i)
            neighborhood_indices.emplace_back (nn_indices[k], i);
        }
      }
    }

    //sort neighborhood indices by id
    std::sort (neighborhood_indices.begin (), neighborhood_indices.end (),
        [] (const auto& p1, const auto& p2) { return p1.first < p2.first; });

    //erase duplicated unexplained points
    neighborhood_indices.erase (
        std::unique (neighborhood_indices.begin (), neighborhood_indices.end (),
            [] (const auto& p1, const auto& p2) { return p1.first == p2.first; }), neighborhood_indices.end ());

    //sort explained points
    std::vector<int> exp_idces (recog_model->explained_);
    std::sort (exp_idces.begin (), exp_idces.end ());

    recog_model->unexplained_in_neighborhood.resize (neighborhood_indices.size ());
    recog_model->unexplained_in_neighborhood_weights.resize (neighborhood_indices.size ());

    std::size_t p = 0;
    std::size_t j = 0;
    for (const auto &neighborhood_index : neighborhood_indices)
    {
      if ((j < exp_idces.size ()) && (neighborhood_index.first == exp_idces[j]))
      {
        //this index is explained by the hypothesis so ignore it, advance j
        j++;
      } else
      {
        //indices_in_nb[i] < exp_idces[j]
        //recog_model->unexplained_in_neighborhood.push_back(neighborhood_indices[i]);
        recog_model->unexplained_in_neighborhood[p] = neighborhood_index.first;

        if ((*clusters_cloud_)[recog_model->explained_[neighborhood_index.second]].intensity != 0.f
            && ((*clusters_cloud_)[recog_model->explained_[neighborhood_index.second]].intensity
                == (*clusters_cloud_)[neighborhood_index.first].intensity))
        {

          recog_model->unexplained_in_neighborhood_weights[p] = clutter_regularizer_;

        } else
        {
          //neighborhood_indices[i].first gives the index to the scene point and second to the explained scene point by the model causing this...
          //calculate weight of this clutter point based on the distance of the scene point and the model point causing it
          float d = static_cast<float> (pow (
              ((*scene_cloud_downsampled_)[recog_model->explained_[neighborhood_index.second]].getVector3fMap ()
                  - (*scene_cloud_downsampled_)[neighborhood_index.first].getVector3fMap ()).norm (), 2));
          float d_weight = -(d / rn_sqr) + 1; //points that are close have a strong weight*/

          //using normals to weight clutter points
          Eigen::Vector3f scene_p_normal = (*scene_normals_)[neighborhood_index.first].getNormalVector3fMap ();
          Eigen::Vector3f model_p_normal = (*scene_normals_)[recog_model->explained_[neighborhood_index.second]].getNormalVector3fMap ();
          float dotp = scene_p_normal.dot (model_p_normal); //[-1,1] from antiparallel trough perpendicular to parallel

          if (dotp < 0)
            dotp = 0.f;

          recog_model->unexplained_in_neighborhood_weights[p] = d_weight * dotp;
        }
        p++;
      }
    }

    recog_model->unexplained_in_neighborhood_weights.resize (p);
    recog_model->unexplained_in_neighborhood.resize (p);
  }
}
