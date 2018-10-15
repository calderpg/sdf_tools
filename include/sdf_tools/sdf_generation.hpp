#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <chrono>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <Eigen/Geometry>
#include <visualization_msgs/Marker.h>
#include <arc_utilities/eigen_helpers.hpp>
#include <arc_utilities/voxel_grid.hpp>
#include <arc_utilities/pretty_print.hpp>
#include <sdf_tools/sdf.hpp>

#ifndef SDF_GENERATION_HPP
#define SDF_GENERATION_HPP

namespace sdf_generation
{
struct bucket_cell
{
  double distance_square;
  int32_t update_direction;
  uint32_t location[3];
  uint32_t closest_point[3];
};

typedef VoxelGrid::VoxelGrid<bucket_cell> DistanceField;

inline int32_t GetDirectionNumber(
    const int32_t dx, const int32_t dy, const int32_t dz)
{
  return ((dx + 1) * 9) + ((dy + 1) * 3) + (dz + 1);
}

inline std::vector<std::vector<std::vector<std::vector<int32_t>>>>
MakeNeighborhoods()
{
  // First vector<>: 2 - the first bucket queue, the points we know are zero
  // distance, start with a complete set of neighbors to check. Every other
  // bucket queue checks fewer neighbors.
  // Second vector<>: 27 (# of source directions in fully-connected 3d grid).
  // Third vector<>:
  std::vector<std::vector<std::vector<std::vector<int32_t>>>> neighborhoods;
  // I don't know why there are 2 initial neighborhoods.
  neighborhoods.resize(2);
  for (size_t n = 0; n < neighborhoods.size(); n++)
  {
    neighborhoods[n].resize(27);
    // Loop through the source directions.
    for (int32_t dx = -1; dx <= 1; dx++)
    {
      for (int32_t dy = -1; dy <= 1; dy++)
      {
        for (int32_t dz = -1; dz <= 1; dz++)
        {
          const int32_t direction_number = GetDirectionNumber(dx, dy, dz);
          // Loop through the target directions.
          for (int32_t tdx = -1; tdx <= 1; tdx++)
          {
            for (int32_t tdy = -1; tdy <= 1; tdy++)
            {
              for (int32_t tdz = -1; tdz <= 1; tdz++)
              {
                // Ignore the case of ourself.
                if (tdx == 0 && tdy == 0 && tdz == 0)
                {
                  continue;
                }
                // Why is one set of neighborhoods larger than the other?
                if (n >= 1)
                {
                  if ((abs(tdx) + abs(tdy) + abs(tdz)) != 1)
                  {
                    continue;
                  }
                  if ((dx * tdx) < 0 || (dy * tdy) < 0 || (dz * tdz) < 0)
                  {
                    continue;
                  }
                }
                std::vector<int32_t> new_point;
                new_point.resize(3);
                new_point[0] = tdx;
                new_point[1] = tdy;
                new_point[2] = tdz;
                neighborhoods[n][direction_number].push_back(new_point);
              }
            }
          }
        }
      }
    }
  }
  return neighborhoods;
}

inline double ComputeDistanceSquared(
    const int32_t x1, const int32_t y1, const int32_t z1,
    const int32_t x2, const int32_t y2, const int32_t z2)
{
  const int32_t dx = x1 - x2;
  const int32_t dy = y1 - y2;
  const int32_t dz = z1 - z2;
  return double((dx * dx) + (dy * dy) + (dz * dz));
}

class MultipleThreadIndexQueueWrapper
{
public:

  explicit MultipleThreadIndexQueueWrapper(const size_t max_queues)
  {
    per_thread_queues_.resize(
          GetNumOMPThreads(), ThreadIndexQueues(max_queues));
  }

  const VoxelGrid::GRID_INDEX& Query(
      const int32_t distance_squared, const size_t idx) const
  {
    size_t working_index = idx;
    for (size_t thread = 0; thread < per_thread_queues_.size(); thread++)
    {
      const auto& current_thread_queue =
          per_thread_queues_.at(thread).at(distance_squared);
      const size_t current_thread_queue_size = current_thread_queue.size();
      if (working_index < current_thread_queue_size)
      {
        return current_thread_queue.at(working_index);
      }
      else
      {
        working_index -= current_thread_queue_size;
      }
    }
    throw std::runtime_error("Failed to find item");
  }

  size_t NumQueues() const
  {
    return per_thread_queues_.at(0).size();
  }

  size_t Size(const int32_t distance_squared) const
  {
    size_t total_size = 0;
    for (size_t thread = 0; thread < per_thread_queues_.size(); thread++)
    {
      total_size += per_thread_queues_.at(thread).at(distance_squared).size();
    }
    return total_size;
  }

  void Enqueue(
      const int32_t distance_squared, const VoxelGrid::GRID_INDEX& index)
  {
#if defined(_OPENMP)
    const size_t thread_num = (size_t)omp_get_thread_num();
#else
    const size_t thread_num = 0;
#endif
    per_thread_queues_.at(thread_num).at(distance_squared).push_back(index);
  }

  void ClearCompletedQueues(const int32_t distance_squared)
  {
    for (size_t thread = 0; thread < per_thread_queues_.size(); thread++)
    {
      per_thread_queues_.at(thread).at(distance_squared).clear();
    }
  }

private:

  inline static size_t GetNumOMPThreads()
  {
#if defined(_OPENMP)
    size_t num_threads = 0;
#pragma omp parallel
    {
      num_threads = (size_t)omp_get_num_threads();
    }
    return num_threads;
#else
    return 1;
#endif
  }

  typedef std::vector<std::vector<VoxelGrid::GRID_INDEX>> ThreadIndexQueues;
  std::vector<ThreadIndexQueues> per_thread_queues_;

};

inline DistanceField BuildDistanceFieldSerial(
    const Eigen::Isometry3d& grid_origin_transform,
    const double grid_resolution,
    const int64_t grid_num_x_cells,
    const int64_t grid_num_y_cells,
    const int64_t grid_num_z_cells,
    const std::vector<VoxelGrid::GRID_INDEX>& points)
{
  const std::chrono::time_point<std::chrono::steady_clock> start_time
      = std::chrono::steady_clock::now();
  // Make the DistanceField container
  bucket_cell default_cell;
  default_cell.distance_square = std::numeric_limits<double>::infinity();
  DistanceField distance_field(grid_origin_transform, grid_resolution,
                               grid_num_x_cells, grid_num_y_cells,
                               grid_num_z_cells, default_cell);
  // Compute maximum distance square
  const int64_t max_distance_square =
      (distance_field.GetNumXCells() * distance_field.GetNumXCells())
      + (distance_field.GetNumYCells() * distance_field.GetNumYCells())
      + (distance_field.GetNumZCells() * distance_field.GetNumZCells());
  // Make bucket queue
  std::vector<std::vector<bucket_cell>> bucket_queue(max_distance_square + 1);
  bucket_queue[0].reserve(points.size());
  // Set initial update direction
  int32_t initial_update_direction = GetDirectionNumber(0, 0, 0);
  // Mark all provided points with distance zero and add to the bucket queue
  for (size_t index = 0; index < points.size(); index++)
  {
    const VoxelGrid::GRID_INDEX& current_index = points[index];
    auto query = distance_field.GetMutable(current_index);
    if (query)
    {
      query.Value().location[0] = (uint32_t)current_index.x;
      query.Value().location[1] = (uint32_t)current_index.y;
      query.Value().location[2] = (uint32_t)current_index.z;
      query.Value().closest_point[0] = (uint32_t)current_index.x;
      query.Value().closest_point[1] = (uint32_t)current_index.y;
      query.Value().closest_point[2] = (uint32_t)current_index.z;
      query.Value().distance_square = 0.0;
      query.Value().update_direction = initial_update_direction;
      bucket_queue[0].push_back(query.Value());
    }
    // If the point is outside the bounds of the SDF, skip
    else
    {
      throw std::runtime_error("Point for BuildDistanceField out of bounds");
    }
  }
  // HERE BE DRAGONS
  // Process the bucket queue
  const std::vector<std::vector<std::vector<std::vector<int>>>> neighborhoods =
      MakeNeighborhoods();
  for (size_t bq_idx = 0; bq_idx < bucket_queue.size(); bq_idx++)
  {
    for (const auto& cur_cell : bucket_queue[bq_idx])
    {
      // Get the current location
      const double x = cur_cell.location[0];
      const double y = cur_cell.location[1];
      const double z = cur_cell.location[2];
      // Pick the update direction
      // Only the first bucket queue gets the larger set of neighborhoods?
      // Don't really userstand why.
      const size_t direction_switch = (bq_idx > 0) ? 1 : 0;
      // Make sure the update direction is valid
      if (cur_cell.update_direction < 0 || cur_cell.update_direction > 26)
      {
        continue;
      }
      // Get the current neighborhood list
      const std::vector<std::vector<int>>& neighborhood =
          neighborhoods[direction_switch][cur_cell.update_direction];
      // Update the distance from the neighboring cells
      for (size_t nh_idx = 0; nh_idx < neighborhood.size(); nh_idx++)
      {
        // Get the direction to check
        const int dx = neighborhood[nh_idx][0];
        const int dy = neighborhood[nh_idx][1];
        const int dz = neighborhood[nh_idx][2];
        const int nx = x + dx;
        const int ny = y + dy;
        const int nz = z + dz;
        auto neighbor_query =
            distance_field.GetMutable((int64_t)nx, (int64_t)ny, (int64_t)nz);
        if (!neighbor_query)
        {
          // "Neighbor" is outside the bounds of the SDF
          continue;
        }
        // Update the neighbor's distance based on the current
        const int32_t new_distance_square =
            (int32_t)ComputeDistanceSquared(nx, ny, nz,
                                            cur_cell.closest_point[0],
                                            cur_cell.closest_point[1],
                                            cur_cell.closest_point[2]);
        if (new_distance_square > max_distance_square)
        {
          // Skip these cases
          continue;
        }
        if (new_distance_square < neighbor_query.Value().distance_square)
        {
          // If the distance is better, time to update the neighbor
          neighbor_query.Value().distance_square = new_distance_square;
          neighbor_query.Value().closest_point[0] = cur_cell.closest_point[0];
          neighbor_query.Value().closest_point[1] = cur_cell.closest_point[1];
          neighbor_query.Value().closest_point[2] = cur_cell.closest_point[2];
          neighbor_query.Value().location[0] = nx;
          neighbor_query.Value().location[1] = ny;
          neighbor_query.Value().location[2] = nz;
          neighbor_query.Value().update_direction =
              GetDirectionNumber(dx, dy, dz);
          // Add the neighbor into the bucket queue
          bucket_queue[new_distance_square].push_back(neighbor_query.Value());
        }
      }
    }
    // Clear the current queue now that we're done with it
    bucket_queue[bq_idx].clear();
  }
  const std::chrono::time_point<std::chrono::steady_clock> end_time
      = std::chrono::steady_clock::now();
  const std::chrono::duration<double> elapsed = end_time - start_time;
  std::cout << "Computed DistanceField for grid size (" << grid_num_x_cells
            << ", " << grid_num_y_cells << ", " << grid_num_z_cells << ") in "
            << elapsed.count() << " seconds" << std::endl;
  return distance_field;
}

inline DistanceField BuildDistanceFieldParallel(
    const Eigen::Isometry3d& grid_origin_transform,
    const double grid_resolution,
    const int64_t grid_num_x_cells,
    const int64_t grid_num_y_cells,
    const int64_t grid_num_z_cells,
    const std::vector<VoxelGrid::GRID_INDEX>& points)
{
  const std::chrono::time_point<std::chrono::steady_clock> start_time
      = std::chrono::steady_clock::now();
  // Make the DistanceField container
  bucket_cell default_cell;
  default_cell.distance_square = std::numeric_limits<double>::infinity();
  DistanceField distance_field(grid_origin_transform, grid_resolution,
                               grid_num_x_cells, grid_num_y_cells,
                               grid_num_z_cells, default_cell);
  // Compute maximum distance square
  const int64_t max_distance_square =
      (distance_field.GetNumXCells() * distance_field.GetNumXCells())
      + (distance_field.GetNumYCells() * distance_field.GetNumYCells())
      + (distance_field.GetNumZCells() * distance_field.GetNumZCells());
  // Make bucket queue
  std::vector<std::vector<bucket_cell>> bucket_queue(max_distance_square + 1);
  bucket_queue[0].reserve(points.size());
  MultipleThreadIndexQueueWrapper bucket_queues(max_distance_square + 1);
  // Set initial update direction
  int32_t initial_update_direction = GetDirectionNumber(0, 0, 0);
  // Mark all provided points with distance zero and add to the bucket queues
  // points MUST NOT CONTAIN DUPLICATE ENTRIES!
#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for (size_t index = 0; index < points.size(); index++)
  {
    const VoxelGrid::GRID_INDEX& current_index = points[index];
    auto query = distance_field.GetMutable(current_index);
    if (query)
    {
      query.Value().location[0] = (uint32_t)current_index.x;
      query.Value().location[1] = (uint32_t)current_index.y;
      query.Value().location[2] = (uint32_t)current_index.z;
      query.Value().closest_point[0] = (uint32_t)current_index.x;
      query.Value().closest_point[1] = (uint32_t)current_index.y;
      query.Value().closest_point[2] = (uint32_t)current_index.z;
      query.Value().distance_square = 0.0;
      query.Value().update_direction = initial_update_direction;
      bucket_queues.Enqueue(0, current_index);
    }
    // If the point is outside the bounds of the SDF, skip
    else
    {
      throw std::runtime_error("Point for BuildDistanceField out of bounds");
    }
  }
  // HERE BE DRAGONS
  // Process the bucket queue
  const std::vector<std::vector<std::vector<std::vector<int>>>> neighborhoods =
      MakeNeighborhoods();
  for (int32_t current_distance_square = 0;
       current_distance_square < (int32_t)bucket_queues.NumQueues();
       current_distance_square++)
  {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t idx = 0; idx < bucket_queues.Size(current_distance_square);
         idx++)
    {
      const VoxelGrid::GRID_INDEX& current_index =
          bucket_queues.Query(current_distance_square, idx);
      // Get the current location
      const bucket_cell& cur_cell =
          distance_field.GetImmutable(current_index).Value();
      const double x = cur_cell.location[0];
      const double y = cur_cell.location[1];
      const double z = cur_cell.location[2];
      // Pick the update direction
      // Only the first bucket queue gets the larger set of neighborhoods?
      // Don't really userstand why.
      const size_t direction_switch = (current_distance_square > 0) ? 1 : 0;
      // Make sure the update direction is valid
      if (cur_cell.update_direction < 0 || cur_cell.update_direction > 26)
      {
        continue;
      }
      // Get the current neighborhood list
      const std::vector<std::vector<int32_t>>& neighborhood =
          neighborhoods[direction_switch][cur_cell.update_direction];
      // Update the distance from the neighboring cells
      for (size_t nh_idx = 0; nh_idx < neighborhood.size(); nh_idx++)
      {
        // Get the direction to check
        const int32_t dx = neighborhood[nh_idx][0];
        const int32_t dy = neighborhood[nh_idx][1];
        const int32_t dz = neighborhood[nh_idx][2];
        const int32_t nx = (int32_t)(x + dx);
        const int32_t ny = (int32_t)(y + dy);
        const int32_t nz = (int32_t)(z + dz);
        const VoxelGrid::GRID_INDEX neighbor_index(
              (int64_t)nx, (int64_t)ny, (int64_t)nz);
        auto neighbor_query = distance_field.GetMutable(neighbor_index);
        if (!neighbor_query)
        {
          // "Neighbor" is outside the bounds of the SDF
          continue;
        }
        // Update the neighbor's distance based on the current
        const int32_t new_distance_square =
            (int32_t)ComputeDistanceSquared(nx, ny, nz,
                                            cur_cell.closest_point[0],
                                            cur_cell.closest_point[1],
                                            cur_cell.closest_point[2]);
        if (new_distance_square > max_distance_square)
        {
          // Skip these cases
          continue;
        }
        if (new_distance_square < neighbor_query.Value().distance_square)
        {
          // If the distance is better, time to update the neighbor
          neighbor_query.Value().distance_square = new_distance_square;
          neighbor_query.Value().closest_point[0] = cur_cell.closest_point[0];
          neighbor_query.Value().closest_point[1] = cur_cell.closest_point[1];
          neighbor_query.Value().closest_point[2] = cur_cell.closest_point[2];
          neighbor_query.Value().location[0] = nx;
          neighbor_query.Value().location[1] = ny;
          neighbor_query.Value().location[2] = nz;
          neighbor_query.Value().update_direction =
              GetDirectionNumber(dx, dy, dz);
          // Add the neighbor into the bucket queue
          bucket_queues.Enqueue(new_distance_square, neighbor_index);
        }
      }
    }
    // Clear the current queues now that we're done with it
    bucket_queues.ClearCompletedQueues(current_distance_square);
  }
  const std::chrono::time_point<std::chrono::steady_clock> end_time
      = std::chrono::steady_clock::now();
  const std::chrono::duration<double> elapsed = end_time - start_time;
  std::cout << "Computed DistanceField for grid size (" << grid_num_x_cells
            << ", " << grid_num_y_cells << ", " << grid_num_z_cells << ") in "
            << elapsed.count() << " seconds" << std::endl;
  return distance_field;
}

#define COMPARE_DISTANCE_FIELD_GENERATION

inline DistanceField BuildDistanceField(
    const Eigen::Isometry3d& grid_origin_transform,
    const double grid_resolution,
    const int64_t grid_num_x_cells,
    const int64_t grid_num_y_cells,
    const int64_t grid_num_z_cells,
    const std::vector<VoxelGrid::GRID_INDEX>& points)
{
#ifdef COMPARE_DISTANCE_FIELD_GENERATION
  const DistanceField legacy_field =
      BuildDistanceFieldSerial(grid_origin_transform, grid_resolution,
                               grid_num_x_cells, grid_num_y_cells,
                               grid_num_z_cells, points);
  const DistanceField new_field =
      BuildDistanceFieldParallel(grid_origin_transform, grid_resolution,
                                 grid_num_x_cells, grid_num_y_cells,
                                 grid_num_z_cells, points);
  for (int64_t x_index = 0; x_index < grid_num_x_cells; x_index++)
  {
    for (int64_t y_index = 0; y_index < grid_num_y_cells; y_index++)
    {
      for (int64_t z_index = 0; z_index < grid_num_z_cells; z_index++)
      {
        const bucket_cell& legacy_cell =
            legacy_field.GetImmutable(x_index, y_index, z_index).Value();
        const bucket_cell& new_cell =
            new_field.GetImmutable(x_index, y_index, z_index).Value();
        const double legacy_distance = legacy_cell.distance_square;
        const double new_distance = new_cell.distance_square;
        const double new_delta = std::abs(new_distance - legacy_distance);
        assert(new_delta < 1e-6);
      }
    }
  }
  return new_field;
#else
  // Should we use the parallelizable new variant or the serial legacy variant?
  if (points.size() > 1000)
  {
    return BuildDistanceFieldParallel(grid_origin_transform, grid_resolution,
                                      grid_num_x_cells, grid_num_y_cells,
                                      grid_num_z_cells, points);
  }
  else
  {
    return BuildDistanceFieldSerial(grid_origin_transform, grid_resolution,
                                    grid_num_x_cells, grid_num_y_cells,
                                    grid_num_z_cells, points);
  }
#endif
}

template<typename T>
inline std::pair<sdf_tools::SignedDistanceField, std::pair<double, double>>
ExtractSignedDistanceField(
    const Eigen::Isometry3d& grid_origin_tranform,
    const double grid_resolution,
    const int64_t grid_num_x_cells,
    const int64_t grid_num_y_cells,
    const int64_t grid_num_z_cells,
    const std::function<bool(const VoxelGrid::GRID_INDEX&)>& is_filled_fn,
    const float oob_value,
    const std::string& frame)
{
  const std::chrono::time_point<std::chrono::steady_clock> start_time
      = std::chrono::steady_clock::now();
  std::vector<VoxelGrid::GRID_INDEX> filled;
  std::vector<VoxelGrid::GRID_INDEX> free;
  for (int64_t x_index = 0; x_index < grid_num_x_cells; x_index++)
  {
    for (int64_t y_index = 0; y_index < grid_num_y_cells; y_index++)
    {
      for (int64_t z_index = 0; z_index < grid_num_z_cells; z_index++)
      {
        const VoxelGrid::GRID_INDEX current_index(x_index, y_index, z_index);
        if (is_filled_fn(current_index))
        {
          // Mark as filled
          filled.push_back(current_index);
        }
        else
        {
          // Mark as free space
          free.push_back(current_index);
        }
      }
    }
  }
  // Make two distance fields, one for distance to filled voxels, one for
  // distance to free voxels.
  const DistanceField filled_distance_field =
      BuildDistanceField(grid_origin_tranform, grid_resolution,
                         grid_num_x_cells, grid_num_y_cells, grid_num_z_cells,
                         filled);
  const DistanceField free_distance_field =
      BuildDistanceField(grid_origin_tranform, grid_resolution,
                         grid_num_x_cells, grid_num_y_cells, grid_num_z_cells,
                         free);
  // Generate the SDF
  sdf_tools::SignedDistanceField new_sdf(
        grid_origin_tranform, frame, grid_resolution, grid_num_x_cells,
        grid_num_y_cells, grid_num_z_cells, oob_value);
  double max_distance = -std::numeric_limits<double>::infinity();
  double min_distance = std::numeric_limits<double>::infinity();
  for (int64_t x_index = 0; x_index < new_sdf.GetNumXCells(); x_index++)
  {
    for (int64_t y_index = 0; y_index < new_sdf.GetNumYCells(); y_index++)
    {
      // Parallelize across the Z-axis, since VoxelGrid ensures Z-axis cells are
      // contiguous in memory.
#if defined(_OPENMP)
#pragma omp parallel for
#endif
      for (int64_t z_index = 0; z_index < new_sdf.GetNumZCells(); z_index++)
      {
        const double distance1 =
            std::sqrt(
              filled_distance_field.GetImmutable(x_index, y_index, z_index)
                .Value().distance_square)
            * new_sdf.GetResolution();
        const double distance2 =
            std::sqrt(
              free_distance_field.GetImmutable(x_index, y_index, z_index)
                .Value().distance_square)
            * new_sdf.GetResolution();
        const double distance = distance1 - distance2;
        if (distance > max_distance)
        {
          max_distance = distance;
        }
        if (distance < min_distance)
        {
          min_distance = distance;
        }
        new_sdf.SetValue(x_index, y_index, z_index, (float)distance);
      }
    }
  }
  const std::chrono::time_point<std::chrono::steady_clock> end_time
      = std::chrono::steady_clock::now();
  const std::chrono::duration<double> elapsed = end_time - start_time;
  std::cout << "Computed SDF for grid size (" << grid_num_x_cells << ", "
            << grid_num_y_cells << ", " << grid_num_z_cells << ") in "
            << elapsed.count() << " seconds" << std::endl;
  std::pair<double, double> extrema(max_distance, min_distance);
  return std::pair<sdf_tools::SignedDistanceField,
                   std::pair<double, double>>(new_sdf, extrema);
}

template<typename T, typename BackingStore=std::vector<T>>
inline std::pair<sdf_tools::SignedDistanceField, std::pair<double, double>>
ExtractSignedDistanceField(
    const VoxelGrid::VoxelGrid<T, BackingStore>& grid,
    const std::function<bool(const VoxelGrid::GRID_INDEX&)>& is_filled_fn,
    const float oob_value, const std::string& frame,
    const bool add_virtual_border)
{
  (void)(add_virtual_border);
  const Eigen::Vector3d cell_sizes = grid.GetCellSizes();
  if ((cell_sizes.x() != cell_sizes.y()) || (cell_sizes.x() != cell_sizes.z()))
  {
    throw std::invalid_argument("Grid must have uniform resolution");
  }
  if (add_virtual_border == false)
  {
    // This is the conventional single-pass result
    return ExtractSignedDistanceField<T>(
          grid.GetOriginTransform(), cell_sizes.x(), grid.GetNumXCells(),
          grid.GetNumYCells(), grid.GetNumZCells(), is_filled_fn, oob_value,
          frame);
  }
  else
  {
    const int64_t x_axis_size_offset =
        (grid.GetNumXCells() > 1) ? (int64_t)2 : (int64_t)0;
    const int64_t x_axis_query_offset =
        (grid.GetNumXCells() > 1) ? (int64_t)1 : (int64_t)0;
    const int64_t y_axis_size_offset =
        (grid.GetNumYCells() > 1) ? (int64_t)2 : (int64_t)0;
    const int64_t y_axis_query_offset =
        (grid.GetNumYCells() > 1) ? (int64_t)1 : (int64_t)0;
    const int64_t z_axis_size_offset =
        (grid.GetNumZCells() > 1) ? (int64_t)2 : (int64_t)0;
    const int64_t z_axis_query_offset =
        (grid.GetNumZCells() > 1) ? (int64_t)1 : (int64_t)0;
    // We need to lie about the size of the grid to add a virtual border
    const int64_t num_x_cells = grid.GetNumXCells() + x_axis_size_offset;
    const int64_t num_y_cells = grid.GetNumYCells() + y_axis_size_offset;
    const int64_t num_z_cells = grid.GetNumZCells() + z_axis_size_offset;
    // Make some deceitful helper functions that hide our lies about size
    // For the free space SDF, we lie and say the virtual border is filled
    const std::function<bool(const VoxelGrid::GRID_INDEX&)> free_is_filled_fn
        = [&] (const VoxelGrid::GRID_INDEX& virtual_border_grid_index)
    {
      // Is there a virtual border on our axis?
      if (x_axis_size_offset > 0)
      {
        // Are we a virtual border cell?
        if ((virtual_border_grid_index.x == 0)
            || (virtual_border_grid_index.x == (num_x_cells - 1)))
        {
          return true;
        }
      }
      // Is there a virtual border on our axis?
      if (y_axis_size_offset > 0)
      {
        // Are we a virtual border cell?
        if ((virtual_border_grid_index.y == 0)
            || (virtual_border_grid_index.y == (num_y_cells - 1)))
        {
          return true;
        }
      }
      // Is there a virtual border on our axis?
      if (z_axis_size_offset > 0)
      {
        // Are we a virtual border cell?
        if ((virtual_border_grid_index.z == 0)
            || (virtual_border_grid_index.z == (num_z_cells - 1)))
        {
          return true;
        }
      }
      const VoxelGrid::GRID_INDEX real_grid_index(
            virtual_border_grid_index.x - x_axis_query_offset,
            virtual_border_grid_index.y - y_axis_query_offset,
            virtual_border_grid_index.z - z_axis_query_offset);
      return is_filled_fn(real_grid_index);
    };
    // For the filled space SDF, we lie and say the virtual border is empty
    const std::function<bool(const VoxelGrid::GRID_INDEX&)> filled_is_filled_fn
        = [&] (const VoxelGrid::GRID_INDEX& virtual_border_grid_index)
    {
      // Is there a virtual border on our axis?
      if (x_axis_size_offset > 0)
      {
        // Are we a virtual border cell?
        if ((virtual_border_grid_index.x == 0)
            || (virtual_border_grid_index.x == (num_x_cells - 1)))
        {
          return false;
        }
      }
      // Is there a virtual border on our axis?
      if (y_axis_size_offset > 0)
      {
        // Are we a virtual border cell?
        if ((virtual_border_grid_index.y == 0)
            || (virtual_border_grid_index.y == (num_y_cells - 1)))
        {
          return false;
        }
      }
      // Is there a virtual border on our axis?
      if (z_axis_size_offset > 0)
      {
        // Are we a virtual border cell?
        if ((virtual_border_grid_index.z == 0)
            || (virtual_border_grid_index.z == (num_z_cells - 1)))
        {
          return false;
        }
      }
      const VoxelGrid::GRID_INDEX real_grid_index(
            virtual_border_grid_index.x - x_axis_query_offset,
            virtual_border_grid_index.y - y_axis_query_offset,
            virtual_border_grid_index.z - z_axis_query_offset);
      return is_filled_fn(real_grid_index);
    };
    // Make both SDFs
    auto free_sdf_result = ExtractSignedDistanceField<T>(
                             grid.GetOriginTransform(), cell_sizes.x(),
                             num_x_cells, num_y_cells, num_z_cells,
                             free_is_filled_fn, oob_value, frame);
    auto filled_sdf_result = ExtractSignedDistanceField<T>(
                               grid.GetOriginTransform(), cell_sizes.x(),
                               num_x_cells, num_y_cells, num_z_cells,
                               filled_is_filled_fn, oob_value, frame);
    // Combine to make a single SDF
    sdf_tools::SignedDistanceField combined_sdf(
          grid.GetOriginTransform(), frame, cell_sizes.x(), grid.GetNumXCells(),
          grid.GetNumYCells(), grid.GetNumZCells(), oob_value);
    for (int64_t x_idx = 0; x_idx < combined_sdf.GetNumXCells(); x_idx++)
    {
      for (int64_t y_idx = 0; y_idx < combined_sdf.GetNumYCells(); y_idx++)
      {
        // Parallelize across the Z-axis, since VoxelGrid ensures Z-axis cells
        // are contiguous in memory.
#if defined(_OPENMP)
#pragma omp parallel for
#endif
        for (int64_t z_idx = 0; z_idx < combined_sdf.GetNumZCells(); z_idx++)
        {
          const int64_t query_x_idx = x_idx + x_axis_query_offset;
          const int64_t query_y_idx = y_idx + y_axis_query_offset;
          const int64_t query_z_idx = z_idx + z_axis_query_offset;
          const float free_sdf_value
              = free_sdf_result.first.GetImmutable(
                  query_x_idx, query_y_idx, query_z_idx).Value();
          const float filled_sdf_value
              = filled_sdf_result.first.GetImmutable(
                  query_x_idx, query_y_idx, query_z_idx).Value();
          if (free_sdf_value >= 0.0)
          {
            combined_sdf.SetValue(x_idx, y_idx, z_idx, free_sdf_value);
          }
          else if (filled_sdf_value <= -0.0)
          {
            combined_sdf.SetValue(x_idx, y_idx, z_idx, filled_sdf_value);
          }
          else
          {
            combined_sdf.SetValue(x_idx, y_idx, z_idx, 0.0f);
          }
        }
      }
    }
    // Get the combined max/min values
    const std::pair<double, double> combined_extrema(
          free_sdf_result.second.first, filled_sdf_result.second.second);
    return std::make_pair(combined_sdf, combined_extrema);
  }
}

template<typename T, typename BackingStore=std::vector<T>>
inline std::pair<sdf_tools::SignedDistanceField, std::pair<double, double>>
ExtractSignedDistanceField(const VoxelGrid::VoxelGrid<T, BackingStore>& grid,
                           const std::function<bool(const T&)>& is_filled_fn,
                           const float oob_value, const std::string& frame)
{
  const std::function<bool(const VoxelGrid::GRID_INDEX&)> real_is_filled_fn =
      [&] (const VoxelGrid::GRID_INDEX& index)
  {
    const T& stored = grid.GetImmutable(index).Value();
    // If it matches an object to use OR there are no objects supplied
    if (is_filled_fn(stored))
    {
      // Mark as filled
      return true;
    }
    else
    {
      // Mark as free space
      return false;
    }
  };
  return ExtractSignedDistanceField(
        grid, real_is_filled_fn, oob_value, frame, false);
}
}

#endif // SDF_GENERATION_HPP
