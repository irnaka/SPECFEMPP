#pragma once

#include "io/reader.hpp"
#include "io/velocity_model/impl/cartesian_grid.hpp"
#include "specfem/assembly.hpp"
#include <memory>
#include <string>

namespace specfem {
namespace io {

/**
 * @brief Reader that injects an external Cartesian velocity model into GLL
 * points.
 *
 * This reader reads a user-supplied velocity model file (ASCII or binary) that
 * defines Vp, Vs, and density on a Cartesian grid and interpolates those
 * values to every GLL quadrature point of the spectral-element mesh.  It
 * replaces the default element-uniform material assignment that SPECFEM++ would
 * otherwise perform from the mesh database.
 *
 * The reader is registered with the assembly through the @c property_reader
 * slot in the assembly constructor.  Passing a non-null @c velocity_model_reader
 * sets @c has_gll_model = true inside the properties container, which allocates
 * the storage without filling it; the actual values are then written by
 * @c read().
 *
 * ## Quick-start (YAML)
 * ```yaml
 * parameters:
 *   databases:
 *     mesh-database: "OUTPUT_FILES/database.bin"
 *     velocity-model:
 *       format: ascii       # "ascii" or "binary"
 *       file:   "MODEL/velocity.dat"
 *       interpolation: bilinear   # "bilinear" (default) or "nearest"
 *       out-of-bounds: clamp      # "clamp" (default) or "error"
 * ```
 *
 * ## Supported media types
 * The reader handles all medium/property combinations that SPECFEM++ supports
 * for 2-D simulations:
 * | Medium       | Stored properties | Formula                                 |
 * |--------------|-------------------|-----------------------------------------|
 * | elastic PSV  | kappa, mu, rho    | kappa=rho*(Vp²−4/3*Vs²), mu=rho*Vs²    |
 * | elastic SH   | kappa, mu, rho    | (same as PSV)                           |
 * | acoustic     | rho_inverse, kappa| kappa=rho*Vp², rho_inverse=1/rho        |
 * | poroelastic  | (not overridden)  | warning printed; mesh materials retained|
 *
 * Other medium types (ELASTIC_PSV_T, anisotropic, Cosserat) that have zero
 * elements in the current simulation are silently skipped.
 *
 * @see specfem::io::velocity_model::CartesianGrid2D   Underlying grid type.
 * @see specfem::runtime_configuration::velocity_model YAML parser.
 */
class velocity_model_reader : public reader {
public:
  /**
   * @brief File format of the external velocity model.
   */
  enum class Format { ascii, binary };

  /**
   * @brief Construct a velocity model reader.
   *
   * @param filename     Path to the velocity model file.
   * @param format       File format (ascii or binary).
   * @param method       Interpolation method (bilinear or nearest).
   * @param oob_policy   Out-of-bounds handling (clamp or error).
   */
  velocity_model_reader(
      const std::string &filename, Format format,
      velocity_model::InterpolationMethod method =
          velocity_model::InterpolationMethod::bilinear,
      velocity_model::OutOfBoundsPolicy oob_policy =
          velocity_model::OutOfBoundsPolicy::clamp);

  /**
   * @brief Inject the velocity model into all GLL points of the 2-D assembly.
   *
   * The method:
   * 1. Loads the Cartesian grid from disk.
   * 2. Iterates over every spectral element, and for every GLL point reads
   *    the physical (x, z) coordinates from @c assembly.mesh.h_coord.
   * 3. Interpolates (Vp, Vs, Rho) at that location.
   * 4. Converts to the native storage format (kappa, mu, rho for elastic;
   *    rho_inverse, kappa for acoustic).
   * 5. Stores the result on host via @c specfem::assembly::store_on_host.
   * 6. Copies all properties to device memory.
   *
   * @param assembly 2-D SPECFEM++ assembly.  Must have been constructed with
   *                 @c has_gll_model = true (i.e. a non-null @c velocity_model_reader
   *                 was passed to the assembly constructor).
   */
  void read(specfem::assembly::assembly<specfem::dimension::type::dim2>
                &assembly) override;

  // Note: the base specfem::io::reader only declares a dim2 read() method.
  // 3-D injection will be added when the base class is extended.

private:
  std::string filename_;
  Format format_;
  velocity_model::InterpolationMethod method_;
  velocity_model::OutOfBoundsPolicy oob_policy_;
};

} // namespace io
} // namespace specfem
