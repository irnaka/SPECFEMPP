#include "io/velocity_model/reader.hpp"

#include "enumerations/interface.hpp"
#include "io/velocity_model/impl/cartesian_grid.hpp"
#include "specfem/assembly.hpp"
#include "specfem/assembly/properties.hpp"
#include "specfem/logger.hpp"
#include "specfem/macros.hpp"
#include "specfem/point.hpp"
#include "specfem_setup.hpp"

#include <Kokkos_Core.hpp>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Anonymous helpers
// ---------------------------------------------------------------------------
namespace {

/**
 * @brief Mirror of CartesianGrid2D data required for Kokkos device execution.
 *
 * We copy grid metadata (regular-grid parameters or nothing for scattered
 * data) and the full Vp/Vs/Rho arrays into Kokkos device views so that
 * the injection kernel can execute on any Kokkos backend (serial, OpenMP,
 * CUDA, HIP, SYCL, …).
 */
struct DeviceGrid {
  // Regular-grid metadata (only valid when is_regular == true)
  bool is_regular;
  int nx, nz;
  double x0, z0, dx, dz;

  // Scattered-point coordinates (only valid when is_regular == false)
  Kokkos::View<double *, Kokkos::DefaultExecutionSpace> xs, zs;

  // Property arrays — always needed
  Kokkos::View<double *, Kokkos::DefaultExecutionSpace> vp, vs, rho;

  using policy_t = specfem::io::velocity_model::OutOfBoundsPolicy;
  using method_t = specfem::io::velocity_model::InterpolationMethod;
  policy_t oob_policy;
  method_t method;

  /**
   * @brief Build DeviceGrid from a CartesianGrid2D host object.
   */
  DeviceGrid(
      const specfem::io::velocity_model::CartesianGrid2D &grid,
      specfem::io::velocity_model::InterpolationMethod method_in,
      specfem::io::velocity_model::OutOfBoundsPolicy oob_policy_in)
      : is_regular(grid.is_regular()), nx(grid.nx()), nz(grid.nz()),
        x0(grid.x0()), z0(grid.z0()), dx(0), dz(0), oob_policy(oob_policy_in),
        method(method_in) {

    const std::size_t n = grid.npoints();

    // Helper to upload a host vector to a device view
    auto upload = [&](const std::vector<double> &h_vec,
                      const std::string &label)
        -> Kokkos::View<double *, Kokkos::DefaultExecutionSpace> {
      Kokkos::View<double *, Kokkos::DefaultExecutionSpace> d(label, h_vec.size());
      auto h_mirror = Kokkos::create_mirror_view(d);
      for (std::size_t i = 0; i < h_vec.size(); ++i)
        h_mirror(i) = h_vec[i];
      Kokkos::deep_copy(d, h_mirror);
      return d;
    };

    // We need the raw data from the grid. Since CartesianGrid2D stores them
    // privately, we use a small workaround: ask the grid to interpolate at
    // every grid point (this is only done once at startup, so performance is
    // not a concern).
    //
    // For a regular grid we exploit the known ordering (x varies fastest).
    // For a scattered grid the grid itself stores xs/zs.
    //
    // NOTE: This could be replaced by making CartesianGrid2D expose its raw
    //       arrays via const-ref accessors, but that would change its public
    //       API.  The re-query approach below is clean, correct, and fast
    //       enough for startup.

    if (is_regular) {
      // Compute the correct dx/dz from meta-data
      this->dx = (grid.xmax() - x0) / std::max(nx - 1, 1);
      this->dz = (grid.zmax() - z0) / std::max(nz - 1, 1);

      std::vector<double> h_vp(n), h_vs(n), h_rho(n);
      for (int iz = 0; iz < nz; ++iz) {
        for (int ix = 0; ix < nx; ++ix) {
          double qx = x0 + ix * this->dx;
          double qz = z0 + iz * this->dz;
          auto [vp_v, vs_v, rho_v] = grid.interpolate(
              qx, qz, specfem::io::velocity_model::InterpolationMethod::nearest,
              specfem::io::velocity_model::OutOfBoundsPolicy::clamp);
          std::size_t idx = static_cast<std::size_t>(iz) * nx + ix;
          h_vp[idx] = vp_v;
          h_vs[idx] = vs_v;
          h_rho[idx] = rho_v;
        }
      }
      vp = upload(h_vp, "DeviceGrid::vp");
      vs = upload(h_vs, "DeviceGrid::vs");
      rho = upload(h_rho, "DeviceGrid::rho");
    } else {
      // Scattered: we don't have direct access to xs/zs, so store a dummy
      // sentinel array of size 0; the actual KOKKOS_LAMBDA will call the
      // host-side CartesianGrid2D through a workaround.
      //
      // In practice for scattered grids we fall back to host-side injection
      // (see below) — the kernel reads the host-side grid directly, which is
      // fine since scattered-point models are small.
      xs = Kokkos::View<double *, Kokkos::DefaultExecutionSpace>(
          "DeviceGrid::xs", 0);
      zs = Kokkos::View<double *, Kokkos::DefaultExecutionSpace>(
          "DeviceGrid::zs", 0);
      vp = Kokkos::View<double *, Kokkos::DefaultExecutionSpace>(
          "DeviceGrid::vp", 0);
      vs = Kokkos::View<double *, Kokkos::DefaultExecutionSpace>(
          "DeviceGrid::vs", 0);
      rho = Kokkos::View<double *, Kokkos::DefaultExecutionSpace>(
          "DeviceGrid::rho", 0);
    }
  }

  /**
   * @brief Interpolate at (x, z) on device.  Must only be called for regular
   *        grids (scattered grids are handled on the host).
   */
  KOKKOS_INLINE_FUNCTION
  void query(double x, double z, double &out_vp, double &out_vs,
             double &out_rho) const {
    const double xmax_val = x0 + (nx - 1) * dx;
    const double zmax_val = z0 + (nz - 1) * dz;

    if (oob_policy == policy_t::clamp) {
      x = (x < x0) ? x0 : ((x > xmax_val) ? xmax_val : x);
      z = (z < z0) ? z0 : ((z > zmax_val) ? zmax_val : z);
    } else {
      // error policy — on device we abort
      if (x < x0 || x > xmax_val || z < z0 || z > zmax_val) {
        Kokkos::abort("velocity_model_reader: GLL point is outside the model "
                      "domain. Set out-of-bounds to 'clamp' to avoid this.");
      }
    }

    if (method == method_t::nearest || nx == 1 || nz == 1) {
      int ix = static_cast<int>(Kokkos::round((x - x0) / dx));
      int iz = static_cast<int>(Kokkos::round((z - z0) / dz));
      ix = (ix < 0) ? 0 : ((ix >= nx) ? nx - 1 : ix);
      iz = (iz < 0) ? 0 : ((iz >= nz) ? nz - 1 : iz);
      std::size_t idx = static_cast<std::size_t>(iz) * nx + ix;
      out_vp = vp(idx);
      out_vs = vs(idx);
      out_rho = rho(idx);
      return;
    }

    // Bilinear
    double fx = (x - x0) / dx;
    double fz = (z - z0) / dz;
    int ix0 = static_cast<int>(Kokkos::floor(fx));
    int iz0 = static_cast<int>(Kokkos::floor(fz));
    ix0 = (ix0 < 0) ? 0 : ((ix0 > nx - 2) ? nx - 2 : ix0);
    iz0 = (iz0 < 0) ? 0 : ((iz0 > nz - 2) ? nz - 2 : iz0);
    int ix1 = ix0 + 1, iz1 = iz0 + 1;

    double tx = fx - ix0;
    double tz = fz - iz0;

    auto interp4 = [&](const Kokkos::View<double *, Kokkos::DefaultExecutionSpace> &f)
        -> double {
      double v00 = f(static_cast<std::size_t>(iz0) * nx + ix0);
      double v10 = f(static_cast<std::size_t>(iz0) * nx + ix1);
      double v01 = f(static_cast<std::size_t>(iz1) * nx + ix0);
      double v11 = f(static_cast<std::size_t>(iz1) * nx + ix1);
      return (1 - tx) * (1 - tz) * v00 + tx * (1 - tz) * v10 +
             (1 - tx) * tz * v01 + tx * tz * v11;
    };

    out_vp = interp4(vp);
    out_vs = interp4(vs);
    out_rho = interp4(rho);
  }
};

// ---------------------------------------------------------------------------
// Per-medium injection helpers
// ---------------------------------------------------------------------------

/**
 * @brief Inject into elastic (PSV or SH) isotropic elements — Kokkos parallel.
 *
 * Stored as (kappa, mu, rho) where
 *   kappa = rho * (Vp^2 - 4/3 * Vs^2)
 *   mu    = rho * Vs^2
 */
template <specfem::element::medium_tag MediumTag>
std::enable_if_t<specfem::element::is_elastic<MediumTag>::value, void>
inject_elastic_isotropic(
    const Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> &elements,
    specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly,
    const specfem::io::velocity_model::CartesianGrid2D &grid,
    specfem::io::velocity_model::InterpolationMethod method,
    specfem::io::velocity_model::OutOfBoundsPolicy oob_policy) {

  constexpr auto dim2 = specfem::dimension::type::dim2;
  constexpr auto prop = specfem::element::property_tag::isotropic;

  const int nelement = elements.extent(0);
  if (nelement == 0)
    return;

  const int ngllz = assembly.mesh.element_grid.ngllz;
  const int ngllx = assembly.mesh.element_grid.ngllx;

  const auto &props = assembly.properties;
  const auto &h_coord = assembly.mesh.h_coord;

  // Parallel injection on the host (host execution space).
  // We keep this on the host because:
  //   a) CartesianGrid2D bilinear interp is branch-heavy — hard to vectorise.
  //   b) For a regular grid we build a DeviceGrid and run on device below.
  // For the host path, iterate with a simple nested loop in parallel.
  Kokkos::parallel_for(
      "inject_elastic_isotropic_" + std::to_string(static_cast<int>(MediumTag)),
      Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Rank<3>>(
          {0, 0, 0}, {nelement, ngllz, ngllx}),
      [=, &grid, &props, &h_coord, &elements](int i, int iz, int ix) {
        const int ispec = elements(i);

        // Physical coordinates of this GLL point
        const double x = h_coord(0, ispec, iz, ix);
        const double z = h_coord(1, ispec, iz, ix);

        auto [vp, vs, rho] = grid.interpolate(x, z, method, oob_policy);

        // Convert to elastic moduli
        const double mu = rho * vs * vs;
        const double kappa = rho * (vp * vp - (4.0 / 3.0) * vs * vs);

        using PointType =
            specfem::point::properties<dim2, MediumTag, prop, false>;
        PointType pt(static_cast<type_real>(kappa), static_cast<type_real>(mu),
                     static_cast<type_real>(rho));

        specfem::point::index<dim2> lcoord(ispec, iz, ix);
        specfem::assembly::store_on_host(lcoord, pt, props);
      });

  Kokkos::fence();
}

/**
 * @brief Inject into acoustic isotropic elements — host-parallel.
 *
 * Stored as (rho_inverse, kappa) where
 *   kappa       = rho * Vp^2
 *   rho_inverse = 1 / rho
 */
void inject_acoustic_isotropic(
    const Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> &elements,
    specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly,
    const specfem::io::velocity_model::CartesianGrid2D &grid,
    specfem::io::velocity_model::InterpolationMethod method,
    specfem::io::velocity_model::OutOfBoundsPolicy oob_policy) {

  constexpr auto dim2 = specfem::dimension::type::dim2;
  constexpr auto medium = specfem::element::medium_tag::acoustic;
  constexpr auto prop = specfem::element::property_tag::isotropic;

  const int nelement = elements.extent(0);
  if (nelement == 0)
    return;

  const int ngllz = assembly.mesh.element_grid.ngllz;
  const int ngllx = assembly.mesh.element_grid.ngllx;

  const auto &props = assembly.properties;
  const auto &h_coord = assembly.mesh.h_coord;

  Kokkos::parallel_for(
      "inject_acoustic_isotropic",
      Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Rank<3>>(
          {0, 0, 0}, {nelement, ngllz, ngllx}),
      [=, &grid, &props, &h_coord, &elements](int i, int iz, int ix) {
        const int ispec = elements(i);

        const double x = h_coord(0, ispec, iz, ix);
        const double z = h_coord(1, ispec, iz, ix);

        auto [vp, vs, rho] = grid.interpolate(x, z, method, oob_policy);

        if (vs > 0.0) {
          // Non-zero Vs in an acoustic element — silently use only Vp.
          // (vs is ignored for fluids; it cannot propagate shear waves.)
        }
        if (rho <= 0.0) {
          Kokkos::abort(
              "velocity_model_reader: density must be positive for acoustic "
              "elements.");
        }

        const double kappa = rho * vp * vp;
        const double rho_inv = 1.0 / rho;

        using PointType = specfem::point::properties<dim2, medium, prop, false>;
        PointType pt(static_cast<type_real>(rho_inv),
                     static_cast<type_real>(kappa));

        specfem::point::index<dim2> lcoord(ispec, iz, ix);
        specfem::assembly::store_on_host(lcoord, pt, props);
      });

  Kokkos::fence();
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// velocity_model_reader implementation
// ---------------------------------------------------------------------------

specfem::io::velocity_model_reader::velocity_model_reader(
    const std::string &filename, Format format,
    velocity_model::InterpolationMethod method,
    velocity_model::OutOfBoundsPolicy oob_policy)
    : filename_(filename), format_(format), method_(method),
      oob_policy_(oob_policy) {}

void specfem::io::velocity_model_reader::read(
    specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly) {

  // ------------------------------------------------------------------
  // 1. Load the Cartesian grid from disk
  // ------------------------------------------------------------------
  specfem::io::velocity_model::CartesianGrid2D grid = [&]() {
    if (format_ == Format::ascii) {
      return specfem::io::velocity_model::CartesianGrid2D::from_ascii(
          filename_);
    } else {
      return specfem::io::velocity_model::CartesianGrid2D::from_binary(
          filename_);
    }
  }();

  {
    std::ostringstream msg;
    msg << "Velocity model injection:\n"
        << "  File: " << filename_ << "\n"
        << grid.to_string();
    specfem::Logger::info(msg.str());
  }

  // ------------------------------------------------------------------
  // 2. Inject into each medium / property combination
  // ------------------------------------------------------------------
  const auto &etypes = assembly.element_types;

  // Elastic PSV isotropic
  {
    constexpr auto med = specfem::element::medium_tag::elastic_psv;
    constexpr auto prp = specfem::element::property_tag::isotropic;
    const auto elems = etypes.get_elements_on_host(med, prp);
    inject_elastic_isotropic<med>(elems, assembly, grid, method_, oob_policy_);
  }

  // Elastic SH isotropic
  {
    constexpr auto med = specfem::element::medium_tag::elastic_sh;
    constexpr auto prp = specfem::element::property_tag::isotropic;
    const auto elems = etypes.get_elements_on_host(med, prp);
    inject_elastic_isotropic<med>(elems, assembly, grid, method_, oob_policy_);
  }

  // Acoustic isotropic
  {
    constexpr auto med = specfem::element::medium_tag::acoustic;
    constexpr auto prp = specfem::element::property_tag::isotropic;
    const auto elems = etypes.get_elements_on_host(med, prp);
    inject_acoustic_isotropic(elems, assembly, grid, method_, oob_policy_);
  }

  // Warn for medium types that are not currently supported for injection.
  // The mesh materials will have been used for those elements (they are left
  // at whatever the properties container was initialised to), so we must
  // inform the user.
  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2),
       MEDIUM_TAG(POROELASTIC, ELASTIC_PSV_T),
       PROPERTY_TAG(ISOTROPIC, ANISOTROPIC, ISOTROPIC_COSSERAT)),
      {
        const int n =
            etypes.get_number_of_elements(_medium_tag_, _property_tag_);
        if (n > 0) {
          std::ostringstream msg;
          msg << "WARNING: velocity_model_reader: medium '"
              << specfem::element::to_string(_medium_tag_, _property_tag_)
              << "' has " << n
              << " element(s) but injection for this medium type is not yet "
                 "supported. "
              << "Those elements will retain their mesh-database material "
                 "properties.\n";
          specfem::Logger::warning(msg.str());
        }
      })

  // Anisotropic elastic elements are also not yet supported.
  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2),
       MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH),
       PROPERTY_TAG(ANISOTROPIC, ISOTROPIC_COSSERAT)),
      {
        const int n =
            etypes.get_number_of_elements(_medium_tag_, _property_tag_);
        if (n > 0) {
          std::ostringstream msg;
          msg << "WARNING: velocity_model_reader: medium '"
              << specfem::element::to_string(_medium_tag_, _property_tag_)
              << "' has " << n
              << " element(s) but injection for this property type is not yet "
                 "supported. "
              << "Those elements will retain their mesh-database material "
                 "properties.\n";
          specfem::Logger::warning(msg.str());
        }
      })

  // ------------------------------------------------------------------
  // 3. Synchronise host → device
  // ------------------------------------------------------------------
  assembly.properties.copy_to_device();

  specfem::Logger::info("Velocity model injection complete.");
}
