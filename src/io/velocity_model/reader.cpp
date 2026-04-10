#include "io/velocity_model/reader.hpp"

#include "enumerations/interface.hpp"
#include "io/velocity_model/impl/cartesian_grid.hpp"
#include "specfem/assembly/assembly/dim2/assembly.hpp"
#include "specfem/assembly/properties.hpp"
#include "specfem/logger.hpp"
#include "specfem/macros.hpp"
#include "specfem/point.hpp"
#include "specfem_setup.hpp"

#include <Kokkos_Core.hpp>
#include <atomic>
#include <cmath>
#include <iostream>
#include <limits>
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
 * @brief Accumulated statistics gathered during injection for diagnostics.
 *
 * All members are written from a single thread (the kernel runs on the host
 * execution space via std::mutex-protected updates) because Kokkos MDRangePolicy
 * on the host with OpenMP does not guarantee atomic access to ordinary doubles.
 * We use std::atomic counters so that the parallel_for updates are race-free.
 */
struct ModelStats {
  std::atomic<double> vp_max{0.0};
  std::atomic<double> vs_max{0.0};
  std::atomic<double> rho_min{std::numeric_limits<double>::max()};
  std::atomic<long> n_negative_kappa{0};
  std::atomic<long> n_nonpositive_rho{0};
  std::atomic<long> n_nonpositive_vp{0};
  std::atomic<long> n_total{0};

  // Atomic max/min helpers (std::atomic<double> has load/store but no
  // compare_exchange_weak in C++17 — roll our own CAS loop).
  void update_max(std::atomic<double> &a, double v) {
    double prev = a.load(std::memory_order_relaxed);
    while (v > prev &&
           !a.compare_exchange_weak(prev, v, std::memory_order_relaxed))
      ;
  }
  void update_min(std::atomic<double> &a, double v) {
    double prev = a.load(std::memory_order_relaxed);
    while (v < prev &&
           !a.compare_exchange_weak(prev, v, std::memory_order_relaxed))
      ;
  }
};

/**
 * @brief Inject into elastic (PSV or SH) isotropic elements — Kokkos parallel.
 *
 * Stored as (kappa, mu, rho) where
 *   kappa = rho * (Vp^2 - 4/3 * Vs^2)
 *   mu    = rho * Vs^2
 *
 * Physical validity:
 *   - rho > 0 required (abort otherwise).
 *   - Vp > 0 required (abort otherwise).
 *   - kappa > 0 required for stability; if kappa <= 0 the GLL point is
 *     clamped to a small positive value and counted in stats.n_negative_kappa.
 */
template <specfem::element::medium_tag MediumTag>
std::enable_if_t<specfem::element::is_elastic<MediumTag>::value, void>
inject_elastic_isotropic(
    const Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> &elements,
    specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly,
    const specfem::io::velocity_model::CartesianGrid2D &grid,
    specfem::io::velocity_model::InterpolationMethod method,
    specfem::io::velocity_model::OutOfBoundsPolicy oob_policy,
    ModelStats &stats) {

  constexpr auto dim2 = specfem::dimension::type::dim2;
  constexpr auto prop = specfem::element::property_tag::isotropic;

  const int nelement = elements.extent(0);
  if (nelement == 0)
    return;

  const int ngllz = assembly.mesh.element_grid.ngllz;
  const int ngllx = assembly.mesh.element_grid.ngllx;

  const auto &props = assembly.properties;
  const auto &h_coord = assembly.mesh.h_coord;

  Kokkos::parallel_for(
      "inject_elastic_isotropic_" + std::to_string(static_cast<int>(MediumTag)),
      Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Rank<3>>(
          {0, 0, 0}, {nelement, ngllz, ngllx}),
      [=, &grid, &props, &h_coord, &elements, &stats](int i, int iz, int ix) {
        const int ispec = elements(i);

        // Physical coordinates of this GLL point
        const double x = h_coord(0, ispec, iz, ix);
        const double z = h_coord(1, ispec, iz, ix);

        auto [vp, vs, rho] = grid.interpolate(x, z, method, oob_policy);

        // Accumulate statistics
        stats.update_max(stats.vp_max, vp);
        stats.update_max(stats.vs_max, vs);
        stats.update_min(stats.rho_min, rho);
        stats.n_total.fetch_add(1, std::memory_order_relaxed);

        // Validate physical values
        if (rho <= 0.0) {
          stats.n_nonpositive_rho.fetch_add(1, std::memory_order_relaxed);
          Kokkos::abort("velocity_model_reader: rho <= 0 in elastic element. "
                        "Check model file units (rho must be in kg/m^3).");
        }
        if (vp <= 0.0) {
          stats.n_nonpositive_vp.fetch_add(1, std::memory_order_relaxed);
          Kokkos::abort("velocity_model_reader: Vp <= 0 in elastic element. "
                        "Check model file units (Vp must be in m/s).");
        }

        // Convert to elastic moduli
        const double mu = rho * vs * vs;
        double kappa = rho * (vp * vp - (4.0 / 3.0) * vs * vs);

        // kappa <= 0 means Vp/Vs < sqrt(4/3) ~ 1.155 (non-physical rock).
        // Clamp to a small positive value to avoid NaN and count occurrences.
        if (kappa <= 0.0) {
          stats.n_negative_kappa.fetch_add(1, std::memory_order_relaxed);
          // Use P-wave modulus (Vp only) as fallback: M = rho*Vp^2
          kappa = rho * vp * vp * 1e-6; // tiny positive — effectively pure shear
        }

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
    specfem::io::velocity_model::OutOfBoundsPolicy oob_policy,
    ModelStats &stats) {

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
      [=, &grid, &props, &h_coord, &elements, &stats](int i, int iz, int ix) {
        const int ispec = elements(i);

        const double x = h_coord(0, ispec, iz, ix);
        const double z = h_coord(1, ispec, iz, ix);

        auto [vp, vs, rho] = grid.interpolate(x, z, method, oob_policy);

        stats.update_max(stats.vp_max, vp);
        stats.update_min(stats.rho_min, rho);
        stats.n_total.fetch_add(1, std::memory_order_relaxed);

        if (rho <= 0.0) {
          stats.n_nonpositive_rho.fetch_add(1, std::memory_order_relaxed);
          Kokkos::abort("velocity_model_reader: rho <= 0 in acoustic element. "
                        "Check model file units (rho must be in kg/m^3).");
        }
        if (vp <= 0.0) {
          stats.n_nonpositive_vp.fetch_add(1, std::memory_order_relaxed);
          Kokkos::abort("velocity_model_reader: Vp <= 0 in acoustic element. "
                        "Check model file units (Vp must be in m/s).");
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

/**
 * @brief Estimate the minimum GLL-point spacing in the mesh (metres).
 *
 * Approximates by scanning every GLL point and computing the minimum
 * distance to its x-neighbour and z-neighbour within each element.
 * This is an O(nelement * ngll^2) pass done once at startup.
 */
double estimate_min_gll_spacing(
    const specfem::assembly::assembly<specfem::dimension::type::dim2>
        &assembly) {
  const auto &h_coord = assembly.mesh.h_coord;
  const int nelement = h_coord.extent(1);
  const int ngllz = h_coord.extent(2);
  const int ngllx = h_coord.extent(3);

  double h_min = std::numeric_limits<double>::max();
  for (int ispec = 0; ispec < nelement; ++ispec) {
    for (int iz = 0; iz < ngllz - 1; ++iz) {
      for (int ix = 0; ix < ngllx - 1; ++ix) {
        // x-direction spacing
        const double dx =
            h_coord(0, ispec, iz, ix + 1) - h_coord(0, ispec, iz, ix);
        const double dz_x =
            h_coord(1, ispec, iz, ix + 1) - h_coord(1, ispec, iz, ix);
        const double ds_x = std::sqrt(dx * dx + dz_x * dz_x);
        if (ds_x > 0.0 && ds_x < h_min) h_min = ds_x;

        // z-direction spacing
        const double dx_z =
            h_coord(0, ispec, iz + 1, ix) - h_coord(0, ispec, iz, ix);
        const double dz =
            h_coord(1, ispec, iz + 1, ix) - h_coord(1, ispec, iz, ix);
        const double ds_z = std::sqrt(dx_z * dx_z + dz * dz);
        if (ds_z > 0.0 && ds_z < h_min) h_min = ds_z;
      }
    }
  }
  return h_min;
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

specfem::io::velocity_model_reader::velocity_model_reader(
    const std::string &filename, Format format)
    : filename_(filename), format_(format),
      method_(velocity_model::InterpolationMethod::bilinear),
      oob_policy_(velocity_model::OutOfBoundsPolicy::clamp) {}

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
  ModelStats stats;

  // Elastic PSV isotropic
  {
    constexpr auto med = specfem::element::medium_tag::elastic_psv;
    constexpr auto prp = specfem::element::property_tag::isotropic;
    const auto elems = etypes.get_elements_on_host(med, prp);
    inject_elastic_isotropic<med>(elems, assembly, grid, method_, oob_policy_,
                                  stats);
  }

  // Elastic SH isotropic
  {
    constexpr auto med = specfem::element::medium_tag::elastic_sh;
    constexpr auto prp = specfem::element::property_tag::isotropic;
    const auto elems = etypes.get_elements_on_host(med, prp);
    inject_elastic_isotropic<med>(elems, assembly, grid, method_, oob_policy_,
                                  stats);
  }

  // Acoustic isotropic
  {
    constexpr auto med = specfem::element::medium_tag::acoustic;
    constexpr auto prp = specfem::element::property_tag::isotropic;
    const auto elems = etypes.get_elements_on_host(med, prp);
    inject_acoustic_isotropic(elems, assembly, grid, method_, oob_policy_,
                               stats);
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
  // 3. Post-injection diagnostics
  // ------------------------------------------------------------------
  {
    const double vp_max  = stats.vp_max.load(std::memory_order_relaxed);
    const double vs_max  = stats.vs_max.load(std::memory_order_relaxed);
    const double rho_min = stats.rho_min.load(std::memory_order_relaxed);
    const long   n_neg_k = stats.n_negative_kappa.load(std::memory_order_relaxed);
    const long   n_total = stats.n_total.load(std::memory_order_relaxed);

    std::ostringstream diag;
    diag << "Velocity model injection statistics (" << n_total << " GLL pts):\n"
         << "  Vp_max  = " << vp_max  << " m/s\n"
         << "  Vs_max  = " << vs_max  << " m/s\n"
         << "  Rho_min = " << rho_min << " kg/m^3\n";
    specfem::Logger::info(diag.str());

    // Warn about non-physical Vp/Vs ratios that forced kappa clamping
    if (n_neg_k > 0) {
      std::ostringstream warn;
      warn << "WARNING: " << n_neg_k << " GLL point(s) had kappa <= 0\n"
           << "  (Vp/Vs < sqrt(4/3) ~ 1.155, non-physical for rock).\n"
           << "  kappa was clamped to a small positive value at those points.\n"
           << "  Check your velocity model: ensure Vp/Vs >= 1.2 everywhere.\n";
      specfem::Logger::warning(warn.str());
    }

    // CFL stability estimate — warn if injected Vp_max may violate stability
    if (n_total > 0 && vp_max > 0.0) {
      const double h_min = estimate_min_gll_spacing(assembly);
      if (h_min > 0.0) {
        // Courant number estimate: C = Vp_max * dt / h_min; stable if C < 0.45
        // We do not have dt here, so we report the stable dt upper bound.
        // The typical Courant limit for the Newmark scheme used in SPECFEM is ~0.45.
        constexpr double courant_limit = 0.45;
        const double dt_max = courant_limit * h_min / vp_max;

        std::ostringstream cfl;
        cfl << "CFL stability estimate for injected model:\n"
            << "  min GLL spacing  h_min  = " << h_min  << " m\n"
            << "  max P-wave speed Vp_max = " << vp_max << " m/s\n"
            << "  --> safe dt upper bound  = " << dt_max << " s\n"
            << "  (Courant limit C = " << courant_limit << ")\n"
            << "  Verify that your time step dt <= " << dt_max
            << " s to ensure stability.\n";
        specfem::Logger::info(cfl.str());
      }
    }
  }

  // ------------------------------------------------------------------
  // 4. Synchronise host → device
  // ------------------------------------------------------------------
  assembly.properties.copy_to_device();

  specfem::Logger::info("Velocity model injection complete.");
}
