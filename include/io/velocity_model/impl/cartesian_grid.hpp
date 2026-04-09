#pragma once

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace specfem {
namespace io {
namespace velocity_model {

/**
 * @brief Interpolation strategy for the Cartesian velocity grid.
 */
enum class InterpolationMethod {
  bilinear, ///< Bilinear interpolation on regular grids (recommended)
  nearest   ///< Nearest-neighbour lookup (fallback for scattered data)
};

/**
 * @brief Out-of-bounds handling strategy.
 */
enum class OutOfBoundsPolicy {
  clamp, ///< Clamp query point to grid boundary (default)
  error  ///< Throw std::runtime_error when query falls outside the grid
};

/**
 * @brief In-memory representation of an external Cartesian velocity model.
 *
 * Stores (Vp, Vs, Rho) on a regular 2-D Cartesian grid and provides
 * bilinear/nearest-neighbour interpolation at arbitrary (x, z) locations.
 *
 * ## ASCII file formats
 *
 * ### Regular grid (recommended)
 * ```
 * # REGULAR_GRID
 * # NX NZ X0 Z0 DX DZ
 * 100 50 0.0 0.0 100.0 100.0
 * # Vp   Vs    Rho
 * 3000.0 1700.0 2700.0
 * 3100.0 1750.0 2750.0
 * ...
 * ```
 * Data is stored row-major with **x varying fastest** (C order): the first NX
 * entries correspond to z = z0, the next NX entries to z = z0+dz, etc.
 *
 * ### Scattered points
 * ```
 * # x    z     Vp      Vs     Rho
 * 0.0   0.0   3000.0  1700.0  2700.0
 * 100.0 0.0   3100.0  1750.0  2750.0
 * ...
 * ```
 * Nearest-neighbour interpolation is used automatically.
 *
 * ## Binary file format
 * ```
 * [4 bytes]  nx         (int32, little-endian)
 * [4 bytes]  nz         (int32, little-endian)
 * [8 bytes]  x0         (float64, little-endian)
 * [8 bytes]  z0         (float64, little-endian)
 * [8 bytes]  dx         (float64, little-endian)
 * [8 bytes]  dz         (float64, little-endian)
 * [nx*nz*8]  Vp values  (float64, row-major, x varies fastest)
 * [nx*nz*8]  Vs values  (float64, row-major)
 * [nx*nz*8]  Rho values (float64, row-major)
 * ```
 */
class CartesianGrid2D {
public:
  /**
   * @brief Construct a regular-grid model from raw arrays.
   *
   * @param nx   Number of grid points in x direction
   * @param nz   Number of grid points in z direction
   * @param x0   Minimum x coordinate
   * @param z0   Minimum z coordinate
   * @param dx   Grid spacing in x
   * @param dz   Grid spacing in z
   * @param vp   Vp values [nz*nx], x varies fastest
   * @param vs   Vs values [nz*nx]
   * @param rho  Density values [nz*nx]
   */
  CartesianGrid2D(int nx, int nz, double x0, double z0, double dx, double dz,
                  std::vector<double> vp, std::vector<double> vs,
                  std::vector<double> rho)
      : nx_(nx), nz_(nz), x0_(x0), z0_(z0), dx_(dx), dz_(dz),
        vp_(std::move(vp)), vs_(std::move(vs)), rho_(std::move(rho)),
        is_regular_(true) {

    if (nx_ <= 0 || nz_ <= 0)
      throw std::runtime_error("CartesianGrid2D: nx and nz must be positive.");
    if (dx_ <= 0.0 || dz_ <= 0.0)
      throw std::runtime_error("CartesianGrid2D: dx and dz must be positive.");
    const std::size_t expected = static_cast<std::size_t>(nx_) * nz_;
    if (vp_.size() != expected || vs_.size() != expected ||
        rho_.size() != expected)
      throw std::runtime_error(
          "CartesianGrid2D: vp/vs/rho arrays must have nx*nz elements.");
  }

  /**
   * @brief Construct a scattered-point model.
   *
   * @param xs   x coordinates of each data point
   * @param zs   z coordinates of each data point
   * @param vp   Vp at each data point
   * @param vs   Vs at each data point
   * @param rho  Density at each data point
   */
  CartesianGrid2D(std::vector<double> xs, std::vector<double> zs,
                  std::vector<double> vp, std::vector<double> vs,
                  std::vector<double> rho)
      : nx_(0), nz_(0), x0_(0), z0_(0), dx_(0), dz_(0), xs_(std::move(xs)),
        zs_(std::move(zs)), vp_(std::move(vp)), vs_(std::move(vs)),
        rho_(std::move(rho)), is_regular_(false) {

    if (xs_.size() != zs_.size() || xs_.size() != vp_.size() ||
        xs_.size() != vs_.size() || xs_.size() != rho_.size())
      throw std::runtime_error(
          "CartesianGrid2D: all scattered-point arrays must have the same "
          "length.");
    if (xs_.empty())
      throw std::runtime_error(
          "CartesianGrid2D: scattered-point model is empty.");
  }

  // -------------------------------------------------------------------------
  // Factory methods (ASCII / binary)
  // -------------------------------------------------------------------------

  /**
   * @brief Load a velocity model from an ASCII file.
   *
   * The file is auto-detected as regular-grid or scattered-point format by
   * searching for the keyword "REGULAR_GRID" on any line (comment or bare)
   * before the first data line.  Data lines are non-empty, non-comment lines
   * that do not contain "REGULAR_GRID".
   *
   * **Regular-grid format:**
   * ```
   * # REGULAR_GRID
   * # NX NZ X0 Z0 DX DZ
   * 51 51 0.0 0.0 100.0 100.0
   * # Vp Vs Rho (one entry per grid point, x varies fastest)
   * 3000.0 1500.0 2500.0
   * ...
   * ```
   *
   * **Scattered-point format** (auto-detected when REGULAR_GRID is absent):
   * ```
   * # x z Vp Vs Rho
   * 0.0 0.0 3000.0 1500.0 2500.0
   * ...
   * ```
   *
   * @param filename Path to the ASCII model file.
   */
  static CartesianGrid2D from_ascii(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open())
      throw std::runtime_error("CartesianGrid2D: cannot open file: " +
                               filename);

    // Scan header lines (comments + optional REGULAR_GRID keyword) to decide
    // format.  We stop at the first *data* line — a non-empty, non-comment
    // line that does not contain "REGULAR_GRID".
    bool is_regular = false;
    std::string line;
    while (std::getline(file, line)) {
      const std::string t = trim(line);
      if (t.empty())
        continue; // blank line → keep scanning
      if (t.find("REGULAR_GRID") != std::string::npos) {
        is_regular = true; // keyword found (may be inside a comment)
        continue;          // keep scanning for the header integers
      }
      if (t[0] == '#')
        continue; // regular comment, keep scanning
      // First real data line reached — stop
      break;
    }

    file.seekg(0); // rewind for the actual loader
    if (is_regular)
      return load_regular_ascii(file);
    else
      return load_scattered_ascii(file);
  }

  /**
   * @brief Load a velocity model from a binary file.
   *
   * Format: [int32 nx][int32 nz][double x0][double z0][double dx][double dz]
   *         [nx*nz doubles Vp][nx*nz doubles Vs][nx*nz doubles Rho]
   *
   * @param filename Path to the binary model file.
   */
  static CartesianGrid2D from_binary(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
      throw std::runtime_error("CartesianGrid2D: cannot open file: " +
                               filename);

    int32_t nx, nz;
    double x0, z0, dx, dz;
    file.read(reinterpret_cast<char *>(&nx), sizeof(int32_t));
    file.read(reinterpret_cast<char *>(&nz), sizeof(int32_t));
    file.read(reinterpret_cast<char *>(&x0), sizeof(double));
    file.read(reinterpret_cast<char *>(&z0), sizeof(double));
    file.read(reinterpret_cast<char *>(&dx), sizeof(double));
    file.read(reinterpret_cast<char *>(&dz), sizeof(double));

    if (!file)
      throw std::runtime_error("CartesianGrid2D: failed to read binary header "
                               "from file: " +
                               filename);

    const std::size_t n = static_cast<std::size_t>(nx) * nz;
    std::vector<double> vp(n), vs(n), rho(n);
    file.read(reinterpret_cast<char *>(vp.data()), n * sizeof(double));
    file.read(reinterpret_cast<char *>(vs.data()), n * sizeof(double));
    file.read(reinterpret_cast<char *>(rho.data()), n * sizeof(double));

    if (!file)
      throw std::runtime_error(
          "CartesianGrid2D: failed to read binary data from file: " + filename);

    return CartesianGrid2D(nx, nz, x0, z0, dx, dz, std::move(vp),
                           std::move(vs), std::move(rho));
  }

  // -------------------------------------------------------------------------
  // Interpolation
  // -------------------------------------------------------------------------

  /**
   * @brief Interpolate (Vp, Vs, Rho) at a query point (x, z).
   *
   * For regular grids, bilinear interpolation is used.
   * For scattered data, nearest-neighbour lookup is used.
   *
   * @param x       Query x coordinate
   * @param z       Query z coordinate
   * @param method  Interpolation method (bilinear or nearest)
   * @param policy  Out-of-bounds policy (clamp or error)
   * @return Tuple of (Vp, Vs, Rho) at (x, z)
   */
  std::tuple<double, double, double>
  interpolate(double x, double z,
              InterpolationMethod method = InterpolationMethod::bilinear,
              OutOfBoundsPolicy policy = OutOfBoundsPolicy::clamp) const {
    if (is_regular_) {
      return interpolate_regular(x, z, method, policy);
    } else {
      return interpolate_nearest(x, z, policy);
    }
  }

  // -------------------------------------------------------------------------
  // Metadata accessors
  // -------------------------------------------------------------------------
  bool is_regular() const { return is_regular_; }
  int nx() const { return nx_; }
  int nz() const { return nz_; }
  std::size_t npoints() const {
    return is_regular_ ? static_cast<std::size_t>(nx_) * nz_ : xs_.size();
  }
  double x0() const { return x0_; }
  double z0() const { return z0_; }
  double xmax() const { return x0_ + (nx_ - 1) * dx_; }
  double zmax() const { return z0_ + (nz_ - 1) * dz_; }

  /**
   * @brief Print a summary of the grid to a string.
   */
  std::string to_string() const {
    std::ostringstream ss;
    if (is_regular_) {
      ss << "CartesianGrid2D [regular grid]\n"
         << "  nx=" << nx_ << "  nz=" << nz_ << "\n"
         << "  x: [" << x0_ << ", " << xmax() << "]  dx=" << dx_ << "\n"
         << "  z: [" << z0_ << ", " << zmax() << "]  dz=" << dz_ << "\n"
         << "  total points: " << npoints() << "\n";
    } else {
      ss << "CartesianGrid2D [scattered points]\n"
         << "  total points: " << xs_.size() << "\n";
    }
    return ss.str();
  }

private:
  // Regular grid
  int nx_, nz_;
  double x0_, z0_, dx_, dz_;

  // Shared data arrays (used by both regular and scattered modes)
  std::vector<double> vp_, vs_, rho_;

  // Scattered-point coordinates (only used when !is_regular_)
  std::vector<double> xs_, zs_;

  bool is_regular_;

  // -------------------------------------------------------------------------
  // Internal helpers
  // -------------------------------------------------------------------------

  static std::string trim(const std::string &s) {
    const auto begin = s.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos)
      return "";
    const auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(begin, end - begin + 1);
  }

  /** Load a regular-grid ASCII file. */
  static CartesianGrid2D load_regular_ascii(std::ifstream &file) {
    int nx = 0, nz = 0;
    double x0 = 0, z0 = 0, dx = 0, dz = 0;
    bool header_found = false;

    std::string line;
    while (std::getline(file, line)) {
      std::string t = trim(line);
      if (t.empty() || t[0] == '#')
        continue;
      if (t.find("REGULAR_GRID") != std::string::npos)
        continue;
      // Expect: NX NZ X0 Z0 DX DZ
      std::istringstream iss(t);
      iss >> nx >> nz >> x0 >> z0 >> dx >> dz;
      if (iss.fail())
        throw std::runtime_error(
            "CartesianGrid2D: failed to parse regular-grid header line: " + t);
      header_found = true;
      break;
    }

    if (!header_found)
      throw std::runtime_error(
          "CartesianGrid2D: regular-grid header (NX NZ X0 Z0 DX DZ) not "
          "found.");
    if (nx <= 0 || nz <= 0)
      throw std::runtime_error(
          "CartesianGrid2D: invalid grid dimensions in header.");

    const std::size_t n = static_cast<std::size_t>(nx) * nz;
    std::vector<double> vp(n), vs(n), rho(n);
    std::size_t idx = 0;
    while (std::getline(file, line) && idx < n) {
      std::string t = trim(line);
      if (t.empty() || t[0] == '#')
        continue;
      std::istringstream iss(t);
      iss >> vp[idx] >> vs[idx] >> rho[idx];
      if (iss.fail())
        throw std::runtime_error(
            "CartesianGrid2D: failed to parse data line: " + t);
      ++idx;
    }
    if (idx != n) {
      std::ostringstream msg;
      msg << "CartesianGrid2D: expected " << n << " data lines, got " << idx
          << ".";
      throw std::runtime_error(msg.str());
    }

    return CartesianGrid2D(nx, nz, x0, z0, dx, dz, std::move(vp),
                           std::move(vs), std::move(rho));
  }

  /** Load a scattered-point ASCII file. */
  static CartesianGrid2D load_scattered_ascii(std::ifstream &file) {
    std::vector<double> xs, zs, vp, vs, rho;
    std::string line;
    while (std::getline(file, line)) {
      std::string t = trim(line);
      if (t.empty() || t[0] == '#')
        continue;
      double x_, z_, vp_, vs_, rho_;
      std::istringstream iss(t);
      iss >> x_ >> z_ >> vp_ >> vs_ >> rho_;
      if (iss.fail())
        throw std::runtime_error(
            "CartesianGrid2D: failed to parse scattered data line: " + t);
      xs.push_back(x_);
      zs.push_back(z_);
      vp.push_back(vp_);
      vs.push_back(vs_);
      rho.push_back(rho_);
    }

    if (xs.empty())
      throw std::runtime_error(
          "CartesianGrid2D: no data found in scattered-point file.");

    return CartesianGrid2D(std::move(xs), std::move(zs), std::move(vp),
                           std::move(vs), std::move(rho));
  }

  // -------------------------------------------------------------------------
  // Interpolation implementations
  // -------------------------------------------------------------------------

  std::tuple<double, double, double>
  interpolate_regular(double x, double z, InterpolationMethod method,
                      OutOfBoundsPolicy policy) const {
    const double xmax_val = x0_ + (nx_ - 1) * dx_;
    const double zmax_val = z0_ + (nz_ - 1) * dz_;

    // Check out-of-bounds
    if (x < x0_ || x > xmax_val || z < z0_ || z > zmax_val) {
      if (policy == OutOfBoundsPolicy::error) {
        std::ostringstream msg;
        msg << "CartesianGrid2D: query point (" << x << ", " << z
            << ") is outside the model domain [" << x0_ << ", " << xmax_val
            << "] x [" << z0_ << ", " << zmax_val << "].";
        throw std::runtime_error(msg.str());
      }
      // Clamp
      x = std::max(x0_, std::min(x, xmax_val));
      z = std::max(z0_, std::min(z, zmax_val));
    }

    if (method == InterpolationMethod::nearest || nx_ == 1 || nz_ == 1) {
      // Nearest-neighbour on regular grid
      int ix = static_cast<int>(std::round((x - x0_) / dx_));
      int iz = static_cast<int>(std::round((z - z0_) / dz_));
      ix = std::max(0, std::min(ix, nx_ - 1));
      iz = std::max(0, std::min(iz, nz_ - 1));
      std::size_t idx = static_cast<std::size_t>(iz) * nx_ + ix;
      return { vp_[idx], vs_[idx], rho_[idx] };
    }

    // Bilinear interpolation
    double fx = (x - x0_) / dx_;
    double fz = (z - z0_) / dz_;
    int ix0 = static_cast<int>(std::floor(fx));
    int iz0 = static_cast<int>(std::floor(fz));
    ix0 = std::max(0, std::min(ix0, nx_ - 2));
    iz0 = std::max(0, std::min(iz0, nz_ - 2));
    int ix1 = ix0 + 1, iz1 = iz0 + 1;

    double tx = fx - ix0;
    double tz = fz - iz0;

    auto interp = [&](const std::vector<double> &field) -> double {
      const double v00 =
          field[static_cast<std::size_t>(iz0) * nx_ + ix0]; // (ix0,iz0)
      const double v10 =
          field[static_cast<std::size_t>(iz0) * nx_ + ix1]; // (ix1,iz0)
      const double v01 =
          field[static_cast<std::size_t>(iz1) * nx_ + ix0]; // (ix0,iz1)
      const double v11 =
          field[static_cast<std::size_t>(iz1) * nx_ + ix1]; // (ix1,iz1)
      return (1 - tx) * (1 - tz) * v00 + tx * (1 - tz) * v10 +
             (1 - tx) * tz * v01 + tx * tz * v11;
    };

    return { interp(vp_), interp(vs_), interp(rho_) };
  }

  std::tuple<double, double, double>
  interpolate_nearest(double x, double z, OutOfBoundsPolicy /*policy*/) const {
    // For scattered data we always do nearest-neighbour (no clamp/error check:
    // the nearest point is always valid).
    double best_d2 = std::numeric_limits<double>::max();
    std::size_t best_idx = 0;
    for (std::size_t i = 0; i < xs_.size(); ++i) {
      double dx = x - xs_[i];
      double dz = z - zs_[i];
      double d2 = dx * dx + dz * dz;
      if (d2 < best_d2) {
        best_d2 = d2;
        best_idx = i;
      }
    }
    return { vp_[best_idx], vs_[best_idx], rho_[best_idx] };
  }
};

} // namespace velocity_model
} // namespace io
} // namespace specfem
