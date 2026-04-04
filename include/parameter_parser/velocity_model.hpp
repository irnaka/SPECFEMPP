#pragma once

#include "io/reader.hpp"
#include "yaml-cpp/yaml.h"
#include <memory>
#include <string>

namespace specfem {
namespace runtime_configuration {

/**
 * @brief YAML configuration parser for the `velocity-model` section.
 *
 * Parses the following block inside `databases:`:
 * ```yaml
 * databases:
 *   mesh-database: "OUTPUT_FILES/database.bin"
 *   velocity-model:
 *     format:        ascii      # "ascii" (default) or "binary"
 *     file:          "MODEL/velocity.dat"
 *     interpolation: bilinear   # "bilinear" (default) or "nearest"
 *     out-of-bounds: clamp      # "clamp" (default) or "error"
 * ```
 *
 * Call @c instantiate_reader() to obtain a @c specfem::io::reader that can be
 * passed to the assembly constructor.
 */
class velocity_model {
public:
  /**
   * @brief Construct from an explicit parameter set.
   *
   * @param file          Path to the velocity model file.
   * @param format        "ascii" or "binary".
   * @param interpolation "bilinear" or "nearest".
   * @param out_of_bounds "clamp" or "error".
   */
  velocity_model(const std::string &file, const std::string &format,
                 const std::string &interpolation,
                 const std::string &out_of_bounds)
      : file_(file), format_(format), interpolation_(interpolation),
        out_of_bounds_(out_of_bounds) {
    validate();
  }

  /**
   * @brief Construct from a YAML node.
   *
   * The node is expected to be the `velocity-model` mapping.
   *
   * @param node YAML node for the `velocity-model` section.
   */
  explicit velocity_model(const YAML::Node &node) {
    if (!node["file"])
      throw std::runtime_error(
          "velocity-model: 'file' is a required field.");

    file_ = node["file"].as<std::string>();
    format_ = node["format"] ? node["format"].as<std::string>() : "ascii";
    interpolation_ = node["interpolation"]
                         ? node["interpolation"].as<std::string>()
                         : "bilinear";
    out_of_bounds_ =
        node["out-of-bounds"] ? node["out-of-bounds"].as<std::string>() : "clamp";
    validate();
  }

  /**
   * @brief Instantiate a velocity_model_reader ready to be passed to the
   *        assembly constructor.
   *
   * @return Shared pointer to a specfem::io::reader.
   */
  std::shared_ptr<specfem::io::reader> instantiate_reader() const;

  const std::string &file() const { return file_; }
  const std::string &format() const { return format_; }
  const std::string &interpolation() const { return interpolation_; }
  const std::string &out_of_bounds() const { return out_of_bounds_; }

private:
  std::string file_;
  std::string format_;
  std::string interpolation_;
  std::string out_of_bounds_;

  void validate() const {
    if (format_ != "ascii" && format_ != "binary")
      throw std::runtime_error(
          "velocity-model: 'format' must be 'ascii' or 'binary', got: " +
          format_);
    if (interpolation_ != "bilinear" && interpolation_ != "nearest")
      throw std::runtime_error(
          "velocity-model: 'interpolation' must be 'bilinear' or 'nearest', "
          "got: " +
          interpolation_);
    if (out_of_bounds_ != "clamp" && out_of_bounds_ != "error")
      throw std::runtime_error(
          "velocity-model: 'out-of-bounds' must be 'clamp' or 'error', got: " +
          out_of_bounds_);
  }
};

} // namespace runtime_configuration
} // namespace specfem
