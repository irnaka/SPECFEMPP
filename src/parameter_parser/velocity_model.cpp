#include "parameter_parser/velocity_model.hpp"
#include "io/velocity_model/impl/cartesian_grid.hpp"
#include "io/velocity_model/reader.hpp"

std::shared_ptr<specfem::io::reader>
specfem::runtime_configuration::velocity_model::instantiate_reader() const {

  // Map format string to enum
  specfem::io::velocity_model_reader::Format fmt =
      (format_ == "binary")
          ? specfem::io::velocity_model_reader::Format::binary
          : specfem::io::velocity_model_reader::Format::ascii;

  // Map interpolation string to enum
  specfem::io::velocity_model::InterpolationMethod method =
      (interpolation_ == "nearest")
          ? specfem::io::velocity_model::InterpolationMethod::nearest
          : specfem::io::velocity_model::InterpolationMethod::bilinear;

  // Map out-of-bounds string to enum
  specfem::io::velocity_model::OutOfBoundsPolicy oob =
      (out_of_bounds_ == "error")
          ? specfem::io::velocity_model::OutOfBoundsPolicy::error
          : specfem::io::velocity_model::OutOfBoundsPolicy::clamp;

  return std::make_shared<specfem::io::velocity_model_reader>(file_, fmt,
                                                              method, oob);
}
