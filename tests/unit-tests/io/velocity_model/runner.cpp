// Minimal test runner for velocity_model_tests.
// Does NOT include dim3/test_fixture.hpp to avoid the uninstantiated
// Assembly3DTest parameterised-suite warning produced by the shared runner.cpp.
#include "SPECFEM_Environment.hpp"
#include "gtest/gtest.h"

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment);
  return RUN_ALL_TESTS();
}
