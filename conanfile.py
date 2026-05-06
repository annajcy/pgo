from conan import ConanFile


class PgoRecipe(ConanFile):
    name = "pgo"
    version = "0.1.0"
    package_type = "header-library"
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps", "CMakeToolchain"

    def requirements(self):
        self.requires("eigen/3.4.0")
        self.requires("cli11/[>=2.4 <3]")

    def build_requirements(self):
        self.test_requires("gtest/[>=1.14 <2]")
