from conan import ConanFile
from conan.errors import ConanInvalidConfiguration
from conan.tools.cmake import CMakeDeps, CMakeToolchain


class PgoRecipe(ConanFile):
    name = "pgo"
    version = "0.1.0"
    package_type = "header-library"
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "with_mkl": [True, False],
    }
    default_options = {
        "with_mkl": False,
    }

    def validate(self):
        if self.options.with_mkl and self.settings.os == "Macos":
            raise ConanInvalidConfiguration(
                "with_mkl=True is disabled on macOS; use Apple Accelerate instead."
            )

    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()

        toolchain = CMakeToolchain(self)
        toolchain.user_presets_path = None
        toolchain.generate()

    def requirements(self):
        self.requires("eigen/3.4.0")
        self.requires("cli11/[>=2.4 <3]")

        if self.options.with_mkl and self.settings.os != "Macos":
            self.requires("mkl/[>=2024 <2027]")

    def build_requirements(self):
        self.test_requires("benchmark/[>=1.9 <2]")
        self.test_requires("gtest/[>=1.14 <2]")
