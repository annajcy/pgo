from conan import ConanFile
from conan.tools.cmake import CMakeDeps, CMakeToolchain


class PgoRecipe(ConanFile):
    name = "pgo"
    version = "0.1.0"
    package_type = "header-library"
    settings = "os", "compiler", "build_type", "arch"

    options = {
        "enable_spdlog": [True, False],
        "enable_alembic": [True, False],
        "enable_tbb": [True, False],
    }
    default_options = {
        "enable_spdlog": False,
        "enable_alembic": False,
        "enable_tbb": False,
        "hwloc/*:shared": True,
    }

    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()

        toolchain = CMakeToolchain(self, generator="Ninja")
        toolchain.user_presets_path = None
        toolchain.generate()

    def requirements(self):
        self.requires("eigen/3.4.0")
        self.requires("cli11/[>=2.4 <3]")
        self.requires("tinyobjloader/2.0.0-rc10", options={"double": True})
        if self.options.enable_alembic:
            self.requires("alembic/1.8.8")
        if self.options.enable_spdlog:
            self.requires("spdlog/[>=1.14 <2]")
        if self.options.enable_tbb:
            self.requires("onetbb/[>=2021.12 <2023]")

    def build_requirements(self):
        self.test_requires("benchmark/[>=1.9 <2]")
        self.test_requires("gtest/[>=1.14 <2]")
