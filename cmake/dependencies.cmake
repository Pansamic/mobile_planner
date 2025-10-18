include(FetchContent)
include(ExternalProject)

set(THIRD_PARTY_PREFIX "${CMAKE_SOURCE_DIR}/third_party")

# Eigen 3.4.0
ExternalProject_Add(eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    DOWNLOAD_NAME eigen-3.4.0.tar.gz
    DOWNLOAD_DIR   "${THIRD_PARTY_PREFIX}/downloads"
    TMP_DIR        "${THIRD_PARTY_PREFIX}/tmp"
    SOURCE_DIR     "${THIRD_PARTY_PREFIX}/src/eigen"
    BINARY_DIR     "${CMAKE_BINARY_DIR}/_deps/eigen-build"
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/_deps/install
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_POLICY_VERSION_MINIMUM=3.10
    INSTALL_COMMAND $(MAKE) install
    TEST_COMMAND ""
)
# list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR}/share/eigen3/cmake)

# yaml-cpp 0.8.0
ExternalProject_Add(yaml-cpp
    URL https://github.com/jbeder/yaml-cpp/archive/refs/tags/0.8.0.tar.gz
    DOWNLOAD_NAME yaml-cpp-0.8.0.tar.gz
    DOWNLOAD_DIR   "${THIRD_PARTY_PREFIX}/downloads"
    TMP_DIR        "${THIRD_PARTY_PREFIX}/tmp"
    SOURCE_DIR     "${THIRD_PARTY_PREFIX}/src/yaml-cpp"
    BINARY_DIR     "${CMAKE_BINARY_DIR}/_deps/yaml-cpp-build"
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/_deps/install
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_POLICY_VERSION_MINIMUM=3.10
        -DYAML_CPP_BUILD_TOOLS=OFF
        -DYAML_BUILD_SHARED_LIBS=OFF
    INSTALL_COMMAND $(MAKE) install
    TEST_COMMAND ""
)
# list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR}/_deps/install/lib/cmake/yaml-cpp)

# BOOST 1.83.0
ExternalProject_Add(boost
    URL https://github.com/boostorg/boost/releases/download/boost-1.89.0/boost-1.89.0-cmake.tar.gz
    DOWNLOAD_NAME boost-1.89.0.tar.gz
    DOWNLOAD_DIR   "${THIRD_PARTY_PREFIX}/downloads"
    TMP_DIR        "${THIRD_PARTY_PREFIX}/tmp"
    SOURCE_DIR     "${THIRD_PARTY_PREFIX}/src/boost"
    BINARY_DIR     "${CMAKE_BINARY_DIR}/_deps/boost-build"
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/_deps/install
        -DCMAKE_BUILD_TYPE=Release
        -DBUILD_SHARED_LIBS=ON
        -DBOOST_CXX_FLAGS=-fPIC
    INSTALL_COMMAND $(MAKE) install
    TEST_COMMAND ""
)

# FLANN 1.9.2
ExternalProject_Add(flann
    URL https://github.com/flann-lib/flann/archive/refs/tags/1.9.2.tar.gz
    DOWNLOAD_NAME flann-1.9.2.tar.gz
    DOWNLOAD_DIR   "${THIRD_PARTY_PREFIX}/downloads"
    TMP_DIR        "${THIRD_PARTY_PREFIX}/tmp"
    SOURCE_DIR     "${THIRD_PARTY_PREFIX}/src/flann"
    BINARY_DIR     "${CMAKE_BINARY_DIR}/_deps/flann-build"
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/_deps/install
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_POLICY_VERSION_MINIMUM=3.10
        -DBUILD_MATLAB_BINDINGS=OFF
        -DBUILD_PYTHON_BINDINGS=OFF
        -DBUILD_EXAMPLES=OFF
        -DBUILD_TESTS=OFF
        -DBUILD_DOC=OFF
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    INSTALL_COMMAND $(MAKE) install
    TEST_COMMAND ""
)

# QHULL 8.0.0
ExternalProject_Add(qhull
    URL http://www.qhull.org/download/qhull-2020-src-8.0.2.tgz
    DOWNLOAD_NAME qhull-8.0.2.tgz
    DOWNLOAD_DIR   "${THIRD_PARTY_PREFIX}/downloads"
    TMP_DIR        "${THIRD_PARTY_PREFIX}/tmp"
    SOURCE_DIR     "${THIRD_PARTY_PREFIX}/src/qhull"
    BINARY_DIR     "${CMAKE_BINARY_DIR}/_deps/qhull-build"
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/_deps/install
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_POLICY_VERSION_MINIMUM=3.10
        # -DBUILD_SHARED_LIBS=OFF
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    INSTALL_COMMAND $(MAKE) install
    TEST_COMMAND ""
)

# VTK 9.2.0
ExternalProject_Add(vtk
    URL https://vtk.org/files/release/9.5/VTK-9.5.2.tar.gz
    DOWNLOAD_NAME vtk-9.5.2.tar.gz
    DOWNLOAD_DIR   "${THIRD_PARTY_PREFIX}/downloads"
    TMP_DIR        "${THIRD_PARTY_PREFIX}/tmp"
    SOURCE_DIR     "${THIRD_PARTY_PREFIX}/src/vtk"
    BINARY_DIR     "${CMAKE_BINARY_DIR}/_deps/vtk-build"
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/_deps/install
        -DCMAKE_BUILD_TYPE=Release
        -DVTK_BUILD_TESTING=OFF
        -DVTK_BUILD_EXAMPLES=OFF
        -DVTK_BUILD_DOCUMENTATION=OFF
        -DVTK_Group_StandAlone=OFF
        -DVTK_Group_Rendering=OFF
        -DModule_vtkCommonCore=ON
        -DModule_vtkCommonDataModel=ON
        -DModule_vtkCommonExecutionModel=ON
        -DModule_vtkCommonMath=ON
        -DModule_vtkCommonMisc=ON
        -DModule_vtkCommonSystem=ON
        -DModule_vtkCommonTransforms=ON
        -DModule_vtkIOCore=ON
        -DModule_vtkIOLegacy=ON
        -DModule_vtkIOXML=ON
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    INSTALL_COMMAND $(MAKE) install
    TEST_COMMAND ""
)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
if(COMPILER_SUPPORTS_AVX2)
    set(PCL_CXX_FLAGS "-mavx2")
else()
    check_cxx_compiler_flag("-mavx" COMPILER_SUPPORTS_AVX)
    if(COMPILER_SUPPORTS_AVX)
        set(PCL_CXX_FLAGS "-mavx")
    else()
        set(PCL_CXX_FLAGS "")  # fallback: no AVX
    endif()
endif()
# PCL 1.15.1
ExternalProject_Add(pcl
    URL https://github.com/PointCloudLibrary/pcl/archive/refs/tags/pcl-1.15.1.zip
    DOWNLOAD_NAME pcl-1.15.1.zip
    DOWNLOAD_DIR   "${THIRD_PARTY_PREFIX}/downloads"
    TMP_DIR        "${THIRD_PARTY_PREFIX}/tmp"
    SOURCE_DIR     "${THIRD_PARTY_PREFIX}/src/pcl"
    BINARY_DIR     "${CMAKE_BINARY_DIR}/_deps/pcl-build"
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/_deps/install
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_MODULE_PATH=${CMAKE_MODULE_PATH}
        -DCMAKE_PREFIX_PATH=${CMAKE_BINARY_DIR}/_deps/install
        -DBoost_USE_STATIC_LIBS=OFF
        -DCMAKE_CXX_FLAGS=${PCL_CXX_FLAGS}
        -DEIGEN_MAX_ALIGN_BYTES=32
    INSTALL_COMMAND $(MAKE) install
    TEST_COMMAND ""
    DEPENDS eigen boost flann qhull vtk
)
# list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR}/_deps/install/share/pcl-1.15)