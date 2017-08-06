cmake_minimum_required(VERSION 3.4.1)
project(wrap_cuda_deformer)

# in case a gcc version is too new for cuda
# cmake -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ .

set(MAYA_VERSION 2017)
set(CUSTOM_DEFINITIONS "REQUIRE_IOSTREAM;_BOOL")
set(MAYA_INSTALL_BASE_SUFFIX "")
set(MAYA_INC_SUFFIX "include")
set(MAYA_LIB_SUFFIX "lib")
set(MAYA_INSTALL_BASE_SUFFIX "")
if(WIN32)
    set(MAYA_INSTALL_BASE_DEFAULT "C:/Program Files/Autodesk")
elseif(APPLE)
    set(MAYA_INSTALL_BASE_DEFAULT /Applications/Autodesk)
else()
    set(MAYA_INSTALL_BASE_DEFAULT /usr/autodesk)
    if(MAYA_VERSION LESS 2016)
        SET(MAYA_INSTALL_BASE_SUFFIX -x64)
    endif()
endif()
set(MAYA_INSTALL_BASE_PATH ${MAYA_INSTALL_BASE_DEFAULT})
set(MAYA_LOCATION
   ${MAYA_INSTALL_BASE_PATH}/maya${MAYA_VERSION}${MAYA_INSTALL_BASE_SUFFIX})


find_package(OpenGL REQUIRED)
find_package(CUDA REQUIRED)


set(SOURCE_FILES
    "point_to_triangle.cc"
    "wrap_cuda_deformer.cc"
    "wrap_cuda_main.cc"
)

set(LIBRARIES ${LIBRARIES}
    "Foundation"
    "OpenMaya"
    "OpenMayaAnim"
    "OpenMayaUI"
    "OpenMayaRender"
    ${OPENGL_LIBRARIES})


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

set(CUDA_NVCC_FLAGS "-arch=sm_50 -Xcompiler -fPIC")

set(_MAYA_LOCATION
    ${MAYA_INSTALL_BASE_PATH}/maya${MAYA_VERSION}${MAYA_INSTALL_BASE_SUFFIX})
set(_PROJECT ${PROJECT_NAME})

include_directories(${_MAYA_LOCATION}/${MAYA_INC_SUFFIX}
                    ${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS})

link_directories(${_MAYA_LOCATION}/${MAYA_LIB_SUFFIX})

#cuda_add_library(cuda_ops STATIC "cuda_ops.cu")

add_library(${_PROJECT} SHARED ${SOURCE_FILES})

#target_link_libraries(${_PROJECT} cuda_ops ${LIBRARIES} ${CUDA_LIBRARIES})
target_link_libraries(${_PROJECT} ${LIBRARIES})

set_target_properties(${_PROJECT} PROPERTIES
                      COMPILE_DEFINITIONS "${CUSTOM_DEFINITIONS}")

set_target_properties(${_PROJECT} PROPERTIES OUTPUT_NAME ${PROJECT_NAME})
set_target_properties(${_PROJECT} PROPERTIES CLEAN_DIRECT_OUTPUT 1)

if(WIN32)
    set_target_properties(${_PROJECT} PROPERTIES SUFFIX ".mll" )
    set_target_properties(${_PROJECT} PROPERTIES LINK_FLAGS
        "/export:initializePlugin /export:uninitializePlugin" )
elseif(APPLE)
    set_target_properties(${_PROJECT} PROPERTIES PREFIX "" )
    set_target_properties(${_PROJECT} PROPERTIES SUFFIX ".bundle" )
else()
    set_target_properties(${_PROJECT} PROPERTIES PREFIX "" )
endif()

