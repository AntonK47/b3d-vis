cmake_minimum_required(VERSION 3.26)

project(b3d_visualisation
    VERSION 0.1
    LANGUAGES CXX C CUDA
)
#set(Boost_NO_WARN_NEW_VERSIONS 1)
#set(VCPKG_LOAD_VCVARS_ENV OFF)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

#it a hack, otherwise it will be included by owl package with wrong settings
find_package(glfw3 CONFIG REQUIRED)
find_package(CUDAToolkit)

#TODO: move owl stuff to its own cmakelists file
set(owl_dir ${PROJECT_SOURCE_DIR}/thirdParty/owl)
OPTION(OWL_BUILD_SAMPLES OFF)
add_subdirectory(${owl_dir} EXCLUDE_FROM_ALL)
add_subdirectory(${owl_dir}/samples/common)


function(add_b3d_renderer)

	set(options)
	set(oneValueArgs NAME)
	set(multiValueArgs SOURCES HEADERS LIBRARIES SHARED_LIBRARIES)
	cmake_parse_arguments(TARGET "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

	if(NOT TARGET_NAME)
		message(FATAL_ERROR "NAME must be defined")
	endif()

	set(MODULE_NAME ${TARGET_NAME})
	set(SRC ${TARGET_SOURCES})

	#TODO: Move static/dynamic to optianal parameter
	add_library(${MODULE_NAME} STATIC ${SRC} ${HEADERS})
	target_compile_features(${MODULE_NAME} PUBLIC cxx_std_20)
	if(MSVC)
		set(MSVC_CXX_COMPILER_FLAGS "/Zc:__cplusplus")
		target_compile_options(${MODULE_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${MSVC_CXX_COMPILER_FLAGS}>)
	endif()
	target_include_directories(${MODULE_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

	target_link_libraries(${MODULE_NAME}
		PRIVATE
			B3D::Renderer::Library
			"${TARGET_LIBRARIES}"
	)
	set_target_properties(${MODULE_NAME}
	PROPERTIES
	CUDA_STANDARD 20
	FOLDER "Renderer"
	)

	add_library(B3D::Renderer::${MODULE_NAME} ALIAS ${MODULE_NAME})
endfunction()


add_subdirectory(renderer)
add_subdirectory(viewer)
add_subdirectory(unityCUDAInterop)
add_subdirectory(tools)



set_property(TARGET owl PROPERTY FOLDER "ThirdParty/owl")
set_property(TARGET owl_viewer PROPERTY FOLDER "ThirdParty/owl")
