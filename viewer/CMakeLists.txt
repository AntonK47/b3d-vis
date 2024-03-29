find_package(imgui CONFIG REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(Vulkan REQUIRED)
find_package(VulkanHeaders CONFIG)
find_package(imguizmo CONFIG REQUIRED)

add_executable(viewer)
target_sources(viewer PRIVATE
	main.cpp
	NanoViewer.cpp
	Vulkan.h
	Vulkan.cpp
	App.h
	App.cpp
	Pass.h
	GLUtils.h
	passes/FullscreenTexturePass.h
	passes/FullscreenTexturePass.cpp
	passes/InfinitGridPass.h
	passes/InfinitGridPass.cpp
	passes/DebugDrawPass.h
	passes/DebugDrawPass.cpp
	DebugDrawList.h
	DebugDrawList.cpp
	DebugDrawVertex.h
	GizmoHelper.h
	GizmoHelper.cpp
)
target_compile_features(viewer PUBLIC cxx_std_20)
target_link_libraries(viewer
  PRIVATE
	owl_viewer
	owl::owl
	glfw
	imgui::imgui
	B3D::Renderer::Library
	B3D::Renderer::Null
	B3D::Renderer::CudaSurfaceObjectWriteTest
	B3D::Renderer::SimpleTriangles
	B3D::Renderer::Nano
	B3D::Renderer::FastVoxelTraversal
	B3D::Renderer::NanoOutOfCore
	OpenGL::GL
	glad::glad
	Vulkan::Vulkan
	Vulkan::Headers
	imguizmo::imguizmo
	B3D::Tools::ColorMap
	${CUDA_nvml_LIBRARY}
)


target_include_directories(viewer
	PRIVATE
		${Stb_INCLUDE_DIR}
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/resources")
	foreach(RESOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/resources")
		list(APPEND commands
			COMMAND ${CMAKE_COMMAND} -E copy_directory 
                ${RESOURCE_DIR}
                "${CMAKE_CURRENT_BINARY_DIR}/resources")
	endforeach()
endif()

add_custom_target(copy_viewer_resources ALL COMMAND ${commands})

add_dependencies(viewer copy_viewer_resources)
