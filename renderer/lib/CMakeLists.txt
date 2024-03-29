set(MODULE_NAME "B3DRendererLibrary")

set(SRC
	Logging.cpp
	RendererBase.cpp
	RenderData.cpp
)

set(SHARED_HEADERS
	Logging.h
	RendererBase.h
	DebugDrawListBase.h
	GizmoHelperBase.h
	NullDebugDrawList.h
	NullGizmoHelper.h
	CudaGpuTimers.h
	OwlExtensions.h
	SharedStructs.h
	RenderFeature.h
	features/ColorMapFeature.h
	features/ColorMapFeature.cpp
	features/TransferFunctionFeature.h
	features/TransferFunctionFeature.cpp
	features/RenderSyncFeature.cpp
	features/RenderSyncFeature.h
	features/BackgroundColorFeature.cpp
	features/BackgroundColorFeature.h
	features/RenderTargetFeature.cpp
	features/RenderTargetFeature.h
	Curve.h
	Curve.cpp
)

add_library(${MODULE_NAME} ${SRC} ${SHARED_HEADERS})
target_include_directories(${MODULE_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(${MODULE_NAME} PUBLIC cxx_std_20)
target_link_libraries(${MODULE_NAME}
  PUBLIC
	owl::owl
)
add_library(B3D::Renderer::Library ALIAS ${MODULE_NAME})

set_property(TARGET ${MODULE_NAME} PROPERTY FOLDER "Renderer")
