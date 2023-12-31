#pragma once
#include <RendererBase.h>

namespace b3d::renderer
{
	class CudaSurfaceObjectWriteTestRenderer final : public RendererBase
	{
	protected:
		auto onRender(const View& view) -> void override;
		auto onInitialize() -> void override;
		auto onDeinitialize() -> void override;
		auto onGui() -> void override;
	};
} // namespace b3d::renderer
