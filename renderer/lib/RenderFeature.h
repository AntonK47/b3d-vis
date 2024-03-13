#pragma once
#include <string>
#include "RenderData.h"

namespace b3d::renderer
{
	class RenderFeature
	{
	protected:
		RenderFeature(const std::string& name) : name_{ name }
		{
		}

	public:
		virtual ~RenderFeature()
		{
		}

		virtual auto onInitialize() -> void{}
		virtual auto onDeinitialize() -> void{}

		auto initialize(b3d::renderer::RenderingDataBuffer& sharedParameters) -> void
		{
			sharedParameters_ = &sharedParameters;
			onInitialize();
		}

		auto deinitialize() -> void
		{
			onDeinitialize();
		}

		virtual auto beginUpdate() -> void
		{
		}
		virtual auto endUpdate() -> void
		{
		}

		virtual auto gui() -> void
		{
		}


		[[nodiscard]] inline auto featureName() const -> std::string
		{
			return name_;
		}

		[[nodiscard]] virtual auto hasGui() const -> bool
		{
			return false;
		}

	protected:
		RenderingDataBuffer* sharedParameters_{ nullptr };
		std::string name_;
	};

} // namespace b3d::renderer
