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

		auto initialize(b3d::renderer::RenderingDataBuffer& sharedParameters)
		{
			sharedParameters_ = &sharedParameters;
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

	protected:
		RenderingDataBuffer* sharedParameters_{ nullptr };
		std::string name_;
	};

} // namespace b3d::renderer