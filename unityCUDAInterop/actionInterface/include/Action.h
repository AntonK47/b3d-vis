#pragma once

#include "IUnityInterface.h"
#pragma once

#include "PluginHandler.h"
// 	TODO: Only required if we're an action which is using RendererBase. Introduce RenderAction and leave this Action as generic as possible.
#include "RendererBase.h"

extern b3d::unity_cuda_interop::PluginHandler sPluginHandler;

namespace b3d::unity_cuda_interop
{

	// See TODO above
	struct NativeRenderingData
	{
		int eyeCount{ 1 };
		renderer::Camera nativeCameradata[2];
	};

	// See TODO above
	struct NativeRenderingDataWrapper
	{
		NativeRenderingData nativeRenderingData{};
		void* additionalDataPointer{ nullptr };
	};

	class PluginLogger;
	class Action
	{
	public:
		static constexpr int eventIdCount = 10;

		Action() = default;

		virtual ~Action()
		{
			isRegistered_ = false;
			logger_ = nullptr;
			renderAPI_ = nullptr;
			renderEventIDOffset_ = -1;
		}

		auto renderEventAndData(const int eventID, void* data) -> void
		{
			const auto actionRenderId = eventID - renderEventIDOffset_;
			customRenderEvent(actionRenderId, data);
		}

		virtual auto getRenderEventIDOffset() const -> int
		{
			return renderEventIDOffset_;
		}

		auto registerAction(const int renderEventIdOffset, PluginLogger* logger, RenderAPI* renderAPI)
		{
			if (isRegistered_)
			{
				return;
			}
			
			renderEventIDOffset_ = renderEventIdOffset;
			logger_ = logger;
			renderAPI_ = renderAPI;

			isRegistered_ = true;
		}

		auto containsEventId(const int eventId) -> bool
		{
			return renderEventIDOffset_ <= eventId && eventId < renderEventIDOffset_ + Action::eventIdCount;
		}

		auto unregisterAction()
		{
			if (!isRegistered_)
			{
				return;
			}

			isRegistered_ = false;

			logger_ = nullptr;
			renderAPI_ = nullptr;
			renderEventIDOffset_ = 0;
		}


		virtual auto initialize(void* data) -> void = 0;
		virtual auto teardown() -> void = 0;

	protected:
		virtual auto customRenderEvent(int eventId, void* data) -> void = 0;
		
		PluginLogger* logger_{ nullptr };
		RenderAPI* renderAPI_{ nullptr };

		int renderEventIDOffset_{ -1 };
		bool isRegistered_{ false };
		bool isInitialized_{ false };
	};

} // namespace b3d::unity_cuda_interop

extern "C"
{
	UNITY_INTERFACE_EXPORT auto UNITY_INTERFACE_API createAction() -> b3d::unity_cuda_interop::Action*;

	UNITY_INTERFACE_EXPORT auto UNITY_INTERFACE_API destroyAction(b3d::unity_cuda_interop::Action* action) -> void;

	UNITY_INTERFACE_EXPORT auto UNITY_INTERFACE_API initializeAction(b3d::unity_cuda_interop::Action* action, void* data) -> void;

	UNITY_INTERFACE_EXPORT auto UNITY_INTERFACE_API teardownAction(b3d::unity_cuda_interop::Action* action) -> void;

	UNITY_INTERFACE_EXPORT auto UNITY_INTERFACE_API getRenderEventIDOffset(const b3d::unity_cuda_interop::Action* action) -> int;
}
