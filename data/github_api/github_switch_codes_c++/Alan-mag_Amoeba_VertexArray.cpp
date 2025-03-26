// Repository: Alan-mag/Amoeba
// File: Amoeba/src/Amoeba/Renderer/VertexArray.cpp

#include "amoebapch.h"
#include "VertexArray.h"

#include "Renderer.h"
#include "Platform/OpenGL/OpenGLVertexArray.h"

namespace Amoeba {

	Ref<VertexArray> VertexArray::Create()
	{
		switch (Renderer::GetAPI())
		{
			case RendererAPI::API::None:    AMOEBA_CORE_ASSERT(false, "RendererAPI::None is currently not supported!"); return nullptr;
			case RendererAPI::API::OpenGL: return std::make_shared<OpenGLVertexArray>();
		}

		AMOEBA_CORE_ASSERT(false, "Unknown RendererAPI!");
		return nullptr;
	}
};