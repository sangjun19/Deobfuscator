// Repository: AdityaGupta1/the-sdoajalizer
// File: src/nodes/types/node_color.cu

#include "node_color.hpp"

#include "cuda_includes.hpp"

NodeColor::NodeColor()
    : Node("color")
{
    addPin(PinType::OUTPUT, "image");
}

bool NodeColor::drawPinExtras(const Pin* pin, int pinNumber)
{
    switch (pinNumber)
    {
    case 0: // image
        ImGui::SameLine();
        return NodeUI::ColorEdit4(constParams.color);
    default:
        throw std::runtime_error("invalid pin number");
    }
}

void NodeColor::_evaluate()
{
    Texture* outTex = nodeEvaluator->requestUniformTexture();
    outTex->setUniformColor(ColorUtils::srgbToLinear(constParams.color));
    outputPins[0].propagateTexture(outTex);
}
