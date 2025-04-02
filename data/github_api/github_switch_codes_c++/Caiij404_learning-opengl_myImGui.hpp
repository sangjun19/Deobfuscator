#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <cmath>
using namespace std;
class myImGui
{
private:
    GLFWwindow *window;

public:
    myImGui(GLFWwindow *w)
    {
        this->window = w;
        // 创建上下文
        ImGui::CreateContext();
        ImGuiIO &io = ImGui::GetIO();
        (void)io;
        // 设置样式
        ImGui::StyleColorsDark();
        // 设置平台和渲染器
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init("#version 330");
    }

    void createSliderFloat(string name, float &val, float min, float max)
    {
        ImGui::Begin("Float Slider");
        ImGui::SliderFloat(name.c_str(), &val, min, max);
        ImGui::End();
    }

    string createFrameInfo()
    {
        float frame = ImGui::GetIO().Framerate;
        int fps_value = (int)round(frame);
        int ms_value = (int)round(1000.0f / frame);

        std::string FPS = std::to_string(fps_value);
        std::string ms = std::to_string(ms_value);
        std::string newTitle = "LearnOpenGL - " + ms + " ms/frame " + FPS;
        glfwSetWindowTitle(window, newTitle.c_str());
    }

    void createButtonSwitchBool(bool &val)
    {
        ImVec2 buttonSize = ImVec2(80, 20);
        ImGui::Begin("imgui");
        if(ImGui::Button("switch", buttonSize))
        {
            val = !val;
        }
        ImGui::End();
    }

    void render()
    {
        // 渲染 gui
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    }

    void newFrame()
    {
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
    }
};