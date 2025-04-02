#include "imgui_include.hpp"

#ifndef NDEBUG
#include "../../input_display/input_display.hpp"
#include "../../diagnostics/diagnostics.hpp"

namespace idsp = input_display;
#endif

#include "../../camera_display/camera_display.hpp"
#include "../../../../libs/sdl/sdl_include.hpp"


namespace img = image;
namespace cdsp = camera_display;


static void set_game_window_icon(SDL_Window* window)
{
#include "../../../../resources/icon_64.c" // this will "paste" the struct my_icon into this function
    sdl::set_window_icon(window, icon_64);
}


static void ui_process_input(sdl::EventInfo& evt, input::Input const& prev, input::Input& curr, ui::UIState& state)
{
    //auto& io = *state.io;

    // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
    // - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application, or clear/overwrite your copy of the mouse data.
    // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application, or clear/overwrite your copy of the keyboard data.
    // Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
    
    /*if (!io.WantCaptureKeyboard)
    {
        evt.has_event = true;
        input::process_keyboard_input(evt, prev.keyboard, curr.keyboard);
        evt.first_in_queue = false;
    }

    if (!io.WantCaptureMouse)
    {
        evt.has_event = true;
        input::process_mouse_input(evt, prev.mouse, curr.mouse);
        evt.first_in_queue = false;
    }*/
}


static void texture_window(cstr title, void* texture, u32 width, u32 height, f32 scale)
{
    auto w = width * scale;
    auto h = height * scale;

    ImGui::Begin(title);

    ImGui::Image(texture, ImVec2(w, h));

    ImGui::End();
}


static void ui_camera_controls_window(cdsp::CameraState& state)
{
    ImGui::Begin("Controls");

    camera_display::show_cameras(state);

    ImGui::End();
}


enum class RunState : int
{
    Begin,
    Run,
    End
};


/* main variables */

namespace
{    
    input::Input user_input[2] = {};
    sdl::ControllerInput sdl_controller = {};
    u8 input_id_curr = 0;
    u8 input_id_prev = 1;
    
    cdsp::CameraState camera_state{};
    img::Buffer32 camera_buffer;
    
    constexpr ogl::TextureId camera_texture_id = { 0 };    

    ui::UIState ui_state{};
    SDL_Window* window = 0;
    SDL_GLContext gl_context;
    
    RunState run_state = RunState::Begin;

#ifndef NDEBUG

    idsp::IOState io_state{};
    constexpr ogl::TextureId input_texture_id = { 1 };
    constexpr u32 N_TEXTURES = 2;

#else

    constexpr u32 N_TEXTURES = 1;

#endif

    ogl::TextureList<N_TEXTURES> textures;
}


static bool init_input_display()
{
#ifndef NDEBUG

    if (!idsp::init(io_state))
    {
        sdl::print_message("Error: idsp::init()");
        sdl::close();
        return false;
    }

    auto data = io_state.display.matrix_data_;
    auto w = io_state.display.width;
    auto h = io_state.display.height;

    ogl::init_texture(data, w, h, textures.get(input_texture_id));
    
#endif

    return true;
}


static void init_camera_display()
{
    // TODO init camera app
    u32 w = 640;
    u32 h = 480;
    camera_buffer = img::create_buffer32(w * h, "camera display");
    camera_state.display = img::make_view(w, h, camera_buffer);
    img::fill(camera_state.display, img::to_pixel(128));

    cdsp::init_async(camera_state);

    auto data = camera_state.display.matrix_data_;

    ogl::init_texture(data, w, h, textures.get(camera_texture_id));
}


static void end_program()
{
    run_state = RunState::End;
}


static bool is_running()
{
    return run_state != RunState::End;
}


static void init_input()
{
    sdl::open_game_controllers(sdl_controller, user_input[0]);
    user_input[1].num_controllers = user_input[0].num_controllers;
    user_input[0].frame = user_input[1].frame = (u64)0 - 1;
}


static void swap_inputs()
{
    input_id_prev = input_id_curr;
    input_id_curr = !input_id_curr;

    auto& input = user_input[input_id_curr];
    auto& input_prev = user_input[input_id_prev];

    input::copy_input(input_prev, input);
}


static void handle_window_event(SDL_Event const& event, SDL_Window* window)
{
    switch (event.type)
    {
    case SDL_QUIT:
        end_program();
        break;

    case SDL_WINDOWEVENT:
    {
        switch (event.window.event)
        {
        case SDL_WINDOWEVENT_SIZE_CHANGED:
        //case SDL_WINDOWEVENT_RESIZED:
            int w, h;
            SDL_GetWindowSize(window, &w, &h);
            glViewport(0, 0, w, h);
            break;

        case SDL_WINDOWEVENT_CLOSE:
            end_program();
            break;
        
        default:
            break;
        }
    } break;

    case SDL_KEYDOWN:
    case SDL_KEYUP:
    {
        auto key_code = event.key.keysym.sym;
        switch (key_code)
        {
        case SDLK_ESCAPE:
            sdl::print_message("ESC");
            end_program();
            break;

        default:
            break;
        }

    } break;
    
    default:
        break;
    }
}


static void process_user_input()
{
    swap_inputs();

    auto& input = user_input[input_id_curr];
    auto& input_prev = user_input[input_id_prev];

    input.frame = input_prev.frame + 1;

    sdl::EventInfo evt{};
    evt.has_event = false;

    // Poll and handle events (inputs, window resize, etc.)
    while (SDL_PollEvent(&evt.event))
    {
        handle_window_event(evt.event, window);
        ImGui_ImplSDL2_ProcessEvent(&evt.event);

        evt.has_event = true;

        input::process_keyboard_input(evt, input_prev.keyboard, input.keyboard);
        input::process_mouse_input(evt, input_prev.mouse, input.mouse);        
    }

    input::process_controller_input(sdl_controller, input_prev, input);

    ui_process_input(evt, input_prev, input, ui_state);    
}


static void render_imgui_frame()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplSDL2_NewFrame();
    ImGui::NewFrame();

    // Rendering
    ImGui::DockSpaceOverViewport(nullptr, ImGuiDockNodeFlags_None);

#ifdef SHOW_IMGUI_DEMO
    ui::show_imgui_demo(ui_state);
#endif

#ifndef NDEBUG
    texture_window("Input", textures.get_imgui_texture(input_texture_id), io_state.display.width, io_state.display.height, 2.0f);
    diagnostics::show_diagnostics();
#endif
    
    texture_window("Camera", textures.get_imgui_texture(camera_texture_id), camera_state.display.width, camera_state.display.height, 1.0f);
    ui_camera_controls_window(camera_state);

    ImGui::Render();
    
    ogl::render(window, gl_context);        
}


static bool main_init()
{
    if(!sdl::init())
    {       
        sdl::print_message("Error: sdl::init()"); 
        return false;
    }

    // From 2.0.18: Enable native IME.
#ifdef SDL_HINT_IME_SHOW_UI
    SDL_SetHint(SDL_HINT_IME_SHOW_UI, "1");
#endif

    // fullscreen window
    SDL_DisplayMode dm;
    SDL_GetCurrentDisplayMode(0, &dm);

    window = ui::create_sdl_ogl_window("Camera", dm.w, dm.h);    
    if (!window)
    {
        sdl::print_error("Error: create_sdl_ogl_window()");
        sdl::close();
        return false;
    }

    set_game_window_icon(window);    

    gl_context = SDL_GL_CreateContext(window);
    SDL_GL_MakeCurrent(window, gl_context);
    SDL_GL_SetSwapInterval(1); // Enable vsync

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;    

    // Setup Platform/Renderer backends
    ImGui_ImplSDL2_InitForOpenGL(window, gl_context);
    ImGui_ImplOpenGL3_Init(ogl::get_glsl_version());

    ui::set_imgui_style();

    glViewport(0, 0, (int)io.DisplaySize.x, (int)io.DisplaySize.y);

    textures = ogl::create_textures<N_TEXTURES>();
    
    ui_state.io = &io;

    init_camera_display();

    if (!init_input_display())
    {
        return false;
    }

    return true;
}


static void main_close()
{ 
    cdsp::close_async(camera_state);

#ifndef NDEBUG
    idsp::close(io_state);
#endif     
    
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();
    SDL_GL_DeleteContext(gl_context);

    SDL_DestroyWindow(window);

    sdl::close_game_controllers(sdl_controller, user_input[0]);
    sdl::close();

    mb::destroy_buffer(camera_buffer);
}


static void main_loop()
{
    init_input();
    
    while(is_running())
    {
        process_user_input();

        auto& input = user_input[input_id_curr];

#ifndef NDEBUG
        idsp::update(input, io_state);
        ogl::render_texture(textures.get(input_texture_id));
#endif        
        ogl::render_texture(textures.get(camera_texture_id));

        render_imgui_frame();
    }
}


int main()
{
    if (!main_init())
    {
        return EXIT_FAILURE;
    }

    run_state = RunState::Run;

    while (is_running())
    {
        main_loop();
    }

    main_close();

    return EXIT_SUCCESS;
}

#include "main_o.cpp"