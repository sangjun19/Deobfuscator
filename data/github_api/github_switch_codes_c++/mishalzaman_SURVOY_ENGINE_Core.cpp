#include "Core.h"

void GLAPIENTRY MessageCallback(GLenum source,
    GLenum type,
    GLuint id,
    GLenum severity,
    GLsizei length,
    const GLchar* message,
    const void* userParam)
{
    // Prefix message to clarify source
    const char* severityStr = "";

    switch (severity) {
    case GL_DEBUG_SEVERITY_HIGH:
        severityStr = "HIGH SEVERITY";
        fprintf(stderr, "GL CALLBACK: ** HIGH SEVERITY ERROR ** %s type = 0x%x, message = %s\n",
            (type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : ""),
            type, message);
        break;
    case GL_DEBUG_SEVERITY_MEDIUM:
        severityStr = "MEDIUM SEVERITY";
        fprintf(stderr, "GL CALLBACK: ** MEDIUM SEVERITY ** %s type = 0x%x, message = %s\n",
            (type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : ""),
            type, message);
        break;
    case GL_DEBUG_SEVERITY_LOW:
        severityStr = "LOW SEVERITY";
        fprintf(stderr, "GL CALLBACK: ** LOW SEVERITY ** %s type = 0x%x, message = %s\n",
            (type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : ""),
            type, message);
        break;
    case GL_DEBUG_SEVERITY_NOTIFICATION:
        // Optionally handle notifications differently or ignore them
        // For example, you might not want to log notifications to keep logs cleaner
        severityStr = "NOTIFICATION";
        break;
    default:
        //severityStr = "UNKNOWN SEVERITY";
        //fprintf(stderr, "GL CALLBACK: ** UNKNOWN SEVERITY ** %s type = 0x%x, message = %s\n",
        //    (type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : ""),
        //    type, message);
        break;
    }

    // Use severityStr if you want to prefix all messages with the severity, for example:
    // fprintf(stderr, "GL CALLBACK: %s %s type = 0x%x, severity = 0x%x, message = %s\n",
    //         severityStr, (type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : ""),
    //         type, severity, message);
}


/*==============================================
CORE
==============================================*/

ENGINE::Core::Core() :
    _window(nullptr),
    _context(NULL),
    _error(1),
    _title(""),
    _quit(false)
{

}

ENGINE::Core::~Core() {
    if (_window != nullptr) {
        //LOG_INFO("Destroy _window");
        SDL_DestroyWindow(_window);
    }

    if (_context) {
        SDL_GL_DeleteContext(_context);
    }

    SDL_Quit();
}


bool ENGINE::Core::CreateDevice(
    const char* title
)
{
    _title = title;

    if (
        !_initSDL() ||
        !_initOpengGL() ||
        !_createWindow() ||
        !_createContext() ||
        !_initGlew()
        ) {
        return false;
    }

    _openGLSettings();
    _initImgui();
    _initializeSubSystems();

    return true;
}

void ENGINE::Core::DestroyDevice()
{
}

void ENGINE::Core::BeginRender()
{
    // Clear the screen
    glClearColor(0.5f, 0.5f, 0.8f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glClear(GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
}

void ENGINE::Core::EndRender()
{
    // Swap buffers
    if (_window) {
        SDL_GL_SwapWindow(_window);
        return;
    }

    throw std::runtime_error("Window was not found during game loop");
}

void ENGINE::Core::BeginShutdown()
{
    std::cout << "beginning shutdown" << std::endl;
    _quit = true;
}

/*==============================================
WINDOW
==============================================*/

void ENGINE::Core::_resizeViewport(const int nWidth, const int nHeight)
{
    int setNewW = 0;
    int setNewH = 0;
    int offsetX = 0;
    int offsetY = 0;

    // Calculate the new dimensions while maintaining the 4:3 ratio
    if (nWidth * 3 > nHeight * 4) {
        setNewW = nHeight * 4 / 3;
        setNewH = nHeight;
        offsetX = (nWidth - setNewW) / 2;
    }
    else {
        setNewW = nWidth;
        setNewH = nWidth * 3 / 4;
        offsetY = (nHeight - setNewH) / 2;
    }

    glViewport(offsetX, offsetY, setNewW, setNewH);
}

/*==============================================
INITIALIZATIONS
==============================================*/

bool ENGINE::Core::_initSDL()
{
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        _error = Code::CORE_SDL;
        return false;
    }

    return true;
}

bool ENGINE::Core::_initOpengGL()
{
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 4);

    return true;
}

bool ENGINE::Core::_createWindow()
{
    _window = SDL_CreateWindow(
        _title,
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        Defaults::BASE_SCREEN_WIDTH,
        Defaults::BASE_SCREEN_HEIGHT,
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE
    );

    if (!_window) {
        SDL_Quit();
        _error = Code::CORE_WINDOW;
        return false;
    }

    return true;
}

bool ENGINE::Core::_createContext()
{
    _context = SDL_GL_CreateContext(_window);
    if (!_context) {
        SDL_DestroyWindow(_window);
        SDL_Quit();
        _error = Code::CORE_CONTEXT;
        return false;
    }

    return true;
}

bool ENGINE::Core::_initGlew()
{
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        SDL_DestroyWindow(_window);
        SDL_Quit();
        _error = Code::CORE_GLEW;
        return false;
    }

    return true;
}

void ENGINE::Core::_openGLSettings()
{
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);

    glEnable(GL_DEBUG_OUTPUT);
    glDebugMessageCallback(MessageCallback, 0);
}

void ENGINE::Core::_initImgui()
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    // Setup Platform/Renderer bindings

    // Setup Platform/Renderer backends
    ImGui_ImplSDL2_InitForOpenGL(_window, _context);
    ImGui_ImplOpenGL3_Init("#version 440");

    // Set your Dear ImGui style
    ImGui::StyleColorsDark();
}

void ENGINE::Core::_initializeSubSystems()
{
    Timer = std::make_unique<ENGINE::Timer>(16.6667);
}
