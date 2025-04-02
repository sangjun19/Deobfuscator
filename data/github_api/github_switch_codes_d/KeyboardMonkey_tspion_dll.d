// Repository: KeyboardMonkey/tspion
// File: src/dll.d

//Public domain

import std.c.windows.windows;
import core.sys.windows.dll;
import hook;

__gshared HINSTANCE g_hInst;

extern (Windows)
BOOL DllMain(HINSTANCE hInstance, ULONG ulReason, LPVOID pvReserved)
{
    switch (ulReason)
    {
	case DLL_PROCESS_ATTACH:
	    g_hInst = hInstance;
	    dll_process_attach(hInstance);
	    break;

	case DLL_PROCESS_DETACH:
	    dll_process_detach(hInstance);
	    break;

	case DLL_THREAD_ATTACH:
	    dll_thread_attach();
	    break;

	case DLL_THREAD_DETACH:
	    dll_thread_detach();
	    break;
    }
    return true;
}
