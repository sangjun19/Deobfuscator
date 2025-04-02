/* Copyright (c) 2002-2012 Croteam Ltd. 
This program is free software; you can redistribute it and/or modify
it under the terms of version 2 of the GNU General Public License as published by
the Free Software Foundation


This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA. */

#include "StdH.h"
#include "MainWindow.h"
#include "resource.h"

extern BOOL _bWindowChanging = FALSE;    // ignores window messages while this is set
extern HWND _hwndMain = NULL;
static char achWindowTitle[256]; // current window title

static HBITMAP _hbmSplash = NULL;
static BITMAP  _bmSplash;

// for window reposition function
static PIX _pixLastSizeI, _pixLastSizeJ;


// window procedure active while window changes are occuring
LRESULT WindowProc_WindowChanging( HWND hWnd, UINT message, 
			    WPARAM wParam, LPARAM lParam )
{
    switch( message ) {
    case WM_PAINT: {
      PAINTSTRUCT ps;
      BeginPaint(hWnd, &ps); 
      EndPaint(hWnd, &ps); 
 
      return 0;
                   } break;
    case WM_ERASEBKGND: {

      PAINTSTRUCT ps;
      BeginPaint(hWnd, &ps); 
      RECT rect;
      GetClientRect(hWnd, &rect); 
      FillRect(ps.hdc, &rect, (HBRUSH)GetStockObject(BLACK_BRUSH));
      HDC hdcMem = CreateCompatibleDC(ps.hdc); 
      SelectObject(hdcMem, _hbmSplash); 
      BitBlt(ps.hdc, 0, 0, _bmSplash.bmWidth, _bmSplash.bmHeight, hdcMem, 0, 0, SRCCOPY); 
//      StretchBlt(ps.hdc, 0, 0, rect.right, rect.bottom,
//        hdcMem, 0,0, _bmSplash.bmWidth, _bmSplash.bmHeight, SRCCOPY); 

      EndPaint(hWnd, &ps); 

      return 1; 
                   } break;
    case WM_CLOSE:
      return 0;
      break;
    }
    return DefWindowProcA(hWnd, message, wParam, lParam);
}

// window procedure active normally
LRESULT WindowProc_Normal( HWND hWnd, UINT message, 
			    WPARAM wParam, LPARAM lParam )
{
  switch( message ) {

  // system commands
  case WM_SYSCOMMAND: {
    switch( wParam & ~0x0F) {
    // window resizing messages
    case SC_MINIMIZE:
    case SC_RESTORE:
    case SC_MAXIMIZE:
      // relay to application
  	  PostMessage(NULL, message, wParam & ~0x0F, lParam);
      // do not allow automatic resizing
      return 0;
      break;
    // prevent screen saver and monitor power down
    case SC_SCREENSAVE:
    case SC_MONITORPOWER:
      return 0;
    }
                      } break;
  // when close box is clicked
  case WM_CLOSE:
    // relay to application
  	PostMessage(NULL, message, wParam, lParam);
    // do not pass to default wndproc
    return 0;

  // some standard focus loose/gain messages
  case WM_LBUTTONUP:
  case WM_LBUTTONDOWN:
  case WM_RBUTTONDOWN:
  case WM_CANCELMODE:
  case WM_KILLFOCUS:
  case WM_ACTIVATEAPP:
    // relay to application
  	PostMessage(NULL, message, wParam, lParam);
    // pass to default wndproc
    break;
  }

  // if we get to here, we pass the message to default procedure
  return DefWindowProcA(hWnd, message, wParam, lParam);
}

// main window procedure
LRESULT CALLBACK WindowProc( HWND hWnd, UINT message, 
			    WPARAM wParam, LPARAM lParam )
{
  // dispatch to proper window procedure
  if(_bWindowChanging) {
    return WindowProc_WindowChanging(hWnd, message, wParam, lParam);
  } else {
    return WindowProc_Normal(hWnd, message, wParam, lParam);
  }
}


// init/end main window management
void MainWindow_Init(void)
{
  // register the window class
  WNDCLASSEXA wc;
  wc.cbSize = sizeof(wc);
  wc.style = CS_HREDRAW | CS_VREDRAW;
  wc.lpfnWndProc = WindowProc;
  wc.cbClsExtra = 0;
  wc.cbWndExtra = 0;
  wc.hInstance = _hInstance;
  wc.hIcon = LoadIconA( _hInstance, (LPCSTR)IDR_MAINFRAME );
  wc.hCursor = NULL;
  wc.hbrBackground = NULL;
  wc.lpszMenuName = APPLICATION_NAME;
  wc.lpszClassName = APPLICATION_NAME;
  wc.hIconSm = NULL;
  if (0 == RegisterClassExA(&wc)) {
    DWORD dwError = GetLastError();
    CTString strErrorMessage(TRANS("Cannot open main window!"));
    CTString strError;
    strError.PrintF("%s Error %d", strErrorMessage, dwError);
    FatalError(strError);
  }

  // load bitmaps
  _hbmSplash = LoadBitmapA(_hInstance, (char*)IDB_SPLASH);
  ASSERT(_hbmSplash!=NULL);
  GetObject(_hbmSplash, sizeof(BITMAP), (LPSTR) &_bmSplash); 
  // here was loading and setting of no-windows-mouse-cursor
}


void MainWindow_End(void)
{
  DeleteObject(_hbmSplash);
}


// close the main application window
void CloseMainWindow(void)
{
  // if window exists
  if( _hwndMain!=NULL) {
    // destroy it
    DestroyWindow(_hwndMain);
    _hwndMain = NULL;
  }
}


void ResetMainWindowNormal(void)
{
  int iFullscreenWidth = ::GetSystemMetrics(SM_CXSCREEN);
  int iFullscreenHeight = ::GetSystemMetrics(SM_CYSCREEN);

  // RT: let the size, that equals to the screen's one, be the borderless covering whole screen
  bool bBorderlessFull = _pixLastSizeI == iFullscreenWidth && _pixLastSizeJ == iFullscreenHeight;

  ShowWindow( _hwndMain, SW_HIDE);

  PIX pixWidth = 0;
  PIX pixHeight = 0;

  if (bBorderlessFull)
  {
    pixWidth = iFullscreenWidth;
    pixHeight = iFullscreenHeight;
  }
  else
  {
    // add edges and title bar to window size so client area would have size that we requested
    RECT rWindow, rClient;
    GetClientRect(_hwndMain, &rClient);
    GetWindowRect(_hwndMain, &rWindow);

    pixWidth = _pixLastSizeI + (rWindow.right - rWindow.left) - (rClient.right - rClient.left);
    pixHeight = _pixLastSizeJ + (rWindow.bottom - rWindow.top) - (rClient.bottom - rClient.top);
  }

  const PIX pixPosX   = (iFullscreenWidth - pixWidth ) /2;
  const PIX pixPosY   = (iFullscreenHeight - pixHeight) /2;

  if (bBorderlessFull)
  {
    LONG lStyle = GetWindowLong(_hwndMain, GWL_STYLE);
    lStyle &= ~(WS_CAPTION | WS_THICKFRAME | WS_MINIMIZE | WS_MAXIMIZE | WS_SYSMENU);
    SetWindowLong(_hwndMain, GWL_STYLE, lStyle);
    LONG lExStyle = GetWindowLong(_hwndMain, GWL_EXSTYLE);
    lExStyle &= ~(WS_EX_DLGMODALFRAME | WS_EX_CLIENTEDGE | WS_EX_STATICEDGE);
    SetWindowLong(_hwndMain, GWL_EXSTYLE, lExStyle);
  }
  else
  {
    // disable maximize button
    LONG lStyle = GetWindowLong(_hwndMain, GWL_STYLE);
    lStyle &= ~(WS_MAXIMIZEBOX | WS_THICKFRAME);
    // for WS_MAXIMIZEBOX
    lStyle |= WS_SYSMENU; 
    SetWindowLong(_hwndMain, GWL_STYLE, lStyle);
  }

  // set new window size and show it
  SetWindowPos( _hwndMain, NULL, pixPosX,pixPosY, pixWidth,pixHeight, SWP_NOZORDER);
  ShowWindow(   _hwndMain, SW_SHOW);
}


// open the main application window for windowed mode
void OpenMainWindowNormal( PIX pixSizeI, PIX pixSizeJ)
{
  ASSERT(_hwndMain==NULL);

  // create a window, invisible initially
  _hwndMain = CreateWindowExA(
	  WS_EX_APPWINDOW,
	  APPLICATION_NAME,
	  "",   // title
    WS_OVERLAPPED|WS_CAPTION|WS_MINIMIZEBOX|WS_MAXIMIZEBOX|WS_SYSMENU,
	  10,10,
	  100,100,  // window size
	  NULL,
	  NULL,
	  _hInstance,
	  NULL);
  // didn't make it?
  if( _hwndMain==NULL) FatalError(TRANS("Cannot open main window!"));
  SE_UpdateWindowHandle( _hwndMain);

  // set window title
  sprintf( achWindowTitle, TRANS("Serious Sam (Window %dx%d)"), pixSizeI, pixSizeJ);
  SetWindowTextA( _hwndMain, achWindowTitle);
  _pixLastSizeI = pixSizeI;
  _pixLastSizeJ = pixSizeJ;
  ResetMainWindowNormal();
}


// open the main application window for fullscreen mode
void OpenMainWindowFullScreen( PIX pixSizeI, PIX pixSizeJ)
{
  ASSERTALWAYS("Fullscreen is replaced by a borderless window that covers whole screen");
  OpenMainWindowNormal(pixSizeI, pixSizeJ);
}


// open the main application window invisible
void OpenMainWindowInvisible(void)
{
  ASSERT(_hwndMain==NULL);
  // create a window, invisible initially
  _hwndMain = CreateWindowExA(
	  WS_EX_APPWINDOW,
	  APPLICATION_NAME,
	  "",   // title
    WS_POPUP,
	  0,0,
	  10, 10,  // window size
	  NULL,
	  NULL,
	  _hInstance,
	  NULL);
  // didn't make it?
  if( _hwndMain==NULL) {
    DWORD dwError = GetLastError();
    CTString strErrorMessage(TRANS("Cannot open main window!"));
    CTString strError;
    strError.PrintF("%s Error %d", strErrorMessage, dwError);
    FatalError(strError);
  }
  SE_UpdateWindowHandle( _hwndMain);

  // set window title
  sprintf( achWindowTitle, "Serious Sam");
  SetWindowTextA( _hwndMain, achWindowTitle);
}
