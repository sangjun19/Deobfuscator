// Repository: vaishnavib2205/RTR2023_118_vaishnavi_bhave
// File: 06_MACOS/04_LogFile/Window.m

#import <Foundation/Foundation.h>
#import <Cocoa/Cocoa.h>

// Global Variable Declaration

FILE *gpFile = NULL;

@interface AppDelegate:NSObject <NSApplicationDelegate , NSWindowDelegate>
@end

@interface View:NSView
@end

// main function

int main(int argc , char* argv[])
{
    // code

    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init]; // create autorelease pool for mem management

    NSApp = [NSApplication sharedApplication]; // create global,shared application object NSApp
    
    [NSApp setDelegate : [[AppDelegate alloc]init]];// set its delegate to our own custom AppDelegate

    [NSApp run]; // start NSApps run loop

    [pool release];// let auto release pool release all from our applications

}

// implimentation of our custom appdelegate interface

@implementation AppDelegate
{
    NSWindow *window;
    View *view;
}

-(void)applicationDidFinishLaunching:(NSNotification *)notification
{
    // code
    
    // Log File creation/Opening code
    NSBundle *appBundle = [NSBundle mainBundle];
    NSString *appDirPath = [appBundle bundlePath];
    NSString *parentDirPath = [appDirPath stringByDeletingLastPathComponent];
    NSString *logFileNameWithPath = [NSString stringWithFormat:@"%@/Log.txt" , parentDirPath];
    const char *pszLogFileNameWithPath = [logFileNameWithPath cStringUsingEncoding:NSASCIIStringEncoding];
    
    gpFile = fopen(pszLogFileNameWithPath , "w");
    if(gpFile == NULL)
    {
        NSLog(@"Log File cannot be created");
        [self release];
        [NSApp terminate:self];
    }

    fprintf(gpFile , "Program Started Successfully \n");
    

    // declare rectangle for frame/border of our window
    NSRect win_rect = NSMakeRect(0.0 , 0.0 , 800.0 , 600.0);
    // create the window
    window = [[NSWindow alloc]initWithContentRect:win_rect
                                        styleMask:  NSWindowStyleMaskTitled |
                                                    NSWindowStyleMaskClosable |
                                                    NSWindowStyleMaskMiniaturizable |
                                                    NSWindowStyleMaskResizable
                                        backing  :  NSBackingStoreBuffered
                                        defer    :  NO ];

    // give title to the window
    [window setTitle:@"VB Cocoa Window"];

    // center the window
    [window center];

    // create the custuom view
    view = [[View alloc] initWithFrame:win_rect];

    // set this newly created custum view as view of our newly created custom view
    [window setContentView:view];

    // setdelegate for window
    [window setDelegate:self];

    // now actually show the window, give its keyboard focus ,and make it top on z order
    [window makeKeyAndOrderFront:self];

}

-(void) applicationWillTerminate:(NSNotification *)notification
{
    // log file closing code
    if(gpFile)
    {
        fprintf(gpFile , "Program Finished successfully \n");
        fclose(gpFile);
        gpFile = NULL;
    }
}

-(void) windowWillClose:(NSNotification *)notification
{
    // code
    [NSApp terminate:self];

}

-(void)dealloc
{
    [super dealloc];
    [view release];
    [window release];

}

@end

// implimentation of custom view interface

@implementation View
{
    NSString *string;
}

-(id) initWithFrame:(NSRect)frame
{
    // code
    self = [super initWithFrame:frame];

    if(self)
    {
        string = @"Hello World !!!" ;
    }

    return self;
}

-(void) drawRect:(NSRect)dirtyRect
{
    // declare bg color as black
    NSColor *bgColor = [NSColor blackColor];
    
    // set this color
    [bgColor set];

    // fill the window rec/dirtyrectangle with above black color
    NSRectFill(dirtyRect);

    //create the font for our screen
    NSFont *stringFont = [NSFont fontWithName:@"Helvetica" size:32 ];

    //crate string color
    NSColor *fgColor = [NSColor greenColor];

    // create dictionary for our string attributes using above two
    NSDictionary *stringDictionary = [NSDictionary dictionaryWithObjectsAndKeys:stringFont,NSFontAttributeName, fgColor,NSForegroundColorAttributeName, nil];

    // create size of the string according to size of font and otehr attributes
    NSSize stringSize = [string sizeWithAttributes:stringDictionary];
    
    // create point for drawing of string
    NSPoint stringPoint = NSMakePoint((dirtyRect.size.width/2 - stringSize.width/2) , (dirtyRect.size.height/2 - stringSize.height/2));

    // draw the string
    [string drawAtPoint:stringPoint withAttributes:stringDictionary ];

}

-(bool)acceptsFirstResponder
{
    // code
    [[self window]makeFirstResponder:self];

    return YES;
}

-(void)keyDown:(NSEvent *)event
{
    int key = (int)[[event characters]characterAtIndex:0];
    switch(key)
    {
        case 27:
            [self release];
            [NSApp terminate:self];
            break;

        case 'F':
        case 'f':
            [[self window]toggleFullScreen:self];
            break;

        default:
            break;

    }
}

-(void)mouseDown:(NSEvent *)event
{
    string = @"LeftMouse Button is clicked";
    [self setNeedsDisplay:YES];  // Repaint = YES;

}

-(void)rightMouseDown:(NSEvent *)event
{
    string = @"Right Mouse button is clicked";
    [self setNeedsDisplay:YES]; // Repaint = YES;
}

-(void)dealloc
{
    [super dealloc];
    
}

@end



