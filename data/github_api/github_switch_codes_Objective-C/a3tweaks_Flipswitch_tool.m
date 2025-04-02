// Repository: a3tweaks/Flipswitch
// File: tool.m

#import <Foundation/Foundation.h>
#import <dlfcn.h>
#import <objc/runtime.h>
#import <unistd.h>

#import "FSSwitchPanel.h"

#ifdef NSLog
#undef NSLog
#endif
#define NSLog(...) fprintf(stderr, "%s\n", [[NSString stringWithFormat:__VA_ARGS__] UTF8String])

#define fprintns(stream, object) fprintf(stream, "%s", [[object description] ?: @"" UTF8String])
#define fprintnsnl(stream, object) fprintf(stream, "%s\n", [[object description] ?: @"" UTF8String])

static void usage(void)
{
	NSLog(@"Usage:\n\tswitch list\n\tswitch get <name>\n\tswitch on <name>\n\tswitch off <name>\n\tswitch toggle <name>");
}

static FSSwitchPanel *FSSwitchPanelMain(void)
{
	dlopen("/usr/lib/libflipswitch.dylib", RTLD_LAZY);
	return [objc_getClass("FSSwitchPanel") sharedPanel];
}

static NSString *getSwitchIdentifier(NSString *name) {
	NSArray *identifiers = FSSwitchPanelMain().switchIdentifiers;
	if ([identifiers containsObject:name]) {
		return name;
	}
	NSString *suffix = [@"." stringByAppendingString:name];
	for (NSString *identifier in identifiers) {
		if ([identifier hasSuffix:suffix]) {
			return identifier;
		}
	}
	return nil;
}

int main(int argc, char *argv[])
{
	NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
	if (argc < 2) {
		usage();
		[pool drain];
		return 0;
	}
	// Read args
	NSMutableArray *args = [NSMutableArray array];
	for (int i = 0; i < argc; i++)
		[args addObject:[NSString stringWithUTF8String:argv[i]]];
	NSString *command = [args objectAtIndex:1];
	switch (argc) {
		case 2: {
			if ([command isEqualToString:@"list"]) {
				for (NSString *switchIdentifier in FSSwitchPanelMain().switchIdentifiers) {
					fprintnsnl(stdout, switchIdentifier);
				}
				return 0;
			}
			break;
		case 3:
			if ([command isEqualToString:@"get"]) {
				NSString *identifier = getSwitchIdentifier([args objectAtIndex:2]);
				if (!identifier) {
					return 1;
				}
				FSSwitchState state = [FSSwitchPanelMain() stateForSwitchIdentifier:identifier];
				switch (state) {
					case FSSwitchStateOff:
						fprintnsnl(stdout, @"off");
						break;
					case FSSwitchStateOn:
						fprintnsnl(stdout, @"on");
						break;
					case FSSwitchStateIndeterminate:
						fprintnsnl(stdout, @"indeterminate");
						break;
				}
				return 0;
			}
			if ([command isEqualToString:@"on"]) {
				NSString *identifier = getSwitchIdentifier([args objectAtIndex:2]);
				if (!identifier) {
					return 1;
				}
				[FSSwitchPanelMain() setState:FSSwitchStateOn forSwitchIdentifier:identifier];
				return 0;
			}
			if ([command isEqualToString:@"off"]) {
				NSString *identifier = getSwitchIdentifier([args objectAtIndex:2]);
				if (!identifier) {
					return 1;
				}
				[FSSwitchPanelMain() setState:FSSwitchStateOff forSwitchIdentifier:identifier];
				return 0;
			}
			if ([command isEqualToString:@"toggle"]) {
				NSString *identifier = getSwitchIdentifier([args objectAtIndex:2]);
				if (!identifier) {
					return 1;
				}
				[FSSwitchPanelMain() applyActionForSwitchIdentifier:identifier];
				return 0;
			}
			break;
		}
	}
	usage();
	[pool drain];
	return 0;
}
