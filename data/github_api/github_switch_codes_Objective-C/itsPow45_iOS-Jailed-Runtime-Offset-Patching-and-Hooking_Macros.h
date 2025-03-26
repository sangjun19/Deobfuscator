// Repository: itsPow45/iOS-Jailed-Runtime-Offset-Patching-and-Hooking
// File: Macros.h

// C++ version made by Lavochka
// C++ version made by Lavochka
//
//  Macros.h
//  ModMenu
//
//  Created by Joey on 4/2/19.
//  Copyright © 2019 Joey. All rights reserved.
//

#include "Menu.h"
#import "Obfuscate.h"
#import "KittyMemory/writeData.hpp"

#include <substrate.h>
#include <mach-o/dyld.h>

// definition at Menu.h
extern Menu *menu;
extern Switches *switches;

// thanks to shmoo for the usefull stuff under this comment.
#define timer(sec) dispatch_after(dispatch_time(DISPATCH_TIME_NOW, sec * NSEC_PER_SEC), dispatch_get_main_queue(), ^

#define getSym(symName) dlsym((void *)RTLD_DEFAULT, symName)

// Convert hex color to UIColor, usage: For the color #BD0000 you'd use: UIColorFromHex(0xBD0000)
#define UIColorFromHex(hexColor) [UIColor colorWithRed:((float)((hexColor & 0xFF0000) >> 16))/255.0 green:((float)((hexColor & 0xFF00) >> 8))/255.0 blue:((float)(hexColor & 0xFF))/255.0 alpha:1.0]

uint64_t getRealOffset(uint64_t offset){
	return KittyMemory::getAbsoluteAddress([menu getFrameworkName], offset);
}

// Patching a offset without switch.
void patchOffset(uint64_t offset, std::string hexBytes) {
	MemoryPatch patch = MemoryPatch::createWithHex([menu getFrameworkName], offset, hexBytes);
	if(!patch.isValid()){
		[menu showPopup:@"Invalid patch" description:[NSString stringWithFormat:@"Failing offset: 0x%llu, please re-check the hex you entered.", offset]];
		return;
	}
	if(!patch.Modify()) {
      [menu showPopup:@"Something went wrong!" description:[NSString stringWithFormat:@"Something went wrong while patching this offset: 0x%llu", offset]];
    }
}
// C++ version made by Lavochka