// Repository: rollbar/rollbar-apple
// File: RollbarNotifier/Sources/RollbarNotifier/DTOs/RollbarAppLanguage.m

#import "RollbarAppLanguage.h"

@implementation RollbarAppLanguageUtil

+ (NSString *) RollbarAppLanguageToString:(RollbarAppLanguage)value {
    
    switch (value) {
        case RollbarAppLanguage_ObjectiveC:
            return @"Objective-C";
        case RollbarAppLanguage_ObjectiveCpp:
            return @"Objective-C++";
        case RollbarAppLanguage_Swift:
            return @"Swift";
        case RollbarAppLanguage_C:
            return @"C";
        case RollbarAppLanguage_Cpp:
            return @"C++";
        default:
            return @"Objective-C";
    }
}

+ (RollbarAppLanguage) RollbarAppLanguageFromString:(NSString *)value {
    
    if (NSOrderedSame == [value caseInsensitiveCompare:@"Objective-C"]) {
        return RollbarAppLanguage_ObjectiveC;
    }
    else if (NSOrderedSame == [value caseInsensitiveCompare:@"Objective-C++"]) {
        return RollbarAppLanguage_ObjectiveCpp;
    }
    else if (NSOrderedSame == [value caseInsensitiveCompare:@"Swift"]) {
        return RollbarAppLanguage_Swift;
    }
    else if (NSOrderedSame == [value caseInsensitiveCompare:@"C"]) {
        return RollbarAppLanguage_C;
    }
    else if (NSOrderedSame == [value caseInsensitiveCompare:@"C++"]) {
        return RollbarAppLanguage_Cpp;
    }
    else {
        return RollbarAppLanguage_ObjectiveC; // default case...
    }
}

@end
