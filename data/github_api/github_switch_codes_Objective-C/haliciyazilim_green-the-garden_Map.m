// Repository: haliciyazilim/green-the-garden
// File: OkParcalari/Models/Map.m

//
//  Map.m
//  GreenTheGarden
//
//  Created by Yunus Eren Guzel on 1/10/13.
//
//

#import "Map.h"
#import "DatabaseManager.h"
#import "GreenTheGardenIAPSpecificValues.h"
#import "GreenTheGardenIAPHelper.h"

@implementation MapPackage

+ (NSArray*)allPackages
{
    NSArray* packages;
    packages =@[
                [MapPackage packageWithId:10000 andName:STANDART_PACKAGE andInAppId:iStandartPackageKey],
                [MapPackage packageWithId:11000 andName:EASY_PACKAGE andInAppId:iEasyPackageKey],
                [MapPackage packageWithId:12000 andName:NORMAL_PACKAGE andInAppId:iNormalPackageKey],
                [MapPackage packageWithId:13000 andName:HARD_PACKAGE andInAppId:iHardPackageKey],
                [MapPackage packageWithId:14000 andName:INSANE_PACKAGE andInAppId:iInsanePackageKey]
                ];
    return packages;
}

+ (MapPackage*) getPackageWithName:(NSString*)name
{
    NSArray* packages = [MapPackage allPackages];
    for (MapPackage* package in packages) {
        if([[package name] isEqualToString:name])
            return package;
    }
    return nil;
}
+ (MapPackage*) packageWithId:(int)packageId andName:(NSString*)name andInAppId:(NSString*) inAppId
{
    MapPackage* package = [[MapPackage alloc] init];
    package.name = name;
    package.packageId = packageId;
    package.inAppId = inAppId;
    return package;
}

- (NSArray *) maps
{
    if(_maps == nil){
        _maps = [[DatabaseManager sharedInstance] getMapsForPackage:self.name];
    }
    return _maps;
}

#pragma mark Alperen dolduracak
- (BOOL)isPurchased
{
    return [[GreenTheGardenIAPHelper sharedInstance] isProductPurchased:self.inAppId];
}

@end

@implementation Map

@dynamic mapId;
@dynamic packageId;
@dynamic score;
@dynamic isFinished;
@dynamic difficulty;
@dynamic stepCount;
@dynamic tileCount;
@dynamic order;
@dynamic isPurchased;
@dynamic isLocked;
@dynamic solveCount;
@dynamic isNotPlayedActiveGame;

@synthesize package;


-(id)init
{
    if(self = [super init]){
        self.isLocked = NO;
    }
    return self;
}

MAP_DIFFICULTY difficultyFromString(NSString* string){
    if([string compare:@"easy"] == 0){
        return EASY;
    }
    else if([string compare:@"normal"] == 0){
        return NORMAL;
    }
    else if([string compare:@"hard"] == 0){
        return HARD;
    }
    else if([string compare:@"insane"] == 0){
        return INSANE;
    }
    return -1;
}
NSString* stringOfDifficulty(MAP_DIFFICULTY difficulty){
    switch (difficulty) {
        case EASY:
            return @"easy";
        case NORMAL:
            return @"normal";
        case HARD:
            return @"hard";
        case INSANE:
            return @"insane";
        default:
            return nil;
    }
}

- (int)getStarCount{
    return [Map starCountForScore:[self.score intValue] andDifficulty:self.difficulty];
}
+ (int) starCountForScore:(int)score andDifficulty:(MAP_DIFFICULTY)difficulty
{
    int oneStarUpperLimit, twoStarUpperLimit, threeStarUpperLimit;
    switch (difficulty) {
        case EASY:
            oneStarUpperLimit = 275;
            twoStarUpperLimit = 175;
            threeStarUpperLimit = 100;
            break;
        case NORMAL:
            oneStarUpperLimit = 400;
            twoStarUpperLimit = 250;
            threeStarUpperLimit = 150;
            break;
        case HARD:
            oneStarUpperLimit = 550;
            twoStarUpperLimit = 350;
            threeStarUpperLimit = 250;
            break;
        case INSANE:
            oneStarUpperLimit = 900;
            twoStarUpperLimit = 600;
            threeStarUpperLimit = 400;
            break;
    }
    
    if(score < threeStarUpperLimit)
        return 3;
    else if(score < twoStarUpperLimit)
        return 2;
    else if(score < oneStarUpperLimit)
        return 1;
    else
        return 0;
}


@end
