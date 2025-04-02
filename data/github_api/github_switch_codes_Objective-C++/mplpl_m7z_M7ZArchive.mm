// Repository: mplpl/m7z
// File: m7z/M7ZArchive.mm

//
//  M7ZArchive.m
//  m7z
//
//  Created by MPL on 21/03/16.
//  Copyright © 2016-2022 MPL. All rights reserved.
//

#import "M7ZArchive.h"
#import "M7ZItem.h"
#import "NSString+wstring.h"

#import <string>
#import <vector>

#import "lib7z.h"
#import "MLUpdateCallback.h"
#import "MLExtractCallback.h"

#import <iconv.h>

using namespace lib7z;

class M7ZArchiveCallback: public MLUpdateCallback, public MLExtractCallback
{
    
public:
    id<M7ZArchiveDelegate> delegat;
    
    M7ZArchiveCallback(id<M7ZArchiveDelegate> delegat): delegat(delegat) {}
    
    int SetTotal(unsigned long long size) {
        delegat.total = size;
        return 0;
    }
    
    int CompressingItem(const wchar_t *name, bool isAnti, int mode){
        [delegat item:[NSString stringWithWstring:name] mode:1000 + mode];
        return delegat.shouldBreak;
    }
    
    int DecompressingItem(const wchar_t *name, bool isFolder, int mode, const unsigned long long *position) {
        [delegat item:[NSString stringWithWstring:name] mode:2000 + mode];
        return delegat.shouldBreak;
    }
    
    int SetCompleted(const unsigned long long *completeValue) {
        delegat.completed = *completeValue;
        return delegat.shouldBreak;
    }
    
    int SetOperationResult(int x, int kind) {
        if (delegat.shouldBreak) return 1;
        if (x == 0) return 0;
        
        BOOL ret = YES;
        switch (kind)
        {
            case 1:
                x = x + 1000;
                ret = [delegat error:x];
                break;
                
            case 2:
                x = x + 2000;
                ret = [delegat error:x];
                break;
        }
        
        return (ret)? 0 : x;
    }
    
    int FinishArchive() {
        [delegat done];
        return 0;
    }
    
    int AskOverwrite (
        const wchar_t *existName, const time_t *existTime, const unsigned long long *existSize,
        const wchar_t *newName, const time_t *newTime, const unsigned long long *newSize,
        int *answer) {
        
        *answer = [delegat shouldOverwriteItem:(existName) ? [NSString stringWithWstring:existName] : @""
                                          date:(existTime) ? [NSDate dateWithTimeIntervalSince1970:*existTime] : nil
                                          size:(existSize) ? *existSize : 0
                                       newName:(newName) ? [NSString stringWithWstring:newName] : @""
                                       newDate:(newTime) ? [NSDate dateWithTimeIntervalSince1970:*newTime] : nil
                                       newSize:(newSize) ? *newSize : 0];
        return 0;
    }
    
    const std::wstring GetPassword() {
        
        NSString *pass = delegat.password;

        if (!pass) {
            return L"";
        } else {
            return [pass wstring];
        }
    }

    int OpenFileError(const wchar_t *name, int systemError) {
        // note, that returning 0 aborts, and 1 make the library skip the given file
        return [delegat openFileError:(name) ? [NSString stringWithWstring:name] : @""
                                 code:systemError] ? 0 : 1;
    }
    
    int MessageError(const wchar_t *message) {
        return ([delegat error:M7Z_ORC_OtherError]) ? 0 : M7Z_RC_FAIL;
    }
};

@implementation M7ZRenameItem

-(instancetype)initWithFrom:(NSString *)from to:(NSString *)to {
    
    if (self = [super init]) {
        self.from = from;
        self.to = to;
    }
    
    return self;
}

@end

@implementation M7ZArchive

-(instancetype)initWithName:(NSString *)name {
    
    if (self = [super init]) {
        _name = name;
    }
    
    return self;
}

-(instancetype)initWithName:(NSString *)name
                   encoding:(NSString *)encoding {
    
    if (self = [super init]) {
        _name = name;
        _encoding = encoding;
    }
    
    return self;
}

-(instancetype)initWithName:(NSString *)name
                   encoding:(NSString *)encoding
           storeCreatedTime:(BOOL)storeCreatedTime {
    
    if (self = [super init]) {
        _name = name;
        _encoding = encoding;
        _storeCreatedTime = storeCreatedTime;
    }
    
    return self;
}

-(int)listItemsTo:(NSMutableArray<M7ZItem *> *)output {
    
    std::vector<DirectoryItem> ra;
    M7ZArchiveCallback cb(self.delegate);
    int ret = MLListArchive([self.name wstring], ra, cb, [self.encoding wstring]);
    if (ret != 0) {
        return ret;
    }
    
    for (std::vector<DirectoryItem>::iterator it = ra.begin(); it < ra.end(); it++)
    {
        DirectoryItem di = *it;
        
        M7ZItem *item = [[M7ZItem alloc] initWithName:[NSString stringWithWstring:di.name]
                                                 size:di.size
                                       sizeCompressed:di.sizeCompressed
                                         creationTime:@(di.creationTime.c_str())
                                     modificationTime:@(di.modificationTime.c_str())
                                           accessTime:@(di.accessTime.c_str())
                                                attrs:@(di.attrs.c_str())
                                            encrypted:di.encrypted
                                    compressionMethod:[NSString stringWithWstring:di.compressionMethod]];
        [output addObject:item];
    }
    
    return 0;
    
}

-(int)addItems:(NSArray<NSString *> *)items {
    
    return [self addItems:items
            encryptHeader:NO
         compressionLevel:5
            moveToArchive:NO
                excluding:@[]];
}

-(int)addItems:(NSArray<NSString *> *)items
 encryptHeader:(BOOL)encryptHeader {
    
    return [self addItems:items
            encryptHeader:encryptHeader
         compressionLevel:5
            moveToArchive:NO
                excluding:@[]];
}

-(int)addItems:(NSArray<NSString *> *)items
 encryptHeader:(BOOL)encryptHeader
compressionLevel:(NSInteger)compressionLevel
 moveToArchive:(BOOL)moveToArchive
     excluding:(NSArray<NSString *> *)exclusionWildcards {
    
    std::vector<std::wstring> itemsW;
    for (NSString *item in items) {
        itemsW.push_back([item wstring]);
    }
    std::vector<std::wstring> exclusionWildcardsW;
    for (NSString *exclusionWildcard in exclusionWildcards) {
        exclusionWildcardsW.push_back([exclusionWildcard wstring]);
    }
    M7ZArchiveCallback cb(self.delegate);
    int ret = MLAddToArchive([self.name wstring], itemsW, cb, encryptHeader, (int)compressionLevel,
                             [self.workDir wstring], [self.encoding wstring], self.storeCreatedTime,
                             moveToArchive, exclusionWildcardsW);
    return ret;
}

-(int)extractAllToDir:(NSString *)dir {
    return [self extractItems:nil toDir:dir];
}

-(int)extractAllToDir:(NSString *)dir
            excluding:(NSArray<NSString *> *)exclusionWildcards {
    
    return [self extractItems:nil toDir:dir excluding:exclusionWildcards];
}

-(int)extractItems:(NSArray<NSString *> *)items
             toDir:(NSString *)dir {

    return [self extractItems:items toDir:dir excluding:@[]];
}


-(int)extractItems:(NSArray<NSString *> *)items
             toDir:(NSString *)dir
         excluding:(NSArray<NSString *> *)exclusionWildcards {
    
    std::vector<std::wstring> files;
    for (NSString *item : items) {
        files.push_back(item.wstring);
    }
    std::vector<std::wstring> exclusionWildcardsW;
    for (NSString *exclusionWildcard in exclusionWildcards) {
        exclusionWildcardsW.push_back([exclusionWildcard wstring]);
    }
    M7ZArchiveCallback cb(self.delegate);
    int ret = MLExtractFromArchive([self.name wstring], [dir wstring], files, cb,
                                   [self.workDir wstring], [self.encoding wstring],
                                   exclusionWildcardsW);
    return ret;
}

-(int)deleteItems:(NSArray<NSString *> *)items {
    
    std::vector<std::wstring> itemsW;
    for (NSString *item in items) {
        itemsW.push_back([item wstring]);
    }
    M7ZArchiveCallback cb(self.delegate);
    int ret = MLDeleteFromArchive([self.name wstring], itemsW, cb, [self.workDir wstring],
                                  [self.encoding wstring]);
    return ret;
}

-(int)renameItem:(NSString *)existingName
         newName:(NSString *)newName {
    
    return [self renameItems:@[[[M7ZRenameItem alloc] initWithFrom:existingName to:newName]]];
}

-(int)renameItems:(NSArray<M7ZRenameItem *> *)items {
    
    std::vector<std::wstring> itemsW;
    for (M7ZRenameItem *item in items) {
        itemsW.push_back([item.from wstring]);
        itemsW.push_back([item.to wstring]);
    }
    M7ZArchiveCallback cb(self.delegate);
    int ret = MLRenameItemsInArchive([self.name wstring], itemsW, cb, [self.workDir wstring],
                                     [self.encoding wstring]);
    return ret;
}

int do_one(unsigned int count, const    char * const *names, void *arg) {
    NSMutableArray<NSString *> *arr = (__bridge NSMutableArray *)arg;
    [arr addObject:[NSString stringWithUTF8String:iconv_canonicalize(names[0])]];
    return 0;
}

+(NSArray<NSString *> *)supportedEncodings {
    
    std::vector<std::wstring> encs = MLSupportedEncodings();
    NSMutableArray<NSString *> *ret = [[NSMutableArray alloc] init];
    
    for (std::vector<std::wstring>::iterator it = encs.begin(); it < encs.end(); it++) {
        [ret addObject:[NSString stringWithWstring:*it]];
    }
    return ret;
}

@end
