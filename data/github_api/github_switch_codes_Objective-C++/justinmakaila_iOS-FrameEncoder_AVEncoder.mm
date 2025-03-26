// Repository: justinmakaila/iOS-FrameEncoder
// File: AVEncoder.mm

//
//  AVEncoder.m
//  Encoder Demo
//
//  Created by Geraint Davies on 14/01/2013.
//  Copyright (c) 2013 GDCL http://www.gdcl.co.uk/license.htm
//

#import "AVEncoder.h"
#import "NALUnit.h"

static unsigned int to_host(unsigned char* p) {
    return (p[0] << 24) + (p[1] << 16) + (p[2] << 8) + p[3];
}

#define OUTPUT_FILE_SWITCH_POINT (50 * 1024 * 1024)  // 50 MB switch point
#define MAX_FILENAME_INDEX  5                        // filenames "capture1.mp4" wraps at capture5.mp4


@interface AVEncoder () {
    VideoEncoder* _headerWriter;        // Initial AVAssetWriter used to write avcC information
    
    VideoEncoder* _writer;              // AVAssetWriter used to write video to file
    
    NSFileHandle* _inputFile;           // Input file handle
    dispatch_queue_t _readQueue;        // Queue to perform read operations
    dispatch_source_t _readSource;      // Creates a source to execute queue's
    
    BOOL _swapping;                     // Indicates if the files are currently swapping
    int _currentFile;                   // Represents the current file
    int _height;                        // Height of the file to be recorded
    int _width;                         // Width of the file to b recorded
    
    NSData* _avcC;                      // avcC Information
    int _lengthSize;                    // Size of the length field
    
    BOOL _foundMDAT;                    // Indicates if an MDAT was found
    uint64_t _posMDAT;                  // Position of found MDAT
    int _bytesToNextAtom;               // Number of bytes to the next atom
    BOOL _needParams;                   // Indicates if avcC params are still needed
    
    int _prev_nal_idc;                  // Previous NAL IDC
    int _prev_nal_type;                 // Previous NAL type
    
    NSMutableArray* _pendingNALU;       // NALUs awaiting to be processed. Contains up to 2 NALUs comprising a single frame.
    NSMutableArray* _times;             // FIFO array of times
    
    encoder_handler_t _outputBlock;     // Called when there is output
    param_handler_t _paramsBlock;       // Called when params are complete
    
    int _bitspersecond;                 // BPS
    double _firstpts;                   // First PTS
}

- (void) initForHeight:(int)height andWidth:(int)width;

@end

@implementation AVEncoder

@synthesize bitspersecond = _bitspersecond;

+ (AVEncoder*) encoderForHeight:(int) height andWidth:(int) width {
    NSLog(@"Encoder for %i by %i", height, width);
    AVEncoder* enc = [AVEncoder alloc];
    [enc initForHeight:height andWidth:width];
    return enc;
}

- (NSString*) makeFilename {
    NSString* filename = [NSString stringWithFormat:@"capture%d.mp4", _currentFile];
    NSString* path = [NSTemporaryDirectory() stringByAppendingPathComponent:filename];
    return path;
}

- (void) initForHeight:(int)height andWidth:(int)width {
    _height = height;
    _width = width;
    NSString* path = [NSTemporaryDirectory() stringByAppendingPathComponent:@"params.mp4"];
    _headerWriter = [VideoEncoder encoderForPath:path Height:height andWidth:width];
    _times = [NSMutableArray arrayWithCapacity:10];
    
    // swap between 3 filenames
    _currentFile = 1;
    _writer = [VideoEncoder encoderForPath:[self makeFilename] Height:height andWidth:width];
}

- (void) encodeWithBlock:(encoder_handler_t) block onParams: (param_handler_t) paramsHandler {
    NSLog(@"Encode with block");
    _outputBlock = block;
    _paramsBlock = paramsHandler;
    _needParams = YES;
    _pendingNALU = nil;
    _firstpts = -1;
    _bitspersecond = 0;
}

- (BOOL) parseParams:(NSString*) path {
    NSFileHandle* file = [NSFileHandle fileHandleForReadingAtPath:path];
    struct stat s;
    fstat([file fileDescriptor], &s);
    
    MP4Atom* movie = [MP4Atom atomAt:0 size:s.st_size type:(OSType)('file') inFile:file];
    MP4Atom* moov = [movie childOfType:(OSType)('moov') startAt:0];
    MP4Atom* trak = nil;
    
    if (moov != nil) {
        for (;;) {
            trak = [moov nextChild];
            if (trak == nil) {
                break;
            }
            
            if (trak.type == (OSType)('trak')) {
                MP4Atom* tkhd = [trak childOfType:(OSType)('tkhd') startAt:0];
                NSData* verflags = [tkhd readAt:0 size:4];
                unsigned char* p = (unsigned char*)[verflags bytes];
                if (p[3] & 1) {
                    break;
                }else {
                    tkhd = nil;
                }
            }
        }
    }
    
    MP4Atom* stsd = nil;
    if (trak != nil) {
        MP4Atom* media = [trak childOfType:(OSType)('mdia') startAt:0];
        if (media != nil) {
            MP4Atom* minf = [media childOfType:(OSType)('minf') startAt:0];
            if (minf != nil) {
                MP4Atom* stbl = [minf childOfType:(OSType)('stbl') startAt:0];
                if (stbl != nil) {
                    stsd = [stbl childOfType:(OSType)('stsd') startAt:0];
                }
            }
        }
    }
    
    if (stsd != nil) {
        MP4Atom* avc1 = [stsd childOfType:(OSType)('avc1') startAt:8];
        if (avc1 != nil) {
            MP4Atom* esd = [avc1 childOfType:(OSType)('avcC') startAt:78];
            if (esd != nil) {
                // this is the avcC record that we are looking for
                _avcC = [esd readAt:0 size:esd.length];
                if (_avcC != nil) {
                    // extract size of length field
                    unsigned char* p = (unsigned char*)[_avcC bytes];
                    _lengthSize = (p[4] & 3) + 1;
                    return YES;
                }
            }
        }
    }
    
    return NO;
}

- (void) onParamsCompletion
{
    // the initial one-frame-only file has been completed
    // Extract the avcC structure and then start monitoring the
    // main file to extract video from the mdat chunk.
    if ([self parseParams:_headerWriter.path]) {
        if (_paramsBlock) {
            
            _paramsBlock(_avcC);
        }
        
        _headerWriter = nil;
        _swapping = NO;
        _inputFile = [NSFileHandle fileHandleForReadingAtPath:_writer.path];
        _readQueue = dispatch_queue_create("uk.co.gdcl.avencoder.read", DISPATCH_QUEUE_SERIAL);
        
        _readSource = dispatch_source_create(DISPATCH_SOURCE_TYPE_READ, [_inputFile fileDescriptor], 0, _readQueue);
        dispatch_source_set_event_handler(_readSource, ^{
            [self onFileUpdate];
        });
        dispatch_resume(_readSource);
    }
}

- (void)encodeFrame:(CMSampleBufferRef) sampleBuffer {
    @synchronized(self) {
        NSLog(@"Need params? %@", _needParams ? @"YES" : @"NO");
        if (_needParams) {
            NSLog(@"Here! %i", _needParams);
            // the avcC record is needed for decoding and it's not written to the file until
            // completion. We get round that by writing the first frame to two files; the first
            // file (containing only one frame) is then finished, so we can extract the avcC record.
            // Only when we've got that do we start reading from the main file.
            _needParams = NO;
            if ([_headerWriter encodeFrame:sampleBuffer]) {
                NSLog(@"Finish writing");
                [_headerWriter finishWithCompletionHandler:^{
                    [self onParamsCompletion];
                }];
            }
        }
    }
    
    // Get the PTS
    CMTime prestime = CMSampleBufferGetPresentationTimeStamp(sampleBuffer);
    
    double dPTS = (double)(prestime.value) / (prestime.timescale);
    
    //NSLog(@"vPTS = %llu / %u = %f", prestime.value, prestime.timescale, dPTS);
    
    NSNumber* pts = [NSNumber numberWithDouble:dPTS];
    
    // Add the PTS to times array
    @synchronized(_times) {
        [_times addObject:pts];
    }
    
    @synchronized(self) {
        // switch output files when we reach a size limit
        // to avoid runaway storage use.
        if (!_swapping) {
            struct stat st;
            fstat([_inputFile fileDescriptor], &st);
            if (st.st_size > OUTPUT_FILE_SWITCH_POINT) {
                _swapping = YES;
                VideoEncoder* oldVideo = _writer;
                
                // construct a new writer to the next filename
                if (++_currentFile > MAX_FILENAME_INDEX) {
                    _currentFile = 1;
                }
                
                NSLog(@"Swap to file %d", _currentFile);
                _writer = [VideoEncoder encoderForPath:[self makeFilename] Height:_height andWidth:_width];
                
                
                // to do this seamlessly requires a few steps in the right order
                // first, suspend the read source
                dispatch_source_cancel(_readSource);
                
                // execute the next step as a block on the same queue, to be sure the suspend is done
                dispatch_async(_readQueue, ^{
                    // finish the file, writing moov, before reading any more from the file
                    // since we don't yet know where the mdat ends
                    _readSource = nil;
                    [oldVideo finishWithCompletionHandler:^{
                        [self swapFiles:oldVideo.path];
                    }];
                });
            }
        }
        
        [_writer encodeFrame:sampleBuffer];
    }
}

- (void) swapFiles:(NSString*) oldPath {
    // save current position
    uint64_t pos = [_inputFile offsetInFile];
    
    // re-read mdat length
    [_inputFile seekToFileOffset:_posMDAT];
    NSData* hdr = [_inputFile readDataOfLength:4];
    unsigned char* p = (unsigned char*) [hdr bytes];
    int lenMDAT = to_host(p);

    // extract nalus from saved position to mdat end
    uint64_t posEnd = _posMDAT + lenMDAT;
    uint32_t cRead = (uint32_t)(posEnd - pos);
    [_inputFile seekToFileOffset:pos];
    [self readAndDeliver:cRead];
    
    // close and remove file
    [_inputFile closeFile];
    _foundMDAT = false;
    _bytesToNextAtom = 0;
    [[NSFileManager defaultManager] removeItemAtPath:oldPath error:nil];
    
    
    // open new file and set up dispatch source
    _inputFile = [NSFileHandle fileHandleForReadingAtPath:_writer.path];
    _readSource = dispatch_source_create(DISPATCH_SOURCE_TYPE_READ, [_inputFile fileDescriptor], 0, _readQueue);
    dispatch_source_set_event_handler(_readSource, ^{
        [self onFileUpdate];
    });
    dispatch_resume(_readSource);
    _swapping = NO;
}


- (void) readAndDeliver:(uint32_t) cReady {
    // Identify the individual NALUs and extract them
    while (cReady > _lengthSize) {
        NSData* lenField = [_inputFile readDataOfLength:_lengthSize];
        cReady -= _lengthSize;
        unsigned char* p = (unsigned char*) [lenField bytes];
        unsigned int lenNALU = to_host(p);
        
        if (lenNALU > cReady) {
            // whole NALU not present -- seek back to start of NALU and wait for more
            [_inputFile seekToFileOffset:[_inputFile offsetInFile] - 4];
            break;
        }
        
        NSData* nalu = [_inputFile readDataOfLength:lenNALU];
        cReady -= lenNALU;
        
        [self onNALU:nalu];
    }
}

- (void) onFileUpdate {
    // called whenever there is more data to read in the main encoder output file.
    struct stat s;
    fstat([_inputFile fileDescriptor], &s);
    int cReady = s.st_size - [_inputFile offsetInFile];
    
    // locate the mdat atom if needed
    while (!_foundMDAT && (cReady > 8)) {
        if (_bytesToNextAtom == 0) {
            NSData* hdr = [_inputFile readDataOfLength:8];
            cReady -= 8;
            unsigned char* p = (unsigned char*) [hdr bytes];
            int lenAtom = to_host(p);
            unsigned int nameAtom = to_host(p+4);
            if (nameAtom == (unsigned int)('mdat')) {
                _foundMDAT = true;
                _posMDAT = [_inputFile offsetInFile] - 8;
            }else {
                _bytesToNextAtom = lenAtom - 8;
            }
        }
        
        if (_bytesToNextAtom > 0) {
            int cThis = cReady < _bytesToNextAtom ? cReady :_bytesToNextAtom;
            _bytesToNextAtom -= cThis;
            [_inputFile seekToFileOffset:[_inputFile offsetInFile]+cThis];
            cReady -= cThis;
        }
    }
    
    if (!_foundMDAT) {
        return;
    }
    
    // the mdat must be just encoded video.
    [self readAndDeliver:cReady];
}

- (void) onEncodedFrame {
    double pts = 0;
    
    @synchronized(_times) {
        if ([_times count] > 0) {
            pts = [_times[0] doubleValue];
            [_times removeObjectAtIndex:0];
            
            if (_firstpts < 0) {
                _firstpts = pts;
            }
            
            if ((pts - _firstpts) < 1) {
                int bytes = 0;
                for (NSData* data in _pendingNALU) {
                    bytes += [data length];
                }
                _bitspersecond += (bytes * 8);
            }
        }else {
            //NSLog(@"no pts for buffer");
        }
    }
    
    if (_outputBlock != nil){
        _outputBlock(_pendingNALU, pts);
    }
}

// combine multiple NALUs into a single frame, and in the process, convert to BSF
// by adding 00 00 01 startcodes before each NALU.
- (void) onNALU:(NSData*) nalu {
    unsigned char* pNal = (unsigned char*)[nalu bytes];
    int idc = pNal[0] & 0x60;
    int naltype = pNal[0] & 0x1f;
    
    if (_pendingNALU) {
        NALUnit nal(pNal, [nalu length]);
        
        // we have existing data — is this the same frame?
        // typically there are a couple of NALUs per frame in iOS encoding.
        // This is not general-purpose: it assumes that arbitrary slice ordering is not allowed.
        BOOL bNew = NO;

        if ((idc != _prev_nal_idc) && ((idc * _prev_nal_idc) == 0)) {
            bNew = YES;
        }else if ((naltype != _prev_nal_type) && ((naltype == 5) || (_prev_nal_type == 5))){
            bNew = YES;
        }else if ((naltype >= 1) && (naltype <= 5)) {
            nal.Skip(8);
            int first_mb = nal.GetUE();
            if (first_mb == 0) {
                bNew = YES;
            }
        }
        
        // Process the existing data
        if (bNew) {
            [self onEncodedFrame];
            _pendingNALU = nil;
        }
    }
    
    // Set the previous NAL type and IDC
    _prev_nal_type = naltype;
    _prev_nal_idc = idc;
    
    // Create a new pendingNALU array
    if (_pendingNALU == nil) {
        _pendingNALU = [NSMutableArray arrayWithCapacity:2];
    }

    /*
    //This will prefix startcodes to the NALU

    int size = [nalu length] + 3;
    
    unsigned char *temp = (unsigned char*)malloc(size);
    
    uint8_t *data = (uint8_t*)malloc(3);
    data[0] = 0x00;
    data[1] = 0x00;
    data[2] = 0x01;
    
    memcpy(temp, data, 3);
    memcpy(temp + 3, pNal, [nalu length]);
    
    NSData *newNALU = [NSData dataWithBytes:temp length:size];
    */
    
    // Add the NALU data to the pendingNALU array
    [_pendingNALU addObject:nalu/*newNALU*/];
}

- (NSData*) getConfigData {
    return [_avcC copy];
}

- (void) shutdown {
    @synchronized(self)
    {
        _readSource = nil;
        if (_headerWriter) {
            [_headerWriter finishWithCompletionHandler:^{
                _headerWriter = nil;
            }];
        }
        if (_writer) {
            [_writer finishWithCompletionHandler:^{
                _writer = nil;
            }];
        }
        // !! wait for these to finish before returning and delete temp files
    }
}

@end
