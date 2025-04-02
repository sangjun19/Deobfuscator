// Repository: fletcher/MMD-Edit
// File: tester.m

/* PEG Markdown Highlight
 * Copyright 2011 Ali Rantakari -- http://hasseg.org
 * Licensed under the GPL2+ and MIT licenses (see LICENSE for more info).
 * 
 * tester.m
 * 
 * Test program that runs the parser on the given Markdown document and
 * highlights the contents using Cocoa's NSMutableAttributedString, outputting
 * the results as ANSI escape -formatted text into stdout.
 */

#include <sys/time.h>
#import <Foundation/Foundation.h>
#import "ANSIEscapeHelper.h"
#import "markdown_parser.h"


void apply_highlighting(NSMutableAttributedString *attrStr, element *elem[])
{
    unsigned long sourceLength = [attrStr length];
    
    int order[] = {
        H1, H2, H3, H4, H5, H6,  
        LINK,
        AUTO_LINK_URL,
        AUTO_LINK_EMAIL,
        IMAGE,
        HTML,
        EMPH,
        STRONG,
        COMMENT,
        CODE,
        LIST_BULLET,
        LIST_ENUMERATOR,
        VERBATIM,
        HTMLBLOCK,
        HRULE,
        REFERENCE,
        NOTE,
        HTML_ENTITY,
        BLOCKQUOTE,
    };
    int order_len = 24;
    
    int i;
    for (i = 0; i < order_len; i++)
    {
        //MKD_PRINTF("apply_highlighting: %i\n", i);
        
        element *cursor = elem[order[i]];
        while (cursor != NULL)
        {
            if (cursor->end <= cursor->pos)
                goto next;
            
            NSColor *fgColor = nil;
            NSColor *bgColor = nil;
            BOOL removeFgColor = NO;
            BOOL removeBgColor = NO;
            
            switch (cursor->type)
            {
                case H1:
                case H2:
                case H3:
                case H4:
                case H5:
                case H6:        fgColor = [NSColor blueColor]; break;
                case EMPH:      fgColor = [NSColor yellowColor]; break;
                case STRONG:    fgColor = [NSColor magentaColor]; break;
                case COMMENT:   fgColor = [NSColor blackColor]; break;
                case CODE:
                case VERBATIM:  fgColor = [NSColor greenColor]; break;
                case HTML_ENTITY:
                case HRULE:     fgColor = [NSColor cyanColor]; break;
                case REFERENCE: fgColor = [NSColor colorWithCalibratedHue:0.67 saturation:0.5 brightness:1.0 alpha:1.0]; break;
                case LIST_ENUMERATOR:
                case LIST_BULLET:fgColor = [NSColor magentaColor]; break;
                case AUTO_LINK_EMAIL:
                case AUTO_LINK_URL:fgColor = [NSColor cyanColor]; break;
                case IMAGE:
                case LINK:      bgColor = [NSColor blackColor];
                                fgColor = [NSColor cyanColor]; break;
                case BLOCKQUOTE:fgColor = [NSColor magentaColor]; break;
                default: break;
            }
            
            //MKD_PRINTF("  %i-%i\n", cursor->pos, cursor->end);
            if (fgColor != nil || bgColor != nil) {
                unsigned long rangePosLimitedLow = MAX(cursor->pos, (unsigned long)0);
                unsigned long rangePos = MIN(rangePosLimitedLow, sourceLength);
                unsigned long len = cursor->end - cursor->pos;
                if (rangePos+len > sourceLength)
                    len = sourceLength-rangePos;
                NSRange range = NSMakeRange(rangePos, len);
                
                if (removeBgColor)
                    [attrStr
                        removeAttribute:NSBackgroundColorAttributeName
                        range:range
                        ];
                else if (bgColor != nil)
                    [attrStr
                        addAttribute:NSBackgroundColorAttributeName
                        value:bgColor
                        range:range
                        ];
                
                if (removeFgColor)
                    [attrStr
                        removeAttribute:NSForegroundColorAttributeName
                        range:range
                        ];
                else if (fgColor != nil)
                    [attrStr
                        addAttribute:NSForegroundColorAttributeName
                        value:fgColor
                        range:range
                        ];
            }
            
            next:
            cursor = cursor->next;
        }
        
        elem[order[i]] = NULL;
    }
}



NSAttributedString *highlight(NSString *str, NSMutableAttributedString *attrStr)
{
    int extensions = 0;
    element **result;
    
    char *md_source = (char *)[str UTF8String];
    markdown_to_elements(md_source, extensions, &result);
    
    if (attrStr == nil)
        attrStr = [[[NSMutableAttributedString alloc] initWithString:str] autorelease];
    apply_highlighting(attrStr, result);
    
    free_elements(result);
    
    return attrStr;
}


void print_result(element *elem[])
{
    for (int i = 0; i < NUM_TYPES; i++)
    {
        element *cursor = elem[i];
        while (cursor != NULL)
        {
            MKD_PRINTF("[%ld-%ld] 0x%x: %s\n", cursor->pos, cursor->end, (int)cursor, type_name(cursor->type));
            cursor = cursor->next;
        }
    }
}

void Print(NSString *aStr)
{
    [aStr writeToFile:@"/dev/stdout" atomically:NO encoding:NSUTF8StringEncoding error:NULL];
}

double get_time()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec*1e-6;
}

int main(int argc, char * argv[])
{
    NSAutoreleasePool *autoReleasePool = [[NSAutoreleasePool alloc] init];
    
    if (argc == 1)
    {
        Print(@"Argument must be path to file\n");
        return(0);
    }
    
    ANSIEscapeHelper *ansiHelper = [[[ANSIEscapeHelper alloc] init] autorelease];
    NSString *filePath = [NSString stringWithUTF8String:argv[argc-1]];
    NSString *contents = [NSString stringWithContentsOfFile:filePath encoding:NSUTF8StringEncoding error:NULL];
    
    if (argc > 2)
    {
        if (strcmp(argv[1], "-d") == 0)
        {
            int extensions = 0;
            element **result;
            markdown_to_elements((char *)[contents UTF8String], extensions, &result);
            print_result(result);
        }
        else
        {
            int iterations = atoi(argv[1]);
            printf("Doing %i iterations.\n", iterations);
            
            NSAttributedString *attrStr = nil;
            
            NSMutableAttributedString *as[iterations];
            for (int j = 0; j < iterations; j++) {
                as[j] = [[[NSMutableAttributedString alloc] initWithString:contents] autorelease];
            }
            
            double starttime = get_time();
            int stepProgress = 0;
            for (int i = 0; i < iterations; i++)
            {
                attrStr = highlight(contents, as[i]);
                
                if (stepProgress == 9) {
                    Print([NSString stringWithFormat:@"%i", i+1]);
                    stepProgress = 0;
                } else {
                    Print(@"-");
                    stepProgress++;
                }
            }
            double endtime = get_time();
            
            //Print([ansiHelper ansiEscapedStringWithAttributedString:attrStr]);
            printf("\n%f\n", (endtime-starttime));
        }
    }
    else
        Print([ansiHelper ansiEscapedStringWithAttributedString:highlight(contents, nil)]);
    
    [autoReleasePool release];
    return(0);
}
