// Repository: andrewwiik/Tweaks
// File: decache/decache.mm


#include "common.h"

#include "reexport.h"

#ifdef TARGET_IPHONE
//	#define READ_VM
#endif

static inline int fsize(int fd)
{
	struct stat results;
    if (fstat(fd, &results) == 0)
	{
		if(results.st_mode & S_IFREG)
			return results.st_size;
	}
    return 0;
}



/*
	0x0BEAE000	// codesign table offset__
	0x0
	0x0
	0x0
	0x0BE5B000
	0x0
	0x00053000
	0x0


*/
/*
struct cache_header {
        char version[16];
        uint32_t baseaddroff;
        uint32_t unk2;
        uint32_t startaddr;
        uint32_t numlibs;

        uint64_t dyldaddr;
        //uint64_t codesignoff;
};*/

struct dyld_cache_header {
	char magic[16];
	uint32_t mappingOffset;
	uint32_t mappingCount;
	uint32_t imagesOffset;
	uint32_t imagesCount;
	uint64_t dyldBaseAddress;
	uint64_t codeSignatureOffset;
	uint64_t codeSignatureSize;
	uint64_t slideInfoOffset;
	uint64_t slideInfoSize;
	uint64_t localSymbolsOffset;
	uint64_t localSymbolsSize;
};


struct dyld_cache_mapping_info {
	uint64_t address;
	uint64_t size;
	uint64_t fileOffset;
	uint32_t maxProt;
	uint32_t initProt;
};

struct dyld_cache_image_info {
	uint64_t address;
	uint64_t modTime;
	uint64_t inode;
	uint32_t pathFileOffset;
	uint32_t pad;
};

struct dyld_cache_slide_info {
	uint32_t version;
	uint32_t toc_offset;
	uint32_t toc_count;
	uint32_t entries_offset;
	uint32_t entries_count;
	uint32_t entries_size;
	// uint16_t toc[toc_count];
	// entrybitmap entries[entries_count];
};

typedef struct _dyld_cache_local_symbols_info {
    uint32_t nlistOffset;
    uint32_t nlistCount;
    uint32_t stringsOffset;
    uint32_t stringsSize;
    uint32_t entriesOffset;
    uint32_t entriesCount;
} dyld_cache_local_symbols_info;

typedef struct _dyld_cache_local_symbols_entry {
    uint32_t dylibOffset;
    uint32_t nlistStartIndex;
    uint32_t nlistCount;
} dyld_cache_local_symbols_entry;


struct dyld_cache_slide_info_entry {
	uint8_t bits[4096/(8*4)];
};


#define SEG_DATA_CONST "__DATA_CONST\0\0\0\0"
#define SEG_DATA_DIRTY "__DATA_DIRTY\0\0\0\0"



uint32_t shared_cache_slide;
uintptr_t dyld_buf;
struct dyld_cache_header *dyldHead = NULL;
uint64_t dyld_vmbase = 0;
uint64_t dyld_vmextent = 0;
dyld_cache_image_info* image_infos = NULL;
dyld_cache_local_symbols_info *localSymbols = NULL;


uintptr_t locate_address(uint32_t addr, bool printSource = 0, uintptr_t* xbuf_ret = 0)
{
	for(uint32_t i=0; i< dyldHead->imagesCount; i++)
	{
		uint64_t vm_address = image_infos[i].address;
		
		uintptr_t xbuf = dyld_buf + vm_address - dyld_vmbase;
		
		
		mach_header* header = (mach_header*) xbuf;

		if(header->magic != 0xfeedface)
		{
			PANIC("Magic does not match");
		}
		
		int ncmds = header->ncmds;
		intptr_t lcptr = xbuf + sizeof(mach_header);
		for(int j=0; j<ncmds; j++)
		{
			uint32_t cmd = ((load_command*) lcptr)->cmd;

			if(cmd == LC_SEGMENT)
			{
				segment_command* seg = (segment_command*) lcptr;
				if(addr >= seg->vmaddr + shared_cache_slide && addr < seg->vmaddr + seg->vmsize + shared_cache_slide)
				{
					uintptr_t base = 0;
					if(!strncmp(seg->segname, SEG_TEXT, 16))
					{
						base = xbuf;
					}
					else if(!strncmp(seg->segname, SEG_DATA, 16) || !strncmp(seg->segname, SEG_LINKEDIT, 16)
						 || !strncmp(seg->segname, SEG_DATA_CONST, 16) || !strncmp(seg->segname, SEG_DATA_DIRTY, 16))
					{
						base = dyld_buf + seg->fileoff;
					}
					else
					{
						PANIC("Unhandled section type!");
					}
					if(printSource)
					{
						const char *filename = (const char *)(dyld_buf + image_infos[i].pathFileOffset);
						
						section *sect = (section *)((uintptr_t)seg + sizeof(segment_command));
						for(uint32_t k = 0; k<seg->nsects; k++)
						{
							if(addr > sect[k].addr && addr < sect[k].addr + sect[k].size)
							{
							//	const char *offset = sect[k].offset+dylbuf+ (addr - sect[k].addr);
								CommonLog("Found at %s : %.16s : %.16s (+%08x)", filename, seg->segname, sect[k].sectname, addr-seg->vmaddr);
								
							//	cout << filename << ": " << sect[k].segname <<"\t"<< sect[k].sectname << "\t";
							//	return offset;
							}
						}
						//CommonLog("Found at %s : %.16s", filename, seg->segname);
					}
					if(xbuf_ret)
						*xbuf_ret = xbuf;
					return base + addr - seg->vmaddr - shared_cache_slide;
					
				}
				//print_segment(seg);
			}
			lcptr += ((load_command*) lcptr)->cmdsize;
		}
	}
	if(printSource)
	{
		CommonLog("Could not locate %x", addr);
	}
	return NULL;
}

segment_command* find_segment(uintptr_t fbuf, const char* name)
{
	mach_header* header = (mach_header*) fbuf;
	
	uint32_t ncmds = header->ncmds;
	//uint32_t sizeofcmds = header->sizeofcmds;
	uintptr_t cmd_base = fbuf + sizeof(mach_header);
	
	
	// first scan: make sure we have space
	{
		//uint32_t lowestOffset;
		
		uintptr_t lcptr = cmd_base;
		for(uint32_t i=0; i<ncmds; i++)
		{
			uint32_t cmd = ((load_command*) lcptr)->cmd;
		
			if(cmd == LC_SEGMENT)
			{
				segment_command* seg = (segment_command*) lcptr;
				if(!strncmp(seg->segname, name, 16))
					return seg;
			}
			lcptr += ((load_command*) lcptr)->cmdsize;
			
		}
	}
	return NULL;
}

load_command* find_command(uintptr_t fbuf, uint32_t matching_cmd)
{
	mach_header* header = (mach_header*) fbuf;
	
	uint32_t ncmds = header->ncmds;
	//uint32_t sizeofcmds = header->sizeofcmds;
	uintptr_t cmd_base = fbuf + sizeof(mach_header);
	
	
	// first scan: make sure we have space
	{
		//uint32_t lowestOffset;
		
		uintptr_t lcptr = cmd_base;
		for(uint32_t i=0; i<ncmds; i++)
		{
			uint32_t cmd = ((load_command*) lcptr)->cmd;
		
			if(cmd == matching_cmd)
			{
				return (load_command*)lcptr;
			}
			lcptr += ((load_command*) lcptr)->cmdsize;
			
		}
	}
	return NULL;
}


static inline bool in_dyld_cache(uint32_t addr)
{
	return (addr >= dyld_vmbase + shared_cache_slide) && (addr < dyld_vmextent + shared_cache_slide);
}

const dyld_cache_image_info* find_file(const char* fname, uint32_t* offset)
{
	for(uint32_t i=0; i< dyldHead->imagesCount; i++)
	{
		uint64_t vm_address = image_infos[i].address;
		
		const char *filename = (const char *)(dyld_buf + image_infos[i].pathFileOffset);
		
		if(strstr(filename, fname))
		//if(strcmp(filename, fname)==0)
		{
			if(offset)
				*offset = vm_address - dyld_vmbase;
			return &image_infos[i];
		}
	}
	return NULL;
}

/*
void print_vmaddr(int dyld_n)
{
	struct dyld_cache_header *dyldHead = (dyld_cache_header *)dyld_buf;
	
	uint64_t base_mapping = *(uint64_t *)(dyld_buf + dyldHead->mappingOffset);

	dyld_cache_image_info* infos = (dyld_cache_image_info*) (dyld_buf + dyldHead->imagesOffset);
	
	for(uint32_t i=0; i< dyldHead->imagesCount; i++)
	{
		uint64_t vm_address = infos[i].address;
		
		const char *filename = (const char *)(dyld_buf + infos[i].pathFileOffset);
		
		uint64_t file_offset = vm_address - base_mapping;
	
		CommonLog("%llx (%llx): %s", vm_address, file_offset, filename);
		
		/ *
		uint64_t mo = la - base_mapping;

		uint32_t f_cmds_count = *(uint32_t *)(dylbuf+mo+0x10);

		int fptr = mo+0x1c;

		for(uint32_t j=0; j<f_cmds_count; j++, fptr+=*(uint32_t *)(dylbuf+fptr+4))
		{
			if(*(uint32_t *)(dylbuf+fptr)==LC_SEGMENT)
			{
				segment_command* cmd = (segment_command *)(dylbuf+fptr);
				if(cmd->vmaddr < vmaddr && cmd->vmaddr + cmd->vmsize > vmaddr)
				{
					section *sect = (section *)(uint64_t)(dylbuf+fptr + sizeof(segment_command));
					for(uint32_t k = 0; k<cmd->nsects; k++)
					{
						if(vmaddr > sect[k].addr && vmaddr < sect[k].addr + sect[k].size)
						{
							const char *offset = sect[k].offset+dylbuf+ (vmaddr - sect[k].addr);
							
							// WHY??
							if(!strcmp(sect[k].segname, "__TEXT"))
							{
								offset+=mo;
							}
							cout << filename << ": " << sect[k].segname <<"\t"<< sect[k].sectname << "\t";
						//	return offset;
						}
					}
				}
			}
		}
		* /
	}
}
*/

const char* section_types[] = 
{
	"S_REGULAR",
	"S_ZEROFILL",
	"S_CSTRING_LITERALS",
	"S_4BYTE_LITERALS",
	"S_8BYTE_LITERALS",
	"S_LITERAL_POINTERS",
	"S_NON_LAZY_SYMBOL_POINTERS",
	"S_LAZY_SYMBOL_POINTERS",
	"S_SYMBOL_STUBS",
	"S_MOD_INIT_FUNC_POINTERS",
	"S_MOD_TERM_FUNC_POINTERS",
	"S_COALESCED",
	"S_GB_ZEROFILL",
	"S_INTERPOSING",
	"S_16BYTE_LITERALS",
	"S_DTRACE_DOF",
	"S_LAZY_DYLIB_SYMBOL_POINTERS",
	"S_THREAD_LOCAL_REGULAR",
	"S_THREAD_LOCAL_ZEROFILL",
	"S_THREAD_LOCAL_VARIABLES",
	"S_THREAD_LOCAL_VARIABLE_POINTERS",
	"S_THREAD_LOCAL_INIT_FUNCTION_POINTERS",
};


void print_segment(segment_command* seg)
{
	CommonLog("%.16s segment: %x %x %x %x",
		seg->segname, seg->fileoff, seg->filesize, seg->vmaddr, seg->vmsize);
	if(seg->nsects)
	{
		section* sects = (section*) ((uintptr_t) seg + sizeof(segment_command));
		for(uint32_t i=0; i<seg->nsects; i++)
		{
			section* sect = &sects[i];
			CommonLog("%.16s has section %.16s (%s) (+%x) (%x+%x)",
				sect->segname, sect->sectname, section_types[sect->flags & SECTION_TYPE],
				sect->addr, sect->offset, sect->size);
			if(sect->reloff || sect->nreloc)
			{
				CommonLog("Relocation offset: %x %x", sect->reloff, sect->nreloc);
			}
		}
	}	
}


int fdcreate(const char* name, int nfile, uintptr_t* buf)
{
	int fd = open(name, O_RDWR | O_CREAT);
	
	fchmod(fd, S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
	
	ftruncate(fd, nfile);
	*buf = (uintptr_t) mmap(NULL, nfile, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	memset((void*)*buf, 0, nfile);

//	CommonLog("fd = %d", fd);

	return fd;
}

void fdresize(int fd, int nfile, uintptr_t* buf)
{
	int oldn = fsize(fd);
	munmap((void*) *buf, oldn);
	
	ftruncate(fd, nfile);
	*buf = (uintptr_t) mmap(NULL, nfile, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
}

void fdclose(int fd, uintptr_t* buf)
{
	int oldn = fsize(fd);
	munmap((void*) *buf, oldn);
	close(fd);
}

void fdtrimclose(int fd, int nfile, uintptr_t* buf)
{
//	CommonLog("fd = %d", fd);
	int oldn = fsize(fd);
	munmap((void*) *buf, oldn);
	ftruncate(fd, nfile);
//	CommonLog("closing fd = %d", fd);
	close(fd);
}


static inline int vmalign(int addr)
{
	int page = 0x1000;
	return (addr & (~(page-1))) + (( addr & (page-1)) ? page : 0);
}


/*
uint32_t dvmaddr_text;
uint32_t dvmaddr_data;

segment_command* oldtext;
segment_command* olddata;

bool resolve_local(uint32_t* ptr)
{
	uintptr_t pointer = *ptr;
	if(!pointer)
		return false;
	if(pointer >= oldtext->vmaddr && pointer < oldtext->vmaddr + oldtext->vmsize)
	{
		*ptr += dvmaddr_text;
		return true;
	}
	else if(pointer >= olddata->vmaddr && pointer < olddata->vmaddr + olddata->vmsize)
	{
		*ptr += dvmaddr_data;
		return true;
	}
	return false;
}
*/

struct seg_range
{
	uint32_t start;
	uint32_t end;
};


struct seg_info
{
	seg_range offset;
	seg_range vmaddr;
	uintptr_t buf;
};

struct seg_adjust
{
	seg_info old;
	seg_info fix;
	uint32_t dvmaddr;
	uint32_t doffset;
};


void seg_applyoffset(segment_command* seg, seg_adjust* adjust)
{
	if(!seg)
		return;
	
	uint32_t dvmaddr = adjust->dvmaddr;
	uint32_t doffset = adjust->doffset;
//	CommonLog("Adjust: %x %x", dvmaddr, doffset);
	
//	CommonLog("old vmaddr/offset: %x %x", seg->vmaddr, seg->fileoff);

	seg->vmaddr += dvmaddr;
	if(seg->fileoff)
		seg->fileoff += doffset;
	
//	CommonLog("new vmaddr/offset: %x %x", seg->vmaddr, seg->fileoff);
	int nsects = seg->nsects;
	section* sects = (section*) ((uintptr_t) seg + sizeof(segment_command));
	for(int i=0; i<nsects; i++)
	{
//		CommonLog("Old offset: %x %x", sects[i].addr, sects[i].offset);

		sects[i].addr += dvmaddr;
		if(sects[i].offset)
			sects[i].offset += doffset;
//		CommonLog("New offset: %x %x", sects[i].addr, sects[i].offset);
	}	
}


bool seg_inoutputs(uint32_t value, seg_adjust* adjust, int nAdjust)
{
	for(int i=0; i<nAdjust; i++)
	{
		if(value >= adjust[i].fix.offset.start && value < adjust[i].fix.offset.end)
			return true;
	}
	return false;
}

/*
bool seg_inoutput(uint32_t value, seg_adjust* adjust)
{
//	if(value >= adjust->fix.offset.start && value < adjust->fix.offset.end)
//		return true;
//	return false;
	return seg_inoutputs(value, adjust, 1);
}*/



bool seg_virtresolves(uint32_t* value, seg_adjust* adjust, int nAdjust, bool noslide = 0)
{
	for(int i=0; i<nAdjust; i++)
	{
		if(*value >= adjust[i].old.vmaddr.start  + (noslide ? 0 : shared_cache_slide)
		 && *value < adjust[i].old.vmaddr.end + (noslide ? 0 : shared_cache_slide))
		{
			*value += adjust[i].dvmaddr - (noslide ? 0 : shared_cache_slide);
			return true;
		}
	}
	return false;
}



struct rebase_info
{
	uintptr_t buf;
	uintptr_t currptr;
	uint32_t nBuf;
	
	uintptr_t dataseg_base;	// dseg->fix.buf
	
//	uint8_t type;
//	uint8_t segIndex;
	uint32_t offset;
	uint32_t delta;
	uint32_t count;
};


void write_bind(rebase_info* rebase)
{
	if(rebase->count == 0)
		return;
	if(rebase->count > 1)
	{
		if(rebase->delta > 4)
		{
			*((uint8_t*)rebase->currptr) = REBASE_OPCODE_DO_REBASE_ULEB_TIMES_SKIPPING_ULEB;
			rebase->currptr++;
			rebase->currptr += append_uleb((char*)rebase->currptr, rebase->count);
			rebase->currptr += append_uleb((char*)rebase->currptr, rebase->delta - 4);

		}
		else if(rebase->count < 16)
		{
			*((uint8_t*)rebase->currptr) = REBASE_OPCODE_DO_REBASE_IMM_TIMES | rebase->count;
			rebase->currptr++;
		}
		else
		{
			*((uint8_t*)rebase->currptr) = REBASE_OPCODE_DO_REBASE_ULEB_TIMES;
			rebase->currptr++;
			rebase->currptr += append_uleb((char*)rebase->currptr, rebase->count);
		}
		
		rebase->offset = rebase->offset + rebase->delta * rebase->count;
		rebase->count = 0;
		rebase->delta = 0;
	}
	else
	{
		if(rebase->delta > 4)
		{
			*((uint8_t*)rebase->currptr) = REBASE_OPCODE_DO_REBASE_ADD_ADDR_ULEB;
			rebase->currptr++;
			rebase->currptr += append_uleb((char*)rebase->currptr, rebase->delta - 4);
			
			
			
			rebase->offset += rebase->delta;
			rebase->delta = 0;
			rebase->count = 0;
		}
		else if(rebase->delta == 4 || rebase->delta == 0)
		{
			*((uint8_t*)rebase->currptr) = REBASE_OPCODE_DO_REBASE_IMM_TIMES | 1;
			rebase->currptr++;
			
			rebase->offset += 4;
			rebase->count --;
			rebase->delta = 0;
		}
		else
		{
			PANIC("WHAT?  GET AWAY FROM HERE!!");
		}
	}
}

void push_rebase_entry(rebase_info* rebase, uintptr_t entry)
{
	if(!rebase->buf)
	{
		rebase->buf = (uintptr_t) malloc(0x8000); // 32k should be plenty, right?
		rebase->currptr = rebase->buf;
		rebase->nBuf = 0x8000;
		*((uint8_t*)rebase->currptr) = REBASE_OPCODE_SET_TYPE_IMM | REBASE_TYPE_POINTER;
		rebase->currptr++;
	}
	else if(rebase->currptr - rebase->buf > rebase->nBuf - 0x10)
	{
		int nBuf_new = rebase->nBuf + 0x8000; // + 32k
		uintptr_t newbuf = (uintptr_t) malloc(nBuf_new);
		memcpy((void*) newbuf, (void*)rebase->buf, rebase->nBuf);
		rebase->nBuf = nBuf_new;
		free((void*)rebase->buf);
		
		rebase->currptr = newbuf + (rebase->currptr - rebase->buf);
		rebase->buf = newbuf;
	}
	
	
	uint32_t addr = entry ? (entry - rebase->dataseg_base) : 0;
	//uint32_t next = rebase->offset + rebase->delta * (rebase->count + 1);
	uint32_t next = rebase->offset + rebase->delta * rebase->count;
	
	// NEED MAX ENTRY SCAN
	
	if(addr)
	{
		if(addr < rebase->offset || !rebase->offset)
		{
			// save out the info.  this only matters if rebase->count > 1
			write_bind(rebase);
			
			
			// write new base address
			*((uint8_t*)rebase->currptr) = REBASE_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB | 1;
			rebase->currptr++;
			rebase->currptr += append_uleb((char*)rebase->currptr, addr);
			
			
			rebase->offset = addr;
			rebase->count = 1;
			rebase->delta = 0;
		}
		else if(!rebase->delta)
		{
			// no delta info; add our own!
			rebase->delta = addr - rebase->offset;
			rebase->count++;
		}
		else if(addr == next)
		{
			// increment counter
			rebase->count++;
		}
		else //if(addr < next) // skip is smaller than planned skip
		{
			// decrement by one
			// save info
			rebase->count --;
			write_bind(rebase);
			rebase->delta = addr - rebase->offset;
			rebase->count = 2;
		}
	}
	else
	{
		write_bind(rebase);
	}
	*((uint8_t*)rebase->currptr) = 0;
	
	
	/*
	
	{
		// write new base address
		*((uint8_t*)rebase->currptr) = REBASE_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB | 1;
		rebase->currptr++;
		rebase->currptr += append_uleb((char*)rebase->currptr, addr);
	
			
		*((uint8_t*)rebase->currptr) = REBASE_OPCODE_DO_REBASE_IMM_TIMES | 1;
		rebase->currptr++;
	}
	*((uint8_t*)rebase->currptr) = 0;
	
	*/
}



/*
bool seg_fileresolve(uint32_t* value, seg_adjust* adjust)
{
	if(*value >= adjust->old.offset.start && *value < adjust->old.offset.end)
	{
		*value += adjust->doffset;
		return true;
	}
	return false;
}
*/


void text_fixlongcalls(uintptr_t fbuf, section* text, section* pss, section* lsp, seg_adjust* tseg, seg_adjust* dsegs, int ndsegs)
{
//	CommonLog("fixlongcalls: %8s %08x %08x", text->sectname, text->offset, text->size);
//	CommonLog("fixlongcalls: %8s %08x %08x", pss->sectname, pss->offset, pss->size);
	
	
	
	
	/*
	CommonLog("fixlongcalls: %8s %08x %08x", sth->sectname, sth->offset, sth->size);
	
	int nsth = 0;
	
	uint32_t sths[sth->size / 12];
	
	
	uintptr_t stbase = fbuf + sth->offset;

	for(int i=0; i<sth->size / 4; i++)
	{
		// LDR R12, (PC+0)
		if(*(uint32_t*)(stbase + i*4) == 0xe59fc000)
		{
			// B PC-...
			uint32_t jmpz = ((-(i+3)) & 0xFFFFFF) | (0xea << 24);
			if(*(uint32_t*)(stbase + (i+1)*4) == jmpz)
			{
			//	CommonLog("%x", sth->offset + i*4);
				sths[nsth] = sth->offset + i*4;
				nsth++;
				//exit(1);
			}
			else
			{
				CommonLog("%x %x", i, *(uint32_t*)(stbase + (i+1)*4));
				exit(1);
			}
			
			// ea ff ff f1
			
		}
	}
	*/
	//CommonLog("Found %d stub helpers\n", nsth);
	//exit(1);
	
	
	
	
	// identify all the dest addresses for the pss
	int npss = pss->size / 16;
	
	uintptr_t psbase = fbuf + pss->offset;
	uintptr_t psvm = tseg->old.vmaddr.start + pss->addr;
	
	uint32_t ptrs[npss];
	for(int i=0; i<npss; i++)
	{
		// this is the easiest match
		if(  *(uint32_t*)(psbase + i*16 + 12) == 0xE7FFDEFE)
		{
			uint32_t offs = *(uint32_t*)(psbase + i*16 + 8);
			
			offs += pss->addr + i*16 + 12;
			//CommonLog("pointer to %x", offs);
			ptrs[i]=offs + tseg->old.vmaddr.start;
			//*(uint32_t*)(psbase + i*16 + 12) =0;
		}
		else
		{
			uint32_t offs = *(uint32_t*)(psbase + i*16 + 12);
			
			offs += pss->addr + i*16 + 12;
			
			offs += tseg->old.vmaddr.start;
			
			if(seg_virtresolves(&offs, tseg, 1))
			{
			//	CommonLog("pointer to %x (%x)", offs, *(uint32_t*)(fbuf + offs));
				
			}
			else if(seg_virtresolves(&offs, dsegs, ndsegs))
			{
			//	CommonLog("pointer to %x (%x)", offs, *(uint32_t*)(fbuf + offs));
				
				offs -= (pss->addr + i*16 + 12);
				*(uint32_t*)(psbase + i*16 + 12) = offs;
			}
			else
			{
				exit(1);
			}
		}
		
		
		// rewrite pss
			
		*(uint32_t*)(psbase + i*16 + 0) = 0xE59FC004;
		*(uint32_t*)(psbase + i*16 + 4) = 0xE08FC00C;
		*(uint32_t*)(psbase + i*16 + 8) = 0xE59CF000;
		*(uint32_t*)(psbase + i*16 + 12) = lsp->addr + i*4-(pss->addr + i*16 + 12);
		
	}
		
	
	
	
	
	// 01 F0 EA 32
	// EA32F010
	
	
	uintptr_t start = fbuf + text->offset;
	uintptr_t end = start + text->size;
	
	uintptr_t pcbase_old = -fbuf + 4 + tseg->old.vmaddr.start + shared_cache_slide;
	uintptr_t pcbase_act = -fbuf + 4 + tseg->fix.vmaddr.start;
	
	for(uintptr_t data = start; data < end; data+=2)
	{
		uint16_t op = *(uint16_t*)data;
		
		// F64C30DA
		// F6C010CC
		
		
		if((op & 0xF800) > 0xE7FF)
		{
			
			
			
			// THUMB32
			uint32_t op32 = (op << 16) | *(uint16_t*)(data+2);
			
			// bl, blx (bit 12?).  Should have no BLX's, but just in case...
			if((op32 & 0xF800E800) == 0xF000E800)
			{
				// we already did + 2 here...hence +4 from base
				int delta = ((op32 & 0x7FF0000) >> 4) | ((op32 & 0x7FF) << 1) + 4;
				
				if(delta & (1<<22))
				{
					delta |= 0xFF800000;
				}
				 // 0x24F48
				
				//if(delta + data-fbuf >= start && delta + data < end)
				//	continue;
				uint32_t addr = tseg->old.vmaddr.start + delta + (data-fbuf);
				
				if((op32 & 0x1000)==0)
				{
					addr &= ~0x3;
					delta = addr - tseg->old.vmaddr.start - (data-fbuf);
				}
				
				if(addr >= tseg->old.vmaddr.start && addr < tseg->old.vmaddr.end)
					continue;
				
				//CommonLog("Found BL(X)! %08x (%08x)", addr - tseg->old.vmaddr.start, data-fbuf);
				//CommonLog("Looking for addr... %x", addr);
				
				// 22661 28e0
				// 24f48
				// 
				
				//locate_address(addr, 0);
				
				
				int i;
				{
					for(i=0; i<npss; i++)
					{
						if(ptrs[i] && (ptrs[i] & (~1)) == addr)
						{
							uint32_t dest = pss->offset + i*16 - (data-fbuf) -4;
							
							*(uint16_t*)(data) = 0xF000 | ((dest & 0x7FF000) >> 12);
							*(uint16_t*)(data+2) = 0xE800 | ((dest & 0xFFE) >> 1);
							
							//CommonLog("hit on entry %d.  patching to +%08x... (%08x)\n", i, dest, (*(uint16_t*)data << 16) |  *(uint16_t*) (data+2));
							
							// BLX +dest to the stub
							break;
						}
					}
					if(i==npss)
					{
						
						//if((data - fbuf) == 0x125e)
						//{
							//PANIC("addr = %08x %08x %08x", delta + (data-fbuf), delta, data-fbuf);
						//}
						
						
						const char* matching_str = 0;
						{
							uintptr_t xbuf;
							locate_address(addr, 0, &xbuf);
							
							
							symtab_command* sym = (symtab_command*) find_command(xbuf, LC_SYMTAB);
							dysymtab_command* dsym = (dysymtab_command*) find_command(xbuf, LC_DYSYMTAB);
							
							
							int nsyms = sym->nsyms;
							struct nlist* nl = (struct nlist*) (dyld_buf + sym->symoff);
							const char* strs = (const char*) (dyld_buf + sym->stroff);
												
							for(int j=dsym->iextdefsym; j < dsym->iextdefsym + dsym->nextdefsym; j++)
							{
								const char* str = &strs[ nl[j].n_un.n_strx];
								uint32_t v = nl[j].n_value;
								if(v == addr)
								{
									matching_str = str;
								//	CommonLog("p=%08lx v=%08x, str=%s", data-fbuf, nl[j].n_value, str);
									break;
								}
							}
						}
						
						//if((data - fbuf) == 0x125e)
						//{
						//	PANIC("addr = %s", matching_str);
						//}
						
						
						{
							symtab_command* sym = (symtab_command*) find_command(fbuf, LC_SYMTAB);
							dysymtab_command* dsym = (dysymtab_command*) find_command(fbuf, LC_DYSYMTAB);
							
							//S_SYMBOL_STUBS							
							
							int nsyms = sym->nsyms;
							uintptr_t dsymoff = (uintptr_t)(dyld_buf + dsym->indirectsymoff);
							
							/*
							CommonLog("indirectsymoff = %08x", dsym->indirectsymoff);
							for(int j=0; j<512; j++)
							{
								fprintf(stderr, "%02x", *(uint8_t*)(dsymoff+j));
								if(((j+1)&7) == 0)
									if(((j+1)&31) == 0)
										fprintf(stderr, "\n");
									else
										fprintf(stderr, " ");
							}
							*/
							struct nlist* nl = (struct nlist*) (dyld_buf + sym->symoff);//sym->symoff);
							const char* strs = (const char*) (dyld_buf + sym->stroff);
							
							//dyld_buf + 
							
							int k;
							for(k=0; k < dsym->nindirectsyms; k++)
							{
								int j = *(uint32_t*)(dsymoff+k*4);
								if(j & 0xC0000000)
									continue;
								//CommonLog("j=%x", j);
								uint32_t v = nl[j].n_value;
								const char* str = &strs[ nl[j].n_un.n_strx];
								//const char* str = &strs[v];
								//
								
								if(str == matching_str)
								{
								//	matching_str = str;
								//	CommonLog("%d %x %s (%x)", j, v, str, k);
									break;
								}
							}
							if(k < dsym->nindirectsyms)
							{
								uint32_t dest = pss->offset + k*16 - (data-fbuf) -4;
								
								*(uint16_t*)(data) = 0xF000 | ((dest & 0x7FF000) >> 12);
								*(uint16_t*)(data+2) = 0xE800 | ((dest & 0xFFE) >> 1);
								
							}
							else
							{
								// this got triggered by two double instructions back-to-back!  lol!
								//PANIC("addr = %08x %08x %08x", delta + (data-fbuf), delta, data-fbuf);
							}
							
						}
						//exit(1);
						
						//segment_command* find_segment(uintptr_t fbuf, const char* name)
						//load_command* find_command(uintptr_t fbuf, uint32_t matching_cmd)
						
						//find_file(extractname, &extract_offs, &xbuf);
						
						
						//uintptr_t extract_buf = dyld_buf + extract_offs;
						//PANIC("Unresolved jump!");
					}
				}

				//exit(1);
			}
			
		}
	}
}

void text_dellvm(uintptr_t fbuf, section* sect, seg_adjust* tseg, seg_adjust* dsegs, int ndsegs)
{
//	CommonLog("dellvm: %8s %p %08x %08x", sect->sectname, sect-fbuf, sect->offset, sect->size);
	
	
	
	// thumb
	// 0x149e + 4
	// d3 f7 41 fb
	// BL              0xFFFD4B24
	// f7d3 fb41  BL SX(23'h7D3682)
	//7d3 << 12 FFFD3682
	//341 << 1  FFFD3682
	
	
	uintptr_t start = fbuf + sect->offset;
	uintptr_t end = start + sect->size;
	
	uintptr_t movw[16];
	uintptr_t movt[16];
	
	//uint32_t val[16];
	//uintptr_t addpc[16];
	
	memset(movw, 0, 16*sizeof(uintptr_t));
	memset(movt, 0, 16*sizeof(uintptr_t));
	//memset(addpc, 0, 16*sizeof(uintptr_t));
	
	uintptr_t pcbase_old = -fbuf + 4 + tseg->old.vmaddr.start + shared_cache_slide;
	uintptr_t pcbase_act = -fbuf + 4 + tseg->fix.vmaddr.start;
	
	for(uintptr_t data = start; data < end; data+=2)
	{
		uint16_t op = *(uint16_t*)data;
		
		// F64C30DA
		// F6C010CC
		
		
		if((op & 0xF800) > 0xE7FF)
		{
			// THUMB32
			data +=2;
			uint32_t op32 = (op << 16) | *(uint16_t*)data;
			//CommonLog("32 bit at %p: %x %x", data - fbuf, op32, op32 & 0xFBF08000);
						
			if((op32 & 0xFBF08000) == 0xF2400000) // T3 encoding
			{
				// MOVW
				int rX = (op32 & 0xF00) >> 8;
				movw[rX] = data-2;
			}
			else if((op32 & 0xFBF08000) == 0xF2C00000)
			{
				// MOVT
				int rX = (op32 & 0xF00) >> 8;
				movt[rX] = data-2;
			}
			else
			{
				data -=2;
			}
		}
		else
		{
			if((op & 0xFF78) == 0x4478)
			{
				int rX = op & 0x7 | ((op & 0x80) ? 0x8 : 0);
				
				uintptr_t lptr = movw[rX];
				uintptr_t hptr = movt[rX];
				
				if(lptr < data - 0x100 || hptr < data - 0x100)
					continue;
				//	PANIC("Out of range! %p=%x %p %p", data - fbuf, op, lptr - fbuf, hptr - fbuf);
				
				uint32_t lop = *(uint32_t*) lptr;
				uint32_t hop = *(uint32_t*) hptr;
				
				//CommonLog("%x %x", lop, hop);
				uint16_t lval = (lop & 0xF) << 12
							  | (lop & 0x400) << 1
							  | (lop & 0x70000000) >> 20
							  | (lop & 0xFF0000) >> 16;
				uint16_t hval = (hop & 0xF) << 12
							  | (hop & 0x400) << 1
							  | (hop & 0x70000000) >> 20
							  | (hop & 0xFF0000) >> 16;
				
				uint32_t pcorig = data + pcbase_old;
				
				uint32_t pcact = data + pcbase_act;
				
				uint32_t ptr = hval << 16 | lval;
				
				
				//CommonLog("pc = %x,%x data = %x", pcorig, pcact, ptr);
				
				ptr+= pcorig;
				
				if(seg_virtresolves(&ptr, dsegs, ndsegs))
				{
					ptr -= pcact;
				//	CommonLog("Adjusted to %x", ptr - pcact);
					
					uint16_t lval = ptr & 0xFFFF;
					uint16_t hval = ptr >> 16;
					//PANIC("Patching to %x, %x", lval, hval);
					//lval = 0;
					//hval = 0;
					
					uint32_t lop = 0xF240//0000
								 | rX << 24
					             | (lval >> 12) & 0xF
					             | (lval >> 1) & 0x400
					             | (lval << 20) & 0x70000000
								 | (lval << 16) & 0xFF0000;
					uint32_t hop = 0xF2C0//0000
						 		 | rX << 24
					             | (hval >> 12) & 0xF
					             | (hval >> 1) & 0x400
					             | (hval << 20) & 0x70000000
								 | (hval << 16) & 0xFF0000;
					*(uint32_t*)lptr = lop;
					*(uint32_t*)hptr = hop;
					movw[rX] = 0;
					movt[rX] = 0;
				//	PANIC("Patching to %x, %x", lop, hop);
				//	CommonLog("Patching %x to %x, %x to %x", lptr +
				}
				else if(seg_virtresolves(&ptr, tseg, 1))
				{
					
				}
				else
				{
					PANIC("pc = %x,%x data = %x", pcorig, pcact, ptr);
				}
				//PANIC("Done.");
				//return;
				//add rX, pc;
			}
		}
		// look for MOV, MOVT, ADD
	}
}


void stub32_fix(uintptr_t fbuf, section* sect, seg_adjust* tseg, seg_adjust* dsegs, int ndsegs, bool panicbit)
{
	uintptr_t start = fbuf + sect->offset;
	uintptr_t end = start + sect->size;
	
	uintptr_t pcbase_old = -fbuf + 8 + tseg->old.vmaddr.start + shared_cache_slide;
	uintptr_t pcbase_act = -fbuf + 8 + tseg->fix.vmaddr.start;
	
	
	uint32_t lastfix = -4;
	
	
	// __picsymbolstub4 289c
	// __la_symbol_ptr 3180
	
	
	for(uintptr_t data = start; data < end; data+=4)
	{
		// e59fc000
		
		// LDR R12, =(PC+offset)
		uint32_t op = *(uint32_t*) data;
		if((op & 0xFFFFFF00) == 0xE59FC000)
		{
			//CommonLog("Test!");
			uint32_t offs = op & 0xFFF;
			
			uint32_t* tofix = (uint32_t*) (data + 8 + offs);
			
			//uint32_t old = *tofix;
			
			uintptr_t pcdata = data + 4;
			//CommonLog("%08x %016lx", *(uint32_t*)pcdata, data-fbuf);
			{
				// E08FC00C // ADD R12, PC, R12 (??)
				// E08FF00C // ADD PC, PC, R12
				// E7FFDEFE
				/*
				__picsymbolstub4:00009E20 _objc_msgSend                           ; CODE XREF: -[UILabel(WeatherAdditions) setFrameOnBaseline:]+1Ep
				__picsymbolstub4:00009E20                                         ; -[UILabel(WeatherAdditions) setFrameOnBaseline:]+56p ...
				__picsymbolstub4:00009E20                 LDR             R12, =(dword_9E2C - 0x9E2C)
				__picsymbolstub4:00009E24                 ADD             PC, PC, R12 ; dword_9E2C
				__picsymbolstub4:00009E24 ; End of function _objc_msgSend
				__picsymbolstub4:00009E24
				__picsymbolstub4:00009E24 ; ---------------------------------------------------------------------------
				__picsymbolstub4:00009E28 off_9E28        DCD dword_9E2C - 0x9E2C ; DATA XREF: _objc_msgSendr
				__picsymbolstub4:00009E2C dword_9E2C      DCD 0xE7FFDEFE          ; DATA XREF: _objc_msgSend+4o
				__picsymbolstub4:00009E2C                                         ; __picsymbolstub4:off_9E28o
				*/
				
				int i = 0;
				for(; i<10 && *(uint32_t*)pcdata != 0xE08FC00C; i++, pcdata+=4)
				{}
				if(*(uint32_t*)pcdata != 0xE08FC00C)
					continue;
			}
			
			
			*tofix += pcdata + pcbase_old;
			
			//CommonLog("old pointer %x", *tofix);
			if(seg_virtresolves(tofix, dsegs, ndsegs))
			{
				//	CommonLog("pointer = %x", *tofix);
				lastfix = *tofix;
				*tofix -= pcdata + pcbase_act;
			}
			else if(in_dyld_cache(*tofix))
			{
				locate_address(*tofix, 1);
				//
				//CommonLog("Dest = %x (from %x)", (uint32_t)pcdata, *tofix);

				if(panicbit)
				{
				//	CommonLog("WARNING: Patching entry from context: %lx = %x", data - fbuf, *tofix);
				
					//uintptr_t dest =
					//locate_address(*tofix, 1);
				}
				lastfix += 4;
				*tofix = lastfix;
				
				/*
				if(panicbit)
				{
					CommonLog("whitening pointer = %x", *tofix);
					*(uint32_t*) (fbuf + *tofix) = 0;
				}
				*/

				//*tofix = 0;
				if(panicbit) // ????
				{
					*tofix = 0;
				}
				else
				{
					*tofix -= pcdata + pcbase_act;
				}
				
				
				
				//*(uint32_t*) data = 0;
			}
			else
			{
				PANIC("HALP!!");
			}
		}
	}
}

/*
void abs32_fix(uintptr_t fbuf, section* sect, seg_adjust* tseg, seg_adjust* dseg)
{
	uintptr_t start = fbuf + sect->offset;
	uintptr_t end = start + sect->size;
	
//	uintptr_t pcbase_old = -fbuf + 8 + tseg->old.vmaddr.start;
//	uintptr_t pcbase_act = -fbuf + 8 + tseg->fix.vmaddr.start;
	
	for(uintptr_t data = start; data < end; data+=4)
	{
		//uint32_t *value = (uint32_t*) data;
		//CommonLog("Fixing %x %p", *value, data - fbuf);
		
		//CommonLog("Range %x %x", tseg->old.vmaddr.start, tseg->old.vmaddr.end);
		if(seg_virtresolve((uint32_t*) data, tseg))
		{
			
		}
		else if(seg_virtresolve((uint32_t*) data, dseg))
		{
			
		}
		
	}
}*/


void extz_fix(uintptr_t fbuf, section* sect, seg_adjust* tseg, seg_adjust* dsegs, int ndsegs)
{
	// this function *WILL* panic if it finds a local offset
	
	uintptr_t start = fbuf + sect->offset;
	uintptr_t end = start + sect->size;
	
	for(uintptr_t data = start; data < end; data+=4)
	{
		if(seg_virtresolves((uint32_t*) data, tseg, 1))
		{
			PANIC("Missed automatic fix at %lx", data - fbuf);
		}
		else if(seg_virtresolves((uint32_t*) data, dsegs, ndsegs))
		{
			PANIC("Missed automatic fix at %lx", data - fbuf);
		}
		else if(in_dyld_cache(*(uint32_t*)data))
		{
			*(uint32_t*) data = 0;
		}
	}
}


void rebasez_fix(uintptr_t fbuf, section* sect, seg_adjust* tseg, seg_adjust* dsegs, int ndsegs, rebase_info* rebase)
{
	// this function *WILL* panic if it finds a local offset
	
	uintptr_t start = fbuf + sect->offset;
	uintptr_t end = start + sect->size;
	
	for(uintptr_t data = start; data < end; data+=4)
	{
		if(seg_virtresolves((uint32_t*) data, tseg, 1))
		{
			push_rebase_entry(rebase, data);
		}
		else if(seg_virtresolves((uint32_t*) data, dsegs, ndsegs))
		{
			push_rebase_entry(rebase, data);
		}
		else if(in_dyld_cache(*(uint32_t*)data))
		{
			*(uint32_t*) data = 0;
		}
	}
}



	
/*
void abs32z_fix(uintptr_t fbuf, section* sect, seg_adjust* tseg, seg_adjust* dseg, bool silent = 0)
{
	//return;
	
	uintptr_t start = fbuf + sect->offset;
	uintptr_t end = start + sect->size;
	
//	uintptr_t pcbase_old = -fbuf + 8 + tseg->old.vmaddr.start;
//	uintptr_t pcbase_act = -fbuf + 8 + tseg->fix.vmaddr.start;
	
	for(uintptr_t data = start; data < end; data+=4)
	{
		//uint32_t *value = (uint32_t*) data;
		//CommonLog("Fixing %x %p", *value, data - fbuf);
		
		//CommonLog("Range %x %x", tseg->old.vmaddr.start, tseg->old.vmaddr.end);
		if(seg_virtresolve((uint32_t*) data, tseg))
		{
			if(!silent)
				CommonLog("Missed automatic fix at %lx", data - fbuf);
		}
		else if(seg_virtresolve((uint32_t*) data, dseg))
		{
			if(!silent)
				CommonLog("Missed automatic fix at %lx", data - fbuf);
		}
		else if(in_dyld_cache(*(uint32_t*)data))
		{
			*(uint32_t*) data = 0;
		}
		/ *
		if(*(uint32_t*) data >= base_mapping)
		{
			*(uint32_t*) data = 0;
		}
		* /
	}
}
*/

// seg_adjust* tseg, seg_adjust* dsegs, int ndsegs, 
void lazy_fix(uintptr_t fbuf, section* sect, section* stubs, rebase_info* rebase)
{
	uint32_t *lazy_arr = (uint32_t*) (fbuf + sect->offset);
	//int nlazy = sect->size / sizeof(uint32_t);
	
	uintptr_t start = fbuf + stubs->offset;
	uintptr_t end = start + stubs->size;
	
	int index = -2;
	for(uintptr_t data = start; data < end; data += 4)
	{
		uint32_t op = *(uint32_t*) data;
		if((op & 0xFFFFFF00) == 0xE59FC000)
		{
			if(index >= 0)
			{
				uint32_t goal = data - fbuf;
				if(lazy_arr[index] != goal)
				{
					//CommonLog("index = %x, goal = %x", lazy_arr[index], goal);
					
					//if(seg_virtresolve((uint32_t*)
					{
						push_rebase_entry(rebase, (uintptr_t) &lazy_arr[index]);
					}
					lazy_arr[index] = goal;
				}
			}
			index++;
		}
	}
}

void resolve_methnames(uintptr_t fbuf, section* sect, section* __objc_methname, seg_adjust* tseg, rebase_info* rebase)
{
	uint32_t* pointers = (uint32_t*) (fbuf + sect->offset);
	int nPointers = sect->size / sizeof(uint32_t);
	
	const char* methods = (const char*) (fbuf + __objc_methname->offset);
	int nMethods = __objc_methname->size;
	
	for(int i = 0; i<nPointers; i++)
	{
		uint32_t* pointer = &pointers[i];
		
		if(seg_virtresolves((uint32_t*) pointer, tseg, 1))
		{
		//	uint32_t tbl = sect->offset(
			

			PANIC("Missed automatic fix at %lx", (uintptr_t) pointer - fbuf);
			continue;
		}
		
		if(!in_dyld_cache(*pointer))// <  base_mapping)
		{
		//	PANIC("Could not resolve method in entire cache!");
			continue;
		}
		
		const char* str = (const char*) locate_address(*pointer, 0); // BOOOO
		int nStr = strlen(str);
		//CommonLog("Str = %s", str);
		if(str)
		{
			int j = 0;
			int c0 = str[0];
			for(; j<nMethods - nStr; j++)
			{
				if(methods[j] == c0)
				{
					int k = 1;
					for(; k < nStr+1 && methods[j+k] == str[k]; k++)
					{
					}
					if(k==nStr+1)
						break;
					j+=k;
				//	PANIC("Done searching: %s", &methods[j]);
				}
				for(; methods[j]!='\0' && j < nMethods - nStr; j++)
				{
				}
			}
			if(j<nMethods - nStr)
			{
				*pointer = j + __objc_methname->addr;
				
				push_rebase_entry(rebase, (uintptr_t) pointer);

			//	CommonLog("String is %s=%s (%x)", str, &methods[j], pointers[i]);
			}
			else
			{
			 	PANIC("Failed to find method \'%s\'(%x, %x) at %x", str, (uint32_t)((uintptr_t)str - (uintptr_t)dyld_buf), *pointer, (uint32_t)((uintptr_t)pointer - fbuf) );
				
			}
		}
		else
		{
			PANIC("Failed to discover string %x", *pointer);
		}
	}
}


void resolve_methname(uintptr_t fbuf, uint32_t* pointer, section* __objc_methname, seg_adjust* tseg, rebase_info* rebase, bool noPanic = 0)
{
	const char* methods = (const char*) (fbuf + __objc_methname->offset);
	int nMethods = __objc_methname->size;
	
	if(seg_virtresolves(pointer, tseg, 1))
	{
		if(!noPanic)
		{
			PANIC("Missed automatic fix at %lx", (uintptr_t) pointer - fbuf);
		}
		push_rebase_entry(rebase, (uintptr_t) pointer);	
		return;
	}
	
	if(!in_dyld_cache(*pointer))
	//if(*pointer <  base_mapping)
		return;
	
	const char* str = (const char*) locate_address(*pointer);
	int nStr = strlen(str);
	if(str)
	{
		int j = 0;
		int c0 = str[0];
		for(; j<nMethods - nStr; j++)
		{
			if(methods[j] == c0)
			{
				int k = 1;
				for(; k < nStr+1 && methods[j+k] == str[k]; k++)
				{
				}
				if(k==nStr+1)
					break;
				j+=k;
			//	PANIC("Done searching: %s", &methods[j]);
			}
			for(; methods[j]!='\0' && j < nMethods - nStr; j++)
			{
			}
		}
		if(j<nMethods - nStr)
		{
			*pointer = j + __objc_methname->addr;
			push_rebase_entry(rebase, (uintptr_t) pointer);	

		//	CommonLog("String is %s=%s (%x)", str, &methods[j], pointers[i]);
		}
		else
		{
			PANIC("Failed to find method \'%s\'(%x) at %lx", str, *pointer, (uintptr_t)pointer - fbuf);
			
		}
	}
	else
	{
		PANIC("Failed to discover string %x", *pointer);
	}
	
}


struct class_t {
    uint32_t isa;
    uint32_t superclass;
    uint32_t cache;
    uint32_t vtable;
    uint32_t data;
};

struct class_ro_t {
    uint32_t flags;
    uint32_t instanceStart;
    uint32_t instanceSize;
    uint32_t ivarLayout;
    uint32_t name;
    uint32_t baseMethods;
    uint32_t baseProtocols;
    uint32_t ivars;
    uint32_t weakIvarLayout;
    uint32_t baseProperties;
};

struct list_t {
	uint32_t entsize;
	uint32_t count;
};

struct method_t {
    uint32_t name;
    uint32_t types;
    uint32_t imp;
};

struct ivar_t {
	uint32_t offset;
	uint32_t name;
	uint32_t type;
	uint32_t alignment;
	uint32_t size;
};

struct property_t {
	uint32_t name;
	uint32_t type;
};






void fix_methods(uintptr_t fbuf, list_t* head, section* __objc_methname, seg_adjust* tseg, seg_adjust* dsegs, int ndsegs, rebase_info* rebase, bool nowarn = 0)
{
//	CommonLog("Fixing method");
	int n = head->count;
	head->entsize = 0xC;	// fix the 0xf in the cache
	
	method_t* methods = (method_t*) ((uintptr_t)head + sizeof(list_t));
	//method_t static_method;
	
	resolve_methname(fbuf, &methods[0].name, __objc_methname, tseg, rebase, nowarn);
	if(seg_virtresolves(&methods[0].types, tseg, 1))
	{
		if(!nowarn)
			PANIC("Missed automatic fix at %lx", (uintptr_t) &methods[0].types - fbuf);
		push_rebase_entry(rebase, (uintptr_t) &methods[0].types);

	}
	if(seg_virtresolves(&methods[0].imp, tseg, 1))
	{
		if(!nowarn)
			PANIC("Missed automatic fix at %lx", (uintptr_t) &methods[0].imp - fbuf);
		push_rebase_entry(rebase, (uintptr_t)&methods[0].imp);		
	}

	for(int i=1; i<n; i++)
	{
		resolve_methname(fbuf, & methods[i].name, __objc_methname, tseg, rebase, nowarn);
		if(seg_virtresolves(&methods[i].types, tseg, 1))
		{
			if(!nowarn)
				PANIC("Missed automatic fix at %lx", (uintptr_t) &methods[i].types - fbuf);
			push_rebase_entry(rebase, (uintptr_t)&methods[i].types);

		}
		if(seg_virtresolves(&methods[i].imp, tseg, 1))
		{
			if(!nowarn)
				PANIC("Missed automatic fix at %lx", (uintptr_t) &methods[i].imp - fbuf);
			push_rebase_entry(rebase, (uintptr_t)&methods[i].imp);		
		}
		/*
		static_method = methods[i];
		
		// no good reason to sort!
		int j = i-1;
		for(; j>=0; j--)
		{
			if(methods[j].imp > static_method.imp )
			// || (methods[j].imp == static_method.imp && methods[j].types > static_method.types))
			{
				methods[j+1] = methods[j];
			}
			else
			{
				break;
			}
		}
		methods[j+1] = static_method;
		*/
		
	}
	
}

void fix_ivars(uintptr_t fbuf, list_t* head, section* __objc_methname, seg_adjust* tseg, seg_adjust* dsegs, int ndsegs, rebase_info* rebase)
{
	int n = head->count;
	
	ivar_t* ivars = (ivar_t*) ((uintptr_t)head + sizeof(list_t));
	
	for(int i=0; i<n; i++)
	{
		if(seg_virtresolves(&ivars[i].offset, dsegs, ndsegs))
		{
			PANIC("Missed automatic fix at %lx", (uintptr_t) &ivars[i].offset - fbuf);

		}
		if(seg_virtresolves(&ivars[i].type, tseg, 1))
		{
			PANIC("Missed automatic fix at %lx", (uintptr_t) &ivars[i].type - fbuf);

		}
		resolve_methname(fbuf, &ivars[i].name, __objc_methname, tseg, rebase);
		
	}
}

void fix_properties(uintptr_t fbuf, list_t* head, seg_adjust* tseg)
{
	int n = head->count;
	
	property_t* props = (property_t*) ((uintptr_t)head + sizeof(list_t));
	
	for(int i=0; i<n; i++)
	{
		if(seg_virtresolves(&props[i].name, tseg, 1))
		{
			PANIC("Missed automatic fix at %lx", (uintptr_t) &props[i].name - fbuf);
		}
		if(seg_virtresolves(&props[i].type, tseg, 1))
		{
			PANIC("Missed automatic fix at %lx", (uintptr_t) &props[i].type - fbuf);

		}
	}
}

struct protocol_list_t
{
	uint32_t count;
	uint32_t list[];
};

void fix_protocols(protocol_list_t *protocols, seg_adjust* dsegs, int ndsegs)
{
	int nProtocols = protocols->count;
	for(int i=0; i<nProtocols; i++)
	{
		if(seg_virtresolves(&protocols->list[i], dsegs, ndsegs))
		{
			PANIC("Missed virtresolve!");
		}
	}
}

void fix_classdata(uintptr_t fbuf, class_ro_t* cls, section* __objc_methname, section* __objc_extradata, seg_adjust* tseg, seg_adjust* dsegs, int ndsegs, rebase_info* rebase)
{
	if(cls->name)
	{
		if(seg_virtresolves(&cls->name, tseg, 1))
		{
			PANIC("Missed automatic fix at %lx", (uintptr_t) &cls->name - fbuf);
		}
//		CommonLog("__OBJC_CLASS_$_%s ...", (char*) (fbuf + cls->name));
	}
	else
	{
		PANIC("No class name!");
	}
	
	if(cls->ivarLayout)
	{
		if(seg_virtresolves(&cls->ivarLayout, dsegs, ndsegs))
		{
			
		}
		else if(in_dyld_cache(cls->ivarLayout))
		{
			PANIC("ivarlayout at %x; dont know what we're doing!", cls->ivarLayout);
		}
		else
		{
			CommonLog("%016lx", (uintptr_t)cls-fbuf);
			CommonLog("WARNING: ivar layout; we don't know how to handle this!");
		}
	}
	
	if(cls->baseMethods)
	{
		bool nowarn = 0;
		if(seg_virtresolves(&cls->baseMethods, dsegs, ndsegs))
		{
			PANIC("Missed automatic fix at %lx", (uintptr_t) &cls->baseMethods - fbuf);
		}
		else if(in_dyld_cache(cls->baseMethods))
		{
			CommonLog("Found baseMethods for %s externally.  bringing locally...", (char*) (fbuf + cls->name));
			
			uintptr_t src = locate_address(cls->baseMethods, 0);
			
			cls->baseMethods = __objc_extradata->offset + __objc_extradata->size;
			
			push_rebase_entry(rebase, (uintptr_t) &cls->baseMethods);
			//CommonLog("Pushed rebase entry %p", (uintptr_t) cls->baseMethods - fbuf);
			
			CommonLog("Pointing to %x", cls->baseMethods);

			uintptr_t dest = fbuf + cls->baseMethods;
			uint32_t size = sizeof(list_t) + ((list_t*)src)->count * sizeof(method_t);
			
			memcpy((void*)dest, (void*)src, size);
			__objc_extradata->size += size;
			// + sizeof(method_t);
			
			
			/*
			CommonLog("WARNING: Could not find method listing locally: %s %x", (char*) (fbuf + cls->name), cls->baseMethods);
			
			list_t* method_list = (list_t*) locate_address(cls->baseMethods, 1);
			
			CommonLog("%d %d", method_list->entsize, method_list->count);
			
			method_t* methods = (method_t*) ((uintptr_t)method_list + sizeof(list_t));
			
			for(int i=0; i<method_list->count; i++)
			{
				method_t method = methods[i];
				
				resolve_methname(fbuf, &method.name, __objc_methname, tseg);
				seg_virtresolve(&method.types, tseg);
				seg_virtresolve(&method.imp, tseg);
				CommonLog("%s %s %x", (char*) (fbuf + method.name), (char*) (fbuf + method.types), method.imp);
			}
			*/
			nowarn = 1;
		}
		
		fix_methods(fbuf, (list_t*) (fbuf + cls->baseMethods), __objc_methname, tseg, dsegs, ndsegs, rebase, nowarn);
	}
	if(cls->baseProtocols)
	{
		if(seg_virtresolves(&cls->baseProtocols, dsegs, ndsegs))
		{
			CommonLog("Missed automatic fix at %lx", (uintptr_t) &cls->baseProtocols - fbuf);
		}
		else if(in_dyld_cache(cls->baseProtocols))
		{
			PANIC("Cannot handle protocol copying yet!");
		}
		fix_protocols((protocol_list_t*) (fbuf + cls->baseProtocols), dsegs, ndsegs);
	}
	if(cls->ivars)
	{
		if(seg_virtresolves(&cls->ivars, dsegs, ndsegs))
		{
			CommonLog("Missed automatic fix at %lx", (uintptr_t) &cls->ivars - fbuf);

		}
		else if(in_dyld_cache(cls->ivars))
		{
			PANIC("Cannot handle ivar copying yet!");
		}
		fix_ivars(fbuf, (list_t*) (fbuf + cls->ivars), __objc_methname, tseg, dsegs, ndsegs, rebase);
	}
	if(cls->weakIvarLayout)
	{
		if(seg_virtresolves(&cls->weakIvarLayout, dsegs, ndsegs))
		{
			CommonLog("Missed automatic fix at %lx", (uintptr_t) &cls->weakIvarLayout - fbuf);
		}
		else if(in_dyld_cache(cls->weakIvarLayout))
		{
			PANIC("Cannot handle weak ivar copying yet!");
		}
		else
		{
			CommonLog("WARNING: Weak ivar layout; we don't know how to handle this!");
		}
		//PANIC("Weak ivar layout?? HALP!");
		
		
		
		
		//seg_virtresolve(&cls->weakIvarLayout, dseg);
	}
	
	
	if(cls->baseProperties)
	{
		if(seg_virtresolves(&cls->baseProperties, dsegs, ndsegs))
		{
			CommonLog("Missed automatic fix at %lx", (uintptr_t) &cls->baseProperties - fbuf);

		}
		else if(in_dyld_cache(cls->baseProperties))
		{
			PANIC("Cannot handle base property copying yet!");
		}
		
		fix_properties(fbuf, (list_t*) (fbuf + cls->baseProperties), tseg);
	}
}


void fix_from_classlist(uintptr_t fbuf, section* sect, section* __objc_methname,  section* __objc_extradata, seg_adjust* tseg, seg_adjust* dsegs, int ndsegs, rebase_info* rebase)
{
	uint32_t *classes = (uint32_t*) (fbuf + sect->offset);
	int nClass = sect->size / sizeof(uint32_t);
	
	for(int i=0; i<nClass; i++)
	{
		//CommonLog("%d\n", i);
		//uint32_t* class = &classes[i];
		//seg_virtresolve(class, dsreg);
		
		class_t* cls = (class_t*) (fbuf + classes[i]);
		class_t* meta = (class_t*) (fbuf + cls->isa);
		
		if(!seg_inoutputs(classes[i], dsegs, ndsegs))
		{
			locate_address(classes[i], 1);
			PANIC("Error finding in correct spot!");// %x %x", dseg->fix.offset.start, dseg->fix.offset.end);
		}
		if(!seg_inoutputs(cls->isa, dsegs, ndsegs))
			PANIC("Error finding in correct spot!");

		// cls->data
		if(!seg_inoutputs(cls->data, dsegs, ndsegs))
		{
			CommonLog("cls->data = %08x", cls->data);
			
			for(int j=0; j<32; j++)
			{
				fprintf(stderr, "%02x", ((uint8_t*)(cls))[j]);
				if(((j+1) & 3)==0)
					if(((j+1) & 31)==0)
						fprintf(stderr, "\n");
					else
						fprintf(stderr, " ");
			}
			fprintf(stderr, "--\n");
			
			
			
			PANIC("Error finding in correct spot!");
		}

		if(!seg_inoutputs(meta->data, dsegs, ndsegs))
			PANIC("Error finding in correct spot!");

		//CommonLog("cls = %x, meta = %x data = %x", classes[i], cls->isa, cls->data);
		fix_classdata(fbuf, (class_ro_t*) (fbuf + cls->data), __objc_methname, __objc_extradata, tseg, dsegs, ndsegs, rebase);
		fix_classdata(fbuf, (class_ro_t*) (fbuf + meta->data), __objc_methname, __objc_extradata, tseg, dsegs, ndsegs, rebase);
		
	}
}


struct protocol_t {
    uint32_t isa;
    uint32_t name;
    uint32_t protocols;
    uint32_t instanceMethods;
    uint32_t classMethods;
    uint32_t optionalInstanceMethods;
    uint32_t optionalClassMethods;
    uint32_t instanceProperties;
};


void fix_from_protolist(uintptr_t fbuf, section* sect, section* __objc_methname, section* __objc_extradata, seg_adjust* tseg, seg_adjust* dsegs, int ndsegs, rebase_info* rebase)
{
	uint32_t *protocols = (uint32_t*) (fbuf + sect->offset);
	int nProtocol = sect->size / sizeof(uint32_t);
	
	for(int i=0; i<nProtocol; i++)
	{
		//uint32_t* class = &classes[i];
		//seg_virtresolve(class, dsreg);
		
		protocol_t* proto = (protocol_t*) (fbuf + protocols[i]);
		
		if(proto->name)
		{
			if(seg_virtresolves(&proto->name, tseg, 1))
			{
				CommonLog("Missed automatic fix at %lx", (uintptr_t) &proto->name - fbuf);
			}
		//	CommonLog("Fixing %s...", (char*) (fbuf + proto->name));
		}
		else
		{
			PANIC("No protocol name!");
		}
		
		if(proto->protocols)
		{
			if(seg_virtresolves(&proto->protocols, dsegs, ndsegs))
			{
				CommonLog("Missed automatic fix at %lx", (uintptr_t) &proto->protocols - fbuf);
			}
			else if(in_dyld_cache(proto->protocols))
			{
				CommonLog("Found protocols externally.  bringing locally...");
				
				uintptr_t src = locate_address(proto->protocols, 0);
				
				proto->protocols = __objc_extradata->offset + __objc_extradata->size;
				
				
				
				
				
				CommonLog("Pointing to %x", proto->protocols);
				uintptr_t dest = fbuf + proto->protocols;

				uint32_t size = sizeof(uint32_t) + ((protocol_list_t*)src)->count * sizeof(uint32_t);
				
				memcpy((void*)dest, (void*)src, size);
				__objc_extradata->size += size;
	
			}
			fix_protocols((protocol_list_t*) (fbuf + proto->protocols), dsegs, ndsegs);
			
		}
		
		if(proto->instanceMethods)
		{
			if(seg_virtresolves(&proto->instanceMethods, dsegs, ndsegs))
			{
				CommonLog("Missed automatic fix at %lx", (uintptr_t) &proto->instanceMethods - fbuf);
			}
			else if(in_dyld_cache(proto->instanceMethods))
			{
				CommonLog("Looking elsewhere for proto->instanceMethods = %x", proto->instanceMethods);

				uintptr_t src = locate_address(proto->instanceMethods, 0);
				if(!src)
					PANIC("Could not find address anywhere");
				proto->instanceMethods = __objc_extradata->offset + __objc_extradata->size;
				
				CommonLog("Pointing to %x", proto->instanceMethods);
				uintptr_t dest = fbuf + proto->instanceMethods;
				LINE();
				uint32_t size = sizeof(list_t) + ((list_t*)src)->count * sizeof(method_t);
				LINE();
				memcpy((void*)dest, (void*)src, size);
				__objc_extradata->size += size;
				
			}
			fix_methods(fbuf, (list_t*) (fbuf + proto->instanceMethods), __objc_methname, tseg, dsegs, ndsegs, rebase);
			
		}
		if(proto->classMethods)
		{
			if(seg_virtresolves(&proto->instanceMethods, dsegs, ndsegs))
			{
				CommonLog("Missed automatic fix at %lx", (uintptr_t) &proto->instanceMethods - fbuf);
			}
			else if(in_dyld_cache(proto->instanceMethods))
			{
				PANIC("Need to search externally for proto->classMethods.  Not handled yet.");
			}
			fix_methods(fbuf, (list_t*) (fbuf + proto->classMethods), __objc_methname, tseg, dsegs, ndsegs, rebase);
			
		}
		if(proto->optionalInstanceMethods)
		{
			if(seg_virtresolves(&proto->optionalInstanceMethods, dsegs, ndsegs))
			{
				CommonLog("Missed automatic fix at %lx", (uintptr_t) &proto->optionalInstanceMethods - fbuf);
			}
			else if(in_dyld_cache(proto->optionalInstanceMethods))
			{
				PANIC("Need to search externally for proto->optionalInstanceMethods.  Not handled yet.");
			}
			fix_methods(fbuf, (list_t*) (fbuf + proto->optionalInstanceMethods), __objc_methname, tseg, dsegs, ndsegs, rebase);
			
		}
		if(proto->optionalClassMethods)
		{
			if(seg_virtresolves(&proto->optionalClassMethods, dsegs, ndsegs))
			{
				CommonLog("Missed automatic fix at %lx", (uintptr_t) &proto->optionalClassMethods - fbuf);
			}
			else if(in_dyld_cache(proto->optionalClassMethods))
			{
				PANIC("Need to search externally for proto->optionalClassMethods.  Not handled yet.");
			}
			fix_methods(fbuf, (list_t*) (fbuf + proto->optionalClassMethods), __objc_methname, tseg, dsegs, ndsegs, rebase);
			
		}
		if(proto->instanceProperties)
		{
			if(seg_virtresolves(&proto->instanceProperties, dsegs, ndsegs))
			{
				CommonLog("Missed automatic fix at %lx", (uintptr_t) &proto->instanceProperties - fbuf);
			}
			else if(in_dyld_cache(proto->instanceProperties))
			{
				PANIC("Need to search externally for proto->instanceProperties.  Not handled yet.");
			}
			fix_properties(fbuf, (list_t*)(fbuf + proto->instanceProperties), tseg);
			
		}
	}
}


struct category_t {
    uint32_t name;
    uint32_t cls;
    uint32_t instanceMethods;
    uint32_t classMethods;
    uint32_t protocols;
    uint32_t instanceProperties;
};

void fix_from_catlist(uintptr_t fbuf, section* sect, section* __objc_methname, section* __objc_extradata, seg_adjust* tseg, seg_adjust* dsegs, int ndsegs, rebase_info* rebase)
{
	uint32_t *categories = (uint32_t*) (fbuf + sect->offset);
	int nCategory = sect->size / sizeof(uint32_t);
	
	for(int i=0; i<nCategory; i++)
	{
		//uint32_t* class = &classes[i];
		//seg_virtresolve(class, dsreg);
		
		category_t* cat = (category_t*) (fbuf + categories[i]);
		
		if(cat->name)
		{
			//if(!seg_virtresolve(&cat->name, tseg))
			//	PANIC("Name did not resolve: %x %x", cat->name);
		//	CommonLog("Fixing %s...", (char*) (fbuf + cat->name));
		}
		else
		{
			PANIC("No category name!");
		}
		
		if(cat->cls)
		{
			//if(!seg_virtresolve(&cat->cls, dseg))
				cat->cls = 0;
		}
		
		if(cat->protocols)
		{
			if(in_dyld_cache(cat->protocols))
				PANIC("Out of range!");
			
			fix_protocols((protocol_list_t*) (fbuf + cat->protocols), dsegs, ndsegs);
		}
		
		if(cat->instanceMethods)
		{
			if(in_dyld_cache(cat->instanceMethods))
				PANIC("Out of range!");
			
			fix_methods(fbuf, (list_t*) (fbuf + cat->instanceMethods), __objc_methname, tseg, dsegs, ndsegs, rebase);
			
		}
		if(cat->classMethods)
		{
			if(in_dyld_cache(cat->classMethods))
				PANIC("Out of range!");
			
			fix_methods(fbuf, (list_t*) (fbuf + cat->classMethods), __objc_methname, tseg, dsegs, ndsegs, rebase);
			
		}
		if(cat->instanceProperties)
		{
			if(in_dyld_cache(cat->classMethods))
				PANIC("Out of range!");
			fix_properties(fbuf, (list_t*)(fbuf + cat->instanceProperties), tseg);
		}

	}

}

struct hack_context
{
	exported_node* basenode;
	seg_adjust* tseg;
	seg_adjust* dsegs;
	int ndsegs;
};

void hack_export_callback(exported_node* node, uintptr_t ctx)
{
	
	hack_context* context = (hack_context*) ctx;
	//CommonLog("basenode = %p %p", context->basenode, ctx);
	
	
//	CommonLog("0node = %p child = %x", node, (uintptr_t)(node)->child);
	
//	print_export_commands_sub(node, NULL);
	
//	CommonLog("2node = %p child = %x", node, (uintptr_t)(node)->child);

	if(node->flags & EXPORT_SYMBOL_FLAGS_STUB_AND_RESOLVER)
	{
		node->stub += context->tseg->old.vmaddr.start;// + shared_cache_slide;
		if(!seg_virtresolves(&node->stub, context->tseg, 1))
		{
			if(!seg_virtresolves(&node->stub, context->dsegs, context->ndsegs))
			{
				CommonLog("Could not locate %x", node->stub);
			}
		}
		
		node->resolver += context->tseg->old.vmaddr.start;// + shared_cache_slide;
		if(!seg_virtresolves(&node->resolver, context->tseg, 1))
		{
			if(!seg_virtresolves(&node->stub, context->dsegs, context->ndsegs))
			{
				CommonLog("Could not locate %x", node->stub);
			}
		}
		
	//	fprintf(stdout, ".EXPORT_RESOLVER %08x %08x %s\n", node->stub, node->resolver, node->base);
	}
	else if(node->flags & EXPORT_SYMBOL_FLAGS_REEXPORT)
	{
	//	fprintf(stdout, ".EXPORT_REEXPORT %02x %s %s\n", node->ordinal, node->base, node->reexport);
	}
	else
	{
		//uint32_t orig = node->stub;
		
		//fprintf(stdout, ".EXPORT(%d) %08x %08x %s\n", node->flags, node->stub, orig, node->base);
		
	//	CommonLog("1node = %p child = %x", node, (uintptr_t)(node)->child);
		
		node->stub += context->tseg->old.vmaddr.start + shared_cache_slide;
		if(!seg_virtresolves(&node->stub, context->dsegs, context->ndsegs))
		{	
			if(!seg_virtresolves(&node->stub, context->tseg, 1))
			{
				CommonLog("Could not locate %x", node->stub);
			}
		}
		
		//LINE();
	}
		
//	CommonLog("node = %p child = %x", node, (uintptr_t)(node)->child);

	//LINE();
	export_construct_terminal(node);
	//LINE();
	
	//CommonLog("node = %p child = %x", node, (uintptr_t)(node)->child);
	
	export_add_node(&(context->basenode), node);
	
//	CommonLog("Basenode = %p child = %x", context->basenode, (uintptr_t)(context->basenode)->child);
	
}

int hack_export_commands(uintptr_t src, uint32_t size, seg_adjust* tseg, seg_adjust* dsegs, int ndsegs)
{
	hack_context context;
	context.basenode = NULL;
	
	context.tseg = tseg;
	context.dsegs = dsegs;
	context.ndsegs = ndsegs;
	
	char strbuf[0x200];
	
	scan_export_tree((uint8_t*)src, size, strbuf, hack_export_callback, (uintptr_t) &context);
	
	//LINE();
	
	
	size = export_finalize((char*)src, context.basenode);
	
	//LINE();
	//exit(1);
	
	
	// DEBUG****
	//print_export_commands(src, size);
	
	
	
	return size;
	
	/*
	uintpr_t out_export = malloc(size + 0x400);
	
	
	
	
	
	
	
	
	
	
	
	return export;
	*/
}




section* append_section(uintptr_t fbuf, section* newsect)
{
	mach_header* header = (mach_header*) fbuf;
	
	uint32_t ncmds = header->ncmds;
	uint32_t sizeofcmds = header->sizeofcmds;
	uintptr_t cmd_base = fbuf + sizeof(mach_header);
	
	
	// first scan: make sure we have space
	{
		uint32_t lowestOffset = 1<<31-1;
		
		uintptr_t lcptr = cmd_base;
		for(uint32_t i=0; i<ncmds; i++)
		{
			uint32_t cmd = ((load_command*) lcptr)->cmd;
		
			if(cmd == LC_SEGMENT)
			{
				segment_command* seg = (segment_command*) lcptr;
				int nsects = seg->nsects;
				if(nsects)
				{
					section* sects = (section*) ((uintptr_t)seg + sizeof(segment_command));
					for(int i=0; i<nsects; i++)
					{
						if( ((sects[i].flags & SECTION_TYPE) != S_ZEROFILL)
							&& sects[i].size && (sects[i].offset < lowestOffset))
						{
							lowestOffset = sects[i].offset;
						}
					}
				}
			}
			/*
				//print_segment(seg);
				if(!strcmp(seg->segname, SEG_TEXT))
				{
					text = seg;
				}
				else if(!strcmp(seg->segname, SEG_DATA))
				{
					data = seg;
				}
				else if(!strcmp(seg->segname, SEG_LINKEDIT))
				{
				//	linkedit = seg;
				}
				else
				{
					PANIC("Unrecognized segment in file");
				}
			*/
			lcptr += ((load_command*) lcptr)->cmdsize;
		}
		if(lowestOffset < sizeof(mach_header) + sizeofcmds + sizeof(section))
		{
		//	CommonLog("Space not available for new section: %x %lx %lx", lowestOffset, sizeof(mach_header) + sizeofcmds, sizeof(mach_header) + sizeofcmds + sizeof(section));
			return NULL;
		}
	}
	{
		uintptr_t lcptr = cmd_base;
		for(uint32_t i=0; i<ncmds; i++)
		{
			uint32_t cmd = ((load_command*) lcptr)->cmd;
	
			if(cmd == LC_SEGMENT)
			{
				segment_command* seg = (segment_command*) lcptr;
				
				if(!strncmp(seg->segname, newsect->segname, 16))
				{
					uintptr_t seg_end = (uintptr_t)seg + seg->cmdsize;
					uint32_t remaining = sizeofcmds - (seg_end - cmd_base);
					memmove((void*) (seg_end + sizeof(section)), (void*)seg_end, remaining);
					
					seg->nsects++;
					seg->cmdsize += sizeof(section);
					memcpy((void*) seg_end, (void*) newsect, sizeof(section));
					
					header->sizeofcmds += sizeof(section);
					
					return (section*) seg_end;
				}
			}
			lcptr += ((load_command*) lcptr)->cmdsize;
		}
	}
	return NULL;
	
}

void remove_commandtype(uintptr_t fbuf, int cmd)
{
	
}





void ProcessSegment(uintptr_t old_data, ptrdiff_t old_offs, segment_command* old_seg, seg_adjust* adjust, seg_info* fdinfo)
{
	// where do you come from
	adjust->old.vmaddr.start = old_seg->vmaddr;
	adjust->old.vmaddr.end = old_seg->vmaddr + old_seg->vmsize;
	
	adjust->old.offset.start = old_offs;
	adjust->old.offset.end = old_offs + old_seg->filesize;
	
	adjust->old.buf = old_data;
	
	
	// where do you go
	adjust->fix.vmaddr.start = fdinfo->vmaddr.end;
	adjust->fix.vmaddr.end = fdinfo->vmaddr.end + old_seg->vmsize;
	
	adjust->fix.offset.start = fdinfo->offset.end;
	adjust->fix.offset.end = fdinfo->offset.end + old_seg->filesize;
	
	adjust->fix.buf = fdinfo->buf + fdinfo->offset.end;
	
	
	// clone the segment
	memcpy((void*) adjust->fix.buf, (void*) adjust->old.buf, old_seg->filesize);
	
	// cotton eyed joe....adjust the out_seg
	fdinfo->vmaddr.end += old_seg->vmsize;
	fdinfo->offset.end += old_seg->filesize;
	
	adjust->dvmaddr = adjust->fix.vmaddr.start - adjust->old.vmaddr.start;
	adjust->doffset = adjust->fix.offset.start - adjust->old.offset.start;
}


void extract_file(uintptr_t xbuf, const char* fname)//name)
{
	mach_header* header = (mach_header*) xbuf;
	
	if(!xbuf)
		PANIC("No file!");
	
	if(header->magic != 0xfeedface)
	{
		PANIC("Magic does not match");
	}
	
	uint32_t ncmds = header->ncmds;
	
	
	
	//const char* fname = "TEMP";
	int fd;
	uintptr_t fbuf;
	int nfile;
	
	
	seg_info fdinfo;
	fdinfo.vmaddr.start = 0;
	fdinfo.vmaddr.end = 0;
	fdinfo.offset.start = 0;
	fdinfo.offset.end = 0;

	seg_adjust tseg;
	
	/*
	seg_adjust dseg;
	seg_adjust dseg_dirty;
	seg_adjust dseg_const;
	*/
	
	int ndsegs = 0;
	seg_adjust dsegs[3];

	
	
//	segment_command* oldtext;
//	segment_command* olddata;
	
//	segment_command* data = NULL;
//	segment_command* data_dirty = NULL;
//	segment_command* data_const = NULL;
	
	segment_command* text = NULL;
	segment_command* datas[3] = {NULL, NULL, NULL};
	
	{
		
		intptr_t lcptr = xbuf + sizeof(mach_header);
		for(uint32_t i=0; i<ncmds; i++)
		{
			uint32_t cmd = ((load_command*) lcptr)->cmd;
			
			if(cmd == LC_SEGMENT)
			{
				segment_command* seg = (segment_command*) lcptr;
				//print_segment(seg);
				if(!strncmp(seg->segname, SEG_TEXT, 16))
				{
					text = seg;
				}
				else if(!strncmp(seg->segname, SEG_DATA, 16) || !strncmp(seg->segname, SEG_DATA_DIRTY, 16) || !strncmp(seg->segname, SEG_DATA_CONST, 16))
				{
					datas[ndsegs] = seg;
					ndsegs++;
				}
				else if(!strncmp(seg->segname, SEG_LINKEDIT, 16))
				{
				//	linkedit = seg;
				}
				else
				{
					PANIC("Unrecognized segment in file");
				}
			}
			lcptr += ((load_command*) lcptr)->cmdsize;
		}
		
		nfile = (text ? text->vmsize : 0);
		for(int i=0; i<ndsegs; i++)
			nfile += datas[i]->vmsize;
		
		
		
		char fpath[0x80];
		
		const char* fnptr = fname;
		{
			while((fnptr = strchr(fnptr, '/')))
			{
				int n = fnptr - fname;
				memcpy(fpath, fname, n);
				fpath[n]=0;
				mkdir(fpath, S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
				fnptr++;
			}
		}
		
		//CommonLog("Creating file of size %x", nfile + 0x1000000);
		
		fd = fdcreate(fname, nfile + 0x1000000, &fbuf);	// allocate 1MB extra for the linkedit.  maybe not enough?
		
		fdinfo.buf = fbuf;
		
		if(text)
		{
			ProcessSegment(xbuf, xbuf-dyld_buf, text, &tseg, &fdinfo);
			
			seg_applyoffset(find_segment(fbuf, SEG_TEXT), &tseg);
		//	CommonLog("Fixed partially text segment");

		}
		
		for(int i=0; i<ndsegs; i++)
		{
			segment_command* data = datas[i];
			
			ProcessSegment(dyld_buf + data->fileoff, data->fileoff, data, &dsegs[i], &fdinfo);
			seg_applyoffset(find_segment(fbuf, data->segname), &dsegs[i]);
		//	CommonLog("Fixed partially data segment %x", i);
		}
		
		/*
		if(data)
		{
			dseg.old.vmaddr.start = data->vmaddr;
			dseg.old.vmaddr.end   = dseg.old.vmaddr.start + data->vmsize;
			dseg.old.offset.start = data->fileoff;
			dseg.old.offset.end   = data->fileoff + data->filesize;
			dseg.old.buf = dyld_buf + data->fileoff;
		
		
			dseg.fix.vmaddr.start = tseg.fix.vmaddr.end;
			dseg.fix.vmaddr.end   = tseg.fix.vmaddr.end + data->vmsize;
			dseg.fix.offset.start = tseg.fix.offset.end;
			dseg.fix.offset.end   = tseg.fix.offset.end + data->filesize;
			dseg.fix.buf = fbuf + text->vmsize;
			
			memcpy((void*)dseg.fix.buf, (void*)dseg.old.buf, data->filesize);
			
			dseg.dvmaddr = dseg.fix.vmaddr.start - dseg.old.vmaddr.start;
			dseg.doffset = dseg.fix.offset.start - dseg.old.offset.start;
			
			
		//	CommonLog("data vmaddr: %x to %x", dseg.old.vmaddr.start, dseg.fix.vmaddr.start);
		//	CommonLog("data offset: %x to %x", dseg.old.offset.start, dseg.fix.offset.start);
			
		}
		else
		{
			dseg.old.vmaddr.start = 0;
			dseg.old.vmaddr.end   = 0;
			dseg.old.offset.start = 0;
			dseg.old.offset.end   = 0;
			dseg.old.buf = 0;
			
			
			dseg.fix.vmaddr.start = tseg.fix.vmaddr.end;
			dseg.fix.vmaddr.end   = tseg.fix.vmaddr.end;
			dseg.fix.offset.start = tseg.fix.offset.end;
			dseg.fix.offset.end   = tseg.fix.offset.end;
			dseg.fix.buf = fbuf + text->vmsize;
			
			dseg.dvmaddr = dseg.fix.vmaddr.start - dseg.old.vmaddr.start;
			dseg.doffset = dseg.fix.offset.start - dseg.old.offset.start;

		}*/
	}
	
	/*
	CommonLog("Outputting stream...\n");
	for(int i=0x149e; i<0x14a2; i++)
	{
		fprintf(stderr, "%02x", ((uint8_t*)fbuf)[i]);
	}
	fprintf(stderr, "\n");
	{
		uint16_t msb = *(uint16_t*)(fbuf+0x149e) & 0x7ff;
		uint16_t lsb = *(uint16_t*)(fbuf+0x14a0) & 0x7ff;
		uint32_t res = (msb<<12) | (lsb<<1);
		if(res & (1<<22))
			res |= 0xFF800000;
		
		CommonLog("offset = %08x %08x %08x", res, msb, lsb);
			
	}
	
	exit(1);
	*/
	
	
	
	header = (mach_header*) fbuf;
	
	// "64k should be plenty" - Bill Gates
	
	
	
	rebase_info rebase = {0, 0, 0, dsegs[0].fix.buf};
	//relocation_table reloc = {0,0,0};
	
	
	// for each data segment
	{
		uint32_t foundt = 0;
		uint32_t foundd = 0;
		uint32_t missed = 0;
		uint32_t missed_z = 0;
		uint32_t missed_dyld = 0;

		for(int ds = 0; ds < ndsegs; ds++)
		{
			seg_adjust* segment = &dsegs[ds];
			
			//CommonLog("Now %d entries", reloc.nEntries);

			dyld_cache_mapping_info* mapping = (dyld_cache_mapping_info*) (dyld_buf + dyldHead->mappingOffset);
			uint32_t dataCacheOffset = mapping[1].address;
			
			
			//CommonLog("number of mappings: %d", dyldHead->mappingCount);
			//for(int i=0; i<dyldHead->mappingCount; i++)
			//{
			//	CommonLog("Entry %d: %016lx %016lx %016lx %08x %08x", i, mapping[i].address, mapping[i].size, mapping[i].fileOffset, mapping[i].maxProt, mapping[i].initProt);
			//
			//}
			
			//CommonLog("Slide info: %08lx %016lx", dyldHead->slideInfoOffset, dyldHead->slideInfoSize * 32);
			
			// dyld_cache_slide_info* entries_count
		
			dyld_cache_slide_info* slide = (dyld_cache_slide_info*) (dyld_buf + dyldHead->slideInfoOffset);
			//int slideSize = dyldHead->slideInfoSize;
		
			uint16_t *slide_toc_index  = (uint16_t*) ((uintptr_t)slide + slide->toc_offset);
			dyld_cache_slide_info_entry* slide_entries = (dyld_cache_slide_info_entry*) ((uintptr_t)slide + slide->entries_offset);
			
			
			uint64_t vmoffset_start = segment->old.vmaddr.start - dataCacheOffset;
			uint64_t vmoffset_end = segment->old.vmaddr.end - dataCacheOffset;
			
			CommonLog("processing data segment: %08Lx %08Lx (%08x)", vmoffset_start, vmoffset_end, segment->old.vmaddr.start);
			

			for(uint32_t i = (vmoffset_start) / 0x1000, k=0; i < (segment->old.vmaddr.end - dataCacheOffset + 0xFFF) / 0x1000; i++, k++)
			{
				// bits: map a 4-byte relocation offset to a 1-bit value
				
				uint8_t *bits = slide_entries[slide_toc_index[i]].bits;
				uintptr_t pagebase = segment->fix.buf + 0x1000 * k;
				//CommonLog("base = %p", 0x1000*(i+k) + 0);
				
				for(uint32_t j=0; j<0x1000/4; j++)
				{
					
					if(bits[j>>3] & (1<<(j&7)))
					{
						uint64_t realvaddr = 0x1000*(i) +j*4;
						if(realvaddr < vmoffset_start || realvaddr > vmoffset_end)
							continue;
						//CommonLog("val = %p", realvaddr);
						
						
						if(seg_virtresolves((uint32_t*)(pagebase + j*4), &tseg, 1))
						{
							foundt++;
							
							push_rebase_entry(&rebase, (pagebase + j*4));
							//CommonLog("0");
							continue;
							
						//	entry_table[k].bits[j>>3] |= (1<<(j&7));
						}
						{
							if (seg_virtresolves((uint32_t*)(pagebase + j*4), dsegs, ndsegs))
							{
							//	CommonLog("Found entry! %d", ds2);
								foundd++;
								push_rebase_entry(&rebase, (pagebase + j*4));
								continue;
							}
						}
						{
							if(in_dyld_cache(*(uint32_t*)(pagebase + j*4)))
							{
							//	CommonLog("%08x", *(uint32_t*)(pagebase + j*4));
								
								missed_dyld++;
							}
							else if(*(uint32_t*)(pagebase + j*4) == 0)
							{
								missed_z++;
							}
							else
							{
								CommonLog("WARNING: slide did not match %x : %x", segment->fix.vmaddr.start + j*4, *(uint32_t*)(pagebase + j*4));
								missed++;
							}
							
						}
						//CommonLog("2");

					}
				}
				//PANIC("Done trying");
			}
			CommonLog("Easily resolved %d text, %d data references.  Missed %d external, %d zero, and %d other symbols.", foundt, foundd, missed_dyld, missed_z, missed);
		}
		//CommonLog("Now %d entries", rebase.count);


		//exit(1);
	}
	
	//seg_adjust dseg = dsegs[0];
	
	
	
	struct section* __objc_extradata = NULL;
	//if(objc)
	{
		
		struct section __objc_extradata_raw = 
			{
				"__objc_xtradata",
				"",
				//"__DATA",
				//dseg.old.vmaddr.end,
				dsegs[ndsegs-1].fix.vmaddr.end,
				0,
				//dseg.old.offset.end,
				dsegs[ndsegs-1].fix.offset.end,
				2,
				0,
				0,
				0,
				0,
				0
			};
		memcpy(&(__objc_extradata_raw.segname), datas[ndsegs-1]->segname, 16);
		__objc_extradata = append_section(fbuf, &__objc_extradata_raw);
		
		if(__objc_extradata)
		{
		//	CommonLog("objc_extradata %x %x", __objc_extradata->addr, __objc_extradata->offset);
		}
	}
	
	
	segment_command* linkedit = NULL;
	
	symtab_command* sym = NULL;
	dysymtab_command* dsym = NULL;
	dyld_info_command* dinfo = NULL;
	linkedit_data_command* fnstart = NULL;
	linkedit_data_command* dic = NULL;
		
	
	section* __objc_methname = NULL;
	section* __objc_classname = NULL;
	section* __objc_methtype = NULL;
	section* __cstring = NULL;
	
	
	section* __stub_helper = NULL;
	/*
	section* __cfstring = NULL;
	section* __objc_selrefs = NULL;
	
	section* __nl_symbol_ptr = NULL;
	section* __objc_classlist = NULL;
	section* __objc_data = NULL;
	section* __objc_const = NULL;
	*/
	section* __objc_classlist = NULL;
	section* __objc_protolist = NULL;
	section* __objc_catlist = NULL;
	section* __objc_const = NULL;
	
	section* __data = NULL;
	section* __dyld = NULL;
	
	section* __text = NULL;
	section* __pss = NULL;
	section* __la_symbol_ptr = NULL;
	//section* __objc_data = NULL;
	
	
	bool objc = 0;
	
	{
		uintptr_t lcptr = fbuf + sizeof(mach_header);
		
		for(uint32_t i=0; i<ncmds; i++)
		{
			uint32_t cmd = ((load_command*) lcptr)->cmd;
			if(lcptr > fbuf + header->sizeofcmds + sizeof(mach_header))
				PANIC("load command went beyond size of table");
		
			switch(cmd)
			{
			case LC_SEGMENT:
				{
					
					
					segment_command* seg = (segment_command*) lcptr;
					if(!strcmp(seg->segname, SEG_TEXT))
					{
					//	seg_applyoffset(seg, &tseg);
						
						int nsects = seg->nsects;
						section* sects = (section*) ((uintptr_t) seg + sizeof(segment_command));
						for(int j=0; j<nsects; j++)
						{
							section* sect = &sects[j];
							int type = sect->flags & SECTION_TYPE;
							switch(type)
							{
								case S_REGULAR:
								{
									if(!strncmp(sect->sectname, "__text", 16))
									{
										__text = sect;
									}
									else if(!strncmp(sect->sectname, "__stub_helper", 16))
									{
										//stub32_fix(fbuf, sect, &tseg, dsegs, ndsegs, 0);
										__stub_helper = sect;
									}
									else if(!strncmp(sect->sectname, "__const", 16))
									{
									//	abs32_fix(fbuf, sect, &tseg, &dseg);
									}
									
									/*
									else if(!strncmp(sect->sectname, "__gcc_except_tab", 16))
									{
										// just one address? 
										//rel32 fix needed
									//	abs32_fix(fbuf, sect, &tseg, &dseg);
									}*/
									else
									{
									
									//	CommonLog("Not handling section of type %s (%.16s.%.16s)", section_types[type], sect->segname, sect->sectname);
									}
									break;
								}
								case S_SYMBOL_STUBS:
								{
									if(!strncmp(sect->sectname, "__picsymbolstub4", 16))
									{
										__pss = sect;
										//stub32_fix(fbuf, sect, &tseg, dsegs, ndsegs, 0);//1);
									//	break;
									}
									else
									{
									
										//PANIC("Not handling section of type %s (%.16s.%.16s)", section_types[type], sect->segname, sect->sectname);
									}
									break;
								}
								case S_CSTRING_LITERALS:
								{
									if(!strncmp(sect->sectname, "__objc_methname", 16))
									{
										objc = 1;
										__objc_methname = sect;
									}
									else if(!strncmp(sect->sectname, "__objc_classname", 16))
									{
										objc = 1;
										__objc_classname = sect;
									}
									else if(!strncmp(sect->sectname, "__objc_methtype", 16))
									{
										objc = 1;
										__objc_methtype = sect;
									}
									else if(!strncmp(sect->sectname, "__cstring", 16))
									{
										__cstring = sect;
									}
									else
									{
									
									//	CommonLog("Not handling section of type %s (%.16s.%.16s)", section_types[type], sect->segname, sect->sectname);
									}
									break;
										
								}
								
								default:
									CommonLog("Not handling section of type %s (%.16s.%.16s)", section_types[type], sect->segname, sect->sectname);
							}
						}
					}
					
					if(!strncmp(seg->segname, SEG_DATA, 16) || !strncmp(seg->segname, SEG_DATA_DIRTY, 16) || !strncmp(seg->segname, SEG_DATA_CONST, 16))
					//if(!strcmp(seg->segname, SEG_DATA))
					{
					//	seg_applyoffset(seg, &dseg);
						int nsects = seg->nsects;
						section* sects = (section*) ((uintptr_t) seg + sizeof(segment_command));
						for(int j=0; j<nsects; j++)
						{

							section* sect = &sects[j];
							int type = sect->flags & SECTION_TYPE;
							switch(type)
							{
								case S_REGULAR:
								{
									if(!strncmp(sect->sectname, "__const", 16))
									{
										extz_fix(fbuf, sect, &tseg, dsegs, ndsegs);
									}
									else if(!strncmp(sect->sectname, "__data", 16))
									{
										__data = sect;
									}
									else if(!strncmp(sect->sectname, "__dyld", 16))
									{
										__dyld = sect;	
									}
									
									else if(!strncmp(sect->sectname, "__cfstring", 16))
									{
										extz_fix(fbuf, sect, &tseg, dsegs, ndsegs);
									}
									else if(!strncmp(sect->sectname, "__objc_data", 16))
									{
										objc = 1;
									//	__objc_data = sect;
										
										// disable the z bit so we can find problems :)
									
										extz_fix(fbuf, sect, &tseg, dsegs, ndsegs);
									}
									else if(!strncmp(sect->sectname, "__objc_classlist", 16))
									{
										objc = 1;
									//	CommonLog("Found __objc_classlist");
										__objc_classlist = sect;
										extz_fix(fbuf, sect, &tseg, dsegs, ndsegs);
									}
									else if(!strncmp(sect->sectname, "__objc_catlist", 16))
									{
										objc = 1;
									//	CommonLog("Found __objc_catlist");
										__objc_catlist = sect;
									//	CommonLog("size is %x", sect->size);
										extz_fix(fbuf, sect, &tseg, dsegs, ndsegs);
									}
									else if(!strncmp(sect->sectname, "__objc_protolist", 16))
									{
										objc = 1;
									//	CommonLog("Found __objc_protolist");
										__objc_protolist = sect;
										extz_fix(fbuf, sect, &tseg, dsegs, ndsegs);
									}
									else if(!strncmp(sect->sectname, "__objc_protorefs", 16))
									{
										objc = 1;
										extz_fix(fbuf, sect, &tseg, dsegs, ndsegs);
									}
									else if(!strncmp(sect->sectname, "__objc_classrefs", 16))
									{
										objc = 1;
										extz_fix(fbuf, sect, &tseg, dsegs, ndsegs);
									}
									else if(!strncmp(sect->sectname, "__objc_superrefs", 16))
									{
										objc = 1;
										extz_fix(fbuf, sect, &tseg, dsegs, ndsegs);
									}
									
									
									else if(!strncmp(sect->sectname, "__objc_const", 16))
									{
										objc = 1;
										__objc_const = sect;
									}
									else if(!strncmp(sect->sectname, "__objc_imageinfo", 16))
									{
										objc = 1;
										((uint32_t*)(fbuf + sect->offset))[1] &= ~8;
										
									}
									else if(!strncmp(sect->sectname, "__objc_ivar", 16))
									{
										objc = 1;
									}
									else if(!strncmp(sect->sectname, "__objc_xtradata", 16))
									{
									}
									
									else
									{
										CommonLog("Not handling section of type %s (%.16s.%.16s)", section_types[type], sect->segname, sect->sectname);
									}
									break;
								}
								case S_NON_LAZY_SYMBOL_POINTERS:
								{
									if(!strncmp(sect->sectname, "__nl_symbol_ptr", 16))
									{
										extz_fix(fbuf, sect, &tseg, dsegs, ndsegs);
									}
									else
									{
										CommonLog("Not handling section of type %s (%.16s.%.16s)", section_types[type], sect->segname, sect->sectname);
									}
									break;
								}
								case S_LAZY_SYMBOL_POINTERS:
								{
									if(!strncmp(sect->sectname, "__la_symbol_ptr", 16))
									{
										// &tseg, dsegs, ndsegs,
										__la_symbol_ptr = sect;
										lazy_fix(fbuf, sect, __stub_helper,  &rebase);
									}
									else
									{
										CommonLog("Not handling section of type %s (%.16s.%.16s)", section_types[type], sect->segname, sect->sectname);
									}
									break;
								}
								case S_LITERAL_POINTERS:
								{
									if(!strncmp(sect->sectname, "__objc_selrefs", 16))
									{
										objc = 1;
										resolve_methnames(fbuf, sect, __objc_methname, &tseg, &rebase);
									}
									
								}
								case S_ZEROFILL:
									break;
								default:
									CommonLog("Not handling section of type %s (%.16s.%.16s)", section_types[type], sect->segname, sect->sectname);
							}
						}
					}
					else if(!strcmp(seg->segname, SEG_LINKEDIT))
					{
						linkedit = seg;
					}
				}
				break;
			
			case LC_SUB_FRAMEWORK:
			case LC_LOAD_UPWARD_DYLIB:
				break;
			/*
			case LC_SUB_FRAMEWORK:
			case LC_REQ_DYLD | LC_LOAD_UPWARD_DYLIB:
			case LC_DATA_IN_CODE:
				break;
				*/
			case LC_SEGMENT_SPLIT_INFO:
				
				// no longer in cache.  ignore.  or delete??
				break;
				/*
				
			 // 0x1e
			
				linkedit_data_command* splitinfo = (linkedit_data_command*) lcptr;
				
				CommonLog("split info: %x %x", splitinfo->dataoff, splitinfo->datasize);
				//exit(1);
				
				break;
				extract_file/2631: Load command 1e not yet processed
				extract_file/2631: Load command 80000023 not yet processed
				extract_file/2631: Load command 80000023 not yet processed
				extract_file/2631: Load command 80000023 not yet processed
				extract_file/2631: Load command 80000023 not yet processed
				extract_file/2631: Load command 80000023 not yet processed
				extract_file/2631: Load command 80000023 not yet processed
				extract_file/2631: Load command 80000023 not yet processed
				extract_file/2631: Load command 12 not yet processed
				extract_file/2631: Load command 29 not yet processed
				*/
			
			
			case LC_SYMTAB:
				sym = (symtab_command*) lcptr;
				break;
			case LC_DYSYMTAB:
				dsym = (dysymtab_command*) lcptr;
				break;
			case LC_LOAD_DYLIB:
				break;
			case LC_ID_DYLIB:
				break;
			case LC_UUID:
				break;
			case LC_DYLD_INFO:
			case LC_DYLD_INFO_ONLY:
				dinfo = (dyld_info_command*) lcptr;
				break;
			case LC_VERSION_MIN_IPHONEOS:
				break;
			case LC_FUNCTION_STARTS:

				fnstart = (linkedit_data_command*) lcptr;
				//CommonLog("fnstart info: %x %x", fnstart->dataoff, fnstart->datasize);
				break;
			case LC_SOURCE_VERSION:
				break;
			case LC_REEXPORT_DYLIB:
				break;
			case LC_LOAD_WEAK_DYLIB:
				break;
			case LC_DATA_IN_CODE:
				dic = (linkedit_data_command*) lcptr;
				break;
				
			default:
				CommonLog("Load command %x not yet processed", cmd);
				break;
			}
			//CommonLog("Load command %x at %x\n", cmd, (uint32_t)((uintptr_t) lcptr -fbuf));

			lcptr += ((load_command*) lcptr)->cmdsize;
		}
		
	}
	
	if(__text)
	{
		text_dellvm(fbuf, __text, &tseg, dsegs, ndsegs);
	}
	if(__text && __pss && __stub_helper)
	{
		text_fixlongcalls(fbuf, __text, __pss, __la_symbol_ptr, &tseg, dsegs, ndsegs);
	}
	

	if(objc && !__objc_extradata)
	{
		PANIC("Unable to add critical objc_xtradata section");
	}
	
	
	
	//__objc_catlist->size = __objc_protolist->offset - __objc_catlist->offset;
	
	
//	linkedit = find_segment(fbuf, SEG_LINKEDIT);
	
	// datas[ndsegs-1]->segname  /// need way to integrate with other two segments...
	
	
	
	if(__objc_classlist)
	{
		fix_from_classlist(fbuf, __objc_classlist, __objc_methname, __objc_extradata, &tseg, dsegs, ndsegs, &rebase);
	}
	
	//CommonLog("%p %p %p %p %p %p %p", fbuf, __objc_protolist, __objc_methname, __objc_extradata, &tseg, &dseg, &rebase);
	if(__objc_protolist)
	{
		fix_from_protolist(fbuf, __objc_protolist, __objc_methname, __objc_extradata, &tseg, dsegs, ndsegs, &rebase);
	}
	if(__objc_const)
		rebasez_fix(fbuf, __objc_const, &tseg, dsegs, ndsegs, &rebase);	// WHY??? :(
	
	if(__objc_catlist)
	{
		fix_from_catlist(fbuf, __objc_catlist, __objc_methname, __objc_extradata, &tseg, dsegs, ndsegs, &rebase);
	}
	
	
	/*
	
	if(__data)
	{
	//	extz_fix(fbuf, __data, &tseg, &dseg);
	}
	
	if(__dyld) // ????
	{
		extz_fix(fbuf, __dyld, &tseg, dsegs, ndsegs);
	}
	*/

	if(__objc_extradata)
	{
		int nExtraData = vmalign(__objc_extradata->size);
		__objc_extradata->size = nExtraData;
		
		segment_command* data = find_segment(fbuf, SEG_DATA);
		data->vmsize += nExtraData;
		data->filesize += nExtraData;
		nfile += nExtraData;
	}
	
	
	
	//CommonLog("2objc_extradata %x %x", __objc_extradata->addr, __objc_extradata->offset);
	
	/*
	if(__cfstring)
	{
		struct __CFString
		{
			uint32_t dc1;
			uint32_t dc2;
			uint32_t ptr;
			uint32_t dc4;
		};
		__CFString* pointers = (__CFString*) (fbuf + __cfstring->offset);
		int nPointers = __cfstring->size / sizeof(__CFString);
		for(int i = 0; i<nPointers; i++)
		{
			if(!resolve_local(&pointers[i].ptr))
			{
				PANIC("CFString out of range");
			}
		}
	}
	
	if(__objc_classlist)
	{
		uint32_t* pointers = (uint32_t*) (fbuf + __objc_classlist->offset);
		int nPointers = __objc_classlist->size / sizeof(uint32_t);
		//memset(pointers, 0, nPointers * sizeof(uint32_t));
		
		uintptr_t dptr = fbuf + data->fileoff - data->vmaddr;
		
		for(int i = 0; i<nPointers; i++)
		{
			uint32_t pointer = pointers[i];
			pointer += dvmaddr_data;
			/ *
			struct __Objc_class
			{
				uint32_t meta;
				uint32_t super;
				uint32_t __empty_cache;
				uint32_t __empty_vtable;
				uint32_t impl;
			};
			
			__Objc_class* cls = dptr + pointer;
			__Objc_class* meta = cls->meta;
			resolve_local(&cls->super);
			
			struct _objc_class_ro
			{
				
			}
			* /
		}
	}
	
	/ *
	if(__objc_data)
	{
		uint32_t* pointers = (uint32_t*) (fbuf + __objc_data->offset);
		int nPointers = __objc_data->size / sizeof(uint32_t);
		//memset(pointers, 0, nPointers * sizeof(uint32_t));
		for(int i = 0; i<nPointers; i++)
		{
			uint32_t pointer = pointers[i];
			if(!pointer)
				continue;
			if(pointer >= oldtext->vmaddr && pointer < oldtext->vmaddr + oldtext->vmsize)
			{
				pointers[i] += dvmaddr_text;
				CommonLog("Locally fixed");
			}
			else if(pointer >= olddata->vmaddr && pointer < olddata->vmaddr + olddata->vmsize)
			{
				pointers[i] += dvmaddr_data;
				CommonLog("Locally fixed");
			}
			else
			{
			//	uintptr_t addr = locate_address(pointer, 1);
				
			//	PANIC("CFString out of range: %d is %x %p", i, pointer, addr);
			}
		//	CommonLog("Changed %x to %x", pointer, pointers[i]);
		}
	}* /
	*/
	
	// symc
	//nfile;
	nfile = vmalign(nfile);
	
	linkedit->vmaddr = nfile;
	linkedit->fileoff = nfile;
	
	/*
	if(dsym)
	{
		if(reloc.nEntries)
		{
			memcpy((void*)(fbuf+nfile), (void*)reloc.entries, reloc.nEntries * sizeof(relocation_info));
			dsym->locreloff = nfile;
			dsym->nlocrel = reloc.nEntries;
			nfile += reloc.nEntries * sizeof(relocation_info);
		}
		if(reloc.entries)
			free(reloc.entries);
	}
	else
	{
		PANIC("No dysymtable?");
	}*/
	
	if(dinfo)
	{
		if(dinfo->rebase_size)
		{
			fprintf(stderr, "%x = ", (uint32_t) ((uintptr_t)dinfo - (uintptr_t) fbuf));
			for(uint32_t i=0; i<sizeof(dyld_info_command); i++)
			{
				fprintf(stderr, "%02x", ((unsigned char*)dinfo)[i]);
			}
			fprintf(stderr, "\n");
			PANIC("HELP!"); // ??????
		}
		
		push_rebase_entry(&rebase, 0);
		int nrebase = rebase.currptr + 1 - rebase.buf;
		if(nrebase > 1)
		{
			dinfo->rebase_off = nfile;
			
			memcpy((void*)(fbuf + nfile), (void*)(rebase.buf), nrebase);
			dinfo->rebase_size = nrebase;
			nfile += dinfo->rebase_size;
			
		}
		
		if(dinfo->bind_off)
		{
		//	CommonLog("Copying %d bytes", dinfo->bind_size);
			
			memcpy((void*)(fbuf + nfile), (void*)(dyld_buf + dinfo->bind_off), dinfo->bind_size);
			
		//	CommonLog("Done.");
			dinfo->bind_off = nfile;
			nfile += dinfo->bind_size;
		}
		if(dinfo->weak_bind_off)
		{			
			memcpy((void*)(fbuf + nfile), (void*)(dyld_buf + dinfo->weak_bind_off), dinfo->weak_bind_size);
			dinfo->weak_bind_off = nfile;
			nfile += dinfo->weak_bind_size;
		}
		if(dinfo->lazy_bind_off)
		{			
			memcpy((void*)(fbuf + nfile), (void*)(dyld_buf + dinfo->lazy_bind_off), dinfo->lazy_bind_size);
			dinfo->lazy_bind_off = nfile;
			nfile += dinfo->lazy_bind_size;
		}
		if(dinfo->export_off)
		{
			memcpy((void*)(fbuf + nfile), (void*)(dyld_buf + dinfo->export_off), dinfo->export_size);
			dinfo->export_off = nfile;
			
		//	int oldsize = dinfo->export_size;
			
			dinfo->export_size = hack_export_commands(fbuf + dinfo->export_off, dinfo->export_size, &tseg, dsegs, ndsegs);
			
		//	CommonLog("Old export size: %x new size: %x", oldsize, dinfo->export_size);
			nfile += dinfo->export_size;
		}
		
		
	}
	else
	{
		PANIC("No dyld_info_only??");
	}
	
	if(fnstart)
	{
		memcpy((void*)(fbuf + nfile), (void*)(dyld_buf + fnstart->dataoff), fnstart->datasize);
		fnstart->dataoff = nfile;
		nfile += fnstart->datasize;
	}
	
	if(dic)
	{
		memcpy((void*)(fbuf + nfile), (void*)(dyld_buf + dic->dataoff), dic->datasize);
		dic->dataoff = nfile;
		nfile += dic->datasize;
	}
	
	// split info was removed already!
	/*
	if(splitinfo)
	{
		//memcpy((void*)(fbuf + nfile), (void*)(dyld_buf + fnstart->dataoff), fnstart->datasize);
		splitinfo->dataoff = nfile;
		splitinfo->datasize = 0;
//		nfile += fnstart->datasize;
	}
	*/
	
	{
		//CommonLog("Syms %x(%x) Stroff %x(%x)", sym->symoff, sym->nsyms*sizeof(struct nlist), sym->stroff, sym->strsize);
		// duplicate sym table
		int nsyms = sym->nsyms;
		memcpy((void*)(fbuf + nfile), (void*)(dyld_buf + sym->symoff), nsyms * sizeof(struct nlist));
		sym->symoff = nfile;
		nfile += nsyms * sizeof(struct nlist);
		
		// recover strings
		struct nlist* nl = (struct nlist*) (fbuf + sym->symoff);
		const char* strs = (const char*) (dyld_buf + sym->stroff);
		
		
		
		
		
		// dysymc.  why it has to be here makes no sense
		{
			{
				int nIndirectSyms = dsym->nindirectsyms;
				if(nIndirectSyms)
				{
					memcpy((void*)(fbuf + nfile), (void*)(dyld_buf + dsym->indirectsymoff), nIndirectSyms * sizeof(uint32_t));
				}
				dsym->indirectsymoff = nfile;
				nfile += nIndirectSyms * sizeof(uint32_t);
			}

			{
				if(dsym->nextrel)
					PANIC("Sorry, we don't handle nextrel!");
				if(dsym->extreloff)
					dsym->extreloff = 0;//nfile;
			}
		}
		
		
		
		
		
		
		struct nlist* nl2 = NULL;
		int nsyms2 = 0;
		const char *cacheStrings = (const char*) ((uintptr_t)localSymbols + localSymbols->stringsOffset);
		{
			dyld_cache_local_symbols_entry *localEntry = NULL;
			
			{
				dyld_cache_local_symbols_entry *localEntries = (dyld_cache_local_symbols_entry *) (((uintptr_t)localSymbols) + localSymbols->entriesOffset);
				
				for(uint32_t i=0; i < localSymbols->entriesCount; i++)
				{
				//	CommonLog("%d %d", i, localSymbols->entriesCount);

					if(localEntries[i].dylibOffset == tseg.old.vmaddr.start - dyld_vmbase) // + shared_cache_offset)
					{
						localEntry = &localEntries[i];
						break;
					}
				}
			}
			if(localEntry)
			{
				nl2 = (struct nlist *) ((uintptr_t)localSymbols + localSymbols->nlistOffset);
				nl2 = &nl2[localEntry->nlistStartIndex];
				nsyms2 = localEntry->nlistCount;
			}
		}
		
		//CommonLog("");
		
		sym->stroff = nfile;
		for(int i=0; i<nsyms; i++)
		{
			const char* str = &strs[ nl[i].n_un.n_strx];
			nl[i].n_un.n_strx = nfile - sym->stroff;
			if((str[0] == 0 || !strcmp(str, "<redacted>")) && nl2)
			{
				for(int j=0; j<nsyms2; j++)
				{
					if(nl2[j].n_value == nl[i].n_value)
					{
						str = &cacheStrings[nl2[i].n_un.n_strx];
					}
				}
			}
			
			{
				// these were already resolved?? not good.
				
				uint32_t* nvalue = &nl[i].n_value;//; + shared_cache_slide;
				if(*nvalue)
				{
					/*
					if(!in_dyld_cache(*nvalue))
					{
						// *nvalue += shared_cache_slide;
					}
					*/
					if(seg_virtresolves(nvalue, dsegs, ndsegs, 1))
					{
					//	PANIC("Missed virtresolve");
					}
					else if(seg_virtresolves(nvalue, &tseg, 1, 1))
					{
					//	PANIC("Missed virtresolve");
					}
					else
					{
					//	PANIC("missed completely! :(");
					}
				}
				
				//CommonLog("%x %s", *nvalue, str);
			}
					
			
			for(; *str !=0; str++)
			{
				((char*)fbuf) [nfile++] = *str;
			}
			((char*)fbuf) [nfile++] = 0;
			
		}
		sym->strsize = nfile - sym->stroff;
		
		
		
		
		
		//exit(1);
		/*
		memcpy(fbuf + nfile;
		symoff = 
		nfile += symoff;
		linkedit->file_offset;
		*/
	}
	
	
	
	
	{
		dyld_cache_mapping_info* mapping = (dyld_cache_mapping_info*) (dyld_buf + dyldHead->mappingOffset);
		uint32_t dataCacheOffset = mapping[1].address;
	
		dyld_cache_slide_info* slide = (dyld_cache_slide_info*) (dyld_buf + dyldHead->slideInfoOffset);
		//int slideSize = dyldHead->slideInfoSize;
	
		uint16_t *slide_toc_index  = (uint16_t*) ((uintptr_t)slide + slide->toc_offset);
		dyld_cache_slide_info_entry* slide_entries = (dyld_cache_slide_info_entry*) ((uintptr_t)slide + slide->entries_offset);
		
		uint32_t missed_dyld = 0;
		
		for(int ds=0; ds<ndsegs; ds++)
		{
			for(uint32_t i = (dsegs[ds].old.vmaddr.start - dataCacheOffset) / 0x1000, k=0; i < (dsegs[ds].old.vmaddr.end - dataCacheOffset) / 0x1000; i++, k++)
			{
				uint8_t *bits = slide_entries[slide_toc_index[i]].bits;
				uintptr_t pagebase = dsegs[ds].fix.buf + 0x1000 * k;
				
				for(uint32_t j=0; j<0x1000/4; j++)
				{
					if(bits[j>>3] & (1<<(j&7)))
					{
						if(in_dyld_cache(*(uint32_t*)(pagebase + j*4)))
						{
							missed_dyld++;
						}
					}
				}
				//PANIC("Done trying");
			}
		}
		
		
		if(missed_dyld)
		{
			CommonLog("Still missed %d slide external references! :(", missed_dyld);
		}
		else
		{
			CommonLog("Yay! Got rid of all the slide external references!");
		}
	}
	
	
	//nfile = vmalign(nfile);
	
	linkedit->filesize = nfile - linkedit->fileoff;
	
	//CommonLog("Aligning to page: %x %x", linkedit->filesize, vmalign(linkedit->filesize));
	linkedit->vmaddr = linkedit->fileoff;//vmalign(linkedit->filesize);
	linkedit->vmsize = linkedit->filesize;
	
	// string operations ****
	
	//CommonLog("Closing file.  Size = %x\n", nfile);
	
	fdtrimclose(fd, nfile, &fbuf);
	
	
	
}

void Usage()
{
	fprintf(stderr, "Usage:\n  decache -c cache -x <extract name> -o <out file name>\n  Sorry, no wildcards :(\n");
	exit(1);
}

uint32_t htoi(const char *str)
{
	//	NSLine();
	
	uint32_t retval=0;
	for(int i=0; i<8; i++)
	{
		char c=str[i];
		if(c==0)
			break;
		c &=0x4F;
		if(c&0x40)
			c-=0x37;
		retval = (retval << 4) + c;
	}
	return retval;
}


int main(int argc, char** argv)
{
	int ch;
	
	const char* dyld_name = NULL; //"dyld_shared_cache_armv7";
	
	//const char* extractname = "/System/Library/PrivateFrameworks/Weather.framework/Weather";
	const char* extractname = NULL;//"MapKit";
	
	const char* outname = NULL;
	
//	uint32_t ptraddr = 0;
//	uint32_t physaddr = 0;
//	uint32_t scanval = 0;
	
	while((ch = getopt (argc, argv, "c:x:o:")) != -1)
	{
		switch(ch)
		{
			case 'c':
				dyld_name = optarg;
				break;
			case 'x':
				extractname = optarg;
				break;
			case 'o':
				outname = optarg;
				break;
				/*
			case 'p':
				ptraddr = htoi(optarg);
				break;
			case 'a':
				physaddr = htoi(optarg);
				break;
			case 's':
				scanval = htoi(optarg);
				break;
				*/
			default:
				CommonLog("Argument %c = %s", ch, optarg);
		}
	}
	
	if(!extractname || !strlen(extractname))
	{
		if(dyld_name)
		{
			// need to print out all the dylds
		}
		else
		{
			Usage();
		}
	}
	
	if(!outname && extractname)
	{
		outname = strrchr(extractname, '/');
		if(!outname || !strlen(outname))
			outname = extractname;
		else
			outname++;
	}
	
	
	int dyld_fd = open(dyld_name, O_RDONLY);
	
	//dyld_fd = dup(dyld_fd);
	//dyld_fd = fcntl(dyld_fd, F_DUPFD, dyld_fd);
	//dyld_fd = 256;
	
	if(dyld_fd == -1)
		PANIC("Could not open dyld cache %s", dyld_name);
	int dyld_n = fsize(dyld_fd);
	
#ifdef READ_VM
	uint32_t cacheFileSize = dyld_n;
	uint32_t cacheAllocatedSize = (cacheFileSize + 4095) & (-4096);
	uint8_t* mappingAddr = NULL;
	if ( vm_allocate(mach_task_self(), (vm_address_t*)(&mappingAddr), cacheAllocatedSize, VM_FLAGS_ANYWHERE) != KERN_SUCCESS )
		PANIC("can't vm_allocate cache of size %u", cacheFileSize);

	fcntl(dyld_fd, F_NOCACHE, 1);
	uint32_t readResult = pread(dyld_fd, mappingAddr, cacheFileSize, 0);

	if(readResult != cacheFileSize)
		PANIC("Unable to load entire cache into memory :(");

	dyld_buf = (uintptr_t) mappingAddr;
#else
	// MAP_NOCACHE.  makes no flippin' difference :(
	fcntl(dyld_fd, F_NOCACHE, 1);
	
//	fcntl(dyld_fd, F_GLOBAL_NOCACHE, 0);
//	dyld_buf = (uintptr_t) mmap(NULL, dyld_n, PROT_READ, MAP_ANON | MAP_NOCACHE, 0, 0);
//	dyld_buf = (uintptr_t) mmap(NULL, dyld_n, PROT_READ, MAP_SHARED, dyld_fd, 0);
	
	dyld_buf = (uintptr_t) mmap(NULL, dyld_n, PROT_READ, MAP_PRIVATE | MAP_NOCACHE, dyld_fd, 0);
//	msync((void*)dyld_buf, dyld_n, MS_SYNC | MS_INVALIDATE | MS_KILLPAGES);
#endif
		   
	
	
	
	
	dyldHead = (dyld_cache_header *)dyld_buf;
	dyld_vmbase = *(uint64_t *)(dyld_buf + dyldHead->mappingOffset);
	
	{
		dyld_cache_mapping_info* mapping = (dyld_cache_mapping_info*) (dyld_buf + dyldHead->mappingOffset);
		for(uint32_t i=0; i<dyldHead->mappingCount; i++)
		{
			if(mapping[i].address + mapping[i].size > dyld_vmextent)
			{
				dyld_vmextent = mapping[i].address + mapping[i].size;
			}
		//	CommonLog("(%x) %x -> %x", (uint32_t) mapping[i].fileOffset, (uint32_t) mapping[i].address, (uint32_t) (mapping[i].address + mapping[i].size));
		}
		
	}
	
	#ifdef TARGET_IPHONE
	uint64_t start_address;
	syscall(SYS_shared_region_check_np, &start_address); // 294
	shared_cache_slide = start_address - dyld_vmbase;
	
	//CommonLog("Slide = %x", shared_cache_slide);
	//exit(1);
	#else
	shared_cache_slide = 0;
	#endif
	
	
	//PANIC("imageOffset %x count %x", dyldHead->imagesOffset, dyldHead->imagesCount);
	
	image_infos = (dyld_cache_image_info*) (dyld_buf + dyldHead->imagesOffset);
	localSymbols = (dyld_cache_local_symbols_info*) (dyld_buf + dyldHead->localSymbolsOffset);
	
	/*
	if(scanval)
	{
		for(uintptr_t ptr = dyld_buf; ptr < dyld_buf + dyld_n; ptr +=4)
		{
			if(*(uint32_t*)ptr==scanval)
			{
				CommonLog("Found at %x", (uint32_t) (ptr - dyld_buf));
			//	exit(1);
			}
		}
		exit(1);
	}
	if(physaddr)
	{
		CommonLog("Value is %x", *(uint32_t*) ((uintptr_t) dyld_buf + physaddr));
		exit(1);
	}
	if(ptraddr)
	{
		if(ptraddr == locate_address(ptraddr, 1))
		{
			
		}
		exit(1);
	}
	*/
	
	//CommonLog("ptraddr = %x", ptraddr);
	//exit(1);
	
	
	
	if(!extractname)
	{
		char outpath[0x80];
		
		for(uint32_t i=0; i< dyldHead->imagesCount; i++)
		{
			//
			const char *filename = (const char *)(dyld_buf + image_infos[i].pathFileOffset);
			
			if(outname)
			{
				uint64_t vm_address = image_infos[i].address;
				uint32_t extract_offs = vm_address - dyld_vmbase;
				sprintf(outpath, "%s/%s", outname, filename);
				uintptr_t extract_buf = dyld_buf + extract_offs;

				fprintf(stderr, "%s\n", filename);
				extract_file(extract_buf, outpath);
			}
			else
			{
				fprintf(stderr, "%s\n", filename);
			}
		}
		exit(1);
	}
	else
	{
		uint32_t extract_offs;
		find_file(extractname, &extract_offs);
		uintptr_t extract_buf = dyld_buf + extract_offs;
		
		CommonLog("offset %x", extract_offs);
		extract_file(extract_buf, outname);
	}
	
	
	//print_vmaddr(dyld_buf, dyld_n);
	
	
#ifdef READ_VM	
	vm_deallocate(mach_task_self(), (vm_address_t)mappingAddr, cacheAllocatedSize);
#else
	munmap((void*)dyld_buf, dyld_n);
#endif
	close(dyld_fd);
	
//	CommonLog("Done!");
//	PANIC("EOF: %x %s", dyld_n, dyld_name);
}