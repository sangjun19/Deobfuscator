	.file	"sayam56_SocketProject_mirror_flatten.c"
	.text
	.globl	_TIG_IZ_wxNp_envp
	.bss
	.align 8
	.type	_TIG_IZ_wxNp_envp, @object
	.size	_TIG_IZ_wxNp_envp, 8
_TIG_IZ_wxNp_envp:
	.zero	8
	.globl	_TIG_IZ_wxNp_argc
	.align 4
	.type	_TIG_IZ_wxNp_argc, @object
	.size	_TIG_IZ_wxNp_argc, 4
_TIG_IZ_wxNp_argc:
	.zero	4
	.globl	_TIG_IZ_wxNp_argv
	.align 8
	.type	_TIG_IZ_wxNp_argv, @object
	.size	_TIG_IZ_wxNp_argv, 8
_TIG_IZ_wxNp_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"r"
.LC1:
	.string	"file_list.txt"
	.align 8
.LC2:
	.string	"Usage: getft <extension list> minimum 1 or maximum 3 extensions are allowed"
.LC3:
	.string	"No file found"
	.align 8
.LC4:
	.string	"tar -czf f23project/temp.tar.gz -T file_list.txt"
.LC5:
	.string	"rb"
.LC6:
	.string	"f23project/temp.tar.gz"
.LC7:
	.string	"fopen"
.LC8:
	.string	"Finished tarring files."
.LC9:
	.string	"Error creating tar.gz file"
.LC10:
	.string	"HOME"
.LC11:
	.string	"w"
	.text
	.globl	getft
	.type	getft, @function
getft:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$1248, %rsp
	movl	%edi, -1236(%rbp)
	movq	%rsi, -1248(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$2, -1176(%rbp)
.L38:
	cmpq	$28, -1176(%rbp)
	ja	.L41
	movq	-1176(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L4(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L4(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L4:
	.long	.L25-.L4
	.long	.L42-.L4
	.long	.L23-.L4
	.long	.L22-.L4
	.long	.L21-.L4
	.long	.L20-.L4
	.long	.L19-.L4
	.long	.L42-.L4
	.long	.L41-.L4
	.long	.L41-.L4
	.long	.L41-.L4
	.long	.L41-.L4
	.long	.L17-.L4
	.long	.L41-.L4
	.long	.L41-.L4
	.long	.L42-.L4
	.long	.L15-.L4
	.long	.L41-.L4
	.long	.L14-.L4
	.long	.L13-.L4
	.long	.L12-.L4
	.long	.L11-.L4
	.long	.L10-.L4
	.long	.L9-.L4
	.long	.L8-.L4
	.long	.L42-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L42-.L4
	.text
.L14:
	movq	-1216(%rbp), %rdx
	movq	-1208(%rbp), %rcx
	movq	-1200(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	filesearchWrite
	movq	-1200(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	leaq	.LC0(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -1088(%rbp)
	movq	-1088(%rbp), %rax
	movq	%rax, -1192(%rbp)
	movq	$22, -1176(%rbp)
	jmp	.L26
.L21:
	movq	-1248(%rbp), %rax
	movq	%rax, %rdi
	call	countExtensions
	movl	%eax, -1220(%rbp)
	movl	-1220(%rbp), %eax
	movl	%eax, -1224(%rbp)
	subl	$1, -1224(%rbp)
	movq	-1248(%rbp), %rax
	addq	$6, %rax
	movq	%rax, -1216(%rbp)
	movq	$16, -1176(%rbp)
	jmp	.L26
.L17:
	leaq	.LC2(%rip), %rax
	movq	%rax, -1152(%rbp)
	movq	-1152(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -1144(%rbp)
	movq	-1144(%rbp), %rdx
	movq	-1152(%rbp), %rsi
	movl	-1236(%rbp), %eax
	movl	$0, %ecx
	movl	%eax, %edi
	call	send@PLT
	movq	$7, -1176(%rbp)
	jmp	.L26
.L9:
	leaq	.LC3(%rip), %rax
	movq	%rax, -1136(%rbp)
	movq	-1136(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -1128(%rbp)
	movq	-1128(%rbp), %rdx
	movq	-1136(%rbp), %rsi
	movl	-1236(%rbp), %eax
	movl	$0, %ecx
	movl	%eax, %edi
	call	send@PLT
	movq	$25, -1176(%rbp)
	jmp	.L26
.L22:
	movq	-1192(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	leaq	-1040(%rbp), %rax
	leaq	.LC4(%rip), %rdx
	movl	$1024, %esi
	movq	%rax, %rdi
	movl	$0, %eax
	call	snprintf@PLT
	leaq	-1040(%rbp), %rax
	movq	%rax, %rdi
	call	system@PLT
	leaq	.LC5(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -1064(%rbp)
	movq	-1064(%rbp), %rax
	movq	%rax, -1184(%rbp)
	movq	$20, -1176(%rbp)
	jmp	.L26
.L15:
	cmpl	$0, -1224(%rbp)
	jne	.L28
	movq	$12, -1176(%rbp)
	jmp	.L26
.L28:
	movq	$6, -1176(%rbp)
	jmp	.L26
.L8:
	cmpq	$0, -1200(%rbp)
	jne	.L30
	movq	$21, -1176(%rbp)
	jmp	.L26
.L30:
	movq	$18, -1176(%rbp)
	jmp	.L26
.L11:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$28, -1176(%rbp)
	jmp	.L26
.L6:
	leaq	.LC2(%rip), %rax
	movq	%rax, -1152(%rbp)
	movq	-1152(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -1144(%rbp)
	movq	-1144(%rbp), %rdx
	movq	-1152(%rbp), %rsi
	movl	-1236(%rbp), %eax
	movl	$0, %ecx
	movl	%eax, %edi
	call	send@PLT
	movq	$15, -1176(%rbp)
	jmp	.L26
.L13:
	movq	-1184(%rbp), %rax
	movl	$2, %edx
	movl	$0, %esi
	movq	%rax, %rdi
	call	fseek@PLT
	movq	-1184(%rbp), %rax
	movq	%rax, %rdi
	call	ftell@PLT
	movq	%rax, -1120(%rbp)
	movq	-1120(%rbp), %rax
	movq	%rax, -1112(%rbp)
	movq	-1184(%rbp), %rax
	movq	%rax, %rdi
	call	rewind@PLT
	movq	-1112(%rbp), %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -1104(%rbp)
	movq	-1104(%rbp), %rax
	movq	%rax, -1096(%rbp)
	movq	-1112(%rbp), %rdx
	movq	-1184(%rbp), %rcx
	movq	-1096(%rbp), %rax
	movl	$1, %esi
	movq	%rax, %rdi
	call	fread@PLT
	movq	-1184(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	-1112(%rbp), %rdx
	movq	-1096(%rbp), %rsi
	movl	-1236(%rbp), %eax
	movl	$0, %ecx
	movl	%eax, %edi
	call	send@PLT
	movq	-1096(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$27, -1176(%rbp)
	jmp	.L26
.L19:
	cmpl	$3, -1224(%rbp)
	jle	.L32
	movq	$26, -1176(%rbp)
	jmp	.L26
.L32:
	movq	$0, -1176(%rbp)
	jmp	.L26
.L5:
	leaq	.LC8(%rip), %rax
	movq	%rax, -1168(%rbp)
	movq	-1168(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -1160(%rbp)
	movq	-1160(%rbp), %rdx
	movq	-1168(%rbp), %rcx
	movl	-1236(%rbp), %eax
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	write@PLT
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	remove@PLT
	movq	$1, -1176(%rbp)
	jmp	.L26
.L10:
	cmpq	$0, -1192(%rbp)
	jne	.L34
	movq	$23, -1176(%rbp)
	jmp	.L26
.L34:
	movq	$3, -1176(%rbp)
	jmp	.L26
.L20:
	leaq	.LC9(%rip), %rax
	movq	%rax, -1080(%rbp)
	movq	-1080(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -1072(%rbp)
	movq	-1072(%rbp), %rdx
	movq	-1080(%rbp), %rsi
	movl	-1236(%rbp), %eax
	movl	$0, %ecx
	movl	%eax, %edi
	call	send@PLT
	movq	$27, -1176(%rbp)
	jmp	.L26
.L25:
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	call	getenv@PLT
	movq	%rax, -1056(%rbp)
	movq	-1056(%rbp), %rax
	movq	%rax, -1208(%rbp)
	leaq	.LC11(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -1048(%rbp)
	movq	-1048(%rbp), %rax
	movq	%rax, -1200(%rbp)
	movq	$24, -1176(%rbp)
	jmp	.L26
.L23:
	movq	$4, -1176(%rbp)
	jmp	.L26
.L12:
	cmpq	$0, -1184(%rbp)
	je	.L36
	movq	$19, -1176(%rbp)
	jmp	.L26
.L36:
	movq	$5, -1176(%rbp)
	jmp	.L26
.L41:
	nop
.L26:
	jmp	.L38
.L42:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L40
	call	__stack_chk_fail@PLT
.L40:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	getft, .-getft
	.section	.rodata
.LC12:
	.string	"unlink"
.LC13:
	.string	"No files found."
.LC14:
	.string	"Files:\n%s\n"
	.align 8
.LC15:
	.string	"Tar.gz archive created: temp.tar.gz"
.LC16:
	.string	"find %s -type f ! -newermt %s"
.LC17:
	.string	"-T"
.LC18:
	.string	"-czvf"
.LC19:
	.string	"tar"
.LC20:
	.string	"execlp"
.LC21:
	.string	"fork"
.LC22:
	.string	"open"
.LC23:
	.string	"fread"
.LC24:
	.string	"popen"
.LC25:
	.string	"temp.tar.gz"
	.text
	.globl	getfdb
	.type	getfdb, @function
getfdb:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$2272, %rsp
	movl	%edi, -2260(%rbp)
	movq	%rsi, -2272(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$12, -2192(%rbp)
.L94:
	cmpq	$39, -2192(%rbp)
	ja	.L97
	movq	-2192(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L46(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L46(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L46:
	.long	.L97-.L46
	.long	.L97-.L46
	.long	.L75-.L46
	.long	.L98-.L46
	.long	.L73-.L46
	.long	.L72-.L46
	.long	.L71-.L46
	.long	.L70-.L46
	.long	.L69-.L46
	.long	.L68-.L46
	.long	.L97-.L46
	.long	.L67-.L46
	.long	.L66-.L46
	.long	.L97-.L46
	.long	.L65-.L46
	.long	.L64-.L46
	.long	.L63-.L46
	.long	.L62-.L46
	.long	.L61-.L46
	.long	.L60-.L46
	.long	.L97-.L46
	.long	.L59-.L46
	.long	.L58-.L46
	.long	.L57-.L46
	.long	.L97-.L46
	.long	.L56-.L46
	.long	.L55-.L46
	.long	.L54-.L46
	.long	.L98-.L46
	.long	.L52-.L46
	.long	.L98-.L46
	.long	.L50-.L46
	.long	.L49-.L46
	.long	.L97-.L46
	.long	.L48-.L46
	.long	.L97-.L46
	.long	.L97-.L46
	.long	.L97-.L46
	.long	.L47-.L46
	.long	.L45-.L46
	.text
.L61:
	leaq	.LC9(%rip), %rax
	movq	%rax, -2128(%rbp)
	movq	-2128(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -2120(%rbp)
	movq	-2120(%rbp), %rdx
	movq	-2128(%rbp), %rsi
	movl	-2260(%rbp), %eax
	movl	$0, %ecx
	movl	%eax, %edi
	call	send@PLT
	movq	$28, -2192(%rbp)
	jmp	.L76
.L56:
	movq	-2216(%rbp), %rax
	movq	%rax, %rdi
	call	feof@PLT
	movl	%eax, -2248(%rbp)
	movq	$31, -2192(%rbp)
	jmp	.L76
.L73:
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$5, -2192(%rbp)
	jmp	.L76
.L65:
	movq	-2216(%rbp), %rdx
	leaq	-1040(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1023, %edx
	movl	$1, %esi
	movq	%rax, %rdi
	call	fread@PLT
	movq	%rax, -2112(%rbp)
	movq	-2112(%rbp), %rax
	movq	%rax, -2208(%rbp)
	movq	$8, -2192(%rbp)
	jmp	.L76
.L64:
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$16, -2192(%rbp)
	jmp	.L76
.L50:
	cmpl	$0, -2248(%rbp)
	je	.L78
	movq	$15, -2192(%rbp)
	jmp	.L76
.L78:
	movq	$34, -2192(%rbp)
	jmp	.L76
.L66:
	movq	$26, -2192(%rbp)
	jmp	.L76
.L69:
	cmpq	$0, -2208(%rbp)
	jne	.L80
	movq	$25, -2192(%rbp)
	jmp	.L76
.L80:
	movq	$23, -2192(%rbp)
	jmp	.L76
.L57:
	leaq	-1040(%rbp), %rdx
	movq	-2208(%rbp), %rax
	addq	%rdx, %rax
	movb	$0, (%rax)
	leaq	-1040(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$16, -2192(%rbp)
	jmp	.L76
.L63:
	movq	-2216(%rbp), %rax
	movq	%rax, %rdi
	call	pclose@PLT
	movb	$47, -2096(%rbp)
	movb	$116, -2095(%rbp)
	movb	$109, -2094(%rbp)
	movb	$112, -2093(%rbp)
	movb	$47, -2092(%rbp)
	movb	$102, -2091(%rbp)
	movb	$105, -2090(%rbp)
	movb	$108, -2089(%rbp)
	movb	$101, -2088(%rbp)
	movb	$95, -2087(%rbp)
	movb	$108, -2086(%rbp)
	movb	$105, -2085(%rbp)
	movb	$115, -2084(%rbp)
	movb	$116, -2083(%rbp)
	movb	$46, -2082(%rbp)
	movb	$116, -2081(%rbp)
	movb	$120, -2080(%rbp)
	movb	$116, -2079(%rbp)
	movb	$0, -2078(%rbp)
	leaq	-2096(%rbp), %rax
	movl	$420, %edx
	movl	$65, %esi
	movq	%rax, %rdi
	movl	$0, %eax
	call	open@PLT
	movl	%eax, -2228(%rbp)
	movl	-2228(%rbp), %eax
	movl	%eax, -2244(%rbp)
	movq	$19, -2192(%rbp)
	jmp	.L76
.L59:
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-2096(%rbp), %rax
	movq	%rax, %rdi
	call	unlink@PLT
	movl	%eax, -2236(%rbp)
	movq	$17, -2192(%rbp)
	jmp	.L76
.L55:
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	call	getenv@PLT
	movq	%rax, -2144(%rbp)
	movq	-2144(%rbp), %rax
	movq	%rax, -2224(%rbp)
	movq	-2272(%rbp), %rcx
	movq	-2224(%rbp), %rdx
	leaq	-2064(%rbp), %rax
	movq	%rcx, %r8
	movq	%rdx, %rcx
	leaq	.LC16(%rip), %rdx
	movl	$1024, %esi
	movq	%rax, %rdi
	movl	$0, %eax
	call	snprintf@PLT
	leaq	-2064(%rbp), %rax
	leaq	.LC0(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	popen@PLT
	movq	%rax, -2136(%rbp)
	movq	-2136(%rbp), %rax
	movq	%rax, -2216(%rbp)
	movq	$7, -2192(%rbp)
	jmp	.L76
.L67:
	cmpq	$0, -2200(%rbp)
	je	.L82
	movq	$29, -2192(%rbp)
	jmp	.L76
.L82:
	movq	$18, -2192(%rbp)
	jmp	.L76
.L68:
	movl	$0, %edi
	call	wait@PLT
	movq	$21, -2192(%rbp)
	jmp	.L76
.L60:
	cmpl	$-1, -2244(%rbp)
	jne	.L84
	movq	$38, -2192(%rbp)
	jmp	.L76
.L84:
	movq	$39, -2192(%rbp)
	jmp	.L76
.L49:
	movq	-2224(%rbp), %rax
	movq	%rax, %rdi
	call	chdir@PLT
	leaq	-2096(%rbp), %rax
	subq	$8, %rsp
	pushq	$0
	movq	%rax, %r9
	leaq	.LC17(%rip), %r8
	leaq	.LC6(%rip), %rax
	movq	%rax, %rcx
	leaq	.LC18(%rip), %rax
	movq	%rax, %rdx
	leaq	.LC19(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC19(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	execlp@PLT
	addq	$16, %rsp
	leaq	.LC20(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L62:
	cmpl	$0, -2236(%rbp)
	je	.L86
	movq	$4, -2192(%rbp)
	jmp	.L76
.L86:
	movq	$5, -2192(%rbp)
	jmp	.L76
.L71:
	leaq	.LC21(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$3, -2192(%rbp)
	jmp	.L76
.L54:
	cmpl	$0, -2240(%rbp)
	jle	.L88
	movq	$9, -2192(%rbp)
	jmp	.L76
.L88:
	movq	$6, -2192(%rbp)
	jmp	.L76
.L47:
	leaq	.LC22(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$30, -2192(%rbp)
	jmp	.L76
.L48:
	leaq	.LC23(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$16, -2192(%rbp)
	jmp	.L76
.L58:
	leaq	.LC24(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L72:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC25(%rip), %rax
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -2104(%rbp)
	movq	-2104(%rbp), %rax
	movq	%rax, -2200(%rbp)
	movq	$11, -2192(%rbp)
	jmp	.L76
.L45:
	leaq	-1040(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -2184(%rbp)
	movq	-2184(%rbp), %rdx
	leaq	-1040(%rbp), %rcx
	movl	-2244(%rbp), %eax
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	write@PLT
	movl	-2244(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	call	fork@PLT
	movl	%eax, -2232(%rbp)
	movl	-2232(%rbp), %eax
	movl	%eax, -2240(%rbp)
	movq	$2, -2192(%rbp)
	jmp	.L76
.L70:
	cmpq	$0, -2216(%rbp)
	jne	.L90
	movq	$22, -2192(%rbp)
	jmp	.L76
.L90:
	movq	$14, -2192(%rbp)
	jmp	.L76
.L52:
	movq	-2200(%rbp), %rax
	movl	$2, %edx
	movl	$0, %esi
	movq	%rax, %rdi
	call	fseek@PLT
	movq	-2200(%rbp), %rax
	movq	%rax, %rdi
	call	ftell@PLT
	movq	%rax, -2176(%rbp)
	movq	-2176(%rbp), %rax
	movq	%rax, -2168(%rbp)
	movq	-2200(%rbp), %rax
	movq	%rax, %rdi
	call	rewind@PLT
	movq	-2168(%rbp), %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -2160(%rbp)
	movq	-2160(%rbp), %rax
	movq	%rax, -2152(%rbp)
	movq	-2168(%rbp), %rdx
	movq	-2200(%rbp), %rcx
	movq	-2152(%rbp), %rax
	movl	$1, %esi
	movq	%rax, %rdi
	call	fread@PLT
	movq	-2200(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	-2168(%rbp), %rdx
	movq	-2152(%rbp), %rsi
	movl	-2260(%rbp), %eax
	movl	$0, %ecx
	movl	%eax, %edi
	call	send@PLT
	movq	-2152(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$28, -2192(%rbp)
	jmp	.L76
.L75:
	cmpl	$0, -2240(%rbp)
	jne	.L92
	movq	$32, -2192(%rbp)
	jmp	.L76
.L92:
	movq	$27, -2192(%rbp)
	jmp	.L76
.L97:
	nop
.L76:
	jmp	.L94
.L98:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L96
	call	__stack_chk_fail@PLT
.L96:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	getfdb, .-getfdb
	.section	.rodata
.LC26:
	.string	"%s/%s"
.LC27:
	.string	"%s\n"
.LC28:
	.string	"stat"
.LC29:
	.string	".."
.LC30:
	.string	"opendir"
.LC31:
	.string	"."
	.text
	.globl	filesearchWrite
	.type	filesearchWrite, @function
filesearchWrite:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$1296, %rsp
	movq	%rdi, -1272(%rbp)
	movq	%rsi, -1280(%rbp)
	movq	%rdx, -1288(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$25, -1208(%rbp)
.L146:
	cmpq	$30, -1208(%rbp)
	ja	.L149
	movq	-1208(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L102(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L102(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L102:
	.long	.L149-.L102
	.long	.L125-.L102
	.long	.L124-.L102
	.long	.L123-.L102
	.long	.L122-.L102
	.long	.L150-.L102
	.long	.L150-.L102
	.long	.L119-.L102
	.long	.L118-.L102
	.long	.L117-.L102
	.long	.L149-.L102
	.long	.L116-.L102
	.long	.L149-.L102
	.long	.L149-.L102
	.long	.L115-.L102
	.long	.L114-.L102
	.long	.L113-.L102
	.long	.L112-.L102
	.long	.L111-.L102
	.long	.L110-.L102
	.long	.L109-.L102
	.long	.L108-.L102
	.long	.L107-.L102
	.long	.L106-.L102
	.long	.L105-.L102
	.long	.L104-.L102
	.long	.L149-.L102
	.long	.L149-.L102
	.long	.L149-.L102
	.long	.L103-.L102
	.long	.L101-.L102
	.text
.L111:
	movq	-1280(%rbp), %rax
	movq	%rax, %rdi
	call	opendir@PLT
	movq	%rax, -1200(%rbp)
	movq	-1200(%rbp), %rax
	movq	%rax, -1240(%rbp)
	movq	$11, -1208(%rbp)
	jmp	.L126
.L104:
	movq	$18, -1208(%rbp)
	jmp	.L126
.L122:
	cmpl	$0, -1252(%rbp)
	jne	.L127
	movq	$22, -1208(%rbp)
	jmp	.L126
.L127:
	movq	$24, -1208(%rbp)
	jmp	.L126
.L101:
	movq	-1232(%rbp), %rax
	leaq	19(%rax), %rcx
	movq	-1280(%rbp), %rdx
	leaq	-1040(%rbp), %rax
	movq	%rcx, %r8
	movq	%rdx, %rcx
	leaq	.LC26(%rip), %rdx
	movl	$1024, %esi
	movq	%rax, %rdi
	movl	$0, %eax
	call	snprintf@PLT
	leaq	-1184(%rbp), %rdx
	leaq	-1040(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	stat@PLT
	movl	%eax, -1244(%rbp)
	movq	$9, -1208(%rbp)
	jmp	.L126
.L115:
	leaq	-1040(%rbp), %rdx
	movq	-1272(%rbp), %rax
	leaq	.LC27(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	$22, -1208(%rbp)
	jmp	.L126
.L114:
	leaq	.LC28(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$22, -1208(%rbp)
	jmp	.L126
.L118:
	movq	-1240(%rbp), %rax
	movq	%rax, %rdi
	call	closedir@PLT
	movq	$6, -1208(%rbp)
	jmp	.L126
.L125:
	cmpl	$0, -1248(%rbp)
	jne	.L129
	movq	$22, -1208(%rbp)
	jmp	.L126
.L129:
	movq	$30, -1208(%rbp)
	jmp	.L126
.L106:
	cmpq	$0, -1232(%rbp)
	je	.L131
	movq	$29, -1208(%rbp)
	jmp	.L126
.L131:
	movq	$8, -1208(%rbp)
	jmp	.L126
.L123:
	cmpq	$0, -1216(%rbp)
	je	.L133
	movq	$14, -1208(%rbp)
	jmp	.L126
.L133:
	movq	$22, -1208(%rbp)
	jmp	.L126
.L113:
	movq	-1288(%rbp), %rdx
	leaq	-1040(%rbp), %rcx
	movq	-1272(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	filesearchWrite
	movq	$22, -1208(%rbp)
	jmp	.L126
.L105:
	movq	-1232(%rbp), %rax
	addq	$19, %rax
	leaq	.LC29(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -1248(%rbp)
	movq	$1, -1208(%rbp)
	jmp	.L126
.L108:
	cmpq	$0, -1224(%rbp)
	je	.L135
	movq	$20, -1208(%rbp)
	jmp	.L126
.L135:
	movq	$22, -1208(%rbp)
	jmp	.L126
.L116:
	cmpq	$0, -1240(%rbp)
	jne	.L137
	movq	$17, -1208(%rbp)
	jmp	.L126
.L137:
	movq	$22, -1208(%rbp)
	jmp	.L126
.L117:
	cmpl	$0, -1244(%rbp)
	je	.L139
	movq	$15, -1208(%rbp)
	jmp	.L126
.L139:
	movq	$2, -1208(%rbp)
	jmp	.L126
.L110:
	movl	-1160(%rbp), %eax
	andl	$61440, %eax
	cmpl	$32768, %eax
	jne	.L141
	movq	$7, -1208(%rbp)
	jmp	.L126
.L141:
	movq	$22, -1208(%rbp)
	jmp	.L126
.L112:
	leaq	.LC30(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$5, -1208(%rbp)
	jmp	.L126
.L107:
	movq	-1240(%rbp), %rax
	movq	%rax, %rdi
	call	readdir@PLT
	movq	%rax, -1232(%rbp)
	movq	$23, -1208(%rbp)
	jmp	.L126
.L119:
	movq	-1232(%rbp), %rax
	addq	$19, %rax
	movl	$46, %esi
	movq	%rax, %rdi
	call	strrchr@PLT
	movq	%rax, -1192(%rbp)
	movq	-1192(%rbp), %rax
	movq	%rax, -1224(%rbp)
	movq	$21, -1208(%rbp)
	jmp	.L126
.L103:
	movq	-1232(%rbp), %rax
	addq	$19, %rax
	leaq	.LC31(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -1252(%rbp)
	movq	$4, -1208(%rbp)
	jmp	.L126
.L124:
	movl	-1160(%rbp), %eax
	andl	$61440, %eax
	cmpl	$16384, %eax
	jne	.L144
	movq	$16, -1208(%rbp)
	jmp	.L126
.L144:
	movq	$19, -1208(%rbp)
	jmp	.L126
.L109:
	movq	-1224(%rbp), %rdx
	movq	-1288(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strstr@PLT
	movq	%rax, -1216(%rbp)
	movq	$3, -1208(%rbp)
	jmp	.L126
.L149:
	nop
.L126:
	jmp	.L146
.L150:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L148
	call	__stack_chk_fail@PLT
.L148:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	filesearchWrite, .-filesearchWrite
	.section	.rodata
.LC32:
	.string	"realloc"
	.text
	.globl	appendFilePath
	.type	appendFilePath, @function
appendFilePath:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$88, %rsp
	.cfi_offset 3, -24
	movq	%rdi, -88(%rbp)
	movq	%rsi, -96(%rbp)
	movq	$4, -64(%rbp)
.L164:
	cmpq	$7, -64(%rbp)
	ja	.L165
	movq	-64(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L154(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L154(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L154:
	.long	.L165-.L154
	.long	.L159-.L154
	.long	.L158-.L154
	.long	.L166-.L154
	.long	.L156-.L154
	.long	.L155-.L154
	.long	.L165-.L154
	.long	.L153-.L154
	.text
.L156:
	movq	$2, -64(%rbp)
	jmp	.L160
.L159:
	leaq	.LC32(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L155:
	cmpq	$0, -72(%rbp)
	jne	.L162
	movq	$1, -64(%rbp)
	jmp	.L160
.L162:
	movq	$7, -64(%rbp)
	jmp	.L160
.L153:
	movq	-88(%rbp), %rax
	movq	-72(%rbp), %rdx
	movq	%rdx, (%rax)
	movq	-88(%rbp), %rax
	movq	(%rax), %rax
	movq	-96(%rbp), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcat@PLT
	movq	-88(%rbp), %rax
	movq	(%rax), %rbx
	movq	%rbx, %rdi
	call	strlen@PLT
	addq	%rbx, %rax
	movw	$10, (%rax)
	movq	$3, -64(%rbp)
	jmp	.L160
.L158:
	movq	-88(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -56(%rbp)
	movq	-56(%rbp), %rax
	movq	%rax, -48(%rbp)
	movq	-96(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -40(%rbp)
	movq	-40(%rbp), %rax
	movq	%rax, -32(%rbp)
	movq	-48(%rbp), %rdx
	movq	-32(%rbp), %rax
	addq	%rdx, %rax
	leaq	2(%rax), %rdx
	movq	-88(%rbp), %rax
	movq	(%rax), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, -72(%rbp)
	movq	$5, -64(%rbp)
	jmp	.L160
.L165:
	nop
.L160:
	jmp	.L164
.L166:
	nop
	movq	-8(%rbp), %rbx
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	appendFilePath, .-appendFilePath
	.globl	searchforFiles
	.type	searchforFiles, @function
searchforFiles:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$1264, %rsp
	movq	%rdi, -1240(%rbp)
	movq	%rsi, -1248(%rbp)
	movq	%rdx, -1256(%rbp)
	movq	%rcx, -1264(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$5, -1200(%rbp)
.L209:
	cmpq	$27, -1200(%rbp)
	ja	.L212
	movq	-1200(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L170(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L170(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L170:
	.long	.L190-.L170
	.long	.L212-.L170
	.long	.L189-.L170
	.long	.L188-.L170
	.long	.L187-.L170
	.long	.L186-.L170
	.long	.L212-.L170
	.long	.L185-.L170
	.long	.L184-.L170
	.long	.L183-.L170
	.long	.L182-.L170
	.long	.L212-.L170
	.long	.L181-.L170
	.long	.L180-.L170
	.long	.L212-.L170
	.long	.L179-.L170
	.long	.L212-.L170
	.long	.L178-.L170
	.long	.L177-.L170
	.long	.L176-.L170
	.long	.L212-.L170
	.long	.L212-.L170
	.long	.L175-.L170
	.long	.L174-.L170
	.long	.L173-.L170
	.long	.L213-.L170
	.long	.L213-.L170
	.long	.L169-.L170
	.text
.L177:
	movq	-1248(%rbp), %rax
	movq	%rax, %rdi
	call	opendir@PLT
	movq	%rax, -1192(%rbp)
	movq	-1192(%rbp), %rax
	movq	%rax, -1216(%rbp)
	movq	$9, -1200(%rbp)
	jmp	.L191
.L187:
	movq	-1264(%rbp), %rcx
	movq	-1256(%rbp), %rdx
	leaq	-1040(%rbp), %rsi
	movq	-1240(%rbp), %rax
	movq	%rax, %rdi
	call	searchforFiles
	movq	$8, -1200(%rbp)
	jmp	.L191
.L179:
	movl	-1160(%rbp), %eax
	andl	$61440, %eax
	cmpl	$16384, %eax
	jne	.L193
	movq	$4, -1200(%rbp)
	jmp	.L191
.L193:
	movq	$7, -1200(%rbp)
	jmp	.L191
.L181:
	movq	-1208(%rbp), %rax
	leaq	19(%rax), %rcx
	movq	-1248(%rbp), %rdx
	leaq	-1040(%rbp), %rax
	movq	%rcx, %r8
	movq	%rdx, %rcx
	leaq	.LC26(%rip), %rdx
	movl	$1024, %esi
	movq	%rax, %rdi
	movl	$0, %eax
	call	snprintf@PLT
	leaq	-1184(%rbp), %rdx
	leaq	-1040(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	stat@PLT
	movl	%eax, -1224(%rbp)
	movq	$2, -1200(%rbp)
	jmp	.L191
.L184:
	movq	-1216(%rbp), %rax
	movq	%rax, %rdi
	call	readdir@PLT
	movq	%rax, -1208(%rbp)
	movq	$24, -1200(%rbp)
	jmp	.L191
.L174:
	leaq	-1040(%rbp), %rdx
	movq	-1240(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcat@PLT
	movq	-1240(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, %rdx
	movq	-1240(%rbp), %rax
	addq	%rdx, %rax
	movw	$32, (%rax)
	movq	-1264(%rbp), %rax
	movl	(%rax), %eax
	leal	1(%rax), %edx
	movq	-1264(%rbp), %rax
	movl	%edx, (%rax)
	movq	$8, -1200(%rbp)
	jmp	.L191
.L188:
	movq	-1208(%rbp), %rax
	addq	$19, %rax
	leaq	.LC31(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -1232(%rbp)
	movq	$10, -1200(%rbp)
	jmp	.L191
.L173:
	cmpq	$0, -1208(%rbp)
	je	.L195
	movq	$3, -1200(%rbp)
	jmp	.L191
.L195:
	movq	$27, -1200(%rbp)
	jmp	.L191
.L183:
	cmpq	$0, -1216(%rbp)
	jne	.L197
	movq	$13, -1200(%rbp)
	jmp	.L191
.L197:
	movq	$8, -1200(%rbp)
	jmp	.L191
.L180:
	leaq	.LC30(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$25, -1200(%rbp)
	jmp	.L191
.L176:
	movq	-1208(%rbp), %rax
	addq	$19, %rax
	leaq	.LC29(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -1228(%rbp)
	movq	$22, -1200(%rbp)
	jmp	.L191
.L178:
	movq	-1208(%rbp), %rax
	leaq	19(%rax), %rdx
	movq	-1256(%rbp), %rax
	movq	%rax, %rsi
	movq	%rdx, %rdi
	call	strcmp@PLT
	movl	%eax, -1220(%rbp)
	movq	$0, -1200(%rbp)
	jmp	.L191
.L169:
	movq	-1216(%rbp), %rax
	movq	%rax, %rdi
	call	closedir@PLT
	movq	$26, -1200(%rbp)
	jmp	.L191
.L175:
	cmpl	$0, -1228(%rbp)
	jne	.L199
	movq	$8, -1200(%rbp)
	jmp	.L191
.L199:
	movq	$12, -1200(%rbp)
	jmp	.L191
.L186:
	movq	$18, -1200(%rbp)
	jmp	.L191
.L182:
	cmpl	$0, -1232(%rbp)
	jne	.L201
	movq	$8, -1200(%rbp)
	jmp	.L191
.L201:
	movq	$19, -1200(%rbp)
	jmp	.L191
.L190:
	cmpl	$0, -1220(%rbp)
	jne	.L203
	movq	$23, -1200(%rbp)
	jmp	.L191
.L203:
	movq	$8, -1200(%rbp)
	jmp	.L191
.L185:
	movl	-1160(%rbp), %eax
	andl	$61440, %eax
	cmpl	$32768, %eax
	jne	.L205
	movq	$17, -1200(%rbp)
	jmp	.L191
.L205:
	movq	$8, -1200(%rbp)
	jmp	.L191
.L189:
	cmpl	$0, -1224(%rbp)
	je	.L207
	movq	$8, -1200(%rbp)
	jmp	.L191
.L207:
	movq	$15, -1200(%rbp)
	jmp	.L191
.L212:
	nop
.L191:
	jmp	.L209
.L213:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L211
	call	__stack_chk_fail@PLT
.L211:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	searchforFiles, .-searchforFiles
	.section	.rodata
.LC33:
	.string	"getfdb "
.LC34:
	.string	"getfda "
.LC35:
	.string	"quitc"
.LC36:
	.string	"%s"
	.align 8
.LC37:
	.string	"Requested files created on or after: %s\n"
.LC38:
	.string	"read"
.LC39:
	.string	"Client Process Terminated!"
.LC40:
	.string	"getfn "
.LC41:
	.string	"getfz "
	.align 8
.LC42:
	.string	"Requested files created on or before: %s\n"
.LC43:
	.string	"Unknown command"
.LC44:
	.string	"getft "
	.text
	.globl	pclientrequest
	.type	pclientrequest, @function
pclientrequest:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$1184, %rsp
	movl	%edi, -1172(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$5, -1136(%rbp)
.L258:
	cmpq	$28, -1136(%rbp)
	ja	.L261
	movq	-1136(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L217(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L217(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L217:
	.long	.L241-.L217
	.long	.L261-.L217
	.long	.L261-.L217
	.long	.L240-.L217
	.long	.L239-.L217
	.long	.L238-.L217
	.long	.L237-.L217
	.long	.L236-.L217
	.long	.L235-.L217
	.long	.L234-.L217
	.long	.L233-.L217
	.long	.L232-.L217
	.long	.L231-.L217
	.long	.L230-.L217
	.long	.L229-.L217
	.long	.L228-.L217
	.long	.L262-.L217
	.long	.L226-.L217
	.long	.L261-.L217
	.long	.L225-.L217
	.long	.L224-.L217
	.long	.L223-.L217
	.long	.L222-.L217
	.long	.L261-.L217
	.long	.L221-.L217
	.long	.L220-.L217
	.long	.L262-.L217
	.long	.L218-.L217
	.long	.L216-.L217
	.text
.L220:
	cmpl	$0, -1148(%rbp)
	jne	.L242
	movq	$17, -1136(%rbp)
	jmp	.L244
.L242:
	movq	$27, -1136(%rbp)
	jmp	.L244
.L239:
	leaq	-1040(%rbp), %rcx
	movl	-1172(%rbp), %eax
	movl	$1023, %edx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	read@PLT
	movq	%rax, -1112(%rbp)
	movq	-1112(%rbp), %rax
	movq	%rax, -1144(%rbp)
	movq	$13, -1136(%rbp)
	jmp	.L244
.L229:
	leaq	-1040(%rbp), %rax
	movl	$6, %edx
	leaq	.LC33(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	strncmp@PLT
	movl	%eax, -1164(%rbp)
	movq	$22, -1136(%rbp)
	jmp	.L244
.L228:
	cmpl	$0, -1152(%rbp)
	jne	.L245
	movq	$3, -1136(%rbp)
	jmp	.L244
.L245:
	movq	$6, -1136(%rbp)
	jmp	.L244
.L231:
	leaq	-1040(%rbp), %rax
	movl	$6, %edx
	leaq	.LC34(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	strncmp@PLT
	movl	%eax, -1168(%rbp)
	movq	$7, -1136(%rbp)
	jmp	.L244
.L235:
	leaq	-1040(%rbp), %rdx
	movq	-1144(%rbp), %rax
	addq	%rdx, %rax
	movb	$0, (%rax)
	leaq	-1040(%rbp), %rax
	leaq	.LC35(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -1148(%rbp)
	movq	$25, -1136(%rbp)
	jmp	.L244
.L240:
	leaq	-1040(%rbp), %rdx
	movl	-1172(%rbp), %eax
	movq	%rdx, %rsi
	movl	%eax, %edi
	call	getfz
	movq	$26, -1136(%rbp)
	jmp	.L244
.L221:
	leaq	-1040(%rbp), %rax
	addq	$7, %rax
	leaq	-1072(%rbp), %rdx
	leaq	.LC36(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_sscanf@PLT
	leaq	-1072(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC37(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-1072(%rbp), %rdx
	movl	-1172(%rbp), %eax
	movq	%rdx, %rsi
	movl	%eax, %edi
	call	getfda
	movq	$26, -1136(%rbp)
	jmp	.L244
.L223:
	cmpl	$0, -1160(%rbp)
	jne	.L248
	movq	$9, -1136(%rbp)
	jmp	.L244
.L248:
	movq	$14, -1136(%rbp)
	jmp	.L244
.L232:
	cmpl	$0, -1156(%rbp)
	jne	.L250
	movq	$20, -1136(%rbp)
	jmp	.L244
.L250:
	movq	$0, -1136(%rbp)
	jmp	.L244
.L234:
	leaq	-1040(%rbp), %rdx
	movl	-1172(%rbp), %eax
	movq	%rdx, %rsi
	movl	%eax, %edi
	call	getft
	movq	$26, -1136(%rbp)
	jmp	.L244
.L230:
	cmpq	$0, -1144(%rbp)
	jg	.L252
	movq	$19, -1136(%rbp)
	jmp	.L244
.L252:
	movq	$8, -1136(%rbp)
	jmp	.L244
.L225:
	leaq	.LC38(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	-1172(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	movq	$16, -1136(%rbp)
	jmp	.L244
.L226:
	leaq	.LC39(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$26, -1136(%rbp)
	jmp	.L244
.L237:
	leaq	-1040(%rbp), %rax
	movl	$5, %edx
	leaq	.LC40(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	strncmp@PLT
	movl	%eax, -1156(%rbp)
	movq	$11, -1136(%rbp)
	jmp	.L244
.L218:
	leaq	-1040(%rbp), %rax
	movl	$5, %edx
	leaq	.LC41(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	strncmp@PLT
	movl	%eax, -1152(%rbp)
	movq	$15, -1136(%rbp)
	jmp	.L244
.L222:
	cmpl	$0, -1164(%rbp)
	jne	.L254
	movq	$28, -1136(%rbp)
	jmp	.L244
.L254:
	movq	$12, -1136(%rbp)
	jmp	.L244
.L216:
	leaq	-1040(%rbp), %rax
	addq	$7, %rax
	leaq	-1104(%rbp), %rdx
	leaq	.LC36(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_sscanf@PLT
	leaq	-1104(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC42(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-1104(%rbp), %rdx
	movl	-1172(%rbp), %eax
	movq	%rdx, %rsi
	movl	%eax, %edi
	call	getfdb
	movq	$26, -1136(%rbp)
	jmp	.L244
.L238:
	movq	$4, -1136(%rbp)
	jmp	.L244
.L233:
	leaq	.LC43(%rip), %rax
	movq	%rax, -1128(%rbp)
	movq	-1128(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -1120(%rbp)
	movq	-1120(%rbp), %rdx
	movq	-1128(%rbp), %rsi
	movl	-1172(%rbp), %eax
	movl	$0, %ecx
	movl	%eax, %edi
	call	send@PLT
	movq	$26, -1136(%rbp)
	jmp	.L244
.L241:
	leaq	-1040(%rbp), %rax
	movl	$5, %edx
	leaq	.LC44(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	strncmp@PLT
	movl	%eax, -1160(%rbp)
	movq	$21, -1136(%rbp)
	jmp	.L244
.L236:
	cmpl	$0, -1168(%rbp)
	jne	.L256
	movq	$24, -1136(%rbp)
	jmp	.L244
.L256:
	movq	$10, -1136(%rbp)
	jmp	.L244
.L224:
	leaq	-1040(%rbp), %rdx
	movl	-1172(%rbp), %eax
	movq	%rdx, %rsi
	movl	%eax, %edi
	call	getfn
	movq	$26, -1136(%rbp)
	jmp	.L244
.L261:
	nop
.L244:
	jmp	.L258
.L262:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L260
	call	__stack_chk_fail@PLT
.L260:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	pclientrequest, .-pclientrequest
	.section	.rodata
.LC45:
	.string	"%llu %llu"
	.align 8
.LC46:
	.string	"Requested size range: %llu - %llu\n"
.LC47:
	.string	"Usage: getfz size1 size2"
.LC48:
	.string	"malloc"
.LC49:
	.string	"/tmp/file_list.txt"
.LC50:
	.string	"Finished tarring files"
	.text
	.globl	getfz
	.type	getfz, @function
getfz:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$192, %rsp
	movl	%edi, -180(%rbp)
	movq	%rsi, -192(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$11, -112(%rbp)
.L311:
	cmpq	$36, -112(%rbp)
	ja	.L314
	movq	-112(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L266(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L266(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L266:
	.long	.L315-.L266
	.long	.L291-.L266
	.long	.L314-.L266
	.long	.L290-.L266
	.long	.L289-.L266
	.long	.L314-.L266
	.long	.L288-.L266
	.long	.L287-.L266
	.long	.L314-.L266
	.long	.L286-.L266
	.long	.L285-.L266
	.long	.L284-.L266
	.long	.L314-.L266
	.long	.L283-.L266
	.long	.L315-.L266
	.long	.L314-.L266
	.long	.L281-.L266
	.long	.L280-.L266
	.long	.L279-.L266
	.long	.L278-.L266
	.long	.L277-.L266
	.long	.L276-.L266
	.long	.L275-.L266
	.long	.L274-.L266
	.long	.L273-.L266
	.long	.L314-.L266
	.long	.L272-.L266
	.long	.L271-.L266
	.long	.L270-.L266
	.long	.L314-.L266
	.long	.L314-.L266
	.long	.L269-.L266
	.long	.L268-.L266
	.long	.L314-.L266
	.long	.L314-.L266
	.long	.L267-.L266
	.long	.L265-.L266
	.text
.L279:
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	call	getenv@PLT
	movq	%rax, -72(%rbp)
	movq	-72(%rbp), %rax
	movq	%rax, -120(%rbp)
	movq	-192(%rbp), %rax
	leaq	9(%rax), %rdi
	leaq	-136(%rbp), %rdx
	leaq	-144(%rbp), %rax
	movq	%rdx, %rcx
	movq	%rax, %rdx
	leaq	.LC45(%rip), %rax
	movq	%rax, %rsi
	movl	$0, %eax
	call	__isoc99_sscanf@PLT
	movl	%eax, -148(%rbp)
	movl	-148(%rbp), %eax
	movl	%eax, -172(%rbp)
	movq	$28, -112(%rbp)
	jmp	.L293
.L289:
	movl	$0, %edi
	call	wait@PLT
	movq	$27, -112(%rbp)
	jmp	.L293
.L269:
	cmpl	$-1, -168(%rbp)
	jne	.L295
	movq	$9, -112(%rbp)
	jmp	.L293
.L295:
	movq	$24, -112(%rbp)
	jmp	.L293
.L291:
	movq	-192(%rbp), %rax
	leaq	9(%rax), %rdi
	leaq	-136(%rbp), %rdx
	leaq	-144(%rbp), %rax
	movq	%rdx, %rcx
	movq	%rax, %rdx
	leaq	.LC45(%rip), %rax
	movq	%rax, %rsi
	movl	$0, %eax
	call	__isoc99_sscanf@PLT
	movq	-136(%rbp), %rdx
	movq	-144(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC46(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$1, %edi
	call	malloc@PLT
	movq	%rax, -40(%rbp)
	movq	-40(%rbp), %rax
	movq	%rax, -128(%rbp)
	movq	$3, -112(%rbp)
	jmp	.L293
.L274:
	cmpl	$0, -164(%rbp)
	jle	.L297
	movq	$4, -112(%rbp)
	jmp	.L293
.L297:
	movq	$26, -112(%rbp)
	jmp	.L293
.L290:
	movq	-128(%rbp), %rax
	testq	%rax, %rax
	jne	.L299
	movq	$19, -112(%rbp)
	jmp	.L293
.L299:
	movq	$6, -112(%rbp)
	jmp	.L293
.L281:
	movq	-128(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$14, -112(%rbp)
	jmp	.L293
.L273:
	movq	-128(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -80(%rbp)
	movq	-128(%rbp), %rcx
	movq	-80(%rbp), %rdx
	movl	-168(%rbp), %eax
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	write@PLT
	movl	-168(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	call	fork@PLT
	movl	%eax, -152(%rbp)
	movl	-152(%rbp), %eax
	movl	%eax, -164(%rbp)
	movq	$13, -112(%rbp)
	jmp	.L293
.L276:
	leaq	.LC47(%rip), %rax
	movq	%rax, -104(%rbp)
	movq	-104(%rbp), %rcx
	movl	-180(%rbp), %eax
	movl	$50, %edx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	write@PLT
	movq	$14, -112(%rbp)
	jmp	.L293
.L265:
	movq	-144(%rbp), %rdx
	movq	-136(%rbp), %rax
	cmpq	%rax, %rdx
	ja	.L301
	movq	$1, -112(%rbp)
	jmp	.L293
.L301:
	movq	$32, -112(%rbp)
	jmp	.L293
.L272:
	leaq	.LC21(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	-128(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movl	$1, %edi
	call	exit@PLT
.L284:
	movq	$18, -112(%rbp)
	jmp	.L293
.L286:
	leaq	.LC22(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	-128(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movl	$1, %edi
	call	exit@PLT
.L283:
	cmpl	$0, -164(%rbp)
	jne	.L303
	movq	$17, -112(%rbp)
	jmp	.L293
.L303:
	movq	$23, -112(%rbp)
	jmp	.L293
.L278:
	leaq	.LC48(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L268:
	leaq	.LC47(%rip), %rax
	movq	%rax, -104(%rbp)
	movq	-104(%rbp), %rcx
	movl	-180(%rbp), %eax
	movl	$50, %edx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	write@PLT
	movq	$14, -112(%rbp)
	jmp	.L293
.L280:
	movq	-120(%rbp), %rax
	movq	%rax, %rdi
	call	chdir@PLT
	leaq	.LC49(%rip), %rax
	movq	%rax, -64(%rbp)
	movq	-64(%rbp), %rax
	subq	$8, %rsp
	pushq	$0
	movq	%rax, %r9
	leaq	.LC17(%rip), %r8
	leaq	.LC6(%rip), %rax
	movq	%rax, %rcx
	leaq	.LC18(%rip), %rax
	movq	%rax, %rdx
	leaq	.LC19(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC19(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	execlp@PLT
	addq	$16, %rsp
	leaq	.LC20(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L288:
	movq	-128(%rbp), %rax
	movb	$0, (%rax)
	movq	-136(%rbp), %rsi
	movq	-144(%rbp), %rax
	movq	-120(%rbp), %rcx
	leaq	-128(%rbp), %rdx
	movq	%rax, %rdi
	call	findsizerange
	movq	$7, -112(%rbp)
	jmp	.L293
.L271:
	leaq	.LC50(%rip), %rax
	movq	%rax, -96(%rbp)
	movq	-96(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -88(%rbp)
	movq	-88(%rbp), %rax
	leaq	1(%rax), %rdx
	movq	-96(%rbp), %rcx
	movl	-180(%rbp), %eax
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	write@PLT
	leaq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	unlink@PLT
	movl	%eax, -160(%rbp)
	movq	$20, -112(%rbp)
	jmp	.L293
.L275:
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$16, -112(%rbp)
	jmp	.L293
.L270:
	cmpl	$1, -172(%rbp)
	jle	.L305
	movq	$36, -112(%rbp)
	jmp	.L293
.L305:
	movq	$21, -112(%rbp)
	jmp	.L293
.L285:
	leaq	.LC3(%rip), %rax
	movq	%rax, -56(%rbp)
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -48(%rbp)
	movq	-48(%rbp), %rax
	leaq	1(%rax), %rdx
	movq	-56(%rbp), %rcx
	movl	-180(%rbp), %eax
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	write@PLT
	movq	-128(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$0, -112(%rbp)
	jmp	.L293
.L287:
	movq	-128(%rbp), %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	jne	.L307
	movq	$10, -112(%rbp)
	jmp	.L293
.L307:
	movq	$35, -112(%rbp)
	jmp	.L293
.L267:
	movb	$47, -32(%rbp)
	movb	$116, -31(%rbp)
	movb	$109, -30(%rbp)
	movb	$112, -29(%rbp)
	movb	$47, -28(%rbp)
	movb	$102, -27(%rbp)
	movb	$105, -26(%rbp)
	movb	$108, -25(%rbp)
	movb	$101, -24(%rbp)
	movb	$95, -23(%rbp)
	movb	$108, -22(%rbp)
	movb	$105, -21(%rbp)
	movb	$115, -20(%rbp)
	movb	$116, -19(%rbp)
	movb	$46, -18(%rbp)
	movb	$116, -17(%rbp)
	movb	$120, -16(%rbp)
	movb	$116, -15(%rbp)
	movb	$0, -14(%rbp)
	leaq	-32(%rbp), %rax
	movl	$420, %edx
	movl	$65, %esi
	movq	%rax, %rdi
	movl	$0, %eax
	call	open@PLT
	movl	%eax, -156(%rbp)
	movl	-156(%rbp), %eax
	movl	%eax, -168(%rbp)
	movq	$31, -112(%rbp)
	jmp	.L293
.L277:
	cmpl	$0, -160(%rbp)
	je	.L309
	movq	$22, -112(%rbp)
	jmp	.L293
.L309:
	movq	$16, -112(%rbp)
	jmp	.L293
.L314:
	nop
.L293:
	jmp	.L311
.L315:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L313
	call	__stack_chk_fail@PLT
.L313:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	getfz, .-getfz
	.section	.rodata
.LC51:
	.string	"find %s -type f -newermt %s"
	.text
	.globl	getfda
	.type	getfda, @function
getfda:
.LFB10:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$2272, %rsp
	movl	%edi, -2260(%rbp)
	movq	%rsi, -2272(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$38, -2192(%rbp)
.L367:
	cmpq	$39, -2192(%rbp)
	ja	.L370
	movq	-2192(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L319(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L319(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L319:
	.long	.L371-.L319
	.long	.L370-.L319
	.long	.L347-.L319
	.long	.L346-.L319
	.long	.L370-.L319
	.long	.L370-.L319
	.long	.L370-.L319
	.long	.L370-.L319
	.long	.L345-.L319
	.long	.L344-.L319
	.long	.L343-.L319
	.long	.L342-.L319
	.long	.L341-.L319
	.long	.L340-.L319
	.long	.L339-.L319
	.long	.L371-.L319
	.long	.L337-.L319
	.long	.L336-.L319
	.long	.L335-.L319
	.long	.L370-.L319
	.long	.L334-.L319
	.long	.L333-.L319
	.long	.L332-.L319
	.long	.L331-.L319
	.long	.L370-.L319
	.long	.L330-.L319
	.long	.L329-.L319
	.long	.L370-.L319
	.long	.L328-.L319
	.long	.L327-.L319
	.long	.L370-.L319
	.long	.L371-.L319
	.long	.L370-.L319
	.long	.L325-.L319
	.long	.L324-.L319
	.long	.L323-.L319
	.long	.L322-.L319
	.long	.L321-.L319
	.long	.L320-.L319
	.long	.L318-.L319
	.text
.L335:
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$11, -2192(%rbp)
	jmp	.L349
.L330:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC25(%rip), %rax
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -2160(%rbp)
	movq	-2160(%rbp), %rax
	movq	%rax, -2200(%rbp)
	movq	$23, -2192(%rbp)
	jmp	.L349
.L339:
	cmpl	$0, -2240(%rbp)
	jle	.L350
	movq	$28, -2192(%rbp)
	jmp	.L349
.L350:
	movq	$20, -2192(%rbp)
	jmp	.L349
.L341:
	leaq	.LC24(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L345:
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	call	getenv@PLT
	movq	%rax, -2112(%rbp)
	movq	-2112(%rbp), %rax
	movq	%rax, -2224(%rbp)
	movq	-2272(%rbp), %rcx
	movq	-2224(%rbp), %rdx
	leaq	-2064(%rbp), %rax
	movq	%rcx, %r8
	movq	%rdx, %rcx
	leaq	.LC51(%rip), %rdx
	movl	$1024, %esi
	movq	%rax, %rdi
	movl	$0, %eax
	call	snprintf@PLT
	leaq	-2064(%rbp), %rax
	leaq	.LC0(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	popen@PLT
	movq	%rax, -2104(%rbp)
	movq	-2104(%rbp), %rax
	movq	%rax, -2216(%rbp)
	movq	$17, -2192(%rbp)
	jmp	.L349
.L331:
	cmpq	$0, -2200(%rbp)
	je	.L353
	movq	$10, -2192(%rbp)
	jmp	.L349
.L353:
	movq	$35, -2192(%rbp)
	jmp	.L349
.L346:
	cmpl	$-1, -2244(%rbp)
	jne	.L355
	movq	$33, -2192(%rbp)
	jmp	.L349
.L355:
	movq	$13, -2192(%rbp)
	jmp	.L349
.L337:
	cmpl	$0, -2240(%rbp)
	jne	.L357
	movq	$9, -2192(%rbp)
	jmp	.L349
.L357:
	movq	$14, -2192(%rbp)
	jmp	.L349
.L333:
	cmpl	$0, -2248(%rbp)
	je	.L359
	movq	$18, -2192(%rbp)
	jmp	.L349
.L359:
	movq	$22, -2192(%rbp)
	jmp	.L349
.L322:
	leaq	-1040(%rbp), %rdx
	movq	-2208(%rbp), %rax
	addq	%rdx, %rax
	movb	$0, (%rax)
	leaq	-1040(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$11, -2192(%rbp)
	jmp	.L349
.L329:
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-2096(%rbp), %rax
	movq	%rax, %rdi
	call	unlink@PLT
	movl	%eax, -2236(%rbp)
	movq	$2, -2192(%rbp)
	jmp	.L349
.L342:
	movq	-2216(%rbp), %rax
	movq	%rax, %rdi
	call	pclose@PLT
	movb	$47, -2096(%rbp)
	movb	$116, -2095(%rbp)
	movb	$109, -2094(%rbp)
	movb	$112, -2093(%rbp)
	movb	$47, -2092(%rbp)
	movb	$102, -2091(%rbp)
	movb	$105, -2090(%rbp)
	movb	$108, -2089(%rbp)
	movb	$101, -2088(%rbp)
	movb	$95, -2087(%rbp)
	movb	$108, -2086(%rbp)
	movb	$105, -2085(%rbp)
	movb	$115, -2084(%rbp)
	movb	$116, -2083(%rbp)
	movb	$46, -2082(%rbp)
	movb	$116, -2081(%rbp)
	movb	$120, -2080(%rbp)
	movb	$116, -2079(%rbp)
	movb	$0, -2078(%rbp)
	leaq	-2096(%rbp), %rax
	movl	$420, %edx
	movl	$65, %esi
	movq	%rax, %rdi
	movl	$0, %eax
	call	open@PLT
	movl	%eax, -2228(%rbp)
	movl	-2228(%rbp), %eax
	movl	%eax, -2244(%rbp)
	movq	$3, -2192(%rbp)
	jmp	.L349
.L344:
	movq	-2224(%rbp), %rax
	movq	%rax, %rdi
	call	chdir@PLT
	leaq	-2096(%rbp), %rax
	subq	$8, %rsp
	pushq	$0
	movq	%rax, %r9
	leaq	.LC17(%rip), %r8
	leaq	.LC6(%rip), %rax
	movq	%rax, %rcx
	leaq	.LC18(%rip), %rax
	movq	%rax, %rdx
	leaq	.LC19(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC19(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	execlp@PLT
	addq	$16, %rsp
	leaq	.LC20(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L340:
	leaq	-1040(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -2152(%rbp)
	movq	-2152(%rbp), %rdx
	leaq	-1040(%rbp), %rcx
	movl	-2244(%rbp), %eax
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	write@PLT
	movl	-2244(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	call	fork@PLT
	movl	%eax, -2232(%rbp)
	movl	-2232(%rbp), %eax
	movl	%eax, -2240(%rbp)
	movq	$16, -2192(%rbp)
	jmp	.L349
.L336:
	cmpq	$0, -2216(%rbp)
	jne	.L361
	movq	$12, -2192(%rbp)
	jmp	.L349
.L361:
	movq	$29, -2192(%rbp)
	jmp	.L349
.L320:
	movq	$8, -2192(%rbp)
	jmp	.L349
.L324:
	cmpq	$0, -2208(%rbp)
	jne	.L363
	movq	$39, -2192(%rbp)
	jmp	.L349
.L363:
	movq	$36, -2192(%rbp)
	jmp	.L349
.L332:
	leaq	.LC23(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$11, -2192(%rbp)
	jmp	.L349
.L328:
	movl	$0, %edi
	call	wait@PLT
	movq	$26, -2192(%rbp)
	jmp	.L349
.L325:
	leaq	.LC22(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$31, -2192(%rbp)
	jmp	.L349
.L321:
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$25, -2192(%rbp)
	jmp	.L349
.L343:
	movq	-2200(%rbp), %rax
	movl	$2, %edx
	movl	$0, %esi
	movq	%rax, %rdi
	call	fseek@PLT
	movq	-2200(%rbp), %rax
	movq	%rax, %rdi
	call	ftell@PLT
	movq	%rax, -2144(%rbp)
	movq	-2144(%rbp), %rax
	movq	%rax, -2136(%rbp)
	movq	-2200(%rbp), %rax
	movq	%rax, %rdi
	call	rewind@PLT
	movq	-2136(%rbp), %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -2128(%rbp)
	movq	-2128(%rbp), %rax
	movq	%rax, -2120(%rbp)
	movq	-2136(%rbp), %rdx
	movq	-2200(%rbp), %rcx
	movq	-2120(%rbp), %rax
	movl	$1, %esi
	movq	%rax, %rdi
	call	fread@PLT
	movq	-2200(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	-2136(%rbp), %rdx
	movq	-2120(%rbp), %rsi
	movl	-2260(%rbp), %eax
	movl	$0, %ecx
	movl	%eax, %edi
	call	send@PLT
	movq	-2120(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$15, -2192(%rbp)
	jmp	.L349
.L318:
	movq	-2216(%rbp), %rax
	movq	%rax, %rdi
	call	feof@PLT
	movl	%eax, -2248(%rbp)
	movq	$21, -2192(%rbp)
	jmp	.L349
.L323:
	leaq	.LC9(%rip), %rax
	movq	%rax, -2184(%rbp)
	movq	-2184(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -2176(%rbp)
	movq	-2176(%rbp), %rdx
	movq	-2184(%rbp), %rsi
	movl	-2260(%rbp), %eax
	movl	$0, %ecx
	movl	%eax, %edi
	call	send@PLT
	movq	$15, -2192(%rbp)
	jmp	.L349
.L327:
	movq	-2216(%rbp), %rdx
	leaq	-1040(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1023, %edx
	movl	$1, %esi
	movq	%rax, %rdi
	call	fread@PLT
	movq	%rax, -2168(%rbp)
	movq	-2168(%rbp), %rax
	movq	%rax, -2208(%rbp)
	movq	$34, -2192(%rbp)
	jmp	.L349
.L347:
	cmpl	$0, -2236(%rbp)
	je	.L365
	movq	$37, -2192(%rbp)
	jmp	.L349
.L365:
	movq	$25, -2192(%rbp)
	jmp	.L349
.L334:
	leaq	.LC21(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$0, -2192(%rbp)
	jmp	.L349
.L370:
	nop
.L349:
	jmp	.L367
.L371:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L369
	call	__stack_chk_fail@PLT
.L369:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	getfda, .-getfda
	.section	.rodata
.LC52:
	.string	"File not found"
	.align 8
.LC53:
	.string	"File: %s\nSize: %ld bytes\nPermissions: %s\nDate Created: %s"
	.text
	.globl	searchDirectory
	.type	searchDirectory, @function
searchDirectory:
.LFB11:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$3376, %rsp
	movl	%edi, -3348(%rbp)
	movq	%rsi, -3360(%rbp)
	movq	%rdx, -3368(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$1, -3304(%rbp)
.L434:
	cmpq	$40, -3304(%rbp)
	ja	.L437
	movq	-3304(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L375(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L375(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L375:
	.long	.L437-.L375
	.long	.L407-.L375
	.long	.L406-.L375
	.long	.L438-.L375
	.long	.L404-.L375
	.long	.L437-.L375
	.long	.L403-.L375
	.long	.L402-.L375
	.long	.L401-.L375
	.long	.L400-.L375
	.long	.L399-.L375
	.long	.L398-.L375
	.long	.L437-.L375
	.long	.L437-.L375
	.long	.L397-.L375
	.long	.L396-.L375
	.long	.L437-.L375
	.long	.L395-.L375
	.long	.L394-.L375
	.long	.L393-.L375
	.long	.L437-.L375
	.long	.L392-.L375
	.long	.L391-.L375
	.long	.L390-.L375
	.long	.L389-.L375
	.long	.L388-.L375
	.long	.L387-.L375
	.long	.L386-.L375
	.long	.L437-.L375
	.long	.L438-.L375
	.long	.L384-.L375
	.long	.L383-.L375
	.long	.L437-.L375
	.long	.L382-.L375
	.long	.L381-.L375
	.long	.L380-.L375
	.long	.L379-.L375
	.long	.L378-.L375
	.long	.L377-.L375
	.long	.L376-.L375
	.long	.L374-.L375
	.text
.L394:
	movl	-3224(%rbp), %eax
	andl	$128, %eax
	testl	%eax, %eax
	je	.L408
	movq	$22, -3304(%rbp)
	jmp	.L410
.L408:
	movq	$33, -3304(%rbp)
	jmp	.L410
.L388:
	movb	$45, -3092(%rbp)
	movq	$18, -3304(%rbp)
	jmp	.L410
.L404:
	cmpl	$0, -3336(%rbp)
	jne	.L411
	movq	$21, -3304(%rbp)
	jmp	.L410
.L411:
	movq	$31, -3304(%rbp)
	jmp	.L410
.L384:
	movl	$0, -3340(%rbp)
	movq	$21, -3304(%rbp)
	jmp	.L410
.L397:
	movl	-3224(%rbp), %eax
	andl	$61440, %eax
	cmpl	$32768, %eax
	jne	.L413
	movq	$34, -3304(%rbp)
	jmp	.L410
.L413:
	movq	$21, -3304(%rbp)
	jmp	.L410
.L396:
	movl	-3224(%rbp), %eax
	andl	$256, %eax
	testl	%eax, %eax
	je	.L415
	movq	$27, -3304(%rbp)
	jmp	.L410
.L415:
	movq	$25, -3304(%rbp)
	jmp	.L410
.L383:
	movq	-3312(%rbp), %rax
	addq	$19, %rax
	leaq	.LC29(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -3332(%rbp)
	movq	$8, -3304(%rbp)
	jmp	.L410
.L401:
	cmpl	$0, -3332(%rbp)
	jne	.L417
	movq	$21, -3304(%rbp)
	jmp	.L410
.L417:
	movq	$9, -3304(%rbp)
	jmp	.L410
.L407:
	movq	$36, -3304(%rbp)
	jmp	.L410
.L390:
	cmpl	$0, -3328(%rbp)
	jne	.L419
	movq	$15, -3304(%rbp)
	jmp	.L410
.L419:
	movq	$21, -3304(%rbp)
	jmp	.L410
.L389:
	cmpq	$0, -3312(%rbp)
	je	.L422
	movq	$7, -3304(%rbp)
	jmp	.L410
.L422:
	movq	$26, -3304(%rbp)
	jmp	.L410
.L392:
	movq	-3320(%rbp), %rax
	movq	%rax, %rdi
	call	readdir@PLT
	movq	%rax, -3312(%rbp)
	movq	$24, -3304(%rbp)
	jmp	.L410
.L379:
	movq	-3360(%rbp), %rax
	movq	%rax, %rdi
	call	opendir@PLT
	movq	%rax, -3296(%rbp)
	movq	-3296(%rbp), %rax
	movq	%rax, -3320(%rbp)
	movq	$6, -3304(%rbp)
	jmp	.L410
.L387:
	movq	-3320(%rbp), %rax
	movq	%rax, %rdi
	call	closedir@PLT
	movq	$39, -3304(%rbp)
	jmp	.L410
.L398:
	leaq	.LC52(%rip), %rax
	movq	%rax, -3264(%rbp)
	movq	-3264(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -3256(%rbp)
	movq	-3256(%rbp), %rax
	leaq	1(%rax), %rdx
	movq	-3264(%rbp), %rcx
	movl	-3348(%rbp), %eax
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	write@PLT
	movq	$29, -3304(%rbp)
	jmp	.L410
.L400:
	movq	-3312(%rbp), %rax
	leaq	19(%rax), %rcx
	movq	-3360(%rbp), %rdx
	leaq	-3088(%rbp), %rax
	movq	%rcx, %r8
	movq	%rdx, %rcx
	leaq	.LC26(%rip), %rdx
	movl	$1024, %esi
	movq	%rax, %rdi
	movl	$0, %eax
	call	snprintf@PLT
	leaq	-3248(%rbp), %rdx
	leaq	-3088(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	stat@PLT
	movl	%eax, -3324(%rbp)
	movq	$10, -3304(%rbp)
	jmp	.L410
.L393:
	movq	-3368(%rbp), %rdx
	leaq	-3088(%rbp), %rcx
	movl	-3348(%rbp), %eax
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	searchDirectory
	movq	$21, -3304(%rbp)
	jmp	.L410
.L395:
	movb	$0, -3089(%rbp)
	leaq	-3248(%rbp), %rax
	addq	$104, %rax
	movq	%rax, %rdi
	call	ctime@PLT
	movq	%rax, -3288(%rbp)
	movq	-3288(%rbp), %rax
	movq	%rax, -3280(%rbp)
	movq	-3200(%rbp), %rcx
	leaq	-3092(%rbp), %rsi
	movq	-3368(%rbp), %rdx
	leaq	-2064(%rbp), %rax
	subq	$8, %rsp
	pushq	-3280(%rbp)
	movq	%rsi, %r9
	movq	%rcx, %r8
	movq	%rdx, %rcx
	leaq	.LC53(%rip), %rdx
	movl	$2048, %esi
	movq	%rax, %rdi
	movl	$0, %eax
	call	snprintf@PLT
	addq	$16, %rsp
	leaq	-2064(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -3272(%rbp)
	movq	-3272(%rbp), %rax
	leaq	1(%rax), %rdx
	leaq	-2064(%rbp), %rcx
	movl	-3348(%rbp), %eax
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	write@PLT
	movl	$1, -3340(%rbp)
	movq	$26, -3304(%rbp)
	jmp	.L410
.L374:
	movb	$45, -3090(%rbp)
	movq	$17, -3304(%rbp)
	jmp	.L410
.L403:
	cmpq	$0, -3320(%rbp)
	jne	.L424
	movq	$2, -3304(%rbp)
	jmp	.L410
.L424:
	movq	$30, -3304(%rbp)
	jmp	.L410
.L386:
	movb	$114, -3092(%rbp)
	movq	$18, -3304(%rbp)
	jmp	.L410
.L377:
	movl	-3224(%rbp), %eax
	andl	$64, %eax
	testl	%eax, %eax
	je	.L426
	movq	$35, -3304(%rbp)
	jmp	.L410
.L426:
	movq	$40, -3304(%rbp)
	jmp	.L410
.L381:
	movq	-3312(%rbp), %rax
	leaq	19(%rax), %rdx
	movq	-3368(%rbp), %rax
	movq	%rax, %rsi
	movq	%rdx, %rdi
	call	strcmp@PLT
	movl	%eax, -3328(%rbp)
	movq	$23, -3304(%rbp)
	jmp	.L410
.L391:
	movb	$119, -3091(%rbp)
	movq	$38, -3304(%rbp)
	jmp	.L410
.L382:
	movb	$45, -3091(%rbp)
	movq	$38, -3304(%rbp)
	jmp	.L410
.L378:
	movl	-3224(%rbp), %eax
	andl	$61440, %eax
	cmpl	$16384, %eax
	jne	.L428
	movq	$19, -3304(%rbp)
	jmp	.L410
.L428:
	movq	$14, -3304(%rbp)
	jmp	.L410
.L399:
	cmpl	$0, -3324(%rbp)
	jne	.L430
	movq	$37, -3304(%rbp)
	jmp	.L410
.L430:
	movq	$21, -3304(%rbp)
	jmp	.L410
.L376:
	cmpl	$0, -3340(%rbp)
	jne	.L432
	movq	$11, -3304(%rbp)
	jmp	.L410
.L432:
	movq	$29, -3304(%rbp)
	jmp	.L410
.L402:
	movq	-3312(%rbp), %rax
	addq	$19, %rax
	leaq	.LC31(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -3336(%rbp)
	movq	$4, -3304(%rbp)
	jmp	.L410
.L380:
	movb	$120, -3090(%rbp)
	movq	$17, -3304(%rbp)
	jmp	.L410
.L406:
	leaq	.LC30(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$3, -3304(%rbp)
	jmp	.L410
.L437:
	nop
.L410:
	jmp	.L434
.L438:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L436
	call	__stack_chk_fail@PLT
.L436:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE11:
	.size	searchDirectory, .-searchDirectory
	.section	.rodata
	.align 8
.LC54:
	.string	"Mirror listening on port: %d...\n"
.LC55:
	.string	"Could not create socket\n"
.LC56:
	.string	"Mirror IP Address: %s\n"
.LC57:
	.string	"inet_ntop"
.LC58:
	.string	"%d"
.LC59:
	.string	"accept"
.LC60:
	.string	"Call model: %s <Port#>\n"
.LC61:
	.string	"error forking"
.LC62:
	.string	"getsockname"
	.text
	.globl	main
	.type	main, @function
main:
.LFB15:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$144, %rsp
	movl	%edi, -116(%rbp)
	movq	%rsi, -128(%rbp)
	movq	%rdx, -136(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_wxNp_envp(%rip)
	nop
.L440:
	movq	$0, _TIG_IZ_wxNp_argv(%rip)
	nop
.L441:
	movl	$0, _TIG_IZ_wxNp_argc(%rip)
	nop
	nop
.L442:
.L443:
#APP
# 157 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-wxNp--0
# 0 "" 2
#NO_APP
	movl	-116(%rbp), %eax
	movl	%eax, _TIG_IZ_wxNp_argc(%rip)
	movq	-128(%rbp), %rax
	movq	%rax, _TIG_IZ_wxNp_argv(%rip)
	movq	-136(%rbp), %rax
	movq	%rax, _TIG_IZ_wxNp_envp(%rip)
	nop
	movq	$9, -72(%rbp)
.L482:
	cmpq	$37, -72(%rbp)
	ja	.L484
	movq	-72(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L446(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L446(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L446:
	.long	.L466-.L446
	.long	.L484-.L446
	.long	.L484-.L446
	.long	.L465-.L446
	.long	.L484-.L446
	.long	.L464-.L446
	.long	.L463-.L446
	.long	.L462-.L446
	.long	.L461-.L446
	.long	.L460-.L446
	.long	.L459-.L446
	.long	.L458-.L446
	.long	.L484-.L446
	.long	.L484-.L446
	.long	.L484-.L446
	.long	.L484-.L446
	.long	.L484-.L446
	.long	.L484-.L446
	.long	.L457-.L446
	.long	.L484-.L446
	.long	.L456-.L446
	.long	.L484-.L446
	.long	.L484-.L446
	.long	.L484-.L446
	.long	.L455-.L446
	.long	.L454-.L446
	.long	.L453-.L446
	.long	.L484-.L446
	.long	.L484-.L446
	.long	.L452-.L446
	.long	.L451-.L446
	.long	.L484-.L446
	.long	.L450-.L446
	.long	.L449-.L446
	.long	.L448-.L446
	.long	.L447-.L446
	.long	.L484-.L446
	.long	.L445-.L446
	.text
.L457:
	cmpl	$0, -100(%rbp)
	jns	.L467
	movq	$5, -72(%rbp)
	jmp	.L469
.L467:
	movq	$35, -72(%rbp)
	jmp	.L469
.L454:
	movl	-112(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC54(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-104(%rbp), %eax
	movl	$0, %edx
	movl	$0, %esi
	movl	%eax, %edi
	call	accept@PLT
	movl	%eax, -100(%rbp)
	movq	$18, -72(%rbp)
	jmp	.L469
.L451:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$24, %edx
	movl	$1, %esi
	leaq	.LC55(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movl	$1, %edi
	call	exit@PLT
.L461:
	cmpl	$-1, -96(%rbp)
	jne	.L470
	movq	$29, -72(%rbp)
	jmp	.L469
.L470:
	movq	$26, -72(%rbp)
	jmp	.L469
.L465:
	movl	$0, %edx
	movl	$1, %esi
	movl	$2, %edi
	call	socket@PLT
	movl	%eax, -104(%rbp)
	movq	$34, -72(%rbp)
	jmp	.L469
.L455:
	leaq	-32(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC56(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-104(%rbp), %eax
	movl	$4, %esi
	movl	%eax, %edi
	call	listen@PLT
	movl	$1, -92(%rbp)
	movq	$25, -72(%rbp)
	jmp	.L469
.L453:
	leaq	-32(%rbp), %rax
	leaq	-48(%rbp), %rdx
	leaq	4(%rdx), %rsi
	movl	$16, %ecx
	movq	%rax, %rdx
	movl	$2, %edi
	call	inet_ntop@PLT
	movq	%rax, -80(%rbp)
	movq	$10, -72(%rbp)
	jmp	.L469
.L458:
	cmpl	$0, -88(%rbp)
	jle	.L472
	movq	$0, -72(%rbp)
	jmp	.L469
.L472:
	movq	$7, -72(%rbp)
	jmp	.L469
.L460:
	cmpl	$2, -116(%rbp)
	je	.L474
	movq	$37, -72(%rbp)
	jmp	.L469
.L474:
	movq	$3, -72(%rbp)
	jmp	.L469
.L450:
	leaq	.LC57(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L463:
	movw	$2, -64(%rbp)
	movl	$0, %edi
	call	htonl@PLT
	movl	%eax, -60(%rbp)
	movq	-128(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	leaq	-112(%rbp), %rdx
	leaq	.LC58(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_sscanf@PLT
	movl	-112(%rbp), %eax
	movzwl	%ax, %eax
	movl	%eax, %edi
	call	htons@PLT
	movw	%ax, -62(%rbp)
	leaq	-64(%rbp), %rcx
	movl	-104(%rbp), %eax
	movl	$16, %edx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	bind@PLT
	movl	$16, -108(%rbp)
	leaq	-108(%rbp), %rdx
	leaq	-48(%rbp), %rcx
	movl	-104(%rbp), %eax
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	getsockname@PLT
	movl	%eax, -96(%rbp)
	movq	$8, -72(%rbp)
	jmp	.L469
.L448:
	cmpl	$0, -104(%rbp)
	jns	.L476
	movq	$30, -72(%rbp)
	jmp	.L469
.L476:
	movq	$6, -72(%rbp)
	jmp	.L469
.L464:
	leaq	.LC59(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$25, -72(%rbp)
	jmp	.L469
.L449:
	cmpl	$0, -88(%rbp)
	jne	.L478
	movq	$20, -72(%rbp)
	jmp	.L469
.L478:
	movq	$11, -72(%rbp)
	jmp	.L469
.L445:
	movq	-128(%rbp), %rax
	movq	(%rax), %rdx
	movq	stderr(%rip), %rax
	leaq	.LC60(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movl	$0, %edi
	call	exit@PLT
.L459:
	cmpq	$0, -80(%rbp)
	jne	.L480
	movq	$32, -72(%rbp)
	jmp	.L469
.L480:
	movq	$24, -72(%rbp)
	jmp	.L469
.L466:
	movl	-100(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	addl	$1, -92(%rbp)
	movq	$25, -72(%rbp)
	jmp	.L469
.L462:
	leaq	.LC61(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$25, -72(%rbp)
	jmp	.L469
.L447:
	call	fork@PLT
	movl	%eax, -84(%rbp)
	movl	-84(%rbp), %eax
	movl	%eax, -88(%rbp)
	movq	$33, -72(%rbp)
	jmp	.L469
.L452:
	leaq	.LC62(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L456:
	movl	-104(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	movl	-100(%rbp), %eax
	movl	%eax, %edi
	call	pclientrequest
	movl	-100(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	movl	$0, %edi
	call	exit@PLT
.L484:
	nop
.L469:
	jmp	.L482
	.cfi_endproc
.LFE15:
	.size	main, .-main
	.section	.rodata
.LC63:
	.string	"Requested file: %s\n"
	.align 8
.LC64:
	.string	"Searching directory tree rooted at: %s\n"
	.text
	.globl	getfn
	.type	getfn, @function
getfn:
.LFB16:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movl	%edi, -52(%rbp)
	movq	%rsi, -64(%rbp)
	movq	$2, -40(%rbp)
.L491:
	cmpq	$2, -40(%rbp)
	je	.L486
	cmpq	$2, -40(%rbp)
	ja	.L492
	cmpq	$0, -40(%rbp)
	je	.L493
	cmpq	$1, -40(%rbp)
	jne	.L492
	movq	-64(%rbp), %rax
	addq	$6, %rax
	movq	%rax, -32(%rbp)
	movq	-32(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC63(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	call	getenv@PLT
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC64(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-32(%rbp), %rdx
	movq	-16(%rbp), %rcx
	movl	-52(%rbp), %eax
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	searchDirectory
	leaq	.LC52(%rip), %rax
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rcx
	movl	-52(%rbp), %eax
	movl	$100, %edx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	write@PLT
	movq	$0, -40(%rbp)
	jmp	.L489
.L486:
	movq	$1, -40(%rbp)
	jmp	.L489
.L492:
	nop
.L489:
	jmp	.L491
.L493:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE16:
	.size	getfn, .-getfn
	.globl	findsizerange
	.type	findsizerange, @function
findsizerange:
.LFB17:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$496, %rsp
	movq	%rdi, -472(%rbp)
	movq	%rsi, -480(%rbp)
	movq	%rdx, -488(%rbp)
	movq	%rcx, -496(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$1, -432(%rbp)
.L539:
	cmpq	$26, -432(%rbp)
	ja	.L542
	movq	-432(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L497(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L497(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L497:
	.long	.L517-.L497
	.long	.L516-.L497
	.long	.L515-.L497
	.long	.L542-.L497
	.long	.L514-.L497
	.long	.L513-.L497
	.long	.L542-.L497
	.long	.L542-.L497
	.long	.L526-.L497
	.long	.L542-.L497
	.long	.L511-.L497
	.long	.L510-.L497
	.long	.L543-.L497
	.long	.L508-.L497
	.long	.L543-.L497
	.long	.L506-.L497
	.long	.L505-.L497
	.long	.L542-.L497
	.long	.L504-.L497
	.long	.L503-.L497
	.long	.L502-.L497
	.long	.L501-.L497
	.long	.L542-.L497
	.long	.L500-.L497
	.long	.L499-.L497
	.long	.L498-.L497
	.long	.L496-.L497
	.text
.L504:
	cmpq	$0, -448(%rbp)
	jne	.L518
	movq	$16, -432(%rbp)
	jmp	.L520
.L518:
	movq	$21, -432(%rbp)
	jmp	.L520
.L498:
	cmpq	$0, -440(%rbp)
	je	.L521
	movq	$24, -432(%rbp)
	jmp	.L520
.L521:
	movq	$2, -432(%rbp)
	jmp	.L520
.L514:
	leaq	-272(%rbp), %rdx
	movq	-488(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	appendFilePath
	movq	$21, -432(%rbp)
	jmp	.L520
.L506:
	movl	-392(%rbp), %eax
	andl	$61440, %eax
	cmpl	$32768, %eax
	jne	.L524
	movq	$0, -432(%rbp)
	jmp	.L520
.L524:
	movq	$8, -432(%rbp)
	jmp	.L520
.L512:
.L526:
	movl	-392(%rbp), %eax
	andl	$61440, %eax
	cmpl	$16384, %eax
	jne	.L527
	movq	$5, -432(%rbp)
	jmp	.L520
.L527:
	movq	$21, -432(%rbp)
	jmp	.L520
.L516:
	movq	$19, -432(%rbp)
	jmp	.L520
.L500:
	movq	-440(%rbp), %rax
	leaq	19(%rax), %rcx
	movq	-496(%rbp), %rdx
	leaq	-272(%rbp), %rax
	movq	%rcx, %r8
	movq	%rdx, %rcx
	leaq	.LC26(%rip), %rdx
	movl	$256, %esi
	movq	%rax, %rdi
	movl	$0, %eax
	call	snprintf@PLT
	leaq	-416(%rbp), %rdx
	leaq	-272(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	stat@PLT
	movl	%eax, -452(%rbp)
	movq	$20, -432(%rbp)
	jmp	.L520
.L505:
	leaq	.LC30(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$12, -432(%rbp)
	jmp	.L520
.L499:
	movq	-440(%rbp), %rax
	addq	$19, %rax
	leaq	.LC31(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -460(%rbp)
	movq	$11, -432(%rbp)
	jmp	.L520
.L501:
	movq	-448(%rbp), %rax
	movq	%rax, %rdi
	call	readdir@PLT
	movq	%rax, -440(%rbp)
	movq	$25, -432(%rbp)
	jmp	.L520
.L496:
	movq	-368(%rbp), %rax
	cmpq	%rax, -480(%rbp)
	jb	.L529
	movq	$4, -432(%rbp)
	jmp	.L520
.L529:
	movq	$8, -432(%rbp)
	jmp	.L520
.L510:
	cmpl	$0, -460(%rbp)
	jne	.L531
	movq	$21, -432(%rbp)
	jmp	.L520
.L531:
	movq	$13, -432(%rbp)
	jmp	.L520
.L508:
	movq	-440(%rbp), %rax
	addq	$19, %rax
	leaq	.LC29(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -456(%rbp)
	movq	$10, -432(%rbp)
	jmp	.L520
.L503:
	movq	-496(%rbp), %rax
	movq	%rax, %rdi
	call	opendir@PLT
	movq	%rax, -424(%rbp)
	movq	-424(%rbp), %rax
	movq	%rax, -448(%rbp)
	movq	$18, -432(%rbp)
	jmp	.L520
.L513:
	leaq	-272(%rbp), %rcx
	movq	-488(%rbp), %rdx
	movq	-480(%rbp), %rsi
	movq	-472(%rbp), %rax
	movq	%rax, %rdi
	call	findsizerange
	movq	$21, -432(%rbp)
	jmp	.L520
.L511:
	cmpl	$0, -456(%rbp)
	jne	.L533
	movq	$21, -432(%rbp)
	jmp	.L520
.L533:
	movq	$23, -432(%rbp)
	jmp	.L520
.L517:
	movq	-368(%rbp), %rax
	cmpq	%rax, -472(%rbp)
	ja	.L535
	movq	$26, -432(%rbp)
	jmp	.L520
.L535:
	movq	$8, -432(%rbp)
	jmp	.L520
.L515:
	movq	-448(%rbp), %rax
	movq	%rax, %rdi
	call	closedir@PLT
	movq	$14, -432(%rbp)
	jmp	.L520
.L502:
	cmpl	$0, -452(%rbp)
	jne	.L537
	movq	$15, -432(%rbp)
	jmp	.L520
.L537:
	movq	$21, -432(%rbp)
	jmp	.L520
.L542:
	nop
.L520:
	jmp	.L539
.L543:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L541
	call	__stack_chk_fail@PLT
.L541:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE17:
	.size	findsizerange, .-findsizerange
	.section	.rodata
.LC65:
	.string	"mkdir"
	.align 8
.LC66:
	.string	"/home/iktider/Desktop/f23project"
	.text
	.globl	createTarDir
	.type	createTarDir, @function
createTarDir:
.LFB18:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	$0, -16(%rbp)
.L556:
	cmpq	$5, -16(%rbp)
	ja	.L557
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L547(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L547(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L547:
	.long	.L551-.L547
	.long	.L550-.L547
	.long	.L549-.L547
	.long	.L557-.L547
	.long	.L548-.L547
	.long	.L558-.L547
	.text
.L548:
	leaq	.LC65(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L550:
	leaq	.LC66(%rip), %rax
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movl	$509, %esi
	movq	%rax, %rdi
	call	mkdir@PLT
	movl	%eax, -20(%rbp)
	movq	$2, -16(%rbp)
	jmp	.L552
.L551:
	movq	$1, -16(%rbp)
	jmp	.L552
.L549:
	cmpl	$0, -20(%rbp)
	jne	.L554
	movq	$5, -16(%rbp)
	jmp	.L552
.L554:
	movq	$4, -16(%rbp)
	jmp	.L552
.L557:
	nop
.L552:
	jmp	.L556
.L558:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE18:
	.size	createTarDir, .-createTarDir
	.section	.rodata
.LC67:
	.string	" "
	.text
	.globl	countExtensions
	.type	countExtensions, @function
countExtensions:
.LFB19:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$320, %rsp
	movq	%rdi, -312(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$2, -288(%rbp)
.L571:
	cmpq	$7, -288(%rbp)
	ja	.L574
	movq	-288(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L562(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L562(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L562:
	.long	.L574-.L562
	.long	.L566-.L562
	.long	.L565-.L562
	.long	.L574-.L562
	.long	.L564-.L562
	.long	.L563-.L562
	.long	.L574-.L562
	.long	.L561-.L562
	.text
.L564:
	movq	-312(%rbp), %rdx
	leaq	-272(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcpy@PLT
	leaq	-272(%rbp), %rax
	leaq	.LC67(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strtok@PLT
	movq	%rax, -280(%rbp)
	movq	-280(%rbp), %rax
	movq	%rax, -296(%rbp)
	movl	$0, -300(%rbp)
	movq	$1, -288(%rbp)
	jmp	.L567
.L566:
	cmpq	$0, -296(%rbp)
	je	.L568
	movq	$5, -288(%rbp)
	jmp	.L567
.L568:
	movq	$7, -288(%rbp)
	jmp	.L567
.L563:
	addl	$1, -300(%rbp)
	leaq	.LC67(%rip), %rax
	movq	%rax, %rsi
	movl	$0, %edi
	call	strtok@PLT
	movq	%rax, -296(%rbp)
	movq	$1, -288(%rbp)
	jmp	.L567
.L561:
	movl	-300(%rbp), %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L572
	jmp	.L573
.L565:
	movq	$4, -288(%rbp)
	jmp	.L567
.L574:
	nop
.L567:
	jmp	.L571
.L573:
	call	__stack_chk_fail@PLT
.L572:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE19:
	.size	countExtensions, .-countExtensions
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
