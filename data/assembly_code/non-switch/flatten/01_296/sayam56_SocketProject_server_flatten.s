	.file	"sayam56_SocketProject_server_flatten.c"
	.text
	.globl	_TIG_IZ_Mhdh_argv
	.bss
	.align 8
	.type	_TIG_IZ_Mhdh_argv, @object
	.size	_TIG_IZ_Mhdh_argv, 8
_TIG_IZ_Mhdh_argv:
	.zero	8
	.globl	_TIG_IZ_Mhdh_argc
	.align 4
	.type	_TIG_IZ_Mhdh_argc, @object
	.size	_TIG_IZ_Mhdh_argc, 4
_TIG_IZ_Mhdh_argc:
	.zero	4
	.globl	_TIG_IZ_Mhdh_envp
	.align 8
	.type	_TIG_IZ_Mhdh_envp, @object
	.size	_TIG_IZ_Mhdh_envp, 8
_TIG_IZ_Mhdh_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"Files:\n%s\n"
.LC1:
	.string	"fork"
	.align 8
.LC2:
	.string	"Tar.gz archive created: temp.tar.gz"
.LC3:
	.string	"Error creating tar.gz file"
.LC4:
	.string	"unlink"
.LC5:
	.string	"popen"
.LC6:
	.string	"HOME"
.LC7:
	.string	"find %s -type f -newermt %s"
.LC8:
	.string	"r"
.LC9:
	.string	"rb"
.LC10:
	.string	"temp.tar.gz"
.LC11:
	.string	"-T"
.LC12:
	.string	"f23project/temp.tar.gz"
.LC13:
	.string	"-czvf"
.LC14:
	.string	"tar"
.LC15:
	.string	"execlp"
.LC16:
	.string	"open"
.LC17:
	.string	"fread"
.LC18:
	.string	"No files found."
	.text
	.globl	getfda
	.type	getfda, @function
getfda:
.LFB0:
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
	movq	$39, -2192(%rbp)
.L52:
	cmpq	$39, -2192(%rbp)
	ja	.L55
	movq	-2192(%rbp), %rax
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
	.long	.L33-.L4
	.long	.L32-.L4
	.long	.L31-.L4
	.long	.L55-.L4
	.long	.L30-.L4
	.long	.L55-.L4
	.long	.L29-.L4
	.long	.L28-.L4
	.long	.L56-.L4
	.long	.L26-.L4
	.long	.L55-.L4
	.long	.L25-.L4
	.long	.L55-.L4
	.long	.L24-.L4
	.long	.L23-.L4
	.long	.L22-.L4
	.long	.L21-.L4
	.long	.L20-.L4
	.long	.L19-.L4
	.long	.L56-.L4
	.long	.L17-.L4
	.long	.L16-.L4
	.long	.L15-.L4
	.long	.L55-.L4
	.long	.L14-.L4
	.long	.L13-.L4
	.long	.L55-.L4
	.long	.L55-.L4
	.long	.L12-.L4
	.long	.L11-.L4
	.long	.L55-.L4
	.long	.L55-.L4
	.long	.L56-.L4
	.long	.L9-.L4
	.long	.L8-.L4
	.long	.L55-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L19:
	leaq	-1040(%rbp), %rdx
	movq	-2208(%rbp), %rax
	addq	%rdx, %rax
	movb	$0, (%rax)
	leaq	-1040(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$34, -2192(%rbp)
	jmp	.L34
.L13:
	cmpq	$0, -2216(%rbp)
	jne	.L35
	movq	$17, -2192(%rbp)
	jmp	.L34
.L35:
	movq	$28, -2192(%rbp)
	jmp	.L34
.L30:
	cmpl	$0, -2248(%rbp)
	je	.L37
	movq	$20, -2192(%rbp)
	jmp	.L34
.L37:
	movq	$2, -2192(%rbp)
	jmp	.L34
.L23:
	cmpl	$0, -2240(%rbp)
	jle	.L39
	movq	$15, -2192(%rbp)
	jmp	.L34
.L39:
	movq	$16, -2192(%rbp)
	jmp	.L34
.L22:
	movl	$0, %edi
	call	wait@PLT
	movq	$36, -2192(%rbp)
	jmp	.L34
.L32:
	movq	-2200(%rbp), %rax
	movl	$2, %edx
	movl	$0, %esi
	movq	%rax, %rdi
	call	fseek@PLT
	movq	-2200(%rbp), %rax
	movq	%rax, %rdi
	call	ftell@PLT
	movq	%rax, -2128(%rbp)
	movq	-2128(%rbp), %rax
	movq	%rax, -2120(%rbp)
	movq	-2200(%rbp), %rax
	movq	%rax, %rdi
	call	rewind@PLT
	movq	-2120(%rbp), %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -2112(%rbp)
	movq	-2112(%rbp), %rax
	movq	%rax, -2104(%rbp)
	movq	-2120(%rbp), %rdx
	movq	-2200(%rbp), %rcx
	movq	-2104(%rbp), %rax
	movl	$1, %esi
	movq	%rax, %rdi
	call	fread@PLT
	movq	-2200(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	-2120(%rbp), %rdx
	movq	-2104(%rbp), %rsi
	movl	-2260(%rbp), %eax
	movl	$0, %ecx
	movl	%eax, %edi
	call	send@PLT
	movq	-2104(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$8, -2192(%rbp)
	jmp	.L34
.L21:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$19, -2192(%rbp)
	jmp	.L34
.L14:
	cmpq	$0, -2208(%rbp)
	jne	.L42
	movq	$37, -2192(%rbp)
	jmp	.L34
.L42:
	movq	$18, -2192(%rbp)
	jmp	.L34
.L16:
	cmpl	$-1, -2244(%rbp)
	jne	.L44
	movq	$29, -2192(%rbp)
	jmp	.L34
.L44:
	movq	$38, -2192(%rbp)
	jmp	.L34
.L7:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-2096(%rbp), %rax
	movq	%rax, %rdi
	call	unlink@PLT
	movl	%eax, -2236(%rbp)
	movq	$13, -2192(%rbp)
	jmp	.L34
.L25:
	leaq	.LC3(%rip), %rax
	movq	%rax, -2160(%rbp)
	movq	-2160(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -2152(%rbp)
	movq	-2152(%rbp), %rdx
	movq	-2160(%rbp), %rsi
	movl	-2260(%rbp), %eax
	movl	$0, %ecx
	movl	%eax, %edi
	call	send@PLT
	movq	$8, -2192(%rbp)
	jmp	.L34
.L26:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$33, -2192(%rbp)
	jmp	.L34
.L24:
	cmpl	$0, -2236(%rbp)
	je	.L46
	movq	$9, -2192(%rbp)
	jmp	.L34
.L46:
	movq	$33, -2192(%rbp)
	jmp	.L34
.L20:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L29:
	leaq	.LC6(%rip), %rax
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
	leaq	.LC7(%rip), %rdx
	movl	$1024, %esi
	movq	%rax, %rdi
	movl	$0, %eax
	call	snprintf@PLT
	leaq	-2064(%rbp), %rax
	leaq	.LC8(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	popen@PLT
	movq	%rax, -2136(%rbp)
	movq	-2136(%rbp), %rax
	movq	%rax, -2216(%rbp)
	movq	$25, -2192(%rbp)
	jmp	.L34
.L5:
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
	movq	$0, -2192(%rbp)
	jmp	.L34
.L8:
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
	movq	$21, -2192(%rbp)
	jmp	.L34
.L15:
	cmpq	$0, -2200(%rbp)
	je	.L48
	movq	$1, -2192(%rbp)
	jmp	.L34
.L48:
	movq	$11, -2192(%rbp)
	jmp	.L34
.L12:
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
	movq	$24, -2192(%rbp)
	jmp	.L34
.L9:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -2176(%rbp)
	movq	-2176(%rbp), %rax
	movq	%rax, -2200(%rbp)
	movq	$22, -2192(%rbp)
	jmp	.L34
.L6:
	movq	-2216(%rbp), %rax
	movq	%rax, %rdi
	call	feof@PLT
	movl	%eax, -2248(%rbp)
	movq	$4, -2192(%rbp)
	jmp	.L34
.L33:
	cmpl	$0, -2240(%rbp)
	jne	.L50
	movq	$7, -2192(%rbp)
	jmp	.L34
.L50:
	movq	$14, -2192(%rbp)
	jmp	.L34
.L3:
	movq	$6, -2192(%rbp)
	jmp	.L34
.L28:
	movq	-2224(%rbp), %rax
	movq	%rax, %rdi
	call	chdir@PLT
	leaq	-2096(%rbp), %rax
	subq	$8, %rsp
	pushq	$0
	movq	%rax, %r9
	leaq	.LC11(%rip), %r8
	leaq	.LC12(%rip), %rax
	movq	%rax, %rcx
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdx
	leaq	.LC14(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	execlp@PLT
	addq	$16, %rsp
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L11:
	leaq	.LC16(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$32, -2192(%rbp)
	jmp	.L34
.L31:
	leaq	.LC17(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$34, -2192(%rbp)
	jmp	.L34
.L17:
	leaq	.LC18(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$34, -2192(%rbp)
	jmp	.L34
.L55:
	nop
.L34:
	jmp	.L52
.L56:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L54
	call	__stack_chk_fail@PLT
.L54:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	getfda, .-getfda
	.section	.rodata
.LC19:
	.string	"."
.LC20:
	.string	"opendir"
.LC21:
	.string	"%s/%s"
.LC22:
	.string	".."
	.text
	.globl	searchforFiles
	.type	searchforFiles, @function
searchforFiles:
.LFB1:
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
	movq	$19, -1200(%rbp)
.L99:
	cmpq	$27, -1200(%rbp)
	ja	.L102
	movq	-1200(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L60(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L60(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L60:
	.long	.L80-.L60
	.long	.L102-.L60
	.long	.L102-.L60
	.long	.L79-.L60
	.long	.L103-.L60
	.long	.L77-.L60
	.long	.L76-.L60
	.long	.L75-.L60
	.long	.L74-.L60
	.long	.L73-.L60
	.long	.L72-.L60
	.long	.L71-.L60
	.long	.L70-.L60
	.long	.L69-.L60
	.long	.L68-.L60
	.long	.L102-.L60
	.long	.L67-.L60
	.long	.L66-.L60
	.long	.L65-.L60
	.long	.L64-.L60
	.long	.L102-.L60
	.long	.L102-.L60
	.long	.L63-.L60
	.long	.L103-.L60
	.long	.L102-.L60
	.long	.L61-.L60
	.long	.L102-.L60
	.long	.L59-.L60
	.text
.L65:
	movq	-1208(%rbp), %rax
	addq	$19, %rax
	leaq	.LC19(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -1232(%rbp)
	movq	$7, -1200(%rbp)
	jmp	.L81
.L61:
	cmpl	$0, -1224(%rbp)
	je	.L82
	movq	$6, -1200(%rbp)
	jmp	.L81
.L82:
	movq	$12, -1200(%rbp)
	jmp	.L81
.L68:
	cmpq	$0, -1216(%rbp)
	jne	.L85
	movq	$3, -1200(%rbp)
	jmp	.L81
.L85:
	movq	$6, -1200(%rbp)
	jmp	.L81
.L70:
	movl	-1160(%rbp), %eax
	andl	$61440, %eax
	cmpl	$16384, %eax
	jne	.L87
	movq	$0, -1200(%rbp)
	jmp	.L81
.L87:
	movq	$10, -1200(%rbp)
	jmp	.L81
.L74:
	cmpl	$0, -1228(%rbp)
	jne	.L89
	movq	$6, -1200(%rbp)
	jmp	.L81
.L89:
	movq	$16, -1200(%rbp)
	jmp	.L81
.L79:
	leaq	.LC20(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$23, -1200(%rbp)
	jmp	.L81
.L67:
	movq	-1208(%rbp), %rax
	leaq	19(%rax), %rcx
	movq	-1248(%rbp), %rdx
	leaq	-1040(%rbp), %rax
	movq	%rcx, %r8
	movq	%rdx, %rcx
	leaq	.LC21(%rip), %rdx
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
	movq	$25, -1200(%rbp)
	jmp	.L81
.L71:
	cmpq	$0, -1208(%rbp)
	je	.L91
	movq	$18, -1200(%rbp)
	jmp	.L81
.L91:
	movq	$22, -1200(%rbp)
	jmp	.L81
.L73:
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
	movq	$6, -1200(%rbp)
	jmp	.L81
.L69:
	movq	-1248(%rbp), %rax
	movq	%rax, %rdi
	call	opendir@PLT
	movq	%rax, -1192(%rbp)
	movq	-1192(%rbp), %rax
	movq	%rax, -1216(%rbp)
	movq	$14, -1200(%rbp)
	jmp	.L81
.L64:
	movq	$13, -1200(%rbp)
	jmp	.L81
.L66:
	movq	-1208(%rbp), %rax
	addq	$19, %rax
	leaq	.LC22(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -1228(%rbp)
	movq	$8, -1200(%rbp)
	jmp	.L81
.L76:
	movq	-1216(%rbp), %rax
	movq	%rax, %rdi
	call	readdir@PLT
	movq	%rax, -1208(%rbp)
	movq	$11, -1200(%rbp)
	jmp	.L81
.L59:
	cmpl	$0, -1220(%rbp)
	jne	.L93
	movq	$9, -1200(%rbp)
	jmp	.L81
.L93:
	movq	$6, -1200(%rbp)
	jmp	.L81
.L63:
	movq	-1216(%rbp), %rax
	movq	%rax, %rdi
	call	closedir@PLT
	movq	$4, -1200(%rbp)
	jmp	.L81
.L77:
	movq	-1208(%rbp), %rax
	leaq	19(%rax), %rdx
	movq	-1256(%rbp), %rax
	movq	%rax, %rsi
	movq	%rdx, %rdi
	call	strcmp@PLT
	movl	%eax, -1220(%rbp)
	movq	$27, -1200(%rbp)
	jmp	.L81
.L72:
	movl	-1160(%rbp), %eax
	andl	$61440, %eax
	cmpl	$32768, %eax
	jne	.L95
	movq	$5, -1200(%rbp)
	jmp	.L81
.L95:
	movq	$6, -1200(%rbp)
	jmp	.L81
.L80:
	movq	-1264(%rbp), %rcx
	movq	-1256(%rbp), %rdx
	leaq	-1040(%rbp), %rsi
	movq	-1240(%rbp), %rax
	movq	%rax, %rdi
	call	searchforFiles
	movq	$6, -1200(%rbp)
	jmp	.L81
.L75:
	cmpl	$0, -1232(%rbp)
	jne	.L97
	movq	$6, -1200(%rbp)
	jmp	.L81
.L97:
	movq	$17, -1200(%rbp)
	jmp	.L81
.L102:
	nop
.L81:
	jmp	.L99
.L103:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L101
	call	__stack_chk_fail@PLT
.L101:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	searchforFiles, .-searchforFiles
	.section	.rodata
.LC23:
	.string	"mkdir"
	.align 8
.LC24:
	.string	"/home/iktider/Desktop/f23project"
	.text
	.globl	createTarDir
	.type	createTarDir, @function
createTarDir:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	$4, -16(%rbp)
.L116:
	cmpq	$5, -16(%rbp)
	ja	.L117
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L107(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L107(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L107:
	.long	.L118-.L107
	.long	.L110-.L107
	.long	.L117-.L107
	.long	.L109-.L107
	.long	.L108-.L107
	.long	.L106-.L107
	.text
.L108:
	movq	$3, -16(%rbp)
	jmp	.L112
.L110:
	leaq	.LC23(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L109:
	leaq	.LC24(%rip), %rax
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movl	$509, %esi
	movq	%rax, %rdi
	call	mkdir@PLT
	movl	%eax, -20(%rbp)
	movq	$5, -16(%rbp)
	jmp	.L112
.L106:
	cmpl	$0, -20(%rbp)
	jne	.L113
	movq	$0, -16(%rbp)
	jmp	.L112
.L113:
	movq	$1, -16(%rbp)
	jmp	.L112
.L117:
	nop
.L112:
	jmp	.L116
.L118:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	createTarDir, .-createTarDir
	.section	.rodata
.LC25:
	.string	"getfn "
.LC26:
	.string	"getfz "
.LC27:
	.string	"getfda "
.LC28:
	.string	"Client Process Terminated!"
.LC29:
	.string	"%s"
	.align 8
.LC30:
	.string	"Requested files created on or after: %s\n"
.LC31:
	.string	"read"
	.align 8
.LC32:
	.string	"Requested files created on or before: %s\n"
.LC33:
	.string	"Unknown command"
.LC34:
	.string	"getft "
.LC35:
	.string	"quitc"
.LC36:
	.string	"getfdb "
	.text
	.globl	pclientrequest
	.type	pclientrequest, @function
pclientrequest:
.LFB3:
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
	movq	$4, -1136(%rbp)
.L163:
	cmpq	$28, -1136(%rbp)
	ja	.L166
	movq	-1136(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L122(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L122(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L122:
	.long	.L146-.L122
	.long	.L166-.L122
	.long	.L145-.L122
	.long	.L144-.L122
	.long	.L143-.L122
	.long	.L142-.L122
	.long	.L141-.L122
	.long	.L140-.L122
	.long	.L139-.L122
	.long	.L138-.L122
	.long	.L166-.L122
	.long	.L137-.L122
	.long	.L136-.L122
	.long	.L135-.L122
	.long	.L134-.L122
	.long	.L133-.L122
	.long	.L166-.L122
	.long	.L132-.L122
	.long	.L167-.L122
	.long	.L130-.L122
	.long	.L167-.L122
	.long	.L128-.L122
	.long	.L127-.L122
	.long	.L126-.L122
	.long	.L166-.L122
	.long	.L125-.L122
	.long	.L124-.L122
	.long	.L123-.L122
	.long	.L121-.L122
	.text
.L125:
	leaq	-1040(%rbp), %rax
	movl	$5, %edx
	leaq	.LC25(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	strncmp@PLT
	movl	%eax, -1156(%rbp)
	movq	$27, -1136(%rbp)
	jmp	.L148
.L143:
	movq	$19, -1136(%rbp)
	jmp	.L148
.L134:
	leaq	-1040(%rbp), %rax
	movl	$5, %edx
	leaq	.LC26(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	strncmp@PLT
	movl	%eax, -1152(%rbp)
	movq	$17, -1136(%rbp)
	jmp	.L148
.L133:
	leaq	-1040(%rbp), %rdx
	movl	-1172(%rbp), %eax
	movq	%rdx, %rsi
	movl	%eax, %edi
	call	getft
	movq	$20, -1136(%rbp)
	jmp	.L148
.L136:
	leaq	-1040(%rbp), %rdx
	movl	-1172(%rbp), %eax
	movq	%rdx, %rsi
	movl	%eax, %edi
	call	getfn
	movq	$20, -1136(%rbp)
	jmp	.L148
.L139:
	leaq	-1040(%rbp), %rax
	movl	$6, %edx
	leaq	.LC27(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	strncmp@PLT
	movl	%eax, -1168(%rbp)
	movq	$11, -1136(%rbp)
	jmp	.L148
.L126:
	leaq	.LC28(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$20, -1136(%rbp)
	jmp	.L148
.L144:
	leaq	-1040(%rbp), %rdx
	movl	-1172(%rbp), %eax
	movq	%rdx, %rsi
	movl	%eax, %edi
	call	getfz
	movq	$20, -1136(%rbp)
	jmp	.L148
.L128:
	leaq	-1040(%rbp), %rax
	addq	$7, %rax
	leaq	-1072(%rbp), %rdx
	leaq	.LC29(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_sscanf@PLT
	leaq	-1072(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC30(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-1072(%rbp), %rdx
	movl	-1172(%rbp), %eax
	movq	%rdx, %rsi
	movl	%eax, %edi
	call	getfda
	movq	$20, -1136(%rbp)
	jmp	.L148
.L124:
	leaq	.LC31(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	-1172(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	movq	$18, -1136(%rbp)
	jmp	.L148
.L137:
	cmpl	$0, -1168(%rbp)
	jne	.L149
	movq	$21, -1136(%rbp)
	jmp	.L148
.L149:
	movq	$13, -1136(%rbp)
	jmp	.L148
.L138:
	leaq	-1040(%rbp), %rax
	addq	$7, %rax
	leaq	-1104(%rbp), %rdx
	leaq	.LC29(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_sscanf@PLT
	leaq	-1104(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC32(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-1104(%rbp), %rdx
	movl	-1172(%rbp), %eax
	movq	%rdx, %rsi
	movl	%eax, %edi
	call	getfdb
	movq	$20, -1136(%rbp)
	jmp	.L148
.L135:
	leaq	.LC33(%rip), %rax
	movq	%rax, -1120(%rbp)
	movq	-1120(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -1112(%rbp)
	movq	-1112(%rbp), %rdx
	movq	-1120(%rbp), %rsi
	movl	-1172(%rbp), %eax
	movl	$0, %ecx
	movl	%eax, %edi
	call	send@PLT
	movq	$20, -1136(%rbp)
	jmp	.L148
.L130:
	leaq	-1040(%rbp), %rcx
	movl	-1172(%rbp), %eax
	movl	$1023, %edx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	read@PLT
	movq	%rax, -1128(%rbp)
	movq	-1128(%rbp), %rax
	movq	%rax, -1144(%rbp)
	movq	$2, -1136(%rbp)
	jmp	.L148
.L132:
	cmpl	$0, -1152(%rbp)
	jne	.L151
	movq	$3, -1136(%rbp)
	jmp	.L148
.L151:
	movq	$25, -1136(%rbp)
	jmp	.L148
.L141:
	leaq	-1040(%rbp), %rax
	movl	$5, %edx
	leaq	.LC34(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	strncmp@PLT
	movl	%eax, -1160(%rbp)
	movq	$0, -1136(%rbp)
	jmp	.L148
.L123:
	cmpl	$0, -1156(%rbp)
	jne	.L153
	movq	$12, -1136(%rbp)
	jmp	.L148
.L153:
	movq	$6, -1136(%rbp)
	jmp	.L148
.L127:
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
	movq	$28, -1136(%rbp)
	jmp	.L148
.L121:
	cmpl	$0, -1148(%rbp)
	jne	.L155
	movq	$23, -1136(%rbp)
	jmp	.L148
.L155:
	movq	$14, -1136(%rbp)
	jmp	.L148
.L142:
	leaq	-1040(%rbp), %rax
	movl	$6, %edx
	leaq	.LC36(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	strncmp@PLT
	movl	%eax, -1164(%rbp)
	movq	$7, -1136(%rbp)
	jmp	.L148
.L146:
	cmpl	$0, -1160(%rbp)
	jne	.L157
	movq	$15, -1136(%rbp)
	jmp	.L148
.L157:
	movq	$5, -1136(%rbp)
	jmp	.L148
.L140:
	cmpl	$0, -1164(%rbp)
	jne	.L159
	movq	$9, -1136(%rbp)
	jmp	.L148
.L159:
	movq	$8, -1136(%rbp)
	jmp	.L148
.L145:
	cmpq	$0, -1144(%rbp)
	jg	.L161
	movq	$26, -1136(%rbp)
	jmp	.L148
.L161:
	movq	$22, -1136(%rbp)
	jmp	.L148
.L166:
	nop
.L148:
	jmp	.L163
.L167:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L165
	call	__stack_chk_fail@PLT
.L165:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	pclientrequest, .-pclientrequest
	.section	.rodata
.LC37:
	.string	"%llu %llu"
	.align 8
.LC38:
	.string	"Requested size range: %llu - %llu\n"
.LC39:
	.string	"malloc"
.LC40:
	.string	"No file found"
.LC41:
	.string	"Usage: getfz size1 size2"
.LC42:
	.string	"/tmp/file_list.txt"
.LC43:
	.string	"Finished tarring files"
	.text
	.globl	getfz
	.type	getfz, @function
getfz:
.LFB5:
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
	movq	$20, -112(%rbp)
.L216:
	cmpq	$38, -112(%rbp)
	ja	.L219
	movq	-112(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L171(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L171(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L171:
	.long	.L197-.L171
	.long	.L219-.L171
	.long	.L219-.L171
	.long	.L196-.L171
	.long	.L195-.L171
	.long	.L194-.L171
	.long	.L193-.L171
	.long	.L192-.L171
	.long	.L191-.L171
	.long	.L219-.L171
	.long	.L190-.L171
	.long	.L219-.L171
	.long	.L219-.L171
	.long	.L219-.L171
	.long	.L189-.L171
	.long	.L219-.L171
	.long	.L219-.L171
	.long	.L188-.L171
	.long	.L187-.L171
	.long	.L186-.L171
	.long	.L185-.L171
	.long	.L219-.L171
	.long	.L184-.L171
	.long	.L220-.L171
	.long	.L219-.L171
	.long	.L182-.L171
	.long	.L181-.L171
	.long	.L220-.L171
	.long	.L219-.L171
	.long	.L179-.L171
	.long	.L178-.L171
	.long	.L177-.L171
	.long	.L176-.L171
	.long	.L175-.L171
	.long	.L219-.L171
	.long	.L174-.L171
	.long	.L173-.L171
	.long	.L172-.L171
	.long	.L170-.L171
	.text
.L187:
	movq	-128(%rbp), %rax
	movb	$0, (%rax)
	movq	-136(%rbp), %rsi
	movq	-144(%rbp), %rax
	movq	-120(%rbp), %rcx
	leaq	-128(%rbp), %rdx
	movq	%rax, %rdi
	call	findsizerange
	movq	$10, -112(%rbp)
	jmp	.L198
.L182:
	movq	-144(%rbp), %rdx
	movq	-136(%rbp), %rax
	cmpq	%rax, %rdx
	ja	.L199
	movq	$4, -112(%rbp)
	jmp	.L198
.L199:
	movq	$29, -112(%rbp)
	jmp	.L198
.L195:
	movq	-192(%rbp), %rax
	leaq	9(%rax), %rdi
	leaq	-136(%rbp), %rdx
	leaq	-144(%rbp), %rax
	movq	%rdx, %rcx
	movq	%rax, %rdx
	leaq	.LC37(%rip), %rax
	movq	%rax, %rsi
	movl	$0, %eax
	call	__isoc99_sscanf@PLT
	movq	-136(%rbp), %rdx
	movq	-144(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC38(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$1, %edi
	call	malloc@PLT
	movq	%rax, -48(%rbp)
	movq	-48(%rbp), %rax
	movq	%rax, -128(%rbp)
	movq	$36, -112(%rbp)
	jmp	.L198
.L178:
	cmpl	$0, -164(%rbp)
	jle	.L201
	movq	$14, -112(%rbp)
	jmp	.L198
.L201:
	movq	$26, -112(%rbp)
	jmp	.L198
.L189:
	movl	$0, %edi
	call	wait@PLT
	movq	$7, -112(%rbp)
	jmp	.L198
.L177:
	cmpl	$0, -160(%rbp)
	je	.L203
	movq	$32, -112(%rbp)
	jmp	.L198
.L203:
	movq	$5, -112(%rbp)
	jmp	.L198
.L191:
	cmpl	$1, -172(%rbp)
	jle	.L205
	movq	$25, -112(%rbp)
	jmp	.L198
.L205:
	movq	$33, -112(%rbp)
	jmp	.L198
.L196:
	cmpl	$0, -164(%rbp)
	jne	.L208
	movq	$0, -112(%rbp)
	jmp	.L198
.L208:
	movq	$30, -112(%rbp)
	jmp	.L198
.L173:
	movq	-128(%rbp), %rax
	testq	%rax, %rax
	jne	.L210
	movq	$19, -112(%rbp)
	jmp	.L198
.L210:
	movq	$18, -112(%rbp)
	jmp	.L198
.L181:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	-128(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movl	$1, %edi
	call	exit@PLT
.L186:
	leaq	.LC39(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L176:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$5, -112(%rbp)
	jmp	.L198
.L188:
	cmpl	$-1, -168(%rbp)
	jne	.L212
	movq	$22, -112(%rbp)
	jmp	.L198
.L212:
	movq	$6, -112(%rbp)
	jmp	.L198
.L193:
	movq	-128(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -56(%rbp)
	movq	-128(%rbp), %rcx
	movq	-56(%rbp), %rdx
	movl	-168(%rbp), %eax
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	write@PLT
	movl	-168(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	call	fork@PLT
	movl	%eax, -148(%rbp)
	movl	-148(%rbp), %eax
	movl	%eax, -164(%rbp)
	movq	$3, -112(%rbp)
	jmp	.L198
.L170:
	leaq	.LC40(%rip), %rax
	movq	%rax, -104(%rbp)
	movq	-104(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -96(%rbp)
	movq	-96(%rbp), %rax
	leaq	1(%rax), %rdx
	movq	-104(%rbp), %rcx
	movl	-180(%rbp), %eax
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	write@PLT
	movq	-128(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$23, -112(%rbp)
	jmp	.L198
.L184:
	leaq	.LC16(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	-128(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movl	$1, %edi
	call	exit@PLT
.L194:
	movq	-128(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$27, -112(%rbp)
	jmp	.L198
.L175:
	leaq	.LC41(%rip), %rax
	movq	%rax, -80(%rbp)
	movq	-80(%rbp), %rcx
	movl	-180(%rbp), %eax
	movl	$50, %edx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	write@PLT
	movq	$27, -112(%rbp)
	jmp	.L198
.L172:
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
	movq	$17, -112(%rbp)
	jmp	.L198
.L190:
	movq	-128(%rbp), %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	jne	.L214
	movq	$38, -112(%rbp)
	jmp	.L198
.L214:
	movq	$37, -112(%rbp)
	jmp	.L198
.L197:
	movq	-120(%rbp), %rax
	movq	%rax, %rdi
	call	chdir@PLT
	leaq	.LC42(%rip), %rax
	movq	%rax, -40(%rbp)
	movq	-40(%rbp), %rax
	subq	$8, %rsp
	pushq	$0
	movq	%rax, %r9
	leaq	.LC11(%rip), %r8
	leaq	.LC12(%rip), %rax
	movq	%rax, %rcx
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdx
	leaq	.LC14(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	execlp@PLT
	addq	$16, %rsp
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L192:
	leaq	.LC43(%rip), %rax
	movq	%rax, -72(%rbp)
	movq	-72(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -64(%rbp)
	movq	-64(%rbp), %rax
	leaq	1(%rax), %rdx
	movq	-72(%rbp), %rcx
	movl	-180(%rbp), %eax
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	write@PLT
	leaq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	unlink@PLT
	movl	%eax, -160(%rbp)
	movq	$31, -112(%rbp)
	jmp	.L198
.L174:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	getenv@PLT
	movq	%rax, -88(%rbp)
	movq	-88(%rbp), %rax
	movq	%rax, -120(%rbp)
	movq	-192(%rbp), %rax
	leaq	9(%rax), %rdi
	leaq	-136(%rbp), %rdx
	leaq	-144(%rbp), %rax
	movq	%rdx, %rcx
	movq	%rax, %rdx
	leaq	.LC37(%rip), %rax
	movq	%rax, %rsi
	movl	$0, %eax
	call	__isoc99_sscanf@PLT
	movl	%eax, -152(%rbp)
	movl	-152(%rbp), %eax
	movl	%eax, -172(%rbp)
	movq	$8, -112(%rbp)
	jmp	.L198
.L179:
	leaq	.LC41(%rip), %rax
	movq	%rax, -80(%rbp)
	movq	-80(%rbp), %rcx
	movl	-180(%rbp), %eax
	movl	$50, %edx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	write@PLT
	movq	$27, -112(%rbp)
	jmp	.L198
.L185:
	movq	$35, -112(%rbp)
	jmp	.L198
.L219:
	nop
.L198:
	jmp	.L216
.L220:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L218
	call	__stack_chk_fail@PLT
.L218:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	getfz, .-getfz
	.section	.rodata
.LC44:
	.string	"Requested file: %s\n"
	.align 8
.LC45:
	.string	"Searching directory tree rooted at: %s\n"
.LC46:
	.string	"File not found"
	.text
	.globl	getfn
	.type	getfn, @function
getfn:
.LFB8:
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
	movq	$1, -40(%rbp)
.L227:
	cmpq	$2, -40(%rbp)
	je	.L228
	cmpq	$2, -40(%rbp)
	ja	.L229
	cmpq	$0, -40(%rbp)
	je	.L224
	cmpq	$1, -40(%rbp)
	jne	.L229
	movq	$0, -40(%rbp)
	jmp	.L225
.L224:
	movq	-64(%rbp), %rax
	addq	$6, %rax
	movq	%rax, -32(%rbp)
	movq	-32(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC44(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	getenv@PLT
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC45(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-32(%rbp), %rdx
	movq	-16(%rbp), %rcx
	movl	-52(%rbp), %eax
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	searchDirectory
	leaq	.LC46(%rip), %rax
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rcx
	movl	-52(%rbp), %eax
	movl	$100, %edx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	write@PLT
	movq	$2, -40(%rbp)
	jmp	.L225
.L229:
	nop
.L225:
	jmp	.L227
.L228:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	getfn, .-getfn
	.section	.rodata
.LC47:
	.string	"%d"
.LC48:
	.string	"Call model: %s <Port#>\n"
.LC49:
	.string	"Could not create socket\n"
.LC50:
	.string	"inet_ntop"
.LC51:
	.string	"getsockname"
.LC52:
	.string	"Server IP Address: %s\n"
	.align 8
.LC53:
	.string	"\nMore than 4 so sending to mirror"
	.align 8
.LC54:
	.string	"Redirecting client %d to mirror.\n"
.LC55:
	.string	"back from mirror redirection"
.LC56:
	.string	"Number of children is : %d\n"
.LC57:
	.string	"accept"
	.align 8
.LC58:
	.string	"Server listening on port: %d...\n"
.LC59:
	.string	"error forking"
	.text
	.globl	main
	.type	main, @function
main:
.LFB9:
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
	movq	$0, _TIG_IZ_Mhdh_envp(%rip)
	nop
.L231:
	movq	$0, _TIG_IZ_Mhdh_argv(%rip)
	nop
.L232:
	movl	$0, _TIG_IZ_Mhdh_argc(%rip)
	nop
	nop
.L233:
.L234:
#APP
# 492 "sayam56_SocketProject_server.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-Mhdh--0
# 0 "" 2
#NO_APP
	movl	-116(%rbp), %eax
	movl	%eax, _TIG_IZ_Mhdh_argc(%rip)
	movq	-128(%rbp), %rax
	movq	%rax, _TIG_IZ_Mhdh_argv(%rip)
	movq	-136(%rbp), %rax
	movq	%rax, _TIG_IZ_Mhdh_envp(%rip)
	nop
	movq	$20, -72(%rbp)
.L287:
	cmpq	$49, -72(%rbp)
	ja	.L289
	movq	-72(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L237(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L237(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L237:
	.long	.L289-.L237
	.long	.L263-.L237
	.long	.L262-.L237
	.long	.L289-.L237
	.long	.L261-.L237
	.long	.L260-.L237
	.long	.L289-.L237
	.long	.L289-.L237
	.long	.L289-.L237
	.long	.L259-.L237
	.long	.L258-.L237
	.long	.L289-.L237
	.long	.L289-.L237
	.long	.L289-.L237
	.long	.L289-.L237
	.long	.L257-.L237
	.long	.L256-.L237
	.long	.L289-.L237
	.long	.L289-.L237
	.long	.L255-.L237
	.long	.L254-.L237
	.long	.L253-.L237
	.long	.L252-.L237
	.long	.L289-.L237
	.long	.L289-.L237
	.long	.L289-.L237
	.long	.L251-.L237
	.long	.L250-.L237
	.long	.L249-.L237
	.long	.L248-.L237
	.long	.L247-.L237
	.long	.L289-.L237
	.long	.L246-.L237
	.long	.L289-.L237
	.long	.L289-.L237
	.long	.L289-.L237
	.long	.L289-.L237
	.long	.L289-.L237
	.long	.L245-.L237
	.long	.L244-.L237
	.long	.L243-.L237
	.long	.L289-.L237
	.long	.L242-.L237
	.long	.L289-.L237
	.long	.L241-.L237
	.long	.L289-.L237
	.long	.L240-.L237
	.long	.L239-.L237
	.long	.L238-.L237
	.long	.L236-.L237
	.text
.L236:
	cmpq	$0, -80(%rbp)
	jne	.L264
	movq	$9, -72(%rbp)
	jmp	.L266
.L264:
	movq	$27, -72(%rbp)
	jmp	.L266
.L261:
	movw	$2, -64(%rbp)
	movl	$0, %edi
	call	htonl@PLT
	movl	%eax, -60(%rbp)
	movq	-128(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	leaq	-112(%rbp), %rdx
	leaq	.LC47(%rip), %rcx
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
	movq	$19, -72(%rbp)
	jmp	.L266
.L247:
	cmpl	$0, -104(%rbp)
	jns	.L267
	movq	$26, -72(%rbp)
	jmp	.L266
.L267:
	movq	$4, -72(%rbp)
	jmp	.L266
.L257:
	cmpl	$0, -88(%rbp)
	jne	.L269
	movq	$47, -72(%rbp)
	jmp	.L266
.L269:
	movq	$46, -72(%rbp)
	jmp	.L266
.L263:
	cmpl	$4, -92(%rbp)
	jle	.L271
	movq	$5, -72(%rbp)
	jmp	.L266
.L271:
	movq	$42, -72(%rbp)
	jmp	.L266
.L256:
	movl	$0, %edx
	movl	$1, %esi
	movl	$2, %edi
	call	socket@PLT
	movl	%eax, -104(%rbp)
	movq	$30, -72(%rbp)
	jmp	.L266
.L253:
	movq	-128(%rbp), %rax
	movq	(%rax), %rdx
	movq	stderr(%rip), %rax
	leaq	.LC48(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movl	$0, %edi
	call	exit@PLT
.L251:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$24, %edx
	movl	$1, %esi
	leaq	.LC49(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movl	$1, %edi
	call	exit@PLT
.L259:
	leaq	.LC50(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L255:
	cmpl	$-1, -96(%rbp)
	jne	.L273
	movq	$32, -72(%rbp)
	jmp	.L266
.L273:
	movq	$10, -72(%rbp)
	jmp	.L266
.L246:
	leaq	.LC51(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L243:
	call	fork@PLT
	movl	%eax, -84(%rbp)
	movl	-84(%rbp), %eax
	movl	%eax, -88(%rbp)
	movq	$15, -72(%rbp)
	jmp	.L266
.L250:
	leaq	-32(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC52(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-104(%rbp), %eax
	movl	$4, %esi
	movl	%eax, %edi
	call	listen@PLT
	movl	$1, -92(%rbp)
	movq	$39, -72(%rbp)
	jmp	.L266
.L245:
	movl	-92(%rbp), %eax
	andl	$1, %eax
	testl	%eax, %eax
	jne	.L275
	movq	$28, -72(%rbp)
	jmp	.L266
.L275:
	movq	$40, -72(%rbp)
	jmp	.L266
.L238:
	leaq	.LC53(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	-92(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC54(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-100(%rbp), %eax
	movl	%eax, %edi
	call	proxy_to_mirror
	leaq	.LC55(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	-100(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	addl	$1, -92(%rbp)
	movl	-92(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC56(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$39, -72(%rbp)
	jmp	.L266
.L252:
	leaq	.LC57(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$39, -72(%rbp)
	jmp	.L266
.L249:
	movl	-92(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC54(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-100(%rbp), %eax
	movl	%eax, %edi
	call	proxy_to_mirror
	movl	-100(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	addl	$1, -92(%rbp)
	movq	$39, -72(%rbp)
	jmp	.L266
.L239:
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
.L241:
	cmpl	$0, -100(%rbp)
	jns	.L277
	movq	$22, -72(%rbp)
	jmp	.L266
.L277:
	movq	$1, -72(%rbp)
	jmp	.L266
.L260:
	cmpl	$8, -92(%rbp)
	jg	.L279
	movq	$48, -72(%rbp)
	jmp	.L266
.L279:
	movq	$42, -72(%rbp)
	jmp	.L266
.L258:
	leaq	-32(%rbp), %rax
	leaq	-48(%rbp), %rdx
	leaq	4(%rdx), %rsi
	movl	$16, %ecx
	movq	%rax, %rdx
	movl	$2, %edi
	call	inet_ntop@PLT
	movq	%rax, -80(%rbp)
	movq	$49, -72(%rbp)
	jmp	.L266
.L242:
	cmpl	$8, -92(%rbp)
	jle	.L281
	movq	$38, -72(%rbp)
	jmp	.L266
.L281:
	movq	$40, -72(%rbp)
	jmp	.L266
.L240:
	cmpl	$0, -88(%rbp)
	jle	.L283
	movq	$29, -72(%rbp)
	jmp	.L266
.L283:
	movq	$2, -72(%rbp)
	jmp	.L266
.L244:
	movl	-112(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC58(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-104(%rbp), %eax
	movl	$0, %edx
	movl	$0, %esi
	movl	%eax, %edi
	call	accept@PLT
	movl	%eax, -100(%rbp)
	movq	$44, -72(%rbp)
	jmp	.L266
.L248:
	movl	-100(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	addl	$1, -92(%rbp)
	movq	$39, -72(%rbp)
	jmp	.L266
.L262:
	leaq	.LC59(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$39, -72(%rbp)
	jmp	.L266
.L254:
	cmpl	$2, -116(%rbp)
	je	.L285
	movq	$21, -72(%rbp)
	jmp	.L266
.L285:
	movq	$16, -72(%rbp)
	jmp	.L266
.L289:
	nop
.L266:
	jmp	.L287
	.cfi_endproc
.LFE9:
	.size	main, .-main
	.section	.rodata
.LC60:
	.string	"find %s -type f ! -newermt %s"
	.text
	.globl	getfdb
	.type	getfdb, @function
getfdb:
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
	movq	$21, -2192(%rbp)
.L341:
	cmpq	$39, -2192(%rbp)
	ja	.L344
	movq	-2192(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L293(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L293(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L293:
	.long	.L344-.L293
	.long	.L344-.L293
	.long	.L322-.L293
	.long	.L321-.L293
	.long	.L320-.L293
	.long	.L319-.L293
	.long	.L318-.L293
	.long	.L317-.L293
	.long	.L345-.L293
	.long	.L344-.L293
	.long	.L315-.L293
	.long	.L314-.L293
	.long	.L313-.L293
	.long	.L344-.L293
	.long	.L312-.L293
	.long	.L311-.L293
	.long	.L344-.L293
	.long	.L310-.L293
	.long	.L309-.L293
	.long	.L308-.L293
	.long	.L307-.L293
	.long	.L306-.L293
	.long	.L305-.L293
	.long	.L304-.L293
	.long	.L303-.L293
	.long	.L302-.L293
	.long	.L301-.L293
	.long	.L344-.L293
	.long	.L344-.L293
	.long	.L344-.L293
	.long	.L300-.L293
	.long	.L345-.L293
	.long	.L298-.L293
	.long	.L344-.L293
	.long	.L297-.L293
	.long	.L296-.L293
	.long	.L295-.L293
	.long	.L345-.L293
	.long	.L344-.L293
	.long	.L292-.L293
	.text
.L309:
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
	movl	%eax, -2232(%rbp)
	movl	-2232(%rbp), %eax
	movl	%eax, -2244(%rbp)
	movq	$39, -2192(%rbp)
	jmp	.L323
.L302:
	movl	$0, %edi
	call	wait@PLT
	movq	$22, -2192(%rbp)
	jmp	.L323
.L320:
	leaq	-1040(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -2128(%rbp)
	movq	-2128(%rbp), %rdx
	leaq	-1040(%rbp), %rcx
	movl	-2244(%rbp), %eax
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	write@PLT
	movl	-2244(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	call	fork@PLT
	movl	%eax, -2228(%rbp)
	movl	-2228(%rbp), %eax
	movl	%eax, -2240(%rbp)
	movq	$20, -2192(%rbp)
	jmp	.L323
.L300:
	leaq	.LC18(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$18, -2192(%rbp)
	jmp	.L323
.L312:
	cmpq	$0, -2216(%rbp)
	jne	.L324
	movq	$32, -2192(%rbp)
	jmp	.L323
.L324:
	movq	$34, -2192(%rbp)
	jmp	.L323
.L311:
	cmpq	$0, -2200(%rbp)
	je	.L326
	movq	$6, -2192(%rbp)
	jmp	.L323
.L326:
	movq	$3, -2192(%rbp)
	jmp	.L323
.L313:
	leaq	-1040(%rbp), %rdx
	movq	-2208(%rbp), %rax
	addq	%rdx, %rax
	movb	$0, (%rax)
	leaq	-1040(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$18, -2192(%rbp)
	jmp	.L323
.L304:
	movq	-2224(%rbp), %rax
	movq	%rax, %rdi
	call	chdir@PLT
	leaq	-2096(%rbp), %rax
	subq	$8, %rsp
	pushq	$0
	movq	%rax, %r9
	leaq	.LC11(%rip), %r8
	leaq	.LC12(%rip), %rax
	movq	%rax, %rcx
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdx
	leaq	.LC14(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	execlp@PLT
	addq	$16, %rsp
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L321:
	leaq	.LC3(%rip), %rax
	movq	%rax, -2120(%rbp)
	movq	-2120(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -2112(%rbp)
	movq	-2112(%rbp), %rdx
	movq	-2120(%rbp), %rsi
	movl	-2260(%rbp), %eax
	movl	$0, %ecx
	movl	%eax, %edi
	call	send@PLT
	movq	$31, -2192(%rbp)
	jmp	.L323
.L303:
	cmpl	$0, -2240(%rbp)
	jle	.L329
	movq	$25, -2192(%rbp)
	jmp	.L323
.L329:
	movq	$5, -2192(%rbp)
	jmp	.L323
.L306:
	movq	$11, -2192(%rbp)
	jmp	.L323
.L295:
	leaq	.LC17(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$18, -2192(%rbp)
	jmp	.L323
.L301:
	leaq	.LC16(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$8, -2192(%rbp)
	jmp	.L323
.L314:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	getenv@PLT
	movq	%rax, -2176(%rbp)
	movq	-2176(%rbp), %rax
	movq	%rax, -2224(%rbp)
	movq	-2272(%rbp), %rcx
	movq	-2224(%rbp), %rdx
	leaq	-2064(%rbp), %rax
	movq	%rcx, %r8
	movq	%rdx, %rcx
	leaq	.LC60(%rip), %rdx
	movl	$1024, %esi
	movq	%rax, %rdi
	movl	$0, %eax
	call	snprintf@PLT
	leaq	-2064(%rbp), %rax
	leaq	.LC8(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	popen@PLT
	movq	%rax, -2168(%rbp)
	movq	-2168(%rbp), %rax
	movq	%rax, -2216(%rbp)
	movq	$14, -2192(%rbp)
	jmp	.L323
.L308:
	movq	-2216(%rbp), %rax
	movq	%rax, %rdi
	call	feof@PLT
	movl	%eax, -2248(%rbp)
	movq	$35, -2192(%rbp)
	jmp	.L323
.L298:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L310:
	cmpl	$0, -2236(%rbp)
	je	.L331
	movq	$10, -2192(%rbp)
	jmp	.L323
.L331:
	movq	$2, -2192(%rbp)
	jmp	.L323
.L318:
	movq	-2200(%rbp), %rax
	movl	$2, %edx
	movl	$0, %esi
	movq	%rax, %rdi
	call	fseek@PLT
	movq	-2200(%rbp), %rax
	movq	%rax, %rdi
	call	ftell@PLT
	movq	%rax, -2160(%rbp)
	movq	-2160(%rbp), %rax
	movq	%rax, -2152(%rbp)
	movq	-2200(%rbp), %rax
	movq	%rax, %rdi
	call	rewind@PLT
	movq	-2152(%rbp), %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -2144(%rbp)
	movq	-2144(%rbp), %rax
	movq	%rax, -2136(%rbp)
	movq	-2152(%rbp), %rdx
	movq	-2200(%rbp), %rcx
	movq	-2136(%rbp), %rax
	movl	$1, %esi
	movq	%rax, %rdi
	call	fread@PLT
	movq	-2200(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	-2152(%rbp), %rdx
	movq	-2136(%rbp), %rsi
	movl	-2260(%rbp), %eax
	movl	$0, %ecx
	movl	%eax, %edi
	call	send@PLT
	movq	-2136(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$31, -2192(%rbp)
	jmp	.L323
.L297:
	movq	-2216(%rbp), %rdx
	leaq	-1040(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1023, %edx
	movl	$1, %esi
	movq	%rax, %rdi
	call	fread@PLT
	movq	%rax, -2184(%rbp)
	movq	-2184(%rbp), %rax
	movq	%rax, -2208(%rbp)
	movq	$7, -2192(%rbp)
	jmp	.L323
.L305:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-2096(%rbp), %rax
	movq	%rax, %rdi
	call	unlink@PLT
	movl	%eax, -2236(%rbp)
	movq	$17, -2192(%rbp)
	jmp	.L323
.L319:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$37, -2192(%rbp)
	jmp	.L323
.L315:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$2, -2192(%rbp)
	jmp	.L323
.L292:
	cmpl	$-1, -2244(%rbp)
	jne	.L333
	movq	$26, -2192(%rbp)
	jmp	.L323
.L333:
	movq	$4, -2192(%rbp)
	jmp	.L323
.L317:
	cmpq	$0, -2208(%rbp)
	jne	.L335
	movq	$19, -2192(%rbp)
	jmp	.L323
.L335:
	movq	$12, -2192(%rbp)
	jmp	.L323
.L296:
	cmpl	$0, -2248(%rbp)
	je	.L337
	movq	$30, -2192(%rbp)
	jmp	.L323
.L337:
	movq	$36, -2192(%rbp)
	jmp	.L323
.L322:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -2104(%rbp)
	movq	-2104(%rbp), %rax
	movq	%rax, -2200(%rbp)
	movq	$15, -2192(%rbp)
	jmp	.L323
.L307:
	cmpl	$0, -2240(%rbp)
	jne	.L339
	movq	$23, -2192(%rbp)
	jmp	.L323
.L339:
	movq	$24, -2192(%rbp)
	jmp	.L323
.L344:
	nop
.L323:
	jmp	.L341
.L345:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L343
	call	__stack_chk_fail@PLT
.L343:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	getfdb, .-getfdb
	.section	.rodata
	.align 8
.LC61:
	.string	"File: %s\nSize: %ld bytes\nPermissions: %s\nDate Created: %s"
	.text
	.globl	searchDirectory
	.type	searchDirectory, @function
searchDirectory:
.LFB13:
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
	movq	$39, -3304(%rbp)
.L408:
	cmpq	$40, -3304(%rbp)
	ja	.L411
	movq	-3304(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L349(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L349(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L349:
	.long	.L381-.L349
	.long	.L380-.L349
	.long	.L379-.L349
	.long	.L378-.L349
	.long	.L377-.L349
	.long	.L411-.L349
	.long	.L411-.L349
	.long	.L376-.L349
	.long	.L375-.L349
	.long	.L374-.L349
	.long	.L411-.L349
	.long	.L373-.L349
	.long	.L372-.L349
	.long	.L371-.L349
	.long	.L412-.L349
	.long	.L369-.L349
	.long	.L368-.L349
	.long	.L367-.L349
	.long	.L411-.L349
	.long	.L366-.L349
	.long	.L365-.L349
	.long	.L364-.L349
	.long	.L411-.L349
	.long	.L363-.L349
	.long	.L362-.L349
	.long	.L411-.L349
	.long	.L361-.L349
	.long	.L360-.L349
	.long	.L412-.L349
	.long	.L358-.L349
	.long	.L357-.L349
	.long	.L411-.L349
	.long	.L356-.L349
	.long	.L411-.L349
	.long	.L355-.L349
	.long	.L354-.L349
	.long	.L353-.L349
	.long	.L352-.L349
	.long	.L351-.L349
	.long	.L350-.L349
	.long	.L348-.L349
	.text
.L377:
	movb	$119, -3091(%rbp)
	movq	$16, -3304(%rbp)
	jmp	.L382
.L357:
	leaq	.LC20(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$14, -3304(%rbp)
	jmp	.L382
.L369:
	movq	-3312(%rbp), %rax
	leaq	19(%rax), %rdx
	movq	-3368(%rbp), %rax
	movq	%rax, %rsi
	movq	%rdx, %rdi
	call	strcmp@PLT
	movl	%eax, -3328(%rbp)
	movq	$24, -3304(%rbp)
	jmp	.L382
.L372:
	movq	-3320(%rbp), %rax
	movq	%rax, %rdi
	call	readdir@PLT
	movq	%rax, -3312(%rbp)
	movq	$19, -3304(%rbp)
	jmp	.L382
.L375:
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
	leaq	.LC61(%rip), %rdx
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
	movq	$40, -3304(%rbp)
	jmp	.L382
.L380:
	leaq	.LC46(%rip), %rax
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
	movq	$28, -3304(%rbp)
	jmp	.L382
.L363:
	movq	-3360(%rbp), %rax
	movq	%rax, %rdi
	call	opendir@PLT
	movq	%rax, -3296(%rbp)
	movq	-3296(%rbp), %rax
	movq	%rax, -3320(%rbp)
	movq	$29, -3304(%rbp)
	jmp	.L382
.L378:
	movl	-3224(%rbp), %eax
	andl	$256, %eax
	testl	%eax, %eax
	je	.L384
	movq	$32, -3304(%rbp)
	jmp	.L382
.L384:
	movq	$38, -3304(%rbp)
	jmp	.L382
.L368:
	movl	-3224(%rbp), %eax
	andl	$64, %eax
	testl	%eax, %eax
	je	.L386
	movq	$13, -3304(%rbp)
	jmp	.L382
.L386:
	movq	$2, -3304(%rbp)
	jmp	.L382
.L362:
	cmpl	$0, -3328(%rbp)
	jne	.L388
	movq	$3, -3304(%rbp)
	jmp	.L382
.L388:
	movq	$12, -3304(%rbp)
	jmp	.L382
.L364:
	movq	-3368(%rbp), %rdx
	leaq	-3088(%rbp), %rcx
	movl	-3348(%rbp), %eax
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	searchDirectory
	movq	$12, -3304(%rbp)
	jmp	.L382
.L353:
	cmpl	$0, -3340(%rbp)
	jne	.L390
	movq	$1, -3304(%rbp)
	jmp	.L382
.L390:
	movq	$28, -3304(%rbp)
	jmp	.L382
.L361:
	movl	$0, -3340(%rbp)
	movq	$12, -3304(%rbp)
	jmp	.L382
.L373:
	movq	-3312(%rbp), %rax
	addq	$19, %rax
	leaq	.LC22(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -3332(%rbp)
	movq	$37, -3304(%rbp)
	jmp	.L382
.L374:
	movl	-3224(%rbp), %eax
	andl	$128, %eax
	testl	%eax, %eax
	je	.L392
	movq	$4, -3304(%rbp)
	jmp	.L382
.L392:
	movq	$35, -3304(%rbp)
	jmp	.L382
.L371:
	movb	$120, -3090(%rbp)
	movq	$8, -3304(%rbp)
	jmp	.L382
.L366:
	cmpq	$0, -3312(%rbp)
	je	.L394
	movq	$0, -3304(%rbp)
	jmp	.L382
.L394:
	movq	$40, -3304(%rbp)
	jmp	.L382
.L356:
	movb	$114, -3092(%rbp)
	movq	$9, -3304(%rbp)
	jmp	.L382
.L367:
	cmpl	$0, -3324(%rbp)
	jne	.L396
	movq	$7, -3304(%rbp)
	jmp	.L382
.L396:
	movq	$12, -3304(%rbp)
	jmp	.L382
.L348:
	movq	-3320(%rbp), %rax
	movq	%rax, %rdi
	call	closedir@PLT
	movq	$36, -3304(%rbp)
	jmp	.L382
.L360:
	movq	-3312(%rbp), %rax
	leaq	19(%rax), %rcx
	movq	-3360(%rbp), %rdx
	leaq	-3088(%rbp), %rax
	movq	%rcx, %r8
	movq	%rdx, %rcx
	leaq	.LC21(%rip), %rdx
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
	movq	$17, -3304(%rbp)
	jmp	.L382
.L351:
	movb	$45, -3092(%rbp)
	movq	$9, -3304(%rbp)
	jmp	.L382
.L355:
	movl	-3224(%rbp), %eax
	andl	$61440, %eax
	cmpl	$32768, %eax
	jne	.L398
	movq	$15, -3304(%rbp)
	jmp	.L382
.L398:
	movq	$12, -3304(%rbp)
	jmp	.L382
.L352:
	cmpl	$0, -3332(%rbp)
	jne	.L400
	movq	$12, -3304(%rbp)
	jmp	.L382
.L400:
	movq	$27, -3304(%rbp)
	jmp	.L382
.L381:
	movq	-3312(%rbp), %rax
	addq	$19, %rax
	leaq	.LC19(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -3336(%rbp)
	movq	$20, -3304(%rbp)
	jmp	.L382
.L350:
	movq	$23, -3304(%rbp)
	jmp	.L382
.L376:
	movl	-3224(%rbp), %eax
	andl	$61440, %eax
	cmpl	$16384, %eax
	jne	.L402
	movq	$21, -3304(%rbp)
	jmp	.L382
.L402:
	movq	$34, -3304(%rbp)
	jmp	.L382
.L354:
	movb	$45, -3091(%rbp)
	movq	$16, -3304(%rbp)
	jmp	.L382
.L358:
	cmpq	$0, -3320(%rbp)
	jne	.L404
	movq	$30, -3304(%rbp)
	jmp	.L382
.L404:
	movq	$26, -3304(%rbp)
	jmp	.L382
.L379:
	movb	$45, -3090(%rbp)
	movq	$8, -3304(%rbp)
	jmp	.L382
.L365:
	cmpl	$0, -3336(%rbp)
	jne	.L406
	movq	$12, -3304(%rbp)
	jmp	.L382
.L406:
	movq	$11, -3304(%rbp)
	jmp	.L382
.L411:
	nop
.L382:
	jmp	.L408
.L412:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L410
	call	__stack_chk_fail@PLT
.L410:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE13:
	.size	searchDirectory, .-searchDirectory
	.section	.rodata
.LC62:
	.string	"realloc"
	.text
	.globl	appendFilePath
	.type	appendFilePath, @function
appendFilePath:
.LFB15:
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
	movq	$2, -64(%rbp)
.L426:
	cmpq	$7, -64(%rbp)
	ja	.L427
	movq	-64(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L416(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L416(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L416:
	.long	.L427-.L416
	.long	.L421-.L416
	.long	.L420-.L416
	.long	.L419-.L416
	.long	.L418-.L416
	.long	.L427-.L416
	.long	.L428-.L416
	.long	.L415-.L416
	.text
.L418:
	cmpq	$0, -72(%rbp)
	jne	.L422
	movq	$3, -64(%rbp)
	jmp	.L424
.L422:
	movq	$1, -64(%rbp)
	jmp	.L424
.L421:
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
	movq	$6, -64(%rbp)
	jmp	.L424
.L419:
	leaq	.LC62(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L415:
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
	movq	$4, -64(%rbp)
	jmp	.L424
.L420:
	movq	$7, -64(%rbp)
	jmp	.L424
.L427:
	nop
.L424:
	jmp	.L426
.L428:
	nop
	movq	-8(%rbp), %rbx
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE15:
	.size	appendFilePath, .-appendFilePath
	.globl	findsizerange
	.type	findsizerange, @function
findsizerange:
.LFB16:
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
	movq	$9, -432(%rbp)
.L474:
	cmpq	$27, -432(%rbp)
	ja	.L477
	movq	-432(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L432(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L432(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L432:
	.long	.L452-.L432
	.long	.L451-.L432
	.long	.L450-.L432
	.long	.L449-.L432
	.long	.L477-.L432
	.long	.L477-.L432
	.long	.L477-.L432
	.long	.L448-.L432
	.long	.L478-.L432
	.long	.L446-.L432
	.long	.L467-.L432
	.long	.L444-.L432
	.long	.L443-.L432
	.long	.L442-.L432
	.long	.L441-.L432
	.long	.L440-.L432
	.long	.L439-.L432
	.long	.L477-.L432
	.long	.L477-.L432
	.long	.L438-.L432
	.long	.L437-.L432
	.long	.L436-.L432
	.long	.L477-.L432
	.long	.L435-.L432
	.long	.L477-.L432
	.long	.L434-.L432
	.long	.L433-.L432
	.long	.L478-.L432
	.text
.L434:
	movq	-448(%rbp), %rax
	movq	%rax, %rdi
	call	readdir@PLT
	movq	%rax, -440(%rbp)
	movq	$14, -432(%rbp)
	jmp	.L453
.L441:
	cmpq	$0, -440(%rbp)
	je	.L454
	movq	$26, -432(%rbp)
	jmp	.L453
.L454:
	movq	$2, -432(%rbp)
	jmp	.L453
.L440:
	movq	-496(%rbp), %rax
	movq	%rax, %rdi
	call	opendir@PLT
	movq	%rax, -424(%rbp)
	movq	-424(%rbp), %rax
	movq	%rax, -448(%rbp)
	movq	$19, -432(%rbp)
	jmp	.L453
.L443:
	movq	-440(%rbp), %rax
	addq	$19, %rax
	leaq	.LC22(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -456(%rbp)
	movq	$20, -432(%rbp)
	jmp	.L453
.L451:
	movq	-368(%rbp), %rax
	cmpq	%rax, -480(%rbp)
	jb	.L457
	movq	$0, -432(%rbp)
	jmp	.L453
.L457:
	movq	$10, -432(%rbp)
	jmp	.L453
.L435:
	leaq	-272(%rbp), %rcx
	movq	-488(%rbp), %rdx
	movq	-480(%rbp), %rsi
	movq	-472(%rbp), %rax
	movq	%rax, %rdi
	call	findsizerange
	movq	$25, -432(%rbp)
	jmp	.L453
.L449:
	leaq	.LC20(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$8, -432(%rbp)
	jmp	.L453
.L439:
	movq	-440(%rbp), %rax
	leaq	19(%rax), %rcx
	movq	-496(%rbp), %rdx
	leaq	-272(%rbp), %rax
	movq	%rcx, %r8
	movq	%rdx, %rcx
	leaq	.LC21(%rip), %rdx
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
	movq	$7, -432(%rbp)
	jmp	.L453
.L436:
	movl	-392(%rbp), %eax
	andl	$61440, %eax
	cmpl	$32768, %eax
	jne	.L459
	movq	$13, -432(%rbp)
	jmp	.L453
.L459:
	movq	$10, -432(%rbp)
	jmp	.L453
.L433:
	movq	-440(%rbp), %rax
	addq	$19, %rax
	leaq	.LC19(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -460(%rbp)
	movq	$11, -432(%rbp)
	jmp	.L453
.L444:
	cmpl	$0, -460(%rbp)
	jne	.L461
	movq	$25, -432(%rbp)
	jmp	.L453
.L461:
	movq	$12, -432(%rbp)
	jmp	.L453
.L446:
	movq	$15, -432(%rbp)
	jmp	.L453
.L442:
	movq	-368(%rbp), %rax
	cmpq	%rax, -472(%rbp)
	ja	.L463
	movq	$1, -432(%rbp)
	jmp	.L453
.L463:
	movq	$10, -432(%rbp)
	jmp	.L453
.L438:
	cmpq	$0, -448(%rbp)
	jne	.L465
	movq	$3, -432(%rbp)
	jmp	.L453
.L465:
	movq	$25, -432(%rbp)
	jmp	.L453
.L445:
.L467:
	movl	-392(%rbp), %eax
	andl	$61440, %eax
	cmpl	$16384, %eax
	jne	.L468
	movq	$23, -432(%rbp)
	jmp	.L453
.L468:
	movq	$25, -432(%rbp)
	jmp	.L453
.L452:
	leaq	-272(%rbp), %rdx
	movq	-488(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	appendFilePath
	movq	$25, -432(%rbp)
	jmp	.L453
.L448:
	cmpl	$0, -452(%rbp)
	jne	.L470
	movq	$21, -432(%rbp)
	jmp	.L453
.L470:
	movq	$25, -432(%rbp)
	jmp	.L453
.L450:
	movq	-448(%rbp), %rax
	movq	%rax, %rdi
	call	closedir@PLT
	movq	$27, -432(%rbp)
	jmp	.L453
.L437:
	cmpl	$0, -456(%rbp)
	jne	.L472
	movq	$25, -432(%rbp)
	jmp	.L453
.L472:
	movq	$16, -432(%rbp)
	jmp	.L453
.L477:
	nop
.L453:
	jmp	.L474
.L478:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L476
	call	__stack_chk_fail@PLT
.L476:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE16:
	.size	findsizerange, .-findsizerange
	.section	.rodata
.LC63:
	.string	" "
	.text
	.globl	countExtensions
	.type	countExtensions, @function
countExtensions:
.LFB17:
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
	movq	$4, -288(%rbp)
.L491:
	cmpq	$7, -288(%rbp)
	ja	.L494
	movq	-288(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L482(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L482(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L482:
	.long	.L494-.L482
	.long	.L494-.L482
	.long	.L486-.L482
	.long	.L485-.L482
	.long	.L484-.L482
	.long	.L483-.L482
	.long	.L494-.L482
	.long	.L481-.L482
	.text
.L484:
	movq	$5, -288(%rbp)
	jmp	.L487
.L485:
	addl	$1, -300(%rbp)
	leaq	.LC63(%rip), %rax
	movq	%rax, %rsi
	movl	$0, %edi
	call	strtok@PLT
	movq	%rax, -296(%rbp)
	movq	$2, -288(%rbp)
	jmp	.L487
.L483:
	movq	-312(%rbp), %rdx
	leaq	-272(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcpy@PLT
	leaq	-272(%rbp), %rax
	leaq	.LC63(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strtok@PLT
	movq	%rax, -280(%rbp)
	movq	-280(%rbp), %rax
	movq	%rax, -296(%rbp)
	movl	$0, -300(%rbp)
	movq	$2, -288(%rbp)
	jmp	.L487
.L481:
	movl	-300(%rbp), %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L492
	jmp	.L493
.L486:
	cmpq	$0, -296(%rbp)
	je	.L489
	movq	$3, -288(%rbp)
	jmp	.L487
.L489:
	movq	$7, -288(%rbp)
	jmp	.L487
.L494:
	nop
.L487:
	jmp	.L491
.L493:
	call	__stack_chk_fail@PLT
.L492:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE17:
	.size	countExtensions, .-countExtensions
	.section	.rodata
.LC64:
	.string	"Select: "
	.align 8
.LC65:
	.string	"Socket creation error for mirror connection"
	.align 8
.LC66:
	.string	"Connection to proxy mirror completed"
.LC67:
	.string	"inside proxy method"
.LC68:
	.string	"Mirror FD created"
.LC69:
	.string	"0.0.0.0"
.LC70:
	.string	"Connection to mirror failed"
	.align 8
.LC71:
	.string	"Invalid mirror address / Address not supported"
	.text
	.globl	proxy_to_mirror
	.type	proxy_to_mirror, @function
proxy_to_mirror:
.LFB18:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$1280, %rsp
	movl	%edi, -1268(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$1, -1208(%rbp)
.L554:
	cmpq	$48, -1208(%rbp)
	ja	.L557
	movq	-1208(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L498(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L498(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L498:
	.long	.L557-.L498
	.long	.L531-.L498
	.long	.L557-.L498
	.long	.L557-.L498
	.long	.L557-.L498
	.long	.L530-.L498
	.long	.L529-.L498
	.long	.L528-.L498
	.long	.L557-.L498
	.long	.L527-.L498
	.long	.L526-.L498
	.long	.L557-.L498
	.long	.L525-.L498
	.long	.L524-.L498
	.long	.L558-.L498
	.long	.L522-.L498
	.long	.L521-.L498
	.long	.L557-.L498
	.long	.L520-.L498
	.long	.L519-.L498
	.long	.L518-.L498
	.long	.L558-.L498
	.long	.L516-.L498
	.long	.L558-.L498
	.long	.L514-.L498
	.long	.L558-.L498
	.long	.L512-.L498
	.long	.L557-.L498
	.long	.L557-.L498
	.long	.L557-.L498
	.long	.L557-.L498
	.long	.L557-.L498
	.long	.L511-.L498
	.long	.L557-.L498
	.long	.L510-.L498
	.long	.L509-.L498
	.long	.L508-.L498
	.long	.L557-.L498
	.long	.L507-.L498
	.long	.L506-.L498
	.long	.L505-.L498
	.long	.L504-.L498
	.long	.L503-.L498
	.long	.L502-.L498
	.long	.L501-.L498
	.long	.L500-.L498
	.long	.L557-.L498
	.long	.L499-.L498
	.long	.L497-.L498
	.text
.L520:
	movl	-1268(%rbp), %eax
	cmpl	-1252(%rbp), %eax
	jle	.L532
	movq	$34, -1208(%rbp)
	jmp	.L534
.L532:
	movq	$16, -1208(%rbp)
	jmp	.L534
.L522:
	cmpl	$0, -1228(%rbp)
	jg	.L536
	movq	$43, -1208(%rbp)
	jmp	.L534
.L536:
	movq	$36, -1208(%rbp)
	jmp	.L534
.L525:
	leaq	-1184(%rbp), %rcx
	movl	-1252(%rbp), %eax
	movl	$16, %edx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	connect@PLT
	movl	%eax, -1244(%rbp)
	movq	$45, -1208(%rbp)
	jmp	.L534
.L500:
	cmpl	$0, -1244(%rbp)
	jns	.L538
	movq	$7, -1208(%rbp)
	jmp	.L534
.L538:
	movq	$38, -1208(%rbp)
	jmp	.L534
.L531:
	movq	$44, -1208(%rbp)
	jmp	.L534
.L521:
	movl	-1252(%rbp), %eax
	movl	%eax, -1236(%rbp)
	movq	$35, -1208(%rbp)
	jmp	.L534
.L514:
	cmpl	$0, -1224(%rbp)
	jg	.L540
	movq	$43, -1208(%rbp)
	jmp	.L534
.L540:
	movq	$10, -1208(%rbp)
	jmp	.L534
.L508:
	movl	-1228(%rbp), %eax
	movslq	%eax, %rdx
	leaq	-1040(%rbp), %rsi
	movl	-1252(%rbp), %eax
	movl	$0, %ecx
	movl	%eax, %edi
	call	send@PLT
	movq	$42, -1208(%rbp)
	jmp	.L534
.L512:
	leaq	.LC64(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$43, -1208(%rbp)
	jmp	.L534
.L527:
	leaq	-1040(%rbp), %rsi
	movl	-1252(%rbp), %eax
	movl	$0, %ecx
	movl	$1024, %edx
	movl	%eax, %edi
	call	recv@PLT
	movq	%rax, -1192(%rbp)
	movq	-1192(%rbp), %rax
	movl	%eax, -1224(%rbp)
	movq	$24, -1208(%rbp)
	jmp	.L534
.L524:
	leaq	-1040(%rbp), %rsi
	movl	-1268(%rbp), %eax
	movl	$0, %ecx
	movl	$1024, %edx
	movl	%eax, %edi
	call	recv@PLT
	movq	%rax, -1200(%rbp)
	movq	-1200(%rbp), %rax
	movl	%eax, -1228(%rbp)
	movq	$15, -1208(%rbp)
	jmp	.L534
.L519:
	cmpl	$15, -1240(%rbp)
	ja	.L542
	movq	$6, -1208(%rbp)
	jmp	.L534
.L542:
	movq	$47, -1208(%rbp)
	jmp	.L534
.L511:
	leaq	-1168(%rbp), %rax
	movq	%rax, -1216(%rbp)
	movl	$0, -1240(%rbp)
	movq	$19, -1208(%rbp)
	jmp	.L534
.L505:
	leaq	.LC65(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$21, -1208(%rbp)
	jmp	.L534
.L529:
	movq	-1216(%rbp), %rax
	movl	-1240(%rbp), %edx
	movq	$0, (%rax,%rdx,8)
	addl	$1, -1240(%rbp)
	movq	$19, -1208(%rbp)
	jmp	.L534
.L507:
	leaq	.LC66(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$32, -1208(%rbp)
	jmp	.L534
.L510:
	movl	-1268(%rbp), %eax
	movl	%eax, -1236(%rbp)
	movq	$35, -1208(%rbp)
	jmp	.L534
.L497:
	cmpl	$0, -1232(%rbp)
	jns	.L544
	movq	$26, -1208(%rbp)
	jmp	.L534
.L544:
	movq	$22, -1208(%rbp)
	jmp	.L534
.L516:
	movl	-1268(%rbp), %eax
	leal	63(%rax), %edx
	testl	%eax, %eax
	cmovs	%edx, %eax
	sarl	$6, %eax
	cltq
	movq	-1168(%rbp,%rax,8), %rdx
	movl	-1268(%rbp), %eax
	andl	$63, %eax
	movl	$1, %esi
	movl	%eax, %ecx
	salq	%cl, %rsi
	movq	%rsi, %rax
	andq	%rdx, %rax
	testq	%rax, %rax
	je	.L546
	movq	$13, -1208(%rbp)
	jmp	.L534
.L546:
	movq	$42, -1208(%rbp)
	jmp	.L534
.L499:
	movl	-1268(%rbp), %eax
	leal	63(%rax), %edx
	testl	%eax, %eax
	cmovs	%edx, %eax
	sarl	$6, %eax
	movl	%eax, %esi
	movslq	%esi, %rax
	movq	-1168(%rbp,%rax,8), %rdx
	movl	-1268(%rbp), %eax
	andl	$63, %eax
	movl	$1, %edi
	movl	%eax, %ecx
	salq	%cl, %rdi
	movq	%rdi, %rax
	orq	%rax, %rdx
	movslq	%esi, %rax
	movq	%rdx, -1168(%rbp,%rax,8)
	movl	-1252(%rbp), %eax
	leal	63(%rax), %edx
	testl	%eax, %eax
	cmovs	%edx, %eax
	sarl	$6, %eax
	movl	%eax, %esi
	movslq	%esi, %rax
	movq	-1168(%rbp,%rax,8), %rdx
	movl	-1252(%rbp), %eax
	andl	$63, %eax
	movl	$1, %edi
	movl	%eax, %ecx
	salq	%cl, %rdi
	movq	%rdi, %rax
	orq	%rax, %rdx
	movslq	%esi, %rax
	movq	%rdx, -1168(%rbp,%rax,8)
	movq	$18, -1208(%rbp)
	jmp	.L534
.L501:
	leaq	.LC67(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$0, %edx
	movl	$1, %esi
	movl	$2, %edi
	call	socket@PLT
	movl	%eax, -1252(%rbp)
	movq	$5, -1208(%rbp)
	jmp	.L534
.L530:
	cmpl	$0, -1252(%rbp)
	jns	.L548
	movq	$40, -1208(%rbp)
	jmp	.L534
.L548:
	movq	$39, -1208(%rbp)
	jmp	.L534
.L504:
	cmpl	$0, -1248(%rbp)
	jg	.L550
	movq	$20, -1208(%rbp)
	jmp	.L534
.L550:
	movq	$12, -1208(%rbp)
	jmp	.L534
.L526:
	movl	-1224(%rbp), %eax
	movslq	%eax, %rdx
	leaq	-1040(%rbp), %rsi
	movl	-1268(%rbp), %eax
	movl	$0, %ecx
	movl	%eax, %edi
	call	send@PLT
	movq	$32, -1208(%rbp)
	jmp	.L534
.L503:
	movl	-1252(%rbp), %eax
	leal	63(%rax), %edx
	testl	%eax, %eax
	cmovs	%edx, %eax
	sarl	$6, %eax
	cltq
	movq	-1168(%rbp,%rax,8), %rdx
	movl	-1252(%rbp), %eax
	andl	$63, %eax
	movl	$1, %esi
	movl	%eax, %ecx
	salq	%cl, %rsi
	movq	%rsi, %rax
	andq	%rdx, %rax
	testq	%rax, %rax
	je	.L552
	movq	$9, -1208(%rbp)
	jmp	.L534
.L552:
	movq	$32, -1208(%rbp)
	jmp	.L534
.L506:
	leaq	.LC68(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movw	$2, -1184(%rbp)
	movl	$9004, %edi
	call	htons@PLT
	movw	%ax, -1182(%rbp)
	leaq	-1184(%rbp), %rax
	addq	$4, %rax
	movq	%rax, %rdx
	leaq	.LC69(%rip), %rax
	movq	%rax, %rsi
	movl	$2, %edi
	call	inet_pton@PLT
	movl	%eax, -1248(%rbp)
	movq	$41, -1208(%rbp)
	jmp	.L534
.L528:
	leaq	.LC70(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$23, -1208(%rbp)
	jmp	.L534
.L509:
	movl	-1236(%rbp), %eax
	movl	%eax, -1220(%rbp)
	movl	-1220(%rbp), %eax
	leal	1(%rax), %edi
	leaq	-1168(%rbp), %rax
	movl	$0, %r8d
	movl	$0, %ecx
	movl	$0, %edx
	movq	%rax, %rsi
	call	select@PLT
	movl	%eax, -1232(%rbp)
	movq	$48, -1208(%rbp)
	jmp	.L534
.L502:
	movl	-1252(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	movq	$25, -1208(%rbp)
	jmp	.L534
.L518:
	leaq	.LC71(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$14, -1208(%rbp)
	jmp	.L534
.L557:
	nop
.L534:
	jmp	.L554
.L558:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L556
	call	__stack_chk_fail@PLT
.L556:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE18:
	.size	proxy_to_mirror, .-proxy_to_mirror
	.section	.rodata
.LC72:
	.string	"w"
.LC73:
	.string	"file_list.txt"
	.align 8
.LC74:
	.string	"tar -czf f23project/temp.tar.gz -T file_list.txt"
	.align 8
.LC75:
	.string	"Usage: getft <extension list> minimum 1 or maximum 3 extensions are allowed"
.LC76:
	.string	"fopen"
.LC77:
	.string	"Finished tarring files."
	.text
	.globl	getft
	.type	getft, @function
getft:
.LFB19:
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
	movq	$30, -1176(%rbp)
.L596:
	cmpq	$30, -1176(%rbp)
	ja	.L599
	movq	-1176(%rbp), %rax
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
	.long	.L583-.L562
	.long	.L600-.L562
	.long	.L600-.L562
	.long	.L600-.L562
	.long	.L599-.L562
	.long	.L599-.L562
	.long	.L579-.L562
	.long	.L578-.L562
	.long	.L599-.L562
	.long	.L577-.L562
	.long	.L599-.L562
	.long	.L600-.L562
	.long	.L599-.L562
	.long	.L575-.L562
	.long	.L599-.L562
	.long	.L574-.L562
	.long	.L573-.L562
	.long	.L572-.L562
	.long	.L571-.L562
	.long	.L599-.L562
	.long	.L570-.L562
	.long	.L569-.L562
	.long	.L599-.L562
	.long	.L600-.L562
	.long	.L567-.L562
	.long	.L566-.L562
	.long	.L565-.L562
	.long	.L564-.L562
	.long	.L563-.L562
	.long	.L599-.L562
	.long	.L561-.L562
	.text
.L571:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	getenv@PLT
	movq	%rax, -1144(%rbp)
	movq	-1144(%rbp), %rax
	movq	%rax, -1208(%rbp)
	leaq	.LC72(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC73(%rip), %rax
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -1136(%rbp)
	movq	-1136(%rbp), %rax
	movq	%rax, -1200(%rbp)
	movq	$13, -1176(%rbp)
	jmp	.L584
.L566:
	cmpq	$0, -1184(%rbp)
	je	.L585
	movq	$0, -1176(%rbp)
	jmp	.L584
.L585:
	movq	$16, -1176(%rbp)
	jmp	.L584
.L561:
	movq	$6, -1176(%rbp)
	jmp	.L584
.L574:
	leaq	.LC40(%rip), %rax
	movq	%rax, -1112(%rbp)
	movq	-1112(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -1104(%rbp)
	movq	-1104(%rbp), %rdx
	movq	-1112(%rbp), %rsi
	movl	-1236(%rbp), %eax
	movl	$0, %ecx
	movl	%eax, %edi
	call	send@PLT
	movq	$11, -1176(%rbp)
	jmp	.L584
.L573:
	leaq	.LC3(%rip), %rax
	movq	%rax, -1128(%rbp)
	movq	-1128(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -1120(%rbp)
	movq	-1120(%rbp), %rdx
	movq	-1128(%rbp), %rsi
	movl	-1236(%rbp), %eax
	movl	$0, %ecx
	movl	%eax, %edi
	call	send@PLT
	movq	$7, -1176(%rbp)
	jmp	.L584
.L567:
	cmpl	$3, -1224(%rbp)
	jle	.L588
	movq	$27, -1176(%rbp)
	jmp	.L584
.L588:
	movq	$18, -1176(%rbp)
	jmp	.L584
.L569:
	movq	-1216(%rbp), %rdx
	movq	-1208(%rbp), %rcx
	movq	-1200(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	filesearchWrite
	movq	-1200(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	leaq	.LC8(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC73(%rip), %rax
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -1152(%rbp)
	movq	-1152(%rbp), %rax
	movq	%rax, -1192(%rbp)
	movq	$17, -1176(%rbp)
	jmp	.L584
.L565:
	cmpl	$0, -1224(%rbp)
	jne	.L590
	movq	$20, -1176(%rbp)
	jmp	.L584
.L590:
	movq	$24, -1176(%rbp)
	jmp	.L584
.L577:
	movq	-1192(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	leaq	-1040(%rbp), %rax
	leaq	.LC74(%rip), %rdx
	movl	$1024, %esi
	movq	%rax, %rdi
	movl	$0, %eax
	call	snprintf@PLT
	leaq	-1040(%rbp), %rax
	movq	%rax, %rdi
	call	system@PLT
	leaq	.LC9(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -1096(%rbp)
	movq	-1096(%rbp), %rax
	movq	%rax, -1184(%rbp)
	movq	$25, -1176(%rbp)
	jmp	.L584
.L575:
	cmpq	$0, -1200(%rbp)
	jne	.L592
	movq	$28, -1176(%rbp)
	jmp	.L584
.L592:
	movq	$21, -1176(%rbp)
	jmp	.L584
.L572:
	cmpq	$0, -1192(%rbp)
	jne	.L594
	movq	$15, -1176(%rbp)
	jmp	.L584
.L594:
	movq	$9, -1176(%rbp)
	jmp	.L584
.L579:
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
	movq	$26, -1176(%rbp)
	jmp	.L584
.L564:
	leaq	.LC75(%rip), %rax
	movq	%rax, -1168(%rbp)
	movq	-1168(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -1160(%rbp)
	movq	-1160(%rbp), %rdx
	movq	-1168(%rbp), %rsi
	movl	-1236(%rbp), %eax
	movl	$0, %ecx
	movl	%eax, %edi
	call	send@PLT
	movq	$2, -1176(%rbp)
	jmp	.L584
.L563:
	leaq	.LC76(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$3, -1176(%rbp)
	jmp	.L584
.L583:
	movq	-1184(%rbp), %rax
	movl	$2, %edx
	movl	$0, %esi
	movq	%rax, %rdi
	call	fseek@PLT
	movq	-1184(%rbp), %rax
	movq	%rax, %rdi
	call	ftell@PLT
	movq	%rax, -1072(%rbp)
	movq	-1072(%rbp), %rax
	movq	%rax, -1064(%rbp)
	movq	-1184(%rbp), %rax
	movq	%rax, %rdi
	call	rewind@PLT
	movq	-1064(%rbp), %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -1056(%rbp)
	movq	-1056(%rbp), %rax
	movq	%rax, -1048(%rbp)
	movq	-1064(%rbp), %rdx
	movq	-1184(%rbp), %rcx
	movq	-1048(%rbp), %rax
	movl	$1, %esi
	movq	%rax, %rdi
	call	fread@PLT
	movq	-1184(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	-1064(%rbp), %rdx
	movq	-1048(%rbp), %rsi
	movl	-1236(%rbp), %eax
	movl	$0, %ecx
	movl	%eax, %edi
	call	send@PLT
	movq	-1048(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$7, -1176(%rbp)
	jmp	.L584
.L578:
	leaq	.LC77(%rip), %rax
	movq	%rax, -1088(%rbp)
	movq	-1088(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -1080(%rbp)
	movq	-1080(%rbp), %rdx
	movq	-1088(%rbp), %rcx
	movl	-1236(%rbp), %eax
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	write@PLT
	leaq	.LC73(%rip), %rax
	movq	%rax, %rdi
	call	remove@PLT
	movq	$1, -1176(%rbp)
	jmp	.L584
.L570:
	leaq	.LC75(%rip), %rax
	movq	%rax, -1168(%rbp)
	movq	-1168(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -1160(%rbp)
	movq	-1160(%rbp), %rdx
	movq	-1168(%rbp), %rsi
	movl	-1236(%rbp), %eax
	movl	$0, %ecx
	movl	%eax, %edi
	call	send@PLT
	movq	$23, -1176(%rbp)
	jmp	.L584
.L599:
	nop
.L584:
	jmp	.L596
.L600:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L598
	call	__stack_chk_fail@PLT
.L598:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE19:
	.size	getft, .-getft
	.section	.rodata
.LC78:
	.string	"stat"
.LC79:
	.string	"%s\n"
	.text
	.globl	filesearchWrite
	.type	filesearchWrite, @function
filesearchWrite:
.LFB20:
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
	movq	$6, -1208(%rbp)
.L648:
	cmpq	$30, -1208(%rbp)
	ja	.L651
	movq	-1208(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L604(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L604(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L604:
	.long	.L651-.L604
	.long	.L627-.L604
	.long	.L651-.L604
	.long	.L652-.L604
	.long	.L651-.L604
	.long	.L652-.L604
	.long	.L624-.L604
	.long	.L623-.L604
	.long	.L651-.L604
	.long	.L622-.L604
	.long	.L621-.L604
	.long	.L651-.L604
	.long	.L620-.L604
	.long	.L619-.L604
	.long	.L618-.L604
	.long	.L617-.L604
	.long	.L616-.L604
	.long	.L615-.L604
	.long	.L614-.L604
	.long	.L613-.L604
	.long	.L612-.L604
	.long	.L611-.L604
	.long	.L610-.L604
	.long	.L609-.L604
	.long	.L608-.L604
	.long	.L607-.L604
	.long	.L651-.L604
	.long	.L651-.L604
	.long	.L606-.L604
	.long	.L605-.L604
	.long	.L603-.L604
	.text
.L614:
	movq	-1280(%rbp), %rax
	movq	%rax, %rdi
	call	opendir@PLT
	movq	%rax, -1200(%rbp)
	movq	-1200(%rbp), %rax
	movq	%rax, -1240(%rbp)
	movq	$21, -1208(%rbp)
	jmp	.L628
.L607:
	movl	-1160(%rbp), %eax
	andl	$61440, %eax
	cmpl	$16384, %eax
	jne	.L629
	movq	$17, -1208(%rbp)
	jmp	.L628
.L629:
	movq	$24, -1208(%rbp)
	jmp	.L628
.L603:
	cmpl	$0, -1244(%rbp)
	je	.L631
	movq	$16, -1208(%rbp)
	jmp	.L628
.L631:
	movq	$25, -1208(%rbp)
	jmp	.L628
.L618:
	cmpq	$0, -1224(%rbp)
	je	.L633
	movq	$23, -1208(%rbp)
	jmp	.L628
.L633:
	movq	$20, -1208(%rbp)
	jmp	.L628
.L617:
	cmpl	$0, -1248(%rbp)
	jne	.L635
	movq	$20, -1208(%rbp)
	jmp	.L628
.L635:
	movq	$7, -1208(%rbp)
	jmp	.L628
.L620:
	cmpq	$0, -1216(%rbp)
	je	.L637
	movq	$10, -1208(%rbp)
	jmp	.L628
.L637:
	movq	$20, -1208(%rbp)
	jmp	.L628
.L627:
	leaq	.LC20(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$5, -1208(%rbp)
	jmp	.L628
.L609:
	movq	-1224(%rbp), %rdx
	movq	-1288(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strstr@PLT
	movq	%rax, -1216(%rbp)
	movq	$12, -1208(%rbp)
	jmp	.L628
.L616:
	leaq	.LC78(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$20, -1208(%rbp)
	jmp	.L628
.L608:
	movl	-1160(%rbp), %eax
	andl	$61440, %eax
	cmpl	$32768, %eax
	jne	.L640
	movq	$13, -1208(%rbp)
	jmp	.L628
.L640:
	movq	$20, -1208(%rbp)
	jmp	.L628
.L611:
	cmpq	$0, -1240(%rbp)
	jne	.L642
	movq	$1, -1208(%rbp)
	jmp	.L628
.L642:
	movq	$20, -1208(%rbp)
	jmp	.L628
.L622:
	cmpq	$0, -1232(%rbp)
	je	.L644
	movq	$28, -1208(%rbp)
	jmp	.L628
.L644:
	movq	$29, -1208(%rbp)
	jmp	.L628
.L619:
	movq	-1232(%rbp), %rax
	addq	$19, %rax
	movl	$46, %esi
	movq	%rax, %rdi
	call	strrchr@PLT
	movq	%rax, -1192(%rbp)
	movq	-1192(%rbp), %rax
	movq	%rax, -1224(%rbp)
	movq	$14, -1208(%rbp)
	jmp	.L628
.L613:
	cmpl	$0, -1252(%rbp)
	jne	.L646
	movq	$20, -1208(%rbp)
	jmp	.L628
.L646:
	movq	$22, -1208(%rbp)
	jmp	.L628
.L615:
	movq	-1288(%rbp), %rdx
	leaq	-1040(%rbp), %rcx
	movq	-1272(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	filesearchWrite
	movq	$20, -1208(%rbp)
	jmp	.L628
.L624:
	movq	$18, -1208(%rbp)
	jmp	.L628
.L610:
	movq	-1232(%rbp), %rax
	addq	$19, %rax
	leaq	.LC22(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -1248(%rbp)
	movq	$15, -1208(%rbp)
	jmp	.L628
.L606:
	movq	-1232(%rbp), %rax
	addq	$19, %rax
	leaq	.LC19(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -1252(%rbp)
	movq	$19, -1208(%rbp)
	jmp	.L628
.L621:
	leaq	-1040(%rbp), %rdx
	movq	-1272(%rbp), %rax
	leaq	.LC79(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	$20, -1208(%rbp)
	jmp	.L628
.L623:
	movq	-1232(%rbp), %rax
	leaq	19(%rax), %rcx
	movq	-1280(%rbp), %rdx
	leaq	-1040(%rbp), %rax
	movq	%rcx, %r8
	movq	%rdx, %rcx
	leaq	.LC21(%rip), %rdx
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
	movq	$30, -1208(%rbp)
	jmp	.L628
.L605:
	movq	-1240(%rbp), %rax
	movq	%rax, %rdi
	call	closedir@PLT
	movq	$3, -1208(%rbp)
	jmp	.L628
.L612:
	movq	-1240(%rbp), %rax
	movq	%rax, %rdi
	call	readdir@PLT
	movq	%rax, -1232(%rbp)
	movq	$9, -1208(%rbp)
	jmp	.L628
.L651:
	nop
.L628:
	jmp	.L648
.L652:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L650
	call	__stack_chk_fail@PLT
.L650:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE20:
	.size	filesearchWrite, .-filesearchWrite
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
