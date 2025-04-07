	.file	"infval_StrikeLZSS_main_flatten.c"
	.text
	.globl	_TIG_IZ_aTfC_envp
	.bss
	.align 8
	.type	_TIG_IZ_aTfC_envp, @object
	.size	_TIG_IZ_aTfC_envp, 8
_TIG_IZ_aTfC_envp:
	.zero	8
	.globl	_TIG_IZ_aTfC_argv
	.align 8
	.type	_TIG_IZ_aTfC_argv, @object
	.size	_TIG_IZ_aTfC_argv, 8
_TIG_IZ_aTfC_argv:
	.zero	8
	.globl	window
	.align 32
	.type	window, @object
	.size	window, 2048
window:
	.zero	2048
	.globl	arguments
	.align 32
	.type	arguments, @object
	.size	arguments, 40
arguments:
	.zero	40
	.globl	_TIG_IZ_aTfC_argc
	.align 4
	.type	_TIG_IZ_aTfC_argc, @object
	.size	_TIG_IZ_aTfC_argc, 4
_TIG_IZ_aTfC_argc:
	.zero	4
	.text
	.globl	LZSS_GetCompressedMaxSize
	.type	LZSS_GetCompressedMaxSize, @function
LZSS_GetCompressedMaxSize:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movq	$0, -8(%rbp)
.L4:
	cmpq	$0, -8(%rbp)
	jne	.L7
	movq	-24(%rbp), %rax
	shrq	$3, %rax
	movq	%rax, %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	addq	$1, %rax
	jmp	.L6
.L7:
	nop
	jmp	.L4
.L6:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	LZSS_GetCompressedMaxSize, .-LZSS_GetCompressedMaxSize
	.globl	Write_u32be
	.type	Write_u32be, @function
Write_u32be:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -20(%rbp)
	movq	%rsi, -32(%rbp)
	movq	$2, -8(%rbp)
.L14:
	cmpq	$2, -8(%rbp)
	je	.L9
	cmpq	$2, -8(%rbp)
	ja	.L16
	cmpq	$0, -8(%rbp)
	je	.L11
	cmpq	$1, -8(%rbp)
	jne	.L16
	jmp	.L15
.L11:
	movl	-20(%rbp), %eax
	shrl	$24, %eax
	movl	%eax, %edx
	movq	-32(%rbp), %rax
	movb	%dl, (%rax)
	movl	-20(%rbp), %eax
	shrl	$16, %eax
	movl	%eax, %edx
	movq	-32(%rbp), %rax
	addq	$1, %rax
	movb	%dl, (%rax)
	movl	-20(%rbp), %eax
	shrl	$8, %eax
	movl	%eax, %edx
	movq	-32(%rbp), %rax
	addq	$2, %rax
	movb	%dl, (%rax)
	movq	-32(%rbp), %rax
	addq	$3, %rax
	movl	-20(%rbp), %edx
	movb	%dl, (%rax)
	movq	$1, -8(%rbp)
	jmp	.L13
.L9:
	movq	$0, -8(%rbp)
	jmp	.L13
.L16:
	nop
.L13:
	jmp	.L14
.L15:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	Write_u32be, .-Write_u32be
	.globl	LZSS_GetVars
	.type	LZSS_GetVars, @function
LZSS_GetVars:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -56(%rbp)
	movq	%rsi, -64(%rbp)
	movq	%rdx, -72(%rbp)
	movq	$0, -24(%rbp)
.L48:
	cmpq	$18, -24(%rbp)
	ja	.L50
	movq	-24(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L20(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L20(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L20:
	.long	.L33-.L20
	.long	.L32-.L20
	.long	.L31-.L20
	.long	.L50-.L20
	.long	.L30-.L20
	.long	.L29-.L20
	.long	.L50-.L20
	.long	.L50-.L20
	.long	.L28-.L20
	.long	.L27-.L20
	.long	.L26-.L20
	.long	.L25-.L20
	.long	.L50-.L20
	.long	.L24-.L20
	.long	.L50-.L20
	.long	.L23-.L20
	.long	.L22-.L20
	.long	.L21-.L20
	.long	.L19-.L20
	.text
.L19:
	movq	-48(%rbp), %rax
	jmp	.L49
.L30:
	movq	-56(%rbp), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	movzbl	%al, %eax
	cmpq	%rax, -32(%rbp)
	jnb	.L35
	movq	$16, -24(%rbp)
	jmp	.L37
.L35:
	movq	$9, -24(%rbp)
	jmp	.L37
.L23:
	cmpq	$0, -72(%rbp)
	jne	.L38
	movq	$5, -24(%rbp)
	jmp	.L37
.L38:
	movq	$17, -24(%rbp)
	jmp	.L37
.L28:
	movq	-40(%rbp), %rdx
	movq	-32(%rbp), %rax
	addq	%rax, %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %edx
	movq	-56(%rbp), %rcx
	movq	-40(%rbp), %rax
	addq	%rcx, %rax
	movzbl	(%rax), %eax
	cmpb	%al, %dl
	jbe	.L40
	movq	$15, -24(%rbp)
	jmp	.L37
.L40:
	movq	$11, -24(%rbp)
	jmp	.L37
.L32:
	movq	-56(%rbp), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	cmpb	$2, %al
	jbe	.L42
	movq	$10, -24(%rbp)
	jmp	.L37
.L42:
	movq	$9, -24(%rbp)
	jmp	.L37
.L22:
	movq	-40(%rbp), %rdx
	movq	-32(%rbp), %rax
	addq	%rdx, %rax
	cmpq	%rax, -64(%rbp)
	jbe	.L44
	movq	$8, -24(%rbp)
	jmp	.L37
.L44:
	movq	$11, -24(%rbp)
	jmp	.L37
.L25:
	addq	$1, -32(%rbp)
	movq	$4, -24(%rbp)
	jmp	.L37
.L27:
	addq	$1, -40(%rbp)
	movq	$13, -24(%rbp)
	jmp	.L37
.L24:
	movq	-40(%rbp), %rax
	cmpq	-64(%rbp), %rax
	jnb	.L46
	movq	$1, -24(%rbp)
	jmp	.L37
.L46:
	movq	$18, -24(%rbp)
	jmp	.L37
.L21:
	movq	-48(%rbp), %rax
	movq	%rax, -16(%rbp)
	addq	$1, -48(%rbp)
	movq	-16(%rbp), %rax
	leaq	0(,%rax,8), %rdx
	movq	-72(%rbp), %rax
	addq	%rax, %rdx
	movq	-40(%rbp), %rax
	movq	%rax, (%rdx)
	movq	-48(%rbp), %rax
	movq	%rax, -8(%rbp)
	addq	$1, -48(%rbp)
	movq	-8(%rbp), %rax
	leaq	0(,%rax,8), %rdx
	movq	-72(%rbp), %rax
	addq	%rax, %rdx
	movq	-32(%rbp), %rax
	movq	%rax, (%rdx)
	movq	$11, -24(%rbp)
	jmp	.L37
.L29:
	addq	$2, -48(%rbp)
	movq	$11, -24(%rbp)
	jmp	.L37
.L26:
	movq	$1, -32(%rbp)
	movq	$4, -24(%rbp)
	jmp	.L37
.L33:
	movq	$2, -24(%rbp)
	jmp	.L37
.L31:
	movq	$0, -48(%rbp)
	movq	$0, -40(%rbp)
	movq	$13, -24(%rbp)
	jmp	.L37
.L50:
	nop
.L37:
	jmp	.L48
.L49:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	LZSS_GetVars, .-LZSS_GetVars
	.globl	LZSS_CalcLength
	.type	LZSS_CalcLength, @function
LZSS_CalcLength:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -40(%rbp)
	movq	%rsi, -48(%rbp)
	movq	$6, -8(%rbp)
.L67:
	cmpq	$7, -8(%rbp)
	ja	.L69
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L54(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L54(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L54:
	.long	.L60-.L54
	.long	.L69-.L54
	.long	.L59-.L54
	.long	.L58-.L54
	.long	.L57-.L54
	.long	.L56-.L54
	.long	.L55-.L54
	.long	.L53-.L54
	.text
.L57:
	movq	-24(%rbp), %rax
	leaq	(%rax,%rax), %rdx
	movq	-32(%rbp), %rax
	addq	%rdx, %rax
	leaq	0(,%rax,8), %rcx
	movq	-24(%rbp), %rdx
	movq	-32(%rbp), %rax
	addq	%rdx, %rax
	addq	%rcx, %rax
	jmp	.L68
.L58:
	addq	$1, -24(%rbp)
	movq	-40(%rbp), %rdx
	movq	-16(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	movzbl	%al, %eax
	addq	%rax, -16(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L62
.L55:
	movq	$7, -8(%rbp)
	jmp	.L62
.L56:
	movq	-16(%rbp), %rax
	cmpq	-48(%rbp), %rax
	jnb	.L63
	movq	$0, -8(%rbp)
	jmp	.L62
.L63:
	movq	$4, -8(%rbp)
	jmp	.L62
.L60:
	movq	-40(%rbp), %rdx
	movq	-16(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	cmpb	$2, %al
	jbe	.L65
	movq	$3, -8(%rbp)
	jmp	.L62
.L65:
	movq	$2, -8(%rbp)
	jmp	.L62
.L53:
	movq	$0, -32(%rbp)
	movq	$0, -24(%rbp)
	movq	$0, -16(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L62
.L59:
	addq	$1, -32(%rbp)
	addq	$1, -16(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L62
.L69:
	nop
.L62:
	jmp	.L67
.L68:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	LZSS_CalcLength, .-LZSS_CalcLength
	.globl	GetFileSize
	.type	GetFileSize, @function
GetFileSize:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movq	$2, -16(%rbp)
.L76:
	cmpq	$2, -16(%rbp)
	je	.L71
	cmpq	$2, -16(%rbp)
	ja	.L78
	cmpq	$0, -16(%rbp)
	je	.L73
	cmpq	$1, -16(%rbp)
	jne	.L78
	movq	-24(%rbp), %rax
	jmp	.L77
.L73:
	movq	-40(%rbp), %rax
	movl	$2, %edx
	movl	$0, %esi
	movq	%rax, %rdi
	call	fseek@PLT
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	ftell@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	-40(%rbp), %rax
	movl	$0, %edx
	movl	$0, %esi
	movq	%rax, %rdi
	call	fseek@PLT
	movq	$1, -16(%rbp)
	jmp	.L75
.L71:
	movq	$0, -16(%rbp)
	jmp	.L75
.L78:
	nop
.L75:
	jmp	.L76
.L77:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	GetFileSize, .-GetFileSize
	.section	.rodata
.LC0:
	.string	"c"
.LC1:
	.string	"d"
.LC2:
	.string	"p"
.LC3:
	.string	"nu"
	.align 8
.LC4:
	.ascii	"LZSS compressor - Desert/Jungle/Urban Strike (Mega Drive) ||"
	.ascii	" v1.2.0 by infval\nusage: %s INPUT OUTPUT [-c | -d] [-p file"
	.ascii	"_pos] [-nu]\npositional arguments:\n  INPUT          input f"
	.ascii	"ile; if decompress,"
	.string	" first 4 bytes (Big-Endian): uncompressed size\n  OUTPUT         output file\noptional arguments:\n  -c             compress (default)\n  -d             decompress\n  -p file_pos    start file position (default: 0)\n  -nu            average compression (faster)\n"
.LC5:
	.string	"Error: LZSS_Compress*()"
.LC6:
	.string	"main"
.LC7:
	.string	"infval_StrikeLZSS_main.c"
.LC8:
	.string	"dsize <= maxsize"
.LC9:
	.string	"Error: malloc()"
.LC10:
	.string	"Can't open: %s"
.LC11:
	.string	"Error: GetFileSize()"
.LC12:
	.string	"rb"
.LC13:
	.string	"Error: LZSS_Decompress()"
.LC14:
	.string	"Can't write: %s"
	.align 8
.LC15:
	.string	"Error: file_pos > input file size"
	.align 8
.LC16:
	.string	"Uncompressed size: %10lu | 0x%08lX\n"
	.align 8
.LC17:
	.string	"Compressed size  : %10lu | 0x%08lX"
.LC18:
	.string	"wb"
	.text
	.globl	main
	.type	main, @function
main:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$272, %rsp
	movl	%edi, -244(%rbp)
	movq	%rsi, -256(%rbp)
	movq	%rdx, -264(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	leaq	.LC0(%rip), %rax
	movq	%rax, arguments(%rip)
	leaq	.LC1(%rip), %rax
	movq	%rax, 8+arguments(%rip)
	leaq	.LC2(%rip), %rax
	movq	%rax, 16+arguments(%rip)
	leaq	.LC3(%rip), %rax
	movq	%rax, 24+arguments(%rip)
	movq	$0, 32+arguments(%rip)
	nop
.L80:
	movl	$0, -232(%rbp)
	jmp	.L81
.L82:
	movl	-232(%rbp), %eax
	cltq
	leaq	window(%rip), %rdx
	movb	$0, (%rax,%rdx)
	addl	$1, -232(%rbp)
.L81:
	cmpl	$2047, -232(%rbp)
	jle	.L82
	nop
.L83:
	movq	$0, _TIG_IZ_aTfC_envp(%rip)
	nop
.L84:
	movq	$0, _TIG_IZ_aTfC_argv(%rip)
	nop
.L85:
	movl	$0, _TIG_IZ_aTfC_argc(%rip)
	nop
	nop
.L86:
.L87:
#APP
# 405 "infval_StrikeLZSS_main.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-aTfC--0
# 0 "" 2
#NO_APP
	movl	-244(%rbp), %eax
	movl	%eax, _TIG_IZ_aTfC_argc(%rip)
	movq	-256(%rbp), %rax
	movq	%rax, _TIG_IZ_aTfC_argv(%rip)
	movq	-264(%rbp), %rax
	movq	%rax, _TIG_IZ_aTfC_envp(%rip)
	nop
	movq	$10, -128(%rbp)
.L194:
	cmpq	$76, -128(%rbp)
	ja	.L197
	movq	-128(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L90(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L90(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L90:
	.long	.L155-.L90
	.long	.L154-.L90
	.long	.L153-.L90
	.long	.L152-.L90
	.long	.L151-.L90
	.long	.L150-.L90
	.long	.L149-.L90
	.long	.L148-.L90
	.long	.L147-.L90
	.long	.L146-.L90
	.long	.L145-.L90
	.long	.L144-.L90
	.long	.L197-.L90
	.long	.L143-.L90
	.long	.L142-.L90
	.long	.L141-.L90
	.long	.L140-.L90
	.long	.L139-.L90
	.long	.L138-.L90
	.long	.L137-.L90
	.long	.L136-.L90
	.long	.L197-.L90
	.long	.L135-.L90
	.long	.L134-.L90
	.long	.L133-.L90
	.long	.L132-.L90
	.long	.L131-.L90
	.long	.L197-.L90
	.long	.L130-.L90
	.long	.L129-.L90
	.long	.L128-.L90
	.long	.L127-.L90
	.long	.L126-.L90
	.long	.L125-.L90
	.long	.L124-.L90
	.long	.L123-.L90
	.long	.L122-.L90
	.long	.L121-.L90
	.long	.L120-.L90
	.long	.L119-.L90
	.long	.L118-.L90
	.long	.L117-.L90
	.long	.L116-.L90
	.long	.L115-.L90
	.long	.L114-.L90
	.long	.L113-.L90
	.long	.L112-.L90
	.long	.L197-.L90
	.long	.L197-.L90
	.long	.L111-.L90
	.long	.L110-.L90
	.long	.L109-.L90
	.long	.L108-.L90
	.long	.L197-.L90
	.long	.L197-.L90
	.long	.L107-.L90
	.long	.L106-.L90
	.long	.L105-.L90
	.long	.L197-.L90
	.long	.L104-.L90
	.long	.L103-.L90
	.long	.L102-.L90
	.long	.L101-.L90
	.long	.L197-.L90
	.long	.L197-.L90
	.long	.L100-.L90
	.long	.L99-.L90
	.long	.L98-.L90
	.long	.L97-.L90
	.long	.L96-.L90
	.long	.L197-.L90
	.long	.L95-.L90
	.long	.L94-.L90
	.long	.L93-.L90
	.long	.L92-.L90
	.long	.L91-.L90
	.long	.L89-.L90
	.text
.L138:
	movl	$1, %eax
	jmp	.L195
.L110:
	movq	-256(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, %rsi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$72, -128(%rbp)
	jmp	.L157
.L132:
	movl	-28(%rbp), %eax
	movl	%eax, %edx
	movq	-216(%rbp), %rax
	subq	-192(%rbp), %rax
	cmpq	%rax, %rdx
	jbe	.L158
	movq	$65, -128(%rbp)
	jmp	.L157
.L158:
	movq	$49, -128(%rbp)
	jmp	.L157
.L111:
	movl	-28(%rbp), %eax
	movl	%eax, %edx
	movq	-208(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, %rdi
	call	Read_u32be
	movl	%eax, -228(%rbp)
	movl	-228(%rbp), %eax
	movq	%rax, -168(%rbp)
	movq	-168(%rbp), %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -88(%rbp)
	movq	-88(%rbp), %rax
	movq	%rax, -184(%rbp)
	movq	$33, -128(%rbp)
	jmp	.L157
.L108:
	movq	-176(%rbp), %rax
	movq	%rax, -160(%rbp)
	movq	-192(%rbp), %rax
	addq	%rax, -176(%rbp)
	movq	$0, -128(%rbp)
	jmp	.L157
.L151:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$23, %edx
	movl	$1, %esi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	$43, -128(%rbp)
	jmp	.L157
.L128:
	cmpq	$0, -184(%rbp)
	jne	.L160
	movq	$55, -128(%rbp)
	jmp	.L157
.L160:
	movq	$9, -128(%rbp)
	jmp	.L157
.L101:
	movq	-256(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, %rsi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$32, -128(%rbp)
	jmp	.L157
.L142:
	movl	$1, %eax
	jmp	.L195
.L141:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rcx
	movl	$491, %edx
	leaq	.LC7(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L106:
	movq	-224(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$4, -192(%rbp)
	movq	$0, -184(%rbp)
	movq	$0, -176(%rbp)
	movq	$0, -168(%rbp)
	movq	$0, -160(%rbp)
	movq	$20, -128(%rbp)
	jmp	.L157
.L127:
	movl	-28(%rbp), %eax
	movl	%eax, %edx
	movq	-216(%rbp), %rax
	subq	%rdx, %rax
	movq	%rax, -168(%rbp)
	movq	-168(%rbp), %rax
	movq	%rax, %rdi
	call	LZSS_GetCompressedMaxSize
	movq	%rax, -72(%rbp)
	movq	-72(%rbp), %rdx
	movq	-192(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, -152(%rbp)
	movq	-152(%rbp), %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -64(%rbp)
	movq	-64(%rbp), %rax
	movq	%rax, -184(%rbp)
	movq	$30, -128(%rbp)
	jmp	.L157
.L96:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$15, %edx
	movl	$1, %esi
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	$74, -128(%rbp)
	jmp	.L157
.L147:
	movl	$1, %eax
	jmp	.L195
.L113:
	movq	-40(%rbp), %rdx
	movq	stderr(%rip), %rax
	leaq	.LC10(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	$51, -128(%rbp)
	jmp	.L157
.L154:
	movl	$1, %eax
	jmp	.L195
.L134:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$20, %edx
	movl	$1, %esi
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	$19, -128(%rbp)
	jmp	.L157
.L152:
	movq	-48(%rbp), %rax
	leaq	.LC12(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -56(%rbp)
	movq	-56(%rbp), %rax
	movq	%rax, -224(%rbp)
	movq	$73, -128(%rbp)
	jmp	.L157
.L140:
	movq	$0, -48(%rbp)
	movq	$0, -40(%rbp)
	movl	$0, -32(%rbp)
	movl	$0, -28(%rbp)
	movb	$0, -24(%rbp)
	leaq	-48(%rbp), %rdx
	movq	-256(%rbp), %rcx
	movl	-244(%rbp), %eax
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	argparse
	movq	$2, -128(%rbp)
	jmp	.L157
.L133:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$24, %edx
	movl	$1, %esi
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	$40, -128(%rbp)
	jmp	.L157
.L122:
	movl	$1, %eax
	jmp	.L195
.L89:
	movq	-184(%rbp), %rdx
	movq	-192(%rbp), %rax
	addq	%rax, %rdx
	movl	-28(%rbp), %eax
	movl	%eax, %ecx
	movq	-208(%rbp), %rax
	addq	%rax, %rcx
	movq	-168(%rbp), %rax
	movq	%rax, %rsi
	movq	%rcx, %rdi
	call	LZSS_CompressUltra
	movq	%rax, -176(%rbp)
	movq	$68, -128(%rbp)
	jmp	.L157
.L105:
	movq	-224(%rbp), %rcx
	movq	-216(%rbp), %rdx
	movq	-208(%rbp), %rax
	movl	$1, %esi
	movq	%rax, %rdi
	call	fread@PLT
	movq	%rax, -96(%rbp)
	movq	-96(%rbp), %rax
	movq	%rax, -200(%rbp)
	movq	$11, -128(%rbp)
	jmp	.L157
.L97:
	cmpq	$-1, -176(%rbp)
	jne	.L162
	movq	$4, -128(%rbp)
	jmp	.L157
.L162:
	movq	$52, -128(%rbp)
	jmp	.L157
.L131:
	movq	-40(%rbp), %rdx
	movq	stderr(%rip), %rax
	leaq	.LC14(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	$14, -128(%rbp)
	jmp	.L157
.L144:
	movq	-200(%rbp), %rax
	cmpq	-216(%rbp), %rax
	je	.L164
	movq	$17, -128(%rbp)
	jmp	.L157
.L164:
	movq	$56, -128(%rbp)
	jmp	.L157
.L146:
	movq	-168(%rbp), %rax
	movl	%eax, %edx
	movq	-184(%rbp), %rax
	movq	%rax, %rsi
	movl	%edx, %edi
	call	Write_u32be
	movq	$37, -128(%rbp)
	jmp	.L157
.L143:
	movl	$1, %eax
	jmp	.L195
.L109:
	movl	$1, %eax
	jmp	.L195
.L137:
	movl	$1, %eax
	jmp	.L195
.L126:
	movl	$0, %eax
	jmp	.L195
.L139:
	movq	-48(%rbp), %rdx
	movq	stderr(%rip), %rax
	leaq	.LC10(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	$34, -128(%rbp)
	jmp	.L157
.L118:
	movl	$1, %eax
	jmp	.L195
.L98:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$15, %edx
	movl	$1, %esi
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	$13, -128(%rbp)
	jmp	.L157
.L107:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$15, %edx
	movl	$1, %esi
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	$36, -128(%rbp)
	jmp	.L157
.L103:
	movq	-224(%rbp), %rax
	movq	%rax, %rdi
	call	GetFileSize
	movq	%rax, -112(%rbp)
	movq	-112(%rbp), %rax
	movq	%rax, -216(%rbp)
	movq	$61, -128(%rbp)
	jmp	.L157
.L104:
	movq	-144(%rbp), %rcx
	movq	-176(%rbp), %rdx
	movq	-184(%rbp), %rax
	movl	$1, %esi
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	%rax, -104(%rbp)
	movq	-104(%rbp), %rax
	movq	%rax, -136(%rbp)
	movq	$6, -128(%rbp)
	jmp	.L157
.L149:
	movq	-136(%rbp), %rax
	cmpq	-176(%rbp), %rax
	je	.L166
	movq	$26, -128(%rbp)
	jmp	.L157
.L166:
	movq	$39, -128(%rbp)
	jmp	.L157
.L120:
	movq	-40(%rbp), %rax
	testq	%rax, %rax
	jne	.L168
	movq	$50, -128(%rbp)
	jmp	.L157
.L168:
	movq	$3, -128(%rbp)
	jmp	.L157
.L102:
	cmpq	$-1, -216(%rbp)
	jne	.L170
	movq	$23, -128(%rbp)
	jmp	.L157
.L170:
	movq	$75, -128(%rbp)
	jmp	.L157
.L124:
	movl	$1, %eax
	jmp	.L195
.L92:
	movl	$1, %eax
	jmp	.L195
.L91:
	movq	-216(%rbp), %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -120(%rbp)
	movq	-120(%rbp), %rax
	movq	%rax, -208(%rbp)
	movq	$7, -128(%rbp)
	jmp	.L157
.L95:
	movq	-184(%rbp), %rdx
	movq	-192(%rbp), %rax
	addq	%rax, %rdx
	movl	-28(%rbp), %eax
	movl	%eax, %ecx
	movq	-208(%rbp), %rax
	leaq	(%rcx,%rax), %rdi
	movq	-168(%rbp), %rax
	movl	$0, %ecx
	movq	%rax, %rsi
	call	LZSS_Compress
	movq	%rax, -176(%rbp)
	movq	$68, -128(%rbp)
	jmp	.L157
.L135:
	movl	-32(%rbp), %eax
	cmpl	$1, %eax
	jne	.L172
	movq	$25, -128(%rbp)
	jmp	.L157
.L172:
	movq	$46, -128(%rbp)
	jmp	.L157
.L130:
	movl	-28(%rbp), %eax
	movl	%eax, %eax
	cmpq	%rax, -216(%rbp)
	jnb	.L174
	movq	$66, -128(%rbp)
	jmp	.L157
.L174:
	movq	$31, -128(%rbp)
	jmp	.L157
.L100:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$33, %edx
	movl	$1, %esi
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	$1, -128(%rbp)
	jmp	.L157
.L93:
	cmpq	$0, -224(%rbp)
	jne	.L176
	movq	$35, -128(%rbp)
	jmp	.L157
.L176:
	movq	$60, -128(%rbp)
	jmp	.L157
.L114:
	movq	-168(%rbp), %rax
	movq	%rax, -176(%rbp)
	movq	$46, -128(%rbp)
	jmp	.L157
.L150:
	cmpq	$0, -144(%rbp)
	jne	.L178
	movq	$45, -128(%rbp)
	jmp	.L157
.L178:
	movq	$59, -128(%rbp)
	jmp	.L157
.L94:
	movl	$0, %eax
	jmp	.L195
.L125:
	cmpq	$0, -184(%rbp)
	jne	.L180
	movq	$67, -128(%rbp)
	jmp	.L157
.L180:
	movq	$42, -128(%rbp)
	jmp	.L157
.L121:
	movzbl	-24(%rbp), %eax
	testb	%al, %al
	je	.L182
	movq	$76, -128(%rbp)
	jmp	.L157
.L182:
	movq	$71, -128(%rbp)
	jmp	.L157
.L117:
	movl	$0, %eax
	jmp	.L195
.L145:
	movq	$16, -128(%rbp)
	jmp	.L157
.L116:
	movq	-168(%rbp), %rax
	movl	%eax, %ecx
	movl	-28(%rbp), %eax
	movl	%eax, %edx
	movq	-216(%rbp), %rax
	subq	%rdx, %rax
	subq	-192(%rbp), %rax
	movl	-28(%rbp), %edx
	movl	%edx, %esi
	movq	-192(%rbp), %rdx
	addq	%rdx, %rsi
	movq	-208(%rbp), %rdx
	leaq	(%rsi,%rdx), %rdi
	movq	-184(%rbp), %rdx
	movq	%rax, %rsi
	call	LZSS_Decompress
	movq	%rax, -160(%rbp)
	movq	$29, -128(%rbp)
	jmp	.L157
.L155:
	movq	-176(%rbp), %rax
	cmpq	-152(%rbp), %rax
	ja	.L184
	movq	$46, -128(%rbp)
	jmp	.L157
.L184:
	movq	$15, -128(%rbp)
	jmp	.L157
.L112:
	movq	-168(%rbp), %rdx
	movq	-168(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC16(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-160(%rbp), %rdx
	movq	-160(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC17(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-40(%rbp), %rax
	leaq	.LC18(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -80(%rbp)
	movq	-80(%rbp), %rax
	movq	%rax, -144(%rbp)
	movq	$5, -128(%rbp)
	jmp	.L157
.L119:
	movq	-144(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	-184(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	-208(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$41, -128(%rbp)
	jmp	.L157
.L99:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$33, %edx
	movl	$1, %esi
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	$8, -128(%rbp)
	jmp	.L157
.L148:
	cmpq	$0, -208(%rbp)
	jne	.L186
	movq	$69, -128(%rbp)
	jmp	.L157
.L186:
	movq	$57, -128(%rbp)
	jmp	.L157
.L123:
	movq	-48(%rbp), %rdx
	movq	stderr(%rip), %rax
	leaq	.LC10(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	$18, -128(%rbp)
	jmp	.L157
.L129:
	cmpq	$-1, -160(%rbp)
	jne	.L188
	movq	$24, -128(%rbp)
	jmp	.L157
.L188:
	movq	$44, -128(%rbp)
	jmp	.L157
.L115:
	movl	$1, %eax
	jmp	.L195
.L153:
	movq	-48(%rbp), %rax
	testq	%rax, %rax
	jne	.L190
	movq	$62, -128(%rbp)
	jmp	.L157
.L190:
	movq	$38, -128(%rbp)
	jmp	.L157
.L136:
	movl	-32(%rbp), %eax
	testl	%eax, %eax
	jne	.L192
	movq	$28, -128(%rbp)
	jmp	.L157
.L192:
	movq	$22, -128(%rbp)
	jmp	.L157
.L197:
	nop
.L157:
	jmp	.L194
.L195:
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L196
	call	__stack_chk_fail@PLT
.L196:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	main, .-main
	.globl	LZSS_Compress
	.type	LZSS_Compress, @function
LZSS_Compress:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$40, %rsp
	movq	%rdi, -136(%rbp)
	movq	%rsi, -144(%rbp)
	movq	%rdx, -152(%rbp)
	movq	%rcx, -160(%rbp)
	movq	$16, -64(%rbp)
.L257:
	cmpq	$42, -64(%rbp)
	ja	.L259
	movq	-64(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L201(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L201(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L201:
	.long	.L259-.L201
	.long	.L228-.L201
	.long	.L227-.L201
	.long	.L226-.L201
	.long	.L225-.L201
	.long	.L224-.L201
	.long	.L223-.L201
	.long	.L259-.L201
	.long	.L259-.L201
	.long	.L222-.L201
	.long	.L221-.L201
	.long	.L259-.L201
	.long	.L259-.L201
	.long	.L220-.L201
	.long	.L259-.L201
	.long	.L219-.L201
	.long	.L218-.L201
	.long	.L259-.L201
	.long	.L217-.L201
	.long	.L216-.L201
	.long	.L215-.L201
	.long	.L259-.L201
	.long	.L214-.L201
	.long	.L213-.L201
	.long	.L259-.L201
	.long	.L259-.L201
	.long	.L212-.L201
	.long	.L211-.L201
	.long	.L210-.L201
	.long	.L209-.L201
	.long	.L259-.L201
	.long	.L208-.L201
	.long	.L259-.L201
	.long	.L207-.L201
	.long	.L206-.L201
	.long	.L259-.L201
	.long	.L205-.L201
	.long	.L259-.L201
	.long	.L204-.L201
	.long	.L203-.L201
	.long	.L202-.L201
	.long	.L259-.L201
	.long	.L200-.L201
	.text
.L217:
	cmpq	$2, -88(%rbp)
	jbe	.L229
	movq	$5, -64(%rbp)
	jmp	.L231
.L229:
	movq	$29, -64(%rbp)
	jmp	.L231
.L225:
	subq	$1, -112(%rbp)
	movq	$38, -64(%rbp)
	jmp	.L231
.L219:
	cmpb	$0, -121(%rbp)
	jne	.L232
	movq	$4, -64(%rbp)
	jmp	.L231
.L232:
	movq	$36, -64(%rbp)
	jmp	.L231
.L208:
	movq	-80(%rbp), %rax
	addq	$2030, %rax
	andl	$2047, %eax
	movq	%rax, -96(%rbp)
	movq	-72(%rbp), %rax
	movq	%rax, -88(%rbp)
	movq	$13, -64(%rbp)
	jmp	.L231
.L228:
	subq	$1, -80(%rbp)
	movq	$20, -64(%rbp)
	jmp	.L231
.L213:
	movq	$0, -96(%rbp)
	movq	$0, -88(%rbp)
	movq	-120(%rbp), %rax
	subq	$1, %rax
	movq	%rax, -80(%rbp)
	movq	$20, -64(%rbp)
	jmp	.L231
.L226:
	movq	-80(%rbp), %rdx
	movq	-72(%rbp), %rax
	addq	%rax, %rdx
	movq	-136(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %edx
	movq	-120(%rbp), %rcx
	movq	-72(%rbp), %rax
	addq	%rax, %rcx
	movq	-136(%rbp), %rax
	addq	%rcx, %rax
	movzbl	(%rax), %eax
	cmpb	%al, %dl
	jne	.L234
	movq	$2, -64(%rbp)
	jmp	.L231
.L234:
	movq	$27, -64(%rbp)
	jmp	.L231
.L218:
	movq	$26, -64(%rbp)
	jmp	.L231
.L205:
	movq	-152(%rbp), %rdx
	movq	-104(%rbp), %rax
	addq	%rax, %rdx
	movzbl	-122(%rbp), %eax
	movb	%al, (%rdx)
	movq	$38, -64(%rbp)
	jmp	.L231
.L212:
	movq	$0, -120(%rbp)
	movq	$0, -112(%rbp)
	movb	$0, -122(%rbp)
	movb	$0, -121(%rbp)
	movq	$0, -104(%rbp)
	movq	-112(%rbp), %rax
	movq	%rax, -40(%rbp)
	addq	$1, -112(%rbp)
	movq	-152(%rbp), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movb	$0, (%rax)
	movq	$6, -64(%rbp)
	jmp	.L231
.L222:
	movq	-160(%rbp), %rdx
	movq	-120(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	movzbl	%al, %eax
	movq	%rax, -88(%rbp)
	movq	$18, -64(%rbp)
	jmp	.L231
.L220:
	cmpq	$17, -88(%rbp)
	jbe	.L236
	movq	$28, -64(%rbp)
	jmp	.L231
.L236:
	movq	$1, -64(%rbp)
	jmp	.L231
.L216:
	movq	-152(%rbp), %rdx
	movq	-104(%rbp), %rax
	addq	%rax, %rdx
	movzbl	-122(%rbp), %eax
	movb	%al, (%rdx)
	movb	$0, -122(%rbp)
	movb	$0, -121(%rbp)
	movq	-112(%rbp), %rax
	movq	%rax, -104(%rbp)
	movq	-112(%rbp), %rax
	movq	%rax, -32(%rbp)
	addq	$1, -112(%rbp)
	movq	-152(%rbp), %rdx
	movq	-32(%rbp), %rax
	addq	%rdx, %rax
	movb	$0, (%rax)
	movq	$6, -64(%rbp)
	jmp	.L231
.L202:
	movzbl	-121(%rbp), %eax
	addl	$1, %eax
	movb	%al, -121(%rbp)
	movq	$34, -64(%rbp)
	jmp	.L231
.L223:
	movq	-120(%rbp), %rax
	cmpq	-144(%rbp), %rax
	jnb	.L238
	movq	$23, -64(%rbp)
	jmp	.L231
.L238:
	movq	$15, -64(%rbp)
	jmp	.L231
.L211:
	movq	-72(%rbp), %rax
	cmpq	-88(%rbp), %rax
	jbe	.L240
	movq	$31, -64(%rbp)
	jmp	.L231
.L240:
	movq	$1, -64(%rbp)
	jmp	.L231
.L204:
	movq	-112(%rbp), %rax
	jmp	.L258
.L206:
	cmpb	$7, -121(%rbp)
	jle	.L243
	movq	$19, -64(%rbp)
	jmp	.L231
.L243:
	movq	$6, -64(%rbp)
	jmp	.L231
.L214:
	movq	-120(%rbp), %rax
	subq	$2048, %rax
	cmpq	%rax, -80(%rbp)
	jl	.L245
	movq	$39, -64(%rbp)
	jmp	.L231
.L245:
	movq	$28, -64(%rbp)
	jmp	.L231
.L210:
	cmpq	$0, -160(%rbp)
	je	.L247
	movq	$9, -64(%rbp)
	jmp	.L231
.L247:
	movq	$18, -64(%rbp)
	jmp	.L231
.L224:
	movq	-96(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movzbl	%al, %edx
	movq	-24(%rbp), %rax
	salq	$4, %rax
	andl	$61440, %eax
	orq	%rdx, %rax
	movq	%rax, -24(%rbp)
	movq	-88(%rbp), %rax
	subq	$3, %rax
	salq	$8, %rax
	orq	%rax, -24(%rbp)
	movq	-112(%rbp), %rax
	movq	%rax, -16(%rbp)
	addq	$1, -112(%rbp)
	movq	-152(%rbp), %rdx
	movq	-16(%rbp), %rax
	addq	%rdx, %rax
	movq	-24(%rbp), %rdx
	movb	%dl, (%rax)
	movq	-112(%rbp), %rax
	movq	%rax, -8(%rbp)
	addq	$1, -112(%rbp)
	movq	-24(%rbp), %rax
	shrq	$8, %rax
	movq	%rax, %rcx
	movq	-152(%rbp), %rdx
	movq	-8(%rbp), %rax
	addq	%rdx, %rax
	movl	%ecx, %edx
	movb	%dl, (%rax)
	movq	-88(%rbp), %rax
	addq	%rax, -120(%rbp)
	movq	$40, -64(%rbp)
	jmp	.L231
.L207:
	movq	$0, -72(%rbp)
	movq	$2, -64(%rbp)
	jmp	.L231
.L221:
	movq	-120(%rbp), %rdx
	movq	-72(%rbp), %rax
	addq	%rdx, %rax
	cmpq	%rax, -144(%rbp)
	jbe	.L249
	movq	$3, -64(%rbp)
	jmp	.L231
.L249:
	movq	$27, -64(%rbp)
	jmp	.L231
.L200:
	cmpq	$17, -72(%rbp)
	ja	.L251
	movq	$10, -64(%rbp)
	jmp	.L231
.L251:
	movq	$27, -64(%rbp)
	jmp	.L231
.L203:
	movq	-80(%rbp), %rdx
	movq	-136(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %edx
	movq	-136(%rbp), %rcx
	movq	-120(%rbp), %rax
	addq	%rcx, %rax
	movzbl	(%rax), %eax
	cmpb	%al, %dl
	jne	.L253
	movq	$33, -64(%rbp)
	jmp	.L231
.L253:
	movq	$1, -64(%rbp)
	jmp	.L231
.L209:
	movq	-112(%rbp), %rax
	movq	%rax, -56(%rbp)
	addq	$1, -112(%rbp)
	movq	-120(%rbp), %rax
	movq	%rax, -48(%rbp)
	addq	$1, -120(%rbp)
	movq	-136(%rbp), %rdx
	movq	-48(%rbp), %rax
	addq	%rdx, %rax
	movq	-152(%rbp), %rcx
	movq	-56(%rbp), %rdx
	addq	%rcx, %rdx
	movzbl	(%rax), %eax
	movb	%al, (%rdx)
	movsbl	-121(%rbp), %eax
	movl	$1, %edx
	movl	%eax, %ecx
	sall	%cl, %edx
	movl	%edx, %eax
	movl	%eax, %edx
	movzbl	-122(%rbp), %eax
	orl	%edx, %eax
	movb	%al, -122(%rbp)
	movq	$40, -64(%rbp)
	jmp	.L231
.L227:
	addq	$1, -72(%rbp)
	movq	$42, -64(%rbp)
	jmp	.L231
.L215:
	cmpq	$0, -80(%rbp)
	js	.L255
	movq	$22, -64(%rbp)
	jmp	.L231
.L255:
	movq	$28, -64(%rbp)
	jmp	.L231
.L259:
	nop
.L231:
	jmp	.L257
.L258:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	LZSS_Compress, .-LZSS_Compress
	.globl	Parse_ulong
	.type	Parse_ulong, @function
Parse_ulong:
.LFB10:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$80, %rsp
	movq	%rdi, -72(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$1, -32(%rbp)
.L283:
	cmpq	$14, -32(%rbp)
	ja	.L286
	movq	-32(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L263(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L263(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L263:
	.long	.L286-.L263
	.long	.L274-.L263
	.long	.L286-.L263
	.long	.L273-.L263
	.long	.L286-.L263
	.long	.L272-.L263
	.long	.L271-.L263
	.long	.L270-.L263
	.long	.L269-.L263
	.long	.L268-.L263
	.long	.L267-.L263
	.long	.L266-.L263
	.long	.L265-.L263
	.long	.L264-.L263
	.long	.L262-.L263
	.text
.L262:
	leaq	-19(%rbp), %rax
	addq	$2, %rax
	movl	$16, %edx
	movl	$0, %esi
	movq	%rax, %rdi
	call	strtoul@PLT
	movq	%rax, -48(%rbp)
	movq	$5, -32(%rbp)
	jmp	.L275
.L265:
	movb	$0, -19(%rbp)
	movl	$1, -56(%rbp)
	movq	$3, -32(%rbp)
	jmp	.L275
.L269:
	movzbl	-19(%rbp), %eax
	cmpb	$48, %al
	jne	.L276
	movq	$6, -32(%rbp)
	jmp	.L275
.L276:
	movq	$7, -32(%rbp)
	jmp	.L275
.L274:
	movq	$12, -32(%rbp)
	jmp	.L275
.L273:
	cmpl	$10, -56(%rbp)
	jbe	.L278
	movq	$11, -32(%rbp)
	jmp	.L275
.L278:
	movq	$13, -32(%rbp)
	jmp	.L275
.L266:
	movq	-72(%rbp), %rcx
	leaq	-19(%rbp), %rax
	movl	$10, %edx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	strncpy@PLT
	movq	$8, -32(%rbp)
	jmp	.L275
.L268:
	cmpl	$88, -52(%rbp)
	jne	.L280
	movq	$14, -32(%rbp)
	jmp	.L275
.L280:
	movq	$7, -32(%rbp)
	jmp	.L275
.L264:
	movl	-56(%rbp), %eax
	movb	$0, -19(%rbp,%rax)
	addl	$1, -56(%rbp)
	movq	$3, -32(%rbp)
	jmp	.L275
.L271:
	movzbl	-18(%rbp), %eax
	movsbl	%al, %eax
	movl	%eax, %edi
	call	toupper@PLT
	movl	%eax, -52(%rbp)
	movq	$9, -32(%rbp)
	jmp	.L275
.L272:
	movq	-48(%rbp), %rax
	jmp	.L284
.L267:
	movq	-40(%rbp), %rax
	jmp	.L284
.L270:
	leaq	-19(%rbp), %rax
	movl	$10, %edx
	movl	$0, %esi
	movq	%rax, %rdi
	call	strtoul@PLT
	movq	%rax, -40(%rbp)
	movq	$10, -32(%rbp)
	jmp	.L275
.L286:
	nop
.L275:
	jmp	.L283
.L284:
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L285
	call	__stack_chk_fail@PLT
.L285:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	Parse_ulong, .-Parse_ulong
	.globl	LZSS_Decompress
	.type	LZSS_Decompress, @function
LZSS_Decompress:
.LFB11:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$8, %rsp
	movq	%rdi, -104(%rbp)
	movq	%rsi, -112(%rbp)
	movq	%rdx, -120(%rbp)
	movl	%ecx, -124(%rbp)
	movq	$17, -40(%rbp)
.L335:
	cmpq	$34, -40(%rbp)
	ja	.L336
	movq	-40(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L290(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L290(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L290:
	.long	.L314-.L290
	.long	.L336-.L290
	.long	.L313-.L290
	.long	.L336-.L290
	.long	.L312-.L290
	.long	.L336-.L290
	.long	.L336-.L290
	.long	.L311-.L290
	.long	.L336-.L290
	.long	.L310-.L290
	.long	.L309-.L290
	.long	.L336-.L290
	.long	.L336-.L290
	.long	.L308-.L290
	.long	.L336-.L290
	.long	.L307-.L290
	.long	.L306-.L290
	.long	.L305-.L290
	.long	.L304-.L290
	.long	.L303-.L290
	.long	.L302-.L290
	.long	.L301-.L290
	.long	.L300-.L290
	.long	.L299-.L290
	.long	.L336-.L290
	.long	.L298-.L290
	.long	.L297-.L290
	.long	.L296-.L290
	.long	.L336-.L290
	.long	.L295-.L290
	.long	.L294-.L290
	.long	.L293-.L290
	.long	.L292-.L290
	.long	.L291-.L290
	.long	.L289-.L290
	.text
.L304:
	cmpl	$0, -124(%rbp)
	jne	.L315
	movq	$33, -40(%rbp)
	jmp	.L317
.L315:
	movq	$16, -40(%rbp)
	jmp	.L317
.L298:
	leaq	window(%rip), %rdx
	movq	-64(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	movb	%al, -91(%rbp)
	movq	-64(%rbp), %rax
	addq	$1, %rax
	andl	$2047, %eax
	movq	%rax, -64(%rbp)
	movq	-80(%rbp), %rax
	movq	%rax, -32(%rbp)
	addq	$1, -80(%rbp)
	movq	-120(%rbp), %rdx
	movq	-32(%rbp), %rax
	addq	%rax, %rdx
	movzbl	-91(%rbp), %eax
	movb	%al, (%rdx)
	subl	$1, -124(%rbp)
	movq	$18, -40(%rbp)
	jmp	.L317
.L312:
	movq	$-1, %rax
	jmp	.L318
.L294:
	cmpq	$1, -112(%rbp)
	ja	.L319
	movq	$4, -40(%rbp)
	jmp	.L317
.L319:
	movq	$31, -40(%rbp)
	jmp	.L317
.L307:
	leaq	window(%rip), %rdx
	movq	-88(%rbp), %rax
	addq	%rax, %rdx
	movzbl	-92(%rbp), %eax
	movb	%al, (%rdx)
	movq	-88(%rbp), %rax
	addq	$1, %rax
	andl	$2047, %eax
	movq	%rax, -88(%rbp)
	movq	$34, -40(%rbp)
	jmp	.L317
.L293:
	subq	$2, -112(%rbp)
	movq	-104(%rbp), %rdx
	movq	-72(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	movzbl	%al, %edx
	movq	-72(%rbp), %rax
	leaq	1(%rax), %rcx
	movq	-104(%rbp), %rax
	addq	%rcx, %rax
	movzbl	(%rax), %eax
	movzbl	%al, %eax
	sall	$4, %eax
	andl	$3840, %eax
	orl	%edx, %eax
	cltq
	movq	%rax, -64(%rbp)
	movq	-72(%rbp), %rax
	leaq	1(%rax), %rdx
	movq	-104(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	movzbl	%al, %eax
	andl	$15, %eax
	addl	$3, %eax
	cltq
	movq	%rax, -56(%rbp)
	addq	$2, -72(%rbp)
	movq	$19, -40(%rbp)
	jmp	.L317
.L299:
	cmpq	$0, -112(%rbp)
	jne	.L321
	movq	$20, -40(%rbp)
	jmp	.L317
.L321:
	movq	$9, -40(%rbp)
	jmp	.L317
.L306:
	leaq	window(%rip), %rdx
	movq	-88(%rbp), %rax
	addq	%rax, %rdx
	movzbl	-91(%rbp), %eax
	movb	%al, (%rdx)
	movq	-88(%rbp), %rax
	addq	$1, %rax
	andl	$2047, %eax
	movq	%rax, -88(%rbp)
	movq	$19, -40(%rbp)
	jmp	.L317
.L301:
	movzwl	-90(%rbp), %eax
	andl	$1, %eax
	testl	%eax, %eax
	je	.L323
	movq	$32, -40(%rbp)
	jmp	.L317
.L323:
	movq	$30, -40(%rbp)
	jmp	.L317
.L297:
	movl	$0, %eax
	jmp	.L318
.L310:
	subq	$1, -112(%rbp)
	movq	-72(%rbp), %rax
	movq	%rax, -8(%rbp)
	addq	$1, -72(%rbp)
	movq	-104(%rbp), %rdx
	movq	-8(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	movzbl	%al, %eax
	orw	$-256, %ax
	movw	%ax, -90(%rbp)
	movq	$21, -40(%rbp)
	jmp	.L317
.L308:
	movq	$-1, %rax
	jmp	.L318
.L303:
	movq	-56(%rbp), %rax
	movq	%rax, -48(%rbp)
	subq	$1, -56(%rbp)
	movq	$22, -40(%rbp)
	jmp	.L317
.L292:
	cmpq	$0, -112(%rbp)
	jne	.L325
	movq	$13, -40(%rbp)
	jmp	.L317
.L325:
	movq	$10, -40(%rbp)
	jmp	.L317
.L305:
	movq	$0, -40(%rbp)
	jmp	.L317
.L296:
	movq	-72(%rbp), %rax
	jmp	.L318
.L289:
	shrw	-90(%rbp)
	movq	$2, -40(%rbp)
	jmp	.L317
.L300:
	cmpq	$0, -48(%rbp)
	je	.L327
	movq	$25, -40(%rbp)
	jmp	.L317
.L327:
	movq	$34, -40(%rbp)
	jmp	.L317
.L291:
	movq	-72(%rbp), %rax
	jmp	.L318
.L309:
	subq	$1, -112(%rbp)
	movq	-72(%rbp), %rax
	movq	%rax, -24(%rbp)
	addq	$1, -72(%rbp)
	movq	-104(%rbp), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	movb	%al, -92(%rbp)
	movq	-80(%rbp), %rax
	movq	%rax, -16(%rbp)
	addq	$1, -80(%rbp)
	movq	-120(%rbp), %rdx
	movq	-16(%rbp), %rax
	addq	%rax, %rdx
	movzbl	-92(%rbp), %eax
	movb	%al, (%rdx)
	subl	$1, -124(%rbp)
	movq	$7, -40(%rbp)
	jmp	.L317
.L314:
	movq	$2030, -88(%rbp)
	movq	$0, -80(%rbp)
	movq	$0, -72(%rbp)
	movw	$0, -90(%rbp)
	movq	$29, -40(%rbp)
	jmp	.L317
.L311:
	cmpl	$0, -124(%rbp)
	jne	.L329
	movq	$27, -40(%rbp)
	jmp	.L317
.L329:
	movq	$15, -40(%rbp)
	jmp	.L317
.L295:
	cmpl	$0, -124(%rbp)
	jne	.L331
	movq	$26, -40(%rbp)
	jmp	.L317
.L331:
	movq	$34, -40(%rbp)
	jmp	.L317
.L313:
	movzwl	-90(%rbp), %eax
	andl	$256, %eax
	testl	%eax, %eax
	jne	.L333
	movq	$23, -40(%rbp)
	jmp	.L317
.L333:
	movq	$21, -40(%rbp)
	jmp	.L317
.L302:
	movq	$-1, %rax
	jmp	.L318
.L336:
	nop
.L317:
	jmp	.L335
.L318:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE11:
	.size	LZSS_Decompress, .-LZSS_Decompress
	.section	.rodata
	.align 8
.LC19:
	.string	"Average comp.size: %10lu | 0x%08lX\n"
	.text
	.globl	LZSS_CompressUltra
	.type	LZSS_CompressUltra, @function
LZSS_CompressUltra:
.LFB12:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$208, %rsp
	movq	%rdi, -184(%rbp)
	movq	%rsi, -192(%rbp)
	movq	%rdx, -200(%rbp)
	movq	$9, -64(%rbp)
.L407:
	cmpq	$55, -64(%rbp)
	ja	.L408
	movq	-64(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L340(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L340(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L340:
	.long	.L376-.L340
	.long	.L375-.L340
	.long	.L374-.L340
	.long	.L373-.L340
	.long	.L372-.L340
	.long	.L371-.L340
	.long	.L408-.L340
	.long	.L408-.L340
	.long	.L408-.L340
	.long	.L370-.L340
	.long	.L408-.L340
	.long	.L369-.L340
	.long	.L408-.L340
	.long	.L368-.L340
	.long	.L367-.L340
	.long	.L408-.L340
	.long	.L366-.L340
	.long	.L365-.L340
	.long	.L408-.L340
	.long	.L408-.L340
	.long	.L364-.L340
	.long	.L363-.L340
	.long	.L362-.L340
	.long	.L361-.L340
	.long	.L360-.L340
	.long	.L359-.L340
	.long	.L358-.L340
	.long	.L357-.L340
	.long	.L408-.L340
	.long	.L408-.L340
	.long	.L356-.L340
	.long	.L408-.L340
	.long	.L355-.L340
	.long	.L408-.L340
	.long	.L354-.L340
	.long	.L408-.L340
	.long	.L353-.L340
	.long	.L352-.L340
	.long	.L408-.L340
	.long	.L351-.L340
	.long	.L350-.L340
	.long	.L408-.L340
	.long	.L349-.L340
	.long	.L408-.L340
	.long	.L408-.L340
	.long	.L408-.L340
	.long	.L348-.L340
	.long	.L408-.L340
	.long	.L347-.L340
	.long	.L346-.L340
	.long	.L345-.L340
	.long	.L344-.L340
	.long	.L343-.L340
	.long	.L342-.L340
	.long	.L341-.L340
	.long	.L339-.L340
	.text
.L345:
	movq	-128(%rbp), %rax
	andl	$3, %eax
	testq	%rax, %rax
	je	.L377
	movq	$14, -64(%rbp)
	jmp	.L379
.L377:
	movq	$52, -64(%rbp)
	jmp	.L379
.L359:
	movq	-160(%rbp), %rdx
	movq	-168(%rbp), %rax
	addq	%rdx, %rax
	movq	-152(%rbp), %rdx
	movb	%dl, (%rax)
	addq	$1, -168(%rbp)
	movq	$21, -64(%rbp)
	jmp	.L379
.L346:
	movq	-80(%rbp), %rax
	cmpq	-104(%rbp), %rax
	jnb	.L380
	movq	$54, -64(%rbp)
	jmp	.L379
.L380:
	movq	$30, -64(%rbp)
	jmp	.L379
.L343:
	movl	$0, -172(%rbp)
	movq	$4, -64(%rbp)
	jmp	.L379
.L372:
	movq	-128(%rbp), %rax
	shrq	$3, %rax
	movq	%rax, %rdx
	movl	-172(%rbp), %eax
	cltq
	addq	%rdx, %rax
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rdx
	movq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC19(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-192(%rbp), %rcx
	movq	-160(%rbp), %rax
	movl	$0, %edx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	LZSS_GetVars
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -120(%rbp)
	movq	-120(%rbp), %rax
	salq	$3, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -112(%rbp)
	movq	$27, -64(%rbp)
	jmp	.L379
.L356:
	movq	-160(%rbp), %rdx
	movq	-88(%rbp), %rax
	addq	%rax, %rdx
	movzbl	-174(%rbp), %eax
	movb	%al, (%rdx)
	movq	$39, -64(%rbp)
	jmp	.L379
.L367:
	movl	$1, -172(%rbp)
	movq	$4, -64(%rbp)
	jmp	.L379
.L341:
	movq	-80(%rbp), %rax
	movq	%rax, -104(%rbp)
	movq	$39, -64(%rbp)
	jmp	.L379
.L375:
	movq	$-1, %rax
	jmp	.L382
.L361:
	movq	-168(%rbp), %rdx
	movq	-136(%rbp), %rax
	addq	%rdx, %rax
	cmpq	%rax, -192(%rbp)
	jbe	.L383
	movq	$36, -64(%rbp)
	jmp	.L379
.L383:
	movq	$0, -64(%rbp)
	jmp	.L379
.L373:
	cmpq	$0, -144(%rbp)
	js	.L385
	movq	$24, -64(%rbp)
	jmp	.L379
.L385:
	movq	$25, -64(%rbp)
	jmp	.L379
.L366:
	movq	$-1, %rax
	jmp	.L382
.L360:
	movq	-168(%rbp), %rax
	subq	$2048, %rax
	cmpq	%rax, -144(%rbp)
	jl	.L387
	movq	$42, -64(%rbp)
	jmp	.L379
.L387:
	movq	$25, -64(%rbp)
	jmp	.L379
.L363:
	movq	-168(%rbp), %rax
	cmpq	-192(%rbp), %rax
	jnb	.L389
	movq	$55, -64(%rbp)
	jmp	.L379
.L389:
	movq	$48, -64(%rbp)
	jmp	.L379
.L353:
	movq	-144(%rbp), %rdx
	movq	-136(%rbp), %rax
	addq	%rax, %rdx
	movq	-184(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %edx
	movq	-168(%rbp), %rcx
	movq	-136(%rbp), %rax
	addq	%rax, %rcx
	movq	-184(%rbp), %rax
	addq	%rcx, %rax
	movzbl	(%rax), %eax
	cmpb	%al, %dl
	jne	.L391
	movq	$11, -64(%rbp)
	jmp	.L379
.L391:
	movq	$0, -64(%rbp)
	jmp	.L379
.L358:
	subq	$1, -144(%rbp)
	movq	$3, -64(%rbp)
	jmp	.L379
.L369:
	addq	$1, -136(%rbp)
	movq	$40, -64(%rbp)
	jmp	.L379
.L370:
	movq	$32, -64(%rbp)
	jmp	.L379
.L368:
	movq	$0, -136(%rbp)
	movq	$11, -64(%rbp)
	jmp	.L379
.L344:
	movq	-112(%rbp), %rdx
	movq	-192(%rbp), %rcx
	movq	-160(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	LZSS_GetVars
	movq	-128(%rbp), %rax
	movq	%rax, -104(%rbp)
	movq	$0, -96(%rbp)
	movq	$37, -64(%rbp)
	jmp	.L379
.L355:
	movq	$0, -168(%rbp)
	movq	-192(%rbp), %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -48(%rbp)
	movq	-48(%rbp), %rax
	movq	%rax, -160(%rbp)
	movq	$17, -64(%rbp)
	jmp	.L379
.L365:
	cmpq	$0, -160(%rbp)
	jne	.L393
	movq	$46, -64(%rbp)
	jmp	.L379
.L393:
	movq	$21, -64(%rbp)
	jmp	.L379
.L350:
	cmpq	$17, -136(%rbp)
	ja	.L395
	movq	$23, -64(%rbp)
	jmp	.L379
.L395:
	movq	$0, -64(%rbp)
	jmp	.L379
.L339:
	movq	$1, -152(%rbp)
	movq	-168(%rbp), %rax
	subq	$1, %rax
	movq	%rax, -144(%rbp)
	movq	$3, -64(%rbp)
	jmp	.L379
.L357:
	cmpq	$0, -112(%rbp)
	jne	.L397
	movq	$20, -64(%rbp)
	jmp	.L379
.L397:
	movq	$51, -64(%rbp)
	jmp	.L379
.L354:
	movq	-72(%rbp), %rax
	jmp	.L382
.L347:
	movq	-192(%rbp), %rdx
	movq	-160(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	LZSS_CalcLength
	movq	%rax, -56(%rbp)
	movq	-56(%rbp), %rax
	movq	%rax, -128(%rbp)
	movq	$50, -64(%rbp)
	jmp	.L379
.L362:
	movq	-120(%rbp), %rax
	subq	-96(%rbp), %rax
	subq	$2, %rax
	leaq	0(,%rax,8), %rdx
	movq	-112(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movq	%rax, -88(%rbp)
	movq	-120(%rbp), %rax
	subq	-96(%rbp), %rax
	subq	$1, %rax
	leaq	0(,%rax,8), %rdx
	movq	-112(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movb	%al, -173(%rbp)
	movq	-160(%rbp), %rdx
	movq	-88(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	movb	%al, -174(%rbp)
	movq	-160(%rbp), %rdx
	movq	-88(%rbp), %rax
	addq	%rax, %rdx
	movzbl	-173(%rbp), %eax
	movb	%al, (%rdx)
	movq	-192(%rbp), %rdx
	movq	-160(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	LZSS_CalcLength
	movq	%rax, -40(%rbp)
	movq	-40(%rbp), %rax
	movq	%rax, -80(%rbp)
	movq	$49, -64(%rbp)
	jmp	.L379
.L342:
	cmpq	$17, -152(%rbp)
	jbe	.L399
	movq	$25, -64(%rbp)
	jmp	.L379
.L399:
	movq	$26, -64(%rbp)
	jmp	.L379
.L371:
	movq	-160(%rbp), %rcx
	movq	-200(%rbp), %rdx
	movq	-192(%rbp), %rsi
	movq	-184(%rbp), %rax
	movq	%rax, %rdi
	call	LZSS_Compress
	movq	%rax, -32(%rbp)
	movq	-32(%rbp), %rax
	movq	%rax, -72(%rbp)
	movq	-112(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	-160(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$34, -64(%rbp)
	jmp	.L379
.L352:
	movq	-96(%rbp), %rax
	cmpq	-120(%rbp), %rax
	jnb	.L401
	movq	$22, -64(%rbp)
	jmp	.L379
.L401:
	movq	$5, -64(%rbp)
	jmp	.L379
.L349:
	movq	-144(%rbp), %rdx
	movq	-184(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %edx
	movq	-184(%rbp), %rcx
	movq	-168(%rbp), %rax
	addq	%rcx, %rax
	movzbl	(%rax), %eax
	cmpb	%al, %dl
	jne	.L403
	movq	$13, -64(%rbp)
	jmp	.L379
.L403:
	movq	$26, -64(%rbp)
	jmp	.L379
.L376:
	movq	-136(%rbp), %rax
	cmpq	-152(%rbp), %rax
	jbe	.L405
	movq	$2, -64(%rbp)
	jmp	.L379
.L405:
	movq	$26, -64(%rbp)
	jmp	.L379
.L348:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$15, %edx
	movl	$1, %esi
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	$1, -64(%rbp)
	jmp	.L379
.L351:
	addq	$2, -96(%rbp)
	movq	$37, -64(%rbp)
	jmp	.L379
.L374:
	movq	-136(%rbp), %rax
	movq	%rax, -152(%rbp)
	movq	$53, -64(%rbp)
	jmp	.L379
.L364:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$15, %edx
	movl	$1, %esi
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	$16, -64(%rbp)
	jmp	.L379
.L408:
	nop
.L379:
	jmp	.L407
.L382:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE12:
	.size	LZSS_CompressUltra, .-LZSS_CompressUltra
	.section	.rodata
.LC20:
	.string	"Unknown option: %s\n"
.LC21:
	.string	"Unknown argument: %s\n"
	.text
	.globl	argparse
	.type	argparse, @function
argparse:
.LFB13:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$80, %rsp
	movl	%edi, -52(%rbp)
	movq	%rsi, -64(%rbp)
	movq	%rdx, -72(%rbp)
	movq	$19, -16(%rbp)
.L456:
	cmpq	$35, -16(%rbp)
	ja	.L457
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L412(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L412(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L412:
	.long	.L434-.L412
	.long	.L433-.L412
	.long	.L432-.L412
	.long	.L457-.L412
	.long	.L457-.L412
	.long	.L431-.L412
	.long	.L430-.L412
	.long	.L429-.L412
	.long	.L458-.L412
	.long	.L427-.L412
	.long	.L457-.L412
	.long	.L457-.L412
	.long	.L426-.L412
	.long	.L457-.L412
	.long	.L425-.L412
	.long	.L457-.L412
	.long	.L457-.L412
	.long	.L424-.L412
	.long	.L423-.L412
	.long	.L422-.L412
	.long	.L421-.L412
	.long	.L420-.L412
	.long	.L457-.L412
	.long	.L419-.L412
	.long	.L418-.L412
	.long	.L417-.L412
	.long	.L416-.L412
	.long	.L457-.L412
	.long	.L457-.L412
	.long	.L415-.L412
	.long	.L414-.L412
	.long	.L413-.L412
	.long	.L457-.L412
	.long	.L457-.L412
	.long	.L457-.L412
	.long	.L411-.L412
	.text
.L423:
	cmpq	$0, -32(%rbp)
	je	.L435
	cmpq	$1, -32(%rbp)
	jne	.L436
	movq	$9, -16(%rbp)
	jmp	.L437
.L435:
	movq	$26, -16(%rbp)
	jmp	.L437
.L436:
	movq	$21, -16(%rbp)
	nop
.L437:
	jmp	.L438
.L417:
	movl	-40(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-64(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rdx
	movq	stderr(%rip), %rax
	leaq	.LC20(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	$5, -16(%rbp)
	jmp	.L438
.L414:
	movq	-72(%rbp), %rax
	movl	$1, 16(%rax)
	movq	$5, -16(%rbp)
	jmp	.L438
.L425:
	movl	-40(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-64(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	leaq	1(%rax), %rdx
	movq	-24(%rbp), %rax
	leaq	0(,%rax,8), %rcx
	leaq	arguments(%rip), %rax
	movq	(%rcx,%rax), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -36(%rbp)
	movq	$6, -16(%rbp)
	jmp	.L438
.L413:
	movq	$0, -24(%rbp)
	movq	$17, -16(%rbp)
	jmp	.L438
.L426:
	cmpq	$3, -24(%rbp)
	je	.L439
	cmpq	$3, -24(%rbp)
	ja	.L440
	cmpq	$2, -24(%rbp)
	je	.L441
	cmpq	$2, -24(%rbp)
	ja	.L440
	cmpq	$0, -24(%rbp)
	je	.L442
	cmpq	$1, -24(%rbp)
	je	.L443
	jmp	.L440
.L439:
	movq	$29, -16(%rbp)
	jmp	.L444
.L441:
	movq	$7, -16(%rbp)
	jmp	.L444
.L443:
	movq	$30, -16(%rbp)
	jmp	.L444
.L442:
	movq	$20, -16(%rbp)
	jmp	.L444
.L440:
	movq	$25, -16(%rbp)
	nop
.L444:
	jmp	.L438
.L433:
	addq	$1, -24(%rbp)
	movq	$17, -16(%rbp)
	jmp	.L438
.L419:
	movl	-40(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-64(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movzbl	(%rax), %eax
	cmpb	$45, %al
	jne	.L446
	movq	$31, -16(%rbp)
	jmp	.L438
.L446:
	movq	$18, -16(%rbp)
	jmp	.L438
.L418:
	movq	-72(%rbp), %rax
	movq	$0, (%rax)
	movq	-72(%rbp), %rax
	movq	$0, 8(%rax)
	movq	-72(%rbp), %rax
	movl	$0, 16(%rax)
	movq	-72(%rbp), %rax
	movl	$0, 20(%rax)
	movq	-72(%rbp), %rax
	movb	$1, 24(%rax)
	movq	$0, -32(%rbp)
	movl	$1, -40(%rbp)
	movq	$2, -16(%rbp)
	jmp	.L438
.L420:
	movl	-40(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-64(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rdx
	movq	stderr(%rip), %rax
	leaq	.LC21(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	$35, -16(%rbp)
	jmp	.L438
.L416:
	movl	-40(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-64(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rdx
	movq	-72(%rbp), %rax
	movq	%rdx, (%rax)
	movq	$35, -16(%rbp)
	jmp	.L438
.L427:
	movl	-40(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-64(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rdx
	movq	-72(%rbp), %rax
	movq	%rdx, 8(%rax)
	movq	$35, -16(%rbp)
	jmp	.L438
.L422:
	movq	$24, -16(%rbp)
	jmp	.L438
.L424:
	movq	-24(%rbp), %rax
	leaq	0(,%rax,8), %rdx
	leaq	arguments(%rip), %rax
	movq	(%rdx,%rax), %rax
	testq	%rax, %rax
	je	.L448
	movq	$14, -16(%rbp)
	jmp	.L438
.L448:
	movq	$12, -16(%rbp)
	jmp	.L438
.L430:
	cmpl	$0, -36(%rbp)
	je	.L450
	movq	$1, -16(%rbp)
	jmp	.L438
.L450:
	movq	$12, -16(%rbp)
	jmp	.L438
.L431:
	addl	$1, -40(%rbp)
	movq	$2, -16(%rbp)
	jmp	.L438
.L434:
	movl	-40(%rbp), %eax
	cltq
	addq	$1, %rax
	leaq	0(,%rax,8), %rdx
	movq	-64(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	Parse_ulong
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movl	%eax, %edx
	movq	-72(%rbp), %rax
	movl	%edx, 20(%rax)
	addl	$1, -40(%rbp)
	movq	$5, -16(%rbp)
	jmp	.L438
.L429:
	movl	-40(%rbp), %eax
	addl	$1, %eax
	cmpl	%eax, -52(%rbp)
	jle	.L452
	movq	$0, -16(%rbp)
	jmp	.L438
.L452:
	movq	$5, -16(%rbp)
	jmp	.L438
.L411:
	addq	$1, -32(%rbp)
	movq	$5, -16(%rbp)
	jmp	.L438
.L415:
	movq	-72(%rbp), %rax
	movb	$0, 24(%rax)
	movq	$5, -16(%rbp)
	jmp	.L438
.L432:
	movl	-40(%rbp), %eax
	cmpl	-52(%rbp), %eax
	jge	.L454
	movq	$23, -16(%rbp)
	jmp	.L438
.L454:
	movq	$8, -16(%rbp)
	jmp	.L438
.L421:
	movq	-72(%rbp), %rax
	movl	$0, 16(%rax)
	movq	$5, -16(%rbp)
	jmp	.L438
.L457:
	nop
.L438:
	jmp	.L456
.L458:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE13:
	.size	argparse, .-argparse
	.globl	Read_u32be
	.type	Read_u32be, @function
Read_u32be:
.LFB15:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movq	$0, -8(%rbp)
.L462:
	cmpq	$0, -8(%rbp)
	jne	.L465
	movq	-24(%rbp), %rax
	movzbl	(%rax), %eax
	movzbl	%al, %eax
	sall	$24, %eax
	movl	%eax, %edx
	movq	-24(%rbp), %rax
	addq	$1, %rax
	movzbl	(%rax), %eax
	movzbl	%al, %eax
	sall	$16, %eax
	orl	%eax, %edx
	movq	-24(%rbp), %rax
	addq	$2, %rax
	movzbl	(%rax), %eax
	movzbl	%al, %eax
	sall	$8, %eax
	orl	%eax, %edx
	movq	-24(%rbp), %rax
	addq	$3, %rax
	movzbl	(%rax), %eax
	movzbl	%al, %eax
	orl	%edx, %eax
	jmp	.L464
.L465:
	nop
	jmp	.L462
.L464:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE15:
	.size	Read_u32be, .-Read_u32be
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
