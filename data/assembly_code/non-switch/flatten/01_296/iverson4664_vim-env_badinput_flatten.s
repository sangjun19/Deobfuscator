	.file	"iverson4664_vim-env_badinput_flatten.c"
	.text
	.globl	_TIG_IZ_BEKB_argc
	.bss
	.align 4
	.type	_TIG_IZ_BEKB_argc, @object
	.size	_TIG_IZ_BEKB_argc, 4
_TIG_IZ_BEKB_argc:
	.zero	4
	.globl	_TIG_IZ_BEKB_envp
	.align 8
	.type	_TIG_IZ_BEKB_envp, @object
	.size	_TIG_IZ_BEKB_envp, 8
_TIG_IZ_BEKB_envp:
	.zero	8
	.globl	_TIG_IZ_BEKB_argv
	.align 8
	.type	_TIG_IZ_BEKB_argv, @object
	.size	_TIG_IZ_BEKB_argv, 8
_TIG_IZ_BEKB_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"Usage:\n"
.LC1:
	.string	"\t%s --help|-h\n"
	.align 8
.LC2:
	.string	"\t%s CMDLINE_TEMPLATE INPUT OUTPUT\n"
	.text
	.type	print_help, @function
print_help:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movl	%edx, -36(%rbp)
	movq	$1, -8(%rbp)
.L5:
	cmpq	$0, -8(%rbp)
	je	.L2
	cmpq	$1, -8(%rbp)
	jne	.L6
	movq	$0, -8(%rbp)
	jmp	.L4
.L2:
	movq	-32(%rbp), %rax
	movq	%rax, %rcx
	movl	$7, %edx
	movl	$1, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	-24(%rbp), %rdx
	movq	-32(%rbp), %rax
	leaq	.LC1(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	-24(%rbp), %rdx
	movq	-32(%rbp), %rax
	leaq	.LC2(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movl	-36(%rbp), %eax
	movl	%eax, %edi
	call	exit@PLT
.L6:
	nop
.L4:
	jmp	.L5
	.cfi_endproc
.LFE0:
	.size	print_help, .-print_help
	.section	.rodata
.LC3:
	.string	"step(end): %d "
.LC4:
	.string	"step(start): %d "
.LC5:
	.string	"Minimal bad input:\n"
	.text
	.type	bisect, @function
bisect:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$96, %rsp
	movq	%rdi, -72(%rbp)
	movq	%rsi, -80(%rbp)
	movq	%rdx, -88(%rbp)
	movl	%ecx, -92(%rbp)
	movq	$12, -8(%rbp)
.L59:
	cmpq	$40, -8(%rbp)
	ja	.L61
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L10(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L10(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L10:
	.long	.L40-.L10
	.long	.L39-.L10
	.long	.L38-.L10
	.long	.L37-.L10
	.long	.L36-.L10
	.long	.L61-.L10
	.long	.L61-.L10
	.long	.L35-.L10
	.long	.L34-.L10
	.long	.L33-.L10
	.long	.L32-.L10
	.long	.L31-.L10
	.long	.L30-.L10
	.long	.L29-.L10
	.long	.L28-.L10
	.long	.L61-.L10
	.long	.L27-.L10
	.long	.L26-.L10
	.long	.L25-.L10
	.long	.L24-.L10
	.long	.L23-.L10
	.long	.L22-.L10
	.long	.L61-.L10
	.long	.L21-.L10
	.long	.L20-.L10
	.long	.L19-.L10
	.long	.L61-.L10
	.long	.L61-.L10
	.long	.L18-.L10
	.long	.L61-.L10
	.long	.L17-.L10
	.long	.L16-.L10
	.long	.L15-.L10
	.long	.L14-.L10
	.long	.L61-.L10
	.long	.L13-.L10
	.long	.L61-.L10
	.long	.L12-.L10
	.long	.L11-.L10
	.long	.L61-.L10
	.long	.L9-.L10
	.text
.L25:
	movl	-52(%rbp), %eax
	cltq
	addq	%rax, -32(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L41
.L19:
	movq	-32(%rbp), %rax
	movq	%rax, -16(%rbp)
	movq	$9, -8(%rbp)
	jmp	.L41
.L36:
	movq	stderr(%rip), %rax
	movl	-56(%rbp), %edx
	leaq	.LC3(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movl	-56(%rbp), %eax
	addl	$1, %eax
	movq	-88(%rbp), %rdx
	movl	%eax, %ecx
	shrq	%cl, %rdx
	movq	%rdx, %rax
	movl	%eax, -52(%rbp)
	movq	$31, -8(%rbp)
	jmp	.L41
.L17:
	cmpl	$0, -44(%rbp)
	jne	.L42
	movq	$25, -8(%rbp)
	jmp	.L41
.L42:
	movq	$7, -8(%rbp)
	jmp	.L41
.L28:
	movq	-88(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	$0, -16(%rbp)
	movq	-88(%rbp), %rax
	movq	%rax, -40(%rbp)
	movq	-88(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	$0, -16(%rbp)
	movl	$0, -56(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L41
.L16:
	cmpl	$0, -52(%rbp)
	jne	.L44
	movq	$1, -8(%rbp)
	jmp	.L41
.L44:
	movq	$17, -8(%rbp)
	jmp	.L41
.L30:
	movq	$14, -8(%rbp)
	jmp	.L41
.L34:
	movq	stderr(%rip), %rax
	movl	-56(%rbp), %edx
	leaq	.LC4(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movl	-56(%rbp), %eax
	addl	$1, %eax
	movq	-88(%rbp), %rdx
	movl	%eax, %ecx
	shrq	%cl, %rdx
	movq	%rdx, %rax
	movl	%eax, -52(%rbp)
	movq	$13, -8(%rbp)
	jmp	.L41
.L39:
	movl	$1, -52(%rbp)
	movq	$17, -8(%rbp)
	jmp	.L41
.L21:
	movq	-40(%rbp), %rax
	addq	$1, %rax
	cmpq	%rax, -24(%rbp)
	jne	.L46
	movq	$38, -8(%rbp)
	jmp	.L41
.L46:
	movq	$21, -8(%rbp)
	jmp	.L41
.L37:
	cmpl	$0, -48(%rbp)
	jne	.L48
	movq	$28, -8(%rbp)
	jmp	.L41
.L48:
	movq	$19, -8(%rbp)
	jmp	.L41
.L27:
	movq	-16(%rbp), %rax
	subq	$1, %rax
	cmpq	%rax, -32(%rbp)
	jne	.L50
	movq	$33, -8(%rbp)
	jmp	.L41
.L50:
	movq	$18, -8(%rbp)
	jmp	.L41
.L20:
	subq	$1, -32(%rbp)
	movq	$33, -8(%rbp)
	jmp	.L41
.L22:
	movl	-52(%rbp), %eax
	cltq
	addq	%rax, -40(%rbp)
	movq	$10, -8(%rbp)
	jmp	.L41
.L31:
	movl	-52(%rbp), %eax
	cltq
	subq	%rax, -32(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L41
.L33:
	movq	-32(%rbp), %rax
	subq	$1, %rax
	cmpq	%rax, -24(%rbp)
	jne	.L52
	movq	$24, -8(%rbp)
	jmp	.L41
.L52:
	movq	$11, -8(%rbp)
	jmp	.L41
.L29:
	cmpl	$0, -52(%rbp)
	jne	.L54
	movq	$40, -8(%rbp)
	jmp	.L41
.L54:
	movq	$0, -8(%rbp)
	jmp	.L41
.L24:
	movq	-40(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	$20, -8(%rbp)
	jmp	.L41
.L15:
	movl	-52(%rbp), %eax
	cltq
	subq	%rax, -40(%rbp)
	movq	$10, -8(%rbp)
	jmp	.L41
.L26:
	movq	-40(%rbp), %rdx
	movl	-92(%rbp), %ecx
	movq	-80(%rbp), %rsi
	movq	-72(%rbp), %rax
	movl	%ecx, %r8d
	movq	%rdx, %rcx
	movl	$0, %edx
	movq	%rax, %rdi
	call	test
	movl	%eax, -48(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L41
.L9:
	movl	$1, -52(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L41
.L11:
	movq	-24(%rbp), %rax
	movq	%rax, -40(%rbp)
	movq	$35, -8(%rbp)
	jmp	.L41
.L18:
	movq	-40(%rbp), %rax
	movq	%rax, -16(%rbp)
	movq	$23, -8(%rbp)
	jmp	.L41
.L14:
	movq	-40(%rbp), %rax
	subq	-32(%rbp), %rax
	movq	%rax, -88(%rbp)
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$19, %edx
	movl	$1, %esi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	stdout(%rip), %rdx
	movq	-32(%rbp), %rcx
	movq	-80(%rbp), %rax
	leaq	(%rcx,%rax), %rdi
	movq	-88(%rbp), %rax
	movq	%rdx, %rcx
	movq	%rax, %rdx
	movl	$1, %esi
	call	fwrite@PLT
	movq	-32(%rbp), %rdx
	movq	-80(%rbp), %rax
	leaq	(%rdx,%rax), %rcx
	movq	-88(%rbp), %rdx
	movl	-92(%rbp), %eax
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	prepare
	movl	$10, %edi
	call	putchar@PLT
	movq	$37, -8(%rbp)
	jmp	.L41
.L12:
	movl	$0, %eax
	jmp	.L60
.L32:
	addl	$1, -56(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L41
.L40:
	movq	-40(%rbp), %rax
	subq	-32(%rbp), %rax
	movq	%rax, %rdi
	movl	-92(%rbp), %ecx
	movq	-32(%rbp), %rdx
	movq	-80(%rbp), %rsi
	movq	-72(%rbp), %rax
	movl	%ecx, %r8d
	movq	%rdi, %rcx
	movq	%rax, %rdi
	call	test
	movl	%eax, -44(%rbp)
	movq	$30, -8(%rbp)
	jmp	.L41
.L35:
	movq	-32(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	$16, -8(%rbp)
	jmp	.L41
.L13:
	movq	-40(%rbp), %rax
	movq	%rax, -88(%rbp)
	movq	$0, -32(%rbp)
	movq	$0, -24(%rbp)
	movq	-40(%rbp), %rax
	movq	%rax, -16(%rbp)
	movl	$0, -56(%rbp)
	movq	$8, -8(%rbp)
	jmp	.L41
.L38:
	addl	$1, -56(%rbp)
	movq	$8, -8(%rbp)
	jmp	.L41
.L23:
	movq	-16(%rbp), %rax
	addq	$1, %rax
	cmpq	%rax, -40(%rbp)
	jne	.L57
	movq	$35, -8(%rbp)
	jmp	.L41
.L57:
	movq	$32, -8(%rbp)
	jmp	.L41
.L61:
	nop
.L41:
	jmp	.L59
.L60:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	bisect, .-bisect
	.section	.rodata
.LC6:
	.string	"[%lu, %lu]..."
.LC7:
	.string	"%d\n"
	.text
	.type	test, @function
test:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movq	%rcx, -48(%rbp)
	movl	%r8d, -52(%rbp)
	movq	$2, -8(%rbp)
.L68:
	cmpq	$2, -8(%rbp)
	je	.L63
	cmpq	$2, -8(%rbp)
	ja	.L70
	cmpq	$0, -8(%rbp)
	je	.L65
	cmpq	$1, -8(%rbp)
	jne	.L70
	movl	-12(%rbp), %eax
	jmp	.L69
.L65:
	movq	-40(%rbp), %rdx
	movq	-32(%rbp), %rax
	leaq	(%rdx,%rax), %rcx
	movq	-48(%rbp), %rdx
	movl	-52(%rbp), %eax
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	prepare
	movq	-40(%rbp), %rdx
	movq	-48(%rbp), %rax
	leaq	(%rdx,%rax), %rcx
	movq	stderr(%rip), %rax
	movq	-40(%rbp), %rdx
	leaq	.LC6(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	system@PLT
	movl	%eax, -12(%rbp)
	movq	stderr(%rip), %rax
	movl	-12(%rbp), %edx
	leaq	.LC7(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	$1, -8(%rbp)
	jmp	.L67
.L63:
	movq	$0, -8(%rbp)
	jmp	.L67
.L70:
	nop
.L67:
	jmp	.L68
.L69:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	test, .-test
	.section	.rodata
.LC8:
	.string	"truncate"
.LC9:
	.string	"write"
.LC10:
	.string	"lseek"
	.text
	.type	prepare, @function
prepare:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	$7, -8(%rbp)
.L92:
	cmpq	$12, -8(%rbp)
	ja	.L93
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L74(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L74(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L74:
	.long	.L83-.L74
	.long	.L82-.L74
	.long	.L93-.L74
	.long	.L93-.L74
	.long	.L81-.L74
	.long	.L80-.L74
	.long	.L79-.L74
	.long	.L78-.L74
	.long	.L77-.L74
	.long	.L76-.L74
	.long	.L93-.L74
	.long	.L75-.L74
	.long	.L94-.L74
	.text
.L81:
	movq	-56(%rbp), %rax
	cmpq	%rax, -16(%rbp)
	je	.L84
	movq	$11, -8(%rbp)
	jmp	.L86
.L84:
	movq	$12, -8(%rbp)
	jmp	.L86
.L77:
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L82:
	cmpl	$-1, -28(%rbp)
	jne	.L88
	movq	$8, -8(%rbp)
	jmp	.L86
.L88:
	movq	$5, -8(%rbp)
	jmp	.L86
.L75:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L76:
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L79:
	movl	-36(%rbp), %eax
	movl	$0, %esi
	movl	%eax, %edi
	call	ftruncate@PLT
	movl	%eax, -28(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L86
.L80:
	movq	-56(%rbp), %rdx
	movq	-48(%rbp), %rcx
	movl	-36(%rbp), %eax
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	write@PLT
	movq	%rax, -16(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L86
.L83:
	cmpq	$0, -24(%rbp)
	je	.L90
	movq	$9, -8(%rbp)
	jmp	.L86
.L90:
	movq	$6, -8(%rbp)
	jmp	.L86
.L78:
	movl	-36(%rbp), %eax
	movl	$0, %edx
	movl	$0, %esi
	movl	%eax, %edi
	call	lseek@PLT
	movq	%rax, -24(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L86
.L93:
	nop
.L86:
	jmp	.L92
.L94:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	prepare, .-prepare
	.section	.rodata
	.align 8
.LC11:
	.string	"no %%s is found in command line template\n"
.LC12:
	.string	"--help"
	.align 8
.LC13:
	.string	"the target command line exits normally against the original input\n"
.LC14:
	.string	"-h"
.LC15:
	.string	"open(output)"
.LC16:
	.string	"wrong number of arguments\n"
.LC17:
	.string	"error in asprintf\n"
.LC18:
	.string	"%s"
	.align 8
.LC19:
	.string	"the target command line exits normally against the empty input\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB7:
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
	movq	$0, _TIG_IZ_BEKB_envp(%rip)
	nop
.L96:
	movq	$0, _TIG_IZ_BEKB_argv(%rip)
	nop
.L97:
	movl	$0, _TIG_IZ_BEKB_argc(%rip)
	nop
	nop
.L98:
.L99:
#APP
# 225 "iverson4664_vim-env_badinput.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-BEKB--0
# 0 "" 2
#NO_APP
	movl	-116(%rbp), %eax
	movl	%eax, _TIG_IZ_BEKB_argc(%rip)
	movq	-128(%rbp), %rax
	movq	%rax, _TIG_IZ_BEKB_argv(%rip)
	movq	-136(%rbp), %rax
	movq	%rax, _TIG_IZ_BEKB_envp(%rip)
	nop
	movq	$7, -16(%rbp)
.L148:
	cmpq	$34, -16(%rbp)
	ja	.L151
	movq	-16(%rbp), %rax
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
	.long	.L127-.L102
	.long	.L126-.L102
	.long	.L151-.L102
	.long	.L151-.L102
	.long	.L151-.L102
	.long	.L151-.L102
	.long	.L125-.L102
	.long	.L124-.L102
	.long	.L123-.L102
	.long	.L122-.L102
	.long	.L121-.L102
	.long	.L120-.L102
	.long	.L119-.L102
	.long	.L118-.L102
	.long	.L117-.L102
	.long	.L116-.L102
	.long	.L115-.L102
	.long	.L114-.L102
	.long	.L151-.L102
	.long	.L113-.L102
	.long	.L151-.L102
	.long	.L112-.L102
	.long	.L111-.L102
	.long	.L110-.L102
	.long	.L109-.L102
	.long	.L108-.L102
	.long	.L107-.L102
	.long	.L106-.L102
	.long	.L105-.L102
	.long	.L104-.L102
	.long	.L151-.L102
	.long	.L151-.L102
	.long	.L103-.L102
	.long	.L151-.L102
	.long	.L101-.L102
	.text
.L108:
	cmpl	$0, -96(%rbp)
	je	.L128
	movq	$26, -16(%rbp)
	jmp	.L130
.L128:
	movq	$16, -16(%rbp)
	jmp	.L130
.L117:
	movq	stderr(%rip), %rax
	leaq	.LC11(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movl	$1, %edi
	call	exit@PLT
.L116:
	movq	-128(%rbp), %rax
	addq	$16, %rax
	movq	(%rax), %rax
	leaq	.LC12(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -96(%rbp)
	movq	$25, -16(%rbp)
	jmp	.L130
.L119:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$66, %edx
	movl	$1, %esi
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movl	$1, %edi
	call	exit@PLT
.L123:
	cmpl	$0, -84(%rbp)
	jne	.L131
	movq	$12, -16(%rbp)
	jmp	.L130
.L131:
	movq	$0, -16(%rbp)
	jmp	.L130
.L126:
	cmpl	$0, -92(%rbp)
	je	.L133
	movq	$29, -16(%rbp)
	jmp	.L130
.L133:
	movq	$6, -16(%rbp)
	jmp	.L130
.L110:
	cmpl	$-1, -88(%rbp)
	jne	.L135
	movq	$17, -16(%rbp)
	jmp	.L130
.L135:
	movq	$11, -16(%rbp)
	jmp	.L130
.L115:
	movq	stdout(%rip), %rcx
	movq	-128(%rbp), %rax
	movq	(%rax), %rax
	movl	$0, %edx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	print_help
	movq	$34, -16(%rbp)
	jmp	.L130
.L109:
	movq	-32(%rbp), %rdx
	movq	-48(%rbp), %rcx
	leaq	-72(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	asprintf@PLT
	movl	%eax, -88(%rbp)
	movq	$23, -16(%rbp)
	jmp	.L130
.L112:
	leaq	-56(%rbp), %rdx
	leaq	-64(%rbp), %rcx
	movq	-40(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	load
	movq	-32(%rbp), %rax
	movl	$438, %edx
	movl	$65, %esi
	movq	%rax, %rdi
	movl	$0, %eax
	call	open@PLT
	movl	%eax, -100(%rbp)
	movq	$22, -16(%rbp)
	jmp	.L130
.L107:
	movq	-128(%rbp), %rax
	addq	$16, %rax
	movq	(%rax), %rax
	leaq	.LC14(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -92(%rbp)
	movq	$1, -16(%rbp)
	jmp	.L130
.L120:
	movq	-56(%rbp), %rdx
	movq	-64(%rbp), %rsi
	movq	-72(%rbp), %rax
	movl	-100(%rbp), %ecx
	movl	%ecx, %r8d
	movq	%rdx, %rcx
	movl	$0, %edx
	movq	%rax, %rdi
	call	test
	movl	%eax, -84(%rbp)
	movq	$8, -16(%rbp)
	jmp	.L130
.L122:
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L118:
	cmpq	$0, -24(%rbp)
	je	.L137
	movq	$21, -16(%rbp)
	jmp	.L130
.L137:
	movq	$14, -16(%rbp)
	jmp	.L130
.L113:
	movq	-56(%rbp), %rdx
	movq	-64(%rbp), %rsi
	movq	-72(%rbp), %rax
	movl	-100(%rbp), %ecx
	movq	%rax, %rdi
	call	bisect
	movl	%eax, -76(%rbp)
	movq	$28, -16(%rbp)
	jmp	.L130
.L103:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$26, %edx
	movl	$1, %esi
	leaq	.LC16(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movl	$1, %edi
	call	exit@PLT
.L114:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$18, %edx
	movl	$1, %esi
	leaq	.LC17(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movl	$1, %edi
	call	exit@PLT
.L125:
	movq	stdout(%rip), %rcx
	movq	-128(%rbp), %rax
	movq	(%rax), %rax
	movl	$0, %edx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	print_help
	movq	$34, -16(%rbp)
	jmp	.L130
.L106:
	cmpl	$0, -80(%rbp)
	je	.L139
	movq	$10, -16(%rbp)
	jmp	.L130
.L139:
	movq	$19, -16(%rbp)
	jmp	.L130
.L101:
	movq	-128(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, -48(%rbp)
	movq	-128(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, -40(%rbp)
	movq	-128(%rbp), %rax
	movq	24(%rax), %rax
	movq	%rax, -32(%rbp)
	movq	-48(%rbp), %rax
	leaq	.LC18(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strstr@PLT
	movq	%rax, -24(%rbp)
	movq	$13, -16(%rbp)
	jmp	.L130
.L111:
	cmpl	$0, -100(%rbp)
	jns	.L141
	movq	$9, -16(%rbp)
	jmp	.L130
.L141:
	movq	$24, -16(%rbp)
	jmp	.L130
.L105:
	movl	-76(%rbp), %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L149
	jmp	.L150
.L121:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$63, %edx
	movl	$1, %esi
	leaq	.LC19(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movl	$1, %edi
	call	exit@PLT
.L127:
	movq	-64(%rbp), %rsi
	movq	-72(%rbp), %rax
	movl	-100(%rbp), %edx
	movl	%edx, %r8d
	movl	$0, %ecx
	movl	$0, %edx
	movq	%rax, %rdi
	call	test
	movl	%eax, -80(%rbp)
	movq	$27, -16(%rbp)
	jmp	.L130
.L124:
	cmpl	$2, -116(%rbp)
	jne	.L144
	movq	$15, -16(%rbp)
	jmp	.L130
.L144:
	movq	$29, -16(%rbp)
	jmp	.L130
.L104:
	cmpl	$4, -116(%rbp)
	je	.L146
	movq	$32, -16(%rbp)
	jmp	.L130
.L146:
	movq	$34, -16(%rbp)
	jmp	.L130
.L151:
	nop
.L130:
	jmp	.L148
.L150:
	call	__stack_chk_fail@PLT
.L149:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	main, .-main
	.section	.rodata
.LC20:
	.string	"fstat"
.LC21:
	.string	"read"
.LC22:
	.string	"memory exhausted\n"
.LC23:
	.string	"open(input)"
	.text
	.type	load, @function
load:
.LFB10:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$224, %rsp
	movq	%rdi, -200(%rbp)
	movq	%rsi, -208(%rbp)
	movq	%rdx, -216(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$8, -176(%rbp)
.L178:
	cmpq	$17, -176(%rbp)
	ja	.L181
	movq	-176(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L155(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L155(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L155:
	.long	.L181-.L155
	.long	.L182-.L155
	.long	.L166-.L155
	.long	.L165-.L155
	.long	.L181-.L155
	.long	.L164-.L155
	.long	.L163-.L155
	.long	.L162-.L155
	.long	.L161-.L155
	.long	.L181-.L155
	.long	.L160-.L155
	.long	.L159-.L155
	.long	.L158-.L155
	.long	.L181-.L155
	.long	.L181-.L155
	.long	.L157-.L155
	.long	.L156-.L155
	.long	.L154-.L155
	.text
.L157:
	movq	-216(%rbp), %rax
	movq	(%rax), %rdx
	movq	-208(%rbp), %rax
	movq	(%rax), %rcx
	movl	-192(%rbp), %eax
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	read@PLT
	movq	%rax, -184(%rbp)
	movq	$12, -176(%rbp)
	jmp	.L168
.L158:
	movq	-216(%rbp), %rax
	movq	(%rax), %rdx
	movq	-184(%rbp), %rax
	cmpq	%rax, %rdx
	je	.L169
	movq	$10, -176(%rbp)
	jmp	.L168
.L169:
	movq	$1, -176(%rbp)
	jmp	.L168
.L161:
	movq	-200(%rbp), %rax
	movl	$0, %esi
	movq	%rax, %rdi
	movl	$0, %eax
	call	open@PLT
	movl	%eax, -192(%rbp)
	movq	$11, -176(%rbp)
	jmp	.L168
.L165:
	movq	-112(%rbp), %rax
	movq	%rax, %rdx
	movq	-216(%rbp), %rax
	movq	%rdx, (%rax)
	movq	-216(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -168(%rbp)
	movq	-208(%rbp), %rax
	movq	-168(%rbp), %rdx
	movq	%rdx, (%rax)
	movq	$5, -176(%rbp)
	jmp	.L168
.L156:
	leaq	-160(%rbp), %rdx
	movl	-192(%rbp), %eax
	movq	%rdx, %rsi
	movl	%eax, %edi
	call	fstat@PLT
	movl	%eax, -188(%rbp)
	movq	$17, -176(%rbp)
	jmp	.L168
.L159:
	cmpl	$0, -192(%rbp)
	jns	.L172
	movq	$2, -176(%rbp)
	jmp	.L168
.L172:
	movq	$16, -176(%rbp)
	jmp	.L168
.L154:
	cmpl	$0, -188(%rbp)
	jns	.L174
	movq	$6, -176(%rbp)
	jmp	.L168
.L174:
	movq	$3, -176(%rbp)
	jmp	.L168
.L163:
	leaq	.LC20(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L164:
	movq	-208(%rbp), %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	jne	.L176
	movq	$7, -176(%rbp)
	jmp	.L168
.L176:
	movq	$15, -176(%rbp)
	jmp	.L168
.L160:
	leaq	.LC21(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L162:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$17, %edx
	movl	$1, %esi
	leaq	.LC22(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movl	$1, %edi
	call	exit@PLT
.L166:
	leaq	.LC23(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L181:
	nop
.L168:
	jmp	.L178
.L182:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L180
	call	__stack_chk_fail@PLT
.L180:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	load, .-load
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
