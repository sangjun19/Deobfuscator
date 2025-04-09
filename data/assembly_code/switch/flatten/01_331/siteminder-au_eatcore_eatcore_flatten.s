	.file	"siteminder-au_eatcore_eatcore_flatten.c"
	.text
	.globl	_TIG_IZ_8agm_argc
	.bss
	.align 4
	.type	_TIG_IZ_8agm_argc, @object
	.size	_TIG_IZ_8agm_argc, 4
_TIG_IZ_8agm_argc:
	.zero	4
	.globl	_TIG_IZ_8agm_envp
	.align 8
	.type	_TIG_IZ_8agm_envp, @object
	.size	_TIG_IZ_8agm_envp, 8
_TIG_IZ_8agm_envp:
	.zero	8
	.globl	_TIG_IZ_8agm_argv
	.align 8
	.type	_TIG_IZ_8agm_argv, @object
	.size	_TIG_IZ_8agm_argv, 8
_TIG_IZ_8agm_argv:
	.zero	8
	.section	.rodata
	.align 8
.LC0:
	.string	"%ld eatcore: Starting with interval=%d, increment=%d, max_increments=%d, random=%d\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	addq	$-128, %rsp
	movl	%edi, -100(%rbp)
	movq	%rsi, -112(%rbp)
	movq	%rdx, -120(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_8agm_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_8agm_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_8agm_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 101 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-8agm--0
# 0 "" 2
#NO_APP
	movl	-100(%rbp), %eax
	movl	%eax, _TIG_IZ_8agm_argc(%rip)
	movq	-112(%rbp), %rax
	movq	%rax, _TIG_IZ_8agm_argv(%rip)
	movq	-120(%rbp), %rax
	movq	%rax, _TIG_IZ_8agm_envp(%rip)
	nop
	movq	$0, -96(%rbp)
.L11:
	cmpq	$2, -96(%rbp)
	je	.L6
	cmpq	$2, -96(%rbp)
	ja	.L14
	cmpq	$0, -96(%rbp)
	je	.L8
	cmpq	$1, -96(%rbp)
	jne	.L14
	movq	-112(%rbp), %rdx
	movl	-100(%rbp), %ecx
	leaq	-80(%rbp), %rax
	movl	%ecx, %esi
	movq	%rax, %rdi
	call	set_defaults
	leaq	-80(%rbp), %rax
	movq	%rax, %rdi
	call	parse_commandline
	movl	$0, %edi
	call	time@PLT
	movq	%rax, -88(%rbp)
	movl	-12(%rbp), %ecx
	movq	-32(%rbp), %rax
	movl	%eax, %r8d
	movq	-56(%rbp), %rax
	movl	%eax, %edi
	movq	-64(%rbp), %rax
	movl	%eax, %esi
	movq	stderr(%rip), %rax
	movq	-88(%rbp), %rdx
	subq	$8, %rsp
	pushq	%rcx
	movl	%r8d, %r9d
	movl	%edi, %r8d
	movl	%esi, %ecx
	leaq	.LC0(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	addq	$16, %rsp
	leaq	-80(%rbp), %rax
	movq	%rax, %rdi
	call	eatcore
	movq	$2, -96(%rbp)
	jmp	.L9
.L8:
	movq	$1, -96(%rbp)
	jmp	.L9
.L6:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L12
	jmp	.L13
.L14:
	nop
.L9:
	jmp	.L11
.L13:
	call	__stack_chk_fail@PLT
.L12:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	main, .-main
	.globl	eatcore
	.type	eatcore, @function
eatcore:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	$0, -8(%rbp)
.L24:
	cmpq	$4, -8(%rbp)
	je	.L16
	cmpq	$4, -8(%rbp)
	ja	.L25
	cmpq	$2, -8(%rbp)
	je	.L18
	cmpq	$2, -8(%rbp)
	ja	.L25
	cmpq	$0, -8(%rbp)
	je	.L19
	cmpq	$1, -8(%rbp)
	je	.L20
	jmp	.L25
.L16:
	movq	-24(%rbp), %rax
	movq	56(%rax), %rdx
	movq	-24(%rbp), %rax
	movq	48(%rax), %rax
	cmpq	%rax, %rdx
	jnb	.L21
	movq	$1, -8(%rbp)
	jmp	.L23
.L21:
	movq	$2, -8(%rbp)
	jmp	.L23
.L20:
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	increment
	movq	$2, -8(%rbp)
	jmp	.L23
.L19:
	movq	$4, -8(%rbp)
	jmp	.L23
.L18:
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	interval
	movq	$4, -8(%rbp)
	jmp	.L23
.L25:
	nop
.L23:
	jmp	.L24
	.cfi_endproc
.LFE1:
	.size	eatcore, .-eatcore
	.section	.rodata
.LC1:
	.string	"Allocation failed, aborting."
	.align 8
.LC2:
	.string	"%ld eatcore: Stomach fed to %zuMB. Touching pages...\n"
	.align 8
.LC3:
	.string	"%ld eatcore: Finished touching pages. Sleeping %d seconds.\n"
	.text
	.globl	increment
	.type	increment, @function
increment:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movq	$7, -24(%rbp)
.L45:
	cmpq	$12, -24(%rbp)
	ja	.L46
	movq	-24(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L29(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L29(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L29:
	.long	.L38-.L29
	.long	.L47-.L29
	.long	.L36-.L29
	.long	.L46-.L29
	.long	.L46-.L29
	.long	.L46-.L29
	.long	.L35-.L29
	.long	.L34-.L29
	.long	.L33-.L29
	.long	.L32-.L29
	.long	.L31-.L29
	.long	.L30-.L29
	.long	.L28-.L29
	.text
.L28:
	movq	-40(%rbp), %rax
	movq	32(%rax), %rdx
	movq	-40(%rbp), %rax
	movq	40(%rax), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	arc4random_buf@PLT
	movq	$0, -24(%rbp)
	jmp	.L39
.L33:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$28, %edx
	movl	$1, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movl	$2, %edi
	call	exit@PLT
.L30:
	movq	-40(%rbp), %rax
	movq	32(%rax), %rdx
	movq	-40(%rbp), %rax
	movq	24(%rax), %rax
	addq	%rdx, %rax
	movq	%rax, -32(%rbp)
	movq	-40(%rbp), %rax
	movq	40(%rax), %rax
	movq	-32(%rbp), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	-40(%rbp), %rdx
	movq	%rax, 40(%rdx)
	movq	$9, -24(%rbp)
	jmp	.L39
.L32:
	movq	-40(%rbp), %rax
	movq	40(%rax), %rax
	testq	%rax, %rax
	je	.L41
	movq	$10, -24(%rbp)
	jmp	.L39
.L41:
	movq	$8, -24(%rbp)
	jmp	.L39
.L35:
	movq	-40(%rbp), %rax
	movl	68(%rax), %eax
	testl	%eax, %eax
	je	.L43
	movq	$12, -24(%rbp)
	jmp	.L39
.L43:
	movq	$2, -24(%rbp)
	jmp	.L39
.L31:
	movq	-40(%rbp), %rax
	movq	-32(%rbp), %rdx
	movq	%rdx, 32(%rax)
	movl	$0, %edi
	call	time@PLT
	movq	%rax, -16(%rbp)
	movq	-40(%rbp), %rax
	movq	32(%rax), %rax
	shrq	$20, %rax
	movq	%rax, %rcx
	movq	stderr(%rip), %rax
	movq	-16(%rbp), %rdx
	leaq	.LC2(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	$6, -24(%rbp)
	jmp	.L39
.L38:
	movl	$0, %edi
	call	time@PLT
	movq	%rax, -8(%rbp)
	movq	-40(%rbp), %rax
	movq	16(%rax), %rax
	movl	%eax, %ecx
	movq	stderr(%rip), %rax
	movq	-8(%rbp), %rdx
	leaq	.LC3(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	-40(%rbp), %rax
	movq	56(%rax), %rax
	leaq	1(%rax), %rdx
	movq	-40(%rbp), %rax
	movq	%rdx, 56(%rax)
	movq	$1, -24(%rbp)
	jmp	.L39
.L34:
	movq	$11, -24(%rbp)
	jmp	.L39
.L36:
	movq	-40(%rbp), %rax
	movq	32(%rax), %rdx
	movq	-40(%rbp), %rax
	movq	40(%rax), %rax
	movl	$0, %esi
	movq	%rax, %rdi
	call	memset@PLT
	movq	$0, -24(%rbp)
	jmp	.L39
.L46:
	nop
.L39:
	jmp	.L45
.L47:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	increment, .-increment
	.globl	set_defaults
	.type	set_defaults, @function
set_defaults:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movl	%esi, -28(%rbp)
	movq	%rdx, -40(%rbp)
	movq	$0, -8(%rbp)
.L54:
	cmpq	$2, -8(%rbp)
	je	.L55
	cmpq	$2, -8(%rbp)
	ja	.L56
	cmpq	$0, -8(%rbp)
	je	.L51
	cmpq	$1, -8(%rbp)
	jne	.L56
	movq	-24(%rbp), %rax
	movl	-28(%rbp), %edx
	movl	%edx, 8(%rax)
	movq	-24(%rbp), %rax
	movq	-40(%rbp), %rdx
	movq	%rdx, (%rax)
	movq	-24(%rbp), %rax
	movq	$60, 16(%rax)
	movq	-24(%rbp), %rax
	movq	$157286400, 24(%rax)
	movq	-24(%rbp), %rax
	movq	$0, 32(%rax)
	movq	-24(%rbp), %rax
	movq	$0, 40(%rax)
	movq	-24(%rbp), %rax
	movq	$10, 48(%rax)
	movq	-24(%rbp), %rax
	movq	$0, 56(%rax)
	movq	-24(%rbp), %rax
	movl	$0, 64(%rax)
	movq	-24(%rbp), %rax
	movl	$0, 68(%rax)
	movq	$2, -8(%rbp)
	jmp	.L52
.L51:
	movq	$1, -8(%rbp)
	jmp	.L52
.L56:
	nop
.L52:
	jmp	.L54
.L55:
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	set_defaults, .-set_defaults
	.section	.rodata
.LC4:
	.string	"%ld eatcore: Sleeping...\n"
	.align 8
.LC5:
	.string	"%ld eatcore: Fully allocated. Exiting.\n"
	.text
	.globl	interval
	.type	interval, @function
interval:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movq	$7, -24(%rbp)
.L72:
	cmpq	$8, -24(%rbp)
	ja	.L73
	movq	-24(%rbp), %rax
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
	.long	.L65-.L60
	.long	.L64-.L60
	.long	.L73-.L60
	.long	.L73-.L60
	.long	.L74-.L60
	.long	.L62-.L60
	.long	.L73-.L60
	.long	.L61-.L60
	.long	.L59-.L60
	.text
.L59:
	movl	$0, %edi
	call	time@PLT
	movq	%rax, -16(%rbp)
	movq	stderr(%rip), %rax
	movq	-16(%rbp), %rdx
	leaq	.LC4(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	-40(%rbp), %rax
	movq	16(%rax), %rax
	movl	%eax, %edi
	call	sleep@PLT
	movq	$4, -24(%rbp)
	jmp	.L67
.L64:
	movq	-40(%rbp), %rax
	movq	56(%rax), %rdx
	movq	-40(%rbp), %rax
	movq	48(%rax), %rax
	cmpq	%rax, %rdx
	jb	.L68
	movq	$5, -24(%rbp)
	jmp	.L67
.L68:
	movq	$0, -24(%rbp)
	jmp	.L67
.L62:
	movl	$0, %edi
	call	time@PLT
	movq	%rax, -8(%rbp)
	movq	stderr(%rip), %rax
	movq	-8(%rbp), %rdx
	leaq	.LC5(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movl	$0, %edi
	call	exit@PLT
.L65:
	movl	$0, %edi
	call	time@PLT
	movq	%rax, -16(%rbp)
	movq	stderr(%rip), %rax
	movq	-16(%rbp), %rdx
	leaq	.LC4(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	-40(%rbp), %rax
	movq	16(%rax), %rax
	movl	%eax, %edi
	call	sleep@PLT
	movq	$4, -24(%rbp)
	jmp	.L67
.L61:
	movq	-40(%rbp), %rax
	movl	64(%rax), %eax
	testl	%eax, %eax
	je	.L70
	movq	$1, -24(%rbp)
	jmp	.L67
.L70:
	movq	$8, -24(%rbp)
	jmp	.L67
.L73:
	nop
.L67:
	jmp	.L72
.L74:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	interval, .-interval
	.section	.rodata
	.align 8
.LC6:
	.string	"%ld eatcore: usage: eatcore [-i interval_in_seconds] [-s increment_in_bytes] [-n max_increments] [-x] [-r]\n"
.LC7:
	.string	"version 0.2.1"
.LC8:
	.string	"hi:n:rs:xv"
	.text
	.globl	parse_commandline
	.type	parse_commandline, @function
parse_commandline:
.LFB10:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movq	$7, -16(%rbp)
.L104:
	cmpq	$24, -16(%rbp)
	ja	.L105
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L78(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L78(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L78:
	.long	.L105-.L78
	.long	.L105-.L78
	.long	.L90-.L78
	.long	.L89-.L78
	.long	.L88-.L78
	.long	.L105-.L78
	.long	.L87-.L78
	.long	.L86-.L78
	.long	.L85-.L78
	.long	.L106-.L78
	.long	.L105-.L78
	.long	.L105-.L78
	.long	.L105-.L78
	.long	.L83-.L78
	.long	.L105-.L78
	.long	.L105-.L78
	.long	.L82-.L78
	.long	.L105-.L78
	.long	.L81-.L78
	.long	.L105-.L78
	.long	.L80-.L78
	.long	.L79-.L78
	.long	.L105-.L78
	.long	.L105-.L78
	.long	.L77-.L78
	.text
.L81:
	cmpl	$-1, -32(%rbp)
	je	.L91
	movq	$20, -16(%rbp)
	jmp	.L93
.L91:
	movq	$2, -16(%rbp)
	jmp	.L93
.L88:
	movq	optarg(%rip), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movl	%eax, -24(%rbp)
	movl	-24(%rbp), %eax
	movslq	%eax, %rdx
	movq	-40(%rbp), %rax
	movq	%rdx, 16(%rax)
	movq	$21, -16(%rbp)
	jmp	.L93
.L85:
	movl	$0, %edi
	call	time@PLT
	movq	%rax, -8(%rbp)
	movq	stderr(%rip), %rax
	movq	-8(%rbp), %rdx
	leaq	.LC6(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movl	$1, %edi
	call	exit@PLT
.L89:
	movq	optarg(%rip), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movl	%eax, -20(%rbp)
	movl	-20(%rbp), %eax
	movslq	%eax, %rdx
	movq	-40(%rbp), %rax
	movq	%rdx, 48(%rax)
	movq	$21, -16(%rbp)
	jmp	.L93
.L82:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$0, %edi
	call	exit@PLT
.L77:
	movq	optarg(%rip), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movl	%eax, -28(%rbp)
	movl	-28(%rbp), %eax
	cltq
	salq	$20, %rax
	movq	%rax, %rdx
	movq	-40(%rbp), %rax
	movq	%rdx, 24(%rax)
	movq	$21, -16(%rbp)
	jmp	.L93
.L79:
	movq	-40(%rbp), %rax
	movq	(%rax), %rcx
	movq	-40(%rbp), %rax
	movl	8(%rax), %eax
	leaq	.LC8(%rip), %rdx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	getopt@PLT
	movl	%eax, -32(%rbp)
	movq	$18, -16(%rbp)
	jmp	.L93
.L83:
	movq	-40(%rbp), %rax
	movl	$1, 64(%rax)
	movq	$21, -16(%rbp)
	jmp	.L93
.L87:
	movq	-40(%rbp), %rax
	movl	$1, 68(%rax)
	movq	$21, -16(%rbp)
	jmp	.L93
.L86:
	movq	$21, -16(%rbp)
	jmp	.L93
.L90:
	movq	-40(%rbp), %rax
	movl	8(%rax), %eax
	movl	optind(%rip), %ecx
	subl	%ecx, %eax
	movl	%eax, %edx
	movq	-40(%rbp), %rax
	movl	%edx, 8(%rax)
	movq	-40(%rbp), %rax
	movq	(%rax), %rdx
	movl	optind(%rip), %eax
	cltq
	salq	$3, %rax
	addq	%rax, %rdx
	movq	-40(%rbp), %rax
	movq	%rdx, (%rax)
	movq	$9, -16(%rbp)
	jmp	.L93
.L80:
	movl	-32(%rbp), %eax
	subl	$105, %eax
	cmpl	$15, %eax
	ja	.L95
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L97(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L97(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L97:
	.long	.L102-.L97
	.long	.L95-.L97
	.long	.L95-.L97
	.long	.L95-.L97
	.long	.L95-.L97
	.long	.L101-.L97
	.long	.L95-.L97
	.long	.L95-.L97
	.long	.L95-.L97
	.long	.L100-.L97
	.long	.L99-.L97
	.long	.L95-.L97
	.long	.L95-.L97
	.long	.L98-.L97
	.long	.L95-.L97
	.long	.L96-.L97
	.text
.L98:
	movq	$16, -16(%rbp)
	jmp	.L103
.L96:
	movq	$13, -16(%rbp)
	jmp	.L103
.L100:
	movq	$6, -16(%rbp)
	jmp	.L103
.L101:
	movq	$3, -16(%rbp)
	jmp	.L103
.L102:
	movq	$4, -16(%rbp)
	jmp	.L103
.L99:
	movq	$24, -16(%rbp)
	jmp	.L103
.L95:
	movq	$8, -16(%rbp)
	nop
.L103:
	jmp	.L93
.L105:
	nop
.L93:
	jmp	.L104
.L106:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	parse_commandline, .-parse_commandline
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
