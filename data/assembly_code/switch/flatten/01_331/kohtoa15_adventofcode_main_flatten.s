	.file	"kohtoa15_adventofcode_main_flatten.c"
	.text
	.globl	_TIG_IZ_a6Sj_argc
	.bss
	.align 4
	.type	_TIG_IZ_a6Sj_argc, @object
	.size	_TIG_IZ_a6Sj_argc, 4
_TIG_IZ_a6Sj_argc:
	.zero	4
	.globl	_TIG_IZ_a6Sj_argv
	.align 8
	.type	_TIG_IZ_a6Sj_argv, @object
	.size	_TIG_IZ_a6Sj_argv, 8
_TIG_IZ_a6Sj_argv:
	.zero	8
	.globl	_TIG_IZ_a6Sj_envp
	.align 8
	.type	_TIG_IZ_a6Sj_envp, @object
	.size	_TIG_IZ_a6Sj_envp, 8
_TIG_IZ_a6Sj_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"invalid row value %x\n"
	.text
	.globl	addHeights
	.type	addHeights, @function
addHeights:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	$5, -8(%rbp)
.L19:
	cmpq	$10, -8(%rbp)
	ja	.L21
	movq	-8(%rbp), %rax
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
	.long	.L10-.L4
	.long	.L21-.L4
	.long	.L21-.L4
	.long	.L9-.L4
	.long	.L22-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L21-.L4
	.long	.L21-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L9:
	movl	-12(%rbp), %eax
	movslq	%eax, %rdx
	movq	-32(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	cmpl	$35, %eax
	je	.L12
	cmpl	$46, %eax
	je	.L13
	jmp	.L20
.L12:
	movq	$6, -8(%rbp)
	jmp	.L15
.L13:
	movq	$9, -8(%rbp)
	jmp	.L15
.L20:
	movq	$10, -8(%rbp)
	nop
.L15:
	jmp	.L16
.L5:
	addl	$1, -12(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L16
.L6:
	movq	-24(%rbp), %rdx
	movl	-12(%rbp), %eax
	cltq
	movzbl	(%rdx,%rax), %eax
	addl	$1, %eax
	movl	%eax, %ecx
	movq	-24(%rbp), %rdx
	movl	-12(%rbp), %eax
	cltq
	movb	%cl, (%rdx,%rax)
	movq	$9, -8(%rbp)
	jmp	.L16
.L7:
	movl	$0, -12(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L16
.L3:
	movl	-12(%rbp), %eax
	movslq	%eax, %rdx
	movq	-32(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$9, -8(%rbp)
	jmp	.L16
.L10:
	cmpl	$4, -12(%rbp)
	jg	.L17
	movq	$3, -8(%rbp)
	jmp	.L16
.L17:
	movq	$4, -8(%rbp)
	jmp	.L16
.L21:
	nop
.L16:
	jmp	.L19
.L22:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	addHeights, .-addHeights
	.section	.rodata
.LC1:
	.string	"\n"
	.text
	.globl	readInput
	.type	readInput, @function
readInput:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$176, %rsp
	movq	%rdi, -168(%rbp)
	movq	%rsi, -176(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$3, -128(%rbp)
.L41:
	cmpq	$10, -128(%rbp)
	ja	.L44
	movq	-128(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L26(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L26(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L26:
	.long	.L34-.L26
	.long	.L44-.L26
	.long	.L33-.L26
	.long	.L32-.L26
	.long	.L31-.L26
	.long	.L44-.L26
	.long	.L30-.L26
	.long	.L29-.L26
	.long	.L28-.L26
	.long	.L27-.L26
	.long	.L25-.L26
	.text
.L31:
	leaq	-80(%rbp), %rdx
	movq	-144(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	feedInput
	movq	$6, -128(%rbp)
	jmp	.L35
.L28:
	cmpl	$0, -148(%rbp)
	jne	.L36
	movq	$10, -128(%rbp)
	jmp	.L35
.L36:
	movq	$4, -128(%rbp)
	jmp	.L35
.L32:
	movq	$9, -128(%rbp)
	jmp	.L35
.L27:
	movq	$0, -112(%rbp)
	movq	$0, -104(%rbp)
	movq	$0, -96(%rbp)
	leaq	-112(%rbp), %rax
	movq	%rax, %rdi
	call	newKeyOrLock
	movq	%rax, -120(%rbp)
	movq	-120(%rbp), %rax
	movq	%rax, -144(%rbp)
	movq	$64, -104(%rbp)
	movq	$6, -128(%rbp)
	jmp	.L35
.L30:
	movq	-176(%rbp), %rdx
	leaq	-80(%rbp), %rax
	movl	$64, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	movq	%rax, -136(%rbp)
	movq	$0, -128(%rbp)
	jmp	.L35
.L25:
	leaq	-112(%rbp), %rax
	movq	%rax, %rdi
	call	newKeyOrLock
	movq	%rax, -144(%rbp)
	movq	$6, -128(%rbp)
	jmp	.L35
.L34:
	cmpq	$0, -136(%rbp)
	je	.L38
	movq	$2, -128(%rbp)
	jmp	.L35
.L38:
	movq	$7, -128(%rbp)
	jmp	.L35
.L29:
	movq	-168(%rbp), %rcx
	movq	-112(%rbp), %rax
	movq	-104(%rbp), %rdx
	movq	%rax, (%rcx)
	movq	%rdx, 8(%rcx)
	movq	-96(%rbp), %rax
	movq	%rax, 16(%rcx)
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L42
	jmp	.L43
.L33:
	leaq	-80(%rbp), %rax
	leaq	.LC1(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -148(%rbp)
	movq	$8, -128(%rbp)
	jmp	.L35
.L44:
	nop
.L35:
	jmp	.L41
.L43:
	call	__stack_chk_fail@PLT
.L42:
	movq	-168(%rbp), %rax
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	readInput, .-readInput
	.section	.rodata
.LC2:
	.string	"invalid input state %d\n"
	.text
	.globl	convertKeysAndLocks
	.type	convertKeysAndLocks, @function
convertKeysAndLocks:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	addq	$-128, %rsp
	movq	%rdi, -120(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$16, -88(%rbp)
.L70:
	cmpq	$23, -88(%rbp)
	ja	.L73
	movq	-88(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L48(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L48(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L48:
	.long	.L60-.L48
	.long	.L59-.L48
	.long	.L73-.L48
	.long	.L58-.L48
	.long	.L73-.L48
	.long	.L57-.L48
	.long	.L73-.L48
	.long	.L73-.L48
	.long	.L73-.L48
	.long	.L73-.L48
	.long	.L56-.L48
	.long	.L73-.L48
	.long	.L55-.L48
	.long	.L73-.L48
	.long	.L54-.L48
	.long	.L73-.L48
	.long	.L53-.L48
	.long	.L52-.L48
	.long	.L51-.L48
	.long	.L50-.L48
	.long	.L73-.L48
	.long	.L73-.L48
	.long	.L49-.L48
	.long	.L47-.L48
	.text
.L51:
	addq	$1, -96(%rbp)
	movq	$10, -88(%rbp)
	jmp	.L61
.L54:
	movq	$0, -96(%rbp)
	movq	$10, -88(%rbp)
	jmp	.L61
.L55:
	movq	-40(%rbp), %rax
	movq	%rax, -64(%rbp)
	movq	-40(%rbp), %rax
	addq	$1, %rax
	movq	%rax, -40(%rbp)
	leaq	-16(%rbp), %rax
	movq	-24(%rbp), %rcx
	movq	-64(%rbp), %rdx
	salq	$3, %rdx
	addq	%rcx, %rdx
	movq	(%rax), %rax
	movq	%rax, (%rdx)
	movq	$18, -88(%rbp)
	jmp	.L61
.L59:
	movl	-100(%rbp), %eax
	movb	$0, -16(%rbp,%rax)
	addl	$1, -100(%rbp)
	movq	$23, -88(%rbp)
	jmp	.L61
.L47:
	cmpl	$7, -100(%rbp)
	jbe	.L62
	movq	$14, -88(%rbp)
	jmp	.L61
.L62:
	movq	$1, -88(%rbp)
	jmp	.L61
.L58:
	movq	-48(%rbp), %rax
	movq	%rax, -56(%rbp)
	movq	-48(%rbp), %rax
	addq	$1, %rax
	movq	%rax, -48(%rbp)
	leaq	-16(%rbp), %rax
	movq	(%rax), %rdx
	movq	-32(%rbp), %rcx
	movq	-56(%rbp), %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movabsq	$-4311810305, %rcx
	addq	%rcx, %rdx
	movq	%rdx, (%rax)
	movq	$18, -88(%rbp)
	jmp	.L61
.L53:
	movq	$19, -88(%rbp)
	jmp	.L61
.L50:
	movq	$0, -48(%rbp)
	movq	$0, -40(%rbp)
	movq	$0, -32(%rbp)
	movq	$0, -24(%rbp)
	movq	16(%rbp), %rax
	salq	$3, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -80(%rbp)
	movq	-80(%rbp), %rax
	movq	%rax, -32(%rbp)
	movq	16(%rbp), %rax
	salq	$3, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -72(%rbp)
	movq	-72(%rbp), %rax
	movq	%rax, -24(%rbp)
	movb	$0, -16(%rbp)
	movl	$1, -100(%rbp)
	movq	$23, -88(%rbp)
	jmp	.L61
.L52:
	movq	32(%rbp), %rcx
	movq	-96(%rbp), %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	addq	%rcx, %rax
	movl	8(%rax), %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$18, -88(%rbp)
	jmp	.L61
.L49:
	movq	-120(%rbp), %rcx
	movq	-48(%rbp), %rax
	movq	-40(%rbp), %rdx
	movq	%rax, (%rcx)
	movq	%rdx, 8(%rcx)
	movq	-32(%rbp), %rax
	movq	-24(%rbp), %rdx
	movq	%rax, 16(%rcx)
	movq	%rdx, 24(%rcx)
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L71
	jmp	.L72
.L57:
	movq	32(%rbp), %rcx
	movq	-96(%rbp), %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	addq	%rcx, %rax
	movl	8(%rax), %eax
	cmpl	$1, %eax
	je	.L65
	cmpl	$2, %eax
	jne	.L66
	movq	$12, -88(%rbp)
	jmp	.L67
.L65:
	movq	$3, -88(%rbp)
	jmp	.L67
.L66:
	movq	$17, -88(%rbp)
	nop
.L67:
	jmp	.L61
.L56:
	movq	16(%rbp), %rax
	cmpq	%rax, -96(%rbp)
	jnb	.L68
	movq	$0, -88(%rbp)
	jmp	.L61
.L68:
	movq	$22, -88(%rbp)
	jmp	.L61
.L60:
	leaq	-16(%rbp), %rax
	movl	$8, %edx
	movl	$0, %esi
	movq	%rax, %rdi
	call	memset@PLT
	movq	32(%rbp), %rcx
	movq	-96(%rbp), %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	addq	%rcx, %rax
	movq	%rax, %rcx
	leaq	-16(%rbp), %rax
	movl	$5, %edx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	memcpy@PLT
	movq	$5, -88(%rbp)
	jmp	.L61
.L73:
	nop
.L61:
	jmp	.L70
.L72:
	call	__stack_chk_fail@PLT
.L71:
	movq	-120(%rbp), %rax
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	convertKeysAndLocks, .-convertKeysAndLocks
	.section	.rodata
.LC3:
	.string	"r"
.LC4:
	.string	"key/lock pairs that fit: %d\n"
.LC5:
	.string	"main"
.LC6:
	.string	"kohtoa15_adventofcode_main.c"
.LC7:
	.string	"argc == 2"
	.text
	.globl	main
	.type	main, @function
main:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$208, %rsp
	movl	%edi, -180(%rbp)
	movq	%rsi, -192(%rbp)
	movq	%rdx, -200(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_a6Sj_envp(%rip)
	nop
.L75:
	movq	$0, _TIG_IZ_a6Sj_argv(%rip)
	nop
.L76:
	movl	$0, _TIG_IZ_a6Sj_argc(%rip)
	nop
	nop
.L77:
.L78:
#APP
# 90 "kohtoa15_adventofcode_main.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-a6Sj--0
# 0 "" 2
#NO_APP
	movl	-180(%rbp), %eax
	movl	%eax, _TIG_IZ_a6Sj_argc(%rip)
	movq	-192(%rbp), %rax
	movq	%rax, _TIG_IZ_a6Sj_argv(%rip)
	movq	-200(%rbp), %rax
	movq	%rax, _TIG_IZ_a6Sj_envp(%rip)
	nop
	movq	$3, -168(%rbp)
.L88:
	cmpq	$4, -168(%rbp)
	je	.L79
	cmpq	$4, -168(%rbp)
	ja	.L91
	cmpq	$3, -168(%rbp)
	je	.L81
	cmpq	$3, -168(%rbp)
	ja	.L91
	cmpq	$0, -168(%rbp)
	je	.L82
	cmpq	$1, -168(%rbp)
	je	.L83
	jmp	.L91
.L79:
	movq	-192(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	leaq	.LC3(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -160(%rbp)
	movq	-160(%rbp), %rax
	movq	%rax, -152(%rbp)
	leaq	-112(%rbp), %rax
	movq	-152(%rbp), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	readInput
	movq	-112(%rbp), %rax
	movq	-104(%rbp), %rdx
	movq	%rax, -144(%rbp)
	movq	%rdx, -136(%rbp)
	movq	-96(%rbp), %rax
	movq	%rax, -128(%rbp)
	movq	-152(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	leaq	-48(%rbp), %rax
	subq	$8, %rsp
	pushq	-128(%rbp)
	pushq	-136(%rbp)
	pushq	-144(%rbp)
	movq	%rax, %rdi
	call	convertKeysAndLocks
	addq	$32, %rsp
	movq	-48(%rbp), %rax
	movq	-40(%rbp), %rdx
	movq	%rax, -80(%rbp)
	movq	%rdx, -72(%rbp)
	movq	-32(%rbp), %rax
	movq	-24(%rbp), %rdx
	movq	%rax, -64(%rbp)
	movq	%rdx, -56(%rbp)
	pushq	-56(%rbp)
	pushq	-64(%rbp)
	pushq	-72(%rbp)
	pushq	-80(%rbp)
	call	getFittingKeysLocks
	addq	$32, %rsp
	movl	%eax, -176(%rbp)
	movl	-176(%rbp), %eax
	movl	%eax, -172(%rbp)
	movl	-172(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$0, -168(%rbp)
	jmp	.L84
.L83:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rcx
	movl	$155, %edx
	leaq	.LC6(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L81:
	cmpl	$2, -180(%rbp)
	jne	.L85
	movq	$4, -168(%rbp)
	jmp	.L84
.L85:
	movq	$1, -168(%rbp)
	jmp	.L84
.L82:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L89
	jmp	.L90
.L91:
	nop
.L84:
	jmp	.L88
.L90:
	call	__stack_chk_fail@PLT
.L89:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	main, .-main
	.globl	newKeyOrLock
	.type	newKeyOrLock, @function
newKeyOrLock:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movq	%rdi, -56(%rbp)
	movq	$9, -24(%rbp)
.L112:
	cmpq	$11, -24(%rbp)
	ja	.L113
	movq	-24(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L95(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L95(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L95:
	.long	.L113-.L95
	.long	.L103-.L95
	.long	.L102-.L95
	.long	.L101-.L95
	.long	.L100-.L95
	.long	.L113-.L95
	.long	.L99-.L95
	.long	.L98-.L95
	.long	.L113-.L95
	.long	.L97-.L95
	.long	.L96-.L95
	.long	.L94-.L95
	.text
.L100:
	movq	-56(%rbp), %rax
	movq	16(%rax), %rcx
	movq	-32(%rbp), %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	addq	%rcx, %rax
	jmp	.L104
.L103:
	movq	-56(%rbp), %rax
	movq	-40(%rbp), %rdx
	movq	%rdx, 16(%rax)
	movq	$3, -24(%rbp)
	jmp	.L105
.L101:
	movq	-56(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, -32(%rbp)
	movq	-56(%rbp), %rax
	movq	16(%rax), %rcx
	movq	-32(%rbp), %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	addq	%rcx, %rax
	movl	$0, 8(%rax)
	movq	-56(%rbp), %rax
	movq	(%rax), %rax
	leaq	1(%rax), %rdx
	movq	-56(%rbp), %rax
	movq	%rdx, (%rax)
	movq	$4, -24(%rbp)
	jmp	.L105
.L94:
	movq	-56(%rbp), %rax
	movq	(%rax), %rdx
	movq	-56(%rbp), %rax
	movq	8(%rax), %rax
	cmpq	%rax, %rdx
	jb	.L106
	movq	$10, -24(%rbp)
	jmp	.L105
.L106:
	movq	$3, -24(%rbp)
	jmp	.L105
.L97:
	movq	-56(%rbp), %rax
	movq	8(%rax), %rax
	testq	%rax, %rax
	jne	.L108
	movq	$2, -24(%rbp)
	jmp	.L105
.L108:
	movq	$11, -24(%rbp)
	jmp	.L105
.L99:
	cmpq	$0, -40(%rbp)
	jne	.L110
	movq	$7, -24(%rbp)
	jmp	.L105
.L110:
	movq	$1, -24(%rbp)
	jmp	.L105
.L96:
	movq	-56(%rbp), %rax
	movq	8(%rax), %rax
	leaq	(%rax,%rax), %rdx
	movq	-56(%rbp), %rax
	movq	%rdx, 8(%rax)
	movq	-56(%rbp), %rax
	movq	8(%rax), %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-56(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -40(%rbp)
	movq	$6, -24(%rbp)
	jmp	.L105
.L98:
	movl	$0, %eax
	jmp	.L104
.L102:
	movq	-56(%rbp), %rax
	movq	$64, 8(%rax)
	movq	-56(%rbp), %rax
	movq	8(%rax), %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-56(%rbp), %rax
	movq	-8(%rbp), %rdx
	movq	%rdx, 16(%rax)
	movq	$3, -24(%rbp)
	jmp	.L105
.L113:
	nop
.L105:
	jmp	.L112
.L104:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	newKeyOrLock, .-newKeyOrLock
	.section	.rodata
.LC8:
	.string	"invalid state %d\n"
	.text
	.globl	feedInput
	.type	feedInput, @function
feedInput:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	$4, -8(%rbp)
.L130:
	cmpq	$9, -8(%rbp)
	ja	.L131
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L117(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L117(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L117:
	.long	.L122-.L117
	.long	.L121-.L117
	.long	.L131-.L117
	.long	.L131-.L117
	.long	.L120-.L117
	.long	.L131-.L117
	.long	.L132-.L117
	.long	.L118-.L117
	.long	.L131-.L117
	.long	.L116-.L117
	.text
.L120:
	movq	-24(%rbp), %rax
	movl	8(%rax), %eax
	cmpl	$2, %eax
	je	.L123
	cmpl	$2, %eax
	jg	.L124
	testl	%eax, %eax
	je	.L125
	cmpl	$1, %eax
	je	.L126
	jmp	.L124
.L123:
	movq	$1, -8(%rbp)
	jmp	.L127
.L126:
	movq	$9, -8(%rbp)
	jmp	.L127
.L125:
	movq	$7, -8(%rbp)
	jmp	.L127
.L124:
	movq	$0, -8(%rbp)
	nop
.L127:
	jmp	.L128
.L121:
	movq	-32(%rbp), %rdx
	movq	-24(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	addHeights
	movq	$6, -8(%rbp)
	jmp	.L128
.L116:
	movq	-32(%rbp), %rdx
	movq	-24(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	addHeights
	movq	$6, -8(%rbp)
	jmp	.L128
.L122:
	movq	-24(%rbp), %rax
	movl	8(%rax), %eax
	movl	%eax, %esi
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$6, -8(%rbp)
	jmp	.L128
.L118:
	movq	-32(%rbp), %rdx
	movq	-24(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	initState
	movq	$6, -8(%rbp)
	jmp	.L128
.L131:
	nop
.L128:
	jmp	.L130
.L132:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	feedInput, .-feedInput
	.globl	getFittingKeysLocks
	.type	getFittingKeysLocks, @function
getFittingKeysLocks:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	$0, -16(%rbp)
.L155:
	cmpq	$16, -16(%rbp)
	ja	.L157
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L136(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L136(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L136:
	.long	.L146-.L136
	.long	.L157-.L136
	.long	.L145-.L136
	.long	.L144-.L136
	.long	.L143-.L136
	.long	.L157-.L136
	.long	.L157-.L136
	.long	.L157-.L136
	.long	.L142-.L136
	.long	.L141-.L136
	.long	.L140-.L136
	.long	.L157-.L136
	.long	.L139-.L136
	.long	.L157-.L136
	.long	.L138-.L136
	.long	.L137-.L136
	.long	.L135-.L136
	.text
.L143:
	movl	-52(%rbp), %eax
	jmp	.L156
.L138:
	addq	$1, -48(%rbp)
	movq	$10, -16(%rbp)
	jmp	.L148
.L137:
	addq	$1, -32(%rbp)
	movq	$8, -16(%rbp)
	jmp	.L148
.L139:
	movq	32(%rbp), %rdx
	movq	-32(%rbp), %rax
	salq	$3, %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movq	%rax, -8(%rbp)
	movq	-40(%rbp), %rdx
	movq	-8(%rbp), %rax
	addq	%rax, %rdx
	movabsq	$43118103050, %rax
	addq	%rdx, %rax
	movq	%rax, -24(%rbp)
	movq	$2, -16(%rbp)
	jmp	.L148
.L142:
	movq	16(%rbp), %rax
	cmpq	%rax, -32(%rbp)
	jnb	.L149
	movq	$12, -16(%rbp)
	jmp	.L148
.L149:
	movq	$14, -16(%rbp)
	jmp	.L148
.L144:
	movl	$0, -52(%rbp)
	movq	$0, -48(%rbp)
	movq	$10, -16(%rbp)
	jmp	.L148
.L135:
	addl	$1, -52(%rbp)
	movq	$15, -16(%rbp)
	jmp	.L148
.L141:
	movq	40(%rbp), %rdx
	movq	-48(%rbp), %rax
	salq	$3, %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movq	%rax, -40(%rbp)
	movq	$0, -32(%rbp)
	movq	$8, -16(%rbp)
	jmp	.L148
.L140:
	movq	24(%rbp), %rax
	cmpq	%rax, -48(%rbp)
	jnb	.L151
	movq	$9, -16(%rbp)
	jmp	.L148
.L151:
	movq	$4, -16(%rbp)
	jmp	.L148
.L146:
	movq	$3, -16(%rbp)
	jmp	.L148
.L145:
	movabsq	$1034834473200, %rax
	andq	-24(%rbp), %rax
	testq	%rax, %rax
	jne	.L153
	movq	$16, -16(%rbp)
	jmp	.L148
.L153:
	movq	$15, -16(%rbp)
	jmp	.L148
.L157:
	nop
.L148:
	jmp	.L155
.L156:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	getFittingKeysLocks, .-getFittingKeysLocks
	.section	.rodata
.LC9:
	.string	"....."
.LC10:
	.string	"#####"
.LC11:
	.string	"invalid init row: %s"
	.text
	.globl	initState
	.type	initState, @function
initState:
.LFB13:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	$1, -8(%rbp)
.L175:
	cmpq	$8, -8(%rbp)
	ja	.L176
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L161(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L161(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L161:
	.long	.L168-.L161
	.long	.L167-.L161
	.long	.L166-.L161
	.long	.L177-.L161
	.long	.L176-.L161
	.long	.L164-.L161
	.long	.L163-.L161
	.long	.L162-.L161
	.long	.L160-.L161
	.text
.L160:
	cmpl	$0, -12(%rbp)
	jne	.L169
	movq	$6, -8(%rbp)
	jmp	.L171
.L169:
	movq	$0, -8(%rbp)
	jmp	.L171
.L167:
	movq	-32(%rbp), %rax
	movl	$5, %edx
	leaq	.LC9(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	strncmp@PLT
	movl	%eax, -12(%rbp)
	movq	$8, -8(%rbp)
	jmp	.L171
.L163:
	movq	-24(%rbp), %rax
	movl	$1, 8(%rax)
	movq	$3, -8(%rbp)
	jmp	.L171
.L164:
	movq	-24(%rbp), %rax
	movl	$2, 8(%rax)
	movq	$3, -8(%rbp)
	jmp	.L171
.L168:
	movq	-32(%rbp), %rax
	movl	$5, %edx
	leaq	.LC10(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	strncmp@PLT
	movl	%eax, -16(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L171
.L162:
	movq	-32(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-24(%rbp), %rax
	movl	$-1, 8(%rax)
	movq	$3, -8(%rbp)
	jmp	.L171
.L166:
	cmpl	$0, -16(%rbp)
	jne	.L173
	movq	$5, -8(%rbp)
	jmp	.L171
.L173:
	movq	$7, -8(%rbp)
	jmp	.L171
.L176:
	nop
.L171:
	jmp	.L175
.L177:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE13:
	.size	initState, .-initState
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
