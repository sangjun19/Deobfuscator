	.file	"lac-dcc_jotai-benchmarks_extr_linuxdriversvideofbdevatafb_utils.h_fill8_col_Final_flatten.c"
	.text
	.globl	rand_primes
	.bss
	.align 32
	.type	rand_primes, @object
	.size	rand_primes, 100
rand_primes:
	.zero	100
	.globl	_TIG_IZ_vfit_envp
	.align 8
	.type	_TIG_IZ_vfit_envp, @object
	.size	_TIG_IZ_vfit_envp, 8
_TIG_IZ_vfit_envp:
	.zero	8
	.globl	_TIG_IZ_vfit_argv
	.align 8
	.type	_TIG_IZ_vfit_argv, @object
	.size	_TIG_IZ_vfit_argv, 8
_TIG_IZ_vfit_argv:
	.zero	8
	.globl	_TIG_IZ_vfit_argc
	.align 4
	.type	_TIG_IZ_vfit_argc, @object
	.size	_TIG_IZ_vfit_argc, 4
_TIG_IZ_vfit_argc:
	.zero	4
	.section	.rodata
	.align 8
.LC0:
	.string	"Usage:\n    prog [ARGS]\n\nARGS:\n       0            big-arr\n       1            big-arr-10x\n"
	.text
	.globl	usage
	.type	usage, @function
usage:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$0, -8(%rbp)
.L6:
	cmpq	$0, -8(%rbp)
	je	.L2
	cmpq	$1, -8(%rbp)
	jne	.L8
	jmp	.L7
.L2:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$1, -8(%rbp)
	jmp	.L5
.L8:
	nop
.L5:
	jmp	.L6
.L7:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	usage, .-usage
	.globl	next_i
	.type	next_i, @function
next_i:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	$1, -8(%rbp)
.L15:
	cmpq	$2, -8(%rbp)
	je	.L10
	cmpq	$2, -8(%rbp)
	ja	.L17
	cmpq	$0, -8(%rbp)
	je	.L12
	cmpq	$1, -8(%rbp)
	jne	.L17
	movq	$0, -8(%rbp)
	jmp	.L13
.L12:
	movl	$0, -12(%rbp)
	addl	$1, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L13
.L10:
	movl	-12(%rbp), %ecx
	movslq	%ecx, %rax
	imulq	$1374389535, %rax, %rax
	shrq	$32, %rax
	sarl	$3, %eax
	movl	%ecx, %esi
	sarl	$31, %esi
	subl	%esi, %eax
	movl	%eax, %edx
	movl	%edx, %eax
	sall	$2, %eax
	addl	%edx, %eax
	leal	0(,%rax,4), %edx
	addl	%edx, %eax
	subl	%eax, %ecx
	movl	%ecx, %edx
	movslq	%edx, %rax
	leaq	0(,%rax,4), %rdx
	leaq	rand_primes(%rip), %rax
	movl	(%rdx,%rax), %eax
	jmp	.L16
.L17:
	nop
.L13:
	jmp	.L15
.L16:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	next_i, .-next_i
	.globl	next_f
	.type	next_f, @function
next_f:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	$2, -8(%rbp)
.L26:
	cmpq	$2, -8(%rbp)
	je	.L19
	cmpq	$2, -8(%rbp)
	ja	.L28
	cmpq	$0, -8(%rbp)
	je	.L21
	cmpq	$1, -8(%rbp)
	jne	.L28
	movl	-12(%rbp), %ecx
	movslq	%ecx, %rax
	imulq	$1374389535, %rax, %rax
	shrq	$32, %rax
	sarl	$3, %eax
	movl	%ecx, %esi
	sarl	$31, %esi
	subl	%esi, %eax
	movl	%eax, %edx
	movl	%edx, %eax
	sall	$2, %eax
	addl	%edx, %eax
	leal	0(,%rax,4), %edx
	addl	%edx, %eax
	subl	%eax, %ecx
	movl	%ecx, %edx
	movslq	%edx, %rax
	leaq	0(,%rax,4), %rdx
	leaq	rand_primes(%rip), %rax
	movl	(%rdx,%rax), %eax
	movl	%eax, %eax
	testq	%rax, %rax
	js	.L22
	pxor	%xmm0, %xmm0
	cvtsi2ssq	%rax, %xmm0
	jmp	.L23
.L22:
	movq	%rax, %rdx
	shrq	%rdx
	andl	$1, %eax
	orq	%rax, %rdx
	pxor	%xmm0, %xmm0
	cvtsi2ssq	%rdx, %xmm0
	addss	%xmm0, %xmm0
.L23:
	movss	.LC1(%rip), %xmm1
	divss	%xmm1, %xmm0
	jmp	.L27
.L21:
	movl	$0, -12(%rbp)
	addl	$1, -12(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L25
.L19:
	movq	$0, -8(%rbp)
	jmp	.L25
.L28:
	nop
.L25:
	jmp	.L26
.L27:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	next_f, .-next_f
	.globl	main
	.type	main, @function
main:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$176, %rsp
	movl	%edi, -148(%rbp)
	movq	%rsi, -160(%rbp)
	movq	%rdx, -168(%rbp)
	movl	$179, rand_primes(%rip)
	movl	$103, 4+rand_primes(%rip)
	movl	$479, 8+rand_primes(%rip)
	movl	$647, 12+rand_primes(%rip)
	movl	$229, 16+rand_primes(%rip)
	movl	$37, 20+rand_primes(%rip)
	movl	$271, 24+rand_primes(%rip)
	movl	$557, 28+rand_primes(%rip)
	movl	$263, 32+rand_primes(%rip)
	movl	$607, 36+rand_primes(%rip)
	movl	$18743, 40+rand_primes(%rip)
	movl	$50359, 44+rand_primes(%rip)
	movl	$21929, 48+rand_primes(%rip)
	movl	$48757, 52+rand_primes(%rip)
	movl	$98179, 56+rand_primes(%rip)
	movl	$12907, 60+rand_primes(%rip)
	movl	$52937, 64+rand_primes(%rip)
	movl	$64579, 68+rand_primes(%rip)
	movl	$49957, 72+rand_primes(%rip)
	movl	$52567, 76+rand_primes(%rip)
	movl	$507163, 80+rand_primes(%rip)
	movl	$149939, 84+rand_primes(%rip)
	movl	$412157, 88+rand_primes(%rip)
	movl	$680861, 92+rand_primes(%rip)
	movl	$757751, 96+rand_primes(%rip)
	nop
.L30:
	movq	$0, _TIG_IZ_vfit_envp(%rip)
	nop
.L31:
	movq	$0, _TIG_IZ_vfit_argv(%rip)
	nop
.L32:
	movl	$0, _TIG_IZ_vfit_argc(%rip)
	nop
	nop
.L33:
.L34:
#APP
# 180 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-vfit--0
# 0 "" 2
#NO_APP
	movl	-148(%rbp), %eax
	movl	%eax, _TIG_IZ_vfit_argc(%rip)
	movq	-160(%rbp), %rax
	movq	%rax, _TIG_IZ_vfit_argv(%rip)
	movq	-168(%rbp), %rax
	movq	%rax, _TIG_IZ_vfit_envp(%rip)
	nop
	movq	$19, -40(%rbp)
.L73:
	cmpq	$42, -40(%rbp)
	ja	.L74
	movq	-40(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L37(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L37(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L37:
	.long	.L57-.L37
	.long	.L56-.L37
	.long	.L55-.L37
	.long	.L54-.L37
	.long	.L53-.L37
	.long	.L74-.L37
	.long	.L74-.L37
	.long	.L52-.L37
	.long	.L51-.L37
	.long	.L74-.L37
	.long	.L74-.L37
	.long	.L50-.L37
	.long	.L74-.L37
	.long	.L74-.L37
	.long	.L74-.L37
	.long	.L49-.L37
	.long	.L48-.L37
	.long	.L47-.L37
	.long	.L74-.L37
	.long	.L46-.L37
	.long	.L45-.L37
	.long	.L44-.L37
	.long	.L74-.L37
	.long	.L43-.L37
	.long	.L74-.L37
	.long	.L74-.L37
	.long	.L74-.L37
	.long	.L42-.L37
	.long	.L41-.L37
	.long	.L74-.L37
	.long	.L74-.L37
	.long	.L74-.L37
	.long	.L40-.L37
	.long	.L74-.L37
	.long	.L39-.L37
	.long	.L38-.L37
	.long	.L74-.L37
	.long	.L74-.L37
	.long	.L74-.L37
	.long	.L74-.L37
	.long	.L74-.L37
	.long	.L74-.L37
	.long	.L36-.L37
	.text
.L53:
	movq	-160(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movl	%eax, -84(%rbp)
	movl	-84(%rbp), %eax
	movl	%eax, -144(%rbp)
	movq	$23, -40(%rbp)
	jmp	.L58
.L49:
	movl	$65025, -132(%rbp)
	movl	-132(%rbp), %eax
	cltq
	salq	$2, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, -64(%rbp)
	movl	$0, -128(%rbp)
	movq	$17, -40(%rbp)
	jmp	.L58
.L51:
	call	next_i
	movl	%eax, -92(%rbp)
	call	next_i
	movl	%eax, -88(%rbp)
	movl	-92(%rbp), %eax
	cltd
	shrl	$31, %edx
	addl	%edx, %eax
	andl	$1, %eax
	subl	%edx, %eax
	movl	%eax, %edx
	movl	$0, %eax
	subl	%edx, %eax
	addl	%eax, %eax
	addl	$1, %eax
	movl	-112(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-48(%rbp), %rdx
	addq	%rcx, %rdx
	imull	-88(%rbp), %eax
	movl	%eax, (%rdx)
	addl	$1, -112(%rbp)
	movq	$20, -40(%rbp)
	jmp	.L58
.L56:
	movl	$65025, -140(%rbp)
	movl	-140(%rbp), %eax
	cltq
	salq	$2, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -72(%rbp)
	movl	$0, -136(%rbp)
	movq	$16, -40(%rbp)
	jmp	.L58
.L43:
	cmpl	$0, -144(%rbp)
	je	.L59
	cmpl	$1, -144(%rbp)
	jne	.L60
	movq	$11, -40(%rbp)
	jmp	.L61
.L59:
	movq	$1, -40(%rbp)
	jmp	.L61
.L60:
	movq	$7, -40(%rbp)
	nop
.L61:
	jmp	.L58
.L54:
	movq	-48(%rbp), %rdx
	movq	-56(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fill8_col
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	-48(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$32, -40(%rbp)
	jmp	.L58
.L48:
	movl	-136(%rbp), %eax
	cmpl	-140(%rbp), %eax
	jge	.L62
	movq	$28, -40(%rbp)
	jmp	.L58
.L62:
	movq	$15, -40(%rbp)
	jmp	.L58
.L44:
	movq	-64(%rbp), %rdx
	movq	-72(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fill8_col
	movq	-72(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$32, -40(%rbp)
	jmp	.L58
.L50:
	movl	$100, -124(%rbp)
	movl	-124(%rbp), %eax
	cltq
	salq	$2, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -56(%rbp)
	movl	$0, -120(%rbp)
	movq	$42, -40(%rbp)
	jmp	.L58
.L46:
	cmpl	$2, -148(%rbp)
	je	.L64
	movq	$0, -40(%rbp)
	jmp	.L58
.L64:
	movq	$4, -40(%rbp)
	jmp	.L58
.L40:
	movl	$0, %eax
	jmp	.L66
.L47:
	movl	-128(%rbp), %eax
	cmpl	-132(%rbp), %eax
	jge	.L67
	movq	$2, -40(%rbp)
	jmp	.L58
.L67:
	movq	$21, -40(%rbp)
	jmp	.L58
.L42:
	call	next_i
	movl	%eax, -100(%rbp)
	call	next_i
	movl	%eax, -96(%rbp)
	movl	-100(%rbp), %eax
	cltd
	shrl	$31, %edx
	addl	%edx, %eax
	andl	$1, %eax
	subl	%edx, %eax
	movl	%eax, %edx
	movl	$0, %eax
	subl	%edx, %eax
	addl	%eax, %eax
	addl	$1, %eax
	movl	-120(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-56(%rbp), %rdx
	addq	%rcx, %rdx
	imull	-96(%rbp), %eax
	movl	%eax, (%rdx)
	addl	$1, -120(%rbp)
	movq	$42, -40(%rbp)
	jmp	.L58
.L39:
	movl	$100, -116(%rbp)
	movl	-116(%rbp), %eax
	cltq
	salq	$2, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -32(%rbp)
	movq	-32(%rbp), %rax
	movq	%rax, -48(%rbp)
	movl	$0, -112(%rbp)
	movq	$20, -40(%rbp)
	jmp	.L58
.L41:
	call	next_i
	movl	%eax, -108(%rbp)
	call	next_i
	movl	%eax, -104(%rbp)
	movl	-108(%rbp), %eax
	cltd
	shrl	$31, %edx
	addl	%edx, %eax
	andl	$1, %eax
	subl	%edx, %eax
	movl	%eax, %edx
	movl	$0, %eax
	subl	%edx, %eax
	addl	%eax, %eax
	addl	$1, %eax
	movl	-136(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-72(%rbp), %rdx
	addq	%rcx, %rdx
	imull	-104(%rbp), %eax
	movl	%eax, (%rdx)
	addl	$1, -136(%rbp)
	movq	$16, -40(%rbp)
	jmp	.L58
.L36:
	movl	-120(%rbp), %eax
	cmpl	-124(%rbp), %eax
	jge	.L69
	movq	$27, -40(%rbp)
	jmp	.L58
.L69:
	movq	$34, -40(%rbp)
	jmp	.L58
.L57:
	call	usage
	movq	$35, -40(%rbp)
	jmp	.L58
.L52:
	call	usage
	movq	$32, -40(%rbp)
	jmp	.L58
.L38:
	movl	$1, %eax
	jmp	.L66
.L55:
	call	next_i
	movl	%eax, -80(%rbp)
	call	next_i
	movl	%eax, -76(%rbp)
	movl	-80(%rbp), %eax
	cltd
	shrl	$31, %edx
	addl	%edx, %eax
	andl	$1, %eax
	subl	%edx, %eax
	movl	%eax, %edx
	movl	$0, %eax
	subl	%edx, %eax
	addl	%eax, %eax
	addl	$1, %eax
	movl	-128(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-64(%rbp), %rdx
	addq	%rcx, %rdx
	imull	-76(%rbp), %eax
	movl	%eax, (%rdx)
	addl	$1, -128(%rbp)
	movq	$17, -40(%rbp)
	jmp	.L58
.L45:
	movl	-112(%rbp), %eax
	cmpl	-116(%rbp), %eax
	jge	.L71
	movq	$8, -40(%rbp)
	jmp	.L58
.L71:
	movq	$3, -40(%rbp)
	jmp	.L58
.L74:
	nop
.L58:
	jmp	.L73
.L66:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	main, .-main
	.type	fill8_col, @function
fill8_col:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	$0, -8(%rbp)
.L81:
	cmpq	$2, -8(%rbp)
	je	.L82
	cmpq	$2, -8(%rbp)
	ja	.L83
	cmpq	$0, -8(%rbp)
	je	.L78
	cmpq	$1, -8(%rbp)
	jne	.L83
	movq	-32(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, -12(%rbp)
	movq	-24(%rbp), %rax
	movl	-12(%rbp), %edx
	movl	%edx, (%rax)
	sarl	$8, -12(%rbp)
	movq	-24(%rbp), %rax
	leaq	8(%rax), %rdx
	movl	-12(%rbp), %eax
	movl	%eax, (%rdx)
	movq	$2, -8(%rbp)
	jmp	.L79
.L78:
	movq	$1, -8(%rbp)
	jmp	.L79
.L83:
	nop
.L79:
	jmp	.L81
.L82:
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	fill8_col, .-fill8_col
	.section	.rodata
	.align 4
.LC1:
	.long	1228472176
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
