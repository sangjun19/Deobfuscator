	.file	"lac-dcc_jotai-benchmarks_extr_linuxdriversnetethernetmellanoxmlx5coreen_ethtool.c_mlx5e_get_rxfh_indir_size_Final_flatten.c"
	.text
	.globl	rand_primes
	.bss
	.align 32
	.type	rand_primes, @object
	.size	rand_primes, 100
rand_primes:
	.zero	100
	.globl	MLX5E_INDIR_RQT_SIZE
	.align 4
	.type	MLX5E_INDIR_RQT_SIZE, @object
	.size	MLX5E_INDIR_RQT_SIZE, 4
MLX5E_INDIR_RQT_SIZE:
	.zero	4
	.globl	_TIG_IZ_5VbJ_envp
	.align 8
	.type	_TIG_IZ_5VbJ_envp, @object
	.size	_TIG_IZ_5VbJ_envp, 8
_TIG_IZ_5VbJ_envp:
	.zero	8
	.globl	_TIG_IZ_5VbJ_argc
	.align 4
	.type	_TIG_IZ_5VbJ_argc, @object
	.size	_TIG_IZ_5VbJ_argc, 4
_TIG_IZ_5VbJ_argc:
	.zero	4
	.globl	_TIG_IZ_5VbJ_argv
	.align 8
	.type	_TIG_IZ_5VbJ_argv, @object
	.size	_TIG_IZ_5VbJ_argv, 8
_TIG_IZ_5VbJ_argv:
	.zero	8
	.text
	.globl	next_i
	.type	next_i, @function
next_i:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	$1, -8(%rbp)
.L7:
	cmpq	$2, -8(%rbp)
	je	.L2
	cmpq	$2, -8(%rbp)
	ja	.L9
	cmpq	$0, -8(%rbp)
	je	.L4
	cmpq	$1, -8(%rbp)
	jne	.L9
	movq	$0, -8(%rbp)
	jmp	.L5
.L4:
	movl	$0, -12(%rbp)
	addl	$1, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L5
.L2:
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
	jmp	.L8
.L9:
	nop
.L5:
	jmp	.L7
.L8:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	next_i, .-next_i
	.section	.rodata
	.align 8
.LC0:
	.string	"Usage:\n    prog [ARGS]\n\nARGS:\n       0            big-arr\n       1            big-arr-10x\n       2            empty\n"
	.text
	.globl	usage
	.type	usage, @function
usage:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$1, -8(%rbp)
.L15:
	cmpq	$0, -8(%rbp)
	je	.L16
	cmpq	$1, -8(%rbp)
	jne	.L17
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$0, -8(%rbp)
	jmp	.L13
.L17:
	nop
.L13:
	jmp	.L15
.L16:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	usage, .-usage
	.globl	next_f
	.type	next_f, @function
next_f:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	$0, -8(%rbp)
.L26:
	cmpq	$2, -8(%rbp)
	je	.L19
	cmpq	$2, -8(%rbp)
	ja	.L28
	cmpq	$0, -8(%rbp)
	je	.L21
	cmpq	$1, -8(%rbp)
	jne	.L28
	movl	$0, -12(%rbp)
	addl	$1, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L22
.L21:
	movq	$1, -8(%rbp)
	jmp	.L22
.L19:
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
	js	.L23
	pxor	%xmm0, %xmm0
	cvtsi2ssq	%rax, %xmm0
	jmp	.L24
.L23:
	movq	%rax, %rdx
	shrq	%rdx
	andl	$1, %eax
	orq	%rax, %rdx
	pxor	%xmm0, %xmm0
	cvtsi2ssq	%rdx, %xmm0
	addss	%xmm0, %xmm0
.L24:
	movss	.LC1(%rip), %xmm1
	divss	%xmm1, %xmm0
	jmp	.L27
.L28:
	nop
.L22:
	jmp	.L26
.L27:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	next_f, .-next_f
	.section	.rodata
.LC2:
	.string	"%d\n"
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
	subq	$176, %rsp
	movl	%edi, -148(%rbp)
	movq	%rsi, -160(%rbp)
	movq	%rdx, -168(%rbp)
	movl	$0, MLX5E_INDIR_RQT_SIZE(%rip)
	nop
.L30:
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
.L31:
	movq	$0, _TIG_IZ_5VbJ_envp(%rip)
	nop
.L32:
	movq	$0, _TIG_IZ_5VbJ_argv(%rip)
	nop
.L33:
	movl	$0, _TIG_IZ_5VbJ_argc(%rip)
	nop
	nop
.L34:
.L35:
#APP
# 143 "lac-dcc_jotai-benchmarks_extr_linuxdriversnetethernetmellanoxmlx5coreen_ethtool.c_mlx5e_get_rxfh_indir_size_Final.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-5VbJ--0
# 0 "" 2
#NO_APP
	movl	-148(%rbp), %eax
	movl	%eax, _TIG_IZ_5VbJ_argc(%rip)
	movq	-160(%rbp), %rax
	movq	%rax, _TIG_IZ_5VbJ_argv(%rip)
	movq	-168(%rbp), %rax
	movq	%rax, _TIG_IZ_5VbJ_envp(%rip)
	nop
	movq	$35, -32(%rbp)
.L72:
	cmpq	$36, -32(%rbp)
	ja	.L73
	movq	-32(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L38(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L38(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L38:
	.long	.L56-.L38
	.long	.L73-.L38
	.long	.L73-.L38
	.long	.L55-.L38
	.long	.L54-.L38
	.long	.L53-.L38
	.long	.L73-.L38
	.long	.L73-.L38
	.long	.L52-.L38
	.long	.L73-.L38
	.long	.L73-.L38
	.long	.L51-.L38
	.long	.L73-.L38
	.long	.L73-.L38
	.long	.L73-.L38
	.long	.L73-.L38
	.long	.L50-.L38
	.long	.L49-.L38
	.long	.L48-.L38
	.long	.L73-.L38
	.long	.L47-.L38
	.long	.L46-.L38
	.long	.L45-.L38
	.long	.L44-.L38
	.long	.L43-.L38
	.long	.L73-.L38
	.long	.L73-.L38
	.long	.L73-.L38
	.long	.L73-.L38
	.long	.L73-.L38
	.long	.L42-.L38
	.long	.L73-.L38
	.long	.L41-.L38
	.long	.L73-.L38
	.long	.L40-.L38
	.long	.L39-.L38
	.long	.L37-.L38
	.text
.L48:
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	mlx5e_get_rxfh_indir_size
	movl	%eax, -88(%rbp)
	movl	-88(%rbp), %eax
	movl	%eax, -84(%rbp)
	movl	-84(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$30, -32(%rbp)
	jmp	.L57
.L54:
	call	usage
	movq	$30, -32(%rbp)
	jmp	.L57
.L42:
	movl	$0, %eax
	jmp	.L58
.L52:
	movl	$65025, -132(%rbp)
	movl	-132(%rbp), %eax
	cltq
	salq	$2, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -56(%rbp)
	movl	$0, -128(%rbp)
	movq	$32, -32(%rbp)
	jmp	.L57
.L44:
	cmpl	$2, -136(%rbp)
	je	.L59
	cmpl	$2, -136(%rbp)
	jg	.L60
	cmpl	$0, -136(%rbp)
	je	.L61
	cmpl	$1, -136(%rbp)
	je	.L62
	jmp	.L60
.L59:
	movq	$5, -32(%rbp)
	jmp	.L63
.L62:
	movq	$24, -32(%rbp)
	jmp	.L63
.L61:
	movq	$8, -32(%rbp)
	jmp	.L63
.L60:
	movq	$4, -32(%rbp)
	nop
.L63:
	jmp	.L57
.L55:
	call	next_i
	movl	%eax, -64(%rbp)
	call	next_i
	movl	%eax, -60(%rbp)
	movl	-64(%rbp), %eax
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
	movq	-40(%rbp), %rdx
	addq	%rcx, %rdx
	imull	-60(%rbp), %eax
	movl	%eax, (%rdx)
	addl	$1, -112(%rbp)
	movq	$0, -32(%rbp)
	jmp	.L57
.L50:
	movl	-120(%rbp), %eax
	cmpl	-124(%rbp), %eax
	jge	.L64
	movq	$34, -32(%rbp)
	jmp	.L57
.L64:
	movq	$36, -32(%rbp)
	jmp	.L57
.L43:
	movl	$100, -124(%rbp)
	movl	-124(%rbp), %eax
	cltq
	salq	$2, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, -48(%rbp)
	movl	$0, -120(%rbp)
	movq	$16, -32(%rbp)
	jmp	.L57
.L46:
	call	usage
	movq	$20, -32(%rbp)
	jmp	.L57
.L37:
	movq	-48(%rbp), %rax
	movq	%rax, %rdi
	call	mlx5e_get_rxfh_indir_size
	movl	%eax, -108(%rbp)
	movl	-108(%rbp), %eax
	movl	%eax, -104(%rbp)
	movl	-104(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-48(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$30, -32(%rbp)
	jmp	.L57
.L51:
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	mlx5e_get_rxfh_indir_size
	movl	%eax, -72(%rbp)
	movl	-72(%rbp), %eax
	movl	%eax, -68(%rbp)
	movl	-68(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$30, -32(%rbp)
	jmp	.L57
.L41:
	movl	-128(%rbp), %eax
	cmpl	-132(%rbp), %eax
	jge	.L66
	movq	$17, -32(%rbp)
	jmp	.L57
.L66:
	movq	$11, -32(%rbp)
	jmp	.L57
.L49:
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
	movq	-56(%rbp), %rdx
	addq	%rcx, %rdx
	imull	-76(%rbp), %eax
	movl	%eax, (%rdx)
	addl	$1, -128(%rbp)
	movq	$32, -32(%rbp)
	jmp	.L57
.L40:
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
	movq	-48(%rbp), %rdx
	addq	%rcx, %rdx
	imull	-96(%rbp), %eax
	movl	%eax, (%rdx)
	addl	$1, -120(%rbp)
	movq	$16, -32(%rbp)
	jmp	.L57
.L45:
	movq	-160(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movl	%eax, -92(%rbp)
	movl	-92(%rbp), %eax
	movl	%eax, -136(%rbp)
	movq	$23, -32(%rbp)
	jmp	.L57
.L53:
	movl	$1, -116(%rbp)
	movl	-116(%rbp), %eax
	cltq
	salq	$2, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -40(%rbp)
	movl	$0, -112(%rbp)
	movq	$0, -32(%rbp)
	jmp	.L57
.L56:
	movl	-112(%rbp), %eax
	cmpl	-116(%rbp), %eax
	jge	.L68
	movq	$3, -32(%rbp)
	jmp	.L57
.L68:
	movq	$18, -32(%rbp)
	jmp	.L57
.L39:
	cmpl	$2, -148(%rbp)
	je	.L70
	movq	$21, -32(%rbp)
	jmp	.L57
.L70:
	movq	$22, -32(%rbp)
	jmp	.L57
.L47:
	movl	$1, %eax
	jmp	.L58
.L73:
	nop
.L57:
	jmp	.L72
.L58:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	main, .-main
	.type	mlx5e_get_rxfh_indir_size, @function
mlx5e_get_rxfh_indir_size:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movq	$0, -8(%rbp)
.L77:
	cmpq	$0, -8(%rbp)
	jne	.L80
	movl	MLX5E_INDIR_RQT_SIZE(%rip), %eax
	jmp	.L79
.L80:
	nop
	jmp	.L77
.L79:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	mlx5e_get_rxfh_indir_size, .-mlx5e_get_rxfh_indir_size
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
