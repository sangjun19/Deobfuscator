	.file	"lac-dcc_jotai-benchmarks_extr_linuxdriversnetethernetmellanoxmlx5coreen_fs.c_mlx5e_set_inner_ttc_ft_params_Final_flatten.c"
	.text
	.globl	_TIG_IZ_Jrl1_argc
	.bss
	.align 4
	.type	_TIG_IZ_Jrl1_argc, @object
	.size	_TIG_IZ_Jrl1_argc, 4
_TIG_IZ_Jrl1_argc:
	.zero	4
	.globl	_TIG_IZ_Jrl1_envp
	.align 8
	.type	_TIG_IZ_Jrl1_envp, @object
	.size	_TIG_IZ_Jrl1_envp, 8
_TIG_IZ_Jrl1_envp:
	.zero	8
	.globl	rand_primes
	.align 32
	.type	rand_primes, @object
	.size	rand_primes, 100
rand_primes:
	.zero	100
	.globl	_TIG_IZ_Jrl1_argv
	.align 8
	.type	_TIG_IZ_Jrl1_argv, @object
	.size	_TIG_IZ_Jrl1_argv, 8
_TIG_IZ_Jrl1_argv:
	.zero	8
	.globl	MLX5E_INNER_TTC_FT_LEVEL
	.align 4
	.type	MLX5E_INNER_TTC_FT_LEVEL, @object
	.size	MLX5E_INNER_TTC_FT_LEVEL, 4
MLX5E_INNER_TTC_FT_LEVEL:
	.zero	4
	.globl	MLX5E_NIC_PRIO
	.align 4
	.type	MLX5E_NIC_PRIO, @object
	.size	MLX5E_NIC_PRIO, 4
MLX5E_NIC_PRIO:
	.zero	4
	.globl	MLX5E_INNER_TTC_TABLE_SIZE
	.align 4
	.type	MLX5E_INNER_TTC_TABLE_SIZE, @object
	.size	MLX5E_INNER_TTC_TABLE_SIZE, 4
MLX5E_INNER_TTC_TABLE_SIZE:
	.zero	4
	.text
	.globl	next_f
	.type	next_f, @function
next_f:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	$0, -8(%rbp)
.L9:
	cmpq	$2, -8(%rbp)
	je	.L2
	cmpq	$2, -8(%rbp)
	ja	.L11
	cmpq	$0, -8(%rbp)
	je	.L4
	cmpq	$1, -8(%rbp)
	jne	.L11
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
	js	.L5
	pxor	%xmm0, %xmm0
	cvtsi2ssq	%rax, %xmm0
	jmp	.L6
.L5:
	movq	%rax, %rdx
	shrq	%rdx
	andl	$1, %eax
	orq	%rax, %rdx
	pxor	%xmm0, %xmm0
	cvtsi2ssq	%rdx, %xmm0
	addss	%xmm0, %xmm0
.L6:
	movss	.LC0(%rip), %xmm1
	divss	%xmm1, %xmm0
	jmp	.L10
.L4:
	movq	$2, -8(%rbp)
	jmp	.L8
.L2:
	movl	$0, -12(%rbp)
	addl	$1, -12(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L8
.L11:
	nop
.L8:
	jmp	.L9
.L10:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	next_f, .-next_f
	.section	.rodata
	.align 8
.LC1:
	.string	"Usage:\n    prog [ARGS]\n\nARGS:\n       0            big-arr\n       1            big-arr-10x\n       2            empty\n"
	.text
	.globl	usage
	.type	usage, @function
usage:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$0, -8(%rbp)
.L17:
	cmpq	$0, -8(%rbp)
	je	.L13
	cmpq	$1, -8(%rbp)
	jne	.L19
	jmp	.L18
.L13:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$1, -8(%rbp)
	jmp	.L16
.L19:
	nop
.L16:
	jmp	.L17
.L18:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	usage, .-usage
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
	subq	$192, %rsp
	movl	%edi, -164(%rbp)
	movq	%rsi, -176(%rbp)
	movq	%rdx, -184(%rbp)
	movl	$0, MLX5E_NIC_PRIO(%rip)
	nop
.L21:
	movl	$0, MLX5E_INNER_TTC_TABLE_SIZE(%rip)
	nop
.L22:
	movl	$0, MLX5E_INNER_TTC_FT_LEVEL(%rip)
	nop
.L23:
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
.L24:
	movq	$0, _TIG_IZ_Jrl1_envp(%rip)
	nop
.L25:
	movq	$0, _TIG_IZ_Jrl1_argv(%rip)
	nop
.L26:
	movl	$0, _TIG_IZ_Jrl1_argc(%rip)
	nop
	nop
.L27:
.L28:
#APP
# 193 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-Jrl1--0
# 0 "" 2
#NO_APP
	movl	-164(%rbp), %eax
	movl	%eax, _TIG_IZ_Jrl1_argc(%rip)
	movq	-176(%rbp), %rax
	movq	%rax, _TIG_IZ_Jrl1_argv(%rip)
	movq	-184(%rbp), %rax
	movq	%rax, _TIG_IZ_Jrl1_envp(%rip)
	nop
	movq	$22, -32(%rbp)
.L65:
	cmpq	$33, -32(%rbp)
	ja	.L66
	movq	-32(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L31(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L31(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L31:
	.long	.L66-.L31
	.long	.L49-.L31
	.long	.L48-.L31
	.long	.L66-.L31
	.long	.L66-.L31
	.long	.L66-.L31
	.long	.L66-.L31
	.long	.L47-.L31
	.long	.L66-.L31
	.long	.L46-.L31
	.long	.L45-.L31
	.long	.L66-.L31
	.long	.L44-.L31
	.long	.L43-.L31
	.long	.L42-.L31
	.long	.L41-.L31
	.long	.L40-.L31
	.long	.L66-.L31
	.long	.L66-.L31
	.long	.L66-.L31
	.long	.L39-.L31
	.long	.L38-.L31
	.long	.L37-.L31
	.long	.L66-.L31
	.long	.L36-.L31
	.long	.L66-.L31
	.long	.L35-.L31
	.long	.L34-.L31
	.long	.L33-.L31
	.long	.L66-.L31
	.long	.L32-.L31
	.long	.L66-.L31
	.long	.L66-.L31
	.long	.L30-.L31
	.text
.L32:
	movl	-144(%rbp), %eax
	cmpl	-148(%rbp), %eax
	jge	.L50
	movq	$14, -32(%rbp)
	jmp	.L52
.L50:
	movq	$28, -32(%rbp)
	jmp	.L52
.L42:
	call	next_i
	movl	%eax, -104(%rbp)
	call	next_i
	movl	%eax, -100(%rbp)
	movl	-104(%rbp), %eax
	cltd
	shrl	$31, %edx
	addl	%edx, %eax
	andl	$1, %eax
	subl	%edx, %eax
	movl	%eax, %edx
	movl	$0, %eax
	subl	%edx, %eax
	addl	%eax, %eax
	leal	1(%rax), %ecx
	movl	-144(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-48(%rbp), %rax
	addq	%rax, %rdx
	movl	%ecx, %eax
	imull	-100(%rbp), %eax
	movl	%eax, (%rdx)
	call	next_i
	movl	%eax, -96(%rbp)
	call	next_i
	movl	%eax, -92(%rbp)
	movl	-96(%rbp), %eax
	cltd
	shrl	$31, %edx
	addl	%edx, %eax
	andl	$1, %eax
	subl	%edx, %eax
	movl	%eax, %edx
	movl	$0, %eax
	subl	%edx, %eax
	addl	%eax, %eax
	leal	1(%rax), %ecx
	movl	-144(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-48(%rbp), %rax
	addq	%rax, %rdx
	movl	%ecx, %eax
	imull	-92(%rbp), %eax
	movl	%eax, 4(%rdx)
	call	next_i
	movl	%eax, -88(%rbp)
	call	next_i
	movl	%eax, -84(%rbp)
	movl	-88(%rbp), %eax
	cltd
	shrl	$31, %edx
	addl	%edx, %eax
	andl	$1, %eax
	subl	%edx, %eax
	movl	%eax, %edx
	movl	$0, %eax
	subl	%edx, %eax
	addl	%eax, %eax
	leal	1(%rax), %ecx
	movl	-144(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-48(%rbp), %rax
	addq	%rax, %rdx
	movl	%ecx, %eax
	imull	-84(%rbp), %eax
	movl	%eax, 8(%rdx)
	addl	$1, -144(%rbp)
	movq	$30, -32(%rbp)
	jmp	.L52
.L41:
	call	next_i
	movl	%eax, -128(%rbp)
	call	next_i
	movl	%eax, -124(%rbp)
	movl	-128(%rbp), %eax
	cltd
	shrl	$31, %edx
	addl	%edx, %eax
	andl	$1, %eax
	subl	%edx, %eax
	movl	%eax, %edx
	movl	$0, %eax
	subl	%edx, %eax
	addl	%eax, %eax
	leal	1(%rax), %ecx
	movl	-136(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-40(%rbp), %rax
	addq	%rax, %rdx
	movl	%ecx, %eax
	imull	-124(%rbp), %eax
	movl	%eax, (%rdx)
	call	next_i
	movl	%eax, -120(%rbp)
	call	next_i
	movl	%eax, -116(%rbp)
	movl	-120(%rbp), %eax
	cltd
	shrl	$31, %edx
	addl	%edx, %eax
	andl	$1, %eax
	subl	%edx, %eax
	movl	%eax, %edx
	movl	$0, %eax
	subl	%edx, %eax
	addl	%eax, %eax
	leal	1(%rax), %ecx
	movl	-136(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-40(%rbp), %rax
	addq	%rax, %rdx
	movl	%ecx, %eax
	imull	-116(%rbp), %eax
	movl	%eax, 4(%rdx)
	call	next_i
	movl	%eax, -112(%rbp)
	call	next_i
	movl	%eax, -108(%rbp)
	movl	-112(%rbp), %eax
	cltd
	shrl	$31, %edx
	addl	%edx, %eax
	andl	$1, %eax
	subl	%edx, %eax
	movl	%eax, %edx
	movl	$0, %eax
	subl	%edx, %eax
	addl	%eax, %eax
	leal	1(%rax), %ecx
	movl	-136(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-40(%rbp), %rax
	addq	%rax, %rdx
	movl	%ecx, %eax
	imull	-108(%rbp), %eax
	movl	%eax, 8(%rdx)
	addl	$1, -136(%rbp)
	movq	$26, -32(%rbp)
	jmp	.L52
.L44:
	movl	$1, %eax
	jmp	.L53
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
	leal	1(%rax), %ecx
	movl	-152(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-56(%rbp), %rax
	addq	%rax, %rdx
	movl	%ecx, %eax
	imull	-76(%rbp), %eax
	movl	%eax, (%rdx)
	call	next_i
	movl	%eax, -72(%rbp)
	call	next_i
	movl	%eax, -68(%rbp)
	movl	-72(%rbp), %eax
	cltd
	shrl	$31, %edx
	addl	%edx, %eax
	andl	$1, %eax
	subl	%edx, %eax
	movl	%eax, %edx
	movl	$0, %eax
	subl	%edx, %eax
	addl	%eax, %eax
	leal	1(%rax), %ecx
	movl	-152(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-56(%rbp), %rax
	addq	%rax, %rdx
	movl	%ecx, %eax
	imull	-68(%rbp), %eax
	movl	%eax, 4(%rdx)
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
	leal	1(%rax), %ecx
	movl	-152(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-56(%rbp), %rax
	addq	%rax, %rdx
	movl	%ecx, %eax
	imull	-60(%rbp), %eax
	movl	%eax, 8(%rdx)
	addl	$1, -152(%rbp)
	movq	$33, -32(%rbp)
	jmp	.L52
.L40:
	movl	$0, %eax
	jmp	.L53
.L36:
	movq	-176(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movl	%eax, -132(%rbp)
	movl	-132(%rbp), %eax
	movl	%eax, -160(%rbp)
	movq	$9, -32(%rbp)
	jmp	.L52
.L38:
	call	usage
	movq	$12, -32(%rbp)
	jmp	.L52
.L35:
	movl	-136(%rbp), %eax
	cmpl	-140(%rbp), %eax
	jge	.L54
	movq	$15, -32(%rbp)
	jmp	.L52
.L54:
	movq	$2, -32(%rbp)
	jmp	.L52
.L46:
	cmpl	$2, -160(%rbp)
	je	.L56
	cmpl	$2, -160(%rbp)
	jg	.L57
	cmpl	$0, -160(%rbp)
	je	.L58
	cmpl	$1, -160(%rbp)
	je	.L59
	jmp	.L57
.L56:
	movq	$13, -32(%rbp)
	jmp	.L60
.L59:
	movq	$27, -32(%rbp)
	jmp	.L60
.L58:
	movq	$20, -32(%rbp)
	jmp	.L60
.L57:
	movq	$10, -32(%rbp)
	nop
.L60:
	jmp	.L52
.L43:
	movl	$1, -140(%rbp)
	movl	-140(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -40(%rbp)
	movl	$0, -136(%rbp)
	movq	$26, -32(%rbp)
	jmp	.L52
.L34:
	movl	$100, -148(%rbp)
	movl	-148(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, -48(%rbp)
	movl	$0, -144(%rbp)
	movq	$30, -32(%rbp)
	jmp	.L52
.L37:
	cmpl	$2, -164(%rbp)
	je	.L61
	movq	$21, -32(%rbp)
	jmp	.L52
.L61:
	movq	$24, -32(%rbp)
	jmp	.L52
.L33:
	movq	-48(%rbp), %rax
	movq	%rax, %rdi
	call	mlx5e_set_inner_ttc_ft_params
	movq	-48(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$16, -32(%rbp)
	jmp	.L52
.L30:
	movl	-152(%rbp), %eax
	cmpl	-156(%rbp), %eax
	jge	.L63
	movq	$1, -32(%rbp)
	jmp	.L52
.L63:
	movq	$7, -32(%rbp)
	jmp	.L52
.L45:
	call	usage
	movq	$16, -32(%rbp)
	jmp	.L52
.L47:
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	mlx5e_set_inner_ttc_ft_params
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$16, -32(%rbp)
	jmp	.L52
.L48:
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	mlx5e_set_inner_ttc_ft_params
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$16, -32(%rbp)
	jmp	.L52
.L39:
	movl	$65025, -156(%rbp)
	movl	-156(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -56(%rbp)
	movl	$0, -152(%rbp)
	movq	$33, -32(%rbp)
	jmp	.L52
.L66:
	nop
.L52:
	jmp	.L65
.L53:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	main, .-main
	.globl	mlx5e_set_inner_ttc_ft_params
	.type	mlx5e_set_inner_ttc_ft_params, @function
mlx5e_set_inner_ttc_ft_params:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movq	$1, -16(%rbp)
.L73:
	cmpq	$2, -16(%rbp)
	je	.L68
	cmpq	$2, -16(%rbp)
	ja	.L74
	cmpq	$0, -16(%rbp)
	je	.L75
	cmpq	$1, -16(%rbp)
	jne	.L74
	movq	$2, -16(%rbp)
	jmp	.L71
.L68:
	movq	-24(%rbp), %rax
	movq	%rax, -8(%rbp)
	movl	MLX5E_INNER_TTC_TABLE_SIZE(%rip), %edx
	movq	-8(%rbp), %rax
	movl	%edx, 8(%rax)
	movl	MLX5E_INNER_TTC_FT_LEVEL(%rip), %edx
	movq	-8(%rbp), %rax
	movl	%edx, 4(%rax)
	movl	MLX5E_NIC_PRIO(%rip), %edx
	movq	-8(%rbp), %rax
	movl	%edx, (%rax)
	movq	$0, -16(%rbp)
	jmp	.L71
.L74:
	nop
.L71:
	jmp	.L73
.L75:
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	mlx5e_set_inner_ttc_ft_params, .-mlx5e_set_inner_ttc_ft_params
	.globl	next_i
	.type	next_i, @function
next_i:
.LFB10:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	$1, -8(%rbp)
.L82:
	cmpq	$2, -8(%rbp)
	je	.L77
	cmpq	$2, -8(%rbp)
	ja	.L84
	cmpq	$0, -8(%rbp)
	je	.L79
	cmpq	$1, -8(%rbp)
	jne	.L84
	movq	$2, -8(%rbp)
	jmp	.L80
.L79:
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
	jmp	.L83
.L77:
	movl	$0, -12(%rbp)
	addl	$1, -12(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L80
.L84:
	nop
.L80:
	jmp	.L82
.L83:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	next_i, .-next_i
	.section	.rodata
	.align 4
.LC0:
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
