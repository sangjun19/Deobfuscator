	.file	"lac-dcc_jotai-benchmarks_extr_FFmpeglibavutilavsscanf.c_ffshlim_Final_flatten.c"
	.text
	.globl	_TIG_IZ_tA2G_envp
	.bss
	.align 8
	.type	_TIG_IZ_tA2G_envp, @object
	.size	_TIG_IZ_tA2G_envp, 8
_TIG_IZ_tA2G_envp:
	.zero	8
	.globl	_TIG_IZ_tA2G_argc
	.align 4
	.type	_TIG_IZ_tA2G_argc, @object
	.size	_TIG_IZ_tA2G_argc, 4
_TIG_IZ_tA2G_argc:
	.zero	4
	.globl	rand_primes
	.align 32
	.type	rand_primes, @object
	.size	rand_primes, 100
rand_primes:
	.zero	100
	.globl	_TIG_IZ_tA2G_argv
	.align 8
	.type	_TIG_IZ_tA2G_argv, @object
	.size	_TIG_IZ_tA2G_argv, 8
_TIG_IZ_tA2G_argv:
	.zero	8
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
	movq	$1, -8(%rbp)
.L9:
	cmpq	$2, -8(%rbp)
	je	.L2
	cmpq	$2, -8(%rbp)
	ja	.L11
	cmpq	$0, -8(%rbp)
	je	.L4
	cmpq	$1, -8(%rbp)
	jne	.L11
	movq	$2, -8(%rbp)
	jmp	.L5
.L4:
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
	js	.L6
	pxor	%xmm0, %xmm0
	cvtsi2ssq	%rax, %xmm0
	jmp	.L7
.L6:
	movq	%rax, %rdx
	shrq	%rdx
	andl	$1, %eax
	orq	%rax, %rdx
	pxor	%xmm0, %xmm0
	cvtsi2ssq	%rdx, %xmm0
	addss	%xmm0, %xmm0
.L7:
	movss	.LC0(%rip), %xmm1
	divss	%xmm1, %xmm0
	jmp	.L10
.L2:
	movl	$0, -12(%rbp)
	addl	$1, -12(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L5
.L11:
	nop
.L5:
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
	.string	"Usage:\n    prog [ARGS]\n\nARGS:\n       0            int-bounds\n       1            big-arr\n       2            big-arr-10x\n       3            empty\n"
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
.L17:
	cmpq	$0, -8(%rbp)
	je	.L18
	cmpq	$1, -8(%rbp)
	jne	.L19
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$0, -8(%rbp)
	jmp	.L15
.L19:
	nop
.L15:
	jmp	.L17
.L18:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	usage, .-usage
	.globl	next_i
	.type	next_i, @function
next_i:
.LFB4:
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
	je	.L21
	cmpq	$2, -8(%rbp)
	ja	.L28
	cmpq	$0, -8(%rbp)
	je	.L23
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
	jmp	.L27
.L23:
	movl	$0, -12(%rbp)
	addl	$1, -12(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L25
.L21:
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
.LFE4:
	.size	next_i, .-next_i
	.globl	main
	.type	main, @function
main:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$384, %rsp
	movl	%edi, -356(%rbp)
	movq	%rsi, -368(%rbp)
	movq	%rdx, -376(%rbp)
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
	movq	$0, _TIG_IZ_tA2G_envp(%rip)
	nop
.L31:
	movq	$0, _TIG_IZ_tA2G_argv(%rip)
	nop
.L32:
	movl	$0, _TIG_IZ_tA2G_argc(%rip)
	nop
	nop
.L33:
.L34:
#APP
# 174 "lac-dcc_jotai-benchmarks_extr_FFmpeglibavutilavsscanf.c_ffshlim_Final.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-tA2G--0
# 0 "" 2
#NO_APP
	movl	-356(%rbp), %eax
	movl	%eax, _TIG_IZ_tA2G_argc(%rip)
	movq	-368(%rbp), %rax
	movq	%rax, _TIG_IZ_tA2G_argv(%rip)
	movq	-376(%rbp), %rax
	movq	%rax, _TIG_IZ_tA2G_envp(%rip)
	nop
	movq	$37, -40(%rbp)
.L78:
	cmpq	$46, -40(%rbp)
	ja	.L79
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
	.long	.L59-.L37
	.long	.L79-.L37
	.long	.L79-.L37
	.long	.L58-.L37
	.long	.L79-.L37
	.long	.L79-.L37
	.long	.L79-.L37
	.long	.L79-.L37
	.long	.L57-.L37
	.long	.L79-.L37
	.long	.L79-.L37
	.long	.L56-.L37
	.long	.L79-.L37
	.long	.L55-.L37
	.long	.L79-.L37
	.long	.L54-.L37
	.long	.L53-.L37
	.long	.L52-.L37
	.long	.L51-.L37
	.long	.L79-.L37
	.long	.L50-.L37
	.long	.L79-.L37
	.long	.L49-.L37
	.long	.L48-.L37
	.long	.L47-.L37
	.long	.L46-.L37
	.long	.L79-.L37
	.long	.L79-.L37
	.long	.L45-.L37
	.long	.L79-.L37
	.long	.L44-.L37
	.long	.L43-.L37
	.long	.L79-.L37
	.long	.L79-.L37
	.long	.L42-.L37
	.long	.L79-.L37
	.long	.L79-.L37
	.long	.L41-.L37
	.long	.L79-.L37
	.long	.L40-.L37
	.long	.L79-.L37
	.long	.L79-.L37
	.long	.L39-.L37
	.long	.L79-.L37
	.long	.L38-.L37
	.long	.L79-.L37
	.long	.L36-.L37
	.text
.L51:
	movl	-320(%rbp), %eax
	cmpl	-324(%rbp), %eax
	jge	.L60
	movq	$25, -40(%rbp)
	jmp	.L62
.L60:
	movq	$22, -40(%rbp)
	jmp	.L62
.L46:
	call	next_i
	movl	%eax, -248(%rbp)
	call	next_i
	movl	%eax, -244(%rbp)
	movl	-248(%rbp), %eax
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
	imull	-244(%rbp), %eax
	movl	%eax, %ecx
	movl	-320(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$4, %rax
	movq	%rax, %rdx
	movq	-64(%rbp), %rax
	addq	%rax, %rdx
	movslq	%ecx, %rax
	movq	%rax, (%rdx)
	call	next_i
	movl	%eax, -240(%rbp)
	call	next_i
	movl	%eax, -236(%rbp)
	movl	-240(%rbp), %eax
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
	imull	-236(%rbp), %eax
	movl	%eax, %ecx
	movl	-320(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$4, %rax
	movq	%rax, %rdx
	movq	-64(%rbp), %rax
	addq	%rax, %rdx
	movslq	%ecx, %rax
	movq	%rax, 8(%rdx)
	call	next_i
	movl	%eax, -232(%rbp)
	call	next_i
	movl	%eax, -228(%rbp)
	movl	-232(%rbp), %eax
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
	imull	-228(%rbp), %eax
	movl	%eax, %ecx
	movl	-320(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$4, %rax
	movq	%rax, %rdx
	movq	-64(%rbp), %rax
	addq	%rax, %rdx
	movslq	%ecx, %rax
	movq	%rax, 16(%rdx)
	call	next_i
	movl	%eax, -224(%rbp)
	call	next_i
	movl	%eax, -220(%rbp)
	movl	-224(%rbp), %eax
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
	imull	-220(%rbp), %eax
	movl	%eax, %ecx
	movl	-320(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$4, %rax
	movq	%rax, %rdx
	movq	-64(%rbp), %rax
	addq	%rax, %rdx
	movslq	%ecx, %rax
	movq	%rax, 24(%rdx)
	call	next_i
	movl	%eax, -216(%rbp)
	call	next_i
	movl	%eax, -212(%rbp)
	movl	-216(%rbp), %eax
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
	imull	-212(%rbp), %eax
	movl	%eax, %ecx
	movl	-320(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$4, %rax
	movq	%rax, %rdx
	movq	-64(%rbp), %rax
	addq	%rax, %rdx
	movslq	%ecx, %rax
	movq	%rax, 32(%rdx)
	call	next_i
	movl	%eax, -208(%rbp)
	call	next_i
	movl	%eax, -204(%rbp)
	movl	-208(%rbp), %eax
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
	imull	-204(%rbp), %eax
	movl	%eax, %ecx
	movl	-320(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$4, %rax
	movq	%rax, %rdx
	movq	-64(%rbp), %rax
	addq	%rax, %rdx
	movslq	%ecx, %rax
	movq	%rax, 40(%rdx)
	addl	$1, -320(%rbp)
	movq	$18, -40(%rbp)
	jmp	.L62
.L44:
	movq	-56(%rbp), %rdx
	movq	-48(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	ffshlim
	movq	-48(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$28, -40(%rbp)
	jmp	.L62
.L54:
	movq	$100, -104(%rbp)
	movl	$1, -340(%rbp)
	movl	-340(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$4, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -96(%rbp)
	movl	$0, -336(%rbp)
	movq	$0, -40(%rbp)
	jmp	.L62
.L43:
	call	next_i
	movl	%eax, -256(%rbp)
	call	next_i
	movl	%eax, -252(%rbp)
	movl	-256(%rbp), %eax
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
	imull	-252(%rbp), %eax
	cltq
	movq	%rax, -56(%rbp)
	movl	$1, -316(%rbp)
	movl	-316(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$4, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -32(%rbp)
	movq	-32(%rbp), %rax
	movq	%rax, -48(%rbp)
	movl	$0, -312(%rbp)
	movq	$23, -40(%rbp)
	jmp	.L62
.L57:
	cmpl	$3, -344(%rbp)
	je	.L63
	cmpl	$3, -344(%rbp)
	jg	.L64
	cmpl	$2, -344(%rbp)
	je	.L65
	cmpl	$2, -344(%rbp)
	jg	.L64
	cmpl	$0, -344(%rbp)
	je	.L66
	cmpl	$1, -344(%rbp)
	je	.L67
	jmp	.L64
.L63:
	movq	$31, -40(%rbp)
	jmp	.L68
.L65:
	movq	$16, -40(%rbp)
	jmp	.L68
.L67:
	movq	$17, -40(%rbp)
	jmp	.L68
.L66:
	movq	$15, -40(%rbp)
	jmp	.L68
.L64:
	movq	$42, -40(%rbp)
	nop
.L68:
	jmp	.L62
.L48:
	movl	-312(%rbp), %eax
	cmpl	-316(%rbp), %eax
	jge	.L69
	movq	$3, -40(%rbp)
	jmp	.L62
.L69:
	movq	$30, -40(%rbp)
	jmp	.L62
.L58:
	call	next_i
	movl	%eax, -152(%rbp)
	call	next_i
	movl	%eax, -148(%rbp)
	movl	-152(%rbp), %eax
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
	imull	-148(%rbp), %eax
	movl	%eax, %ecx
	movl	-312(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$4, %rax
	movq	%rax, %rdx
	movq	-48(%rbp), %rax
	addq	%rax, %rdx
	movslq	%ecx, %rax
	movq	%rax, (%rdx)
	call	next_i
	movl	%eax, -144(%rbp)
	call	next_i
	movl	%eax, -140(%rbp)
	movl	-144(%rbp), %eax
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
	imull	-140(%rbp), %eax
	movl	%eax, %ecx
	movl	-312(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$4, %rax
	movq	%rax, %rdx
	movq	-48(%rbp), %rax
	addq	%rax, %rdx
	movslq	%ecx, %rax
	movq	%rax, 8(%rdx)
	call	next_i
	movl	%eax, -136(%rbp)
	call	next_i
	movl	%eax, -132(%rbp)
	movl	-136(%rbp), %eax
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
	imull	-132(%rbp), %eax
	movl	%eax, %ecx
	movl	-312(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$4, %rax
	movq	%rax, %rdx
	movq	-48(%rbp), %rax
	addq	%rax, %rdx
	movslq	%ecx, %rax
	movq	%rax, 16(%rdx)
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
	addl	$1, %eax
	imull	-124(%rbp), %eax
	movl	%eax, %ecx
	movl	-312(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$4, %rax
	movq	%rax, %rdx
	movq	-48(%rbp), %rax
	addq	%rax, %rdx
	movslq	%ecx, %rax
	movq	%rax, 24(%rdx)
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
	addl	$1, %eax
	imull	-116(%rbp), %eax
	movl	%eax, %ecx
	movl	-312(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$4, %rax
	movq	%rax, %rdx
	movq	-48(%rbp), %rax
	addq	%rax, %rdx
	movslq	%ecx, %rax
	movq	%rax, 32(%rdx)
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
	addl	$1, %eax
	imull	-108(%rbp), %eax
	movl	%eax, %ecx
	movl	-312(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$4, %rax
	movq	%rax, %rdx
	movq	-48(%rbp), %rax
	addq	%rax, %rdx
	movslq	%ecx, %rax
	movq	%rax, 40(%rdx)
	addl	$1, -312(%rbp)
	movq	$23, -40(%rbp)
	jmp	.L62
.L53:
	movq	$10, -72(%rbp)
	movl	$100, -324(%rbp)
	movl	-324(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$4, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -64(%rbp)
	movl	$0, -320(%rbp)
	movq	$18, -40(%rbp)
	jmp	.L62
.L47:
	movq	-104(%rbp), %rdx
	movq	-96(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	ffshlim
	movq	-96(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$28, -40(%rbp)
	jmp	.L62
.L56:
	call	next_i
	movl	%eax, -200(%rbp)
	call	next_i
	movl	%eax, -196(%rbp)
	movl	-200(%rbp), %eax
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
	imull	-196(%rbp), %eax
	movl	%eax, %ecx
	movl	-336(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$4, %rax
	movq	%rax, %rdx
	movq	-96(%rbp), %rax
	addq	%rax, %rdx
	movslq	%ecx, %rax
	movq	%rax, (%rdx)
	call	next_i
	movl	%eax, -192(%rbp)
	call	next_i
	movl	%eax, -188(%rbp)
	movl	-192(%rbp), %eax
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
	imull	-188(%rbp), %eax
	movl	%eax, %ecx
	movl	-336(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$4, %rax
	movq	%rax, %rdx
	movq	-96(%rbp), %rax
	addq	%rax, %rdx
	movslq	%ecx, %rax
	movq	%rax, 8(%rdx)
	call	next_i
	movl	%eax, -184(%rbp)
	call	next_i
	movl	%eax, -180(%rbp)
	movl	-184(%rbp), %eax
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
	imull	-180(%rbp), %eax
	movl	%eax, %ecx
	movl	-336(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$4, %rax
	movq	%rax, %rdx
	movq	-96(%rbp), %rax
	addq	%rax, %rdx
	movslq	%ecx, %rax
	movq	%rax, 16(%rdx)
	call	next_i
	movl	%eax, -176(%rbp)
	call	next_i
	movl	%eax, -172(%rbp)
	movl	-176(%rbp), %eax
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
	imull	-172(%rbp), %eax
	movl	%eax, %ecx
	movl	-336(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$4, %rax
	movq	%rax, %rdx
	movq	-96(%rbp), %rax
	addq	%rax, %rdx
	movslq	%ecx, %rax
	movq	%rax, 24(%rdx)
	call	next_i
	movl	%eax, -168(%rbp)
	call	next_i
	movl	%eax, -164(%rbp)
	movl	-168(%rbp), %eax
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
	imull	-164(%rbp), %eax
	movl	%eax, %ecx
	movl	-336(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$4, %rax
	movq	%rax, %rdx
	movq	-96(%rbp), %rax
	addq	%rax, %rdx
	movslq	%ecx, %rax
	movq	%rax, 32(%rdx)
	call	next_i
	movl	%eax, -160(%rbp)
	call	next_i
	movl	%eax, -156(%rbp)
	movl	-160(%rbp), %eax
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
	imull	-156(%rbp), %eax
	movl	%eax, %ecx
	movl	-336(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$4, %rax
	movq	%rax, %rdx
	movq	-96(%rbp), %rax
	addq	%rax, %rdx
	movslq	%ecx, %rax
	movq	%rax, 40(%rdx)
	addl	$1, -336(%rbp)
	movq	$0, -40(%rbp)
	jmp	.L62
.L55:
	call	usage
	movq	$44, -40(%rbp)
	jmp	.L62
.L52:
	movq	$255, -88(%rbp)
	movl	$65025, -332(%rbp)
	movl	-332(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$4, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, -80(%rbp)
	movl	$0, -328(%rbp)
	movq	$20, -40(%rbp)
	jmp	.L62
.L42:
	movq	-368(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movl	%eax, -260(%rbp)
	movl	-260(%rbp), %eax
	movl	%eax, -344(%rbp)
	movq	$8, -40(%rbp)
	jmp	.L62
.L49:
	movq	-72(%rbp), %rdx
	movq	-64(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	ffshlim
	movq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$28, -40(%rbp)
	jmp	.L62
.L45:
	movl	$0, %eax
	jmp	.L71
.L38:
	movl	$1, %eax
	jmp	.L71
.L41:
	cmpl	$2, -356(%rbp)
	je	.L72
	movq	$13, -40(%rbp)
	jmp	.L62
.L72:
	movq	$34, -40(%rbp)
	jmp	.L62
.L39:
	call	usage
	movq	$28, -40(%rbp)
	jmp	.L62
.L59:
	movl	-336(%rbp), %eax
	cmpl	-340(%rbp), %eax
	jge	.L74
	movq	$11, -40(%rbp)
	jmp	.L62
.L74:
	movq	$24, -40(%rbp)
	jmp	.L62
.L36:
	movq	-88(%rbp), %rdx
	movq	-80(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	ffshlim
	movq	-80(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$28, -40(%rbp)
	jmp	.L62
.L40:
	call	next_i
	movl	%eax, -308(%rbp)
	call	next_i
	movl	%eax, -304(%rbp)
	movl	-308(%rbp), %eax
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
	imull	-304(%rbp), %eax
	movl	%eax, %ecx
	movl	-328(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$4, %rax
	movq	%rax, %rdx
	movq	-80(%rbp), %rax
	addq	%rax, %rdx
	movslq	%ecx, %rax
	movq	%rax, (%rdx)
	call	next_i
	movl	%eax, -300(%rbp)
	call	next_i
	movl	%eax, -296(%rbp)
	movl	-300(%rbp), %eax
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
	imull	-296(%rbp), %eax
	movl	%eax, %ecx
	movl	-328(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$4, %rax
	movq	%rax, %rdx
	movq	-80(%rbp), %rax
	addq	%rax, %rdx
	movslq	%ecx, %rax
	movq	%rax, 8(%rdx)
	call	next_i
	movl	%eax, -292(%rbp)
	call	next_i
	movl	%eax, -288(%rbp)
	movl	-292(%rbp), %eax
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
	imull	-288(%rbp), %eax
	movl	%eax, %ecx
	movl	-328(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$4, %rax
	movq	%rax, %rdx
	movq	-80(%rbp), %rax
	addq	%rax, %rdx
	movslq	%ecx, %rax
	movq	%rax, 16(%rdx)
	call	next_i
	movl	%eax, -284(%rbp)
	call	next_i
	movl	%eax, -280(%rbp)
	movl	-284(%rbp), %eax
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
	imull	-280(%rbp), %eax
	movl	%eax, %ecx
	movl	-328(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$4, %rax
	movq	%rax, %rdx
	movq	-80(%rbp), %rax
	addq	%rax, %rdx
	movslq	%ecx, %rax
	movq	%rax, 24(%rdx)
	call	next_i
	movl	%eax, -276(%rbp)
	call	next_i
	movl	%eax, -272(%rbp)
	movl	-276(%rbp), %eax
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
	imull	-272(%rbp), %eax
	movl	%eax, %ecx
	movl	-328(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$4, %rax
	movq	%rax, %rdx
	movq	-80(%rbp), %rax
	addq	%rax, %rdx
	movslq	%ecx, %rax
	movq	%rax, 32(%rdx)
	call	next_i
	movl	%eax, -268(%rbp)
	call	next_i
	movl	%eax, -264(%rbp)
	movl	-268(%rbp), %eax
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
	imull	-264(%rbp), %eax
	movl	%eax, %ecx
	movl	-328(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$4, %rax
	movq	%rax, %rdx
	movq	-80(%rbp), %rax
	addq	%rax, %rdx
	movslq	%ecx, %rax
	movq	%rax, 40(%rdx)
	addl	$1, -328(%rbp)
	movq	$20, -40(%rbp)
	jmp	.L62
.L50:
	movl	-328(%rbp), %eax
	cmpl	-332(%rbp), %eax
	jge	.L76
	movq	$39, -40(%rbp)
	jmp	.L62
.L76:
	movq	$46, -40(%rbp)
	jmp	.L62
.L79:
	nop
.L62:
	jmp	.L78
.L71:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	main, .-main
	.type	ffshlim, @function
ffshlim:
.LFB10:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	$3, -8(%rbp)
.L97:
	cmpq	$7, -8(%rbp)
	ja	.L98
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L83(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L83(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L83:
	.long	.L90-.L83
	.long	.L89-.L83
	.long	.L88-.L83
	.long	.L87-.L83
	.long	.L99-.L83
	.long	.L85-.L83
	.long	.L84-.L83
	.long	.L82-.L83
	.text
.L89:
	cmpq	$0, -32(%rbp)
	je	.L92
	movq	$5, -8(%rbp)
	jmp	.L94
.L92:
	movq	$2, -8(%rbp)
	jmp	.L94
.L87:
	movq	$0, -8(%rbp)
	jmp	.L94
.L84:
	movq	-24(%rbp), %rax
	movq	16(%rax), %rdx
	movq	-32(%rbp), %rax
	addq	%rax, %rdx
	movq	-24(%rbp), %rax
	movq	%rdx, 8(%rax)
	movq	$4, -8(%rbp)
	jmp	.L94
.L85:
	movq	-24(%rbp), %rax
	movq	(%rax), %rdx
	movq	-24(%rbp), %rax
	movq	16(%rax), %rcx
	movq	%rdx, %rax
	subq	%rcx, %rax
	cmpq	%rax, -32(%rbp)
	jge	.L95
	movq	$6, -8(%rbp)
	jmp	.L94
.L95:
	movq	$7, -8(%rbp)
	jmp	.L94
.L90:
	movq	-24(%rbp), %rax
	movq	-32(%rbp), %rdx
	movq	%rdx, 40(%rax)
	movq	-24(%rbp), %rax
	movq	24(%rax), %rdx
	movq	-24(%rbp), %rax
	movq	16(%rax), %rax
	subq	%rax, %rdx
	movq	-24(%rbp), %rax
	movq	%rdx, 32(%rax)
	movq	$1, -8(%rbp)
	jmp	.L94
.L82:
	movq	-24(%rbp), %rax
	movq	(%rax), %rdx
	movq	-24(%rbp), %rax
	movq	%rdx, 8(%rax)
	movq	$4, -8(%rbp)
	jmp	.L94
.L88:
	movq	-24(%rbp), %rax
	movq	(%rax), %rdx
	movq	-24(%rbp), %rax
	movq	%rdx, 8(%rax)
	movq	$4, -8(%rbp)
	jmp	.L94
.L98:
	nop
.L94:
	jmp	.L97
.L99:
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	ffshlim, .-ffshlim
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
