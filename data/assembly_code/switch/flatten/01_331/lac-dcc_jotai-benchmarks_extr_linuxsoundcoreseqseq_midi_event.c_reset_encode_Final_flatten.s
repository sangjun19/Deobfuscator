	.file	"lac-dcc_jotai-benchmarks_extr_linuxsoundcoreseqseq_midi_event.c_reset_encode_Final_flatten.c"
	.text
	.globl	_TIG_IZ_yKIw_envp
	.bss
	.align 8
	.type	_TIG_IZ_yKIw_envp, @object
	.size	_TIG_IZ_yKIw_envp, 8
_TIG_IZ_yKIw_envp:
	.zero	8
	.globl	rand_primes
	.align 32
	.type	rand_primes, @object
	.size	rand_primes, 100
rand_primes:
	.zero	100
	.globl	_TIG_IZ_yKIw_argv
	.align 8
	.type	_TIG_IZ_yKIw_argv, @object
	.size	_TIG_IZ_yKIw_argv, 8
_TIG_IZ_yKIw_argv:
	.zero	8
	.globl	_TIG_IZ_yKIw_argc
	.align 4
	.type	_TIG_IZ_yKIw_argc, @object
	.size	_TIG_IZ_yKIw_argc, 4
_TIG_IZ_yKIw_argc:
	.zero	4
	.globl	ST_INVALID
	.align 4
	.type	ST_INVALID, @object
	.size	ST_INVALID, 4
ST_INVALID:
	.zero	4
	.text
	.globl	next_f
	.type	next_f, @function
next_f:
.LFB2:
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
.LFE2:
	.size	next_f, .-next_f
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
	movq	$0, -8(%rbp)
.L18:
	cmpq	$2, -8(%rbp)
	je	.L13
	cmpq	$2, -8(%rbp)
	ja	.L20
	cmpq	$0, -8(%rbp)
	je	.L15
	cmpq	$1, -8(%rbp)
	jne	.L20
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
	jmp	.L19
.L15:
	movq	$2, -8(%rbp)
	jmp	.L17
.L13:
	movl	$0, -12(%rbp)
	addl	$1, -12(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L17
.L20:
	nop
.L17:
	jmp	.L18
.L19:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	next_i, .-next_i
	.type	reset_encode, @function
reset_encode:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movq	$1, -8(%rbp)
.L27:
	cmpq	$2, -8(%rbp)
	je	.L22
	cmpq	$2, -8(%rbp)
	ja	.L28
	cmpq	$0, -8(%rbp)
	je	.L29
	cmpq	$1, -8(%rbp)
	jne	.L28
	movq	$2, -8(%rbp)
	jmp	.L25
.L22:
	movq	-24(%rbp), %rax
	movq	$0, 16(%rax)
	movq	-24(%rbp), %rax
	movq	$0, 8(%rax)
	movl	ST_INVALID(%rip), %edx
	movq	-24(%rbp), %rax
	movl	%edx, (%rax)
	movq	$0, -8(%rbp)
	jmp	.L25
.L28:
	nop
.L25:
	jmp	.L27
.L29:
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	reset_encode, .-reset_encode
	.section	.rodata
	.align 8
.LC1:
	.string	"Usage:\n    prog [ARGS]\n\nARGS:\n       0            big-arr\n       1            big-arr-10x\n       2            empty\n"
	.text
	.globl	usage
	.type	usage, @function
usage:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$0, -8(%rbp)
.L35:
	cmpq	$0, -8(%rbp)
	je	.L31
	cmpq	$1, -8(%rbp)
	jne	.L37
	jmp	.L36
.L31:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$1, -8(%rbp)
	jmp	.L34
.L37:
	nop
.L34:
	jmp	.L35
.L36:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	usage, .-usage
	.globl	main
	.type	main, @function
main:
.LFB10:
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
	movl	$0, ST_INVALID(%rip)
	nop
.L39:
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
.L40:
	movq	$0, _TIG_IZ_yKIw_envp(%rip)
	nop
.L41:
	movq	$0, _TIG_IZ_yKIw_argv(%rip)
	nop
.L42:
	movl	$0, _TIG_IZ_yKIw_argc(%rip)
	nop
	nop
.L43:
.L44:
#APP
# 147 "lac-dcc_jotai-benchmarks_extr_linuxsoundcoreseqseq_midi_event.c_reset_encode_Final.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-yKIw--0
# 0 "" 2
#NO_APP
	movl	-164(%rbp), %eax
	movl	%eax, _TIG_IZ_yKIw_argc(%rip)
	movq	-176(%rbp), %rax
	movq	%rax, _TIG_IZ_yKIw_argv(%rip)
	movq	-184(%rbp), %rax
	movq	%rax, _TIG_IZ_yKIw_envp(%rip)
	nop
	movq	$22, -32(%rbp)
.L81:
	cmpq	$33, -32(%rbp)
	ja	.L82
	movq	-32(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L47(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L47(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L47:
	.long	.L82-.L47
	.long	.L65-.L47
	.long	.L64-.L47
	.long	.L82-.L47
	.long	.L82-.L47
	.long	.L82-.L47
	.long	.L82-.L47
	.long	.L63-.L47
	.long	.L82-.L47
	.long	.L62-.L47
	.long	.L61-.L47
	.long	.L82-.L47
	.long	.L60-.L47
	.long	.L59-.L47
	.long	.L58-.L47
	.long	.L57-.L47
	.long	.L56-.L47
	.long	.L82-.L47
	.long	.L82-.L47
	.long	.L82-.L47
	.long	.L55-.L47
	.long	.L54-.L47
	.long	.L53-.L47
	.long	.L82-.L47
	.long	.L52-.L47
	.long	.L82-.L47
	.long	.L51-.L47
	.long	.L50-.L47
	.long	.L49-.L47
	.long	.L82-.L47
	.long	.L48-.L47
	.long	.L82-.L47
	.long	.L82-.L47
	.long	.L46-.L47
	.text
.L48:
	movl	-144(%rbp), %eax
	cmpl	-148(%rbp), %eax
	jge	.L66
	movq	$14, -32(%rbp)
	jmp	.L68
.L66:
	movq	$28, -32(%rbp)
	jmp	.L68
.L58:
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
	salq	$3, %rax
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
	addl	$1, %eax
	imull	-92(%rbp), %eax
	movl	%eax, %ecx
	movl	-144(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-48(%rbp), %rax
	addq	%rax, %rdx
	movslq	%ecx, %rax
	movq	%rax, 8(%rdx)
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
	addl	$1, %eax
	imull	-84(%rbp), %eax
	movl	%eax, %ecx
	movl	-144(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-48(%rbp), %rax
	addq	%rax, %rdx
	movslq	%ecx, %rax
	movq	%rax, 16(%rdx)
	addl	$1, -144(%rbp)
	movq	$30, -32(%rbp)
	jmp	.L68
.L57:
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
	salq	$3, %rax
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
	addl	$1, %eax
	imull	-116(%rbp), %eax
	movl	%eax, %ecx
	movl	-136(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-40(%rbp), %rax
	addq	%rax, %rdx
	movslq	%ecx, %rax
	movq	%rax, 8(%rdx)
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
	movl	-136(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-40(%rbp), %rax
	addq	%rax, %rdx
	movslq	%ecx, %rax
	movq	%rax, 16(%rdx)
	addl	$1, -136(%rbp)
	movq	$26, -32(%rbp)
	jmp	.L68
.L60:
	movl	$1, %eax
	jmp	.L69
.L65:
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
	salq	$3, %rax
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
	addl	$1, %eax
	imull	-68(%rbp), %eax
	movl	%eax, %ecx
	movl	-152(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-56(%rbp), %rax
	addq	%rax, %rdx
	movslq	%ecx, %rax
	movq	%rax, 8(%rdx)
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
	imull	-60(%rbp), %eax
	movl	%eax, %ecx
	movl	-152(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-56(%rbp), %rax
	addq	%rax, %rdx
	movslq	%ecx, %rax
	movq	%rax, 16(%rdx)
	addl	$1, -152(%rbp)
	movq	$33, -32(%rbp)
	jmp	.L68
.L56:
	movl	$0, %eax
	jmp	.L69
.L52:
	movq	-176(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movl	%eax, -132(%rbp)
	movl	-132(%rbp), %eax
	movl	%eax, -160(%rbp)
	movq	$9, -32(%rbp)
	jmp	.L68
.L54:
	call	usage
	movq	$12, -32(%rbp)
	jmp	.L68
.L51:
	movl	-136(%rbp), %eax
	cmpl	-140(%rbp), %eax
	jge	.L70
	movq	$15, -32(%rbp)
	jmp	.L68
.L70:
	movq	$2, -32(%rbp)
	jmp	.L68
.L62:
	cmpl	$2, -160(%rbp)
	je	.L72
	cmpl	$2, -160(%rbp)
	jg	.L73
	cmpl	$0, -160(%rbp)
	je	.L74
	cmpl	$1, -160(%rbp)
	je	.L75
	jmp	.L73
.L72:
	movq	$13, -32(%rbp)
	jmp	.L76
.L75:
	movq	$27, -32(%rbp)
	jmp	.L76
.L74:
	movq	$20, -32(%rbp)
	jmp	.L76
.L73:
	movq	$10, -32(%rbp)
	nop
.L76:
	jmp	.L68
.L59:
	movl	$1, -140(%rbp)
	movl	-140(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -40(%rbp)
	movl	$0, -136(%rbp)
	movq	$26, -32(%rbp)
	jmp	.L68
.L50:
	movl	$100, -148(%rbp)
	movl	-148(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, -48(%rbp)
	movl	$0, -144(%rbp)
	movq	$30, -32(%rbp)
	jmp	.L68
.L53:
	cmpl	$2, -164(%rbp)
	je	.L77
	movq	$21, -32(%rbp)
	jmp	.L68
.L77:
	movq	$24, -32(%rbp)
	jmp	.L68
.L49:
	movq	-48(%rbp), %rax
	movq	%rax, %rdi
	call	reset_encode
	movq	-48(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$16, -32(%rbp)
	jmp	.L68
.L46:
	movl	-152(%rbp), %eax
	cmpl	-156(%rbp), %eax
	jge	.L79
	movq	$1, -32(%rbp)
	jmp	.L68
.L79:
	movq	$7, -32(%rbp)
	jmp	.L68
.L61:
	call	usage
	movq	$16, -32(%rbp)
	jmp	.L68
.L63:
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	reset_encode
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$16, -32(%rbp)
	jmp	.L68
.L64:
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	reset_encode
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$16, -32(%rbp)
	jmp	.L68
.L55:
	movl	$65025, -156(%rbp)
	movl	-156(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -56(%rbp)
	movl	$0, -152(%rbp)
	movq	$33, -32(%rbp)
	jmp	.L68
.L82:
	nop
.L68:
	jmp	.L81
.L69:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	main, .-main
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
