	.file	"lac-dcc_jotai-benchmarks_extr_linuxtoolsperfutilevsel.c_perf_evsel__read_size_Final_flatten.c"
	.text
	.globl	_TIG_IZ_XHF7_argc
	.bss
	.align 4
	.type	_TIG_IZ_XHF7_argc, @object
	.size	_TIG_IZ_XHF7_argc, 4
_TIG_IZ_XHF7_argc:
	.zero	4
	.globl	PERF_FORMAT_TOTAL_TIME_ENABLED
	.align 4
	.type	PERF_FORMAT_TOTAL_TIME_ENABLED, @object
	.size	PERF_FORMAT_TOTAL_TIME_ENABLED, 4
PERF_FORMAT_TOTAL_TIME_ENABLED:
	.zero	4
	.globl	_TIG_IZ_XHF7_envp
	.align 8
	.type	_TIG_IZ_XHF7_envp, @object
	.size	_TIG_IZ_XHF7_envp, 8
_TIG_IZ_XHF7_envp:
	.zero	8
	.globl	PERF_FORMAT_GROUP
	.align 4
	.type	PERF_FORMAT_GROUP, @object
	.size	PERF_FORMAT_GROUP, 4
PERF_FORMAT_GROUP:
	.zero	4
	.globl	rand_primes
	.align 32
	.type	rand_primes, @object
	.size	rand_primes, 100
rand_primes:
	.zero	100
	.globl	PERF_FORMAT_TOTAL_TIME_RUNNING
	.align 4
	.type	PERF_FORMAT_TOTAL_TIME_RUNNING, @object
	.size	PERF_FORMAT_TOTAL_TIME_RUNNING, 4
PERF_FORMAT_TOTAL_TIME_RUNNING:
	.zero	4
	.globl	_TIG_IZ_XHF7_argv
	.align 8
	.type	_TIG_IZ_XHF7_argv, @object
	.size	_TIG_IZ_XHF7_argv, 8
_TIG_IZ_XHF7_argv:
	.zero	8
	.globl	PERF_FORMAT_ID
	.align 4
	.type	PERF_FORMAT_ID, @object
	.size	PERF_FORMAT_ID, 4
PERF_FORMAT_ID:
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
	movl	$0, -12(%rbp)
	addl	$1, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L5
.L4:
	movq	$1, -8(%rbp)
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
.L18:
	cmpq	$2, -8(%rbp)
	je	.L13
	cmpq	$2, -8(%rbp)
	ja	.L20
	cmpq	$0, -8(%rbp)
	je	.L15
	cmpq	$1, -8(%rbp)
	jne	.L20
	movq	$0, -8(%rbp)
	jmp	.L16
.L15:
	movl	$0, -12(%rbp)
	addl	$1, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L16
.L13:
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
.L20:
	nop
.L16:
	jmp	.L18
.L19:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	next_i, .-next_i
	.type	perf_evsel__read_size, @function
perf_evsel__read_size:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -40(%rbp)
	movq	$4, -8(%rbp)
.L46:
	cmpq	$12, -8(%rbp)
	ja	.L48
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L24(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L24(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L24:
	.long	.L35-.L24
	.long	.L34-.L24
	.long	.L48-.L24
	.long	.L33-.L24
	.long	.L32-.L24
	.long	.L31-.L24
	.long	.L30-.L24
	.long	.L29-.L24
	.long	.L28-.L24
	.long	.L27-.L24
	.long	.L26-.L24
	.long	.L25-.L24
	.long	.L23-.L24
	.text
.L32:
	movq	$0, -8(%rbp)
	jmp	.L36
.L23:
	movq	-40(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, -12(%rbp)
	movl	-16(%rbp), %eax
	addl	$4, %eax
	movl	%eax, -16(%rbp)
	movq	$9, -8(%rbp)
	jmp	.L36
.L28:
	movl	PERF_FORMAT_GROUP(%rip), %eax
	andl	-24(%rbp), %eax
	testl	%eax, %eax
	je	.L37
	movq	$12, -8(%rbp)
	jmp	.L36
.L37:
	movq	$9, -8(%rbp)
	jmp	.L36
.L34:
	movl	-16(%rbp), %eax
	jmp	.L47
.L33:
	movl	-16(%rbp), %eax
	addl	$4, %eax
	movl	%eax, -16(%rbp)
	movq	$10, -8(%rbp)
	jmp	.L36
.L25:
	movl	PERF_FORMAT_ID(%rip), %eax
	andl	-24(%rbp), %eax
	testl	%eax, %eax
	je	.L40
	movq	$5, -8(%rbp)
	jmp	.L36
.L40:
	movq	$8, -8(%rbp)
	jmp	.L36
.L27:
	movl	-20(%rbp), %eax
	imull	-12(%rbp), %eax
	addl	%eax, -16(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L36
.L30:
	movl	-16(%rbp), %eax
	addl	$4, %eax
	movl	%eax, -16(%rbp)
	movq	$11, -8(%rbp)
	jmp	.L36
.L31:
	movl	-20(%rbp), %eax
	addl	$4, %eax
	movl	%eax, -20(%rbp)
	movq	$8, -8(%rbp)
	jmp	.L36
.L26:
	movl	PERF_FORMAT_TOTAL_TIME_RUNNING(%rip), %eax
	andl	-24(%rbp), %eax
	testl	%eax, %eax
	je	.L42
	movq	$6, -8(%rbp)
	jmp	.L36
.L42:
	movq	$11, -8(%rbp)
	jmp	.L36
.L35:
	movq	-40(%rbp), %rax
	movl	4(%rax), %eax
	movl	%eax, -24(%rbp)
	movl	$4, -20(%rbp)
	movl	$0, -16(%rbp)
	movl	$1, -12(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L36
.L29:
	movl	PERF_FORMAT_TOTAL_TIME_ENABLED(%rip), %eax
	andl	-24(%rbp), %eax
	testl	%eax, %eax
	je	.L44
	movq	$3, -8(%rbp)
	jmp	.L36
.L44:
	movq	$10, -8(%rbp)
	jmp	.L36
.L48:
	nop
.L36:
	jmp	.L46
.L47:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	perf_evsel__read_size, .-perf_evsel__read_size
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
.L54:
	cmpq	$0, -8(%rbp)
	je	.L50
	cmpq	$1, -8(%rbp)
	jne	.L56
	jmp	.L55
.L50:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$1, -8(%rbp)
	jmp	.L53
.L56:
	nop
.L53:
	jmp	.L54
.L55:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	usage, .-usage
	.section	.rodata
.LC2:
	.string	"%d\n"
	.text
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
	movl	$0, PERF_FORMAT_TOTAL_TIME_RUNNING(%rip)
	nop
.L58:
	movl	$0, PERF_FORMAT_TOTAL_TIME_ENABLED(%rip)
	nop
.L59:
	movl	$0, PERF_FORMAT_ID(%rip)
	nop
.L60:
	movl	$0, PERF_FORMAT_GROUP(%rip)
	nop
.L61:
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
.L62:
	movq	$0, _TIG_IZ_XHF7_envp(%rip)
	nop
.L63:
	movq	$0, _TIG_IZ_XHF7_argv(%rip)
	nop
.L64:
	movl	$0, _TIG_IZ_XHF7_argc(%rip)
	nop
	nop
.L65:
.L66:
#APP
# 168 "lac-dcc_jotai-benchmarks_extr_linuxtoolsperfutilevsel.c_perf_evsel__read_size_Final.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-XHF7--0
# 0 "" 2
#NO_APP
	movl	-164(%rbp), %eax
	movl	%eax, _TIG_IZ_XHF7_argc(%rip)
	movq	-176(%rbp), %rax
	movq	%rax, _TIG_IZ_XHF7_argv(%rip)
	movq	-184(%rbp), %rax
	movq	%rax, _TIG_IZ_XHF7_envp(%rip)
	nop
	movq	$33, -32(%rbp)
.L103:
	cmpq	$36, -32(%rbp)
	ja	.L104
	movq	-32(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L69(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L69(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L69:
	.long	.L87-.L69
	.long	.L104-.L69
	.long	.L104-.L69
	.long	.L104-.L69
	.long	.L86-.L69
	.long	.L104-.L69
	.long	.L85-.L69
	.long	.L84-.L69
	.long	.L83-.L69
	.long	.L82-.L69
	.long	.L81-.L69
	.long	.L80-.L69
	.long	.L79-.L69
	.long	.L78-.L69
	.long	.L104-.L69
	.long	.L104-.L69
	.long	.L104-.L69
	.long	.L77-.L69
	.long	.L76-.L69
	.long	.L75-.L69
	.long	.L104-.L69
	.long	.L74-.L69
	.long	.L73-.L69
	.long	.L104-.L69
	.long	.L104-.L69
	.long	.L72-.L69
	.long	.L104-.L69
	.long	.L104-.L69
	.long	.L104-.L69
	.long	.L104-.L69
	.long	.L104-.L69
	.long	.L104-.L69
	.long	.L104-.L69
	.long	.L71-.L69
	.long	.L70-.L69
	.long	.L104-.L69
	.long	.L68-.L69
	.text
.L76:
	movl	-152(%rbp), %eax
	cmpl	-156(%rbp), %eax
	jge	.L88
	movq	$34, -32(%rbp)
	jmp	.L90
.L88:
	movq	$36, -32(%rbp)
	jmp	.L90
.L72:
	movl	$65025, -156(%rbp)
	movl	-156(%rbp), %eax
	cltq
	salq	$3, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, -56(%rbp)
	movl	$0, -152(%rbp)
	movq	$18, -32(%rbp)
	jmp	.L90
.L86:
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
	movl	-144(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,8), %rcx
	movq	-48(%rbp), %rdx
	addq	%rcx, %rdx
	imull	-68(%rbp), %eax
	movl	%eax, (%rdx)
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
	movl	-144(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,8), %rcx
	movq	-48(%rbp), %rdx
	addq	%rcx, %rdx
	imull	-60(%rbp), %eax
	movl	%eax, 4(%rdx)
	addl	$1, -144(%rbp)
	movq	$0, -32(%rbp)
	jmp	.L90
.L79:
	movq	-176(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movl	%eax, -92(%rbp)
	movl	-92(%rbp), %eax
	movl	%eax, -160(%rbp)
	movq	$8, -32(%rbp)
	jmp	.L90
.L83:
	cmpl	$2, -160(%rbp)
	je	.L91
	cmpl	$2, -160(%rbp)
	jg	.L92
	cmpl	$0, -160(%rbp)
	je	.L93
	cmpl	$1, -160(%rbp)
	je	.L94
	jmp	.L92
.L91:
	movq	$11, -32(%rbp)
	jmp	.L95
.L94:
	movq	$6, -32(%rbp)
	jmp	.L95
.L93:
	movq	$25, -32(%rbp)
	jmp	.L95
.L92:
	movq	$19, -32(%rbp)
	nop
.L95:
	jmp	.L90
.L74:
	movl	$1, %eax
	jmp	.L96
.L68:
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	perf_evsel__read_size
	movl	%eax, -132(%rbp)
	movl	-132(%rbp), %eax
	movl	%eax, -128(%rbp)
	movl	-128(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$9, -32(%rbp)
	jmp	.L90
.L80:
	movl	$1, -140(%rbp)
	movl	-140(%rbp), %eax
	cltq
	salq	$3, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -40(%rbp)
	movl	$0, -136(%rbp)
	movq	$17, -32(%rbp)
	jmp	.L90
.L82:
	movl	$0, %eax
	jmp	.L96
.L78:
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	perf_evsel__read_size
	movl	%eax, -100(%rbp)
	movl	-100(%rbp), %eax
	movl	%eax, -96(%rbp)
	movl	-96(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$9, -32(%rbp)
	jmp	.L90
.L75:
	call	usage
	movq	$9, -32(%rbp)
	jmp	.L90
.L77:
	movl	-136(%rbp), %eax
	cmpl	-140(%rbp), %eax
	jge	.L97
	movq	$10, -32(%rbp)
	jmp	.L90
.L97:
	movq	$13, -32(%rbp)
	jmp	.L90
.L85:
	movl	$100, -148(%rbp)
	movl	-148(%rbp), %eax
	cltq
	salq	$3, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -48(%rbp)
	movl	$0, -144(%rbp)
	movq	$0, -32(%rbp)
	jmp	.L90
.L70:
	call	next_i
	movl	%eax, -124(%rbp)
	call	next_i
	movl	%eax, -120(%rbp)
	movl	-124(%rbp), %eax
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
	movl	-152(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,8), %rcx
	movq	-56(%rbp), %rdx
	addq	%rcx, %rdx
	imull	-120(%rbp), %eax
	movl	%eax, (%rdx)
	call	next_i
	movl	%eax, -116(%rbp)
	call	next_i
	movl	%eax, -112(%rbp)
	movl	-116(%rbp), %eax
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
	movl	-152(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,8), %rcx
	movq	-56(%rbp), %rdx
	addq	%rcx, %rdx
	imull	-112(%rbp), %eax
	movl	%eax, 4(%rdx)
	addl	$1, -152(%rbp)
	movq	$18, -32(%rbp)
	jmp	.L90
.L73:
	movq	-48(%rbp), %rax
	movq	%rax, %rdi
	call	perf_evsel__read_size
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
	movq	$9, -32(%rbp)
	jmp	.L90
.L71:
	cmpl	$2, -164(%rbp)
	je	.L99
	movq	$7, -32(%rbp)
	jmp	.L90
.L99:
	movq	$12, -32(%rbp)
	jmp	.L90
.L81:
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
	movl	-136(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,8), %rcx
	movq	-40(%rbp), %rdx
	addq	%rcx, %rdx
	imull	-84(%rbp), %eax
	movl	%eax, (%rdx)
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
	movl	-136(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,8), %rcx
	movq	-40(%rbp), %rdx
	addq	%rcx, %rdx
	imull	-76(%rbp), %eax
	movl	%eax, 4(%rdx)
	addl	$1, -136(%rbp)
	movq	$17, -32(%rbp)
	jmp	.L90
.L87:
	movl	-144(%rbp), %eax
	cmpl	-148(%rbp), %eax
	jge	.L101
	movq	$4, -32(%rbp)
	jmp	.L90
.L101:
	movq	$22, -32(%rbp)
	jmp	.L90
.L84:
	call	usage
	movq	$21, -32(%rbp)
	jmp	.L90
.L104:
	nop
.L90:
	jmp	.L103
.L96:
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
