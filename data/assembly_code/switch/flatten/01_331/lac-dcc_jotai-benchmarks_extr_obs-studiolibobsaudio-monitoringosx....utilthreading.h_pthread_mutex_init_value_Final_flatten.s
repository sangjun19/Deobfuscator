	.file	"lac-dcc_jotai-benchmarks_extr_obs-studiolibobsaudio-monitoringosx....utilthreading.h_pthread_mutex_init_value_Final_flatten.c"
	.text
	.globl	PTHREAD_MUTEX_INITIALIZER
	.bss
	.align 4
	.type	PTHREAD_MUTEX_INITIALIZER, @object
	.size	PTHREAD_MUTEX_INITIALIZER, 4
PTHREAD_MUTEX_INITIALIZER:
	.zero	4
	.globl	rand_primes
	.align 32
	.type	rand_primes, @object
	.size	rand_primes, 100
rand_primes:
	.zero	100
	.globl	_TIG_IZ_51VV_argv
	.align 8
	.type	_TIG_IZ_51VV_argv, @object
	.size	_TIG_IZ_51VV_argv, 8
_TIG_IZ_51VV_argv:
	.zero	8
	.globl	_TIG_IZ_51VV_envp
	.align 8
	.type	_TIG_IZ_51VV_envp, @object
	.size	_TIG_IZ_51VV_envp, 8
_TIG_IZ_51VV_envp:
	.zero	8
	.globl	_TIG_IZ_51VV_argc
	.align 4
	.type	_TIG_IZ_51VV_argc, @object
	.size	_TIG_IZ_51VV_argc, 4
_TIG_IZ_51VV_argc:
	.zero	4
	.text
	.globl	main
	.type	main, @function
main:
.LFB1:
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
	movl	$0, PTHREAD_MUTEX_INITIALIZER(%rip)
	nop
.L2:
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
.L3:
	movq	$0, _TIG_IZ_51VV_envp(%rip)
	nop
.L4:
	movq	$0, _TIG_IZ_51VV_argv(%rip)
	nop
.L5:
	movl	$0, _TIG_IZ_51VV_argc(%rip)
	nop
	nop
.L6:
.L7:
#APP
# 179 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-51VV--0
# 0 "" 2
#NO_APP
	movl	-116(%rbp), %eax
	movl	%eax, _TIG_IZ_51VV_argc(%rip)
	movq	-128(%rbp), %rax
	movq	%rax, _TIG_IZ_51VV_argv(%rip)
	movq	-136(%rbp), %rax
	movq	%rax, _TIG_IZ_51VV_envp(%rip)
	nop
	movq	$22, -32(%rbp)
.L44:
	cmpq	$38, -32(%rbp)
	ja	.L45
	movq	-32(%rbp), %rax
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
	.long	.L45-.L10
	.long	.L28-.L10
	.long	.L45-.L10
	.long	.L27-.L10
	.long	.L45-.L10
	.long	.L45-.L10
	.long	.L26-.L10
	.long	.L45-.L10
	.long	.L45-.L10
	.long	.L25-.L10
	.long	.L24-.L10
	.long	.L45-.L10
	.long	.L45-.L10
	.long	.L23-.L10
	.long	.L22-.L10
	.long	.L21-.L10
	.long	.L45-.L10
	.long	.L45-.L10
	.long	.L20-.L10
	.long	.L45-.L10
	.long	.L45-.L10
	.long	.L19-.L10
	.long	.L18-.L10
	.long	.L45-.L10
	.long	.L17-.L10
	.long	.L45-.L10
	.long	.L45-.L10
	.long	.L16-.L10
	.long	.L45-.L10
	.long	.L15-.L10
	.long	.L45-.L10
	.long	.L14-.L10
	.long	.L13-.L10
	.long	.L12-.L10
	.long	.L45-.L10
	.long	.L45-.L10
	.long	.L45-.L10
	.long	.L11-.L10
	.long	.L9-.L10
	.text
.L20:
	call	usage
	movq	$9, -32(%rbp)
	jmp	.L29
.L22:
	movl	-96(%rbp), %eax
	cmpl	-100(%rbp), %eax
	jge	.L30
	movq	$38, -32(%rbp)
	jmp	.L29
.L30:
	movq	$13, -32(%rbp)
	jmp	.L29
.L21:
	movl	-88(%rbp), %eax
	cmpl	-92(%rbp), %eax
	jge	.L32
	movq	$3, -32(%rbp)
	jmp	.L29
.L32:
	movq	$31, -32(%rbp)
	jmp	.L29
.L14:
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	pthread_mutex_init_value
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$9, -32(%rbp)
	jmp	.L29
.L28:
	movl	-104(%rbp), %eax
	cmpl	-108(%rbp), %eax
	jge	.L34
	movq	$29, -32(%rbp)
	jmp	.L29
.L34:
	movq	$27, -32(%rbp)
	jmp	.L29
.L27:
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
	movl	-88(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-40(%rbp), %rdx
	addq	%rcx, %rdx
	imull	-60(%rbp), %eax
	movl	%eax, (%rdx)
	addl	$1, -88(%rbp)
	movq	$15, -32(%rbp)
	jmp	.L29
.L17:
	call	usage
	movq	$10, -32(%rbp)
	jmp	.L29
.L19:
	movq	-128(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movl	%eax, -68(%rbp)
	movl	-68(%rbp), %eax
	movl	%eax, -112(%rbp)
	movq	$33, -32(%rbp)
	jmp	.L29
.L25:
	movl	$0, %eax
	jmp	.L36
.L23:
	movq	-48(%rbp), %rax
	movq	%rax, %rdi
	call	pthread_mutex_init_value
	movq	-48(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$9, -32(%rbp)
	jmp	.L29
.L13:
	movl	$1, -92(%rbp)
	movl	-92(%rbp), %eax
	cltq
	salq	$2, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -40(%rbp)
	movl	$0, -88(%rbp)
	movq	$15, -32(%rbp)
	jmp	.L29
.L26:
	movl	$65025, -108(%rbp)
	movl	-108(%rbp), %eax
	cltq
	salq	$2, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -56(%rbp)
	movl	$0, -104(%rbp)
	movq	$1, -32(%rbp)
	jmp	.L29
.L16:
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	pthread_mutex_init_value
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$9, -32(%rbp)
	jmp	.L29
.L9:
	call	next_i
	movl	%eax, -84(%rbp)
	call	next_i
	movl	%eax, -80(%rbp)
	movl	-84(%rbp), %eax
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
	movl	-96(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-48(%rbp), %rdx
	addq	%rcx, %rdx
	imull	-80(%rbp), %eax
	movl	%eax, (%rdx)
	addl	$1, -96(%rbp)
	movq	$14, -32(%rbp)
	jmp	.L29
.L18:
	cmpl	$2, -116(%rbp)
	je	.L37
	movq	$24, -32(%rbp)
	jmp	.L29
.L37:
	movq	$21, -32(%rbp)
	jmp	.L29
.L12:
	cmpl	$2, -112(%rbp)
	je	.L39
	cmpl	$2, -112(%rbp)
	jg	.L40
	cmpl	$0, -112(%rbp)
	je	.L41
	cmpl	$1, -112(%rbp)
	je	.L42
	jmp	.L40
.L39:
	movq	$32, -32(%rbp)
	jmp	.L43
.L42:
	movq	$37, -32(%rbp)
	jmp	.L43
.L41:
	movq	$6, -32(%rbp)
	jmp	.L43
.L40:
	movq	$18, -32(%rbp)
	nop
.L43:
	jmp	.L29
.L11:
	movl	$100, -100(%rbp)
	movl	-100(%rbp), %eax
	cltq
	salq	$2, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, -48(%rbp)
	movl	$0, -96(%rbp)
	movq	$14, -32(%rbp)
	jmp	.L29
.L24:
	movl	$1, %eax
	jmp	.L36
.L15:
	call	next_i
	movl	%eax, -76(%rbp)
	call	next_i
	movl	%eax, -72(%rbp)
	movl	-76(%rbp), %eax
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
	movl	-104(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-56(%rbp), %rdx
	addq	%rcx, %rdx
	imull	-72(%rbp), %eax
	movl	%eax, (%rdx)
	addl	$1, -104(%rbp)
	movq	$1, -32(%rbp)
	jmp	.L29
.L45:
	nop
.L29:
	jmp	.L44
.L36:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	main, .-main
	.globl	next_i
	.type	next_i, @function
next_i:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	$2, -8(%rbp)
.L52:
	cmpq	$2, -8(%rbp)
	je	.L47
	cmpq	$2, -8(%rbp)
	ja	.L54
	cmpq	$0, -8(%rbp)
	je	.L49
	cmpq	$1, -8(%rbp)
	jne	.L54
	movl	$0, -12(%rbp)
	addl	$1, -12(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L50
.L49:
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
	jmp	.L53
.L47:
	movq	$1, -8(%rbp)
	jmp	.L50
.L54:
	nop
.L50:
	jmp	.L52
.L53:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	next_i, .-next_i
	.globl	next_f
	.type	next_f, @function
next_f:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	$2, -8(%rbp)
.L63:
	cmpq	$2, -8(%rbp)
	je	.L56
	cmpq	$2, -8(%rbp)
	ja	.L65
	cmpq	$0, -8(%rbp)
	je	.L58
	cmpq	$1, -8(%rbp)
	jne	.L65
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
	js	.L59
	pxor	%xmm0, %xmm0
	cvtsi2ssq	%rax, %xmm0
	jmp	.L60
.L59:
	movq	%rax, %rdx
	shrq	%rdx
	andl	$1, %eax
	orq	%rax, %rdx
	pxor	%xmm0, %xmm0
	cvtsi2ssq	%rdx, %xmm0
	addss	%xmm0, %xmm0
.L60:
	movss	.LC0(%rip), %xmm1
	divss	%xmm1, %xmm0
	jmp	.L64
.L58:
	movl	$0, -12(%rbp)
	addl	$1, -12(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L62
.L56:
	movq	$0, -8(%rbp)
	jmp	.L62
.L65:
	nop
.L62:
	jmp	.L63
.L64:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
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
.L71:
	cmpq	$0, -8(%rbp)
	je	.L67
	cmpq	$1, -8(%rbp)
	jne	.L73
	jmp	.L72
.L67:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$1, -8(%rbp)
	jmp	.L70
.L73:
	nop
.L70:
	jmp	.L71
.L72:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	usage, .-usage
	.type	pthread_mutex_init_value, @function
pthread_mutex_init_value:
.LFB10:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movq	$3, -8(%rbp)
.L86:
	cmpq	$4, -8(%rbp)
	ja	.L87
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L77(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L77(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L77:
	.long	.L81-.L77
	.long	.L88-.L77
	.long	.L88-.L77
	.long	.L78-.L77
	.long	.L76-.L77
	.text
.L76:
	movq	-24(%rbp), %rax
	movl	-12(%rbp), %edx
	movl	%edx, (%rax)
	movq	$2, -8(%rbp)
	jmp	.L82
.L78:
	movl	PTHREAD_MUTEX_INITIALIZER(%rip), %eax
	movl	%eax, -12(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L82
.L81:
	cmpq	$0, -24(%rbp)
	jne	.L84
	movq	$1, -8(%rbp)
	jmp	.L82
.L84:
	movq	$4, -8(%rbp)
	jmp	.L82
.L87:
	nop
.L82:
	jmp	.L86
.L88:
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	pthread_mutex_init_value, .-pthread_mutex_init_value
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
