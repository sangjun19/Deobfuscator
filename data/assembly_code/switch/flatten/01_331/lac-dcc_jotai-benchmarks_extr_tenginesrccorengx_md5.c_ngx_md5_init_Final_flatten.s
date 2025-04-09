	.file	"lac-dcc_jotai-benchmarks_extr_tenginesrccorengx_md5.c_ngx_md5_init_Final_flatten.c"
	.text
	.globl	_TIG_IZ_rcxy_argc
	.bss
	.align 4
	.type	_TIG_IZ_rcxy_argc, @object
	.size	_TIG_IZ_rcxy_argc, 4
_TIG_IZ_rcxy_argc:
	.zero	4
	.globl	_TIG_IZ_rcxy_envp
	.align 8
	.type	_TIG_IZ_rcxy_envp, @object
	.size	_TIG_IZ_rcxy_envp, 8
_TIG_IZ_rcxy_envp:
	.zero	8
	.globl	rand_primes
	.align 32
	.type	rand_primes, @object
	.size	rand_primes, 100
rand_primes:
	.zero	100
	.globl	_TIG_IZ_rcxy_argv
	.align 8
	.type	_TIG_IZ_rcxy_argv, @object
	.size	_TIG_IZ_rcxy_argv, 8
_TIG_IZ_rcxy_argv:
	.zero	8
	.text
	.globl	ngx_md5_init
	.type	ngx_md5_init, @function
ngx_md5_init:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movq	$1, -8(%rbp)
.L7:
	cmpq	$2, -8(%rbp)
	je	.L2
	cmpq	$2, -8(%rbp)
	ja	.L8
	cmpq	$0, -8(%rbp)
	je	.L9
	cmpq	$1, -8(%rbp)
	jne	.L8
	movq	$2, -8(%rbp)
	jmp	.L5
.L2:
	movq	-24(%rbp), %rax
	movl	$1732584193, (%rax)
	movq	-24(%rbp), %rax
	movl	$-271733879, 4(%rax)
	movq	-24(%rbp), %rax
	movl	$-1732584194, 8(%rax)
	movq	-24(%rbp), %rax
	movl	$271733878, 12(%rax)
	movq	-24(%rbp), %rax
	movq	$0, 16(%rax)
	movq	$0, -8(%rbp)
	jmp	.L5
.L8:
	nop
.L5:
	jmp	.L7
.L9:
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	ngx_md5_init, .-ngx_md5_init
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
	movq	$0, -8(%rbp)
.L15:
	cmpq	$0, -8(%rbp)
	je	.L11
	cmpq	$1, -8(%rbp)
	jne	.L17
	jmp	.L16
.L11:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$1, -8(%rbp)
	jmp	.L14
.L17:
	nop
.L14:
	jmp	.L15
.L16:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	usage, .-usage
	.globl	next_i
	.type	next_i, @function
next_i:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	$0, -8(%rbp)
.L24:
	cmpq	$2, -8(%rbp)
	je	.L19
	cmpq	$2, -8(%rbp)
	ja	.L26
	cmpq	$0, -8(%rbp)
	je	.L21
	cmpq	$1, -8(%rbp)
	jne	.L26
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
	jmp	.L25
.L21:
	movq	$2, -8(%rbp)
	jmp	.L23
.L19:
	movl	$0, -12(%rbp)
	addl	$1, -12(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L23
.L26:
	nop
.L23:
	jmp	.L24
.L25:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	next_i, .-next_i
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
	subq	$240, %rsp
	movl	%edi, -212(%rbp)
	movq	%rsi, -224(%rbp)
	movq	%rdx, -232(%rbp)
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
.L28:
	movq	$0, _TIG_IZ_rcxy_envp(%rip)
	nop
.L29:
	movq	$0, _TIG_IZ_rcxy_argv(%rip)
	nop
.L30:
	movl	$0, _TIG_IZ_rcxy_argc(%rip)
	nop
	nop
.L31:
.L32:
#APP
# 141 "lac-dcc_jotai-benchmarks_extr_tenginesrccorengx_md5.c_ngx_md5_init_Final.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-rcxy--0
# 0 "" 2
#NO_APP
	movl	-212(%rbp), %eax
	movl	%eax, _TIG_IZ_rcxy_argc(%rip)
	movq	-224(%rbp), %rax
	movq	%rax, _TIG_IZ_rcxy_argv(%rip)
	movq	-232(%rbp), %rax
	movq	%rax, _TIG_IZ_rcxy_envp(%rip)
	nop
	movq	$22, -32(%rbp)
.L69:
	cmpq	$33, -32(%rbp)
	ja	.L70
	movq	-32(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L35(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L35(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L35:
	.long	.L70-.L35
	.long	.L53-.L35
	.long	.L52-.L35
	.long	.L70-.L35
	.long	.L70-.L35
	.long	.L70-.L35
	.long	.L70-.L35
	.long	.L51-.L35
	.long	.L70-.L35
	.long	.L50-.L35
	.long	.L49-.L35
	.long	.L70-.L35
	.long	.L48-.L35
	.long	.L47-.L35
	.long	.L46-.L35
	.long	.L45-.L35
	.long	.L44-.L35
	.long	.L70-.L35
	.long	.L70-.L35
	.long	.L70-.L35
	.long	.L43-.L35
	.long	.L42-.L35
	.long	.L41-.L35
	.long	.L70-.L35
	.long	.L40-.L35
	.long	.L70-.L35
	.long	.L39-.L35
	.long	.L38-.L35
	.long	.L37-.L35
	.long	.L70-.L35
	.long	.L36-.L35
	.long	.L70-.L35
	.long	.L70-.L35
	.long	.L34-.L35
	.text
.L36:
	movl	-192(%rbp), %eax
	cmpl	-196(%rbp), %eax
	jge	.L54
	movq	$14, -32(%rbp)
	jmp	.L56
.L54:
	movq	$28, -32(%rbp)
	jmp	.L56
.L46:
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
	leal	1(%rax), %ecx
	movl	-192(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-48(%rbp), %rax
	addq	%rax, %rdx
	movl	%ecx, %eax
	imull	-132(%rbp), %eax
	movl	%eax, (%rdx)
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
	movl	-192(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-48(%rbp), %rax
	addq	%rax, %rdx
	movl	%ecx, %eax
	imull	-124(%rbp), %eax
	movl	%eax, 4(%rdx)
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
	movl	-192(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-48(%rbp), %rax
	addq	%rax, %rdx
	movl	%ecx, %eax
	imull	-116(%rbp), %eax
	movl	%eax, 8(%rdx)
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
	movl	-192(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-48(%rbp), %rax
	addq	%rax, %rdx
	movl	%ecx, %eax
	imull	-108(%rbp), %eax
	movl	%eax, 12(%rdx)
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
	addl	$1, %eax
	imull	-100(%rbp), %eax
	movl	%eax, %ecx
	movl	-192(%rbp), %eax
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
	addl	$1, -192(%rbp)
	movq	$30, -32(%rbp)
	jmp	.L56
.L45:
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
	leal	1(%rax), %ecx
	movl	-184(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-40(%rbp), %rax
	addq	%rax, %rdx
	movl	%ecx, %eax
	imull	-172(%rbp), %eax
	movl	%eax, (%rdx)
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
	leal	1(%rax), %ecx
	movl	-184(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-40(%rbp), %rax
	addq	%rax, %rdx
	movl	%ecx, %eax
	imull	-164(%rbp), %eax
	movl	%eax, 4(%rdx)
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
	leal	1(%rax), %ecx
	movl	-184(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-40(%rbp), %rax
	addq	%rax, %rdx
	movl	%ecx, %eax
	imull	-156(%rbp), %eax
	movl	%eax, 8(%rdx)
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
	leal	1(%rax), %ecx
	movl	-184(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-40(%rbp), %rax
	addq	%rax, %rdx
	movl	%ecx, %eax
	imull	-148(%rbp), %eax
	movl	%eax, 12(%rdx)
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
	movl	-184(%rbp), %eax
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
	addl	$1, -184(%rbp)
	movq	$26, -32(%rbp)
	jmp	.L56
.L48:
	movl	$1, %eax
	jmp	.L57
.L53:
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
	movl	-200(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-56(%rbp), %rax
	addq	%rax, %rdx
	movl	%ecx, %eax
	imull	-92(%rbp), %eax
	movl	%eax, (%rdx)
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
	movl	-200(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-56(%rbp), %rax
	addq	%rax, %rdx
	movl	%ecx, %eax
	imull	-84(%rbp), %eax
	movl	%eax, 4(%rdx)
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
	movl	-200(%rbp), %eax
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
	movl	%eax, 8(%rdx)
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
	movl	-200(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-56(%rbp), %rax
	addq	%rax, %rdx
	movl	%ecx, %eax
	imull	-68(%rbp), %eax
	movl	%eax, 12(%rdx)
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
	movl	-200(%rbp), %eax
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
	addl	$1, -200(%rbp)
	movq	$33, -32(%rbp)
	jmp	.L56
.L44:
	movl	$0, %eax
	jmp	.L57
.L40:
	movq	-224(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movl	%eax, -180(%rbp)
	movl	-180(%rbp), %eax
	movl	%eax, -208(%rbp)
	movq	$9, -32(%rbp)
	jmp	.L56
.L42:
	call	usage
	movq	$12, -32(%rbp)
	jmp	.L56
.L39:
	movl	-184(%rbp), %eax
	cmpl	-188(%rbp), %eax
	jge	.L58
	movq	$15, -32(%rbp)
	jmp	.L56
.L58:
	movq	$2, -32(%rbp)
	jmp	.L56
.L50:
	cmpl	$2, -208(%rbp)
	je	.L60
	cmpl	$2, -208(%rbp)
	jg	.L61
	cmpl	$0, -208(%rbp)
	je	.L62
	cmpl	$1, -208(%rbp)
	je	.L63
	jmp	.L61
.L60:
	movq	$13, -32(%rbp)
	jmp	.L64
.L63:
	movq	$27, -32(%rbp)
	jmp	.L64
.L62:
	movq	$20, -32(%rbp)
	jmp	.L64
.L61:
	movq	$10, -32(%rbp)
	nop
.L64:
	jmp	.L56
.L47:
	movl	$1, -188(%rbp)
	movl	-188(%rbp), %eax
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
	movl	$0, -184(%rbp)
	movq	$26, -32(%rbp)
	jmp	.L56
.L38:
	movl	$100, -196(%rbp)
	movl	-196(%rbp), %eax
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
	movl	$0, -192(%rbp)
	movq	$30, -32(%rbp)
	jmp	.L56
.L41:
	cmpl	$2, -212(%rbp)
	je	.L65
	movq	$21, -32(%rbp)
	jmp	.L56
.L65:
	movq	$24, -32(%rbp)
	jmp	.L56
.L37:
	movq	-48(%rbp), %rax
	movq	%rax, %rdi
	call	ngx_md5_init
	movq	-48(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$16, -32(%rbp)
	jmp	.L56
.L34:
	movl	-200(%rbp), %eax
	cmpl	-204(%rbp), %eax
	jge	.L67
	movq	$1, -32(%rbp)
	jmp	.L56
.L67:
	movq	$7, -32(%rbp)
	jmp	.L56
.L49:
	call	usage
	movq	$16, -32(%rbp)
	jmp	.L56
.L51:
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	ngx_md5_init
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$16, -32(%rbp)
	jmp	.L56
.L52:
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	ngx_md5_init
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$16, -32(%rbp)
	jmp	.L56
.L43:
	movl	$65025, -204(%rbp)
	movl	-204(%rbp), %eax
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
	movl	$0, -200(%rbp)
	movq	$33, -32(%rbp)
	jmp	.L56
.L70:
	nop
.L56:
	jmp	.L69
.L57:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	main, .-main
	.globl	next_f
	.type	next_f, @function
next_f:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	$1, -8(%rbp)
.L79:
	cmpq	$2, -8(%rbp)
	je	.L72
	cmpq	$2, -8(%rbp)
	ja	.L81
	cmpq	$0, -8(%rbp)
	je	.L74
	cmpq	$1, -8(%rbp)
	jne	.L81
	movq	$2, -8(%rbp)
	jmp	.L75
.L74:
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
	js	.L76
	pxor	%xmm0, %xmm0
	cvtsi2ssq	%rax, %xmm0
	jmp	.L77
.L76:
	movq	%rax, %rdx
	shrq	%rdx
	andl	$1, %eax
	orq	%rax, %rdx
	pxor	%xmm0, %xmm0
	cvtsi2ssq	%rdx, %xmm0
	addss	%xmm0, %xmm0
.L77:
	movss	.LC1(%rip), %xmm1
	divss	%xmm1, %xmm0
	jmp	.L80
.L72:
	movl	$0, -12(%rbp)
	addl	$1, -12(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L75
.L81:
	nop
.L75:
	jmp	.L79
.L80:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	next_f, .-next_f
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
