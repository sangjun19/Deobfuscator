	.file	"lac-dcc_jotai-benchmarks_extr_nginxsrccorengx_connection.c_ngx_clone_listening_Final_flatten.c"
	.text
	.globl	_TIG_IZ_CCrj_envp
	.bss
	.align 8
	.type	_TIG_IZ_CCrj_envp, @object
	.size	_TIG_IZ_CCrj_envp, 8
_TIG_IZ_CCrj_envp:
	.zero	8
	.globl	_TIG_IZ_CCrj_argc
	.align 4
	.type	_TIG_IZ_CCrj_argc, @object
	.size	_TIG_IZ_CCrj_argc, 4
_TIG_IZ_CCrj_argc:
	.zero	4
	.globl	rand_primes
	.align 32
	.type	rand_primes, @object
	.size	rand_primes, 100
rand_primes:
	.zero	100
	.globl	_TIG_IZ_CCrj_argv
	.align 8
	.type	_TIG_IZ_CCrj_argv, @object
	.size	_TIG_IZ_CCrj_argv, 8
_TIG_IZ_CCrj_argv:
	.zero	8
	.globl	NGX_OK
	.align 4
	.type	NGX_OK, @object
	.size	NGX_OK, 4
NGX_OK:
	.zero	4
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
	movq	$2, -8(%rbp)
.L7:
	cmpq	$2, -8(%rbp)
	je	.L2
	cmpq	$2, -8(%rbp)
	ja	.L9
	cmpq	$0, -8(%rbp)
	je	.L4
	cmpq	$1, -8(%rbp)
	jne	.L9
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
.L4:
	movl	$0, -12(%rbp)
	addl	$1, -12(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L6
.L2:
	movq	$0, -8(%rbp)
	jmp	.L6
.L9:
	nop
.L6:
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
	.globl	ngx_clone_listening
	.type	ngx_clone_listening, @function
ngx_clone_listening:
.LFB4:
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
.L21:
	cmpq	$0, -8(%rbp)
	jne	.L24
	movl	NGX_OK(%rip), %eax
	jmp	.L23
.L24:
	nop
	jmp	.L21
.L23:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	ngx_clone_listening, .-ngx_clone_listening
	.section	.rodata
.LC1:
	.string	"%d\n"
	.text
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
	subq	$272, %rsp
	movl	%edi, -244(%rbp)
	movq	%rsi, -256(%rbp)
	movq	%rdx, -264(%rbp)
	movl	$0, NGX_OK(%rip)
	nop
.L26:
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
.L27:
	movq	$0, _TIG_IZ_CCrj_envp(%rip)
	nop
.L28:
	movq	$0, _TIG_IZ_CCrj_argv(%rip)
	nop
.L29:
	movl	$0, _TIG_IZ_CCrj_argc(%rip)
	nop
	nop
.L30:
.L31:
#APP
# 181 "lac-dcc_jotai-benchmarks_extr_nginxsrccorengx_connection.c_ngx_clone_listening_Final.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-CCrj--0
# 0 "" 2
#NO_APP
	movl	-244(%rbp), %eax
	movl	%eax, _TIG_IZ_CCrj_argc(%rip)
	movq	-256(%rbp), %rax
	movq	%rax, _TIG_IZ_CCrj_argv(%rip)
	movq	-264(%rbp), %rax
	movq	%rax, _TIG_IZ_CCrj_envp(%rip)
	nop
	movq	$43, -56(%rbp)
.L83:
	cmpq	$59, -56(%rbp)
	ja	.L84
	movq	-56(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L34(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L34(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L34:
	.long	.L84-.L34
	.long	.L61-.L34
	.long	.L84-.L34
	.long	.L60-.L34
	.long	.L59-.L34
	.long	.L84-.L34
	.long	.L84-.L34
	.long	.L58-.L34
	.long	.L84-.L34
	.long	.L57-.L34
	.long	.L56-.L34
	.long	.L84-.L34
	.long	.L84-.L34
	.long	.L55-.L34
	.long	.L84-.L34
	.long	.L84-.L34
	.long	.L54-.L34
	.long	.L53-.L34
	.long	.L52-.L34
	.long	.L51-.L34
	.long	.L50-.L34
	.long	.L49-.L34
	.long	.L84-.L34
	.long	.L84-.L34
	.long	.L48-.L34
	.long	.L47-.L34
	.long	.L84-.L34
	.long	.L84-.L34
	.long	.L84-.L34
	.long	.L46-.L34
	.long	.L84-.L34
	.long	.L84-.L34
	.long	.L84-.L34
	.long	.L45-.L34
	.long	.L44-.L34
	.long	.L84-.L34
	.long	.L84-.L34
	.long	.L84-.L34
	.long	.L43-.L34
	.long	.L84-.L34
	.long	.L84-.L34
	.long	.L84-.L34
	.long	.L42-.L34
	.long	.L41-.L34
	.long	.L84-.L34
	.long	.L84-.L34
	.long	.L84-.L34
	.long	.L84-.L34
	.long	.L40-.L34
	.long	.L84-.L34
	.long	.L84-.L34
	.long	.L39-.L34
	.long	.L84-.L34
	.long	.L38-.L34
	.long	.L84-.L34
	.long	.L84-.L34
	.long	.L37-.L34
	.long	.L36-.L34
	.long	.L35-.L34
	.long	.L33-.L34
	.text
.L52:
	movl	$0, %eax
	jmp	.L62
.L47:
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
	movl	-216(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-96(%rbp), %rdx
	addq	%rcx, %rdx
	imull	-148(%rbp), %eax
	movl	%eax, (%rdx)
	addl	$1, -216(%rbp)
	movq	$34, -56(%rbp)
	jmp	.L63
.L59:
	movl	$1, %eax
	jmp	.L62
.L37:
	movl	$1, -196(%rbp)
	movl	-196(%rbp), %eax
	cltq
	salq	$2, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -40(%rbp)
	movq	-40(%rbp), %rax
	movq	%rax, -72(%rbp)
	movl	$0, -192(%rbp)
	movq	$33, -56(%rbp)
	jmp	.L63
.L61:
	cmpl	$2, -232(%rbp)
	je	.L64
	cmpl	$2, -232(%rbp)
	jg	.L65
	cmpl	$0, -232(%rbp)
	je	.L66
	cmpl	$1, -232(%rbp)
	je	.L67
	jmp	.L65
.L64:
	movq	$56, -56(%rbp)
	jmp	.L68
.L67:
	movq	$58, -56(%rbp)
	jmp	.L68
.L66:
	movq	$19, -56(%rbp)
	jmp	.L68
.L65:
	movq	$48, -56(%rbp)
	nop
.L68:
	jmp	.L63
.L60:
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
	movl	-200(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-80(%rbp), %rdx
	addq	%rcx, %rdx
	imull	-108(%rbp), %eax
	movl	%eax, (%rdx)
	addl	$1, -200(%rbp)
	movq	$9, -56(%rbp)
	jmp	.L63
.L54:
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
	movl	-224(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-104(%rbp), %rdx
	addq	%rcx, %rdx
	imull	-132(%rbp), %eax
	movl	%eax, (%rdx)
	addl	$1, -224(%rbp)
	movq	$29, -56(%rbp)
	jmp	.L63
.L48:
	movl	$100, -204(%rbp)
	movl	-204(%rbp), %eax
	cltq
	salq	$2, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -32(%rbp)
	movq	-32(%rbp), %rax
	movq	%rax, -80(%rbp)
	movl	$0, -200(%rbp)
	movq	$9, -56(%rbp)
	jmp	.L63
.L49:
	movl	$1, -188(%rbp)
	movl	-188(%rbp), %eax
	cltq
	salq	$2, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, -64(%rbp)
	movl	$0, -184(%rbp)
	movq	$53, -56(%rbp)
	jmp	.L63
.L36:
	call	next_i
	movl	%eax, -172(%rbp)
	call	next_i
	movl	%eax, -168(%rbp)
	movl	-172(%rbp), %eax
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
	movl	-192(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-72(%rbp), %rdx
	addq	%rcx, %rdx
	imull	-168(%rbp), %eax
	movl	%eax, (%rdx)
	addl	$1, -192(%rbp)
	movq	$33, -56(%rbp)
	jmp	.L63
.L57:
	movl	-200(%rbp), %eax
	cmpl	-204(%rbp), %eax
	jge	.L69
	movq	$3, -56(%rbp)
	jmp	.L63
.L69:
	movq	$7, -56(%rbp)
	jmp	.L63
.L55:
	movq	-96(%rbp), %rdx
	movq	-104(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	ngx_clone_listening
	movl	%eax, -128(%rbp)
	movl	-128(%rbp), %eax
	movl	%eax, -124(%rbp)
	movl	-124(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-104(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	-96(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$18, -56(%rbp)
	jmp	.L63
.L39:
	call	usage
	movq	$4, -56(%rbp)
	jmp	.L63
.L51:
	movl	$65025, -228(%rbp)
	movl	-228(%rbp), %eax
	cltq
	salq	$2, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -104(%rbp)
	movl	$0, -224(%rbp)
	movq	$29, -56(%rbp)
	jmp	.L63
.L53:
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
	movl	-184(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-64(%rbp), %rdx
	addq	%rcx, %rdx
	imull	-140(%rbp), %eax
	movl	%eax, (%rdx)
	addl	$1, -184(%rbp)
	movq	$53, -56(%rbp)
	jmp	.L63
.L33:
	movq	-64(%rbp), %rdx
	movq	-72(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	ngx_clone_listening
	movl	%eax, -180(%rbp)
	movl	-180(%rbp), %eax
	movl	%eax, -176(%rbp)
	movl	-176(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-72(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$18, -56(%rbp)
	jmp	.L63
.L43:
	movq	-256(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movl	%eax, -156(%rbp)
	movl	-156(%rbp), %eax
	movl	%eax, -232(%rbp)
	movq	$1, -56(%rbp)
	jmp	.L63
.L35:
	movl	$100, -212(%rbp)
	movl	-212(%rbp), %eax
	cltq
	salq	$2, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -48(%rbp)
	movq	-48(%rbp), %rax
	movq	%rax, -88(%rbp)
	movl	$0, -208(%rbp)
	movq	$10, -56(%rbp)
	jmp	.L63
.L44:
	movl	-216(%rbp), %eax
	cmpl	-220(%rbp), %eax
	jge	.L71
	movq	$25, -56(%rbp)
	jmp	.L63
.L71:
	movq	$13, -56(%rbp)
	jmp	.L63
.L40:
	call	usage
	movq	$18, -56(%rbp)
	jmp	.L63
.L38:
	movl	-184(%rbp), %eax
	cmpl	-188(%rbp), %eax
	jge	.L73
	movq	$17, -56(%rbp)
	jmp	.L63
.L73:
	movq	$59, -56(%rbp)
	jmp	.L63
.L45:
	movl	-192(%rbp), %eax
	cmpl	-196(%rbp), %eax
	jge	.L75
	movq	$57, -56(%rbp)
	jmp	.L63
.L75:
	movq	$21, -56(%rbp)
	jmp	.L63
.L56:
	movl	-208(%rbp), %eax
	cmpl	-212(%rbp), %eax
	jge	.L77
	movq	$42, -56(%rbp)
	jmp	.L63
.L77:
	movq	$24, -56(%rbp)
	jmp	.L63
.L42:
	call	next_i
	movl	%eax, -164(%rbp)
	call	next_i
	movl	%eax, -160(%rbp)
	movl	-164(%rbp), %eax
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
	movl	-208(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-88(%rbp), %rdx
	addq	%rcx, %rdx
	imull	-160(%rbp), %eax
	movl	%eax, (%rdx)
	addl	$1, -208(%rbp)
	movq	$10, -56(%rbp)
	jmp	.L63
.L58:
	movq	-80(%rbp), %rdx
	movq	-88(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	ngx_clone_listening
	movl	%eax, -120(%rbp)
	movl	-120(%rbp), %eax
	movl	%eax, -116(%rbp)
	movl	-116(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-88(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	-80(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$18, -56(%rbp)
	jmp	.L63
.L46:
	movl	-224(%rbp), %eax
	cmpl	-228(%rbp), %eax
	jge	.L79
	movq	$16, -56(%rbp)
	jmp	.L63
.L79:
	movq	$20, -56(%rbp)
	jmp	.L63
.L41:
	cmpl	$2, -244(%rbp)
	je	.L81
	movq	$51, -56(%rbp)
	jmp	.L63
.L81:
	movq	$38, -56(%rbp)
	jmp	.L63
.L50:
	movl	$65025, -220(%rbp)
	movl	-220(%rbp), %eax
	cltq
	salq	$2, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -96(%rbp)
	movl	$0, -216(%rbp)
	movq	$34, -56(%rbp)
	jmp	.L63
.L84:
	nop
.L63:
	jmp	.L83
.L62:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	main, .-main
	.globl	next_f
	.type	next_f, @function
next_f:
.LFB10:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	$1, -8(%rbp)
.L93:
	cmpq	$2, -8(%rbp)
	je	.L86
	cmpq	$2, -8(%rbp)
	ja	.L95
	cmpq	$0, -8(%rbp)
	je	.L88
	cmpq	$1, -8(%rbp)
	jne	.L95
	movq	$0, -8(%rbp)
	jmp	.L89
.L88:
	movl	$0, -12(%rbp)
	addl	$1, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L89
.L86:
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
	js	.L90
	pxor	%xmm0, %xmm0
	cvtsi2ssq	%rax, %xmm0
	jmp	.L91
.L90:
	movq	%rax, %rdx
	shrq	%rdx
	andl	$1, %eax
	orq	%rax, %rdx
	pxor	%xmm0, %xmm0
	cvtsi2ssq	%rdx, %xmm0
	addss	%xmm0, %xmm0
.L91:
	movss	.LC2(%rip), %xmm1
	divss	%xmm1, %xmm0
	jmp	.L94
.L95:
	nop
.L89:
	jmp	.L93
.L94:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	next_f, .-next_f
	.section	.rodata
	.align 4
.LC2:
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
