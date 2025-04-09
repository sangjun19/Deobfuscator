	.file	"lac-dcc_jotai-benchmarks_extr_linuxsoundsoccodecsi-sabre-codec.c_i_sabre_codec_volatile_Final_flatten.c"
	.text
	.globl	_TIG_IZ_je6J_argv
	.bss
	.align 8
	.type	_TIG_IZ_je6J_argv, @object
	.size	_TIG_IZ_je6J_argv, 8
_TIG_IZ_je6J_argv:
	.zero	8
	.globl	_TIG_IZ_je6J_argc
	.align 4
	.type	_TIG_IZ_je6J_argc, @object
	.size	_TIG_IZ_je6J_argc, 4
_TIG_IZ_je6J_argc:
	.zero	4
	.globl	rand_primes
	.align 32
	.type	rand_primes, @object
	.size	rand_primes, 100
rand_primes:
	.zero	100
	.globl	_TIG_IZ_je6J_envp
	.align 8
	.type	_TIG_IZ_je6J_envp, @object
	.size	_TIG_IZ_je6J_envp, 8
_TIG_IZ_je6J_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"%d\n"
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
.L2:
	movq	$0, _TIG_IZ_je6J_envp(%rip)
	nop
.L3:
	movq	$0, _TIG_IZ_je6J_argv(%rip)
	nop
.L4:
	movl	$0, _TIG_IZ_je6J_argc(%rip)
	nop
	nop
.L5:
.L6:
#APP
# 173 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-je6J--0
# 0 "" 2
#NO_APP
	movl	-212(%rbp), %eax
	movl	%eax, _TIG_IZ_je6J_argc(%rip)
	movq	-224(%rbp), %rax
	movq	%rax, _TIG_IZ_je6J_argv(%rip)
	movq	-232(%rbp), %rax
	movq	%rax, _TIG_IZ_je6J_envp(%rip)
	nop
	movq	$15, -40(%rbp)
.L50:
	cmpq	$46, -40(%rbp)
	ja	.L51
	movq	-40(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L9(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L9(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L9:
	.long	.L31-.L9
	.long	.L51-.L9
	.long	.L30-.L9
	.long	.L29-.L9
	.long	.L28-.L9
	.long	.L51-.L9
	.long	.L27-.L9
	.long	.L26-.L9
	.long	.L51-.L9
	.long	.L25-.L9
	.long	.L24-.L9
	.long	.L51-.L9
	.long	.L51-.L9
	.long	.L23-.L9
	.long	.L51-.L9
	.long	.L22-.L9
	.long	.L21-.L9
	.long	.L20-.L9
	.long	.L51-.L9
	.long	.L51-.L9
	.long	.L19-.L9
	.long	.L18-.L9
	.long	.L51-.L9
	.long	.L51-.L9
	.long	.L17-.L9
	.long	.L51-.L9
	.long	.L51-.L9
	.long	.L16-.L9
	.long	.L15-.L9
	.long	.L51-.L9
	.long	.L51-.L9
	.long	.L51-.L9
	.long	.L51-.L9
	.long	.L14-.L9
	.long	.L51-.L9
	.long	.L13-.L9
	.long	.L12-.L9
	.long	.L51-.L9
	.long	.L51-.L9
	.long	.L51-.L9
	.long	.L51-.L9
	.long	.L51-.L9
	.long	.L51-.L9
	.long	.L11-.L9
	.long	.L51-.L9
	.long	.L10-.L9
	.long	.L8-.L9
	.text
.L28:
	movl	-184(%rbp), %edx
	movq	-64(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	i_sabre_codec_volatile
	movl	%eax, -96(%rbp)
	movl	-96(%rbp), %eax
	movl	%eax, -92(%rbp)
	movl	-92(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$17, -40(%rbp)
	jmp	.L32
.L22:
	cmpl	$2, -212(%rbp)
	je	.L33
	movq	$43, -40(%rbp)
	jmp	.L32
.L33:
	movq	$9, -40(%rbp)
	jmp	.L32
.L10:
	movl	$10, -172(%rbp)
	movl	$100, -168(%rbp)
	movl	-168(%rbp), %eax
	cltq
	salq	$2, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -32(%rbp)
	movq	-32(%rbp), %rax
	movq	%rax, -56(%rbp)
	movl	$0, -164(%rbp)
	movq	$27, -40(%rbp)
	jmp	.L32
.L29:
	movl	-188(%rbp), %eax
	cmpl	-192(%rbp), %eax
	jge	.L35
	movq	$7, -40(%rbp)
	jmp	.L32
.L35:
	movq	$24, -40(%rbp)
	jmp	.L32
.L21:
	movl	-172(%rbp), %edx
	movq	-56(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	i_sabre_codec_volatile
	movl	%eax, -116(%rbp)
	movl	-116(%rbp), %eax
	movl	%eax, -112(%rbp)
	movl	-112(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$17, -40(%rbp)
	jmp	.L32
.L17:
	movl	-196(%rbp), %edx
	movq	-72(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	i_sabre_codec_volatile
	movl	%eax, -124(%rbp)
	movl	-124(%rbp), %eax
	movl	%eax, -120(%rbp)
	movl	-120(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-72(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$17, -40(%rbp)
	jmp	.L32
.L18:
	cmpl	$3, -200(%rbp)
	je	.L37
	cmpl	$3, -200(%rbp)
	jg	.L38
	cmpl	$2, -200(%rbp)
	je	.L39
	cmpl	$2, -200(%rbp)
	jg	.L38
	cmpl	$0, -200(%rbp)
	je	.L40
	cmpl	$1, -200(%rbp)
	je	.L41
	jmp	.L38
.L37:
	movq	$36, -40(%rbp)
	jmp	.L42
.L39:
	movq	$45, -40(%rbp)
	jmp	.L42
.L41:
	movq	$20, -40(%rbp)
	jmp	.L42
.L40:
	movq	$10, -40(%rbp)
	jmp	.L42
.L38:
	movq	$33, -40(%rbp)
	nop
.L42:
	jmp	.L32
.L12:
	call	next_i
	movl	%eax, -140(%rbp)
	call	next_i
	movl	%eax, -136(%rbp)
	movl	-140(%rbp), %eax
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
	imull	-136(%rbp), %eax
	movl	%eax, -160(%rbp)
	movl	$1, -156(%rbp)
	movl	-156(%rbp), %eax
	cltq
	salq	$2, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, -48(%rbp)
	movl	$0, -152(%rbp)
	movq	$13, -40(%rbp)
	jmp	.L32
.L25:
	movq	-224(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movl	%eax, -108(%rbp)
	movl	-108(%rbp), %eax
	movl	%eax, -200(%rbp)
	movq	$21, -40(%rbp)
	jmp	.L32
.L23:
	movl	-152(%rbp), %eax
	cmpl	-156(%rbp), %eax
	jge	.L43
	movq	$0, -40(%rbp)
	jmp	.L32
.L43:
	movq	$35, -40(%rbp)
	jmp	.L32
.L20:
	movl	$0, %eax
	jmp	.L45
.L27:
	movl	-176(%rbp), %eax
	cmpl	-180(%rbp), %eax
	jge	.L46
	movq	$2, -40(%rbp)
	jmp	.L32
.L46:
	movq	$4, -40(%rbp)
	jmp	.L32
.L16:
	movl	-164(%rbp), %eax
	cmpl	-168(%rbp), %eax
	jge	.L48
	movq	$46, -40(%rbp)
	jmp	.L32
.L48:
	movq	$16, -40(%rbp)
	jmp	.L32
.L15:
	movl	$1, %eax
	jmp	.L45
.L14:
	call	usage
	movq	$17, -40(%rbp)
	jmp	.L32
.L24:
	movl	$100, -196(%rbp)
	movl	$1, -192(%rbp)
	movl	-192(%rbp), %eax
	cltq
	salq	$2, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -72(%rbp)
	movl	$0, -188(%rbp)
	movq	$3, -40(%rbp)
	jmp	.L32
.L31:
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
	movl	-152(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-48(%rbp), %rdx
	addq	%rcx, %rdx
	imull	-76(%rbp), %eax
	movl	%eax, (%rdx)
	addl	$1, -152(%rbp)
	movq	$13, -40(%rbp)
	jmp	.L32
.L8:
	call	next_i
	movl	%eax, -148(%rbp)
	call	next_i
	movl	%eax, -144(%rbp)
	movl	-148(%rbp), %eax
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
	movl	-164(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-56(%rbp), %rdx
	addq	%rcx, %rdx
	imull	-144(%rbp), %eax
	movl	%eax, (%rdx)
	addl	$1, -164(%rbp)
	movq	$27, -40(%rbp)
	jmp	.L32
.L26:
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
	movl	-188(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-72(%rbp), %rdx
	addq	%rcx, %rdx
	imull	-100(%rbp), %eax
	movl	%eax, (%rdx)
	addl	$1, -188(%rbp)
	movq	$3, -40(%rbp)
	jmp	.L32
.L13:
	movl	-160(%rbp), %edx
	movq	-48(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	i_sabre_codec_volatile
	movl	%eax, -132(%rbp)
	movl	-132(%rbp), %eax
	movl	%eax, -128(%rbp)
	movl	-128(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-48(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$17, -40(%rbp)
	jmp	.L32
.L11:
	call	usage
	movq	$28, -40(%rbp)
	jmp	.L32
.L30:
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
	movl	-176(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-64(%rbp), %rdx
	addq	%rcx, %rdx
	imull	-84(%rbp), %eax
	movl	%eax, (%rdx)
	addl	$1, -176(%rbp)
	movq	$6, -40(%rbp)
	jmp	.L32
.L19:
	movl	$255, -184(%rbp)
	movl	$65025, -180(%rbp)
	movl	-180(%rbp), %eax
	cltq
	salq	$2, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -64(%rbp)
	movl	$0, -176(%rbp)
	movq	$6, -40(%rbp)
	jmp	.L32
.L51:
	nop
.L32:
	jmp	.L50
.L45:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	main, .-main
	.section	.rodata
	.align 8
.LC1:
	.string	"Usage:\n    prog [ARGS]\n\nARGS:\n       0            int-bounds\n       1            big-arr\n       2            big-arr-10x\n       3            empty\n"
	.text
	.globl	usage
	.type	usage, @function
usage:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$0, -8(%rbp)
.L57:
	cmpq	$0, -8(%rbp)
	je	.L53
	cmpq	$1, -8(%rbp)
	jne	.L59
	jmp	.L58
.L53:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$1, -8(%rbp)
	jmp	.L56
.L59:
	nop
.L56:
	jmp	.L57
.L58:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	usage, .-usage
	.type	i_sabre_codec_volatile, @function
i_sabre_codec_volatile:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movl	%esi, -28(%rbp)
	movq	$1, -8(%rbp)
.L69:
	cmpq	$3, -8(%rbp)
	je	.L61
	cmpq	$3, -8(%rbp)
	ja	.L70
	cmpq	$1, -8(%rbp)
	je	.L63
	cmpq	$2, -8(%rbp)
	je	.L64
	jmp	.L70
.L63:
	movl	-28(%rbp), %eax
	addl	$-128, %eax
	cmpl	$1, %eax
	ja	.L65
	movq	$2, -8(%rbp)
	jmp	.L66
.L65:
	movq	$3, -8(%rbp)
	nop
.L66:
	jmp	.L67
.L61:
	movl	$0, %eax
	jmp	.L68
.L64:
	movl	$1, %eax
	jmp	.L68
.L70:
	nop
.L67:
	jmp	.L69
.L68:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	i_sabre_codec_volatile, .-i_sabre_codec_volatile
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
.L77:
	cmpq	$2, -8(%rbp)
	je	.L72
	cmpq	$2, -8(%rbp)
	ja	.L79
	cmpq	$0, -8(%rbp)
	je	.L74
	cmpq	$1, -8(%rbp)
	jne	.L79
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
	jmp	.L78
.L74:
	movq	$2, -8(%rbp)
	jmp	.L76
.L72:
	movl	$0, -12(%rbp)
	addl	$1, -12(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L76
.L79:
	nop
.L76:
	jmp	.L77
.L78:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	next_i, .-next_i
	.globl	next_f
	.type	next_f, @function
next_f:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	$1, -8(%rbp)
.L88:
	cmpq	$2, -8(%rbp)
	je	.L81
	cmpq	$2, -8(%rbp)
	ja	.L90
	cmpq	$0, -8(%rbp)
	je	.L83
	cmpq	$1, -8(%rbp)
	jne	.L90
	movq	$2, -8(%rbp)
	jmp	.L84
.L83:
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
	js	.L85
	pxor	%xmm0, %xmm0
	cvtsi2ssq	%rax, %xmm0
	jmp	.L86
.L85:
	movq	%rax, %rdx
	shrq	%rdx
	andl	$1, %eax
	orq	%rax, %rdx
	pxor	%xmm0, %xmm0
	cvtsi2ssq	%rdx, %xmm0
	addss	%xmm0, %xmm0
.L86:
	movss	.LC2(%rip), %xmm1
	divss	%xmm1, %xmm0
	jmp	.L89
.L81:
	movl	$0, -12(%rbp)
	addl	$1, -12(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L84
.L90:
	nop
.L84:
	jmp	.L88
.L89:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
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
