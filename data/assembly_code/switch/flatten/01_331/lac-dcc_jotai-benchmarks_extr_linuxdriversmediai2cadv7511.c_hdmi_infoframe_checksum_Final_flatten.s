	.file	"lac-dcc_jotai-benchmarks_extr_linuxdriversmediai2cadv7511.c_hdmi_infoframe_checksum_Final_flatten.c"
	.text
	.globl	_TIG_IZ_WII9_argc
	.bss
	.align 4
	.type	_TIG_IZ_WII9_argc, @object
	.size	_TIG_IZ_WII9_argc, 4
_TIG_IZ_WII9_argc:
	.zero	4
	.globl	rand_primes
	.align 32
	.type	rand_primes, @object
	.size	rand_primes, 100
rand_primes:
	.zero	100
	.globl	_TIG_IZ_WII9_envp
	.align 8
	.type	_TIG_IZ_WII9_envp, @object
	.size	_TIG_IZ_WII9_envp, 8
_TIG_IZ_WII9_envp:
	.zero	8
	.globl	_TIG_IZ_WII9_argv
	.align 8
	.type	_TIG_IZ_WII9_argv, @object
	.size	_TIG_IZ_WII9_argv, 8
_TIG_IZ_WII9_argv:
	.zero	8
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
	movq	$1, -8(%rbp)
.L6:
	cmpq	$0, -8(%rbp)
	je	.L7
	cmpq	$1, -8(%rbp)
	jne	.L8
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$0, -8(%rbp)
	jmp	.L4
.L8:
	nop
.L4:
	jmp	.L6
.L7:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	usage, .-usage
	.section	.rodata
.LC1:
	.string	"%ld\n"
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
	subq	$160, %rsp
	movl	%edi, -132(%rbp)
	movq	%rsi, -144(%rbp)
	movq	%rdx, -152(%rbp)
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
.L10:
	movq	$0, _TIG_IZ_WII9_envp(%rip)
	nop
.L11:
	movq	$0, _TIG_IZ_WII9_argv(%rip)
	nop
.L12:
	movl	$0, _TIG_IZ_WII9_argc(%rip)
	nop
	nop
.L13:
.L14:
#APP
# 131 "lac-dcc_jotai-benchmarks_extr_linuxdriversmediai2cadv7511.c_hdmi_infoframe_checksum_Final.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-WII9--0
# 0 "" 2
#NO_APP
	movl	-132(%rbp), %eax
	movl	%eax, _TIG_IZ_WII9_argc(%rip)
	movq	-144(%rbp), %rax
	movq	%rax, _TIG_IZ_WII9_argv(%rip)
	movq	-152(%rbp), %rax
	movq	%rax, _TIG_IZ_WII9_envp(%rip)
	nop
	movq	$2, -56(%rbp)
.L43:
	cmpq	$28, -56(%rbp)
	ja	.L44
	movq	-56(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L17(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L17(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L17:
	.long	.L44-.L17
	.long	.L44-.L17
	.long	.L31-.L17
	.long	.L30-.L17
	.long	.L44-.L17
	.long	.L29-.L17
	.long	.L44-.L17
	.long	.L44-.L17
	.long	.L28-.L17
	.long	.L44-.L17
	.long	.L44-.L17
	.long	.L27-.L17
	.long	.L26-.L17
	.long	.L25-.L17
	.long	.L24-.L17
	.long	.L23-.L17
	.long	.L22-.L17
	.long	.L21-.L17
	.long	.L20-.L17
	.long	.L44-.L17
	.long	.L44-.L17
	.long	.L44-.L17
	.long	.L19-.L17
	.long	.L44-.L17
	.long	.L18-.L17
	.long	.L44-.L17
	.long	.L44-.L17
	.long	.L44-.L17
	.long	.L16-.L17
	.text
.L20:
	movq	-88(%rbp), %rdx
	movq	-80(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	hdmi_infoframe_checksum
	movq	%rax, -48(%rbp)
	movq	-48(%rbp), %rax
	movq	%rax, -40(%rbp)
	movq	-40(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-80(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$13, -56(%rbp)
	jmp	.L32
.L24:
	movq	$10, -72(%rbp)
	movl	$100, -116(%rbp)
	movl	-116(%rbp), %eax
	cltq
	salq	$3, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, -64(%rbp)
	movl	$0, -112(%rbp)
	movq	$8, -56(%rbp)
	jmp	.L32
.L23:
	call	usage
	movq	$22, -56(%rbp)
	jmp	.L32
.L26:
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
	imull	-96(%rbp), %eax
	movl	-120(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,8), %rcx
	movq	-80(%rbp), %rdx
	addq	%rcx, %rdx
	cltq
	movq	%rax, (%rdx)
	addl	$1, -120(%rbp)
	movq	$24, -56(%rbp)
	jmp	.L32
.L28:
	movl	-112(%rbp), %eax
	cmpl	-116(%rbp), %eax
	jge	.L33
	movq	$28, -56(%rbp)
	jmp	.L32
.L33:
	movq	$5, -56(%rbp)
	jmp	.L32
.L30:
	movq	-144(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movl	%eax, -92(%rbp)
	movl	-92(%rbp), %eax
	movl	%eax, -128(%rbp)
	movq	$11, -56(%rbp)
	jmp	.L32
.L22:
	movq	$255, -88(%rbp)
	movl	$65025, -124(%rbp)
	movl	-124(%rbp), %eax
	cltq
	salq	$3, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -32(%rbp)
	movq	-32(%rbp), %rax
	movq	%rax, -80(%rbp)
	movl	$0, -120(%rbp)
	movq	$24, -56(%rbp)
	jmp	.L32
.L18:
	movl	-120(%rbp), %eax
	cmpl	-124(%rbp), %eax
	jge	.L35
	movq	$12, -56(%rbp)
	jmp	.L32
.L35:
	movq	$18, -56(%rbp)
	jmp	.L32
.L27:
	cmpl	$0, -128(%rbp)
	je	.L37
	cmpl	$1, -128(%rbp)
	jne	.L38
	movq	$14, -56(%rbp)
	jmp	.L39
.L37:
	movq	$16, -56(%rbp)
	jmp	.L39
.L38:
	movq	$17, -56(%rbp)
	nop
.L39:
	jmp	.L32
.L25:
	movl	$0, %eax
	jmp	.L40
.L21:
	call	usage
	movq	$13, -56(%rbp)
	jmp	.L32
.L19:
	movl	$1, %eax
	jmp	.L40
.L16:
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
	imull	-104(%rbp), %eax
	movl	-112(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,8), %rcx
	movq	-64(%rbp), %rdx
	addq	%rcx, %rdx
	cltq
	movq	%rax, (%rdx)
	addl	$1, -112(%rbp)
	movq	$8, -56(%rbp)
	jmp	.L32
.L29:
	movq	-72(%rbp), %rdx
	movq	-64(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	hdmi_infoframe_checksum
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$13, -56(%rbp)
	jmp	.L32
.L31:
	cmpl	$2, -132(%rbp)
	je	.L41
	movq	$15, -56(%rbp)
	jmp	.L32
.L41:
	movq	$3, -56(%rbp)
	jmp	.L32
.L44:
	nop
.L32:
	jmp	.L43
.L40:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	main, .-main
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
	movq	$2, -8(%rbp)
.L51:
	cmpq	$2, -8(%rbp)
	je	.L46
	cmpq	$2, -8(%rbp)
	ja	.L53
	cmpq	$0, -8(%rbp)
	je	.L48
	cmpq	$1, -8(%rbp)
	jne	.L53
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
	jmp	.L52
.L48:
	movl	$0, -12(%rbp)
	addl	$1, -12(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L50
.L46:
	movq	$0, -8(%rbp)
	jmp	.L50
.L53:
	nop
.L50:
	jmp	.L51
.L52:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	next_i, .-next_i
	.globl	next_f
	.type	next_f, @function
next_f:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	$0, -8(%rbp)
.L62:
	cmpq	$2, -8(%rbp)
	je	.L55
	cmpq	$2, -8(%rbp)
	ja	.L64
	cmpq	$0, -8(%rbp)
	je	.L57
	cmpq	$1, -8(%rbp)
	jne	.L64
	movl	$0, -12(%rbp)
	addl	$1, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L58
.L57:
	movq	$1, -8(%rbp)
	jmp	.L58
.L55:
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
	movss	.LC2(%rip), %xmm1
	divss	%xmm1, %xmm0
	jmp	.L63
.L64:
	nop
.L58:
	jmp	.L62
.L63:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	next_f, .-next_f
	.type	hdmi_infoframe_checksum, @function
hdmi_infoframe_checksum:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -40(%rbp)
	movq	%rsi, -48(%rbp)
	movq	$3, -8(%rbp)
.L77:
	cmpq	$6, -8(%rbp)
	ja	.L79
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L68(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L68(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L68:
	.long	.L72-.L68
	.long	.L79-.L68
	.long	.L79-.L68
	.long	.L71-.L68
	.long	.L70-.L68
	.long	.L69-.L68
	.long	.L67-.L68
	.text
.L70:
	movq	-16(%rbp), %rax
	leaq	0(,%rax,8), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	addq	%rax, -24(%rbp)
	addq	$1, -16(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L73
.L71:
	movq	$0, -8(%rbp)
	jmp	.L73
.L67:
	movl	$256, %eax
	subq	-24(%rbp), %rax
	jmp	.L78
.L69:
	movq	-16(%rbp), %rax
	cmpq	-48(%rbp), %rax
	jnb	.L75
	movq	$4, -8(%rbp)
	jmp	.L73
.L75:
	movq	$6, -8(%rbp)
	jmp	.L73
.L72:
	movq	$0, -24(%rbp)
	movq	$0, -16(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L73
.L79:
	nop
.L73:
	jmp	.L77
.L78:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	hdmi_infoframe_checksum, .-hdmi_infoframe_checksum
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
