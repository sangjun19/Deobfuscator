	.file	"lac-dcc_jotai-benchmarks_extr_linuxdriversgpudrmamddisplaydccoredc_link_ddc.c_ddc_service_set_dongle_type_Final_flatten.c"
	.text
	.globl	_TIG_IZ_Gz0r_envp
	.bss
	.align 8
	.type	_TIG_IZ_Gz0r_envp, @object
	.size	_TIG_IZ_Gz0r_envp, 8
_TIG_IZ_Gz0r_envp:
	.zero	8
	.globl	_TIG_IZ_Gz0r_argc
	.align 4
	.type	_TIG_IZ_Gz0r_argc, @object
	.size	_TIG_IZ_Gz0r_argc, 4
_TIG_IZ_Gz0r_argc:
	.zero	4
	.globl	rand_primes
	.align 32
	.type	rand_primes, @object
	.size	rand_primes, 100
rand_primes:
	.zero	100
	.globl	_TIG_IZ_Gz0r_argv
	.align 8
	.type	_TIG_IZ_Gz0r_argv, @object
	.size	_TIG_IZ_Gz0r_argv, 8
_TIG_IZ_Gz0r_argv:
	.zero	8
	.text
	.globl	ddc_service_set_dongle_type
	.type	ddc_service_set_dongle_type, @function
ddc_service_set_dongle_type:
.LFB2:
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
.L6:
	cmpq	$0, -8(%rbp)
	je	.L7
	cmpq	$1, -8(%rbp)
	jne	.L8
	movl	-28(%rbp), %edx
	movq	-24(%rbp), %rax
	movl	%edx, (%rax)
	movq	$0, -8(%rbp)
	jmp	.L4
.L8:
	nop
.L4:
	jmp	.L6
.L7:
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	ddc_service_set_dongle_type, .-ddc_service_set_dongle_type
	.globl	next_f
	.type	next_f, @function
next_f:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	$0, -8(%rbp)
.L17:
	cmpq	$2, -8(%rbp)
	je	.L10
	cmpq	$2, -8(%rbp)
	ja	.L19
	cmpq	$0, -8(%rbp)
	je	.L12
	cmpq	$1, -8(%rbp)
	jne	.L19
	movl	$0, -12(%rbp)
	addl	$1, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L13
.L12:
	movq	$1, -8(%rbp)
	jmp	.L13
.L10:
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
	js	.L14
	pxor	%xmm0, %xmm0
	cvtsi2ssq	%rax, %xmm0
	jmp	.L15
.L14:
	movq	%rax, %rdx
	shrq	%rdx
	andl	$1, %eax
	orq	%rax, %rdx
	pxor	%xmm0, %xmm0
	cvtsi2ssq	%rdx, %xmm0
	addss	%xmm0, %xmm0
.L15:
	movss	.LC0(%rip), %xmm1
	divss	%xmm1, %xmm0
	jmp	.L18
.L19:
	nop
.L13:
	jmp	.L17
.L18:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	next_f, .-next_f
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
	movq	$1, -8(%rbp)
.L26:
	cmpq	$2, -8(%rbp)
	je	.L21
	cmpq	$2, -8(%rbp)
	ja	.L28
	cmpq	$0, -8(%rbp)
	je	.L23
	cmpq	$1, -8(%rbp)
	jne	.L28
	movq	$0, -8(%rbp)
	jmp	.L24
.L23:
	movl	$0, -12(%rbp)
	addl	$1, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L24
.L21:
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
.L28:
	nop
.L24:
	jmp	.L26
.L27:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	next_i, .-next_i
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
	movq	$1, -8(%rbp)
.L34:
	cmpq	$0, -8(%rbp)
	je	.L35
	cmpq	$1, -8(%rbp)
	jne	.L36
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$0, -8(%rbp)
	jmp	.L32
.L36:
	nop
.L32:
	jmp	.L34
.L35:
	nop
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
.L38:
	movq	$0, _TIG_IZ_Gz0r_envp(%rip)
	nop
.L39:
	movq	$0, _TIG_IZ_Gz0r_argv(%rip)
	nop
.L40:
	movl	$0, _TIG_IZ_Gz0r_argc(%rip)
	nop
	nop
.L41:
.L42:
#APP
# 134 "lac-dcc_jotai-benchmarks_extr_linuxdriversgpudrmamddisplaydccoredc_link_ddc.c_ddc_service_set_dongle_type_Final.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-Gz0r--0
# 0 "" 2
#NO_APP
	movl	-132(%rbp), %eax
	movl	%eax, _TIG_IZ_Gz0r_argc(%rip)
	movq	-144(%rbp), %rax
	movq	%rax, _TIG_IZ_Gz0r_argv(%rip)
	movq	-152(%rbp), %rax
	movq	%rax, _TIG_IZ_Gz0r_envp(%rip)
	nop
	movq	$26, -32(%rbp)
.L79:
	cmpq	$38, -32(%rbp)
	ja	.L80
	movq	-32(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L45(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L45(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L45:
	.long	.L63-.L45
	.long	.L80-.L45
	.long	.L80-.L45
	.long	.L80-.L45
	.long	.L62-.L45
	.long	.L61-.L45
	.long	.L80-.L45
	.long	.L80-.L45
	.long	.L80-.L45
	.long	.L60-.L45
	.long	.L80-.L45
	.long	.L80-.L45
	.long	.L80-.L45
	.long	.L59-.L45
	.long	.L80-.L45
	.long	.L58-.L45
	.long	.L80-.L45
	.long	.L80-.L45
	.long	.L57-.L45
	.long	.L56-.L45
	.long	.L55-.L45
	.long	.L80-.L45
	.long	.L54-.L45
	.long	.L53-.L45
	.long	.L80-.L45
	.long	.L52-.L45
	.long	.L51-.L45
	.long	.L80-.L45
	.long	.L50-.L45
	.long	.L80-.L45
	.long	.L80-.L45
	.long	.L80-.L45
	.long	.L49-.L45
	.long	.L48-.L45
	.long	.L80-.L45
	.long	.L47-.L45
	.long	.L80-.L45
	.long	.L46-.L45
	.long	.L44-.L45
	.text
.L57:
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
	movl	-100(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-48(%rbp), %rdx
	addq	%rcx, %rdx
	imull	-76(%rbp), %eax
	movl	%eax, (%rdx)
	addl	$1, -100(%rbp)
	movq	$4, -32(%rbp)
	jmp	.L64
.L52:
	movl	-108(%rbp), %edx
	movq	-48(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	ddc_service_set_dongle_type
	movq	-48(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$22, -32(%rbp)
	jmp	.L64
.L62:
	movl	-100(%rbp), %eax
	cmpl	-104(%rbp), %eax
	jge	.L65
	movq	$18, -32(%rbp)
	jmp	.L64
.L65:
	movq	$25, -32(%rbp)
	jmp	.L64
.L58:
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
	movl	-88(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-40(%rbp), %rdx
	addq	%rcx, %rdx
	imull	-68(%rbp), %eax
	movl	%eax, (%rdx)
	addl	$1, -88(%rbp)
	movq	$20, -32(%rbp)
	jmp	.L64
.L53:
	movl	$0, -108(%rbp)
	movl	$100, -104(%rbp)
	movl	-104(%rbp), %eax
	cltq
	salq	$2, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -48(%rbp)
	movl	$0, -100(%rbp)
	movq	$4, -32(%rbp)
	jmp	.L64
.L51:
	cmpl	$2, -132(%rbp)
	je	.L67
	movq	$28, -32(%rbp)
	jmp	.L64
.L67:
	movq	$37, -32(%rbp)
	jmp	.L64
.L60:
	movl	-96(%rbp), %edx
	movq	-40(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	ddc_service_set_dongle_type
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$22, -32(%rbp)
	jmp	.L64
.L59:
	movl	$0, -96(%rbp)
	movl	$1, -92(%rbp)
	movl	-92(%rbp), %eax
	cltq
	salq	$2, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -40(%rbp)
	movl	$0, -88(%rbp)
	movq	$20, -32(%rbp)
	jmp	.L64
.L56:
	movl	$1, %eax
	jmp	.L69
.L49:
	movl	-120(%rbp), %edx
	movq	-56(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	ddc_service_set_dongle_type
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$22, -32(%rbp)
	jmp	.L64
.L44:
	movl	-112(%rbp), %eax
	cmpl	-116(%rbp), %eax
	jge	.L70
	movq	$0, -32(%rbp)
	jmp	.L64
.L70:
	movq	$32, -32(%rbp)
	jmp	.L64
.L54:
	movl	$0, %eax
	jmp	.L69
.L50:
	call	usage
	movq	$19, -32(%rbp)
	jmp	.L64
.L61:
	call	usage
	movq	$22, -32(%rbp)
	jmp	.L64
.L48:
	cmpl	$2, -124(%rbp)
	je	.L72
	cmpl	$2, -124(%rbp)
	jg	.L73
	cmpl	$0, -124(%rbp)
	je	.L74
	cmpl	$1, -124(%rbp)
	je	.L75
	jmp	.L73
.L72:
	movq	$13, -32(%rbp)
	jmp	.L76
.L75:
	movq	$23, -32(%rbp)
	jmp	.L76
.L74:
	movq	$35, -32(%rbp)
	jmp	.L76
.L73:
	movq	$5, -32(%rbp)
	nop
.L76:
	jmp	.L64
.L46:
	movq	-144(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movl	%eax, -84(%rbp)
	movl	-84(%rbp), %eax
	movl	%eax, -124(%rbp)
	movq	$33, -32(%rbp)
	jmp	.L64
.L63:
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
	movq	-56(%rbp), %rdx
	addq	%rcx, %rdx
	imull	-60(%rbp), %eax
	movl	%eax, (%rdx)
	addl	$1, -112(%rbp)
	movq	$38, -32(%rbp)
	jmp	.L64
.L47:
	movl	$0, -120(%rbp)
	movl	$65025, -116(%rbp)
	movl	-116(%rbp), %eax
	cltq
	salq	$2, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, -56(%rbp)
	movl	$0, -112(%rbp)
	movq	$38, -32(%rbp)
	jmp	.L64
.L55:
	movl	-88(%rbp), %eax
	cmpl	-92(%rbp), %eax
	jge	.L77
	movq	$15, -32(%rbp)
	jmp	.L64
.L77:
	movq	$9, -32(%rbp)
	jmp	.L64
.L80:
	nop
.L64:
	jmp	.L79
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
