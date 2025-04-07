	.file	"brianprogh1_ds_5_flatten.c"
	.text
	.globl	_TIG_IZ_fcAG_argv
	.bss
	.align 8
	.type	_TIG_IZ_fcAG_argv, @object
	.size	_TIG_IZ_fcAG_argv, 8
_TIG_IZ_fcAG_argv:
	.zero	8
	.globl	_TIG_IZ_fcAG_envp
	.align 8
	.type	_TIG_IZ_fcAG_envp, @object
	.size	_TIG_IZ_fcAG_envp, 8
_TIG_IZ_fcAG_envp:
	.zero	8
	.globl	_TIG_IZ_fcAG_argc
	.align 4
	.type	_TIG_IZ_fcAG_argc, @object
	.size	_TIG_IZ_fcAG_argc, 4
_TIG_IZ_fcAG_argc:
	.zero	4
	.section	.rodata
.LC1:
	.string	"stack is underflow"
	.text
	.globl	pop
	.type	pop, @function
pop:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	$3, -8(%rbp)
.L13:
	cmpq	$6, -8(%rbp)
	ja	.L14
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L4(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L4(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L4:
	.long	.L8-.L4
	.long	.L7-.L4
	.long	.L14-.L4
	.long	.L6-.L4
	.long	.L14-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L7:
	movss	-12(%rbp), %xmm0
	jmp	.L9
.L6:
	movq	-32(%rbp), %rax
	movl	(%rax), %eax
	cmpl	$-1, %eax
	jne	.L10
	movq	$0, -8(%rbp)
	jmp	.L12
.L10:
	movq	$6, -8(%rbp)
	jmp	.L12
.L3:
	movq	-32(%rbp), %rax
	movl	(%rax), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm0
	movss	%xmm0, -12(%rbp)
	movq	-32(%rbp), %rax
	movl	(%rax), %eax
	leal	-1(%rax), %edx
	movq	-32(%rbp), %rax
	movl	%edx, (%rax)
	movq	$1, -8(%rbp)
	jmp	.L12
.L5:
	pxor	%xmm0, %xmm0
	jmp	.L9
.L8:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$5, -8(%rbp)
	jmp	.L12
.L14:
	nop
.L12:
	jmp	.L13
.L9:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	pop, .-pop
	.section	.rodata
.LC2:
	.string	"\n move disc %d from %c to %c"
	.text
	.globl	tower
	.type	tower, @function
tower:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, -20(%rbp)
	movl	%esi, -24(%rbp)
	movl	%edx, -28(%rbp)
	movl	%ecx, -32(%rbp)
	movq	$4, -8(%rbp)
.L25:
	cmpq	$4, -8(%rbp)
	je	.L16
	cmpq	$4, -8(%rbp)
	ja	.L26
	cmpq	$3, -8(%rbp)
	je	.L27
	cmpq	$3, -8(%rbp)
	ja	.L26
	cmpq	$1, -8(%rbp)
	je	.L28
	cmpq	$2, -8(%rbp)
	je	.L20
	jmp	.L26
.L16:
	cmpl	$0, -20(%rbp)
	jne	.L21
	movq	$3, -8(%rbp)
	jmp	.L23
.L21:
	movq	$2, -8(%rbp)
	jmp	.L23
.L20:
	movl	-20(%rbp), %eax
	leal	-1(%rax), %edi
	movl	-28(%rbp), %ecx
	movl	-32(%rbp), %edx
	movl	-24(%rbp), %eax
	movl	%eax, %esi
	call	tower
	movl	-32(%rbp), %ecx
	movl	-24(%rbp), %edx
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-20(%rbp), %eax
	leal	-1(%rax), %edi
	movl	-32(%rbp), %ecx
	movl	-24(%rbp), %edx
	movl	-28(%rbp), %eax
	movl	%eax, %esi
	call	tower
	movq	$1, -8(%rbp)
	jmp	.L23
.L26:
	nop
.L23:
	jmp	.L25
.L27:
	nop
	jmp	.L15
.L28:
	nop
.L15:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	tower, .-tower
	.section	.rodata
.LC3:
	.string	"enter the valid postfix expn:"
.LC4:
	.string	"%s"
	.align 8
.LC5:
	.string	"the result of the evaluation of postfix expn is %f"
	.text
	.globl	evalpostfix
	.type	evalpostfix, @function
evalpostfix:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$176, %rsp
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$10, -120(%rbp)
.L47:
	cmpq	$14, -120(%rbp)
	ja	.L50
	movq	-120(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L32(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L32(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L32:
	.long	.L50-.L32
	.long	.L40-.L32
	.long	.L50-.L32
	.long	.L39-.L32
	.long	.L50-.L32
	.long	.L50-.L32
	.long	.L50-.L32
	.long	.L38-.L32
	.long	.L37-.L32
	.long	.L50-.L32
	.long	.L36-.L32
	.long	.L35-.L32
	.long	.L34-.L32
	.long	.L33-.L32
	.long	.L51-.L32
	.text
.L34:
	movl	-156(%rbp), %eax
	cltq
	movzbl	-32(%rbp,%rax), %eax
	testb	%al, %al
	je	.L42
	movq	$1, -120(%rbp)
	jmp	.L44
.L42:
	movq	$7, -120(%rbp)
	jmp	.L44
.L37:
	leaq	-160(%rbp), %rdx
	leaq	-112(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	pop
	movd	%xmm0, %eax
	movl	%eax, -148(%rbp)
	leaq	-160(%rbp), %rdx
	leaq	-112(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	pop
	movd	%xmm0, %eax
	movl	%eax, -144(%rbp)
	movsbl	-161(%rbp), %edx
	movss	-148(%rbp), %xmm0
	movl	-144(%rbp), %eax
	movl	%edx, %edi
	movaps	%xmm0, %xmm1
	movd	%eax, %xmm0
	call	compute
	movd	%xmm0, %eax
	movl	%eax, -140(%rbp)
	movl	-140(%rbp), %ecx
	leaq	-160(%rbp), %rdx
	leaq	-112(%rbp), %rax
	movd	%ecx, %xmm0
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	push
	movq	$12, -120(%rbp)
	jmp	.L44
.L40:
	movl	-156(%rbp), %eax
	movl	%eax, -132(%rbp)
	addl	$1, -156(%rbp)
	movl	-132(%rbp), %eax
	cltq
	movzbl	-32(%rbp,%rax), %eax
	movb	%al, -161(%rbp)
	call	__ctype_b_loc@PLT
	movq	%rax, -128(%rbp)
	movq	$11, -120(%rbp)
	jmp	.L44
.L39:
	movl	$0, -156(%rbp)
	movl	$0, -136(%rbp)
	movl	$-1, -160(%rbp)
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-32(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$12, -120(%rbp)
	jmp	.L44
.L35:
	movq	-128(%rbp), %rax
	movq	(%rax), %rdx
	movsbq	-161(%rbp), %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$2048, %eax
	testl	%eax, %eax
	je	.L45
	movq	$13, -120(%rbp)
	jmp	.L44
.L45:
	movq	$8, -120(%rbp)
	jmp	.L44
.L33:
	movsbl	-161(%rbp), %eax
	subl	$48, %eax
	pxor	%xmm0, %xmm0
	cvtsi2ssl	%eax, %xmm0
	movss	%xmm0, -152(%rbp)
	movl	-152(%rbp), %ecx
	leaq	-160(%rbp), %rdx
	leaq	-112(%rbp), %rax
	movd	%ecx, %xmm0
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	push
	movq	$12, -120(%rbp)
	jmp	.L44
.L36:
	movq	$3, -120(%rbp)
	jmp	.L44
.L38:
	leaq	-160(%rbp), %rdx
	leaq	-112(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	pop
	movd	%xmm0, %eax
	movl	%eax, -140(%rbp)
	pxor	%xmm2, %xmm2
	cvtss2sd	-140(%rbp), %xmm2
	movq	%xmm2, %rax
	movq	%rax, %xmm0
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$14, -120(%rbp)
	jmp	.L44
.L50:
	nop
.L44:
	jmp	.L47
.L51:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L49
	call	__stack_chk_fail@PLT
.L49:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	evalpostfix, .-evalpostfix
	.globl	push
	.type	push, @function
push:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movss	%xmm0, -36(%rbp)
	movq	$2, -8(%rbp)
.L58:
	cmpq	$2, -8(%rbp)
	je	.L53
	cmpq	$2, -8(%rbp)
	ja	.L59
	cmpq	$0, -8(%rbp)
	je	.L60
	cmpq	$1, -8(%rbp)
	jne	.L59
	movq	-32(%rbp), %rax
	movl	(%rax), %eax
	leal	1(%rax), %edx
	movq	-32(%rbp), %rax
	movl	%edx, (%rax)
	movq	-32(%rbp), %rax
	movl	(%rax), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movss	-36(%rbp), %xmm0
	movss	%xmm0, (%rax)
	movq	$0, -8(%rbp)
	jmp	.L56
.L53:
	movq	$1, -8(%rbp)
	jmp	.L56
.L59:
	nop
.L56:
	jmp	.L58
.L60:
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	push, .-push
	.globl	compute
	.type	compute, @function
compute:
.LFB10:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movss	%xmm0, -20(%rbp)
	movss	%xmm1, -24(%rbp)
	movl	%edi, %eax
	movb	%al, -28(%rbp)
	movq	$4, -8(%rbp)
.L84:
	cmpq	$8, -8(%rbp)
	ja	.L85
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L64(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L64(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L64:
	.long	.L72-.L64
	.long	.L71-.L64
	.long	.L70-.L64
	.long	.L69-.L64
	.long	.L68-.L64
	.long	.L67-.L64
	.long	.L66-.L64
	.long	.L65-.L64
	.long	.L63-.L64
	.text
.L68:
	movsbl	-28(%rbp), %eax
	cmpl	$47, %eax
	jg	.L73
	cmpl	$36, %eax
	jl	.L74
	subl	$36, %eax
	cmpl	$11, %eax
	ja	.L74
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L76(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L76(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L76:
	.long	.L80-.L76
	.long	.L74-.L76
	.long	.L74-.L76
	.long	.L74-.L76
	.long	.L74-.L76
	.long	.L74-.L76
	.long	.L79-.L76
	.long	.L78-.L76
	.long	.L74-.L76
	.long	.L77-.L76
	.long	.L74-.L76
	.long	.L75-.L76
	.text
.L73:
	cmpl	$94, %eax
	jne	.L74
.L80:
	movq	$8, -8(%rbp)
	jmp	.L81
.L75:
	movq	$7, -8(%rbp)
	jmp	.L81
.L79:
	movq	$6, -8(%rbp)
	jmp	.L81
.L77:
	movq	$2, -8(%rbp)
	jmp	.L81
.L78:
	movq	$0, -8(%rbp)
	jmp	.L81
.L74:
	movq	$5, -8(%rbp)
	nop
.L81:
	jmp	.L82
.L63:
	pxor	%xmm0, %xmm0
	cvtss2sd	-20(%rbp), %xmm0
	pxor	%xmm2, %xmm2
	cvtss2sd	-24(%rbp), %xmm2
	movq	%xmm2, %rax
	movapd	%xmm0, %xmm1
	movq	%rax, %xmm0
	call	pow@PLT
	movq	%xmm0, %rax
	movq	%rax, -16(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L82
.L71:
	pxor	%xmm0, %xmm0
	jmp	.L83
.L69:
	pxor	%xmm0, %xmm0
	cvtsd2ss	-16(%rbp), %xmm0
	jmp	.L83
.L66:
	movss	-20(%rbp), %xmm0
	mulss	-24(%rbp), %xmm0
	jmp	.L83
.L67:
	movq	$1, -8(%rbp)
	jmp	.L82
.L72:
	movss	-20(%rbp), %xmm0
	addss	-24(%rbp), %xmm0
	jmp	.L83
.L65:
	movss	-20(%rbp), %xmm0
	divss	-24(%rbp), %xmm0
	jmp	.L83
.L70:
	movss	-20(%rbp), %xmm0
	subss	-24(%rbp), %xmm0
	jmp	.L83
.L85:
	nop
.L82:
	jmp	.L84
.L83:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	compute, .-compute
	.section	.rodata
.LC6:
	.string	"\n enter the number of discs:"
.LC7:
	.string	"%d"
.LC8:
	.string	"\n total number of moves are"
	.align 8
.LC9:
	.string	"\n 1. evaluation of postfix expn"
.LC10:
	.string	"\n 2. tower of hanoi"
.LC11:
	.string	"\n 3. exit"
.LC12:
	.string	"\n enter your choice:"
.LC13:
	.string	"\n main menu"
	.text
	.globl	main
	.type	main, @function
main:
.LFB11:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_fcAG_envp(%rip)
	nop
.L87:
	movq	$0, _TIG_IZ_fcAG_argv(%rip)
	nop
.L88:
	movl	$0, _TIG_IZ_fcAG_argc(%rip)
	nop
	nop
.L89:
.L90:
#APP
# 124 "brianprogh1_ds_5.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-fcAG--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_fcAG_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_fcAG_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_fcAG_envp(%rip)
	nop
	movq	$7, -16(%rbp)
.L106:
	cmpq	$10, -16(%rbp)
	ja	.L108
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L93(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L93(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L93:
	.long	.L99-.L93
	.long	.L108-.L93
	.long	.L98-.L93
	.long	.L97-.L93
	.long	.L108-.L93
	.long	.L108-.L93
	.long	.L96-.L93
	.long	.L95-.L93
	.long	.L108-.L93
	.long	.L94-.L93
	.long	.L92-.L93
	.text
.L97:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-24(%rbp), %eax
	movl	$67, %ecx
	movl	$66, %edx
	movl	$65, %esi
	movl	%eax, %edi
	call	tower
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$10, -16(%rbp)
	jmp	.L100
.L94:
	movq	$10, -16(%rbp)
	jmp	.L100
.L96:
	movl	$0, %edi
	call	exit@PLT
.L92:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-20(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$0, -16(%rbp)
	jmp	.L100
.L99:
	movl	-20(%rbp), %eax
	cmpl	$3, %eax
	je	.L101
	cmpl	$3, %eax
	jg	.L102
	cmpl	$1, %eax
	je	.L103
	cmpl	$2, %eax
	je	.L104
	jmp	.L102
.L101:
	movq	$6, -16(%rbp)
	jmp	.L105
.L103:
	movq	$2, -16(%rbp)
	jmp	.L105
.L104:
	movq	$3, -16(%rbp)
	jmp	.L105
.L102:
	movq	$9, -16(%rbp)
	nop
.L105:
	jmp	.L100
.L95:
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$10, -16(%rbp)
	jmp	.L100
.L98:
	call	evalpostfix
	movq	$10, -16(%rbp)
	jmp	.L100
.L108:
	nop
.L100:
	jmp	.L106
	.cfi_endproc
.LFE11:
	.size	main, .-main
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
