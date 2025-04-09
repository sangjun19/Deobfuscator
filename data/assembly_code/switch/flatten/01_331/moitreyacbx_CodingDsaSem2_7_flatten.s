	.file	"moitreyacbx_CodingDsaSem2_7_flatten.c"
	.text
	.globl	_TIG_IZ_5hz8_envp
	.bss
	.align 8
	.type	_TIG_IZ_5hz8_envp, @object
	.size	_TIG_IZ_5hz8_envp, 8
_TIG_IZ_5hz8_envp:
	.zero	8
	.globl	_TIG_IZ_5hz8_argc
	.align 4
	.type	_TIG_IZ_5hz8_argc, @object
	.size	_TIG_IZ_5hz8_argc, 4
_TIG_IZ_5hz8_argc:
	.zero	4
	.globl	_TIG_IZ_5hz8_argv
	.align 8
	.type	_TIG_IZ_5hz8_argv, @object
	.size	_TIG_IZ_5hz8_argv, 8
_TIG_IZ_5hz8_argv:
	.zero	8
	.text
	.globl	deletefirst
	.type	deletefirst, @function
deletefirst:
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
	ja	.L9
	cmpq	$0, -8(%rbp)
	je	.L4
	cmpq	$1, -8(%rbp)
	jne	.L9
	movq	$0, -8(%rbp)
	jmp	.L5
.L4:
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, -16(%rbp)
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	movq	8(%rax), %rdx
	movq	-24(%rbp), %rax
	movq	%rdx, (%rax)
	movq	$2, -8(%rbp)
	jmp	.L5
.L2:
	movq	-16(%rbp), %rax
	movss	(%rax), %xmm0
	jmp	.L8
.L9:
	nop
.L5:
	jmp	.L7
.L8:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	deletefirst, .-deletefirst
	.section	.rodata
	.align 8
.LC0:
	.string	"Enter the postfix expression :"
.LC1:
	.string	"Evaluated value = %f\n"
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
	subq	$112, %rsp
	movl	%edi, -84(%rbp)
	movq	%rsi, -96(%rbp)
	movq	%rdx, -104(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_5hz8_envp(%rip)
	nop
.L11:
	movq	$0, _TIG_IZ_5hz8_argv(%rip)
	nop
.L12:
	movl	$0, _TIG_IZ_5hz8_argc(%rip)
	nop
	nop
.L13:
.L14:
#APP
# 104 "moitreyacbx_CodingDsaSem2_7.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-5hz8--0
# 0 "" 2
#NO_APP
	movl	-84(%rbp), %eax
	movl	%eax, _TIG_IZ_5hz8_argc(%rip)
	movq	-96(%rbp), %rax
	movq	%rax, _TIG_IZ_5hz8_argv(%rip)
	movq	-104(%rbp), %rax
	movq	%rax, _TIG_IZ_5hz8_envp(%rip)
	nop
	movq	$2, -72(%rbp)
.L20:
	cmpq	$2, -72(%rbp)
	je	.L15
	cmpq	$2, -72(%rbp)
	ja	.L23
	cmpq	$0, -72(%rbp)
	je	.L17
	cmpq	$1, -72(%rbp)
	jne	.L23
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L21
	jmp	.L22
.L17:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-64(%rbp), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	gets@PLT
	leaq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	postfix_evaluation
	movd	%xmm0, %eax
	movl	%eax, -76(%rbp)
	pxor	%xmm1, %xmm1
	cvtss2sd	-76(%rbp), %xmm1
	movq	%xmm1, %rax
	movq	%rax, %xmm0
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$1, -72(%rbp)
	jmp	.L19
.L15:
	movq	$0, -72(%rbp)
	jmp	.L19
.L23:
	nop
.L19:
	jmp	.L20
.L22:
	call	__stack_chk_fail@PLT
.L21:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	main, .-main
	.globl	addfirst
	.type	addfirst, @function
addfirst:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movss	%xmm0, -44(%rbp)
	movq	$0, -24(%rbp)
.L30:
	cmpq	$2, -24(%rbp)
	je	.L25
	cmpq	$2, -24(%rbp)
	ja	.L32
	cmpq	$0, -24(%rbp)
	je	.L27
	cmpq	$1, -24(%rbp)
	jne	.L32
	jmp	.L31
.L27:
	movq	$2, -24(%rbp)
	jmp	.L29
.L25:
	movl	-44(%rbp), %eax
	movd	%eax, %xmm0
	call	createnode
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -8(%rbp)
	movq	-40(%rbp), %rax
	movq	(%rax), %rdx
	movq	-8(%rbp), %rax
	movq	%rdx, 8(%rax)
	movq	-40(%rbp), %rax
	movq	-8(%rbp), %rdx
	movq	%rdx, (%rax)
	movq	$1, -24(%rbp)
	jmp	.L29
.L32:
	nop
.L29:
	jmp	.L30
.L31:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	addfirst, .-addfirst
	.globl	isempty
	.type	isempty, @function
isempty:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movq	$0, -8(%rbp)
.L41:
	cmpq	$3, -8(%rbp)
	je	.L34
	cmpq	$3, -8(%rbp)
	ja	.L42
	cmpq	$0, -8(%rbp)
	je	.L36
	cmpq	$1, -8(%rbp)
	jne	.L42
	movl	$1, %eax
	jmp	.L37
.L34:
	movl	$0, %eax
	jmp	.L37
.L36:
	movq	-24(%rbp), %rax
	testq	%rax, %rax
	jne	.L38
	movq	$1, -8(%rbp)
	jmp	.L40
.L38:
	movq	$3, -8(%rbp)
	jmp	.L40
.L42:
	nop
.L40:
	jmp	.L41
.L37:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	isempty, .-isempty
	.globl	createnode
	.type	createnode, @function
createnode:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movss	%xmm0, -36(%rbp)
	movq	$0, -16(%rbp)
.L49:
	cmpq	$2, -16(%rbp)
	je	.L44
	cmpq	$2, -16(%rbp)
	ja	.L51
	cmpq	$0, -16(%rbp)
	je	.L46
	cmpq	$1, -16(%rbp)
	jne	.L51
	movl	$16, %edi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movss	-36(%rbp), %xmm0
	movss	%xmm0, (%rax)
	movq	-24(%rbp), %rax
	movq	$0, 8(%rax)
	movq	$2, -16(%rbp)
	jmp	.L47
.L46:
	movq	$1, -16(%rbp)
	jmp	.L47
.L44:
	movq	-24(%rbp), %rax
	jmp	.L50
.L51:
	nop
.L47:
	jmp	.L49
.L50:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	createnode, .-createnode
	.globl	peek
	.type	peek, @function
peek:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$24, %rsp
	movq	%rdi, -24(%rbp)
	movq	$1, -8(%rbp)
.L61:
	cmpq	$3, -8(%rbp)
	je	.L53
	cmpq	$3, -8(%rbp)
	ja	.L62
	cmpq	$2, -8(%rbp)
	je	.L55
	cmpq	$2, -8(%rbp)
	ja	.L62
	cmpq	$0, -8(%rbp)
	je	.L56
	cmpq	$1, -8(%rbp)
	jne	.L62
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	isempty
	movb	%al, -9(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L57
.L53:
	cmpb	$0, -9(%rbp)
	je	.L58
	movq	$2, -8(%rbp)
	jmp	.L57
.L58:
	movq	$0, -8(%rbp)
	jmp	.L57
.L56:
	movq	-24(%rbp), %rax
	movss	(%rax), %xmm0
	jmp	.L60
.L55:
	movss	.LC2(%rip), %xmm0
	jmp	.L60
.L62:
	nop
.L57:
	jmp	.L61
.L60:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	peek, .-peek
	.globl	postfix_evaluation
	.type	postfix_evaluation, @function
postfix_evaluation:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$80, %rsp
	movq	%rdi, -72(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$7, -24(%rbp)
.L98:
	cmpq	$27, -24(%rbp)
	ja	.L101
	movq	-24(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L66(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L66(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L66:
	.long	.L101-.L66
	.long	.L101-.L66
	.long	.L82-.L66
	.long	.L81-.L66
	.long	.L80-.L66
	.long	.L101-.L66
	.long	.L101-.L66
	.long	.L79-.L66
	.long	.L101-.L66
	.long	.L78-.L66
	.long	.L101-.L66
	.long	.L77-.L66
	.long	.L76-.L66
	.long	.L75-.L66
	.long	.L74-.L66
	.long	.L101-.L66
	.long	.L73-.L66
	.long	.L72-.L66
	.long	.L71-.L66
	.long	.L101-.L66
	.long	.L70-.L66
	.long	.L101-.L66
	.long	.L69-.L66
	.long	.L101-.L66
	.long	.L68-.L66
	.long	.L67-.L66
	.long	.L101-.L66
	.long	.L65-.L66
	.text
.L71:
	movl	-60(%rbp), %eax
	movslq	%eax, %rdx
	movq	-72(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	je	.L83
	movq	$25, -24(%rbp)
	jmp	.L85
.L83:
	movq	$9, -24(%rbp)
	jmp	.L85
.L67:
	movl	-60(%rbp), %eax
	movslq	%eax, %rdx
	movq	-72(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	cmpb	$47, %al
	jle	.L86
	movq	$17, -24(%rbp)
	jmp	.L85
.L86:
	movq	$27, -24(%rbp)
	jmp	.L85
.L80:
	pxor	%xmm0, %xmm0
	cvtss2sd	-56(%rbp), %xmm0
	pxor	%xmm2, %xmm2
	cvtss2sd	-52(%rbp), %xmm2
	movq	%xmm2, %rax
	movapd	%xmm0, %xmm1
	movq	%rax, %xmm0
	call	pow@PLT
	movq	%xmm0, %rax
	movq	%rax, -16(%rbp)
	pxor	%xmm0, %xmm0
	cvtsd2ss	-16(%rbp), %xmm0
	movss	%xmm0, -48(%rbp)
	movq	$13, -24(%rbp)
	jmp	.L85
.L74:
	movss	-40(%rbp), %xmm0
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L99
	jmp	.L100
.L76:
	movss	-52(%rbp), %xmm0
	divss	-56(%rbp), %xmm0
	movss	%xmm0, -48(%rbp)
	movq	$13, -24(%rbp)
	jmp	.L85
.L81:
	movss	-52(%rbp), %xmm0
	addss	-56(%rbp), %xmm0
	movss	%xmm0, -48(%rbp)
	movq	$13, -24(%rbp)
	jmp	.L85
.L73:
	movl	-60(%rbp), %eax
	movl	%eax, -36(%rbp)
	addl	$1, -60(%rbp)
	movl	-36(%rbp), %eax
	movslq	%eax, %rdx
	movq	-72(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	subl	$48, %eax
	pxor	%xmm3, %xmm3
	cvtsi2ssl	%eax, %xmm3
	movd	%xmm3, %edx
	leaq	-32(%rbp), %rax
	movd	%edx, %xmm0
	movq	%rax, %rdi
	call	push
	movq	$18, -24(%rbp)
	jmp	.L85
.L68:
	movl	-44(%rbp), %eax
	movslq	%eax, %rdx
	movq	-72(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	cmpl	$94, %eax
	je	.L89
	cmpl	$94, %eax
	jg	.L90
	cmpl	$47, %eax
	je	.L91
	cmpl	$47, %eax
	jg	.L90
	cmpl	$45, %eax
	je	.L92
	cmpl	$45, %eax
	jg	.L90
	cmpl	$42, %eax
	je	.L93
	cmpl	$43, %eax
	je	.L94
	jmp	.L90
.L89:
	movq	$4, -24(%rbp)
	jmp	.L95
.L91:
	movq	$12, -24(%rbp)
	jmp	.L95
.L93:
	movq	$22, -24(%rbp)
	jmp	.L95
.L92:
	movq	$2, -24(%rbp)
	jmp	.L95
.L94:
	movq	$3, -24(%rbp)
	jmp	.L95
.L90:
	movq	$20, -24(%rbp)
	nop
.L95:
	jmp	.L85
.L77:
	movl	$0, -60(%rbp)
	movq	$0, -32(%rbp)
	movq	$18, -24(%rbp)
	jmp	.L85
.L78:
	leaq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	pop
	movd	%xmm0, %eax
	movl	%eax, -40(%rbp)
	movq	$14, -24(%rbp)
	jmp	.L85
.L75:
	movl	-48(%rbp), %edx
	leaq	-32(%rbp), %rax
	movd	%edx, %xmm0
	movq	%rax, %rdi
	call	push
	movq	$18, -24(%rbp)
	jmp	.L85
.L72:
	movl	-60(%rbp), %eax
	movslq	%eax, %rdx
	movq	-72(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	cmpb	$57, %al
	jg	.L96
	movq	$16, -24(%rbp)
	jmp	.L85
.L96:
	movq	$27, -24(%rbp)
	jmp	.L85
.L65:
	leaq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	pop
	movd	%xmm0, %eax
	movl	%eax, -56(%rbp)
	leaq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	pop
	movd	%xmm0, %eax
	movl	%eax, -52(%rbp)
	movl	-60(%rbp), %eax
	movl	%eax, -44(%rbp)
	addl	$1, -60(%rbp)
	movq	$24, -24(%rbp)
	jmp	.L85
.L69:
	movss	-52(%rbp), %xmm0
	mulss	-56(%rbp), %xmm0
	movss	%xmm0, -48(%rbp)
	movq	$13, -24(%rbp)
	jmp	.L85
.L79:
	movq	$11, -24(%rbp)
	jmp	.L85
.L82:
	movss	-52(%rbp), %xmm0
	subss	-56(%rbp), %xmm0
	movss	%xmm0, -48(%rbp)
	movq	$13, -24(%rbp)
	jmp	.L85
.L70:
	movq	$13, -24(%rbp)
	jmp	.L85
.L101:
	nop
.L85:
	jmp	.L98
.L100:
	call	__stack_chk_fail@PLT
.L99:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	postfix_evaluation, .-postfix_evaluation
	.globl	pop
	.type	pop, @function
pop:
.LFB10:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$24, %rsp
	movq	%rdi, -24(%rbp)
	movq	$1, -8(%rbp)
.L114:
	cmpq	$4, -8(%rbp)
	ja	.L115
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L105(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L105(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L105:
	.long	.L109-.L105
	.long	.L108-.L105
	.long	.L107-.L105
	.long	.L106-.L105
	.long	.L104-.L105
	.text
.L104:
	movss	-12(%rbp), %xmm0
	jmp	.L110
.L108:
	movq	-24(%rbp), %rax
	movq	(%rax), %rdi
	call	isempty
	movb	%al, -13(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L111
.L106:
	movss	.LC2(%rip), %xmm0
	jmp	.L110
.L109:
	cmpb	$0, -13(%rbp)
	je	.L112
	movq	$3, -8(%rbp)
	jmp	.L111
.L112:
	movq	$2, -8(%rbp)
	jmp	.L111
.L107:
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	deletefirst
	movd	%xmm0, %eax
	movl	%eax, -12(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L111
.L115:
	nop
.L111:
	jmp	.L114
.L110:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	pop, .-pop
	.globl	push
	.type	push, @function
push:
.LFB12:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movss	%xmm0, -28(%rbp)
	movq	$0, -8(%rbp)
.L121:
	cmpq	$0, -8(%rbp)
	je	.L117
	cmpq	$1, -8(%rbp)
	jne	.L123
	jmp	.L122
.L117:
	movq	-24(%rbp), %rax
	movl	-28(%rbp), %edx
	movd	%edx, %xmm0
	movq	%rax, %rdi
	call	addfirst
	movq	$1, -8(%rbp)
	jmp	.L120
.L123:
	nop
.L120:
	jmp	.L121
.L122:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE12:
	.size	push, .-push
	.section	.rodata
	.align 4
.LC2:
	.long	-1082130432
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
