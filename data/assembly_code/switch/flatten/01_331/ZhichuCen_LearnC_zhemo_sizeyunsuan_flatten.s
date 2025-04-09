	.file	"ZhichuCen_LearnC_zhemo_sizeyunsuan_flatten.c"
	.text
	.globl	_TIG_IZ_Z0qH_argc
	.bss
	.align 4
	.type	_TIG_IZ_Z0qH_argc, @object
	.size	_TIG_IZ_Z0qH_argc, 4
_TIG_IZ_Z0qH_argc:
	.zero	4
	.globl	_TIG_IZ_Z0qH_argv
	.align 8
	.type	_TIG_IZ_Z0qH_argv, @object
	.size	_TIG_IZ_Z0qH_argv, 8
_TIG_IZ_Z0qH_argv:
	.zero	8
	.globl	_TIG_IZ_Z0qH_envp
	.align 8
	.type	_TIG_IZ_Z0qH_envp, @object
	.size	_TIG_IZ_Z0qH_envp, 8
_TIG_IZ_Z0qH_envp:
	.zero	8
	.text
	.globl	ab
	.type	ab, @function
ab:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -20(%rbp)
	movq	$1, -8(%rbp)
.L10:
	cmpq	$3, -8(%rbp)
	je	.L2
	cmpq	$3, -8(%rbp)
	ja	.L12
	cmpq	$2, -8(%rbp)
	je	.L4
	cmpq	$2, -8(%rbp)
	ja	.L12
	cmpq	$0, -8(%rbp)
	je	.L5
	cmpq	$1, -8(%rbp)
	jne	.L12
	cmpl	$0, -20(%rbp)
	jns	.L6
	movq	$3, -8(%rbp)
	jmp	.L8
.L6:
	movq	$2, -8(%rbp)
	jmp	.L8
.L2:
	movl	-20(%rbp), %eax
	negl	%eax
	movl	%eax, -12(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L8
.L5:
	movl	-12(%rbp), %eax
	jmp	.L11
.L4:
	movl	-20(%rbp), %eax
	movl	%eax, -12(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L8
.L12:
	nop
.L8:
	jmp	.L10
.L11:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	ab, .-ab
	.globl	r
	.type	r, @function
r:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movss	%xmm0, -20(%rbp)
	movq	$2, -8(%rbp)
.L23:
	cmpq	$3, -8(%rbp)
	je	.L14
	cmpq	$3, -8(%rbp)
	ja	.L27
	cmpq	$2, -8(%rbp)
	je	.L16
	cmpq	$2, -8(%rbp)
	ja	.L27
	cmpq	$0, -8(%rbp)
	je	.L17
	cmpq	$1, -8(%rbp)
	jne	.L27
	pxor	%xmm1, %xmm1
	cvtss2sd	-20(%rbp), %xmm1
	movsd	.LC0(%rip), %xmm0
	addsd	%xmm1, %xmm0
	cvttsd2sil	%xmm0, %eax
	movl	%eax, -12(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L18
.L14:
	pxor	%xmm0, %xmm0
	cvtss2sd	-20(%rbp), %xmm0
	movsd	.LC0(%rip), %xmm1
	subsd	%xmm1, %xmm0
	cvttsd2sil	%xmm0, %eax
	movl	%eax, -12(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L18
.L17:
	movl	-12(%rbp), %eax
	jmp	.L25
.L16:
	movss	-20(%rbp), %xmm0
	pxor	%xmm1, %xmm1
	comiss	%xmm1, %xmm0
	jbe	.L26
	movq	$1, -8(%rbp)
	jmp	.L18
.L26:
	movq	$3, -8(%rbp)
	jmp	.L18
.L27:
	nop
.L18:
	jmp	.L23
.L25:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	r, .-r
	.section	.rodata
.LC2:
	.string	"%d/%d %c %d/%d"
.LC3:
	.string	"%d/%d %c %d/%d = %d\n"
.LC4:
	.string	"%d/%d %c %d/%d = %d/%d\n"
.LC5:
	.string	"%d/%d %c %d/%d = 0\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB4:
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
	movq	$0, _TIG_IZ_Z0qH_envp(%rip)
	nop
.L29:
	movq	$0, _TIG_IZ_Z0qH_argv(%rip)
	nop
.L30:
	movl	$0, _TIG_IZ_Z0qH_argc(%rip)
	nop
	nop
.L31:
.L32:
#APP
# 98 "ZhichuCen_LearnC_zhemo_sizeyunsuan.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-Z0qH--0
# 0 "" 2
#NO_APP
	movl	-84(%rbp), %eax
	movl	%eax, _TIG_IZ_Z0qH_argc(%rip)
	movq	-96(%rbp), %rax
	movq	%rax, _TIG_IZ_Z0qH_argv(%rip)
	movq	-104(%rbp), %rax
	movq	%rax, _TIG_IZ_Z0qH_envp(%rip)
	nop
	movq	$17, -16(%rbp)
.L62:
	cmpq	$18, -16(%rbp)
	ja	.L65
	movq	-16(%rbp), %rax
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
	.long	.L65-.L35
	.long	.L49-.L35
	.long	.L48-.L35
	.long	.L47-.L35
	.long	.L46-.L35
	.long	.L45-.L35
	.long	.L44-.L35
	.long	.L43-.L35
	.long	.L42-.L35
	.long	.L41-.L35
	.long	.L65-.L35
	.long	.L40-.L35
	.long	.L39-.L35
	.long	.L38-.L35
	.long	.L65-.L35
	.long	.L37-.L35
	.long	.L65-.L35
	.long	.L36-.L35
	.long	.L34-.L35
	.text
.L34:
	movss	-36(%rbp), %xmm0
	divss	-32(%rbp), %xmm0
	movss	%xmm0, -40(%rbp)
	movq	$8, -16(%rbp)
	jmp	.L50
.L46:
	movss	-36(%rbp), %xmm0
	addss	-32(%rbp), %xmm0
	movss	%xmm0, -40(%rbp)
	movq	$2, -16(%rbp)
	jmp	.L50
.L37:
	movss	-36(%rbp), %xmm0
	mulss	-32(%rbp), %xmm0
	movss	%xmm0, -40(%rbp)
	movq	$2, -16(%rbp)
	jmp	.L50
.L39:
	pxor	%xmm0, %xmm0
	movss	%xmm0, -40(%rbp)
	leaq	-52(%rbp), %rdi
	leaq	-56(%rbp), %rsi
	leaq	-65(%rbp), %rcx
	leaq	-60(%rbp), %rdx
	leaq	-64(%rbp), %rax
	movq	%rdi, %r9
	movq	%rsi, %r8
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	scanf@PLT
	movl	-64(%rbp), %eax
	pxor	%xmm0, %xmm0
	cvtsi2ssl	%eax, %xmm0
	movl	-60(%rbp), %eax
	pxor	%xmm1, %xmm1
	cvtsi2ssl	%eax, %xmm1
	divss	%xmm1, %xmm0
	movss	%xmm0, -36(%rbp)
	movl	-56(%rbp), %eax
	pxor	%xmm0, %xmm0
	cvtsi2ssl	%eax, %xmm0
	movl	-52(%rbp), %eax
	pxor	%xmm1, %xmm1
	cvtsi2ssl	%eax, %xmm1
	divss	%xmm1, %xmm0
	movss	%xmm0, -32(%rbp)
	movq	$11, -16(%rbp)
	jmp	.L50
.L42:
	movq	$2, -16(%rbp)
	jmp	.L50
.L49:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L63
	jmp	.L64
.L47:
	movl	-52(%rbp), %r8d
	movl	-56(%rbp), %edi
	movzbl	-65(%rbp), %eax
	movsbl	%al, %ecx
	movl	-60(%rbp), %edx
	movl	-64(%rbp), %eax
	subq	$8, %rsp
	movl	-48(%rbp), %esi
	pushq	%rsi
	movl	%r8d, %r9d
	movl	%edi, %r8d
	movl	%eax, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addq	$16, %rsp
	movq	$1, -16(%rbp)
	jmp	.L50
.L40:
	movzbl	-65(%rbp), %eax
	movsbl	%al, %eax
	cmpl	$47, %eax
	je	.L52
	cmpl	$47, %eax
	jg	.L53
	cmpl	$45, %eax
	je	.L54
	cmpl	$45, %eax
	jg	.L53
	cmpl	$42, %eax
	je	.L55
	cmpl	$43, %eax
	je	.L56
	jmp	.L53
.L52:
	movq	$18, -16(%rbp)
	jmp	.L57
.L55:
	movq	$15, -16(%rbp)
	jmp	.L57
.L54:
	movq	$5, -16(%rbp)
	jmp	.L57
.L56:
	movq	$4, -16(%rbp)
	jmp	.L57
.L53:
	movq	$8, -16(%rbp)
	nop
.L57:
	jmp	.L50
.L41:
	cmpl	$1, -44(%rbp)
	jne	.L58
	movq	$3, -16(%rbp)
	jmp	.L50
.L58:
	movq	$6, -16(%rbp)
	jmp	.L50
.L38:
	cmpl	$0, -48(%rbp)
	jne	.L60
	movq	$7, -16(%rbp)
	jmp	.L50
.L60:
	movq	$9, -16(%rbp)
	jmp	.L50
.L36:
	movq	$12, -16(%rbp)
	jmp	.L50
.L44:
	movl	-52(%rbp), %r8d
	movl	-56(%rbp), %edi
	movzbl	-65(%rbp), %eax
	movsbl	%al, %ecx
	movl	-60(%rbp), %edx
	movl	-64(%rbp), %eax
	movl	-44(%rbp), %esi
	pushq	%rsi
	movl	-48(%rbp), %esi
	pushq	%rsi
	movl	%r8d, %r9d
	movl	%edi, %r8d
	movl	%eax, %esi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addq	$16, %rsp
	movq	$1, -16(%rbp)
	jmp	.L50
.L45:
	movss	-36(%rbp), %xmm0
	subss	-32(%rbp), %xmm0
	movss	%xmm0, -40(%rbp)
	movq	$2, -16(%rbp)
	jmp	.L50
.L43:
	movl	-52(%rbp), %edi
	movl	-56(%rbp), %esi
	movzbl	-65(%rbp), %eax
	movsbl	%al, %ecx
	movl	-60(%rbp), %edx
	movl	-64(%rbp), %eax
	movl	%edi, %r9d
	movl	%esi, %r8d
	movl	%eax, %esi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -16(%rbp)
	jmp	.L50
.L48:
	movl	-60(%rbp), %eax
	pxor	%xmm0, %xmm0
	cvtsi2ssl	%eax, %xmm0
	movaps	%xmm0, %xmm1
	mulss	-40(%rbp), %xmm1
	movl	-52(%rbp), %eax
	pxor	%xmm0, %xmm0
	cvtsi2ssl	%eax, %xmm0
	mulss	%xmm0, %xmm1
	movd	%xmm1, %eax
	movd	%eax, %xmm0
	call	r
	movl	%eax, -28(%rbp)
	movl	-60(%rbp), %edx
	movl	-52(%rbp), %eax
	imull	%eax, %edx
	movl	-28(%rbp), %eax
	movl	%edx, %esi
	movl	%eax, %edi
	call	g
	movl	%eax, -24(%rbp)
	movl	-24(%rbp), %eax
	movl	%eax, %edi
	call	ab
	movl	%eax, -20(%rbp)
	movl	-60(%rbp), %eax
	pxor	%xmm0, %xmm0
	cvtsi2ssl	%eax, %xmm0
	movaps	%xmm0, %xmm1
	mulss	-40(%rbp), %xmm1
	movl	-52(%rbp), %eax
	pxor	%xmm0, %xmm0
	cvtsi2ssl	%eax, %xmm0
	mulss	%xmm1, %xmm0
	pxor	%xmm1, %xmm1
	cvtsi2ssl	-20(%rbp), %xmm1
	divss	%xmm1, %xmm0
	movd	%xmm0, %eax
	movd	%eax, %xmm0
	call	r
	movl	%eax, -48(%rbp)
	movl	-60(%rbp), %edx
	movl	-52(%rbp), %eax
	imull	%edx, %eax
	cltd
	idivl	-20(%rbp)
	movl	%eax, -44(%rbp)
	movq	$13, -16(%rbp)
	jmp	.L50
.L65:
	nop
.L50:
	jmp	.L62
.L64:
	call	__stack_chk_fail@PLT
.L63:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	main, .-main
	.globl	g
	.type	g, @function
g:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movl	%edi, -36(%rbp)
	movl	%esi, -40(%rbp)
	movq	$3, -8(%rbp)
.L82:
	cmpq	$7, -8(%rbp)
	ja	.L84
	movq	-8(%rbp), %rax
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
	.long	.L75-.L69
	.long	.L74-.L69
	.long	.L84-.L69
	.long	.L73-.L69
	.long	.L72-.L69
	.long	.L71-.L69
	.long	.L70-.L69
	.long	.L68-.L69
	.text
.L72:
	movl	-20(%rbp), %eax
	movl	%eax, -16(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L76
.L74:
	movl	$1, -20(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L76
.L73:
	cmpl	$0, -40(%rbp)
	je	.L77
	movq	$5, -8(%rbp)
	jmp	.L76
.L77:
	movq	$6, -8(%rbp)
	jmp	.L76
.L70:
	cmpl	$0, -36(%rbp)
	je	.L79
	movq	$7, -8(%rbp)
	jmp	.L76
.L79:
	movq	$1, -8(%rbp)
	jmp	.L76
.L71:
	movl	-36(%rbp), %eax
	cltd
	idivl	-40(%rbp)
	movl	-40(%rbp), %eax
	movl	%edx, %esi
	movl	%eax, %edi
	call	g
	movl	%eax, -12(%rbp)
	movl	-12(%rbp), %eax
	movl	%eax, -16(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L76
.L75:
	movl	-16(%rbp), %eax
	jmp	.L83
.L68:
	movl	-36(%rbp), %eax
	movl	%eax, -20(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L76
.L84:
	nop
.L76:
	jmp	.L82
.L83:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	g, .-g
	.globl	l
	.type	l, @function
l:
.LFB9:
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
	movq	$1, -8(%rbp)
.L90:
	cmpq	$0, -8(%rbp)
	je	.L86
	cmpq	$1, -8(%rbp)
	jne	.L92
	movl	-24(%rbp), %edx
	movl	-20(%rbp), %eax
	movl	%edx, %esi
	movl	%eax, %edi
	call	g
	movl	%eax, -12(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L88
.L86:
	movl	-20(%rbp), %eax
	imull	-24(%rbp), %eax
	cltd
	idivl	-12(%rbp)
	jmp	.L91
.L92:
	nop
.L88:
	jmp	.L90
.L91:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	l, .-l
	.section	.rodata
	.align 8
.LC0:
	.long	0
	.long	1071644672
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
