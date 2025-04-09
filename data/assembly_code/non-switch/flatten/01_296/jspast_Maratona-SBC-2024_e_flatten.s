	.file	"jspast_Maratona-SBC-2024_e_flatten.c"
	.text
	.globl	_TIG_IZ_5S1Y_envp
	.bss
	.align 8
	.type	_TIG_IZ_5S1Y_envp, @object
	.size	_TIG_IZ_5S1Y_envp, 8
_TIG_IZ_5S1Y_envp:
	.zero	8
	.globl	_TIG_IZ_5S1Y_argv
	.align 8
	.type	_TIG_IZ_5S1Y_argv, @object
	.size	_TIG_IZ_5S1Y_argv, 8
_TIG_IZ_5S1Y_argv:
	.zero	8
	.globl	_TIG_IZ_5S1Y_argc
	.align 4
	.type	_TIG_IZ_5S1Y_argc, @object
	.size	_TIG_IZ_5S1Y_argc, 4
_TIG_IZ_5S1Y_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"%d"
	.text
	.globl	main
	.type	main, @function
main:
.LFB6:
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
	movq	$0, _TIG_IZ_5S1Y_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_5S1Y_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_5S1Y_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 135 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-5S1Y--0
# 0 "" 2
#NO_APP
	movl	-84(%rbp), %eax
	movl	%eax, _TIG_IZ_5S1Y_argc(%rip)
	movq	-96(%rbp), %rax
	movq	%rax, _TIG_IZ_5S1Y_argv(%rip)
	movq	-104(%rbp), %rax
	movq	%rax, _TIG_IZ_5S1Y_envp(%rip)
	nop
	movq	$40, -24(%rbp)
.L90:
	cmpq	$59, -24(%rbp)
	ja	.L93
	movq	-24(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L8(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L8(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L8:
	.long	.L46-.L8
	.long	.L45-.L8
	.long	.L44-.L8
	.long	.L43-.L8
	.long	.L93-.L8
	.long	.L93-.L8
	.long	.L42-.L8
	.long	.L93-.L8
	.long	.L93-.L8
	.long	.L93-.L8
	.long	.L93-.L8
	.long	.L41-.L8
	.long	.L40-.L8
	.long	.L39-.L8
	.long	.L93-.L8
	.long	.L38-.L8
	.long	.L93-.L8
	.long	.L37-.L8
	.long	.L36-.L8
	.long	.L35-.L8
	.long	.L34-.L8
	.long	.L33-.L8
	.long	.L32-.L8
	.long	.L93-.L8
	.long	.L31-.L8
	.long	.L30-.L8
	.long	.L29-.L8
	.long	.L28-.L8
	.long	.L93-.L8
	.long	.L27-.L8
	.long	.L93-.L8
	.long	.L93-.L8
	.long	.L26-.L8
	.long	.L93-.L8
	.long	.L25-.L8
	.long	.L24-.L8
	.long	.L93-.L8
	.long	.L93-.L8
	.long	.L23-.L8
	.long	.L22-.L8
	.long	.L21-.L8
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L93-.L8
	.long	.L93-.L8
	.long	.L93-.L8
	.long	.L17-.L8
	.long	.L93-.L8
	.long	.L93-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L93-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L36:
	movl	-72(%rbp), %eax
	cltq
	movl	-72(%rbp), %edx
	movslq	%edx, %rdx
	salq	$5, %rdx
	addq	$31, %rdx
	andq	$-32, %rdx
	imulq	%rdx, %rax
	addq	$31, %rax
	shrq	$3, %rax
	movq	%rax, %rdx
	movabsq	$2305843009213693948, %rax
	andq	%rdx, %rax
	movq	%rax, -16(%rbp)
	movl	-72(%rbp), %eax
	cltq
	salq	$5, %rax
	addq	$31, %rax
	shrq	$3, %rax
	movq	%rax, %rdx
	movabsq	$2305843009213693948, %rax
	andq	%rdx, %rax
	movq	%rax, -32(%rbp)
	movq	-16(%rbp), %rax
	leaq	8(%rax), %rdx
	movl	$16, %eax
	subq	$1, %rax
	addq	%rdx, %rax
	movl	$16, %esi
	movl	$0, %edx
	divq	%rsi
	imulq	$16, %rax, %rax
	movq	%rax, %rcx
	andq	$-4096, %rcx
	movq	%rsp, %rdx
	subq	%rcx, %rdx
.L47:
	cmpq	%rdx, %rsp
	je	.L48
	subq	$4096, %rsp
	orq	$0, 4088(%rsp)
	jmp	.L47
.L48:
	movq	%rax, %rdx
	andl	$4095, %edx
	subq	%rdx, %rsp
	movq	%rax, %rdx
	andl	$4095, %edx
	testq	%rdx, %rdx
	je	.L49
	andl	$4095, %eax
	subq	$8, %rax
	addq	%rsp, %rax
	orq	$0, (%rax)
.L49:
	movq	%rsp, %rax
	addq	$15, %rax
	shrq	$4, %rax
	salq	$4, %rax
	movq	%rax, -40(%rbp)
	movq	$53, -24(%rbp)
	jmp	.L50
.L16:
	movl	-56(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$55, -24(%rbp)
	jmp	.L50
.L30:
	cmpl	$0, -52(%rbp)
	je	.L51
	movq	$32, -24(%rbp)
	jmp	.L50
.L51:
	movq	$51, -24(%rbp)
	jmp	.L50
.L14:
	movl	$0, -52(%rbp)
	movq	$47, -24(%rbp)
	jmp	.L50
.L38:
	leaq	-72(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$18, -24(%rbp)
	jmp	.L50
.L40:
	cmpl	$1, -56(%rbp)
	jne	.L53
	movq	$59, -24(%rbp)
	jmp	.L50
.L53:
	movq	$22, -24(%rbp)
	jmp	.L50
.L12:
	movl	-44(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movl	-48(%rbp), %eax
	cltq
	imulq	-32(%rbp), %rax
	addq	%rax, %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	movl	-44(%rbp), %eax
	addl	$1, %eax
	cltq
	leaq	0(,%rax,4), %rcx
	movl	-48(%rbp), %eax
	cltq
	imulq	-32(%rbp), %rax
	addq	%rax, %rcx
	movq	-40(%rbp), %rax
	addq	%rcx, %rax
	movl	(%rax), %eax
	cmpl	%eax, %edx
	jle	.L55
	movq	$52, -24(%rbp)
	jmp	.L50
.L55:
	movq	$12, -24(%rbp)
	jmp	.L50
.L45:
	movl	-72(%rbp), %eax
	cmpl	%eax, -68(%rbp)
	jge	.L57
	movq	$24, -24(%rbp)
	jmp	.L50
.L57:
	movq	$34, -24(%rbp)
	jmp	.L50
.L43:
	movl	$1, -60(%rbp)
	movq	$41, -24(%rbp)
	jmp	.L50
.L31:
	movl	$0, -64(%rbp)
	movq	$11, -24(%rbp)
	jmp	.L50
.L33:
	cmpl	$0, -60(%rbp)
	jne	.L59
	movq	$6, -24(%rbp)
	jmp	.L50
.L59:
	movq	$50, -24(%rbp)
	jmp	.L50
.L10:
	cmpl	$0, -56(%rbp)
	jne	.L61
	movq	$54, -24(%rbp)
	jmp	.L50
.L61:
	movq	$19, -24(%rbp)
	jmp	.L50
.L29:
	movl	$0, -52(%rbp)
	movq	$25, -24(%rbp)
	jmp	.L50
.L41:
	movl	-72(%rbp), %eax
	cmpl	%eax, -64(%rbp)
	jge	.L63
	movq	$58, -24(%rbp)
	jmp	.L50
.L63:
	movq	$35, -24(%rbp)
	jmp	.L50
.L39:
	movl	-48(%rbp), %eax
	cltq
	imulq	-32(%rbp), %rax
	movq	%rax, %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	movl	-48(%rbp), %eax
	addl	$1, %eax
	cltq
	imulq	-32(%rbp), %rax
	movq	%rax, %rcx
	movq	-40(%rbp), %rax
	addq	%rcx, %rax
	movl	(%rax), %eax
	cmpl	%eax, %edx
	jge	.L65
	movq	$26, -24(%rbp)
	jmp	.L50
.L65:
	movq	$25, -24(%rbp)
	jmp	.L50
.L15:
	addl	$1, -48(%rbp)
	movq	$2, -24(%rbp)
	jmp	.L50
.L35:
	cmpl	$3, -56(%rbp)
	jne	.L67
	movq	$54, -24(%rbp)
	jmp	.L50
.L67:
	movq	$12, -24(%rbp)
	jmp	.L50
.L26:
	movl	-44(%rbp), %eax
	leal	1(%rax), %edx
	movl	-72(%rbp), %eax
	cmpl	%eax, %edx
	jge	.L69
	movq	$57, -24(%rbp)
	jmp	.L50
.L69:
	movq	$51, -24(%rbp)
	jmp	.L50
.L37:
	movl	-48(%rbp), %eax
	cltq
	imulq	-32(%rbp), %rax
	movq	%rax, %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	movl	-48(%rbp), %eax
	addl	$1, %eax
	cltq
	imulq	-32(%rbp), %rax
	movq	%rax, %rcx
	movq	-40(%rbp), %rax
	addq	%rcx, %rax
	movl	(%rax), %eax
	cmpl	%eax, %edx
	jle	.L71
	movq	$39, -24(%rbp)
	jmp	.L50
.L71:
	movq	$20, -24(%rbp)
	jmp	.L50
.L21:
	movq	$15, -24(%rbp)
	jmp	.L50
.L11:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L91
	jmp	.L92
.L7:
	movl	-44(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movl	-48(%rbp), %eax
	cltq
	imulq	-32(%rbp), %rax
	addq	%rax, %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	movl	-44(%rbp), %eax
	addl	$1, %eax
	cltq
	leaq	0(,%rax,4), %rcx
	movl	-48(%rbp), %eax
	cltq
	imulq	-32(%rbp), %rax
	addq	%rax, %rcx
	movq	-40(%rbp), %rax
	addq	%rcx, %rax
	movl	(%rax), %eax
	cmpl	%eax, %edx
	jge	.L74
	movq	$0, -24(%rbp)
	jmp	.L50
.L74:
	movq	$47, -24(%rbp)
	jmp	.L50
.L42:
	movl	$1, -52(%rbp)
	movl	$0, -48(%rbp)
	movq	$2, -24(%rbp)
	jmp	.L50
.L28:
	movl	-48(%rbp), %eax
	leal	1(%rax), %edx
	movl	-72(%rbp), %eax
	cmpl	%eax, %edx
	jge	.L76
	movq	$38, -24(%rbp)
	jmp	.L50
.L76:
	movq	$29, -24(%rbp)
	jmp	.L50
.L23:
	movl	$0, -44(%rbp)
	movq	$25, -24(%rbp)
	jmp	.L50
.L9:
	movl	-64(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movl	-68(%rbp), %eax
	cltq
	imulq	-32(%rbp), %rax
	addq	%rax, %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	addl	$1, -64(%rbp)
	movq	$11, -24(%rbp)
	jmp	.L50
.L25:
	movl	$0, -60(%rbp)
	movl	$0, -56(%rbp)
	movq	$41, -24(%rbp)
	jmp	.L50
.L32:
	cmpl	$2, -56(%rbp)
	jne	.L78
	movq	$59, -24(%rbp)
	jmp	.L50
.L78:
	movq	$47, -24(%rbp)
	jmp	.L50
.L13:
	movl	$0, -68(%rbp)
	movq	$1, -24(%rbp)
	jmp	.L50
.L17:
	addl	$1, -44(%rbp)
	movq	$42, -24(%rbp)
	jmp	.L50
.L20:
	cmpl	$3, -56(%rbp)
	jg	.L80
	movq	$21, -24(%rbp)
	jmp	.L50
.L80:
	movq	$50, -24(%rbp)
	jmp	.L50
.L19:
	cmpl	$1, -56(%rbp)
	jg	.L82
	movq	$17, -24(%rbp)
	jmp	.L50
.L82:
	movq	$20, -24(%rbp)
	jmp	.L50
.L46:
	movl	$0, -52(%rbp)
	movq	$47, -24(%rbp)
	jmp	.L50
.L22:
	movl	$0, -52(%rbp)
	movq	$25, -24(%rbp)
	jmp	.L50
.L24:
	addl	$1, -68(%rbp)
	movq	$1, -24(%rbp)
	jmp	.L50
.L27:
	cmpl	$0, -52(%rbp)
	je	.L84
	movq	$3, -24(%rbp)
	jmp	.L50
.L84:
	movq	$43, -24(%rbp)
	jmp	.L50
.L18:
	addl	$1, -56(%rbp)
	movq	$41, -24(%rbp)
	jmp	.L50
.L44:
	cmpl	$0, -52(%rbp)
	je	.L86
	movq	$27, -24(%rbp)
	jmp	.L50
.L86:
	movq	$29, -24(%rbp)
	jmp	.L50
.L34:
	cmpl	$1, -56(%rbp)
	jle	.L88
	movq	$13, -24(%rbp)
	jmp	.L50
.L88:
	movq	$25, -24(%rbp)
	jmp	.L50
.L93:
	nop
.L50:
	jmp	.L90
.L92:
	call	__stack_chk_fail@PLT
.L91:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
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
