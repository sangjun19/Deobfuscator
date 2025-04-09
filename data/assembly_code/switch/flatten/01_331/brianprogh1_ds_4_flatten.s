	.file	"brianprogh1_ds_4_flatten.c"
	.text
	.globl	_TIG_IZ_Aq55_argc
	.bss
	.align 4
	.type	_TIG_IZ_Aq55_argc, @object
	.size	_TIG_IZ_Aq55_argc, 4
_TIG_IZ_Aq55_argc:
	.zero	4
	.globl	_TIG_IZ_Aq55_argv
	.align 8
	.type	_TIG_IZ_Aq55_argv, @object
	.size	_TIG_IZ_Aq55_argv, 8
_TIG_IZ_Aq55_argv:
	.zero	8
	.globl	_TIG_IZ_Aq55_envp
	.align 8
	.type	_TIG_IZ_Aq55_envp, @object
	.size	_TIG_IZ_Aq55_envp, 8
_TIG_IZ_Aq55_envp:
	.zero	8
	.section	.rodata
	.align 8
.LC0:
	.string	"Enter the valid infix expression:"
.LC1:
	.string	"%s"
	.align 8
.LC2:
	.string	"The given infix expression is %s and the equivalent postfix expression is %s"
	.text
	.globl	main
	.type	main, @function
main:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$176, %rsp
	movl	%edi, -148(%rbp)
	movq	%rsi, -160(%rbp)
	movq	%rdx, -168(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_Aq55_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_Aq55_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_Aq55_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 122 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-Aq55--0
# 0 "" 2
#NO_APP
	movl	-148(%rbp), %eax
	movl	%eax, _TIG_IZ_Aq55_argc(%rip)
	movq	-160(%rbp), %rax
	movq	%rax, _TIG_IZ_Aq55_argv(%rip)
	movq	-168(%rbp), %rax
	movq	%rax, _TIG_IZ_Aq55_envp(%rip)
	nop
	movq	$0, -136(%rbp)
.L11:
	cmpq	$2, -136(%rbp)
	je	.L6
	cmpq	$2, -136(%rbp)
	ja	.L14
	cmpq	$0, -136(%rbp)
	je	.L8
	cmpq	$1, -136(%rbp)
	jne	.L14
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-128(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	-64(%rbp), %rdx
	leaq	-128(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	infixtopostfix
	leaq	-64(%rbp), %rdx
	leaq	-128(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$2, -136(%rbp)
	jmp	.L9
.L8:
	movq	$1, -136(%rbp)
	jmp	.L9
.L6:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L12
	jmp	.L13
.L14:
	nop
.L9:
	jmp	.L11
.L13:
	call	__stack_chk_fail@PLT
.L12:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	main, .-main
	.globl	g
	.type	g, @function
g:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, %eax
	movb	%al, -20(%rbp)
	movq	$2, -8(%rbp)
.L35:
	cmpq	$7, -8(%rbp)
	ja	.L36
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L18(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L18(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L18:
	.long	.L24-.L18
	.long	.L23-.L18
	.long	.L22-.L18
	.long	.L36-.L18
	.long	.L21-.L18
	.long	.L20-.L18
	.long	.L19-.L18
	.long	.L17-.L18
	.text
.L21:
	movl	$1, %eax
	jmp	.L25
.L23:
	movl	$0, %eax
	jmp	.L25
.L19:
	movl	$3, %eax
	jmp	.L25
.L20:
	movl	$9, %eax
	jmp	.L25
.L24:
	movl	$6, %eax
	jmp	.L25
.L17:
	movl	$7, %eax
	jmp	.L25
.L22:
	movsbl	-20(%rbp), %eax
	subl	$36, %eax
	cmpl	$58, %eax
	ja	.L26
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L28(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L28(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L28:
	.long	.L27-.L28
	.long	.L29-.L28
	.long	.L26-.L28
	.long	.L26-.L28
	.long	.L32-.L28
	.long	.L31-.L28
	.long	.L29-.L28
	.long	.L30-.L28
	.long	.L26-.L28
	.long	.L30-.L28
	.long	.L26-.L28
	.long	.L29-.L28
	.long	.L26-.L28
	.long	.L26-.L28
	.long	.L26-.L28
	.long	.L26-.L28
	.long	.L26-.L28
	.long	.L26-.L28
	.long	.L26-.L28
	.long	.L26-.L28
	.long	.L26-.L28
	.long	.L26-.L28
	.long	.L26-.L28
	.long	.L26-.L28
	.long	.L26-.L28
	.long	.L26-.L28
	.long	.L26-.L28
	.long	.L26-.L28
	.long	.L26-.L28
	.long	.L26-.L28
	.long	.L26-.L28
	.long	.L26-.L28
	.long	.L26-.L28
	.long	.L26-.L28
	.long	.L26-.L28
	.long	.L26-.L28
	.long	.L26-.L28
	.long	.L26-.L28
	.long	.L26-.L28
	.long	.L26-.L28
	.long	.L26-.L28
	.long	.L26-.L28
	.long	.L26-.L28
	.long	.L26-.L28
	.long	.L26-.L28
	.long	.L26-.L28
	.long	.L26-.L28
	.long	.L26-.L28
	.long	.L26-.L28
	.long	.L26-.L28
	.long	.L26-.L28
	.long	.L26-.L28
	.long	.L26-.L28
	.long	.L26-.L28
	.long	.L26-.L28
	.long	.L26-.L28
	.long	.L26-.L28
	.long	.L26-.L28
	.long	.L27-.L28
	.text
.L31:
	movq	$1, -8(%rbp)
	jmp	.L33
.L32:
	movq	$5, -8(%rbp)
	jmp	.L33
.L27:
	movq	$0, -8(%rbp)
	jmp	.L33
.L29:
	movq	$6, -8(%rbp)
	jmp	.L33
.L30:
	movq	$4, -8(%rbp)
	jmp	.L33
.L26:
	movq	$7, -8(%rbp)
	nop
.L33:
	jmp	.L34
.L36:
	nop
.L34:
	jmp	.L35
.L25:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	g, .-g
	.globl	infixtopostfix
	.type	infixtopostfix, @function
infixtopostfix:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	addq	$-128, %rsp
	movq	%rdi, -120(%rbp)
	movq	%rsi, -128(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$13, -40(%rbp)
.L65:
	cmpq	$24, -40(%rbp)
	ja	.L68
	movq	-40(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L40(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L40(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L40:
	.long	.L54-.L40
	.long	.L68-.L40
	.long	.L68-.L40
	.long	.L53-.L40
	.long	.L68-.L40
	.long	.L52-.L40
	.long	.L51-.L40
	.long	.L68-.L40
	.long	.L68-.L40
	.long	.L50-.L40
	.long	.L49-.L40
	.long	.L68-.L40
	.long	.L68-.L40
	.long	.L48-.L40
	.long	.L68-.L40
	.long	.L47-.L40
	.long	.L68-.L40
	.long	.L46-.L40
	.long	.L45-.L40
	.long	.L68-.L40
	.long	.L44-.L40
	.long	.L43-.L40
	.long	.L42-.L40
	.long	.L69-.L40
	.long	.L39-.L40
	.text
.L45:
	movl	-96(%rbp), %eax
	movslq	%eax, %rdx
	movq	-120(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	je	.L55
	movq	$6, -40(%rbp)
	jmp	.L57
.L55:
	movq	$20, -40(%rbp)
	jmp	.L57
.L47:
	movl	-92(%rbp), %eax
	movl	%eax, -68(%rbp)
	addl	$1, -92(%rbp)
	movl	-88(%rbp), %eax
	movl	%eax, -64(%rbp)
	subl	$1, -88(%rbp)
	movl	-68(%rbp), %eax
	movslq	%eax, %rdx
	movq	-128(%rbp), %rax
	addq	%rax, %rdx
	movl	-64(%rbp), %eax
	cltq
	movzbl	-32(%rbp,%rax), %eax
	movb	%al, (%rdx)
	movq	$20, -40(%rbp)
	jmp	.L57
.L53:
	movl	-92(%rbp), %eax
	movl	%eax, -52(%rbp)
	addl	$1, -92(%rbp)
	movl	-88(%rbp), %eax
	movl	%eax, -48(%rbp)
	subl	$1, -88(%rbp)
	movl	-52(%rbp), %eax
	movslq	%eax, %rdx
	movq	-128(%rbp), %rax
	addq	%rax, %rdx
	movl	-48(%rbp), %eax
	cltq
	movzbl	-32(%rbp,%rax), %eax
	movb	%al, (%rdx)
	movq	$9, -40(%rbp)
	jmp	.L57
.L39:
	movl	$0, -96(%rbp)
	movl	$0, -92(%rbp)
	movl	$-1, -88(%rbp)
	addl	$1, -88(%rbp)
	movl	-88(%rbp), %eax
	cltq
	movb	$35, -32(%rbp,%rax)
	movq	$18, -40(%rbp)
	jmp	.L57
.L43:
	movl	-76(%rbp), %eax
	cmpl	-72(%rbp), %eax
	je	.L59
	movq	$22, -40(%rbp)
	jmp	.L57
.L59:
	movq	$0, -40(%rbp)
	jmp	.L57
.L50:
	movl	-88(%rbp), %eax
	cltq
	movzbl	-32(%rbp,%rax), %eax
	movsbl	%al, %eax
	movl	%eax, %edi
	call	f
	movl	%eax, -84(%rbp)
	movsbl	-97(%rbp), %eax
	movl	%eax, %edi
	call	g
	movl	%eax, -80(%rbp)
	movq	$17, -40(%rbp)
	jmp	.L57
.L48:
	movq	$24, -40(%rbp)
	jmp	.L57
.L46:
	movl	-84(%rbp), %eax
	cmpl	-80(%rbp), %eax
	jle	.L61
	movq	$3, -40(%rbp)
	jmp	.L57
.L61:
	movq	$10, -40(%rbp)
	jmp	.L57
.L51:
	movl	-96(%rbp), %eax
	movl	%eax, -60(%rbp)
	addl	$1, -96(%rbp)
	movl	-60(%rbp), %eax
	movslq	%eax, %rdx
	movq	-120(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	movb	%al, -97(%rbp)
	movq	$9, -40(%rbp)
	jmp	.L57
.L42:
	addl	$1, -88(%rbp)
	movl	-88(%rbp), %eax
	cltq
	movzbl	-97(%rbp), %edx
	movb	%dl, -32(%rbp,%rax)
	movq	$18, -40(%rbp)
	jmp	.L57
.L52:
	movl	-92(%rbp), %eax
	movl	%eax, -56(%rbp)
	addl	$1, -92(%rbp)
	movl	-56(%rbp), %eax
	movslq	%eax, %rdx
	movq	-128(%rbp), %rax
	addq	%rdx, %rax
	movb	$0, (%rax)
	movq	$23, -40(%rbp)
	jmp	.L57
.L49:
	movl	-88(%rbp), %eax
	cltq
	movzbl	-32(%rbp,%rax), %eax
	movsbl	%al, %eax
	movl	%eax, %edi
	call	f
	movl	%eax, -76(%rbp)
	movsbl	-97(%rbp), %eax
	movl	%eax, %edi
	call	g
	movl	%eax, -72(%rbp)
	movq	$21, -40(%rbp)
	jmp	.L57
.L54:
	movl	-88(%rbp), %eax
	movl	%eax, -44(%rbp)
	subl	$1, -88(%rbp)
	movq	$18, -40(%rbp)
	jmp	.L57
.L44:
	movl	-88(%rbp), %eax
	cltq
	movzbl	-32(%rbp,%rax), %eax
	cmpb	$35, %al
	je	.L63
	movq	$15, -40(%rbp)
	jmp	.L57
.L63:
	movq	$5, -40(%rbp)
	jmp	.L57
.L68:
	nop
.L57:
	jmp	.L65
.L69:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L67
	call	__stack_chk_fail@PLT
.L67:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	infixtopostfix, .-infixtopostfix
	.globl	f
	.type	f, @function
f:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, %eax
	movb	%al, -20(%rbp)
	movq	$3, -8(%rbp)
.L90:
	cmpq	$7, -8(%rbp)
	ja	.L91
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L73(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L73(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L73:
	.long	.L79-.L73
	.long	.L78-.L73
	.long	.L77-.L73
	.long	.L76-.L73
	.long	.L75-.L73
	.long	.L74-.L73
	.long	.L91-.L73
	.long	.L72-.L73
	.text
.L75:
	movl	$-1, %eax
	jmp	.L80
.L78:
	movl	$4, %eax
	jmp	.L80
.L76:
	movsbl	-20(%rbp), %eax
	subl	$35, %eax
	cmpl	$59, %eax
	ja	.L81
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L83(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L83(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L83:
	.long	.L87-.L83
	.long	.L82-.L83
	.long	.L84-.L83
	.long	.L81-.L83
	.long	.L81-.L83
	.long	.L86-.L83
	.long	.L81-.L83
	.long	.L84-.L83
	.long	.L85-.L83
	.long	.L81-.L83
	.long	.L85-.L83
	.long	.L81-.L83
	.long	.L84-.L83
	.long	.L81-.L83
	.long	.L81-.L83
	.long	.L81-.L83
	.long	.L81-.L83
	.long	.L81-.L83
	.long	.L81-.L83
	.long	.L81-.L83
	.long	.L81-.L83
	.long	.L81-.L83
	.long	.L81-.L83
	.long	.L81-.L83
	.long	.L81-.L83
	.long	.L81-.L83
	.long	.L81-.L83
	.long	.L81-.L83
	.long	.L81-.L83
	.long	.L81-.L83
	.long	.L81-.L83
	.long	.L81-.L83
	.long	.L81-.L83
	.long	.L81-.L83
	.long	.L81-.L83
	.long	.L81-.L83
	.long	.L81-.L83
	.long	.L81-.L83
	.long	.L81-.L83
	.long	.L81-.L83
	.long	.L81-.L83
	.long	.L81-.L83
	.long	.L81-.L83
	.long	.L81-.L83
	.long	.L81-.L83
	.long	.L81-.L83
	.long	.L81-.L83
	.long	.L81-.L83
	.long	.L81-.L83
	.long	.L81-.L83
	.long	.L81-.L83
	.long	.L81-.L83
	.long	.L81-.L83
	.long	.L81-.L83
	.long	.L81-.L83
	.long	.L81-.L83
	.long	.L81-.L83
	.long	.L81-.L83
	.long	.L81-.L83
	.long	.L82-.L83
	.text
.L87:
	movq	$4, -8(%rbp)
	jmp	.L88
.L86:
	movq	$7, -8(%rbp)
	jmp	.L88
.L82:
	movq	$2, -8(%rbp)
	jmp	.L88
.L84:
	movq	$1, -8(%rbp)
	jmp	.L88
.L85:
	movq	$0, -8(%rbp)
	jmp	.L88
.L81:
	movq	$5, -8(%rbp)
	nop
.L88:
	jmp	.L89
.L74:
	movl	$8, %eax
	jmp	.L80
.L79:
	movl	$2, %eax
	jmp	.L80
.L72:
	movl	$0, %eax
	jmp	.L80
.L77:
	movl	$5, %eax
	jmp	.L80
.L91:
	nop
.L89:
	jmp	.L90
.L80:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	f, .-f
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
