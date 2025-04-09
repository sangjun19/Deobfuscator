	.file	"Dhruvanarayana_DSA-LAB_4_flatten.c"
	.text
	.globl	_TIG_IZ_AI0d_argc
	.bss
	.align 4
	.type	_TIG_IZ_AI0d_argc, @object
	.size	_TIG_IZ_AI0d_argc, 4
_TIG_IZ_AI0d_argc:
	.zero	4
	.globl	_TIG_IZ_AI0d_envp
	.align 8
	.type	_TIG_IZ_AI0d_envp, @object
	.size	_TIG_IZ_AI0d_envp, 8
_TIG_IZ_AI0d_envp:
	.zero	8
	.globl	_TIG_IZ_AI0d_argv
	.align 8
	.type	_TIG_IZ_AI0d_argv, @object
	.size	_TIG_IZ_AI0d_argv, 8
_TIG_IZ_AI0d_argv:
	.zero	8
	.section	.rodata
	.align 8
.LC0:
	.string	"Enter the valid infix expression"
.LC1:
	.string	"%s"
.LC2:
	.string	"\nThe postfix expression is:"
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
	subq	$112, %rsp
	movl	%edi, -84(%rbp)
	movq	%rsi, -96(%rbp)
	movq	%rdx, -104(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_AI0d_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_AI0d_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_AI0d_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 122 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-AI0d--0
# 0 "" 2
#NO_APP
	movl	-84(%rbp), %eax
	movl	%eax, _TIG_IZ_AI0d_argc(%rip)
	movq	-96(%rbp), %rax
	movq	%rax, _TIG_IZ_AI0d_argv(%rip)
	movq	-104(%rbp), %rax
	movq	%rax, _TIG_IZ_AI0d_envp(%rip)
	nop
	movq	$1, -72(%rbp)
.L11:
	cmpq	$2, -72(%rbp)
	je	.L14
	cmpq	$2, -72(%rbp)
	ja	.L15
	cmpq	$0, -72(%rbp)
	je	.L8
	cmpq	$1, -72(%rbp)
	jne	.L15
	movq	$0, -72(%rbp)
	jmp	.L9
.L8:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-64(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	-32(%rbp), %rdx
	leaq	-64(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	infixpostfix
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-32(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$2, -72(%rbp)
	jmp	.L9
.L15:
	nop
.L9:
	jmp	.L11
.L14:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L13
	call	__stack_chk_fail@PLT
.L13:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	main, .-main
	.globl	inpre
	.type	inpre, @function
inpre:
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
	movq	$5, -8(%rbp)
.L36:
	cmpq	$7, -8(%rbp)
	ja	.L37
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L19(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L19(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L19:
	.long	.L25-.L19
	.long	.L37-.L19
	.long	.L24-.L19
	.long	.L23-.L19
	.long	.L22-.L19
	.long	.L21-.L19
	.long	.L20-.L19
	.long	.L18-.L19
	.text
.L22:
	movl	$6, %eax
	jmp	.L26
.L23:
	movl	$0, %eax
	jmp	.L26
.L20:
	movl	$1, %eax
	jmp	.L26
.L21:
	movsbl	-20(%rbp), %eax
	subl	$36, %eax
	cmpl	$58, %eax
	ja	.L27
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L29(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L29(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L29:
	.long	.L28-.L29
	.long	.L30-.L29
	.long	.L27-.L29
	.long	.L27-.L29
	.long	.L33-.L29
	.long	.L32-.L29
	.long	.L30-.L29
	.long	.L31-.L29
	.long	.L27-.L29
	.long	.L31-.L29
	.long	.L27-.L29
	.long	.L30-.L29
	.long	.L27-.L29
	.long	.L27-.L29
	.long	.L27-.L29
	.long	.L27-.L29
	.long	.L27-.L29
	.long	.L27-.L29
	.long	.L27-.L29
	.long	.L27-.L29
	.long	.L27-.L29
	.long	.L27-.L29
	.long	.L27-.L29
	.long	.L27-.L29
	.long	.L27-.L29
	.long	.L27-.L29
	.long	.L27-.L29
	.long	.L27-.L29
	.long	.L27-.L29
	.long	.L27-.L29
	.long	.L27-.L29
	.long	.L27-.L29
	.long	.L27-.L29
	.long	.L27-.L29
	.long	.L27-.L29
	.long	.L27-.L29
	.long	.L27-.L29
	.long	.L27-.L29
	.long	.L27-.L29
	.long	.L27-.L29
	.long	.L27-.L29
	.long	.L27-.L29
	.long	.L27-.L29
	.long	.L27-.L29
	.long	.L27-.L29
	.long	.L27-.L29
	.long	.L27-.L29
	.long	.L27-.L29
	.long	.L27-.L29
	.long	.L27-.L29
	.long	.L27-.L29
	.long	.L27-.L29
	.long	.L27-.L29
	.long	.L27-.L29
	.long	.L27-.L29
	.long	.L27-.L29
	.long	.L27-.L29
	.long	.L27-.L29
	.long	.L28-.L29
	.text
.L32:
	movq	$3, -8(%rbp)
	jmp	.L34
.L33:
	movq	$0, -8(%rbp)
	jmp	.L34
.L28:
	movq	$4, -8(%rbp)
	jmp	.L34
.L30:
	movq	$2, -8(%rbp)
	jmp	.L34
.L31:
	movq	$6, -8(%rbp)
	jmp	.L34
.L27:
	movq	$7, -8(%rbp)
	nop
.L34:
	jmp	.L35
.L25:
	movl	$3, %eax
	jmp	.L26
.L18:
	movl	$7, %eax
	jmp	.L26
.L24:
	movl	$3, %eax
	jmp	.L26
.L37:
	nop
.L35:
	jmp	.L36
.L26:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	inpre, .-inpre
	.globl	pre
	.type	pre, @function
pre:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, %eax
	movb	%al, -20(%rbp)
	movq	$4, -8(%rbp)
.L58:
	cmpq	$7, -8(%rbp)
	ja	.L59
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L41(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L41(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L41:
	.long	.L47-.L41
	.long	.L46-.L41
	.long	.L59-.L41
	.long	.L45-.L41
	.long	.L44-.L41
	.long	.L43-.L41
	.long	.L42-.L41
	.long	.L40-.L41
	.text
.L44:
	movsbl	-20(%rbp), %eax
	subl	$35, %eax
	cmpl	$59, %eax
	ja	.L48
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L50(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L50(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L50:
	.long	.L54-.L50
	.long	.L49-.L50
	.long	.L51-.L50
	.long	.L48-.L50
	.long	.L48-.L50
	.long	.L53-.L50
	.long	.L48-.L50
	.long	.L51-.L50
	.long	.L52-.L50
	.long	.L48-.L50
	.long	.L52-.L50
	.long	.L48-.L50
	.long	.L51-.L50
	.long	.L48-.L50
	.long	.L48-.L50
	.long	.L48-.L50
	.long	.L48-.L50
	.long	.L48-.L50
	.long	.L48-.L50
	.long	.L48-.L50
	.long	.L48-.L50
	.long	.L48-.L50
	.long	.L48-.L50
	.long	.L48-.L50
	.long	.L48-.L50
	.long	.L48-.L50
	.long	.L48-.L50
	.long	.L48-.L50
	.long	.L48-.L50
	.long	.L48-.L50
	.long	.L48-.L50
	.long	.L48-.L50
	.long	.L48-.L50
	.long	.L48-.L50
	.long	.L48-.L50
	.long	.L48-.L50
	.long	.L48-.L50
	.long	.L48-.L50
	.long	.L48-.L50
	.long	.L48-.L50
	.long	.L48-.L50
	.long	.L48-.L50
	.long	.L48-.L50
	.long	.L48-.L50
	.long	.L48-.L50
	.long	.L48-.L50
	.long	.L48-.L50
	.long	.L48-.L50
	.long	.L48-.L50
	.long	.L48-.L50
	.long	.L48-.L50
	.long	.L48-.L50
	.long	.L48-.L50
	.long	.L48-.L50
	.long	.L48-.L50
	.long	.L48-.L50
	.long	.L48-.L50
	.long	.L48-.L50
	.long	.L48-.L50
	.long	.L49-.L50
	.text
.L54:
	movq	$1, -8(%rbp)
	jmp	.L55
.L53:
	movq	$5, -8(%rbp)
	jmp	.L55
.L49:
	movq	$0, -8(%rbp)
	jmp	.L55
.L51:
	movq	$3, -8(%rbp)
	jmp	.L55
.L52:
	movq	$7, -8(%rbp)
	jmp	.L55
.L48:
	movq	$6, -8(%rbp)
	nop
.L55:
	jmp	.L56
.L46:
	movl	$-1, %eax
	jmp	.L57
.L45:
	movl	$4, %eax
	jmp	.L57
.L42:
	movl	$8, %eax
	jmp	.L57
.L43:
	movl	$0, %eax
	jmp	.L57
.L47:
	movl	$5, %eax
	jmp	.L57
.L40:
	movl	$2, %eax
	jmp	.L57
.L59:
	nop
.L56:
	jmp	.L58
.L57:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	pre, .-pre
	.globl	infixpostfix
	.type	infixpostfix, @function
infixpostfix:
.LFB8:
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
	movq	$21, -56(%rbp)
.L90:
	cmpq	$27, -56(%rbp)
	ja	.L93
	movq	-56(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L63(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L63(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L63:
	.long	.L79-.L63
	.long	.L78-.L63
	.long	.L77-.L63
	.long	.L76-.L63
	.long	.L93-.L63
	.long	.L93-.L63
	.long	.L75-.L63
	.long	.L93-.L63
	.long	.L93-.L63
	.long	.L93-.L63
	.long	.L93-.L63
	.long	.L74-.L63
	.long	.L93-.L63
	.long	.L73-.L63
	.long	.L72-.L63
	.long	.L71-.L63
	.long	.L70-.L63
	.long	.L69-.L63
	.long	.L93-.L63
	.long	.L93-.L63
	.long	.L68-.L63
	.long	.L67-.L63
	.long	.L93-.L63
	.long	.L66-.L63
	.long	.L94-.L63
	.long	.L64-.L63
	.long	.L93-.L63
	.long	.L62-.L63
	.text
.L64:
	movl	$-1, -104(%rbp)
	movl	$0, -100(%rbp)
	addl	$1, -104(%rbp)
	movl	-104(%rbp), %eax
	cltq
	movb	$35, -48(%rbp,%rax)
	movl	$0, -96(%rbp)
	movq	$14, -56(%rbp)
	jmp	.L80
.L72:
	movq	-120(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -64(%rbp)
	movq	$3, -56(%rbp)
	jmp	.L80
.L71:
	movl	-104(%rbp), %eax
	cltq
	movzbl	-48(%rbp,%rax), %eax
	movsbl	%al, %eax
	movl	%eax, %edi
	call	pre
	movl	%eax, -92(%rbp)
	movsbl	-105(%rbp), %eax
	movl	%eax, %edi
	call	inpre
	movl	%eax, -88(%rbp)
	movq	$16, -56(%rbp)
	jmp	.L80
.L78:
	movl	-84(%rbp), %eax
	cmpl	-80(%rbp), %eax
	je	.L81
	movq	$0, -56(%rbp)
	jmp	.L80
.L81:
	movq	$6, -56(%rbp)
	jmp	.L80
.L66:
	movl	-104(%rbp), %eax
	cltq
	movzbl	-48(%rbp,%rax), %eax
	movsbl	%al, %eax
	movl	%eax, %edi
	call	pre
	movl	%eax, -84(%rbp)
	movsbl	-105(%rbp), %eax
	movl	%eax, %edi
	call	inpre
	movl	%eax, -80(%rbp)
	movq	$1, -56(%rbp)
	jmp	.L80
.L76:
	movl	-96(%rbp), %eax
	cltq
	cmpq	%rax, -64(%rbp)
	jbe	.L83
	movq	$13, -56(%rbp)
	jmp	.L80
.L83:
	movq	$17, -56(%rbp)
	jmp	.L80
.L70:
	movl	-92(%rbp), %eax
	cmpl	-88(%rbp), %eax
	jle	.L85
	movq	$11, -56(%rbp)
	jmp	.L80
.L85:
	movq	$23, -56(%rbp)
	jmp	.L80
.L67:
	movq	$25, -56(%rbp)
	jmp	.L80
.L74:
	movl	-104(%rbp), %eax
	movl	%eax, -68(%rbp)
	subl	$1, -104(%rbp)
	movl	-100(%rbp), %eax
	movslq	%eax, %rdx
	movq	-128(%rbp), %rax
	addq	%rax, %rdx
	movl	-68(%rbp), %eax
	cltq
	movzbl	-48(%rbp,%rax), %eax
	movb	%al, (%rdx)
	addl	$1, -100(%rbp)
	movq	$15, -56(%rbp)
	jmp	.L80
.L73:
	movl	-96(%rbp), %eax
	movslq	%eax, %rdx
	movq	-120(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	movb	%al, -105(%rbp)
	movq	$15, -56(%rbp)
	jmp	.L80
.L69:
	movl	-104(%rbp), %eax
	cltq
	movzbl	-48(%rbp,%rax), %eax
	cmpb	$35, %al
	je	.L88
	movq	$20, -56(%rbp)
	jmp	.L80
.L88:
	movq	$27, -56(%rbp)
	jmp	.L80
.L75:
	subl	$1, -104(%rbp)
	movq	$2, -56(%rbp)
	jmp	.L80
.L62:
	movl	-100(%rbp), %eax
	movslq	%eax, %rdx
	movq	-128(%rbp), %rax
	addq	%rdx, %rax
	movb	$0, (%rax)
	movq	$24, -56(%rbp)
	jmp	.L80
.L79:
	addl	$1, -104(%rbp)
	movl	-104(%rbp), %eax
	cltq
	movzbl	-105(%rbp), %edx
	movb	%dl, -48(%rbp,%rax)
	movq	$2, -56(%rbp)
	jmp	.L80
.L77:
	addl	$1, -96(%rbp)
	movq	$14, -56(%rbp)
	jmp	.L80
.L68:
	movl	-100(%rbp), %eax
	movl	%eax, -76(%rbp)
	addl	$1, -100(%rbp)
	movl	-104(%rbp), %eax
	movl	%eax, -72(%rbp)
	subl	$1, -104(%rbp)
	movl	-76(%rbp), %eax
	movslq	%eax, %rdx
	movq	-128(%rbp), %rax
	addq	%rax, %rdx
	movl	-72(%rbp), %eax
	cltq
	movzbl	-48(%rbp,%rax), %eax
	movb	%al, (%rdx)
	movq	$17, -56(%rbp)
	jmp	.L80
.L93:
	nop
.L80:
	jmp	.L90
.L94:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L92
	call	__stack_chk_fail@PLT
.L92:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	infixpostfix, .-infixpostfix
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
