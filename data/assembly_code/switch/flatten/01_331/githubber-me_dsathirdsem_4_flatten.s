	.file	"githubber-me_dsathirdsem_4_flatten.c"
	.text
	.globl	_TIG_IZ_ogvP_argv
	.bss
	.align 8
	.type	_TIG_IZ_ogvP_argv, @object
	.size	_TIG_IZ_ogvP_argv, 8
_TIG_IZ_ogvP_argv:
	.zero	8
	.globl	_TIG_IZ_ogvP_envp
	.align 8
	.type	_TIG_IZ_ogvP_envp, @object
	.size	_TIG_IZ_ogvP_envp, 8
_TIG_IZ_ogvP_envp:
	.zero	8
	.globl	_TIG_IZ_ogvP_argc
	.align 4
	.type	_TIG_IZ_ogvP_argc, @object
	.size	_TIG_IZ_ogvP_argc, 4
_TIG_IZ_ogvP_argc:
	.zero	4
	.section	.rodata
	.align 8
.LC0:
	.string	"Enter a valid infix expression"
.LC1:
	.string	"%s"
.LC2:
	.string	"Postfix expression is %s"
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
	addq	$-128, %rsp
	movl	%edi, -100(%rbp)
	movq	%rsi, -112(%rbp)
	movq	%rdx, -120(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_ogvP_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_ogvP_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_ogvP_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 125 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-ogvP--0
# 0 "" 2
#NO_APP
	movl	-100(%rbp), %eax
	movl	%eax, _TIG_IZ_ogvP_argc(%rip)
	movq	-112(%rbp), %rax
	movq	%rax, _TIG_IZ_ogvP_argv(%rip)
	movq	-120(%rbp), %rax
	movq	%rax, _TIG_IZ_ogvP_envp(%rip)
	nop
	movq	$2, -88(%rbp)
.L11:
	cmpq	$2, -88(%rbp)
	je	.L6
	cmpq	$2, -88(%rbp)
	ja	.L15
	cmpq	$0, -88(%rbp)
	je	.L8
	cmpq	$1, -88(%rbp)
	jne	.L15
	jmp	.L14
.L8:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-80(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	-48(%rbp), %rdx
	leaq	-80(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	convert
	leaq	-48(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -88(%rbp)
	jmp	.L10
.L6:
	movq	$0, -88(%rbp)
	jmp	.L10
.L15:
	nop
.L10:
	jmp	.L11
.L14:
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
	.globl	in_prec
	.type	in_prec, @function
in_prec:
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
	movl	$7, %eax
	jmp	.L26
.L23:
	movl	$1, %eax
	jmp	.L26
.L20:
	movl	$9, %eax
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
	.long	.L27-.L29
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
	movq	$0, -8(%rbp)
	jmp	.L34
.L33:
	movq	$6, -8(%rbp)
	jmp	.L34
.L28:
	movq	$2, -8(%rbp)
	jmp	.L34
.L30:
	movq	$7, -8(%rbp)
	jmp	.L34
.L31:
	movq	$3, -8(%rbp)
	jmp	.L34
.L27:
	movq	$4, -8(%rbp)
	nop
.L34:
	jmp	.L35
.L25:
	movl	$0, %eax
	jmp	.L26
.L18:
	movl	$3, %eax
	jmp	.L26
.L24:
	movl	$6, %eax
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
	.size	in_prec, .-in_prec
	.globl	stack_prec
	.type	stack_prec, @function
stack_prec:
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
	movq	$6, -8(%rbp)
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
	.long	.L45-.L41
	.long	.L44-.L41
	.long	.L59-.L41
	.long	.L43-.L41
	.long	.L42-.L41
	.long	.L40-.L41
	.text
.L46:
	movl	$2, %eax
	jmp	.L48
.L44:
	movl	$-1, %eax
	jmp	.L48
.L42:
	movsbl	-20(%rbp), %eax
	subl	$35, %eax
	cmpl	$59, %eax
	ja	.L49
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L51(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L51(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L51:
	.long	.L55-.L51
	.long	.L50-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L54-.L51
	.long	.L49-.L51
	.long	.L52-.L51
	.long	.L53-.L51
	.long	.L49-.L51
	.long	.L53-.L51
	.long	.L49-.L51
	.long	.L52-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L49-.L51
	.long	.L50-.L51
	.text
.L55:
	movq	$3, -8(%rbp)
	jmp	.L56
.L54:
	movq	$0, -8(%rbp)
	jmp	.L56
.L50:
	movq	$5, -8(%rbp)
	jmp	.L56
.L52:
	movq	$2, -8(%rbp)
	jmp	.L56
.L53:
	movq	$1, -8(%rbp)
	jmp	.L56
.L49:
	movq	$7, -8(%rbp)
	nop
.L56:
	jmp	.L57
.L43:
	movl	$5, %eax
	jmp	.L48
.L47:
	movl	$0, %eax
	jmp	.L48
.L40:
	movl	$8, %eax
	jmp	.L48
.L45:
	movl	$4, %eax
	jmp	.L48
.L59:
	nop
.L57:
	jmp	.L58
.L48:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	stack_prec, .-stack_prec
	.globl	convert
	.type	convert, @function
convert:
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
	movq	$8, -56(%rbp)
.L89:
	cmpq	$25, -56(%rbp)
	ja	.L92
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
	.long	.L92-.L63
	.long	.L78-.L63
	.long	.L92-.L63
	.long	.L92-.L63
	.long	.L77-.L63
	.long	.L92-.L63
	.long	.L76-.L63
	.long	.L92-.L63
	.long	.L75-.L63
	.long	.L93-.L63
	.long	.L92-.L63
	.long	.L73-.L63
	.long	.L72-.L63
	.long	.L71-.L63
	.long	.L70-.L63
	.long	.L69-.L63
	.long	.L92-.L63
	.long	.L68-.L63
	.long	.L67-.L63
	.long	.L66-.L63
	.long	.L92-.L63
	.long	.L65-.L63
	.long	.L92-.L63
	.long	.L92-.L63
	.long	.L64-.L63
	.long	.L62-.L63
	.text
.L67:
	movl	-92(%rbp), %eax
	movl	%eax, -72(%rbp)
	addl	$1, -92(%rbp)
	movl	-100(%rbp), %eax
	movl	%eax, -68(%rbp)
	subl	$1, -100(%rbp)
	movl	-72(%rbp), %eax
	movslq	%eax, %rdx
	movq	-128(%rbp), %rax
	addq	%rax, %rdx
	movl	-68(%rbp), %eax
	cltq
	movzbl	-48(%rbp,%rax), %eax
	movb	%al, (%rdx)
	movq	$6, -56(%rbp)
	jmp	.L79
.L62:
	movl	-96(%rbp), %eax
	movslq	%eax, %rdx
	movq	-120(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	je	.L80
	movq	$15, -56(%rbp)
	jmp	.L79
.L80:
	movq	$6, -56(%rbp)
	jmp	.L79
.L77:
	movl	-100(%rbp), %eax
	cltq
	movzbl	-48(%rbp,%rax), %eax
	movsbl	%al, %eax
	movl	%eax, %edi
	call	stack_prec
	movl	%eax, -80(%rbp)
	movsbl	-101(%rbp), %eax
	movl	%eax, %edi
	call	in_prec
	movl	%eax, -76(%rbp)
	movq	$12, -56(%rbp)
	jmp	.L79
.L70:
	movl	-100(%rbp), %eax
	cltq
	movzbl	-48(%rbp,%rax), %eax
	movsbl	%al, %eax
	movl	%eax, %edi
	call	stack_prec
	movl	%eax, -88(%rbp)
	movsbl	-101(%rbp), %eax
	movl	%eax, %edi
	call	in_prec
	movl	%eax, -84(%rbp)
	movq	$13, -56(%rbp)
	jmp	.L79
.L69:
	movl	-96(%rbp), %eax
	movslq	%eax, %rdx
	movq	-120(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	movb	%al, -101(%rbp)
	movq	$14, -56(%rbp)
	jmp	.L79
.L72:
	movl	-80(%rbp), %eax
	cmpl	-76(%rbp), %eax
	je	.L82
	movq	$24, -56(%rbp)
	jmp	.L79
.L82:
	movq	$1, -56(%rbp)
	jmp	.L79
.L75:
	movq	$21, -56(%rbp)
	jmp	.L79
.L78:
	subl	$1, -100(%rbp)
	movq	$11, -56(%rbp)
	jmp	.L79
.L64:
	addl	$1, -100(%rbp)
	movl	-100(%rbp), %eax
	cltq
	movzbl	-101(%rbp), %edx
	movb	%dl, -48(%rbp,%rax)
	movq	$11, -56(%rbp)
	jmp	.L79
.L65:
	movl	$-1, -100(%rbp)
	movl	$0, -92(%rbp)
	addl	$1, -100(%rbp)
	movl	-100(%rbp), %eax
	cltq
	movb	$35, -48(%rbp,%rax)
	movl	$0, -96(%rbp)
	movq	$25, -56(%rbp)
	jmp	.L79
.L73:
	addl	$1, -96(%rbp)
	movq	$25, -56(%rbp)
	jmp	.L79
.L71:
	movl	-88(%rbp), %eax
	cmpl	-84(%rbp), %eax
	jle	.L85
	movq	$17, -56(%rbp)
	jmp	.L79
.L85:
	movq	$4, -56(%rbp)
	jmp	.L79
.L66:
	movl	-92(%rbp), %eax
	movslq	%eax, %rdx
	movq	-128(%rbp), %rax
	addq	%rdx, %rax
	movb	$0, (%rax)
	movq	$9, -56(%rbp)
	jmp	.L79
.L68:
	movl	-92(%rbp), %eax
	movl	%eax, -64(%rbp)
	addl	$1, -92(%rbp)
	movl	-100(%rbp), %eax
	movl	%eax, -60(%rbp)
	subl	$1, -100(%rbp)
	movl	-64(%rbp), %eax
	movslq	%eax, %rdx
	movq	-128(%rbp), %rax
	addq	%rax, %rdx
	movl	-60(%rbp), %eax
	cltq
	movzbl	-48(%rbp,%rax), %eax
	movb	%al, (%rdx)
	movq	$14, -56(%rbp)
	jmp	.L79
.L76:
	movl	-100(%rbp), %eax
	cltq
	movzbl	-48(%rbp,%rax), %eax
	cmpb	$35, %al
	je	.L87
	movq	$18, -56(%rbp)
	jmp	.L79
.L87:
	movq	$19, -56(%rbp)
	jmp	.L79
.L92:
	nop
.L79:
	jmp	.L89
.L93:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L91
	call	__stack_chk_fail@PLT
.L91:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	convert, .-convert
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
