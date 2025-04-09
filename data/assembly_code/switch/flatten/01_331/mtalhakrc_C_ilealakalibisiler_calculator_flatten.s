	.file	"mtalhakrc_C_ilealakalibisiler_calculator_flatten.c"
	.text
	.globl	bufp
	.bss
	.align 4
	.type	bufp, @object
	.size	bufp, 4
bufp:
	.zero	4
	.globl	val
	.align 32
	.type	val, @object
	.size	val, 800
val:
	.zero	800
	.globl	_TIG_IZ_Cikp_argc
	.align 4
	.type	_TIG_IZ_Cikp_argc, @object
	.size	_TIG_IZ_Cikp_argc, 4
_TIG_IZ_Cikp_argc:
	.zero	4
	.globl	buf
	.align 32
	.type	buf, @object
	.size	buf, 100
buf:
	.zero	100
	.globl	_TIG_IZ_Cikp_argv
	.align 8
	.type	_TIG_IZ_Cikp_argv, @object
	.size	_TIG_IZ_Cikp_argv, 8
_TIG_IZ_Cikp_argv:
	.zero	8
	.globl	sp
	.align 4
	.type	sp, @object
	.size	sp, 4
sp:
	.zero	4
	.globl	_TIG_IZ_Cikp_envp
	.align 8
	.type	_TIG_IZ_Cikp_envp, @object
	.size	_TIG_IZ_Cikp_envp, 8
_TIG_IZ_Cikp_envp:
	.zero	8
	.section	.rodata
	.align 8
.LC0:
	.string	"Error: stack full, can't push %g.\n"
	.text
	.globl	push
	.type	push, @function
push:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movsd	%xmm0, -24(%rbp)
	movq	$1, -8(%rbp)
.L11:
	cmpq	$4, -8(%rbp)
	je	.L2
	cmpq	$4, -8(%rbp)
	ja	.L12
	cmpq	$2, -8(%rbp)
	je	.L13
	cmpq	$2, -8(%rbp)
	ja	.L12
	cmpq	$0, -8(%rbp)
	je	.L5
	cmpq	$1, -8(%rbp)
	je	.L6
	jmp	.L12
.L2:
	movq	-24(%rbp), %rax
	movq	%rax, %xmm0
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$2, -8(%rbp)
	jmp	.L7
.L6:
	movl	sp(%rip), %eax
	cmpl	$99, %eax
	jg	.L8
	movq	$0, -8(%rbp)
	jmp	.L7
.L8:
	movq	$4, -8(%rbp)
	jmp	.L7
.L5:
	movl	sp(%rip), %eax
	movl	%eax, -12(%rbp)
	movl	sp(%rip), %eax
	addl	$1, %eax
	movl	%eax, sp(%rip)
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	leaq	val(%rip), %rax
	movsd	-24(%rbp), %xmm0
	movsd	%xmm0, (%rdx,%rax)
	movq	$2, -8(%rbp)
	jmp	.L7
.L12:
	nop
.L7:
	jmp	.L11
.L13:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	push, .-push
	.section	.rodata
.LC1:
	.string	"ungetch: too many characters"
	.text
	.globl	ungetch
	.type	ungetch, @function
ungetch:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, -20(%rbp)
	movq	$3, -8(%rbp)
.L23:
	cmpq	$3, -8(%rbp)
	je	.L15
	cmpq	$3, -8(%rbp)
	ja	.L24
	cmpq	$2, -8(%rbp)
	je	.L17
	cmpq	$2, -8(%rbp)
	ja	.L24
	cmpq	$0, -8(%rbp)
	je	.L25
	cmpq	$1, -8(%rbp)
	jne	.L24
	movl	bufp(%rip), %eax
	movl	%eax, -12(%rbp)
	movl	bufp(%rip), %eax
	addl	$1, %eax
	movl	%eax, bufp(%rip)
	movl	-20(%rbp), %eax
	movl	%eax, %ecx
	movl	-12(%rbp), %eax
	cltq
	leaq	buf(%rip), %rdx
	movb	%cl, (%rax,%rdx)
	movq	$0, -8(%rbp)
	jmp	.L19
.L15:
	movl	bufp(%rip), %eax
	cmpl	$99, %eax
	jle	.L20
	movq	$2, -8(%rbp)
	jmp	.L19
.L20:
	movq	$1, -8(%rbp)
	jmp	.L19
.L17:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$0, -8(%rbp)
	jmp	.L19
.L24:
	nop
.L19:
	jmp	.L23
.L25:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	ungetch, .-ungetch
	.globl	getop
	.type	getop, @function
getop:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$96, %rsp
	movq	%rdi, -88(%rbp)
	movq	$27, -8(%rbp)
.L83:
	cmpq	$37, -8(%rbp)
	ja	.L84
	movq	-8(%rbp), %rax
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
	.long	.L54-.L29
	.long	.L53-.L29
	.long	.L84-.L29
	.long	.L52-.L29
	.long	.L51-.L29
	.long	.L50-.L29
	.long	.L84-.L29
	.long	.L49-.L29
	.long	.L84-.L29
	.long	.L48-.L29
	.long	.L84-.L29
	.long	.L47-.L29
	.long	.L46-.L29
	.long	.L45-.L29
	.long	.L84-.L29
	.long	.L84-.L29
	.long	.L84-.L29
	.long	.L44-.L29
	.long	.L43-.L29
	.long	.L42-.L29
	.long	.L84-.L29
	.long	.L41-.L29
	.long	.L40-.L29
	.long	.L39-.L29
	.long	.L38-.L29
	.long	.L84-.L29
	.long	.L37-.L29
	.long	.L36-.L29
	.long	.L84-.L29
	.long	.L35-.L29
	.long	.L34-.L29
	.long	.L33-.L29
	.long	.L32-.L29
	.long	.L84-.L29
	.long	.L31-.L29
	.long	.L84-.L29
	.long	.L30-.L29
	.long	.L28-.L29
	.text
.L43:
	cmpl	$45, -60(%rbp)
	jne	.L55
	movq	$9, -8(%rbp)
	jmp	.L57
.L55:
	movq	$23, -8(%rbp)
	jmp	.L57
.L51:
	cmpl	$9, -60(%rbp)
	jne	.L58
	movq	$32, -8(%rbp)
	jmp	.L57
.L58:
	movq	$37, -8(%rbp)
	jmp	.L57
.L34:
	cmpl	$-1, -60(%rbp)
	je	.L60
	movq	$3, -8(%rbp)
	jmp	.L57
.L60:
	movq	$21, -8(%rbp)
	jmp	.L57
.L33:
	movl	-56(%rbp), %eax
	jmp	.L62
.L46:
	cmpl	$46, -56(%rbp)
	je	.L63
	movq	$31, -8(%rbp)
	jmp	.L57
.L63:
	movq	$1, -8(%rbp)
	jmp	.L57
.L53:
	movl	-64(%rbp), %eax
	movslq	%eax, %rdx
	movq	-88(%rbp), %rax
	addq	%rdx, %rax
	movl	-60(%rbp), %edx
	movb	%dl, (%rax)
	movl	-56(%rbp), %eax
	movl	%eax, -60(%rbp)
	movl	-60(%rbp), %eax
	movl	%eax, %edi
	call	ungetch
	movq	$29, -8(%rbp)
	jmp	.L57
.L39:
	call	getch
	movl	%eax, -60(%rbp)
	movq	$29, -8(%rbp)
	jmp	.L57
.L52:
	movl	-60(%rbp), %eax
	movl	%eax, %edi
	call	ungetch
	movq	$21, -8(%rbp)
	jmp	.L57
.L38:
	cmpb	$32, -67(%rbp)
	jne	.L65
	movq	$32, -8(%rbp)
	jmp	.L57
.L65:
	movq	$4, -8(%rbp)
	jmp	.L57
.L41:
	movl	$48, %eax
	jmp	.L62
.L30:
	movq	-16(%rbp), %rax
	movq	(%rax), %rdx
	movsbq	-65(%rbp), %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$2048, %eax
	testl	%eax, %eax
	je	.L67
	movq	$17, -8(%rbp)
	jmp	.L57
.L67:
	movq	$30, -8(%rbp)
	jmp	.L57
.L37:
	cmpl	$46, -60(%rbp)
	je	.L69
	movq	$22, -8(%rbp)
	jmp	.L57
.L69:
	movq	$18, -8(%rbp)
	jmp	.L57
.L47:
	movq	-32(%rbp), %rax
	movq	(%rax), %rdx
	movsbq	-66(%rbp), %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$2048, %eax
	testl	%eax, %eax
	je	.L71
	movq	$19, -8(%rbp)
	jmp	.L57
.L71:
	movq	$13, -8(%rbp)
	jmp	.L57
.L48:
	call	getch
	movl	%eax, -52(%rbp)
	movl	-52(%rbp), %eax
	movl	%eax, -56(%rbp)
	call	__ctype_b_loc@PLT
	movq	%rax, -40(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L57
.L45:
	cmpl	$46, -60(%rbp)
	jne	.L73
	movq	$17, -8(%rbp)
	jmp	.L57
.L73:
	movq	$30, -8(%rbp)
	jmp	.L57
.L42:
	call	__ctype_b_loc@PLT
	movq	%rax, -32(%rbp)
	addl	$1, -64(%rbp)
	call	getch
	movl	%eax, -60(%rbp)
	movl	-60(%rbp), %eax
	movb	%al, -66(%rbp)
	movl	-64(%rbp), %eax
	movslq	%eax, %rdx
	movq	-88(%rbp), %rax
	addq	%rax, %rdx
	movzbl	-66(%rbp), %eax
	movb	%al, (%rdx)
	movq	$11, -8(%rbp)
	jmp	.L57
.L32:
	call	getch
	movl	%eax, -60(%rbp)
	movl	-60(%rbp), %eax
	movb	%al, -67(%rbp)
	movq	-88(%rbp), %rax
	movzbl	-67(%rbp), %edx
	movb	%dl, (%rax)
	movq	$24, -8(%rbp)
	jmp	.L57
.L44:
	call	__ctype_b_loc@PLT
	movq	%rax, -16(%rbp)
	addl	$1, -64(%rbp)
	call	getch
	movl	%eax, -60(%rbp)
	movl	-60(%rbp), %eax
	movb	%al, -65(%rbp)
	movl	-64(%rbp), %eax
	movslq	%eax, %rdx
	movq	-88(%rbp), %rax
	addq	%rax, %rdx
	movzbl	-65(%rbp), %eax
	movb	%al, (%rdx)
	movq	$36, -8(%rbp)
	jmp	.L57
.L36:
	movl	$0, -64(%rbp)
	movq	$32, -8(%rbp)
	jmp	.L57
.L31:
	movq	-24(%rbp), %rax
	movq	(%rax), %rdx
	movl	-60(%rbp), %eax
	cltq
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$2048, %eax
	testl	%eax, %eax
	je	.L75
	movq	$19, -8(%rbp)
	jmp	.L57
.L75:
	movq	$13, -8(%rbp)
	jmp	.L57
.L40:
	cmpl	$45, -60(%rbp)
	je	.L77
	movq	$7, -8(%rbp)
	jmp	.L57
.L77:
	movq	$18, -8(%rbp)
	jmp	.L57
.L50:
	movq	-40(%rbp), %rax
	movq	(%rax), %rdx
	movl	-56(%rbp), %eax
	cltq
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$2048, %eax
	testl	%eax, %eax
	je	.L79
	movq	$1, -8(%rbp)
	jmp	.L57
.L79:
	movq	$12, -8(%rbp)
	jmp	.L57
.L28:
	movq	-88(%rbp), %rax
	addq	$1, %rax
	movb	$0, (%rax)
	call	__ctype_b_loc@PLT
	movq	%rax, -48(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L57
.L54:
	movq	-48(%rbp), %rax
	movq	(%rax), %rdx
	movl	-60(%rbp), %eax
	cltq
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$2048, %eax
	testl	%eax, %eax
	je	.L81
	movq	$18, -8(%rbp)
	jmp	.L57
.L81:
	movq	$26, -8(%rbp)
	jmp	.L57
.L49:
	movl	-60(%rbp), %eax
	jmp	.L62
.L35:
	call	__ctype_b_loc@PLT
	movq	%rax, -24(%rbp)
	movq	$34, -8(%rbp)
	jmp	.L57
.L84:
	nop
.L57:
	jmp	.L83
.L62:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	getop, .-getop
	.globl	getch
	.type	getch, @function
getch:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$3, -8(%rbp)
.L95:
	cmpq	$5, -8(%rbp)
	je	.L86
	cmpq	$5, -8(%rbp)
	ja	.L97
	cmpq	$4, -8(%rbp)
	je	.L88
	cmpq	$4, -8(%rbp)
	ja	.L97
	cmpq	$2, -8(%rbp)
	je	.L89
	cmpq	$3, -8(%rbp)
	je	.L90
	jmp	.L97
.L88:
	call	getchar@PLT
	movl	%eax, -12(%rbp)
	movl	-12(%rbp), %eax
	movl	%eax, -16(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L91
.L90:
	movl	bufp(%rip), %eax
	testl	%eax, %eax
	jle	.L92
	movq	$2, -8(%rbp)
	jmp	.L91
.L92:
	movq	$4, -8(%rbp)
	jmp	.L91
.L86:
	movl	-16(%rbp), %eax
	jmp	.L96
.L89:
	movl	bufp(%rip), %eax
	subl	$1, %eax
	movl	%eax, bufp(%rip)
	movl	bufp(%rip), %eax
	cltq
	leaq	buf(%rip), %rdx
	movzbl	(%rax,%rdx), %eax
	movsbl	%al, %eax
	movl	%eax, -16(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L91
.L97:
	nop
.L91:
	jmp	.L95
.L96:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	getch, .-getch
	.section	.rodata
.LC3:
	.string	"Error: zero divisor."
.LC4:
	.string	"Error: unknown command %s.\n"
.LC5:
	.string	"result: %.8g\n"
	.text
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
	subq	$240, %rsp
	movl	%edi, -212(%rbp)
	movq	%rsi, -224(%rbp)
	movq	%rdx, -232(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movl	$0, -208(%rbp)
	jmp	.L99
.L100:
	movl	-208(%rbp), %eax
	cltq
	leaq	buf(%rip), %rdx
	movb	$0, (%rax,%rdx)
	addl	$1, -208(%rbp)
.L99:
	cmpl	$99, -208(%rbp)
	jle	.L100
	nop
.L101:
	movl	$0, bufp(%rip)
	nop
.L102:
	pxor	%xmm0, %xmm0
	movsd	%xmm0, val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 8+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 16+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 24+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 32+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 40+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 48+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 56+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 64+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 72+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 80+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 88+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 96+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 104+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 112+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 120+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 128+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 136+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 144+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 152+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 160+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 168+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 176+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 184+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 192+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 200+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 208+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 216+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 224+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 232+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 240+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 248+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 256+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 264+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 272+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 280+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 288+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 296+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 304+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 312+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 320+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 328+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 336+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 344+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 352+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 360+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 368+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 376+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 384+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 392+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 400+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 408+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 416+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 424+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 432+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 440+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 448+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 456+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 464+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 472+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 480+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 488+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 496+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 504+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 512+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 520+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 528+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 536+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 544+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 552+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 560+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 568+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 576+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 584+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 592+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 600+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 608+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 616+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 624+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 632+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 640+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 648+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 656+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 664+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 672+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 680+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 688+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 696+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 704+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 712+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 720+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 728+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 736+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 744+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 752+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 760+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 768+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 776+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 784+val(%rip)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 792+val(%rip)
	nop
.L103:
	movl	$0, sp(%rip)
	nop
.L104:
	movq	$0, _TIG_IZ_Cikp_envp(%rip)
	nop
.L105:
	movq	$0, _TIG_IZ_Cikp_argv(%rip)
	nop
.L106:
	movl	$0, _TIG_IZ_Cikp_argc(%rip)
	nop
	nop
.L107:
.L108:
#APP
# 305 "mtalhakrc_C_ilealakalibisiler_calculator.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-Cikp--0
# 0 "" 2
#NO_APP
	movl	-212(%rbp), %eax
	movl	%eax, _TIG_IZ_Cikp_argc(%rip)
	movq	-224(%rbp), %rax
	movq	%rax, _TIG_IZ_Cikp_argv(%rip)
	movq	-232(%rbp), %rax
	movq	%rax, _TIG_IZ_Cikp_envp(%rip)
	nop
	movq	$29, -192(%rbp)
.L150:
	cmpq	$33, -192(%rbp)
	ja	.L155
	movq	-192(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L111(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L111(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L111:
	.long	.L129-.L111
	.long	.L155-.L111
	.long	.L128-.L111
	.long	.L127-.L111
	.long	.L155-.L111
	.long	.L155-.L111
	.long	.L126-.L111
	.long	.L155-.L111
	.long	.L125-.L111
	.long	.L124-.L111
	.long	.L155-.L111
	.long	.L155-.L111
	.long	.L155-.L111
	.long	.L123-.L111
	.long	.L155-.L111
	.long	.L122-.L111
	.long	.L155-.L111
	.long	.L155-.L111
	.long	.L121-.L111
	.long	.L155-.L111
	.long	.L120-.L111
	.long	.L155-.L111
	.long	.L155-.L111
	.long	.L119-.L111
	.long	.L118-.L111
	.long	.L117-.L111
	.long	.L116-.L111
	.long	.L155-.L111
	.long	.L115-.L111
	.long	.L114-.L111
	.long	.L113-.L111
	.long	.L155-.L111
	.long	.L112-.L111
	.long	.L110-.L111
	.text
.L121:
	movl	-204(%rbp), %eax
	subl	$10, %eax
	cmpl	$38, %eax
	ja	.L130
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L132(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L132(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L132:
	.long	.L138-.L132
	.long	.L130-.L132
	.long	.L130-.L132
	.long	.L130-.L132
	.long	.L130-.L132
	.long	.L130-.L132
	.long	.L130-.L132
	.long	.L130-.L132
	.long	.L130-.L132
	.long	.L130-.L132
	.long	.L130-.L132
	.long	.L130-.L132
	.long	.L130-.L132
	.long	.L130-.L132
	.long	.L130-.L132
	.long	.L130-.L132
	.long	.L130-.L132
	.long	.L130-.L132
	.long	.L130-.L132
	.long	.L130-.L132
	.long	.L130-.L132
	.long	.L130-.L132
	.long	.L130-.L132
	.long	.L130-.L132
	.long	.L130-.L132
	.long	.L130-.L132
	.long	.L130-.L132
	.long	.L137-.L132
	.long	.L130-.L132
	.long	.L130-.L132
	.long	.L130-.L132
	.long	.L130-.L132
	.long	.L136-.L132
	.long	.L135-.L132
	.long	.L130-.L132
	.long	.L134-.L132
	.long	.L130-.L132
	.long	.L133-.L132
	.long	.L131-.L132
	.text
.L138:
	movq	$23, -192(%rbp)
	jmp	.L139
.L137:
	movq	$0, -192(%rbp)
	jmp	.L139
.L133:
	movq	$28, -192(%rbp)
	jmp	.L139
.L136:
	movq	$33, -192(%rbp)
	jmp	.L139
.L134:
	movq	$3, -192(%rbp)
	jmp	.L139
.L135:
	movq	$13, -192(%rbp)
	jmp	.L139
.L131:
	movq	$32, -192(%rbp)
	jmp	.L139
.L130:
	movq	$15, -192(%rbp)
	nop
.L139:
	jmp	.L140
.L117:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$26, -192(%rbp)
	jmp	.L140
.L113:
	call	pop
	movq	%xmm0, %rax
	movq	%rax, -160(%rbp)
	movsd	-160(%rbp), %xmm0
	divsd	-200(%rbp), %xmm0
	movq	%xmm0, %rax
	movq	%rax, %xmm0
	call	push
	movq	$26, -192(%rbp)
	jmp	.L140
.L122:
	leaq	-112(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$26, -192(%rbp)
	jmp	.L140
.L125:
	cmpl	$-1, -204(%rbp)
	je	.L141
	movq	$18, -192(%rbp)
	jmp	.L140
.L141:
	movq	$9, -192(%rbp)
	jmp	.L140
.L119:
	call	pop
	movq	%xmm0, %rax
	movq	%rax, -152(%rbp)
	movq	-152(%rbp), %rax
	movq	%rax, %xmm0
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$26, -192(%rbp)
	jmp	.L140
.L127:
	call	pop
	movq	%xmm0, %rax
	movq	%rax, -200(%rbp)
	call	pop
	movq	%xmm0, %rax
	movq	%rax, -120(%rbp)
	movsd	-120(%rbp), %xmm0
	subsd	-200(%rbp), %xmm0
	movq	%xmm0, %rax
	movq	%rax, %xmm0
	call	push
	movq	$26, -192(%rbp)
	jmp	.L140
.L118:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$26, -192(%rbp)
	jmp	.L140
.L116:
	leaq	-112(%rbp), %rax
	movq	%rax, %rdi
	call	getop
	movl	%eax, -204(%rbp)
	movq	$8, -192(%rbp)
	jmp	.L140
.L124:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L151
	jmp	.L154
.L123:
	call	pop
	movq	%xmm0, %rax
	movq	%rax, -136(%rbp)
	call	pop
	movq	%xmm0, %rax
	movq	%rax, -128(%rbp)
	movsd	-136(%rbp), %xmm0
	addsd	-128(%rbp), %xmm0
	movq	%xmm0, %rax
	movq	%rax, %xmm0
	call	push
	movq	$26, -192(%rbp)
	jmp	.L140
.L112:
	leaq	-112(%rbp), %rax
	movq	%rax, %rdi
	call	atof@PLT
	movq	%xmm0, %rax
	movq	%rax, -168(%rbp)
	movq	-168(%rbp), %rax
	movq	%rax, %xmm0
	call	push
	movq	$26, -192(%rbp)
	jmp	.L140
.L126:
	pxor	%xmm0, %xmm0
	ucomisd	-200(%rbp), %xmm0
	jp	.L152
	pxor	%xmm0, %xmm0
	ucomisd	-200(%rbp), %xmm0
	je	.L144
.L152:
	movq	$20, -192(%rbp)
	jmp	.L140
.L144:
	movq	$24, -192(%rbp)
	jmp	.L140
.L115:
	call	pop
	movq	%xmm0, %rax
	movq	%rax, -200(%rbp)
	movq	$2, -192(%rbp)
	jmp	.L140
.L110:
	call	pop
	movq	%xmm0, %rax
	movq	%rax, -184(%rbp)
	call	pop
	movq	%xmm0, %rax
	movq	%rax, -176(%rbp)
	movsd	-184(%rbp), %xmm0
	mulsd	-176(%rbp), %xmm0
	movq	%xmm0, %rax
	movq	%rax, %xmm0
	call	push
	movq	$26, -192(%rbp)
	jmp	.L140
.L129:
	call	pop
	movq	%xmm0, %rax
	movq	%rax, -200(%rbp)
	movq	$6, -192(%rbp)
	jmp	.L140
.L114:
	movq	$26, -192(%rbp)
	jmp	.L140
.L128:
	pxor	%xmm0, %xmm0
	ucomisd	-200(%rbp), %xmm0
	jp	.L153
	pxor	%xmm0, %xmm0
	ucomisd	-200(%rbp), %xmm0
	je	.L147
.L153:
	movq	$30, -192(%rbp)
	jmp	.L140
.L147:
	movq	$25, -192(%rbp)
	jmp	.L140
.L120:
	call	pop
	movq	%xmm0, %rax
	movq	%rax, -144(%rbp)
	movsd	-144(%rbp), %xmm0
	cvttsd2sil	%xmm0, %eax
	movsd	-200(%rbp), %xmm0
	cvttsd2sil	%xmm0, %ecx
	cltd
	idivl	%ecx
	movl	%edx, %eax
	pxor	%xmm1, %xmm1
	cvtsi2sdl	%eax, %xmm1
	movq	%xmm1, %rax
	movq	%rax, %xmm0
	call	push
	movq	$26, -192(%rbp)
	jmp	.L140
.L155:
	nop
.L140:
	jmp	.L150
.L154:
	call	__stack_chk_fail@PLT
.L151:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	main, .-main
	.section	.rodata
.LC6:
	.string	"Error: stack empty."
	.text
	.globl	pop
	.type	pop, @function
pop:
.LFB11:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$4, -8(%rbp)
.L168:
	cmpq	$5, -8(%rbp)
	ja	.L169
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L159(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L159(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L159:
	.long	.L163-.L159
	.long	.L169-.L159
	.long	.L162-.L159
	.long	.L161-.L159
	.long	.L160-.L159
	.long	.L158-.L159
	.text
.L160:
	movl	sp(%rip), %eax
	testl	%eax, %eax
	jle	.L164
	movq	$2, -8(%rbp)
	jmp	.L166
.L164:
	movq	$5, -8(%rbp)
	jmp	.L166
.L161:
	movl	sp(%rip), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	leaq	val(%rip), %rax
	movsd	(%rdx,%rax), %xmm0
	jmp	.L167
.L158:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$0, -8(%rbp)
	jmp	.L166
.L163:
	pxor	%xmm0, %xmm0
	jmp	.L167
.L162:
	movl	sp(%rip), %eax
	subl	$1, %eax
	movl	%eax, sp(%rip)
	movq	$3, -8(%rbp)
	jmp	.L166
.L169:
	nop
.L166:
	jmp	.L168
.L167:
	movq	%xmm0, %rax
	movq	%rax, %xmm0
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE11:
	.size	pop, .-pop
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
