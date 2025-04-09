	.file	"jayy-patell_dsa-practice_main_flatten.c"
	.text
	.globl	_TIG_IZ_ZV9a_argv
	.bss
	.align 8
	.type	_TIG_IZ_ZV9a_argv, @object
	.size	_TIG_IZ_ZV9a_argv, 8
_TIG_IZ_ZV9a_argv:
	.zero	8
	.globl	stack
	.align 32
	.type	stack, @object
	.size	stack, 400
stack:
	.zero	400
	.globl	_TIG_IZ_ZV9a_argc
	.align 4
	.type	_TIG_IZ_ZV9a_argc, @object
	.size	_TIG_IZ_ZV9a_argc, 4
_TIG_IZ_ZV9a_argc:
	.zero	4
	.globl	_TIG_IZ_ZV9a_envp
	.align 8
	.type	_TIG_IZ_ZV9a_envp, @object
	.size	_TIG_IZ_ZV9a_envp, 8
_TIG_IZ_ZV9a_envp:
	.zero	8
	.globl	top
	.align 4
	.type	top, @object
	.size	top, 4
top:
	.zero	4
	.section	.rodata
.LC0:
	.string	"%c"
	.align 8
.LC1:
	.string	"ASSUMPTION: There are only four operators(*, /, +, -) in an expression and operand is single digit only."
	.align 8
.LC2:
	.string	" \nEnter postfix expression,\npress right parenthesis ')' for end expression : "
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
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movl	$-1, top(%rip)
	nop
.L2:
	movl	$0, -128(%rbp)
	jmp	.L3
.L4:
	movl	-128(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	stack(%rip), %rax
	movl	$0, (%rdx,%rax)
	addl	$1, -128(%rbp)
.L3:
	cmpl	$99, -128(%rbp)
	jle	.L4
	nop
.L5:
	movq	$0, _TIG_IZ_ZV9a_envp(%rip)
	nop
.L6:
	movq	$0, _TIG_IZ_ZV9a_argv(%rip)
	nop
.L7:
	movl	$0, _TIG_IZ_ZV9a_argc(%rip)
	nop
	nop
.L8:
.L9:
#APP
# 141 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-ZV9a--0
# 0 "" 2
#NO_APP
	movl	-132(%rbp), %eax
	movl	%eax, _TIG_IZ_ZV9a_argc(%rip)
	movq	-144(%rbp), %rax
	movq	%rax, _TIG_IZ_ZV9a_argv(%rip)
	movq	-152(%rbp), %rax
	movq	%rax, _TIG_IZ_ZV9a_envp(%rip)
	nop
	movq	$4, -120(%rbp)
.L26:
	cmpq	$10, -120(%rbp)
	ja	.L29
	movq	-120(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L12(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L12(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L12:
	.long	.L19-.L12
	.long	.L29-.L12
	.long	.L18-.L12
	.long	.L17-.L12
	.long	.L16-.L12
	.long	.L15-.L12
	.long	.L14-.L12
	.long	.L29-.L12
	.long	.L13-.L12
	.long	.L29-.L12
	.long	.L11-.L12
	.text
.L16:
	movq	$0, -120(%rbp)
	jmp	.L20
.L13:
	addl	$1, -124(%rbp)
	movq	$2, -120(%rbp)
	jmp	.L20
.L17:
	leaq	-112(%rbp), %rdx
	movl	-124(%rbp), %eax
	cltq
	addq	%rdx, %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$6, -120(%rbp)
	jmp	.L20
.L14:
	movl	-124(%rbp), %eax
	cltq
	movzbl	-112(%rbp,%rax), %eax
	cmpb	$41, %al
	jne	.L21
	movq	$5, -120(%rbp)
	jmp	.L20
.L21:
	movq	$8, -120(%rbp)
	jmp	.L20
.L15:
	leaq	-112(%rbp), %rax
	movq	%rax, %rdi
	call	EvalPostfix
	movq	$10, -120(%rbp)
	jmp	.L20
.L11:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L27
	jmp	.L28
.L19:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$0, -124(%rbp)
	movq	$2, -120(%rbp)
	jmp	.L20
.L18:
	cmpl	$99, -124(%rbp)
	jg	.L24
	movq	$3, -120(%rbp)
	jmp	.L20
.L24:
	movq	$5, -120(%rbp)
	jmp	.L20
.L29:
	nop
.L20:
	jmp	.L26
.L28:
	call	__stack_chk_fail@PLT
.L27:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	main, .-main
	.section	.rodata
.LC3:
	.string	"stack over flow"
	.text
	.globl	push
	.type	push, @function
push:
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
	movq	$0, -8(%rbp)
.L42:
	cmpq	$5, -8(%rbp)
	ja	.L43
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L33(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L33(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L33:
	.long	.L37-.L33
	.long	.L43-.L33
	.long	.L44-.L33
	.long	.L35-.L33
	.long	.L34-.L33
	.long	.L44-.L33
	.text
.L34:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$2, -8(%rbp)
	jmp	.L38
.L35:
	movl	top(%rip), %eax
	addl	$1, %eax
	movl	%eax, top(%rip)
	movl	top(%rip), %eax
	cltq
	leaq	0(,%rax,4), %rcx
	leaq	stack(%rip), %rdx
	movl	-20(%rbp), %eax
	movl	%eax, (%rcx,%rdx)
	movq	$5, -8(%rbp)
	jmp	.L38
.L37:
	movl	top(%rip), %eax
	cmpl	$98, %eax
	jle	.L40
	movq	$4, -8(%rbp)
	jmp	.L38
.L40:
	movq	$3, -8(%rbp)
	jmp	.L38
.L43:
	nop
.L38:
	jmp	.L42
.L44:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	push, .-push
	.section	.rodata
	.align 8
.LC4:
	.string	" \n Result of expression evaluation : %d \n"
	.text
	.globl	EvalPostfix
	.type	EvalPostfix, @function
EvalPostfix:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movq	%rdi, -56(%rbp)
	movq	$29, -8(%rbp)
.L88:
	cmpq	$29, -8(%rbp)
	ja	.L89
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L48(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L48(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L48:
	.long	.L67-.L48
	.long	.L89-.L48
	.long	.L66-.L48
	.long	.L65-.L48
	.long	.L64-.L48
	.long	.L63-.L48
	.long	.L89-.L48
	.long	.L89-.L48
	.long	.L89-.L48
	.long	.L89-.L48
	.long	.L89-.L48
	.long	.L62-.L48
	.long	.L61-.L48
	.long	.L60-.L48
	.long	.L90-.L48
	.long	.L58-.L48
	.long	.L57-.L48
	.long	.L56-.L48
	.long	.L55-.L48
	.long	.L54-.L48
	.long	.L89-.L48
	.long	.L53-.L48
	.long	.L89-.L48
	.long	.L89-.L48
	.long	.L52-.L48
	.long	.L51-.L48
	.long	.L50-.L48
	.long	.L49-.L48
	.long	.L89-.L48
	.long	.L47-.L48
	.text
.L55:
	movsbl	-37(%rbp), %eax
	cmpl	$47, %eax
	je	.L68
	cmpl	$47, %eax
	jg	.L69
	cmpl	$45, %eax
	je	.L70
	cmpl	$45, %eax
	jg	.L69
	cmpl	$42, %eax
	je	.L71
	cmpl	$43, %eax
	je	.L72
	jmp	.L69
.L70:
	movq	$26, -8(%rbp)
	jmp	.L73
.L72:
	movq	$21, -8(%rbp)
	jmp	.L73
.L68:
	movq	$13, -8(%rbp)
	jmp	.L73
.L71:
	movq	$3, -8(%rbp)
	jmp	.L73
.L69:
	movq	$5, -8(%rbp)
	nop
.L73:
	jmp	.L74
.L51:
	cmpb	$42, -37(%rbp)
	jne	.L75
	movq	$11, -8(%rbp)
	jmp	.L74
.L75:
	movq	$0, -8(%rbp)
	jmp	.L74
.L64:
	movq	-16(%rbp), %rax
	movq	(%rax), %rdx
	movsbq	-37(%rbp), %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$2048, %eax
	testl	%eax, %eax
	je	.L77
	movq	$12, -8(%rbp)
	jmp	.L74
.L77:
	movq	$24, -8(%rbp)
	jmp	.L74
.L58:
	cmpb	$45, -37(%rbp)
	jne	.L80
	movq	$11, -8(%rbp)
	jmp	.L74
.L80:
	movq	$25, -8(%rbp)
	jmp	.L74
.L61:
	movsbl	-37(%rbp), %eax
	subl	$48, %eax
	movl	%eax, %edi
	call	push
	movq	$16, -8(%rbp)
	jmp	.L74
.L65:
	movl	-24(%rbp), %eax
	imull	-28(%rbp), %eax
	movl	%eax, -32(%rbp)
	movq	$19, -8(%rbp)
	jmp	.L74
.L57:
	addl	$1, -36(%rbp)
	movq	$17, -8(%rbp)
	jmp	.L74
.L52:
	cmpb	$43, -37(%rbp)
	jne	.L82
	movq	$11, -8(%rbp)
	jmp	.L74
.L82:
	movq	$15, -8(%rbp)
	jmp	.L74
.L53:
	movl	-24(%rbp), %edx
	movl	-28(%rbp), %eax
	addl	%edx, %eax
	movl	%eax, -32(%rbp)
	movq	$19, -8(%rbp)
	jmp	.L74
.L50:
	movl	-24(%rbp), %eax
	subl	-28(%rbp), %eax
	movl	%eax, -32(%rbp)
	movq	$19, -8(%rbp)
	jmp	.L74
.L62:
	call	pop
	movl	%eax, -28(%rbp)
	call	pop
	movl	%eax, -24(%rbp)
	movq	$18, -8(%rbp)
	jmp	.L74
.L60:
	movl	-24(%rbp), %eax
	cltd
	idivl	-28(%rbp)
	movl	%eax, -32(%rbp)
	movq	$19, -8(%rbp)
	jmp	.L74
.L54:
	movl	-32(%rbp), %eax
	movl	%eax, %edi
	call	push
	movq	$16, -8(%rbp)
	jmp	.L74
.L56:
	movl	-36(%rbp), %eax
	movslq	%eax, %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	cmpb	$41, %al
	je	.L84
	movq	$27, -8(%rbp)
	jmp	.L74
.L84:
	movq	$2, -8(%rbp)
	jmp	.L74
.L49:
	movl	-36(%rbp), %eax
	movslq	%eax, %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	movb	%al, -37(%rbp)
	call	__ctype_b_loc@PLT
	movq	%rax, -16(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L74
.L63:
	movq	$19, -8(%rbp)
	jmp	.L74
.L67:
	cmpb	$47, -37(%rbp)
	jne	.L86
	movq	$11, -8(%rbp)
	jmp	.L74
.L86:
	movq	$16, -8(%rbp)
	jmp	.L74
.L47:
	movl	$0, -36(%rbp)
	movq	$17, -8(%rbp)
	jmp	.L74
.L66:
	call	pop
	movl	%eax, -20(%rbp)
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$14, -8(%rbp)
	jmp	.L74
.L89:
	nop
.L74:
	jmp	.L88
.L90:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	EvalPostfix, .-EvalPostfix
	.section	.rodata
.LC5:
	.string	"stack under flow"
	.text
	.globl	pop
	.type	pop, @function
pop:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$0, -8(%rbp)
.L103:
	cmpq	$5, -8(%rbp)
	ja	.L104
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L94(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L94(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L94:
	.long	.L98-.L94
	.long	.L97-.L94
	.long	.L104-.L94
	.long	.L96-.L94
	.long	.L95-.L94
	.long	.L93-.L94
	.text
.L95:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -8(%rbp)
	jmp	.L99
.L97:
	movl	$0, %eax
	jmp	.L100
.L96:
	movl	top(%rip), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	stack(%rip), %rax
	movl	(%rdx,%rax), %eax
	movl	%eax, -12(%rbp)
	movl	top(%rip), %eax
	subl	$1, %eax
	movl	%eax, top(%rip)
	movq	$5, -8(%rbp)
	jmp	.L99
.L93:
	movl	-12(%rbp), %eax
	jmp	.L100
.L98:
	movl	top(%rip), %eax
	testl	%eax, %eax
	jns	.L101
	movq	$4, -8(%rbp)
	jmp	.L99
.L101:
	movq	$3, -8(%rbp)
	jmp	.L99
.L104:
	nop
.L99:
	jmp	.L103
.L100:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
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
