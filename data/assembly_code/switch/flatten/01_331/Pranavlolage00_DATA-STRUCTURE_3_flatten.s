	.file	"Pranavlolage00_DATA-STRUCTURE_3_flatten.c"
	.text
	.globl	_TIG_IZ_uqzm_envp
	.bss
	.align 8
	.type	_TIG_IZ_uqzm_envp, @object
	.size	_TIG_IZ_uqzm_envp, 8
_TIG_IZ_uqzm_envp:
	.zero	8
	.globl	_TIG_IZ_uqzm_argc
	.align 4
	.type	_TIG_IZ_uqzm_argc, @object
	.size	_TIG_IZ_uqzm_argc, 4
_TIG_IZ_uqzm_argc:
	.zero	4
	.globl	_TIG_IZ_uqzm_argv
	.align 8
	.type	_TIG_IZ_uqzm_argv, @object
	.size	_TIG_IZ_uqzm_argv, 8
_TIG_IZ_uqzm_argv:
	.zero	8
	.globl	top
	.align 8
	.type	top, @object
	.size	top, 8
top:
	.zero	8
	.section	.rodata
.LC0:
	.string	"result=%d"
.LC1:
	.string	"enter postfix string :"
.LC2:
	.string	"%s"
.LC3:
	.string	"enter value of A:"
.LC4:
	.string	"%d"
.LC5:
	.string	"enter value of B:"
.LC6:
	.string	"enter value of C:"
.LC7:
	.string	"enter value of D:"
.LC8:
	.string	"enter value of E:"
	.text
	.globl	main
	.type	main, @function
main:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$144, %rsp
	movl	%edi, -116(%rbp)
	movq	%rsi, -128(%rbp)
	movq	%rdx, -136(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, top(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_uqzm_envp(%rip)
	nop
.L3:
	movq	$0, _TIG_IZ_uqzm_argv(%rip)
	nop
.L4:
	movl	$0, _TIG_IZ_uqzm_argc(%rip)
	nop
	nop
.L5:
.L6:
#APP
# 123 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-uqzm--0
# 0 "" 2
#NO_APP
	movl	-116(%rbp), %eax
	movl	%eax, _TIG_IZ_uqzm_argc(%rip)
	movq	-128(%rbp), %rax
	movq	%rax, _TIG_IZ_uqzm_argv(%rip)
	movq	-136(%rbp), %rax
	movq	%rax, _TIG_IZ_uqzm_envp(%rip)
	nop
	movq	$36, -56(%rbp)
.L52:
	cmpq	$36, -56(%rbp)
	ja	.L55
	movq	-56(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L9(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L9(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L9:
	.long	.L31-.L9
	.long	.L30-.L9
	.long	.L29-.L9
	.long	.L55-.L9
	.long	.L55-.L9
	.long	.L28-.L9
	.long	.L27-.L9
	.long	.L55-.L9
	.long	.L26-.L9
	.long	.L25-.L9
	.long	.L55-.L9
	.long	.L55-.L9
	.long	.L24-.L9
	.long	.L23-.L9
	.long	.L55-.L9
	.long	.L55-.L9
	.long	.L55-.L9
	.long	.L22-.L9
	.long	.L21-.L9
	.long	.L55-.L9
	.long	.L20-.L9
	.long	.L19-.L9
	.long	.L55-.L9
	.long	.L18-.L9
	.long	.L17-.L9
	.long	.L16-.L9
	.long	.L55-.L9
	.long	.L15-.L9
	.long	.L55-.L9
	.long	.L55-.L9
	.long	.L14-.L9
	.long	.L13-.L9
	.long	.L12-.L9
	.long	.L55-.L9
	.long	.L11-.L9
	.long	.L10-.L9
	.long	.L8-.L9
	.text
.L21:
	movq	-64(%rbp), %rax
	movq	(%rax), %rdx
	movl	-84(%rbp), %eax
	cltq
	movzbl	-48(%rbp,%rax), %eax
	movsbq	%al, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$1024, %eax
	testl	%eax, %eax
	je	.L32
	movq	$9, -56(%rbp)
	jmp	.L34
.L32:
	movq	$2, -56(%rbp)
	jmp	.L34
.L16:
	movl	-84(%rbp), %eax
	cltq
	movzbl	-48(%rbp,%rax), %eax
	testb	%al, %al
	je	.L35
	movq	$20, -56(%rbp)
	jmp	.L34
.L35:
	movq	$8, -56(%rbp)
	jmp	.L34
.L14:
	movl	-92(%rbp), %eax
	movl	%eax, %edi
	call	push
	movq	$13, -56(%rbp)
	jmp	.L34
.L13:
	movl	-76(%rbp), %eax
	cltd
	idivl	-80(%rbp)
	movl	%eax, -72(%rbp)
	movq	$5, -56(%rbp)
	jmp	.L34
.L24:
	movl	-84(%rbp), %eax
	cltq
	movzbl	-48(%rbp,%rax), %eax
	movsbl	%al, %eax
	cmpl	$47, %eax
	je	.L37
	cmpl	$47, %eax
	jg	.L38
	cmpl	$45, %eax
	je	.L39
	cmpl	$45, %eax
	jg	.L38
	cmpl	$42, %eax
	je	.L40
	cmpl	$43, %eax
	je	.L41
	jmp	.L38
.L40:
	movq	$24, -56(%rbp)
	jmp	.L42
.L37:
	movq	$31, -56(%rbp)
	jmp	.L42
.L39:
	movq	$35, -56(%rbp)
	jmp	.L42
.L41:
	movq	$27, -56(%rbp)
	jmp	.L42
.L38:
	movq	$17, -56(%rbp)
	nop
.L42:
	jmp	.L34
.L26:
	call	pop
	movl	%eax, -68(%rbp)
	movl	-68(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$23, -56(%rbp)
	jmp	.L34
.L30:
	movl	-96(%rbp), %eax
	movl	%eax, %edi
	call	push
	movq	$13, -56(%rbp)
	jmp	.L34
.L18:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L53
	jmp	.L54
.L17:
	movl	-76(%rbp), %eax
	imull	-80(%rbp), %eax
	movl	%eax, -72(%rbp)
	movq	$5, -56(%rbp)
	jmp	.L34
.L19:
	movq	$13, -56(%rbp)
	jmp	.L34
.L8:
	movq	$0, -56(%rbp)
	jmp	.L34
.L25:
	movl	-84(%rbp), %eax
	cltq
	movzbl	-48(%rbp,%rax), %eax
	movsbl	%al, %eax
	subl	$97, %eax
	cmpl	$4, %eax
	ja	.L44
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L46(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L46(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L46:
	.long	.L50-.L46
	.long	.L49-.L46
	.long	.L48-.L46
	.long	.L47-.L46
	.long	.L45-.L46
	.text
.L45:
	movq	$34, -56(%rbp)
	jmp	.L51
.L47:
	movq	$30, -56(%rbp)
	jmp	.L51
.L48:
	movq	$1, -56(%rbp)
	jmp	.L51
.L49:
	movq	$32, -56(%rbp)
	jmp	.L51
.L50:
	movq	$6, -56(%rbp)
	jmp	.L51
.L44:
	movq	$21, -56(%rbp)
	nop
.L51:
	jmp	.L34
.L23:
	addl	$1, -84(%rbp)
	movq	$25, -56(%rbp)
	jmp	.L34
.L12:
	movl	-100(%rbp), %eax
	movl	%eax, %edi
	call	push
	movq	$13, -56(%rbp)
	jmp	.L34
.L22:
	movq	$5, -56(%rbp)
	jmp	.L34
.L27:
	movl	-104(%rbp), %eax
	movl	%eax, %edi
	call	push
	movq	$13, -56(%rbp)
	jmp	.L34
.L15:
	movl	-76(%rbp), %edx
	movl	-80(%rbp), %eax
	addl	%edx, %eax
	movl	%eax, -72(%rbp)
	movq	$5, -56(%rbp)
	jmp	.L34
.L11:
	movl	-88(%rbp), %eax
	movl	%eax, %edi
	call	push
	movq	$13, -56(%rbp)
	jmp	.L34
.L28:
	movl	-72(%rbp), %eax
	movl	%eax, %edi
	call	push
	movq	$13, -56(%rbp)
	jmp	.L34
.L31:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-48(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-104(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-100(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-96(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-92(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-88(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	$0, -84(%rbp)
	movq	$25, -56(%rbp)
	jmp	.L34
.L10:
	movl	-80(%rbp), %eax
	movl	%eax, -76(%rbp)
	movl	-76(%rbp), %eax
	movl	%eax, -72(%rbp)
	movq	$5, -56(%rbp)
	jmp	.L34
.L29:
	call	pop
	movl	%eax, -80(%rbp)
	call	pop
	movl	%eax, -76(%rbp)
	movq	$12, -56(%rbp)
	jmp	.L34
.L20:
	call	__ctype_b_loc@PLT
	movq	%rax, -64(%rbp)
	movq	$18, -56(%rbp)
	jmp	.L34
.L55:
	nop
.L34:
	jmp	.L52
.L54:
	call	__stack_chk_fail@PLT
.L53:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	main, .-main
	.globl	init
	.type	init, @function
init:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	$0, -8(%rbp)
.L61:
	cmpq	$0, -8(%rbp)
	je	.L57
	cmpq	$1, -8(%rbp)
	jne	.L63
	jmp	.L62
.L57:
	movq	$0, top(%rip)
	movq	$1, -8(%rbp)
	jmp	.L60
.L63:
	nop
.L60:
	jmp	.L61
.L62:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	init, .-init
	.section	.rodata
.LC9:
	.string	"stack is empty..."
	.text
	.globl	pop
	.type	pop, @function
pop:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	$0, -8(%rbp)
.L78:
	cmpq	$7, -8(%rbp)
	ja	.L79
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L67(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L67(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L67:
	.long	.L73-.L67
	.long	.L72-.L67
	.long	.L71-.L67
	.long	.L70-.L67
	.long	.L69-.L67
	.long	.L68-.L67
	.long	.L79-.L67
	.long	.L66-.L67
	.text
.L69:
	movl	-24(%rbp), %eax
	jmp	.L74
.L72:
	movq	-16(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, -24(%rbp)
	movq	top(%rip), %rax
	movq	8(%rax), %rax
	movq	%rax, top(%rip)
	movq	-16(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$4, -8(%rbp)
	jmp	.L75
.L70:
	movl	$0, %eax
	jmp	.L74
.L68:
	cmpl	$0, -20(%rbp)
	je	.L76
	movq	$7, -8(%rbp)
	jmp	.L75
.L76:
	movq	$1, -8(%rbp)
	jmp	.L75
.L73:
	movq	$2, -8(%rbp)
	jmp	.L75
.L66:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -8(%rbp)
	jmp	.L75
.L71:
	movq	top(%rip), %rax
	movq	%rax, -16(%rbp)
	call	isempty
	movl	%eax, -20(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L75
.L79:
	nop
.L75:
	jmp	.L78
.L74:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	pop, .-pop
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
	movq	$0, -8(%rbp)
.L89:
	cmpq	$3, -8(%rbp)
	je	.L81
	cmpq	$3, -8(%rbp)
	ja	.L90
	cmpq	$0, -8(%rbp)
	je	.L83
	cmpq	$2, -8(%rbp)
	je	.L84
	jmp	.L90
.L81:
	movl	$1, %eax
	jmp	.L85
.L83:
	movq	top(%rip), %rax
	testq	%rax, %rax
	jne	.L86
	movq	$3, -8(%rbp)
	jmp	.L88
.L86:
	movq	$2, -8(%rbp)
	jmp	.L88
.L84:
	movl	$0, %eax
	jmp	.L85
.L90:
	nop
.L88:
	jmp	.L89
.L85:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	isempty, .-isempty
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
	subq	$48, %rsp
	movl	%edi, -36(%rbp)
	movq	$2, -24(%rbp)
.L97:
	cmpq	$2, -24(%rbp)
	je	.L92
	cmpq	$2, -24(%rbp)
	ja	.L99
	cmpq	$0, -24(%rbp)
	je	.L94
	cmpq	$1, -24(%rbp)
	jne	.L99
	jmp	.L98
.L94:
	movl	$16, %edi
	call	malloc@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movl	-36(%rbp), %edx
	movl	%edx, (%rax)
	movq	top(%rip), %rdx
	movq	-8(%rbp), %rax
	movq	%rdx, 8(%rax)
	movq	-8(%rbp), %rax
	movq	%rax, top(%rip)
	movq	$1, -24(%rbp)
	jmp	.L96
.L92:
	movq	$0, -24(%rbp)
	jmp	.L96
.L99:
	nop
.L96:
	jmp	.L97
.L98:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	push, .-push
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
