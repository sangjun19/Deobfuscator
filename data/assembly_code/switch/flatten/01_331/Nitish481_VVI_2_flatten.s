	.file	"Nitish481_VVI_2_flatten.c"
	.text
	.globl	_TIG_IZ_maBq_envp
	.bss
	.align 8
	.type	_TIG_IZ_maBq_envp, @object
	.size	_TIG_IZ_maBq_envp, 8
_TIG_IZ_maBq_envp:
	.zero	8
	.globl	_TIG_IZ_maBq_argv
	.align 8
	.type	_TIG_IZ_maBq_argv, @object
	.size	_TIG_IZ_maBq_argv, 8
_TIG_IZ_maBq_argv:
	.zero	8
	.globl	_TIG_IZ_maBq_argc
	.align 4
	.type	_TIG_IZ_maBq_argc, @object
	.size	_TIG_IZ_maBq_argc, 4
_TIG_IZ_maBq_argc:
	.zero	4
	.text
	.globl	isfull
	.type	isfull, @function
isfull:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	$0, -8(%rbp)
.L9:
	cmpq	$2, -8(%rbp)
	je	.L2
	cmpq	$2, -8(%rbp)
	ja	.L10
	cmpq	$0, -8(%rbp)
	je	.L4
	cmpq	$1, -8(%rbp)
	jne	.L10
	movl	$1, %eax
	jmp	.L5
.L4:
	movl	16(%rbp), %eax
	movl	32(%rbp), %edx
	subl	$1, %edx
	cmpl	%edx, %eax
	jne	.L6
	movq	$1, -8(%rbp)
	jmp	.L8
.L6:
	movq	$2, -8(%rbp)
	jmp	.L8
.L2:
	movl	$0, %eax
	jmp	.L5
.L10:
	nop
.L8:
	jmp	.L9
.L5:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	isfull, .-isfull
	.globl	isempty
	.type	isempty, @function
isempty:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	$0, -8(%rbp)
.L19:
	cmpq	$2, -8(%rbp)
	je	.L12
	cmpq	$2, -8(%rbp)
	ja	.L20
	cmpq	$0, -8(%rbp)
	je	.L14
	cmpq	$1, -8(%rbp)
	jne	.L20
	movl	$1, %eax
	jmp	.L15
.L14:
	movl	16(%rbp), %eax
	cmpl	$-1, %eax
	jne	.L16
	movq	$1, -8(%rbp)
	jmp	.L18
.L16:
	movq	$2, -8(%rbp)
	jmp	.L18
.L12:
	movl	$0, %eax
	jmp	.L15
.L20:
	nop
.L18:
	jmp	.L19
.L15:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	isempty, .-isempty
	.section	.rodata
.LC0:
	.string	"System has %d stacks\n"
	.align 8
.LC1:
	.string	"Enter in which stack you want to push = "
.LC2:
	.string	"%d"
.LC3:
	.string	"Enter element to be pushed = "
	.align 8
.LC4:
	.string	"System has only %d number of stack.\nTry again\n\n"
.LC5:
	.string	"No stack in the system"
	.align 8
.LC6:
	.string	"Enter right option between 0 to 3"
.LC7:
	.string	"Top element is %d\n"
	.align 8
.LC8:
	.string	"\nEnter your choice:\nEnter 0 for exit\nEnter 1 for create\nEnter 2 for PUSH\nEnter 3 for POP\nEnter 4 for PEEk"
.LC9:
	.string	"Popped out element=%d\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$96, %rsp
	movl	%edi, -68(%rbp)
	movq	%rsi, -80(%rbp)
	movq	%rdx, -88(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_maBq_envp(%rip)
	nop
.L22:
	movq	$0, _TIG_IZ_maBq_argv(%rip)
	nop
.L23:
	movl	$0, _TIG_IZ_maBq_argc(%rip)
	nop
	nop
.L24:
.L25:
#APP
# 205 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-maBq--0
# 0 "" 2
#NO_APP
	movl	-68(%rbp), %eax
	movl	%eax, _TIG_IZ_maBq_argc(%rip)
	movq	-80(%rbp), %rax
	movq	%rax, _TIG_IZ_maBq_argv(%rip)
	movq	-88(%rbp), %rax
	movq	%rax, _TIG_IZ_maBq_envp(%rip)
	nop
	movq	$20, -32(%rbp)
.L100:
	cmpq	$59, -32(%rbp)
	ja	.L103
	movq	-32(%rbp), %rax
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
	.long	.L65-.L28
	.long	.L64-.L28
	.long	.L63-.L28
	.long	.L62-.L28
	.long	.L61-.L28
	.long	.L103-.L28
	.long	.L60-.L28
	.long	.L59-.L28
	.long	.L58-.L28
	.long	.L57-.L28
	.long	.L103-.L28
	.long	.L56-.L28
	.long	.L55-.L28
	.long	.L103-.L28
	.long	.L54-.L28
	.long	.L103-.L28
	.long	.L103-.L28
	.long	.L53-.L28
	.long	.L52-.L28
	.long	.L103-.L28
	.long	.L51-.L28
	.long	.L50-.L28
	.long	.L49-.L28
	.long	.L48-.L28
	.long	.L47-.L28
	.long	.L46-.L28
	.long	.L103-.L28
	.long	.L45-.L28
	.long	.L44-.L28
	.long	.L43-.L28
	.long	.L103-.L28
	.long	.L103-.L28
	.long	.L42-.L28
	.long	.L103-.L28
	.long	.L103-.L28
	.long	.L103-.L28
	.long	.L103-.L28
	.long	.L41-.L28
	.long	.L103-.L28
	.long	.L40-.L28
	.long	.L39-.L28
	.long	.L38-.L28
	.long	.L103-.L28
	.long	.L37-.L28
	.long	.L36-.L28
	.long	.L103-.L28
	.long	.L35-.L28
	.long	.L34-.L28
	.long	.L103-.L28
	.long	.L33-.L28
	.long	.L103-.L28
	.long	.L32-.L28
	.long	.L31-.L28
	.long	.L103-.L28
	.long	.L30-.L28
	.long	.L29-.L28
	.long	.L103-.L28
	.long	.L103-.L28
	.long	.L103-.L28
	.long	.L27-.L28
	.text
.L52:
	movl	-52(%rbp), %eax
	cmpl	%eax, -44(%rbp)
	jge	.L66
	movq	$51, -32(%rbp)
	jmp	.L68
.L66:
	movq	$1, -32(%rbp)
	jmp	.L68
.L46:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L101
	jmp	.L102
.L33:
	movl	-52(%rbp), %eax
	cmpl	%eax, -44(%rbp)
	jge	.L70
	movq	$3, -32(%rbp)
	jmp	.L68
.L70:
	movq	$21, -32(%rbp)
	jmp	.L68
.L31:
	movl	-48(%rbp), %eax
	cmpl	$4, %eax
	ja	.L72
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L74(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L74(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L74:
	.long	.L78-.L74
	.long	.L77-.L74
	.long	.L76-.L74
	.long	.L75-.L74
	.long	.L73-.L74
	.text
.L73:
	movq	$7, -32(%rbp)
	jmp	.L79
.L75:
	movq	$39, -32(%rbp)
	jmp	.L79
.L76:
	movq	$12, -32(%rbp)
	jmp	.L79
.L77:
	movq	$54, -32(%rbp)
	jmp	.L79
.L78:
	movq	$44, -32(%rbp)
	jmp	.L79
.L72:
	movq	$37, -32(%rbp)
	nop
.L79:
	jmp	.L68
.L61:
	cmpl	$0, -44(%rbp)
	jne	.L80
	movq	$47, -32(%rbp)
	jmp	.L68
.L80:
	movq	$3, -32(%rbp)
	jmp	.L68
.L54:
	movl	-52(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	leaq	-24(%rax), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	subq	$8, %rsp
	pushq	16(%rax)
	pushq	8(%rax)
	pushq	(%rax)
	call	peek
	addq	$32, %rsp
	movl	%eax, -56(%rbp)
	movq	$23, -32(%rbp)
	jmp	.L68
.L55:
	movl	-44(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$4, -32(%rbp)
	jmp	.L68
.L58:
	movl	-52(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	leaq	-24(%rax), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	subq	$8, %rsp
	pushq	16(%rax)
	pushq	8(%rax)
	pushq	(%rax)
	call	display
	addq	$32, %rsp
	movq	$44, -32(%rbp)
	jmp	.L68
.L30:
	movl	-44(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, %rdi
	call	create
	addl	$1, -44(%rbp)
	movl	-44(%rbp), %eax
	addl	$1, %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-40(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -40(%rbp)
	movl	-44(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$44, -32(%rbp)
	jmp	.L68
.L64:
	movl	-52(%rbp), %eax
	cmpl	%eax, -44(%rbp)
	jge	.L82
	movq	$40, -32(%rbp)
	jmp	.L68
.L82:
	movq	$14, -32(%rbp)
	jmp	.L68
.L48:
	movl	-56(%rbp), %eax
	cmpl	$-1, %eax
	je	.L84
	movq	$46, -32(%rbp)
	jmp	.L68
.L84:
	movq	$8, -32(%rbp)
	jmp	.L68
.L62:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-52(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$24, -32(%rbp)
	jmp	.L68
.L47:
	movl	-52(%rbp), %eax
	cmpl	%eax, -44(%rbp)
	jge	.L86
	movq	$6, -32(%rbp)
	jmp	.L68
.L86:
	movq	$49, -32(%rbp)
	jmp	.L68
.L50:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-56(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-56(%rbp), %edx
	movl	-52(%rbp), %eax
	movslq	%eax, %rcx
	movq	%rcx, %rax
	addq	%rax, %rax
	addq	%rcx, %rax
	salq	$3, %rax
	leaq	-24(%rax), %rcx
	movq	-40(%rbp), %rax
	addq	%rcx, %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	push
	movq	$44, -32(%rbp)
	jmp	.L68
.L56:
	movl	-56(%rbp), %eax
	cmpl	$-1, %eax
	je	.L88
	movq	$2, -32(%rbp)
	jmp	.L68
.L88:
	movq	$59, -32(%rbp)
	jmp	.L68
.L57:
	cmpl	$0, -44(%rbp)
	jne	.L90
	movq	$0, -32(%rbp)
	jmp	.L68
.L90:
	movq	$27, -32(%rbp)
	jmp	.L68
.L32:
	movl	-44(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -32(%rbp)
	jmp	.L68
.L42:
	cmpl	$0, -44(%rbp)
	jne	.L92
	movq	$22, -32(%rbp)
	jmp	.L68
.L92:
	movq	$40, -32(%rbp)
	jmp	.L68
.L53:
	movl	-52(%rbp), %eax
	cmpl	%eax, -44(%rbp)
	jge	.L94
	movq	$27, -32(%rbp)
	jmp	.L68
.L94:
	movq	$41, -32(%rbp)
	jmp	.L68
.L39:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-52(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$18, -32(%rbp)
	jmp	.L68
.L29:
	movl	$0, -44(%rbp)
	movl	$24, %edi
	call	malloc@PLT
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, -40(%rbp)
	movq	$43, -32(%rbp)
	jmp	.L68
.L27:
	movl	-52(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	leaq	-24(%rax), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	subq	$8, %rsp
	pushq	16(%rax)
	pushq	8(%rax)
	pushq	(%rax)
	call	display
	addq	$32, %rsp
	movq	$44, -32(%rbp)
	jmp	.L68
.L60:
	movl	-44(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$49, -32(%rbp)
	jmp	.L68
.L45:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-52(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$29, -32(%rbp)
	jmp	.L68
.L49:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$44, -32(%rbp)
	jmp	.L68
.L44:
	movl	-44(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$17, -32(%rbp)
	jmp	.L68
.L34:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$44, -32(%rbp)
	jmp	.L68
.L36:
	movl	-48(%rbp), %eax
	testl	%eax, %eax
	je	.L96
	movq	$43, -32(%rbp)
	jmp	.L68
.L96:
	movq	$25, -32(%rbp)
	jmp	.L68
.L41:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$44, -32(%rbp)
	jmp	.L68
.L38:
	movl	-52(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	leaq	-24(%rax), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, %rdi
	call	pop
	movl	%eax, -56(%rbp)
	movq	$11, -32(%rbp)
	jmp	.L68
.L65:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$44, -32(%rbp)
	jmp	.L68
.L35:
	movl	-56(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$8, -32(%rbp)
	jmp	.L68
.L40:
	movl	-44(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$9, -32(%rbp)
	jmp	.L68
.L59:
	movl	-44(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$32, -32(%rbp)
	jmp	.L68
.L43:
	movl	-52(%rbp), %eax
	cmpl	%eax, -44(%rbp)
	jge	.L98
	movq	$28, -32(%rbp)
	jmp	.L68
.L98:
	movq	$17, -32(%rbp)
	jmp	.L68
.L37:
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-48(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$52, -32(%rbp)
	jmp	.L68
.L63:
	movl	-56(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$59, -32(%rbp)
	jmp	.L68
.L51:
	movq	$55, -32(%rbp)
	jmp	.L68
.L103:
	nop
.L68:
	jmp	.L100
.L102:
	call	__stack_chk_fail@PLT
.L101:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	main, .-main
	.section	.rodata
.LC10:
	.string	"%d\n"
.LC11:
	.string	"\nStack :"
.LC12:
	.string	"\nStack is empty"
	.text
	.globl	display
	.type	display, @function
display:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$0, -8(%rbp)
.L120:
	cmpq	$10, -8(%rbp)
	ja	.L121
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L107(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L107(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L107:
	.long	.L113-.L107
	.long	.L122-.L107
	.long	.L121-.L107
	.long	.L121-.L107
	.long	.L121-.L107
	.long	.L121-.L107
	.long	.L111-.L107
	.long	.L122-.L107
	.long	.L109-.L107
	.long	.L108-.L107
	.long	.L106-.L107
	.text
.L109:
	cmpl	$0, -12(%rbp)
	js	.L114
	movq	$9, -8(%rbp)
	jmp	.L116
.L114:
	movq	$1, -8(%rbp)
	jmp	.L116
.L108:
	movq	24(%rbp), %rdx
	movl	-12(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	subl	$1, -12(%rbp)
	movq	$8, -8(%rbp)
	jmp	.L116
.L111:
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	16(%rbp), %eax
	movl	%eax, -12(%rbp)
	movq	$8, -8(%rbp)
	jmp	.L116
.L106:
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$7, -8(%rbp)
	jmp	.L116
.L113:
	movl	16(%rbp), %eax
	cmpl	$-1, %eax
	jne	.L118
	movq	$10, -8(%rbp)
	jmp	.L116
.L118:
	movq	$6, -8(%rbp)
	jmp	.L116
.L121:
	nop
.L116:
	jmp	.L120
.L122:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	display, .-display
	.section	.rodata
.LC13:
	.string	"\nEnter the size of the :"
	.text
	.globl	create
	.type	create, @function
create:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$1, -24(%rbp)
.L129:
	cmpq	$2, -24(%rbp)
	je	.L124
	cmpq	$2, -24(%rbp)
	ja	.L132
	cmpq	$0, -24(%rbp)
	je	.L133
	cmpq	$1, -24(%rbp)
	jne	.L132
	movq	$2, -24(%rbp)
	jmp	.L127
.L124:
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-28(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	-40(%rbp), %rax
	movl	$-1, (%rax)
	movl	-28(%rbp), %edx
	movq	-40(%rbp), %rax
	movl	%edx, 16(%rax)
	movl	-28(%rbp), %eax
	cltq
	salq	$2, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -16(%rbp)
	movq	-40(%rbp), %rax
	movq	-16(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movq	$0, -24(%rbp)
	jmp	.L127
.L132:
	nop
.L127:
	jmp	.L129
.L133:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L131
	call	__stack_chk_fail@PLT
.L131:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	create, .-create
	.section	.rodata
.LC14:
	.string	"Stack underflow"
	.text
	.globl	pop
	.type	pop, @function
pop:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	$1, -8(%rbp)
.L147:
	cmpq	$6, -8(%rbp)
	ja	.L148
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L137(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L137(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L137:
	.long	.L142-.L137
	.long	.L141-.L137
	.long	.L148-.L137
	.long	.L140-.L137
	.long	.L139-.L137
	.long	.L138-.L137
	.long	.L136-.L137
	.text
.L139:
	cmpb	$0, -13(%rbp)
	je	.L143
	movq	$0, -8(%rbp)
	jmp	.L145
.L143:
	movq	$5, -8(%rbp)
	jmp	.L145
.L141:
	movq	-24(%rbp), %rax
	pushq	16(%rax)
	pushq	8(%rax)
	pushq	(%rax)
	call	isempty
	addq	$24, %rsp
	movb	%al, -13(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L145
.L140:
	movl	$-1, %eax
	jmp	.L146
.L136:
	movl	-12(%rbp), %eax
	jmp	.L146
.L138:
	movq	-24(%rbp), %rax
	movq	8(%rax), %rdx
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, -12(%rbp)
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	leal	-1(%rax), %edx
	movq	-24(%rbp), %rax
	movl	%edx, (%rax)
	movq	$6, -8(%rbp)
	jmp	.L145
.L142:
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$3, -8(%rbp)
	jmp	.L145
.L148:
	nop
.L145:
	jmp	.L147
.L146:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	pop, .-pop
	.globl	peek
	.type	peek, @function
peek:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$2, -8(%rbp)
.L161:
	cmpq	$4, -8(%rbp)
	ja	.L162
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L152(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L152(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L152:
	.long	.L156-.L152
	.long	.L155-.L152
	.long	.L154-.L152
	.long	.L153-.L152
	.long	.L151-.L152
	.text
.L151:
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$1, -8(%rbp)
	jmp	.L157
.L155:
	movl	$-1, %eax
	jmp	.L158
.L153:
	cmpb	$0, -9(%rbp)
	je	.L159
	movq	$4, -8(%rbp)
	jmp	.L157
.L159:
	movq	$0, -8(%rbp)
	jmp	.L157
.L156:
	movq	24(%rbp), %rdx
	movl	16(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	jmp	.L158
.L154:
	subq	$8, %rsp
	pushq	32(%rbp)
	pushq	24(%rbp)
	pushq	16(%rbp)
	call	isempty
	addq	$32, %rsp
	movb	%al, -9(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L157
.L162:
	nop
.L157:
	jmp	.L161
.L158:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	peek, .-peek
	.section	.rodata
.LC15:
	.string	"Stack overflow"
.LC16:
	.string	"Element pushed successfully"
	.text
	.globl	push
	.type	push, @function
push:
.LFB11:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movl	%esi, -28(%rbp)
	movq	$1, -8(%rbp)
.L176:
	cmpq	$6, -8(%rbp)
	ja	.L177
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L166(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L166(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L166:
	.long	.L178-.L166
	.long	.L170-.L166
	.long	.L169-.L166
	.long	.L168-.L166
	.long	.L177-.L166
	.long	.L167-.L166
	.long	.L178-.L166
	.text
.L170:
	movq	-24(%rbp), %rax
	pushq	16(%rax)
	pushq	8(%rax)
	pushq	(%rax)
	call	isfull
	addq	$24, %rsp
	movb	%al, -9(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L172
.L168:
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$0, -8(%rbp)
	jmp	.L172
.L167:
	cmpb	$0, -9(%rbp)
	je	.L174
	movq	$3, -8(%rbp)
	jmp	.L172
.L174:
	movq	$2, -8(%rbp)
	jmp	.L172
.L169:
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	leal	1(%rax), %edx
	movq	-24(%rbp), %rax
	movl	%edx, (%rax)
	movq	-24(%rbp), %rax
	movq	8(%rax), %rdx
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	cltq
	salq	$2, %rax
	addq	%rax, %rdx
	movl	-28(%rbp), %eax
	movl	%eax, (%rdx)
	leaq	.LC16(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	subq	$8, %rsp
	movq	-24(%rbp), %rax
	pushq	16(%rax)
	pushq	8(%rax)
	pushq	(%rax)
	call	display
	addq	$32, %rsp
	movq	$6, -8(%rbp)
	jmp	.L172
.L177:
	nop
.L172:
	jmp	.L176
.L178:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE11:
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
