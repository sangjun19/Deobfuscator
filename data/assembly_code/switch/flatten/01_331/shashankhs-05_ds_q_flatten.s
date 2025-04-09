	.file	"shashankhs-05_ds_q_flatten.c"
	.text
	.globl	front
	.bss
	.align 4
	.type	front, @object
	.size	front, 4
front:
	.zero	4
	.globl	_TIG_IZ_lBrb_argv
	.align 8
	.type	_TIG_IZ_lBrb_argv, @object
	.size	_TIG_IZ_lBrb_argv, 8
_TIG_IZ_lBrb_argv:
	.zero	8
	.globl	rear
	.align 4
	.type	rear, @object
	.size	rear, 4
rear:
	.zero	4
	.globl	_TIG_IZ_lBrb_argc
	.align 4
	.type	_TIG_IZ_lBrb_argc, @object
	.size	_TIG_IZ_lBrb_argc, 4
_TIG_IZ_lBrb_argc:
	.zero	4
	.globl	_TIG_IZ_lBrb_envp
	.align 8
	.type	_TIG_IZ_lBrb_envp, @object
	.size	_TIG_IZ_lBrb_envp, 8
_TIG_IZ_lBrb_envp:
	.zero	8
	.globl	a
	.align 8
	.type	a, @object
	.size	a, 12
a:
	.zero	12
	.section	.rodata
.LC0:
	.string	"\nUnderflow"
.LC1:
	.string	"\nDeleted element=%d"
	.text
	.globl	delete
	.type	delete, @function
delete:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$6, -8(%rbp)
.L16:
	cmpq	$6, -8(%rbp)
	ja	.L17
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L4(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L4(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L4:
	.long	.L9-.L4
	.long	.L8-.L4
	.long	.L18-.L4
	.long	.L6-.L4
	.long	.L17-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L8:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$2, -8(%rbp)
	jmp	.L10
.L6:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$2, -8(%rbp)
	jmp	.L10
.L3:
	movl	rear(%rip), %eax
	cmpl	$-1, %eax
	jne	.L11
	movq	$3, -8(%rbp)
	jmp	.L10
.L11:
	movq	$0, -8(%rbp)
	jmp	.L10
.L5:
	movl	front(%rip), %eax
	movl	%eax, -12(%rbp)
	movl	front(%rip), %eax
	addl	$1, %eax
	movl	%eax, front(%rip)
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	a(%rip), %rax
	movl	(%rdx,%rax), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$2, -8(%rbp)
	jmp	.L10
.L9:
	movl	front(%rip), %eax
	cmpl	$3, %eax
	jne	.L13
	movq	$1, -8(%rbp)
	jmp	.L10
.L13:
	movq	$5, -8(%rbp)
	jmp	.L10
.L17:
	nop
.L10:
	jmp	.L16
.L18:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	delete, .-delete
	.section	.rodata
.LC2:
	.string	"The elements are: "
.LC3:
	.string	"%d\t"
	.text
	.globl	display
	.type	display, @function
display:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$11, -8(%rbp)
.L39:
	cmpq	$11, -8(%rbp)
	ja	.L40
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L22(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L22(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L22:
	.long	.L30-.L22
	.long	.L40-.L22
	.long	.L29-.L22
	.long	.L40-.L22
	.long	.L28-.L22
	.long	.L41-.L22
	.long	.L26-.L22
	.long	.L25-.L22
	.long	.L40-.L22
	.long	.L24-.L22
	.long	.L23-.L22
	.long	.L21-.L22
	.text
.L28:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$10, -8(%rbp)
	jmp	.L31
.L21:
	movl	rear(%rip), %eax
	cmpl	$-1, %eax
	jne	.L32
	movq	$4, -8(%rbp)
	jmp	.L31
.L32:
	movq	$6, -8(%rbp)
	jmp	.L31
.L24:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$10, -8(%rbp)
	jmp	.L31
.L26:
	movl	front(%rip), %eax
	cmpl	$3, %eax
	jne	.L34
	movq	$0, -8(%rbp)
	jmp	.L31
.L34:
	movq	$9, -8(%rbp)
	jmp	.L31
.L23:
	movl	front(%rip), %eax
	movl	%eax, -12(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L31
.L30:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$10, -8(%rbp)
	jmp	.L31
.L25:
	movl	rear(%rip), %eax
	cmpl	%eax, -12(%rbp)
	jg	.L37
	movq	$2, -8(%rbp)
	jmp	.L31
.L37:
	movq	$5, -8(%rbp)
	jmp	.L31
.L29:
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	a(%rip), %rax
	movl	(%rdx,%rax), %eax
	movl	%eax, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -12(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L31
.L40:
	nop
.L31:
	jmp	.L39
.L41:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	display, .-display
	.section	.rodata
	.align 8
.LC4:
	.string	"\nEnter choice\n1.Insert\n2.Delete\n3.Display\n4.Exit"
.LC5:
	.string	"%d"
.LC6:
	.string	"Invalid"
	.text
	.globl	main
	.type	main, @function
main:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movl	$0, front(%rip)
	nop
.L43:
	movl	$-1, rear(%rip)
	nop
.L44:
	movl	$0, -20(%rbp)
	jmp	.L45
.L46:
	movl	-20(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	a(%rip), %rax
	movl	$0, (%rdx,%rax)
	addl	$1, -20(%rbp)
.L45:
	cmpl	$2, -20(%rbp)
	jle	.L46
	nop
.L47:
	movq	$0, _TIG_IZ_lBrb_envp(%rip)
	nop
.L48:
	movq	$0, _TIG_IZ_lBrb_argv(%rip)
	nop
.L49:
	movl	$0, _TIG_IZ_lBrb_argc(%rip)
	nop
	nop
.L50:
.L51:
#APP
# 147 "shashankhs-05_ds_q.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-lBrb--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_lBrb_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_lBrb_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_lBrb_envp(%rip)
	nop
	movq	$1, -16(%rbp)
.L69:
	cmpq	$12, -16(%rbp)
	ja	.L71
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L54(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L54(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L54:
	.long	.L61-.L54
	.long	.L60-.L54
	.long	.L71-.L54
	.long	.L59-.L54
	.long	.L71-.L54
	.long	.L71-.L54
	.long	.L58-.L54
	.long	.L71-.L54
	.long	.L57-.L54
	.long	.L71-.L54
	.long	.L56-.L54
	.long	.L55-.L54
	.long	.L53-.L54
	.text
.L53:
	movl	-24(%rbp), %eax
	cmpl	$4, %eax
	je	.L62
	cmpl	$4, %eax
	jg	.L63
	cmpl	$3, %eax
	je	.L64
	cmpl	$3, %eax
	jg	.L63
	cmpl	$1, %eax
	je	.L65
	cmpl	$2, %eax
	je	.L66
	jmp	.L63
.L62:
	movq	$0, -16(%rbp)
	jmp	.L67
.L64:
	movq	$10, -16(%rbp)
	jmp	.L67
.L66:
	movq	$11, -16(%rbp)
	jmp	.L67
.L65:
	movq	$3, -16(%rbp)
	jmp	.L67
.L63:
	movq	$6, -16(%rbp)
	nop
.L67:
	jmp	.L68
.L57:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$12, -16(%rbp)
	jmp	.L68
.L60:
	movq	$8, -16(%rbp)
	jmp	.L68
.L59:
	call	insert
	movq	$8, -16(%rbp)
	jmp	.L68
.L55:
	call	delete
	movq	$8, -16(%rbp)
	jmp	.L68
.L58:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$8, -16(%rbp)
	jmp	.L68
.L56:
	call	display
	movq	$8, -16(%rbp)
	jmp	.L68
.L61:
	movl	$0, %edi
	call	exit@PLT
.L71:
	nop
.L68:
	jmp	.L69
	.cfi_endproc
.LFE5:
	.size	main, .-main
	.section	.rodata
.LC7:
	.string	"Overflow"
	.align 8
.LC8:
	.string	"\nEnter element to be inserted:"
	.text
	.globl	insert
	.type	insert, @function
insert:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$3, -8(%rbp)
.L82:
	cmpq	$4, -8(%rbp)
	je	.L83
	cmpq	$4, -8(%rbp)
	ja	.L84
	cmpq	$3, -8(%rbp)
	je	.L75
	cmpq	$3, -8(%rbp)
	ja	.L84
	cmpq	$1, -8(%rbp)
	je	.L76
	cmpq	$2, -8(%rbp)
	je	.L77
	jmp	.L84
.L76:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$4, -8(%rbp)
	jmp	.L79
.L75:
	movl	rear(%rip), %eax
	cmpl	$2, %eax
	jne	.L80
	movq	$1, -8(%rbp)
	jmp	.L79
.L80:
	movq	$2, -8(%rbp)
	jmp	.L79
.L77:
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	rear(%rip), %eax
	addl	$1, %eax
	movl	%eax, rear(%rip)
	movl	rear(%rip), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	a(%rip), %rax
	addq	%rdx, %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$4, -8(%rbp)
	jmp	.L79
.L84:
	nop
.L79:
	jmp	.L82
.L83:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	insert, .-insert
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
