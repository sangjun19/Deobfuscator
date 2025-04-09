	.file	"DemonicAK_DSA_COURSE_LAB_5_flatten.c"
	.text
	.globl	front
	.bss
	.align 8
	.type	front, @object
	.size	front, 8
front:
	.zero	8
	.globl	rear
	.align 8
	.type	rear, @object
	.size	rear, 8
rear:
	.zero	8
	.globl	_TIG_IZ_Svy5_argc
	.align 4
	.type	_TIG_IZ_Svy5_argc, @object
	.size	_TIG_IZ_Svy5_argc, 4
_TIG_IZ_Svy5_argc:
	.zero	4
	.globl	_TIG_IZ_Svy5_envp
	.align 8
	.type	_TIG_IZ_Svy5_envp, @object
	.size	_TIG_IZ_Svy5_envp, 8
_TIG_IZ_Svy5_envp:
	.zero	8
	.globl	_TIG_IZ_Svy5_argv
	.align 8
	.type	_TIG_IZ_Svy5_argv, @object
	.size	_TIG_IZ_Svy5_argv, 8
_TIG_IZ_Svy5_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"\nUNDERFLOW"
	.text
	.globl	del
	.type	del, @function
del:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$3, -16(%rbp)
.L13:
	cmpq	$5, -16(%rbp)
	ja	.L14
	movq	-16(%rbp), %rax
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
	.long	.L8-.L4
	.long	.L15-.L4
	.long	.L14-.L4
	.long	.L6-.L4
	.long	.L15-.L4
	.long	.L3-.L4
	.text
.L6:
	movq	front(%rip), %rax
	testq	%rax, %rax
	jne	.L10
	movq	$0, -16(%rbp)
	jmp	.L12
.L10:
	movq	$5, -16(%rbp)
	jmp	.L12
.L3:
	movq	front(%rip), %rax
	movq	%rax, -8(%rbp)
	movq	front(%rip), %rax
	movq	8(%rax), %rax
	movq	%rax, front(%rip)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$4, -16(%rbp)
	jmp	.L12
.L8:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$1, -16(%rbp)
	jmp	.L12
.L14:
	nop
.L12:
	jmp	.L13
.L15:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	del, .-del
	.section	.rodata
.LC1:
	.string	"Elements of Queue:"
.LC2:
	.string	"\nEmpty queue"
.LC3:
	.string	" %d"
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
	movq	$9, -8(%rbp)
.L33:
	cmpq	$10, -8(%rbp)
	ja	.L34
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
	.long	.L26-.L19
	.long	.L34-.L19
	.long	.L34-.L19
	.long	.L25-.L19
	.long	.L24-.L19
	.long	.L34-.L19
	.long	.L35-.L19
	.long	.L22-.L19
	.long	.L21-.L19
	.long	.L20-.L19
	.long	.L18-.L19
	.text
.L24:
	movl	$10, %edi
	call	putchar@PLT
	movq	$6, -8(%rbp)
	jmp	.L27
.L21:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$10, -8(%rbp)
	jmp	.L27
.L25:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$6, -8(%rbp)
	jmp	.L27
.L20:
	movq	front(%rip), %rax
	movq	%rax, -16(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L27
.L18:
	cmpq	$0, -16(%rbp)
	je	.L29
	movq	$0, -8(%rbp)
	jmp	.L27
.L29:
	movq	$4, -8(%rbp)
	jmp	.L27
.L26:
	movq	-16(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-16(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, -16(%rbp)
	movq	$10, -8(%rbp)
	jmp	.L27
.L22:
	movq	front(%rip), %rax
	testq	%rax, %rax
	jne	.L31
	movq	$3, -8(%rbp)
	jmp	.L27
.L31:
	movq	$8, -8(%rbp)
	jmp	.L27
.L34:
	nop
.L27:
	jmp	.L33
.L35:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	display, .-display
	.section	.rodata
.LC4:
	.string	"Enter value:"
.LC5:
	.string	"%d"
.LC6:
	.string	"\nOVERFLOW"
	.text
	.globl	ins
	.type	ins, @function
ins:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$2, -24(%rbp)
.L55:
	cmpq	$12, -24(%rbp)
	ja	.L58
	movq	-24(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L39(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L39(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L39:
	.long	.L48-.L39
	.long	.L58-.L39
	.long	.L47-.L39
	.long	.L46-.L39
	.long	.L45-.L39
	.long	.L44-.L39
	.long	.L43-.L39
	.long	.L58-.L39
	.long	.L42-.L39
	.long	.L58-.L39
	.long	.L59-.L39
	.long	.L59-.L39
	.long	.L38-.L39
	.text
.L45:
	movq	-32(%rbp), %rax
	movq	%rax, front(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, rear(%rip)
	movq	front(%rip), %rax
	movq	$0, 8(%rax)
	movq	rear(%rip), %rax
	movq	$0, 8(%rax)
	movq	$11, -24(%rbp)
	jmp	.L49
.L38:
	cmpq	$0, -32(%rbp)
	jne	.L50
	movq	$0, -24(%rbp)
	jmp	.L49
.L50:
	movq	$3, -24(%rbp)
	jmp	.L49
.L42:
	movq	front(%rip), %rax
	testq	%rax, %rax
	jne	.L52
	movq	$4, -24(%rbp)
	jmp	.L49
.L52:
	movq	$6, -24(%rbp)
	jmp	.L49
.L46:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-36(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-36(%rbp), %edx
	movq	-32(%rbp), %rax
	movl	%edx, (%rax)
	movq	$8, -24(%rbp)
	jmp	.L49
.L43:
	movq	rear(%rip), %rax
	movq	-32(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movq	-32(%rbp), %rax
	movq	%rax, rear(%rip)
	movq	rear(%rip), %rax
	movq	$0, 8(%rax)
	movq	$11, -24(%rbp)
	jmp	.L49
.L44:
	movl	$16, %edi
	call	malloc@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -32(%rbp)
	movq	$12, -24(%rbp)
	jmp	.L49
.L48:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$10, -24(%rbp)
	jmp	.L49
.L47:
	movq	$5, -24(%rbp)
	jmp	.L49
.L58:
	nop
.L49:
	jmp	.L55
.L59:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L57
	call	__stack_chk_fail@PLT
.L57:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	ins, .-ins
	.section	.rodata
	.align 8
.LC7:
	.string	"\n1.Insert an element\n2.Delete an element\n3.Display the queue\n4.Exit"
.LC8:
	.string	"Enter your choice:"
.LC9:
	.string	"\nEnter valid choice!!"
	.text
	.globl	main
	.type	main, @function
main:
.LFB9:
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
	movq	$0, rear(%rip)
	nop
.L61:
	movq	$0, front(%rip)
	nop
.L62:
	movq	$0, _TIG_IZ_Svy5_envp(%rip)
	nop
.L63:
	movq	$0, _TIG_IZ_Svy5_argv(%rip)
	nop
.L64:
	movl	$0, _TIG_IZ_Svy5_argc(%rip)
	nop
	nop
.L65:
.L66:
#APP
# 117 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-Svy5--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_Svy5_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_Svy5_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_Svy5_envp(%rip)
	nop
	movq	$6, -16(%rbp)
.L89:
	cmpq	$13, -16(%rbp)
	ja	.L92
	movq	-16(%rbp), %rax
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
	.long	.L78-.L69
	.long	.L77-.L69
	.long	.L92-.L69
	.long	.L92-.L69
	.long	.L92-.L69
	.long	.L92-.L69
	.long	.L76-.L69
	.long	.L93-.L69
	.long	.L74-.L69
	.long	.L73-.L69
	.long	.L72-.L69
	.long	.L71-.L69
	.long	.L70-.L69
	.long	.L68-.L69
	.text
.L70:
	movl	$0, %edi
	call	exit@PLT
.L74:
	call	del
	movq	$9, -16(%rbp)
	jmp	.L79
.L77:
	call	display
	movq	$9, -16(%rbp)
	jmp	.L79
.L71:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-20(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$0, -16(%rbp)
	jmp	.L79
.L73:
	movl	-20(%rbp), %eax
	cmpl	$4, %eax
	je	.L80
	movq	$11, -16(%rbp)
	jmp	.L79
.L80:
	movq	$7, -16(%rbp)
	jmp	.L79
.L68:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$9, -16(%rbp)
	jmp	.L79
.L76:
	movq	$9, -16(%rbp)
	jmp	.L79
.L72:
	call	ins
	movq	$9, -16(%rbp)
	jmp	.L79
.L78:
	movl	-20(%rbp), %eax
	cmpl	$4, %eax
	je	.L82
	cmpl	$4, %eax
	jg	.L83
	cmpl	$3, %eax
	je	.L84
	cmpl	$3, %eax
	jg	.L83
	cmpl	$1, %eax
	je	.L85
	cmpl	$2, %eax
	je	.L86
	jmp	.L83
.L82:
	movq	$12, -16(%rbp)
	jmp	.L87
.L84:
	movq	$1, -16(%rbp)
	jmp	.L87
.L86:
	movq	$8, -16(%rbp)
	jmp	.L87
.L85:
	movq	$10, -16(%rbp)
	jmp	.L87
.L83:
	movq	$13, -16(%rbp)
	nop
.L87:
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
.LFE9:
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
