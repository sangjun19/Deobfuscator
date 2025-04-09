	.file	"sreejithliterally_dslab_q_flatten.c"
	.text
	.globl	front
	.bss
	.align 4
	.type	front, @object
	.size	front, 4
front:
	.zero	4
	.globl	_TIG_IZ_y7WV_envp
	.align 8
	.type	_TIG_IZ_y7WV_envp, @object
	.size	_TIG_IZ_y7WV_envp, 8
_TIG_IZ_y7WV_envp:
	.zero	8
	.globl	_TIG_IZ_y7WV_argc
	.align 4
	.type	_TIG_IZ_y7WV_argc, @object
	.size	_TIG_IZ_y7WV_argc, 4
_TIG_IZ_y7WV_argc:
	.zero	4
	.globl	rear
	.align 4
	.type	rear, @object
	.size	rear, 4
rear:
	.zero	4
	.globl	Queue_array
	.align 16
	.type	Queue_array, @object
	.size	Queue_array, 20
Queue_array:
	.zero	20
	.globl	_TIG_IZ_y7WV_argv
	.align 8
	.type	_TIG_IZ_y7WV_argv, @object
	.size	_TIG_IZ_y7WV_argv, 8
_TIG_IZ_y7WV_argv:
	.zero	8
	.section	.rodata
	.align 8
.LC0:
	.string	"Element deleted from the Queue is %d\n"
.LC1:
	.string	"Queue underflow"
.LC2:
	.string	"\n\n"
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
	movq	$2, -8(%rbp)
.L19:
	cmpq	$9, -8(%rbp)
	ja	.L20
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
	.long	.L20-.L4
	.long	.L21-.L4
	.long	.L11-.L4
	.long	.L10-.L4
	.long	.L9-.L4
	.long	.L21-.L4
	.long	.L21-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L9:
	movl	front(%rip), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	Queue_array(%rip), %rax
	movl	(%rdx,%rax), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	front(%rip), %eax
	addl	$1, %eax
	movl	%eax, front(%rip)
	movq	$3, -8(%rbp)
	jmp	.L13
.L5:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$1, -8(%rbp)
	jmp	.L13
.L10:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$5, -8(%rbp)
	jmp	.L13
.L3:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$6, -8(%rbp)
	jmp	.L13
.L6:
	movl	front(%rip), %edx
	movl	rear(%rip), %eax
	cmpl	%eax, %edx
	jle	.L15
	movq	$8, -8(%rbp)
	jmp	.L13
.L15:
	movq	$4, -8(%rbp)
	jmp	.L13
.L11:
	movl	front(%rip), %eax
	cmpl	$-1, %eax
	jne	.L17
	movq	$9, -8(%rbp)
	jmp	.L13
.L17:
	movq	$7, -8(%rbp)
	jmp	.L13
.L20:
	nop
.L13:
	jmp	.L19
.L21:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	delete, .-delete
	.section	.rodata
.LC3:
	.string	"%d"
.LC4:
	.string	"Queue is empty"
.LC5:
	.string	"queue is : "
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
	movq	$8, -8(%rbp)
.L39:
	cmpq	$10, -8(%rbp)
	ja	.L40
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L25(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L25(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L25:
	.long	.L32-.L25
	.long	.L31-.L25
	.long	.L30-.L25
	.long	.L40-.L25
	.long	.L40-.L25
	.long	.L29-.L25
	.long	.L40-.L25
	.long	.L28-.L25
	.long	.L27-.L25
	.long	.L41-.L25
	.long	.L24-.L25
	.text
.L27:
	movl	front(%rip), %eax
	cmpl	$-1, %eax
	jne	.L33
	movq	$0, -8(%rbp)
	jmp	.L35
.L33:
	movq	$7, -8(%rbp)
	jmp	.L35
.L31:
	addl	$1, -12(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L35
.L29:
	movl	rear(%rip), %eax
	cmpl	%eax, -12(%rbp)
	jg	.L37
	movq	$1, -8(%rbp)
	jmp	.L35
.L37:
	movq	$10, -8(%rbp)
	jmp	.L35
.L24:
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	Queue_array(%rip), %rax
	movl	(%rdx,%rax), %eax
	movl	%eax, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$10, %edi
	call	putchar@PLT
	movq	$2, -8(%rbp)
	jmp	.L35
.L32:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$2, -8(%rbp)
	jmp	.L35
.L28:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	front(%rip), %eax
	movl	%eax, -12(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L35
.L30:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$9, -8(%rbp)
	jmp	.L35
.L40:
	nop
.L35:
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
.LC6:
	.string	"Wrong choice"
.LC7:
	.string	"1.Insert element to a Queue"
	.align 8
.LC8:
	.string	"2.Delete elements from a queue"
.LC9:
	.string	"3.Display all elements "
.LC10:
	.string	"4.Quit"
.LC11:
	.string	"5.Enter your choice"
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
	movl	$-1, front(%rip)
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
	leaq	Queue_array(%rip), %rax
	movl	$0, (%rdx,%rax)
	addl	$1, -20(%rbp)
.L45:
	cmpl	$4, -20(%rbp)
	jle	.L46
	nop
.L47:
	movq	$0, _TIG_IZ_y7WV_envp(%rip)
	nop
.L48:
	movq	$0, _TIG_IZ_y7WV_argv(%rip)
	nop
.L49:
	movl	$0, _TIG_IZ_y7WV_argc(%rip)
	nop
	nop
.L50:
.L51:
#APP
# 208 "sreejithliterally_dslab_q.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-y7WV--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_y7WV_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_y7WV_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_y7WV_envp(%rip)
	nop
	movq	$8, -16(%rbp)
.L69:
	cmpq	$13, -16(%rbp)
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
	.long	.L71-.L54
	.long	.L61-.L54
	.long	.L71-.L54
	.long	.L71-.L54
	.long	.L60-.L54
	.long	.L59-.L54
	.long	.L71-.L54
	.long	.L58-.L54
	.long	.L57-.L54
	.long	.L71-.L54
	.long	.L56-.L54
	.long	.L55-.L54
	.long	.L71-.L54
	.long	.L53-.L54
	.text
.L60:
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
	movq	$13, -16(%rbp)
	jmp	.L67
.L64:
	movq	$11, -16(%rbp)
	jmp	.L67
.L66:
	movq	$10, -16(%rbp)
	jmp	.L67
.L65:
	movq	$1, -16(%rbp)
	jmp	.L67
.L63:
	movq	$5, -16(%rbp)
	nop
.L67:
	jmp	.L68
.L57:
	movq	$7, -16(%rbp)
	jmp	.L68
.L61:
	call	insert
	movq	$7, -16(%rbp)
	jmp	.L68
.L55:
	call	display
	movq	$7, -16(%rbp)
	jmp	.L68
.L53:
	movl	$0, %edi
	call	exit@PLT
.L59:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$7, -16(%rbp)
	jmp	.L68
.L56:
	call	delete
	movq	$7, -16(%rbp)
	jmp	.L68
.L58:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$4, -16(%rbp)
	jmp	.L68
.L71:
	nop
.L68:
	jmp	.L69
	.cfi_endproc
.LFE5:
	.size	main, .-main
	.section	.rodata
.LC12:
	.string	"Insert the element"
.LC13:
	.string	"Queue overflow"
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
	subq	$32, %rsp
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$4, -16(%rbp)
.L88:
	cmpq	$7, -16(%rbp)
	ja	.L91
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L75(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L75(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L75:
	.long	.L81-.L75
	.long	.L91-.L75
	.long	.L80-.L75
	.long	.L92-.L75
	.long	.L78-.L75
	.long	.L77-.L75
	.long	.L76-.L75
	.long	.L74-.L75
	.text
.L78:
	movl	rear(%rip), %eax
	cmpl	$4, %eax
	jne	.L82
	movq	$5, -16(%rbp)
	jmp	.L84
.L82:
	movq	$2, -16(%rbp)
	jmp	.L84
.L76:
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-20(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	rear(%rip), %eax
	addl	$1, %eax
	movl	%eax, rear(%rip)
	movl	rear(%rip), %edx
	movl	-20(%rbp), %eax
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	leaq	Queue_array(%rip), %rdx
	movl	%eax, (%rcx,%rdx)
	movq	$0, -16(%rbp)
	jmp	.L84
.L77:
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$0, -16(%rbp)
	jmp	.L84
.L81:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$3, -16(%rbp)
	jmp	.L84
.L74:
	movl	$0, front(%rip)
	movq	$6, -16(%rbp)
	jmp	.L84
.L80:
	movl	front(%rip), %eax
	cmpl	$-1, %eax
	jne	.L86
	movq	$7, -16(%rbp)
	jmp	.L84
.L86:
	movq	$6, -16(%rbp)
	jmp	.L84
.L91:
	nop
.L84:
	jmp	.L88
.L92:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L90
	call	__stack_chk_fail@PLT
.L90:
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
