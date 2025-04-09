	.file	"JoyceMK_lab-works_8_1_queue_flatten.c"
	.text
	.globl	front
	.bss
	.align 4
	.type	front, @object
	.size	front, 4
front:
	.zero	4
	.globl	_TIG_IZ_HpnX_argc
	.align 4
	.type	_TIG_IZ_HpnX_argc, @object
	.size	_TIG_IZ_HpnX_argc, 4
_TIG_IZ_HpnX_argc:
	.zero	4
	.globl	rear
	.align 4
	.type	rear, @object
	.size	rear, 4
rear:
	.zero	4
	.globl	_TIG_IZ_HpnX_envp
	.align 8
	.type	_TIG_IZ_HpnX_envp, @object
	.size	_TIG_IZ_HpnX_envp, 8
_TIG_IZ_HpnX_envp:
	.zero	8
	.globl	queue
	.align 16
	.type	queue, @object
	.size	queue, 20
queue:
	.zero	20
	.globl	_TIG_IZ_HpnX_argv
	.align 8
	.type	_TIG_IZ_HpnX_argv, @object
	.size	_TIG_IZ_HpnX_argv, 8
_TIG_IZ_HpnX_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"Queue is full"
.LC1:
	.string	"Enter element : "
.LC2:
	.string	"%d"
	.text
	.globl	enqueue
	.type	enqueue, @function
enqueue:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$9, -8(%rbp)
.L18:
	cmpq	$9, -8(%rbp)
	ja	.L19
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
	.long	.L11-.L4
	.long	.L20-.L4
	.long	.L9-.L4
	.long	.L8-.L4
	.long	.L20-.L4
	.long	.L19-.L4
	.long	.L6-.L4
	.long	.L19-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L5:
	movl	$0, rear(%rip)
	movl	rear(%rip), %eax
	movl	%eax, front(%rip)
	movq	$6, -8(%rbp)
	jmp	.L13
.L8:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$4, -8(%rbp)
	jmp	.L13
.L3:
	movl	rear(%rip), %eax
	cmpl	$4, %eax
	jne	.L14
	movq	$3, -8(%rbp)
	jmp	.L13
.L14:
	movq	$0, -8(%rbp)
	jmp	.L13
.L6:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	rear(%rip), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	queue(%rip), %rax
	addq	%rdx, %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$1, -8(%rbp)
	jmp	.L13
.L11:
	movl	rear(%rip), %eax
	cmpl	$-1, %eax
	jne	.L16
	movq	$8, -8(%rbp)
	jmp	.L13
.L16:
	movq	$2, -8(%rbp)
	jmp	.L13
.L9:
	movl	rear(%rip), %eax
	addl	$1, %eax
	movl	%eax, rear(%rip)
	movq	$6, -8(%rbp)
	jmp	.L13
.L19:
	nop
.L13:
	jmp	.L18
.L20:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	enqueue, .-enqueue
	.section	.rodata
.LC3:
	.string	"Queue is empty"
.LC4:
	.string	"\n\n%d Dequeued\n"
	.text
	.globl	dequeue
	.type	dequeue, @function
dequeue:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$1, -8(%rbp)
.L38:
	cmpq	$7, -8(%rbp)
	ja	.L39
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L24(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L24(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L24:
	.long	.L40-.L24
	.long	.L30-.L24
	.long	.L29-.L24
	.long	.L28-.L24
	.long	.L27-.L24
	.long	.L26-.L24
	.long	.L40-.L24
	.long	.L23-.L24
	.text
.L27:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$6, -8(%rbp)
	jmp	.L32
.L30:
	movl	front(%rip), %eax
	cmpl	$-1, %eax
	jne	.L33
	movq	$4, -8(%rbp)
	jmp	.L32
.L33:
	movq	$2, -8(%rbp)
	jmp	.L32
.L28:
	movl	front(%rip), %eax
	addl	$1, %eax
	movl	%eax, front(%rip)
	movq	$0, -8(%rbp)
	jmp	.L32
.L26:
	movl	front(%rip), %edx
	movl	rear(%rip), %eax
	cmpl	%eax, %edx
	jne	.L36
	movq	$7, -8(%rbp)
	jmp	.L32
.L36:
	movq	$3, -8(%rbp)
	jmp	.L32
.L23:
	movl	$-1, rear(%rip)
	movl	rear(%rip), %eax
	movl	%eax, front(%rip)
	movq	$0, -8(%rbp)
	jmp	.L32
.L29:
	movl	front(%rip), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	queue(%rip), %rax
	movl	(%rdx,%rax), %eax
	movl	%eax, %esi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$5, -8(%rbp)
	jmp	.L32
.L39:
	nop
.L32:
	jmp	.L38
.L40:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	dequeue, .-dequeue
	.section	.rodata
	.align 8
.LC5:
	.string	"\n1. Enqueue \n2. Dequeue \n3. Exit \nEnter your option : "
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
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movl	$-1, rear(%rip)
	nop
.L42:
	movl	$-1, front(%rip)
	nop
.L43:
	movl	$0, -20(%rbp)
	jmp	.L44
.L45:
	movl	-20(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	queue(%rip), %rax
	movl	$0, (%rdx,%rax)
	addl	$1, -20(%rbp)
.L44:
	cmpl	$4, -20(%rbp)
	jle	.L45
	nop
.L46:
	movq	$0, _TIG_IZ_HpnX_envp(%rip)
	nop
.L47:
	movq	$0, _TIG_IZ_HpnX_argv(%rip)
	nop
.L48:
	movl	$0, _TIG_IZ_HpnX_argc(%rip)
	nop
	nop
.L49:
.L50:
#APP
# 105 "JoyceMK_lab-works_8_1_queue.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-HpnX--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_HpnX_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_HpnX_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_HpnX_envp(%rip)
	nop
	movq	$4, -16(%rbp)
.L67:
	cmpq	$9, -16(%rbp)
	ja	.L70
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L53(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L53(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L53:
	.long	.L70-.L53
	.long	.L59-.L53
	.long	.L58-.L53
	.long	.L70-.L53
	.long	.L57-.L53
	.long	.L70-.L53
	.long	.L56-.L53
	.long	.L55-.L53
	.long	.L54-.L53
	.long	.L52-.L53
	.text
.L57:
	movq	$7, -16(%rbp)
	jmp	.L60
.L54:
	movl	-24(%rbp), %eax
	cmpl	$3, %eax
	je	.L61
	cmpl	$3, %eax
	jg	.L62
	cmpl	$1, %eax
	je	.L63
	cmpl	$2, %eax
	je	.L64
	jmp	.L62
.L61:
	movq	$2, -16(%rbp)
	jmp	.L65
.L64:
	movq	$6, -16(%rbp)
	jmp	.L65
.L63:
	movq	$9, -16(%rbp)
	jmp	.L65
.L62:
	movq	$1, -16(%rbp)
	nop
.L65:
	jmp	.L60
.L59:
	movq	$7, -16(%rbp)
	jmp	.L60
.L52:
	call	enqueue
	movq	$7, -16(%rbp)
	jmp	.L60
.L56:
	call	dequeue
	movq	$7, -16(%rbp)
	jmp	.L60
.L55:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$8, -16(%rbp)
	jmp	.L60
.L58:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L68
	jmp	.L69
.L70:
	nop
.L60:
	jmp	.L67
.L69:
	call	__stack_chk_fail@PLT
.L68:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
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
