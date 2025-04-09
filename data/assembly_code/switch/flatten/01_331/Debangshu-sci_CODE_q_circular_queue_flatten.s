	.file	"Debangshu-sci_CODE_q_circular_queue_flatten.c"
	.text
	.globl	front
	.bss
	.align 4
	.type	front, @object
	.size	front, 4
front:
	.zero	4
	.globl	_TIG_IZ_9qdq_argc
	.align 4
	.type	_TIG_IZ_9qdq_argc, @object
	.size	_TIG_IZ_9qdq_argc, 4
_TIG_IZ_9qdq_argc:
	.zero	4
	.globl	rear
	.align 4
	.type	rear, @object
	.size	rear, 4
rear:
	.zero	4
	.globl	_TIG_IZ_9qdq_envp
	.align 8
	.type	_TIG_IZ_9qdq_envp, @object
	.size	_TIG_IZ_9qdq_envp, 8
_TIG_IZ_9qdq_envp:
	.zero	8
	.globl	_TIG_IZ_9qdq_argv
	.align 8
	.type	_TIG_IZ_9qdq_argv, @object
	.size	_TIG_IZ_9qdq_argv, 8
_TIG_IZ_9qdq_argv:
	.zero	8
	.globl	queue
	.align 16
	.type	queue, @object
	.size	queue, 24
queue:
	.zero	24
	.section	.rodata
.LC0:
	.string	"Queue is overflow.."
	.text
	.globl	enqueue
	.type	enqueue, @function
enqueue:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, -20(%rbp)
	movq	$6, -8(%rbp)
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
	.long	.L10-.L4
	.long	.L20-.L4
	.long	.L9-.L4
	.long	.L21-.L4
	.long	.L20-.L4
	.long	.L20-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L5:
	movl	$0, front(%rip)
	movl	$0, rear(%rip)
	movl	rear(%rip), %eax
	cltq
	leaq	0(,%rax,4), %rcx
	leaq	queue(%rip), %rdx
	movl	-20(%rbp), %eax
	movl	%eax, (%rcx,%rdx)
	movq	$3, -8(%rbp)
	jmp	.L11
.L3:
	movl	rear(%rip), %eax
	cmpl	$-1, %eax
	jne	.L13
	movq	$8, -8(%rbp)
	jmp	.L11
.L13:
	movq	$7, -8(%rbp)
	jmp	.L11
.L7:
	movl	front(%rip), %eax
	cmpl	$-1, %eax
	jne	.L15
	movq	$9, -8(%rbp)
	jmp	.L11
.L15:
	movq	$7, -8(%rbp)
	jmp	.L11
.L10:
	movl	rear(%rip), %eax
	leal	1(%rax), %ecx
	movslq	%ecx, %rax
	imulq	$715827883, %rax, %rax
	shrq	$32, %rax
	movl	%ecx, %esi
	sarl	$31, %esi
	movl	%eax, %edx
	subl	%esi, %edx
	movl	%edx, %eax
	addl	%eax, %eax
	addl	%edx, %eax
	addl	%eax, %eax
	subl	%eax, %ecx
	movl	%ecx, %edx
	movl	%edx, rear(%rip)
	movl	rear(%rip), %eax
	cltq
	leaq	0(,%rax,4), %rcx
	leaq	queue(%rip), %rdx
	movl	-20(%rbp), %eax
	movl	%eax, (%rcx,%rdx)
	movq	$3, -8(%rbp)
	jmp	.L11
.L6:
	movl	rear(%rip), %eax
	leal	1(%rax), %ecx
	movslq	%ecx, %rax
	imulq	$715827883, %rax, %rax
	shrq	$32, %rax
	movl	%ecx, %esi
	sarl	$31, %esi
	movl	%eax, %edx
	subl	%esi, %edx
	movl	%edx, %eax
	addl	%eax, %eax
	addl	%edx, %eax
	addl	%eax, %eax
	subl	%eax, %ecx
	movl	%ecx, %edx
	movl	front(%rip), %eax
	cmpl	%eax, %edx
	jne	.L17
	movq	$2, -8(%rbp)
	jmp	.L11
.L17:
	movq	$0, -8(%rbp)
	jmp	.L11
.L9:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -8(%rbp)
	jmp	.L11
.L20:
	nop
.L11:
	jmp	.L19
.L21:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	enqueue, .-enqueue
	.section	.rodata
.LC1:
	.string	"\nThe dequeued element is %d"
.LC2:
	.string	"\nQueue is underflow.."
	.text
	.globl	dequeue
	.type	dequeue, @function
dequeue:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$7, -8(%rbp)
.L40:
	cmpq	$9, -8(%rbp)
	ja	.L42
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
	.long	.L31-.L25
	.long	.L30-.L25
	.long	.L42-.L25
	.long	.L29-.L25
	.long	.L28-.L25
	.long	.L42-.L25
	.long	.L42-.L25
	.long	.L27-.L25
	.long	.L26-.L25
	.long	.L24-.L25
	.text
.L28:
	movl	front(%rip), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	queue(%rip), %rax
	movl	(%rdx,%rax), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$-1, front(%rip)
	movl	$-1, rear(%rip)
	movq	$3, -8(%rbp)
	jmp	.L32
.L26:
	movl	rear(%rip), %eax
	cmpl	$-1, %eax
	jne	.L33
	movq	$9, -8(%rbp)
	jmp	.L32
.L33:
	movq	$1, -8(%rbp)
	jmp	.L32
.L30:
	movl	front(%rip), %edx
	movl	rear(%rip), %eax
	cmpl	%eax, %edx
	jne	.L35
	movq	$4, -8(%rbp)
	jmp	.L32
.L35:
	movq	$0, -8(%rbp)
	jmp	.L32
.L29:
	movl	$0, %eax
	jmp	.L41
.L24:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -8(%rbp)
	jmp	.L32
.L31:
	movl	front(%rip), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	queue(%rip), %rax
	movl	(%rdx,%rax), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	front(%rip), %eax
	leal	1(%rax), %ecx
	movslq	%ecx, %rax
	imulq	$715827883, %rax, %rax
	shrq	$32, %rax
	movl	%ecx, %esi
	sarl	$31, %esi
	movl	%eax, %edx
	subl	%esi, %edx
	movl	%edx, %eax
	addl	%eax, %eax
	addl	%edx, %eax
	addl	%eax, %eax
	subl	%eax, %ecx
	movl	%ecx, %edx
	movl	%edx, front(%rip)
	movq	$3, -8(%rbp)
	jmp	.L32
.L27:
	movl	front(%rip), %eax
	cmpl	$-1, %eax
	jne	.L38
	movq	$8, -8(%rbp)
	jmp	.L32
.L38:
	movq	$1, -8(%rbp)
	jmp	.L32
.L42:
	nop
.L32:
	jmp	.L40
.L41:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	dequeue, .-dequeue
	.section	.rodata
.LC3:
	.string	"\n Press 1: Insert an element"
.LC4:
	.string	"\nPress 2: Delete an element"
.LC5:
	.string	"\nPress 3: Display the element"
.LC6:
	.string	"\nEnter your choice"
.LC7:
	.string	"%d"
	.align 8
.LC8:
	.string	"Enter the element which is to be inserted"
	.text
	.globl	main
	.type	main, @function
main:
.LFB7:
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
.L44:
	movl	$-1, front(%rip)
	nop
.L45:
	movl	$0, -20(%rbp)
	jmp	.L46
.L47:
	movl	-20(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	queue(%rip), %rax
	movl	$0, (%rdx,%rax)
	addl	$1, -20(%rbp)
.L46:
	cmpl	$5, -20(%rbp)
	jle	.L47
	nop
.L48:
	movq	$0, _TIG_IZ_9qdq_envp(%rip)
	nop
.L49:
	movq	$0, _TIG_IZ_9qdq_argv(%rip)
	nop
.L50:
	movl	$0, _TIG_IZ_9qdq_argc(%rip)
	nop
	nop
.L51:
.L52:
#APP
# 130 "Debangshu-sci_CODE_q_circular_queue.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-9qdq--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_9qdq_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_9qdq_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_9qdq_envp(%rip)
	nop
	movq	$7, -16(%rbp)
.L76:
	cmpq	$14, -16(%rbp)
	ja	.L79
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L55(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L55(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L55:
	.long	.L64-.L55
	.long	.L63-.L55
	.long	.L62-.L55
	.long	.L61-.L55
	.long	.L60-.L55
	.long	.L59-.L55
	.long	.L58-.L55
	.long	.L57-.L55
	.long	.L79-.L55
	.long	.L56-.L55
	.long	.L79-.L55
	.long	.L79-.L55
	.long	.L79-.L55
	.long	.L79-.L55
	.long	.L54-.L55
	.text
.L60:
	call	dequeue
	movq	$3, -16(%rbp)
	jmp	.L65
.L54:
	movq	$3, -16(%rbp)
	jmp	.L65
.L63:
	call	display
	movq	$14, -16(%rbp)
	jmp	.L65
.L61:
	movl	-28(%rbp), %eax
	cmpl	$3, %eax
	jg	.L66
	movq	$9, -16(%rbp)
	jmp	.L65
.L66:
	movq	$6, -16(%rbp)
	jmp	.L65
.L56:
	movl	-28(%rbp), %eax
	testl	%eax, %eax
	je	.L68
	movq	$5, -16(%rbp)
	jmp	.L65
.L68:
	movq	$6, -16(%rbp)
	jmp	.L65
.L58:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L77
	jmp	.L78
.L59:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-28(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$0, -16(%rbp)
	jmp	.L65
.L64:
	movl	-28(%rbp), %eax
	cmpl	$3, %eax
	je	.L71
	cmpl	$3, %eax
	jg	.L72
	cmpl	$1, %eax
	je	.L73
	cmpl	$2, %eax
	je	.L74
	jmp	.L72
.L71:
	movq	$1, -16(%rbp)
	jmp	.L75
.L74:
	movq	$4, -16(%rbp)
	jmp	.L75
.L73:
	movq	$2, -16(%rbp)
	jmp	.L75
.L72:
	movq	$14, -16(%rbp)
	nop
.L75:
	jmp	.L65
.L57:
	movl	$1, -28(%rbp)
	movq	$3, -16(%rbp)
	jmp	.L65
.L62:
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-24(%rbp), %eax
	movl	%eax, %edi
	call	enqueue
	movq	$3, -16(%rbp)
	jmp	.L65
.L79:
	nop
.L65:
	jmp	.L76
.L78:
	call	__stack_chk_fail@PLT
.L77:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	main, .-main
	.section	.rodata
.LC9:
	.string	"\nElements in a Queue are :"
.LC10:
	.string	"\n Queue is empty.."
.LC11:
	.string	"%d,"
	.text
	.globl	display
	.type	display, @function
display:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$8, -8(%rbp)
.L99:
	cmpq	$9, -8(%rbp)
	ja	.L100
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L83(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L83(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L83:
	.long	.L90-.L83
	.long	.L89-.L83
	.long	.L88-.L83
	.long	.L100-.L83
	.long	.L87-.L83
	.long	.L86-.L83
	.long	.L100-.L83
	.long	.L85-.L83
	.long	.L84-.L83
	.long	.L101-.L83
	.text
.L87:
	movl	rear(%rip), %eax
	cmpl	$-1, %eax
	jne	.L91
	movq	$7, -8(%rbp)
	jmp	.L93
.L91:
	movq	$5, -8(%rbp)
	jmp	.L93
.L84:
	movl	front(%rip), %eax
	movl	%eax, -12(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L93
.L89:
	movl	rear(%rip), %eax
	cmpl	%eax, -12(%rbp)
	jg	.L94
	movq	$2, -8(%rbp)
	jmp	.L93
.L94:
	movq	$9, -8(%rbp)
	jmp	.L93
.L86:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -8(%rbp)
	jmp	.L93
.L90:
	movl	front(%rip), %eax
	cmpl	$-1, %eax
	jne	.L97
	movq	$4, -8(%rbp)
	jmp	.L93
.L97:
	movq	$5, -8(%rbp)
	jmp	.L93
.L85:
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$9, -8(%rbp)
	jmp	.L93
.L88:
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	queue(%rip), %rax
	movl	(%rdx,%rax), %eax
	movl	%eax, %esi
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-12(%rbp), %eax
	leal	1(%rax), %edx
	movslq	%edx, %rax
	imulq	$715827883, %rax, %rax
	shrq	$32, %rax
	movl	%edx, %ecx
	sarl	$31, %ecx
	subl	%ecx, %eax
	movl	%eax, -12(%rbp)
	movl	-12(%rbp), %ecx
	movl	%ecx, %eax
	addl	%eax, %eax
	addl	%ecx, %eax
	addl	%eax, %eax
	subl	%eax, %edx
	movl	%edx, -12(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L93
.L100:
	nop
.L93:
	jmp	.L99
.L101:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	display, .-display
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
