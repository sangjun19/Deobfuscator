	.file	"nikitam614_DSA_assignments_queue_linkedlist_flatten.c"
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
	.globl	_TIG_IZ_7hFP_envp
	.align 8
	.type	_TIG_IZ_7hFP_envp, @object
	.size	_TIG_IZ_7hFP_envp, 8
_TIG_IZ_7hFP_envp:
	.zero	8
	.globl	_TIG_IZ_7hFP_argc
	.align 4
	.type	_TIG_IZ_7hFP_argc, @object
	.size	_TIG_IZ_7hFP_argc, 4
_TIG_IZ_7hFP_argc:
	.zero	4
	.globl	_TIG_IZ_7hFP_argv
	.align 8
	.type	_TIG_IZ_7hFP_argv, @object
	.size	_TIG_IZ_7hFP_argv, 8
_TIG_IZ_7hFP_argv:
	.zero	8
	.text
	.globl	getnode
	.type	getnode, @function
getnode:
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
.L6:
	cmpq	$0, -8(%rbp)
	je	.L2
	cmpq	$1, -8(%rbp)
	jne	.L8
	movl	$16, %edi
	call	malloc@PLT
	movq	%rax, -16(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L4
.L2:
	movq	-16(%rbp), %rax
	jmp	.L7
.L8:
	nop
.L4:
	jmp	.L6
.L7:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	getnode, .-getnode
	.globl	empty
	.type	empty, @function
empty:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	$1, -8(%rbp)
.L18:
	cmpq	$3, -8(%rbp)
	je	.L10
	cmpq	$3, -8(%rbp)
	ja	.L19
	cmpq	$1, -8(%rbp)
	je	.L12
	cmpq	$2, -8(%rbp)
	je	.L13
	jmp	.L19
.L12:
	movq	rear(%rip), %rax
	testq	%rax, %rax
	jne	.L14
	movq	$3, -8(%rbp)
	jmp	.L16
.L14:
	movq	$2, -8(%rbp)
	jmp	.L16
.L10:
	movl	$1, %eax
	jmp	.L17
.L13:
	movl	$0, %eax
	jmp	.L17
.L19:
	nop
.L16:
	jmp	.L18
.L17:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	empty, .-empty
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
	subq	$48, %rsp
	movl	%edi, -36(%rbp)
	movq	$4, -8(%rbp)
.L33:
	cmpq	$7, -8(%rbp)
	ja	.L34
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L23(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L23(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L23:
	.long	.L34-.L23
	.long	.L28-.L23
	.long	.L34-.L23
	.long	.L27-.L23
	.long	.L26-.L23
	.long	.L35-.L23
	.long	.L24-.L23
	.long	.L22-.L23
	.text
.L26:
	movq	$1, -8(%rbp)
	jmp	.L29
.L28:
	call	getnode
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movl	-36(%rbp), %edx
	movl	%edx, (%rax)
	call	empty
	movl	%eax, -20(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L29
.L27:
	movq	-16(%rbp), %rax
	movq	$0, 8(%rax)
	movq	-16(%rbp), %rax
	movq	%rax, front(%rip)
	movq	-16(%rbp), %rax
	movq	%rax, rear(%rip)
	movq	$5, -8(%rbp)
	jmp	.L29
.L24:
	movq	rear(%rip), %rax
	movq	8(%rax), %rdx
	movq	-16(%rbp), %rax
	movq	%rdx, 8(%rax)
	movq	rear(%rip), %rax
	movq	-16(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movq	-16(%rbp), %rax
	movq	%rax, rear(%rip)
	movq	$5, -8(%rbp)
	jmp	.L29
.L22:
	cmpl	$0, -20(%rbp)
	je	.L31
	movq	$3, -8(%rbp)
	jmp	.L29
.L31:
	movq	$6, -8(%rbp)
	jmp	.L29
.L34:
	nop
.L29:
	jmp	.L33
.L35:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	enqueue, .-enqueue
	.section	.rodata
.LC0:
	.string	"\n Wrong Choice"
.LC1:
	.string	"\n Queue is empty"
	.align 8
.LC2:
	.string	"\nDo you want to continue? (Press 1=yes, 0=no)"
.LC3:
	.string	"%d"
.LC4:
	.string	"\n %d is deleted"
.LC5:
	.string	"\n -----Operation on Queue----"
.LC6:
	.string	"\n1. Insert/Enqueue"
.LC7:
	.string	"\n2. Delete/ Dequeue"
.LC8:
	.string	"\n3. Display"
.LC9:
	.string	"\n4. Quit"
.LC10:
	.string	"\n Enter your choice:"
	.align 8
.LC11:
	.string	"\n Enter the value to be inserted :"
	.text
	.globl	main
	.type	main, @function
main:
.LFB6:
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
.L37:
	movq	$0, front(%rip)
	nop
.L38:
	movq	$0, _TIG_IZ_7hFP_envp(%rip)
	nop
.L39:
	movq	$0, _TIG_IZ_7hFP_argv(%rip)
	nop
.L40:
	movl	$0, _TIG_IZ_7hFP_argc(%rip)
	nop
	nop
.L41:
.L42:
#APP
# 153 "nikitam614_DSA_assignments_queue_linkedlist.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-7hFP--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_7hFP_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_7hFP_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_7hFP_envp(%rip)
	nop
	movq	$23, -16(%rbp)
.L71:
	cmpq	$23, -16(%rbp)
	ja	.L74
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L45(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L45(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L45:
	.long	.L58-.L45
	.long	.L57-.L45
	.long	.L74-.L45
	.long	.L56-.L45
	.long	.L55-.L45
	.long	.L75-.L45
	.long	.L53-.L45
	.long	.L52-.L45
	.long	.L74-.L45
	.long	.L74-.L45
	.long	.L51-.L45
	.long	.L74-.L45
	.long	.L50-.L45
	.long	.L74-.L45
	.long	.L49-.L45
	.long	.L48-.L45
	.long	.L74-.L45
	.long	.L74-.L45
	.long	.L74-.L45
	.long	.L74-.L45
	.long	.L47-.L45
	.long	.L46-.L45
	.long	.L74-.L45
	.long	.L44-.L45
	.text
.L55:
	movl	$1, %edi
	call	exit@PLT
.L49:
	call	display
	movq	$21, -16(%rbp)
	jmp	.L59
.L48:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$21, -16(%rbp)
	jmp	.L59
.L50:
	movl	-28(%rbp), %eax
	cmpl	$4, %eax
	je	.L60
	cmpl	$4, %eax
	jg	.L61
	cmpl	$3, %eax
	je	.L62
	cmpl	$3, %eax
	jg	.L61
	cmpl	$1, %eax
	je	.L63
	cmpl	$2, %eax
	je	.L64
	jmp	.L61
.L60:
	movq	$4, -16(%rbp)
	jmp	.L65
.L62:
	movq	$14, -16(%rbp)
	jmp	.L65
.L64:
	movq	$20, -16(%rbp)
	jmp	.L65
.L63:
	movq	$7, -16(%rbp)
	jmp	.L65
.L61:
	movq	$15, -16(%rbp)
	nop
.L65:
	jmp	.L59
.L57:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$21, -16(%rbp)
	jmp	.L59
.L44:
	call	init
	movq	$10, -16(%rbp)
	jmp	.L59
.L56:
	cmpl	$0, -20(%rbp)
	je	.L66
	movq	$1, -16(%rbp)
	jmp	.L59
.L66:
	movq	$6, -16(%rbp)
	jmp	.L59
.L46:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$0, -16(%rbp)
	jmp	.L59
.L53:
	call	dequeue
	movl	%eax, -32(%rbp)
	movl	-32(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$21, -16(%rbp)
	jmp	.L59
.L51:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-28(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$12, -16(%rbp)
	jmp	.L59
.L58:
	movl	-24(%rbp), %eax
	cmpl	$1, %eax
	jne	.L69
	movq	$10, -16(%rbp)
	jmp	.L59
.L69:
	movq	$5, -16(%rbp)
	jmp	.L59
.L52:
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-32(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-32(%rbp), %eax
	movl	%eax, %edi
	call	enqueue
	movq	$21, -16(%rbp)
	jmp	.L59
.L47:
	call	empty
	movl	%eax, -20(%rbp)
	movq	$3, -16(%rbp)
	jmp	.L59
.L74:
	nop
.L59:
	jmp	.L71
.L75:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L73
	call	__stack_chk_fail@PLT
.L73:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	main, .-main
	.globl	freenode
	.type	freenode, @function
freenode:
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
.L81:
	cmpq	$0, -8(%rbp)
	je	.L82
	cmpq	$1, -8(%rbp)
	jne	.L83
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$0, -8(%rbp)
	jmp	.L79
.L83:
	nop
.L79:
	jmp	.L81
.L82:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	freenode, .-freenode
	.globl	dequeue
	.type	dequeue, @function
dequeue:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	$3, -8(%rbp)
.L98:
	cmpq	$6, -8(%rbp)
	ja	.L100
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L87(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L87(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L87:
	.long	.L93-.L87
	.long	.L92-.L87
	.long	.L91-.L87
	.long	.L90-.L87
	.long	.L89-.L87
	.long	.L88-.L87
	.long	.L86-.L87
	.text
.L89:
	call	init
	movq	$6, -8(%rbp)
	jmp	.L94
.L92:
	movq	front(%rip), %rax
	movq	%rax, %rdx
	movq	rear(%rip), %rax
	cmpq	%rax, %rdx
	jne	.L95
	movq	$4, -8(%rbp)
	jmp	.L94
.L95:
	movq	$5, -8(%rbp)
	jmp	.L94
.L90:
	movq	$0, -8(%rbp)
	jmp	.L94
.L86:
	movq	-16(%rbp), %rax
	movq	%rax, %rdi
	call	freenode
	movq	$2, -8(%rbp)
	jmp	.L94
.L88:
	movq	front(%rip), %rax
	movq	8(%rax), %rax
	movq	%rax, front(%rip)
	movq	$6, -8(%rbp)
	jmp	.L94
.L93:
	movq	front(%rip), %rax
	movq	%rax, -16(%rbp)
	movq	front(%rip), %rax
	movl	(%rax), %eax
	movl	%eax, -20(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L94
.L91:
	movl	-20(%rbp), %eax
	jmp	.L99
.L100:
	nop
.L94:
	jmp	.L98
.L99:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	dequeue, .-dequeue
	.section	.rodata
.LC12:
	.string	"-> %d "
	.text
	.globl	display
	.type	display, @function
display:
.LFB10:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	$4, -8(%rbp)
.L117:
	cmpq	$9, -8(%rbp)
	ja	.L118
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L104(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L104(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L104:
	.long	.L110-.L104
	.long	.L118-.L104
	.long	.L119-.L104
	.long	.L108-.L104
	.long	.L107-.L104
	.long	.L106-.L104
	.long	.L118-.L104
	.long	.L118-.L104
	.long	.L105-.L104
	.long	.L103-.L104
	.text
.L107:
	call	empty
	movl	%eax, -20(%rbp)
	movq	$8, -8(%rbp)
	jmp	.L111
.L105:
	cmpl	$0, -20(%rbp)
	je	.L112
	movq	$5, -8(%rbp)
	jmp	.L111
.L112:
	movq	$9, -8(%rbp)
	jmp	.L111
.L108:
	movq	-16(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-16(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, -16(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L111
.L103:
	movq	front(%rip), %rax
	movq	%rax, -16(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L111
.L106:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$2, -8(%rbp)
	jmp	.L111
.L110:
	cmpq	$0, -16(%rbp)
	je	.L114
	movq	$3, -8(%rbp)
	jmp	.L111
.L114:
	movq	$2, -8(%rbp)
	jmp	.L111
.L118:
	nop
.L111:
	jmp	.L117
.L119:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	display, .-display
	.globl	init
	.type	init, @function
init:
.LFB13:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	$0, -8(%rbp)
.L126:
	cmpq	$2, -8(%rbp)
	je	.L121
	cmpq	$2, -8(%rbp)
	ja	.L128
	cmpq	$0, -8(%rbp)
	je	.L123
	cmpq	$1, -8(%rbp)
	jne	.L128
	jmp	.L127
.L123:
	movq	$2, -8(%rbp)
	jmp	.L125
.L121:
	movq	$0, rear(%rip)
	movq	rear(%rip), %rax
	movq	%rax, front(%rip)
	movq	$1, -8(%rbp)
	jmp	.L125
.L128:
	nop
.L125:
	jmp	.L126
.L127:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE13:
	.size	init, .-init
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
