	.file	"Nitish481_VVI_4_flatten.c"
	.text
	.globl	_TIG_IZ_jHwA_argv
	.bss
	.align 8
	.type	_TIG_IZ_jHwA_argv, @object
	.size	_TIG_IZ_jHwA_argv, 8
_TIG_IZ_jHwA_argv:
	.zero	8
	.globl	_TIG_IZ_jHwA_envp
	.align 8
	.type	_TIG_IZ_jHwA_envp, @object
	.size	_TIG_IZ_jHwA_envp, 8
_TIG_IZ_jHwA_envp:
	.zero	8
	.globl	_TIG_IZ_jHwA_argc
	.align 4
	.type	_TIG_IZ_jHwA_argc, @object
	.size	_TIG_IZ_jHwA_argc, 4
_TIG_IZ_jHwA_argc:
	.zero	4
	.text
	.globl	searchiterative
	.type	searchiterative, @function
searchiterative:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movl	%esi, -28(%rbp)
	movq	$1, -8(%rbp)
.L24:
	cmpq	$11, -8(%rbp)
	ja	.L25
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
	.long	.L13-.L4
	.long	.L12-.L4
	.long	.L11-.L4
	.long	.L25-.L4
	.long	.L10-.L4
	.long	.L9-.L4
	.long	.L25-.L4
	.long	.L8-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L10:
	movl	$0, %eax
	jmp	.L14
.L7:
	movq	-16(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, -16(%rbp)
	movq	$9, -8(%rbp)
	jmp	.L15
.L12:
	movq	-24(%rbp), %rax
	movq	%rax, -16(%rbp)
	movq	$11, -8(%rbp)
	jmp	.L15
.L3:
	cmpq	$0, -24(%rbp)
	jne	.L16
	movq	$4, -8(%rbp)
	jmp	.L15
.L16:
	movq	$9, -8(%rbp)
	jmp	.L15
.L6:
	cmpq	$0, -16(%rbp)
	je	.L18
	movq	$5, -8(%rbp)
	jmp	.L15
.L18:
	movq	$0, -8(%rbp)
	jmp	.L15
.L9:
	movq	-16(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, -28(%rbp)
	jne	.L20
	movq	$2, -8(%rbp)
	jmp	.L15
.L20:
	movq	$7, -8(%rbp)
	jmp	.L15
.L5:
	movq	-16(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, -16(%rbp)
	movq	$9, -8(%rbp)
	jmp	.L15
.L13:
	movl	$0, %eax
	jmp	.L14
.L8:
	movq	-16(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, -28(%rbp)
	jle	.L22
	movq	$8, -8(%rbp)
	jmp	.L15
.L22:
	movq	$10, -8(%rbp)
	jmp	.L15
.L11:
	movq	-16(%rbp), %rax
	jmp	.L14
.L25:
	nop
.L15:
	jmp	.L24
.L14:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	searchiterative, .-searchiterative
	.section	.rodata
	.align 8
.LC0:
	.string	"Enter the data to be inserted = "
.LC1:
	.string	"%d"
.LC2:
	.string	"\nInorder traversal : "
.LC3:
	.string	"\nPreorder traversal : "
.LC4:
	.string	"Value not found"
.LC5:
	.string	"\nPostorder traversal : "
.LC6:
	.string	"Value found"
	.align 8
.LC7:
	.string	"Enter the data to be deleted = "
	.align 8
.LC8:
	.string	"Enter the data to be search = "
.LC9:
	.string	"\n\nEnter option :"
	.align 8
.LC10:
	.string	"0 -> Exit\n1 -> Create Tree\n2 -> Insert iterative\n3 -> Pre order traversal\n4 -> Inorder traversal\n5 -> Post order traversal\n6 -> Search iterative\n7 -> Delete"
	.align 8
.LC11:
	.string	"Enter a number between 0 to 12"
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
	subq	$80, %rsp
	movl	%edi, -52(%rbp)
	movq	%rsi, -64(%rbp)
	movq	%rdx, -72(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_jHwA_envp(%rip)
	nop
.L27:
	movq	$0, _TIG_IZ_jHwA_argv(%rip)
	nop
.L28:
	movl	$0, _TIG_IZ_jHwA_argc(%rip)
	nop
	nop
.L29:
.L30:
#APP
# 189 "Nitish481_VVI_4.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-jHwA--0
# 0 "" 2
#NO_APP
	movl	-52(%rbp), %eax
	movl	%eax, _TIG_IZ_jHwA_argc(%rip)
	movq	-64(%rbp), %rax
	movq	%rax, _TIG_IZ_jHwA_argv(%rip)
	movq	-72(%rbp), %rax
	movq	%rax, _TIG_IZ_jHwA_envp(%rip)
	nop
	movq	$14, -16(%rbp)
.L66:
	movq	-16(%rbp), %rax
	subq	$3, %rax
	cmpq	$29, %rax
	ja	.L69
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
	.long	.L48-.L33
	.long	.L69-.L33
	.long	.L69-.L33
	.long	.L69-.L33
	.long	.L69-.L33
	.long	.L69-.L33
	.long	.L47-.L33
	.long	.L46-.L33
	.long	.L69-.L33
	.long	.L69-.L33
	.long	.L69-.L33
	.long	.L45-.L33
	.long	.L69-.L33
	.long	.L44-.L33
	.long	.L43-.L33
	.long	.L42-.L33
	.long	.L69-.L33
	.long	.L69-.L33
	.long	.L69-.L33
	.long	.L41-.L33
	.long	.L40-.L33
	.long	.L39-.L33
	.long	.L69-.L33
	.long	.L38-.L33
	.long	.L37-.L33
	.long	.L36-.L33
	.long	.L35-.L33
	.long	.L69-.L33
	.long	.L34-.L33
	.long	.L32-.L33
	.text
.L42:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-36(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-36(%rbp), %edx
	leaq	-32(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	insertiterative
	movq	$22, -16(%rbp)
	jmp	.L49
.L45:
	movq	$10, -16(%rbp)
	jmp	.L49
.L34:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	inorder
	movq	$22, -16(%rbp)
	jmp	.L49
.L40:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	preorder
	movq	$22, -16(%rbp)
	jmp	.L49
.L48:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L67
	jmp	.L68
.L44:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$22, -16(%rbp)
	jmp	.L49
.L39:
	movl	-40(%rbp), %eax
	cmpl	$7, %eax
	ja	.L51
	movl	%eax, %eax
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
	.long	.L60-.L53
	.long	.L59-.L53
	.long	.L58-.L53
	.long	.L57-.L53
	.long	.L56-.L53
	.long	.L55-.L53
	.long	.L54-.L53
	.long	.L52-.L53
	.text
.L52:
	movq	$17, -16(%rbp)
	jmp	.L61
.L54:
	movq	$28, -16(%rbp)
	jmp	.L61
.L55:
	movq	$26, -16(%rbp)
	jmp	.L61
.L56:
	movq	$31, -16(%rbp)
	jmp	.L61
.L57:
	movq	$23, -16(%rbp)
	jmp	.L61
.L58:
	movq	$18, -16(%rbp)
	jmp	.L61
.L59:
	movq	$27, -16(%rbp)
	jmp	.L61
.L60:
	movq	$22, -16(%rbp)
	jmp	.L61
.L51:
	movq	$29, -16(%rbp)
	nop
.L61:
	jmp	.L49
.L38:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	postorder
	movq	$22, -16(%rbp)
	jmp	.L49
.L47:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$22, -16(%rbp)
	jmp	.L49
.L32:
	cmpq	$0, -24(%rbp)
	je	.L62
	movq	$9, -16(%rbp)
	jmp	.L49
.L62:
	movq	$16, -16(%rbp)
	jmp	.L49
.L43:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-36(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-36(%rbp), %edx
	movq	-32(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	deletenode
	movq	%rax, -32(%rbp)
	movq	$22, -16(%rbp)
	jmp	.L49
.L37:
	leaq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	create_tree
	movq	$22, -16(%rbp)
	jmp	.L49
.L41:
	movl	-40(%rbp), %eax
	testl	%eax, %eax
	je	.L64
	movq	$10, -16(%rbp)
	jmp	.L49
.L64:
	movq	$3, -16(%rbp)
	jmp	.L49
.L36:
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-36(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-36(%rbp), %edx
	movq	-32(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	searchiterative
	movq	%rax, -24(%rbp)
	movq	$32, -16(%rbp)
	jmp	.L49
.L46:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-40(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$24, -16(%rbp)
	jmp	.L49
.L35:
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$22, -16(%rbp)
	jmp	.L49
.L69:
	nop
.L49:
	jmp	.L66
.L68:
	call	__stack_chk_fail@PLT
.L67:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	main, .-main
	.section	.rodata
.LC12:
	.string	"%d "
	.text
	.globl	inorder
	.type	inorder, @function
inorder:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	$3, -8(%rbp)
.L80:
	cmpq	$4, -8(%rbp)
	je	.L71
	cmpq	$4, -8(%rbp)
	ja	.L81
	cmpq	$3, -8(%rbp)
	je	.L73
	cmpq	$3, -8(%rbp)
	ja	.L81
	cmpq	$1, -8(%rbp)
	je	.L82
	cmpq	$2, -8(%rbp)
	je	.L83
	jmp	.L81
.L71:
	movq	-24(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, %rdi
	call	inorder
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-24(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, %rdi
	call	inorder
	movq	$1, -8(%rbp)
	jmp	.L76
.L73:
	cmpq	$0, -24(%rbp)
	jne	.L78
	movq	$2, -8(%rbp)
	jmp	.L76
.L78:
	movq	$4, -8(%rbp)
	jmp	.L76
.L81:
	nop
.L76:
	jmp	.L80
.L82:
	nop
	jmp	.L70
.L83:
	nop
.L70:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	inorder, .-inorder
	.section	.rodata
.LC13:
	.string	"No dublicate data please"
	.text
	.globl	insertiterative
	.type	insertiterative, @function
insertiterative:
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
	movl	%esi, -44(%rbp)
	movq	$2, -8(%rbp)
.L120:
	cmpq	$20, -8(%rbp)
	ja	.L121
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
	.long	.L105-.L87
	.long	.L122-.L87
	.long	.L103-.L87
	.long	.L121-.L87
	.long	.L102-.L87
	.long	.L122-.L87
	.long	.L122-.L87
	.long	.L99-.L87
	.long	.L98-.L87
	.long	.L97-.L87
	.long	.L122-.L87
	.long	.L95-.L87
	.long	.L121-.L87
	.long	.L94-.L87
	.long	.L93-.L87
	.long	.L92-.L87
	.long	.L91-.L87
	.long	.L90-.L87
	.long	.L89-.L87
	.long	.L88-.L87
	.long	.L122-.L87
	.text
.L89:
	movq	-16(%rbp), %rax
	movq	8(%rax), %rax
	testq	%rax, %rax
	je	.L106
	movq	$15, -8(%rbp)
	jmp	.L108
.L106:
	movq	$4, -8(%rbp)
	jmp	.L108
.L102:
	movq	-16(%rbp), %rax
	movq	-24(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movq	$1, -8(%rbp)
	jmp	.L108
.L93:
	cmpq	$0, -16(%rbp)
	je	.L109
	movq	$0, -8(%rbp)
	jmp	.L108
.L109:
	movq	$20, -8(%rbp)
	jmp	.L108
.L92:
	movq	-16(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, -16(%rbp)
	movq	$14, -8(%rbp)
	jmp	.L108
.L98:
	movq	-16(%rbp), %rax
	movq	-24(%rbp), %rdx
	movq	%rdx, 16(%rax)
	movq	$5, -8(%rbp)
	jmp	.L108
.L91:
	movq	-40(%rbp), %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	jne	.L112
	movq	$11, -8(%rbp)
	jmp	.L108
.L112:
	movq	$14, -8(%rbp)
	jmp	.L108
.L95:
	movq	-40(%rbp), %rax
	movq	-24(%rbp), %rdx
	movq	%rdx, (%rax)
	movq	$6, -8(%rbp)
	jmp	.L108
.L97:
	movq	-16(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, -16(%rbp)
	movq	$14, -8(%rbp)
	jmp	.L108
.L94:
	movq	-16(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, -44(%rbp)
	jle	.L114
	movq	$7, -8(%rbp)
	jmp	.L108
.L114:
	movq	$18, -8(%rbp)
	jmp	.L108
.L88:
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$10, -8(%rbp)
	jmp	.L108
.L90:
	movq	-40(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, -16(%rbp)
	movl	-44(%rbp), %eax
	movl	%eax, %edi
	call	create
	movq	%rax, -24(%rbp)
	movq	$16, -8(%rbp)
	jmp	.L108
.L105:
	movq	-16(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, -44(%rbp)
	jne	.L116
	movq	$19, -8(%rbp)
	jmp	.L108
.L116:
	movq	$13, -8(%rbp)
	jmp	.L108
.L99:
	movq	-16(%rbp), %rax
	movq	16(%rax), %rax
	testq	%rax, %rax
	je	.L118
	movq	$9, -8(%rbp)
	jmp	.L108
.L118:
	movq	$8, -8(%rbp)
	jmp	.L108
.L103:
	movq	$17, -8(%rbp)
	jmp	.L108
.L121:
	nop
.L108:
	jmp	.L120
.L122:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	insertiterative, .-insertiterative
	.globl	small
	.type	small, @function
small:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movq	$2, -8(%rbp)
.L133:
	cmpq	$5, -8(%rbp)
	je	.L124
	cmpq	$5, -8(%rbp)
	ja	.L135
	cmpq	$4, -8(%rbp)
	je	.L126
	cmpq	$4, -8(%rbp)
	ja	.L135
	cmpq	$1, -8(%rbp)
	je	.L127
	cmpq	$2, -8(%rbp)
	je	.L128
	jmp	.L135
.L126:
	movq	-16(%rbp), %rax
	movq	8(%rax), %rax
	testq	%rax, %rax
	je	.L129
	movq	$5, -8(%rbp)
	jmp	.L131
.L129:
	movq	$1, -8(%rbp)
	jmp	.L131
.L127:
	movq	-16(%rbp), %rax
	jmp	.L134
.L124:
	movq	-16(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, -16(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L131
.L128:
	movq	-24(%rbp), %rax
	movq	%rax, -16(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L131
.L135:
	nop
.L131:
	jmp	.L133
.L134:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	small, .-small
	.globl	deletenode
	.type	deletenode, @function
deletenode:
.LFB10:
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
	movq	$16, -8(%rbp)
.L165:
	cmpq	$16, -8(%rbp)
	ja	.L166
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L139(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L139(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L139:
	.long	.L152-.L139
	.long	.L151-.L139
	.long	.L150-.L139
	.long	.L149-.L139
	.long	.L166-.L139
	.long	.L148-.L139
	.long	.L166-.L139
	.long	.L147-.L139
	.long	.L146-.L139
	.long	.L145-.L139
	.long	.L144-.L139
	.long	.L143-.L139
	.long	.L166-.L139
	.long	.L142-.L139
	.long	.L141-.L139
	.long	.L140-.L139
	.long	.L138-.L139
	.text
.L141:
	movq	-24(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, %rdi
	call	small
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movl	(%rax), %edx
	movq	-24(%rbp), %rax
	movl	%edx, (%rax)
	movq	-16(%rbp), %rax
	movl	(%rax), %edx
	movq	-24(%rbp), %rax
	movq	16(%rax), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	deletenode
	movq	-24(%rbp), %rdx
	movq	%rax, 16(%rdx)
	movq	$15, -8(%rbp)
	jmp	.L153
.L140:
	movq	-24(%rbp), %rax
	jmp	.L154
.L146:
	movq	-24(%rbp), %rax
	movq	8(%rax), %rax
	movl	-28(%rbp), %edx
	movl	%edx, %esi
	movq	%rax, %rdi
	call	deletenode
	movq	-24(%rbp), %rdx
	movq	%rax, 8(%rdx)
	movq	$15, -8(%rbp)
	jmp	.L153
.L151:
	movq	-24(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, -16(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$10, -8(%rbp)
	jmp	.L153
.L149:
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, -28(%rbp)
	jge	.L155
	movq	$8, -8(%rbp)
	jmp	.L153
.L155:
	movq	$7, -8(%rbp)
	jmp	.L153
.L138:
	cmpq	$0, -24(%rbp)
	jne	.L157
	movq	$0, -8(%rbp)
	jmp	.L153
.L157:
	movq	$11, -8(%rbp)
	jmp	.L153
.L143:
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, -28(%rbp)
	jle	.L159
	movq	$9, -8(%rbp)
	jmp	.L153
.L159:
	movq	$3, -8(%rbp)
	jmp	.L153
.L145:
	movq	-24(%rbp), %rax
	movq	16(%rax), %rax
	movl	-28(%rbp), %edx
	movl	%edx, %esi
	movq	%rax, %rdi
	call	deletenode
	movq	-24(%rbp), %rdx
	movq	%rax, 16(%rdx)
	movq	$15, -8(%rbp)
	jmp	.L153
.L142:
	movq	-24(%rbp), %rax
	movq	16(%rax), %rax
	testq	%rax, %rax
	jne	.L161
	movq	$1, -8(%rbp)
	jmp	.L153
.L161:
	movq	$14, -8(%rbp)
	jmp	.L153
.L148:
	movq	-16(%rbp), %rax
	jmp	.L154
.L144:
	movq	-16(%rbp), %rax
	jmp	.L154
.L152:
	movq	-24(%rbp), %rax
	jmp	.L154
.L147:
	movq	-24(%rbp), %rax
	movq	8(%rax), %rax
	testq	%rax, %rax
	jne	.L163
	movq	$2, -8(%rbp)
	jmp	.L153
.L163:
	movq	$13, -8(%rbp)
	jmp	.L153
.L150:
	movq	-24(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, -16(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$5, -8(%rbp)
	jmp	.L153
.L166:
	nop
.L153:
	jmp	.L165
.L154:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	deletenode, .-deletenode
	.globl	create
	.type	create, @function
create:
.LFB11:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movl	%edi, -36(%rbp)
	movq	$2, -16(%rbp)
.L173:
	cmpq	$2, -16(%rbp)
	je	.L168
	cmpq	$2, -16(%rbp)
	ja	.L175
	cmpq	$0, -16(%rbp)
	je	.L170
	cmpq	$1, -16(%rbp)
	jne	.L175
	movq	-24(%rbp), %rax
	jmp	.L174
.L170:
	movl	$24, %edi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movl	-36(%rbp), %edx
	movl	%edx, (%rax)
	movq	-24(%rbp), %rax
	movq	$0, 8(%rax)
	movq	-24(%rbp), %rax
	movq	$0, 16(%rax)
	movq	$1, -16(%rbp)
	jmp	.L172
.L168:
	movq	$0, -16(%rbp)
	jmp	.L172
.L175:
	nop
.L172:
	jmp	.L173
.L174:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE11:
	.size	create, .-create
	.globl	postorder
	.type	postorder, @function
postorder:
.LFB12:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	$0, -8(%rbp)
.L186:
	cmpq	$4, -8(%rbp)
	je	.L187
	cmpq	$4, -8(%rbp)
	ja	.L188
	cmpq	$3, -8(%rbp)
	je	.L189
	cmpq	$3, -8(%rbp)
	ja	.L188
	cmpq	$0, -8(%rbp)
	je	.L180
	cmpq	$2, -8(%rbp)
	je	.L181
	jmp	.L188
.L180:
	cmpq	$0, -24(%rbp)
	jne	.L183
	movq	$3, -8(%rbp)
	jmp	.L185
.L183:
	movq	$2, -8(%rbp)
	jmp	.L185
.L181:
	movq	-24(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, %rdi
	call	postorder
	movq	-24(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, %rdi
	call	postorder
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$4, -8(%rbp)
	jmp	.L185
.L188:
	nop
.L185:
	jmp	.L186
.L187:
	nop
	jmp	.L176
.L189:
	nop
.L176:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE12:
	.size	postorder, .-postorder
	.globl	preorder
	.type	preorder, @function
preorder:
.LFB13:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	$3, -8(%rbp)
.L199:
	cmpq	$3, -8(%rbp)
	je	.L191
	cmpq	$3, -8(%rbp)
	ja	.L200
	cmpq	$2, -8(%rbp)
	je	.L201
	cmpq	$2, -8(%rbp)
	ja	.L200
	cmpq	$0, -8(%rbp)
	je	.L202
	cmpq	$1, -8(%rbp)
	jne	.L200
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-24(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, %rdi
	call	preorder
	movq	-24(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, %rdi
	call	preorder
	movq	$2, -8(%rbp)
	jmp	.L195
.L191:
	cmpq	$0, -24(%rbp)
	jne	.L196
	movq	$0, -8(%rbp)
	jmp	.L195
.L196:
	movq	$1, -8(%rbp)
	jmp	.L195
.L200:
	nop
.L195:
	jmp	.L199
.L201:
	nop
	jmp	.L190
.L202:
	nop
.L190:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE13:
	.size	preorder, .-preorder
	.section	.rodata
.LC14:
	.string	"Binary Search Tree is created"
	.text
	.globl	create_tree
	.type	create_tree, @function
create_tree:
.LFB15:
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
.L209:
	cmpq	$2, -8(%rbp)
	je	.L210
	cmpq	$2, -8(%rbp)
	ja	.L211
	cmpq	$0, -8(%rbp)
	je	.L206
	cmpq	$1, -8(%rbp)
	jne	.L211
	movq	$0, -8(%rbp)
	jmp	.L207
.L206:
	movq	-24(%rbp), %rax
	movq	$0, (%rax)
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$2, -8(%rbp)
	jmp	.L207
.L211:
	nop
.L207:
	jmp	.L209
.L210:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE15:
	.size	create_tree, .-create_tree
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
