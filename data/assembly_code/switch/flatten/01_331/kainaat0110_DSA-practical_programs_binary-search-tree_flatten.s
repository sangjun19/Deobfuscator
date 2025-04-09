	.file	"kainaat0110_DSA-practical_programs_binary-search-tree_flatten.c"
	.text
	.globl	root
	.bss
	.align 8
	.type	root, @object
	.size	root, 8
root:
	.zero	8
	.globl	_TIG_IZ_6KIQ_envp
	.align 8
	.type	_TIG_IZ_6KIQ_envp, @object
	.size	_TIG_IZ_6KIQ_envp, 8
_TIG_IZ_6KIQ_envp:
	.zero	8
	.globl	_TIG_IZ_6KIQ_argc
	.align 4
	.type	_TIG_IZ_6KIQ_argc, @object
	.size	_TIG_IZ_6KIQ_argc, 4
_TIG_IZ_6KIQ_argc:
	.zero	4
	.globl	node
	.align 16
	.type	node, @object
	.size	node, 24
node:
	.zero	24
	.globl	_TIG_IZ_6KIQ_argv
	.align 8
	.type	_TIG_IZ_6KIQ_argv, @object
	.size	_TIG_IZ_6KIQ_argv, 8
_TIG_IZ_6KIQ_argv:
	.zero	8
	.text
	.globl	freenode
	.type	freenode, @function
freenode:
.LFB1:
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
.L6:
	cmpq	$0, -8(%rbp)
	je	.L2
	cmpq	$1, -8(%rbp)
	jne	.L8
	jmp	.L7
.L2:
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$1, -8(%rbp)
	jmp	.L5
.L8:
	nop
.L5:
	jmp	.L6
.L7:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	freenode, .-freenode
	.section	.rodata
.LC0:
	.string	"\nElement not found"
.LC1:
	.string	"\n The %d Element is Present"
	.text
	.globl	search
	.type	search, @function
search:
.LFB2:
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
	movq	$7, -8(%rbp)
.L33:
	cmpq	$14, -8(%rbp)
	ja	.L34
	movq	-8(%rbp), %rax
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
	.long	.L22-.L12
	.long	.L35-.L12
	.long	.L20-.L12
	.long	.L19-.L12
	.long	.L34-.L12
	.long	.L18-.L12
	.long	.L17-.L12
	.long	.L16-.L12
	.long	.L34-.L12
	.long	.L15-.L12
	.long	.L14-.L12
	.long	.L13-.L12
	.long	.L34-.L12
	.long	.L34-.L12
	.long	.L11-.L12
	.text
.L11:
	cmpl	$0, -20(%rbp)
	jne	.L23
	movq	$3, -8(%rbp)
	jmp	.L25
.L23:
	movq	$1, -8(%rbp)
	jmp	.L25
.L19:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -8(%rbp)
	jmp	.L25
.L13:
	movq	-16(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, -44(%rbp)
	jge	.L27
	movq	$9, -8(%rbp)
	jmp	.L25
.L27:
	movq	$10, -8(%rbp)
	jmp	.L25
.L15:
	movq	-16(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, -16(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L25
.L17:
	movl	$0, -20(%rbp)
	movq	root(%rip), %rax
	movq	%rax, -16(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L25
.L18:
	movq	-16(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$1, -20(%rbp)
	movq	$14, -8(%rbp)
	jmp	.L25
.L14:
	movq	-16(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, -16(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L25
.L22:
	movq	-16(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, -44(%rbp)
	jne	.L29
	movq	$5, -8(%rbp)
	jmp	.L25
.L29:
	movq	$11, -8(%rbp)
	jmp	.L25
.L16:
	movq	$6, -8(%rbp)
	jmp	.L25
.L20:
	cmpq	$0, -16(%rbp)
	je	.L31
	movq	$0, -8(%rbp)
	jmp	.L25
.L31:
	movq	$14, -8(%rbp)
	jmp	.L25
.L34:
	nop
.L25:
	jmp	.L33
.L35:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	search, .-search
	.section	.rodata
.LC2:
	.string	"\n\t %d "
	.text
	.globl	inorder
	.type	inorder, @function
inorder:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	$2, -8(%rbp)
.L44:
	cmpq	$2, -8(%rbp)
	je	.L37
	cmpq	$2, -8(%rbp)
	ja	.L46
	cmpq	$0, -8(%rbp)
	je	.L39
	cmpq	$1, -8(%rbp)
	jne	.L46
	jmp	.L45
.L39:
	movq	-24(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, %rdi
	call	inorder
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-24(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, %rdi
	call	inorder
	movq	$1, -8(%rbp)
	jmp	.L41
.L37:
	cmpq	$0, -24(%rbp)
	je	.L42
	movq	$0, -8(%rbp)
	jmp	.L41
.L42:
	movq	$1, -8(%rbp)
	jmp	.L41
.L46:
	nop
.L41:
	jmp	.L44
.L45:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	inorder, .-inorder
	.globl	preorder
	.type	preorder, @function
preorder:
.LFB4:
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
.L56:
	cmpq	$3, -8(%rbp)
	je	.L48
	cmpq	$3, -8(%rbp)
	ja	.L57
	cmpq	$1, -8(%rbp)
	je	.L50
	cmpq	$2, -8(%rbp)
	je	.L58
	jmp	.L57
.L50:
	cmpq	$0, -24(%rbp)
	je	.L52
	movq	$3, -8(%rbp)
	jmp	.L54
.L52:
	movq	$2, -8(%rbp)
	jmp	.L54
.L48:
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
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
	jmp	.L54
.L57:
	nop
.L54:
	jmp	.L56
.L58:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	preorder, .-preorder
	.globl	getnode
	.type	getnode, @function
getnode:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$0, -8(%rbp)
.L64:
	cmpq	$0, -8(%rbp)
	je	.L60
	cmpq	$1, -8(%rbp)
	jne	.L66
	movq	-16(%rbp), %rax
	jmp	.L65
.L60:
	movl	$24, %edi
	call	malloc@PLT
	movq	%rax, -16(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L63
.L66:
	nop
.L63:
	jmp	.L64
.L65:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	getnode, .-getnode
	.section	.rodata
.LC3:
	.string	"\n Duplicate Values."
	.text
	.globl	insert
	.type	insert, @function
insert:
.LFB9:
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
	movq	$8, -16(%rbp)
.L93:
	cmpq	$12, -16(%rbp)
	ja	.L95
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L70(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L70(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L70:
	.long	.L80-.L70
	.long	.L95-.L70
	.long	.L79-.L70
	.long	.L95-.L70
	.long	.L78-.L70
	.long	.L77-.L70
	.long	.L76-.L70
	.long	.L75-.L70
	.long	.L74-.L70
	.long	.L73-.L70
	.long	.L72-.L70
	.long	.L71-.L70
	.long	.L69-.L70
	.text
.L78:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$12, -16(%rbp)
	jmp	.L81
.L69:
	movl	$0, %eax
	jmp	.L94
.L74:
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, -28(%rbp)
	jne	.L83
	movq	$4, -16(%rbp)
	jmp	.L81
.L83:
	movq	$5, -16(%rbp)
	jmp	.L81
.L71:
	movq	-24(%rbp), %rax
	movq	8(%rax), %rax
	movl	-28(%rbp), %edx
	movl	%edx, %esi
	movq	%rax, %rdi
	call	insert
	movq	$12, -16(%rbp)
	jmp	.L81
.L73:
	movq	-24(%rbp), %rax
	movq	8(%rax), %rax
	testq	%rax, %rax
	jne	.L85
	movq	$2, -16(%rbp)
	jmp	.L81
.L85:
	movq	$11, -16(%rbp)
	jmp	.L81
.L76:
	movq	-24(%rbp), %rax
	movq	16(%rax), %rax
	testq	%rax, %rax
	jne	.L87
	movq	$10, -16(%rbp)
	jmp	.L81
.L87:
	movq	$7, -16(%rbp)
	jmp	.L81
.L77:
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, -28(%rbp)
	jge	.L89
	movq	$9, -16(%rbp)
	jmp	.L81
.L89:
	movq	$0, -16(%rbp)
	jmp	.L81
.L72:
	call	getnode
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movl	-28(%rbp), %edx
	movl	%edx, (%rax)
	movq	-8(%rbp), %rax
	movq	$0, 8(%rax)
	movq	-8(%rbp), %rax
	movq	$0, 16(%rax)
	movq	-24(%rbp), %rax
	movq	-8(%rbp), %rdx
	movq	%rdx, 16(%rax)
	movq	$12, -16(%rbp)
	jmp	.L81
.L80:
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, -28(%rbp)
	jle	.L91
	movq	$6, -16(%rbp)
	jmp	.L81
.L91:
	movq	$12, -16(%rbp)
	jmp	.L81
.L75:
	movq	-24(%rbp), %rax
	movq	16(%rax), %rax
	movl	-28(%rbp), %edx
	movl	%edx, %esi
	movq	%rax, %rdi
	call	insert
	movq	$12, -16(%rbp)
	jmp	.L81
.L79:
	call	getnode
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movl	-28(%rbp), %edx
	movl	%edx, (%rax)
	movq	-8(%rbp), %rax
	movq	$0, 8(%rax)
	movq	-8(%rbp), %rax
	movq	$0, 16(%rax)
	movq	-24(%rbp), %rax
	movq	-8(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movq	$12, -16(%rbp)
	jmp	.L81
.L95:
	nop
.L81:
	jmp	.L93
.L94:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	insert, .-insert
	.section	.rodata
.LC4:
	.string	" Enter the node value:"
.LC5:
	.string	"\n %d"
.LC6:
	.string	"\n  ----- Linked List -----"
.LC7:
	.string	"\n 1. Insert"
.LC8:
	.string	"\n 2. PreOrder"
.LC9:
	.string	"\n 3. InOrder"
.LC10:
	.string	"\n 4. PostOrder"
.LC11:
	.string	"\n 5. Search"
.LC12:
	.string	"\n 6. Exit"
.LC13:
	.string	"\n Enter your Choice : "
.LC14:
	.string	"%d"
.LC15:
	.string	"\n Wrong Choice"
	.text
	.globl	main
	.type	main, @function
main:
.LFB10:
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
	movq	$0, root(%rip)
	nop
.L97:
	movl	$0, node(%rip)
	movq	$0, 8+node(%rip)
	movq	$0, 16+node(%rip)
	nop
.L98:
	movq	$0, _TIG_IZ_6KIQ_envp(%rip)
	nop
.L99:
	movq	$0, _TIG_IZ_6KIQ_argv(%rip)
	nop
.L100:
	movl	$0, _TIG_IZ_6KIQ_argc(%rip)
	nop
	nop
.L101:
.L102:
#APP
# 106 "kainaat0110_DSA-practical_programs_binary-search-tree.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-6KIQ--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_6KIQ_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_6KIQ_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_6KIQ_envp(%rip)
	nop
	movq	$3, -24(%rbp)
.L130:
	cmpq	$22, -24(%rbp)
	ja	.L132
	movq	-24(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L105(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L105(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L105:
	.long	.L117-.L105
	.long	.L116-.L105
	.long	.L132-.L105
	.long	.L115-.L105
	.long	.L114-.L105
	.long	.L113-.L105
	.long	.L112-.L105
	.long	.L132-.L105
	.long	.L111-.L105
	.long	.L110-.L105
	.long	.L109-.L105
	.long	.L132-.L105
	.long	.L132-.L105
	.long	.L132-.L105
	.long	.L132-.L105
	.long	.L132-.L105
	.long	.L132-.L105
	.long	.L108-.L105
	.long	.L107-.L105
	.long	.L106-.L105
	.long	.L132-.L105
	.long	.L132-.L105
	.long	.L104-.L105
	.text
.L107:
	movl	$1, %edi
	call	exit@PLT
.L114:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-28(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$9, -24(%rbp)
	jmp	.L118
.L111:
	movq	root(%rip), %rax
	movq	%rax, %rdi
	call	postorder
	movq	$19, -24(%rbp)
	jmp	.L118
.L116:
	movl	-32(%rbp), %eax
	cmpl	$6, %eax
	ja	.L119
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L121(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L121(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L121:
	.long	.L119-.L121
	.long	.L126-.L121
	.long	.L125-.L121
	.long	.L124-.L121
	.long	.L123-.L121
	.long	.L122-.L121
	.long	.L120-.L121
	.text
.L120:
	movq	$18, -24(%rbp)
	jmp	.L127
.L122:
	movq	$17, -24(%rbp)
	jmp	.L127
.L123:
	movq	$8, -24(%rbp)
	jmp	.L127
.L124:
	movq	$5, -24(%rbp)
	jmp	.L127
.L125:
	movq	$0, -24(%rbp)
	jmp	.L127
.L126:
	movq	$4, -24(%rbp)
	jmp	.L127
.L119:
	movq	$10, -24(%rbp)
	nop
.L127:
	jmp	.L118
.L115:
	movq	$19, -24(%rbp)
	jmp	.L118
.L110:
	movq	root(%rip), %rax
	testq	%rax, %rax
	jne	.L128
	movq	$22, -24(%rbp)
	jmp	.L118
.L128:
	movq	$6, -24(%rbp)
	jmp	.L118
.L106:
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
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-32(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$1, -24(%rbp)
	jmp	.L118
.L108:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-28(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-28(%rbp), %edx
	movq	root(%rip), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	search
	movq	$19, -24(%rbp)
	jmp	.L118
.L112:
	movl	-28(%rbp), %edx
	movq	root(%rip), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	insert
	movq	$19, -24(%rbp)
	jmp	.L118
.L104:
	call	getnode
	movq	%rax, -16(%rbp)
	movl	-28(%rbp), %edx
	movq	-16(%rbp), %rax
	movl	%edx, (%rax)
	movq	-16(%rbp), %rax
	movq	$0, 8(%rax)
	movq	-16(%rbp), %rax
	movq	$0, 16(%rax)
	movq	-16(%rbp), %rax
	movq	%rax, root(%rip)
	movq	$19, -24(%rbp)
	jmp	.L118
.L113:
	movq	root(%rip), %rax
	movq	%rax, %rdi
	call	inorder
	movq	$19, -24(%rbp)
	jmp	.L118
.L109:
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$19, -24(%rbp)
	jmp	.L118
.L117:
	movq	root(%rip), %rax
	movq	%rax, %rdi
	call	preorder
	movq	$19, -24(%rbp)
	jmp	.L118
.L132:
	nop
.L118:
	jmp	.L130
	.cfi_endproc
.LFE10:
	.size	main, .-main
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
	movq	$2, -8(%rbp)
.L141:
	cmpq	$2, -8(%rbp)
	je	.L134
	cmpq	$2, -8(%rbp)
	ja	.L143
	cmpq	$0, -8(%rbp)
	je	.L136
	cmpq	$1, -8(%rbp)
	jne	.L143
	jmp	.L142
.L136:
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
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -8(%rbp)
	jmp	.L138
.L134:
	cmpq	$0, -24(%rbp)
	je	.L139
	movq	$0, -8(%rbp)
	jmp	.L138
.L139:
	movq	$1, -8(%rbp)
	jmp	.L138
.L143:
	nop
.L138:
	jmp	.L141
.L142:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE12:
	.size	postorder, .-postorder
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
