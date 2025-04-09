	.file	"mvanbraeckel_BAET-Downheap20x10_2520A3_q1_flatten.c"
	.text
	.globl	_TIG_IZ_SIK6_envp
	.bss
	.align 8
	.type	_TIG_IZ_SIK6_envp, @object
	.size	_TIG_IZ_SIK6_envp, 8
_TIG_IZ_SIK6_envp:
	.zero	8
	.globl	_TIG_IZ_SIK6_argc
	.align 4
	.type	_TIG_IZ_SIK6_argc, @object
	.size	_TIG_IZ_SIK6_argc, 4
_TIG_IZ_SIK6_argc:
	.zero	4
	.globl	_TIG_IZ_SIK6_argv
	.align 8
	.type	_TIG_IZ_SIK6_argv, @object
	.size	_TIG_IZ_SIK6_argv, 8
_TIG_IZ_SIK6_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	" %s"
	.text
	.globl	postorder
	.type	postorder, @function
postorder:
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
	movq	$1, -8(%rbp)
.L10:
	cmpq	$3, -8(%rbp)
	je	.L11
	cmpq	$3, -8(%rbp)
	ja	.L12
	cmpq	$2, -8(%rbp)
	je	.L13
	cmpq	$2, -8(%rbp)
	ja	.L12
	cmpq	$0, -8(%rbp)
	je	.L5
	cmpq	$1, -8(%rbp)
	jne	.L12
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	jne	.L6
	movq	$2, -8(%rbp)
	jmp	.L8
.L6:
	movq	$0, -8(%rbp)
	jmp	.L8
.L5:
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	postorder
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	addq	$8, %rax
	movq	%rax, %rdi
	call	postorder
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	movq	16(%rax), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -8(%rbp)
	jmp	.L8
.L12:
	nop
.L8:
	jmp	.L10
.L11:
	nop
	jmp	.L1
.L13:
	nop
.L1:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	postorder, .-postorder
	.section	.rodata
	.align 8
.LC1:
	.string	"\nError: Invalid value - please try again\n"
	.align 8
.LC2:
	.string	"Enter the new value for '%s': "
	.align 8
.LC3:
	.string	"\nError: Invalid value - must enter a valid decimal number\n"
	.align 8
.LC4:
	.string	"\nError: Invalid value - max of 10 digits\n"
	.text
	.globl	find_variable
	.type	find_variable, @function
find_variable:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$120, %rsp
	.cfi_offset 3, -24
	movq	%rdi, -104(%rbp)
	movq	%rsi, -112(%rbp)
	movq	%rdx, -120(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -24(%rbp)
	xorl	%eax, %eax
	movq	$5, -56(%rbp)
.L57:
	cmpq	$27, -56(%rbp)
	ja	.L60
	movq	-56(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L17(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L17(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L17:
	.long	.L38-.L17
	.long	.L37-.L17
	.long	.L60-.L17
	.long	.L36-.L17
	.long	.L61-.L17
	.long	.L34-.L17
	.long	.L33-.L17
	.long	.L32-.L17
	.long	.L31-.L17
	.long	.L30-.L17
	.long	.L60-.L17
	.long	.L29-.L17
	.long	.L28-.L17
	.long	.L61-.L17
	.long	.L26-.L17
	.long	.L25-.L17
	.long	.L60-.L17
	.long	.L24-.L17
	.long	.L60-.L17
	.long	.L23-.L17
	.long	.L22-.L17
	.long	.L60-.L17
	.long	.L21-.L17
	.long	.L20-.L17
	.long	.L61-.L17
	.long	.L60-.L17
	.long	.L18-.L17
	.long	.L16-.L17
	.text
.L26:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$12, -56(%rbp)
	jmp	.L40
.L25:
	movq	-104(%rbp), %rax
	movq	(%rax), %rax
	movq	8(%rax), %rax
	testq	%rax, %rax
	jne	.L41
	movq	$8, -56(%rbp)
	jmp	.L40
.L41:
	movq	$23, -56(%rbp)
	jmp	.L40
.L28:
	cmpl	$0, -88(%rbp)
	jne	.L43
	movq	$9, -56(%rbp)
	jmp	.L40
.L43:
	movq	$26, -56(%rbp)
	jmp	.L40
.L31:
	movq	-104(%rbp), %rax
	movq	(%rax), %rax
	movq	16(%rax), %rdx
	movq	-112(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -76(%rbp)
	movq	$7, -56(%rbp)
	jmp	.L40
.L37:
	cmpq	$10, -64(%rbp)
	jbe	.L45
	movq	$27, -56(%rbp)
	jmp	.L40
.L45:
	movq	$11, -56(%rbp)
	jmp	.L40
.L20:
	movq	-104(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, %rcx
	movq	-120(%rbp), %rdx
	movq	-112(%rbp), %rax
	movq	%rax, %rsi
	movq	%rcx, %rdi
	call	find_variable
	movq	-104(%rbp), %rax
	movq	(%rax), %rax
	leaq	8(%rax), %rcx
	movq	-120(%rbp), %rdx
	movq	-112(%rbp), %rax
	movq	%rax, %rsi
	movq	%rcx, %rdi
	call	find_variable
	movq	$4, -56(%rbp)
	jmp	.L40
.L36:
	movl	$1, -88(%rbp)
	movq	$12, -56(%rbp)
	jmp	.L40
.L18:
	movq	-104(%rbp), %rax
	movq	(%rax), %rbx
	leaq	-72(%rbp), %rdx
	leaq	-36(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strtod@PLT
	movq	%xmm0, %rax
	movq	%rax, 24(%rbx)
	movq	$24, -56(%rbp)
	jmp	.L40
.L29:
	leaq	-36(%rbp), %rcx
	leaq	-36(%rbp), %rax
	movl	$10, %edx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	strncpy@PLT
	leaq	-36(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -48(%rbp)
	leaq	-36(%rbp), %rdx
	movq	-48(%rbp), %rax
	addq	%rdx, %rax
	movb	$0, (%rax)
	leaq	-36(%rbp), %rax
	movq	%rax, %rdi
	call	isWhitespace
	movl	%eax, -80(%rbp)
	movq	$22, -56(%rbp)
	jmp	.L40
.L30:
	movq	-112(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	stdin(%rip), %rdx
	leaq	-36(%rbp), %rax
	movl	$12, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	leaq	-36(%rbp), %rax
	movq	%rax, %rdi
	call	flush_input
	leaq	-36(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -64(%rbp)
	movq	$1, -56(%rbp)
	jmp	.L40
.L23:
	leaq	-36(%rbp), %rax
	movq	%rax, %rdi
	call	isDouble
	movl	%eax, -84(%rbp)
	movq	$20, -56(%rbp)
	jmp	.L40
.L24:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$12, -56(%rbp)
	jmp	.L40
.L33:
	movq	-104(%rbp), %rax
	movq	(%rax), %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	jne	.L47
	movq	$15, -56(%rbp)
	jmp	.L40
.L47:
	movq	$23, -56(%rbp)
	jmp	.L40
.L16:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$12, -56(%rbp)
	jmp	.L40
.L21:
	cmpl	$0, -80(%rbp)
	je	.L49
	movq	$14, -56(%rbp)
	jmp	.L40
.L49:
	movq	$19, -56(%rbp)
	jmp	.L40
.L34:
	movq	-104(%rbp), %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	jne	.L51
	movq	$13, -56(%rbp)
	jmp	.L40
.L51:
	movq	$6, -56(%rbp)
	jmp	.L40
.L38:
	movq	-120(%rbp), %rax
	movl	$1, (%rax)
	movl	$0, -88(%rbp)
	movq	$12, -56(%rbp)
	jmp	.L40
.L32:
	cmpl	$0, -76(%rbp)
	jne	.L53
	movq	$0, -56(%rbp)
	jmp	.L40
.L53:
	movq	$23, -56(%rbp)
	jmp	.L40
.L22:
	cmpl	$0, -84(%rbp)
	je	.L55
	movq	$3, -56(%rbp)
	jmp	.L40
.L55:
	movq	$17, -56(%rbp)
	jmp	.L40
.L60:
	nop
.L40:
	jmp	.L57
.L61:
	nop
	movq	-24(%rbp), %rax
	subq	%fs:40, %rax
	je	.L59
	call	__stack_chk_fail@PLT
.L59:
	movq	-8(%rbp), %rbx
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	find_variable, .-find_variable
	.globl	get_tree_height
	.type	get_tree_height, @function
get_tree_height:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movq	$0, -8(%rbp)
.L71:
	cmpq	$3, -8(%rbp)
	je	.L63
	cmpq	$3, -8(%rbp)
	ja	.L72
	cmpq	$2, -8(%rbp)
	je	.L65
	cmpq	$2, -8(%rbp)
	ja	.L72
	cmpq	$0, -8(%rbp)
	je	.L66
	cmpq	$1, -8(%rbp)
	jne	.L72
	movl	-20(%rbp), %eax
	addl	$1, %eax
	jmp	.L67
.L63:
	movl	$0, %eax
	jmp	.L67
.L66:
	movq	-40(%rbp), %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	jne	.L68
	movq	$3, -8(%rbp)
	jmp	.L70
.L68:
	movq	$2, -8(%rbp)
	jmp	.L70
.L65:
	movq	-40(%rbp), %rax
	movq	(%rax), %rax
	addq	$8, %rax
	movq	%rax, %rdi
	call	get_tree_height
	movl	%eax, -16(%rbp)
	movq	-40(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	get_tree_height
	movl	%eax, -12(%rbp)
	movl	-16(%rbp), %edx
	movl	-12(%rbp), %eax
	movl	%edx, %esi
	movl	%eax, %edi
	call	max
	movl	%eax, -20(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L70
.L72:
	nop
.L70:
	jmp	.L71
.L67:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	get_tree_height, .-get_tree_height
	.section	.rodata
.LC5:
	.string	"\342\224\206"
.LC6:
	.string	"\342\225\214"
	.text
	.globl	display_tree2
	.type	display_tree2, @function
display_tree2:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movq	%rdi, -40(%rbp)
	movl	%esi, -44(%rbp)
	movq	%rdx, -56(%rbp)
	movl	%ecx, -48(%rbp)
	movq	$15, -8(%rbp)
.L132:
	cmpq	$43, -8(%rbp)
	ja	.L133
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L76(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L76(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L76:
	.long	.L133-.L76
	.long	.L107-.L76
	.long	.L106-.L76
	.long	.L105-.L76
	.long	.L133-.L76
	.long	.L104-.L76
	.long	.L103-.L76
	.long	.L102-.L76
	.long	.L101-.L76
	.long	.L100-.L76
	.long	.L99-.L76
	.long	.L133-.L76
	.long	.L98-.L76
	.long	.L97-.L76
	.long	.L96-.L76
	.long	.L95-.L76
	.long	.L133-.L76
	.long	.L94-.L76
	.long	.L93-.L76
	.long	.L92-.L76
	.long	.L133-.L76
	.long	.L134-.L76
	.long	.L90-.L76
	.long	.L89-.L76
	.long	.L133-.L76
	.long	.L88-.L76
	.long	.L133-.L76
	.long	.L87-.L76
	.long	.L133-.L76
	.long	.L86-.L76
	.long	.L85-.L76
	.long	.L84-.L76
	.long	.L83-.L76
	.long	.L133-.L76
	.long	.L134-.L76
	.long	.L133-.L76
	.long	.L81-.L76
	.long	.L133-.L76
	.long	.L80-.L76
	.long	.L79-.L76
	.long	.L78-.L76
	.long	.L133-.L76
	.long	.L77-.L76
	.long	.L75-.L76
	.text
.L93:
	movq	-40(%rbp), %rax
	movq	(%rax), %rax
	movq	16(%rax), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$0, -16(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L108
.L88:
	movl	$32, %edi
	call	putchar@PLT
	movq	$29, -8(%rbp)
	jmp	.L108
.L85:
	movq	-40(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	movq	-56(%rbp), %rdx
	movl	-44(%rbp), %eax
	movl	$0, %ecx
	movl	%eax, %esi
	call	display_tree2
	movq	$34, -8(%rbp)
	jmp	.L108
.L96:
	movl	-44(%rbp), %eax
	subl	$1, %eax
	cmpl	%eax, -24(%rbp)
	jge	.L109
	movq	$39, -8(%rbp)
	jmp	.L108
.L109:
	movq	$18, -8(%rbp)
	jmp	.L108
.L95:
	movq	-40(%rbp), %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	jne	.L111
	movq	$21, -8(%rbp)
	jmp	.L108
.L111:
	movq	$36, -8(%rbp)
	jmp	.L108
.L84:
	movl	-24(%rbp), %eax
	movslq	%eax, %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	cmpb	$49, %al
	jne	.L113
	movq	$8, -8(%rbp)
	jmp	.L108
.L113:
	movq	$27, -8(%rbp)
	jmp	.L108
.L98:
	movl	-44(%rbp), %eax
	cltq
	leaq	-2(%rax), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movb	$49, (%rax)
	movq	$40, -8(%rbp)
	jmp	.L108
.L101:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$17, -8(%rbp)
	jmp	.L108
.L107:
	movl	-16(%rbp), %eax
	cmpl	-44(%rbp), %eax
	jge	.L115
	movq	$3, -8(%rbp)
	jmp	.L108
.L115:
	movq	$30, -8(%rbp)
	jmp	.L108
.L89:
	movl	$32, %edi
	call	putchar@PLT
	addl	$1, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L108
.L105:
	movl	-16(%rbp), %eax
	movslq	%eax, %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	cmpb	$49, %al
	jne	.L117
	movq	$9, -8(%rbp)
	jmp	.L108
.L117:
	movq	$43, -8(%rbp)
	jmp	.L108
.L81:
	addl	$1, -44(%rbp)
	movq	-40(%rbp), %rax
	movq	(%rax), %rax
	leaq	8(%rax), %rdi
	movq	-56(%rbp), %rdx
	movl	-44(%rbp), %eax
	movl	$1, %ecx
	movl	%eax, %esi
	call	display_tree2
	movl	-44(%rbp), %eax
	cltq
	leaq	-2(%rax), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movb	$48, (%rax)
	movq	$38, -8(%rbp)
	jmp	.L108
.L100:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$19, -8(%rbp)
	jmp	.L108
.L97:
	addl	$1, -24(%rbp)
	movq	$14, -8(%rbp)
	jmp	.L108
.L92:
	movl	$1, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L108
.L83:
	movl	-44(%rbp), %eax
	subl	$2, %eax
	cmpl	%eax, -24(%rbp)
	jge	.L120
	movq	$25, -8(%rbp)
	jmp	.L108
.L120:
	movq	$5, -8(%rbp)
	jmp	.L108
.L94:
	movl	$1, -20(%rbp)
	movq	$42, -8(%rbp)
	jmp	.L108
.L78:
	movq	-40(%rbp), %rax
	movq	(%rax), %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	je	.L122
	movq	$7, -8(%rbp)
	jmp	.L108
.L122:
	movq	$22, -8(%rbp)
	jmp	.L108
.L103:
	addl	$1, -16(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L108
.L87:
	movl	$32, %edi
	call	putchar@PLT
	movq	$17, -8(%rbp)
	jmp	.L108
.L80:
	cmpl	$0, -48(%rbp)
	je	.L124
	movq	$12, -8(%rbp)
	jmp	.L108
.L124:
	movq	$40, -8(%rbp)
	jmp	.L108
.L90:
	movl	$10, %edi
	call	putchar@PLT
	movl	$0, -24(%rbp)
	movq	$14, -8(%rbp)
	jmp	.L108
.L104:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$29, -8(%rbp)
	jmp	.L108
.L99:
	movl	$32, %edi
	call	putchar@PLT
	movq	$17, -8(%rbp)
	jmp	.L108
.L77:
	cmpl	$5, -20(%rbp)
	jg	.L126
	movq	$32, -8(%rbp)
	jmp	.L108
.L126:
	movq	$13, -8(%rbp)
	jmp	.L108
.L79:
	movl	-44(%rbp), %eax
	subl	$2, %eax
	cmpl	%eax, -24(%rbp)
	jne	.L128
	movq	$10, -8(%rbp)
	jmp	.L108
.L128:
	movq	$31, -8(%rbp)
	jmp	.L108
.L102:
	movl	-44(%rbp), %eax
	cltq
	leaq	-1(%rax), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movb	$49, (%rax)
	movq	$22, -8(%rbp)
	jmp	.L108
.L86:
	addl	$1, -20(%rbp)
	movq	$42, -8(%rbp)
	jmp	.L108
.L75:
	movl	$32, %edi
	call	putchar@PLT
	movq	$19, -8(%rbp)
	jmp	.L108
.L106:
	cmpl	$5, -12(%rbp)
	jg	.L130
	movq	$23, -8(%rbp)
	jmp	.L108
.L130:
	movq	$6, -8(%rbp)
	jmp	.L108
.L133:
	nop
.L108:
	jmp	.L132
.L134:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	display_tree2, .-display_tree2
	.section	.rodata
	.align 8
.LC7:
	.string	"Invalid number input <%s>: numbers must be of format '0.00'\n"
	.align 8
.LC8:
	.string	"\nError: Invalid variable name - must be of form 'x0#'\n"
	.align 8
.LC9:
	.string	"\nError: invald input - You must choose one of the menu options by number: [1,7]\nPlease try again\n"
	.align 8
.LC10:
	.string	"Invalid expression <%s>: entire expression must be contained by parentheses: '(' at the start, and ')' at the end\n"
.LC11:
	.string	"\nGood bye!"
	.align 8
.LC12:
	.string	"Usage: %s <input: String expression_fully_paranthesized_with_no_spaces>\n"
.LC13:
	.string	"\nPreorder:"
.LC14:
	.string	"\n"
	.align 8
.LC15:
	.string	"\t...still NULL here -I am root\n"
	.align 8
.LC17:
	.string	"\nEnter the name of variable to be updated: "
	.align 8
.LC18:
	.string	"Invalid expression <%s>: expression must contain an equal number of left and right parentheses\n"
	.align 8
.LC19:
	.string	"\nError: Invalid variable name - please try again\n"
	.align 8
.LC20:
	.string	"What would you like to do?\n(Please enter the number of the option you would like to do)\n\n\t1. Display\n\t2. Preorder\n\t3. Inorder\n\t4. Postorder\n\t5. Update\n\t6. Calculate\n\t7. Exit.\n\nSelect option: "
	.align 8
.LC21:
	.string	"Invalid expression <%s>: expression must contain one more operator than operand\n"
	.align 8
.LC22:
	.string	"Invalid variable input <%s>: variable name of format 'x0#' is too long\n"
	.align 8
.LC23:
	.string	"Invalid variable input <%s>: variables must be of format 'x0#'\n"
	.align 8
.LC24:
	.string	"\nAnswer = undefined --> cannot divide by zero\n"
.LC25:
	.string	"\nAnswer = %.2lf\n\n"
.LC26:
	.string	"\nPostorder:"
.LC27:
	.string	"\nInorder: "
	.align 8
.LC28:
	.string	"Invalid input character <%c>: expression must only contain numbers of format '0.00', variables of form 'x0#', and operators: +, -, *, or / and parentheses: ( or )\n"
	.align 8
.LC29:
	.string	"\nError: Invalid variable name - must be 10 characters or less\n"
	.align 8
.LC30:
	.string	"\nSorry, that variable name is not in the expression - please try again\n"
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
	subq	$240, %rsp
	movl	%edi, -212(%rbp)
	movq	%rsi, -224(%rbp)
	movq	%rdx, -232(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_SIK6_envp(%rip)
	nop
.L136:
	movq	$0, _TIG_IZ_SIK6_argv(%rip)
	nop
.L137:
	movl	$0, _TIG_IZ_SIK6_argc(%rip)
	nop
	nop
.L138:
.L139:
#APP
# 744 "mvanbraeckel_BAET-Downheap20x10_2520A3_q1.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-SIK6--0
# 0 "" 2
#NO_APP
	movl	-212(%rbp), %eax
	movl	%eax, _TIG_IZ_SIK6_argc(%rip)
	movq	-224(%rbp), %rax
	movq	%rax, _TIG_IZ_SIK6_argv(%rip)
	movq	-232(%rbp), %rax
	movq	%rax, _TIG_IZ_SIK6_envp(%rip)
	nop
	movq	$49, -56(%rbp)
.L317:
	cmpq	$132, -56(%rbp)
	ja	.L320
	movq	-56(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L142(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L142(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L142:
	.long	.L238-.L142
	.long	.L237-.L142
	.long	.L236-.L142
	.long	.L320-.L142
	.long	.L235-.L142
	.long	.L320-.L142
	.long	.L234-.L142
	.long	.L233-.L142
	.long	.L232-.L142
	.long	.L320-.L142
	.long	.L231-.L142
	.long	.L230-.L142
	.long	.L229-.L142
	.long	.L228-.L142
	.long	.L227-.L142
	.long	.L226-.L142
	.long	.L225-.L142
	.long	.L224-.L142
	.long	.L223-.L142
	.long	.L222-.L142
	.long	.L221-.L142
	.long	.L320-.L142
	.long	.L320-.L142
	.long	.L220-.L142
	.long	.L219-.L142
	.long	.L218-.L142
	.long	.L320-.L142
	.long	.L217-.L142
	.long	.L320-.L142
	.long	.L320-.L142
	.long	.L216-.L142
	.long	.L320-.L142
	.long	.L215-.L142
	.long	.L214-.L142
	.long	.L213-.L142
	.long	.L212-.L142
	.long	.L211-.L142
	.long	.L210-.L142
	.long	.L209-.L142
	.long	.L208-.L142
	.long	.L207-.L142
	.long	.L206-.L142
	.long	.L320-.L142
	.long	.L205-.L142
	.long	.L204-.L142
	.long	.L203-.L142
	.long	.L320-.L142
	.long	.L202-.L142
	.long	.L201-.L142
	.long	.L200-.L142
	.long	.L199-.L142
	.long	.L320-.L142
	.long	.L198-.L142
	.long	.L320-.L142
	.long	.L197-.L142
	.long	.L196-.L142
	.long	.L195-.L142
	.long	.L194-.L142
	.long	.L193-.L142
	.long	.L192-.L142
	.long	.L191-.L142
	.long	.L320-.L142
	.long	.L190-.L142
	.long	.L189-.L142
	.long	.L320-.L142
	.long	.L320-.L142
	.long	.L320-.L142
	.long	.L188-.L142
	.long	.L320-.L142
	.long	.L187-.L142
	.long	.L186-.L142
	.long	.L185-.L142
	.long	.L184-.L142
	.long	.L183-.L142
	.long	.L182-.L142
	.long	.L181-.L142
	.long	.L320-.L142
	.long	.L180-.L142
	.long	.L179-.L142
	.long	.L178-.L142
	.long	.L320-.L142
	.long	.L177-.L142
	.long	.L176-.L142
	.long	.L175-.L142
	.long	.L320-.L142
	.long	.L174-.L142
	.long	.L320-.L142
	.long	.L173-.L142
	.long	.L172-.L142
	.long	.L320-.L142
	.long	.L171-.L142
	.long	.L170-.L142
	.long	.L169-.L142
	.long	.L320-.L142
	.long	.L168-.L142
	.long	.L167-.L142
	.long	.L166-.L142
	.long	.L320-.L142
	.long	.L165-.L142
	.long	.L164-.L142
	.long	.L320-.L142
	.long	.L320-.L142
	.long	.L163-.L142
	.long	.L162-.L142
	.long	.L161-.L142
	.long	.L160-.L142
	.long	.L320-.L142
	.long	.L159-.L142
	.long	.L320-.L142
	.long	.L158-.L142
	.long	.L157-.L142
	.long	.L156-.L142
	.long	.L155-.L142
	.long	.L154-.L142
	.long	.L320-.L142
	.long	.L153-.L142
	.long	.L152-.L142
	.long	.L151-.L142
	.long	.L150-.L142
	.long	.L149-.L142
	.long	.L148-.L142
	.long	.L320-.L142
	.long	.L147-.L142
	.long	.L146-.L142
	.long	.L320-.L142
	.long	.L145-.L142
	.long	.L320-.L142
	.long	.L144-.L142
	.long	.L320-.L142
	.long	.L320-.L142
	.long	.L143-.L142
	.long	.L320-.L142
	.long	.L141-.L142
	.text
.L223:
	movq	stderr(%rip), %rax
	leaq	-31(%rbp), %rdx
	leaq	.LC7(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movl	$-1, %edi
	call	exit@PLT
.L199:
	call	__ctype_b_loc@PLT
	movq	%rax, -120(%rbp)
	movq	$60, -56(%rbp)
	jmp	.L239
.L161:
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$67, -56(%rbp)
	jmp	.L239
.L143:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$19, -56(%rbp)
	jmp	.L239
.L218:
	movl	-200(%rbp), %eax
	testl	%eax, %eax
	jne	.L240
	movq	$43, -56(%rbp)
	jmp	.L239
.L240:
	movq	$19, -56(%rbp)
	jmp	.L239
.L200:
	cmpl	$2, -212(%rbp)
	je	.L242
	movq	$69, -56(%rbp)
	jmp	.L239
.L242:
	movq	$39, -56(%rbp)
	jmp	.L239
.L198:
	movl	-176(%rbp), %eax
	movslq	%eax, %rdx
	movl	-172(%rbp), %eax
	cltq
	addq	%rax, %rdx
	movq	-144(%rbp), %rax
	leaq	(%rdx,%rax), %rcx
	leaq	-31(%rbp), %rax
	movl	$1, %edx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	strncat@PLT
	addl	$1, -172(%rbp)
	movq	$11, -56(%rbp)
	jmp	.L239
.L235:
	movq	-224(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rdx
	movq	stderr(%rip), %rax
	leaq	.LC10(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movl	$-1, %edi
	call	exit@PLT
.L216:
	cmpq	$1, -96(%rbp)
	ja	.L244
	movq	$113, -56(%rbp)
	jmp	.L239
.L244:
	movq	$12, -56(%rbp)
	jmp	.L239
.L190:
	movq	-224(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, -144(%rbp)
	movq	$0, -160(%rbp)
	movl	$0, -192(%rbp)
	movl	$0, -188(%rbp)
	movl	$0, -184(%rbp)
	movl	$0, -180(%rbp)
	movl	$0, -176(%rbp)
	movq	$15, -56(%rbp)
	jmp	.L239
.L162:
	movq	-224(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rdx
	movq	-152(%rbp), %rax
	subq	$1, %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	cmpb	$41, %al
	je	.L246
	movq	$79, -56(%rbp)
	jmp	.L239
.L246:
	movq	$62, -56(%rbp)
	jmp	.L239
.L163:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L318
	jmp	.L319
.L227:
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-160(%rbp), %rax
	movq	%rax, %rdi
	call	free_tree
	movq	$102, -56(%rbp)
	jmp	.L239
.L156:
	movzbl	-201(%rbp), %eax
	cmpb	$40, %al
	jne	.L249
	movq	$74, -56(%rbp)
	jmp	.L239
.L249:
	movq	$58, -56(%rbp)
	jmp	.L239
.L226:
	movq	-144(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -88(%rbp)
	movq	$36, -56(%rbp)
	jmp	.L239
.L176:
	movzbl	-34(%rbp), %eax
	cmpb	$54, %al
	jne	.L251
	movq	$59, -56(%rbp)
	jmp	.L239
.L251:
	movq	$38, -56(%rbp)
	jmp	.L239
.L195:
	movzbl	-201(%rbp), %eax
	cmpb	$47, %al
	jne	.L253
	movq	$132, -56(%rbp)
	jmp	.L239
.L253:
	movq	$115, -56(%rbp)
	jmp	.L239
.L178:
	movq	-224(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rdx
	movq	stderr(%rip), %rax
	leaq	.LC10(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movl	$-1, %edi
	call	exit@PLT
.L229:
	movl	-172(%rbp), %eax
	subl	$1, %eax
	addl	%eax, -176(%rbp)
	movq	$7, -56(%rbp)
	jmp	.L239
.L187:
	movq	-224(%rbp), %rax
	movq	(%rax), %rdx
	movq	stderr(%rip), %rax
	leaq	.LC12(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movl	$-1, %edi
	call	exit@PLT
.L232:
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-160(%rbp), %rax
	movq	%rax, %rdi
	call	preorder
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$19, -56(%rbp)
	jmp	.L239
.L166:
	movzbl	-34(%rbp), %eax
	cmpb	$52, %al
	jne	.L255
	movq	$37, -56(%rbp)
	jmp	.L239
.L255:
	movq	$20, -56(%rbp)
	jmp	.L239
.L203:
	movzbl	-201(%rbp), %eax
	cmpb	$45, %al
	jne	.L257
	movq	$90, -56(%rbp)
	jmp	.L239
.L257:
	movq	$55, -56(%rbp)
	jmp	.L239
.L197:
	movq	stderr(%rip), %rax
	leaq	-31(%rbp), %rdx
	leaq	.LC7(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movl	$-1, %edi
	call	exit@PLT
.L179:
	leaq	-20(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -72(%rbp)
	movq	$23, -56(%rbp)
	jmp	.L239
.L150:
	leaq	-20(%rbp), %rcx
	leaq	-20(%rbp), %rax
	movl	$10, %edx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	strncpy@PLT
	leaq	-20(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -48(%rbp)
	leaq	-20(%rbp), %rdx
	movq	-48(%rbp), %rax
	addq	%rdx, %rax
	movb	$0, (%rax)
	movl	$0, -200(%rbp)
	leaq	-200(%rbp), %rdx
	leaq	-20(%rbp), %rcx
	leaq	-160(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	find_variable
	movq	$25, -56(%rbp)
	jmp	.L239
.L237:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$31, %edx
	movl	$1, %esi
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	$81, -56(%rbp)
	jmp	.L239
.L177:
	movl	$0, -200(%rbp)
	movl	$0, -168(%rbp)
	movl	$0, -196(%rbp)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, -80(%rbp)
	movq	$19, -56(%rbp)
	jmp	.L239
.L220:
	cmpq	$10, -72(%rbp)
	jbe	.L259
	movq	$10, -56(%rbp)
	jmp	.L239
.L259:
	movq	$63, -56(%rbp)
	jmp	.L239
.L180:
	movl	$0, -168(%rbp)
	movq	$67, -56(%rbp)
	jmp	.L239
.L186:
	cmpq	$1, -64(%rbp)
	jbe	.L261
	movq	$44, -56(%rbp)
	jmp	.L239
.L261:
	movq	$34, -56(%rbp)
	jmp	.L239
.L225:
	movq	-128(%rbp), %rax
	movq	(%rax), %rdx
	movzbl	-31(%rbp), %eax
	movsbq	%al, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$2048, %eax
	testl	%eax, %eax
	je	.L263
	movq	$24, -56(%rbp)
	jmp	.L239
.L263:
	movq	$18, -56(%rbp)
	jmp	.L239
.L219:
	movzbl	-30(%rbp), %eax
	cmpb	$46, %al
	je	.L265
	movq	$105, -56(%rbp)
	jmp	.L239
.L265:
	movq	$50, -56(%rbp)
	jmp	.L239
.L168:
	movzbl	-201(%rbp), %eax
	cmpb	$47, %al
	jle	.L267
	movq	$2, -56(%rbp)
	jmp	.L239
.L267:
	movq	$33, -56(%rbp)
	jmp	.L239
.L158:
	cmpl	$10, -172(%rbp)
	jne	.L269
	movq	$17, -56(%rbp)
	jmp	.L239
.L269:
	movq	$52, -56(%rbp)
	jmp	.L239
.L211:
	movl	-176(%rbp), %eax
	cltq
	cmpq	%rax, -88(%rbp)
	jbe	.L271
	movq	$40, -56(%rbp)
	jmp	.L239
.L271:
	movq	$72, -56(%rbp)
	jmp	.L239
.L194:
	leaq	.LC17(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	stdin(%rip), %rdx
	leaq	-20(%rbp), %rax
	movl	$12, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	leaq	-20(%rbp), %rax
	movq	%rax, %rdi
	call	flush_input
	movq	$91, -56(%rbp)
	jmp	.L239
.L147:
	cmpl	$0, -164(%rbp)
	je	.L273
	movq	$98, -56(%rbp)
	jmp	.L239
.L273:
	movq	$27, -56(%rbp)
	jmp	.L239
.L174:
	movq	stderr(%rip), %rax
	movq	-144(%rbp), %rdx
	leaq	.LC18(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movl	$-1, %edi
	call	exit@PLT
.L165:
	leaq	.LC19(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$67, -56(%rbp)
	jmp	.L239
.L160:
	movq	stderr(%rip), %rax
	leaq	-31(%rbp), %rdx
	leaq	.LC7(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movl	$-1, %edi
	call	exit@PLT
.L230:
	call	__ctype_b_loc@PLT
	movq	%rax, -104(%rbp)
	movq	$117, -56(%rbp)
	jmp	.L239
.L228:
	addl	$1, -188(%rbp)
	movq	$7, -56(%rbp)
	jmp	.L239
.L189:
	leaq	-20(%rbp), %rax
	movq	%rax, %rdi
	call	isWhitespace
	movl	%eax, -164(%rbp)
	movq	$122, -56(%rbp)
	jmp	.L239
.L159:
	movl	-196(%rbp), %eax
	testl	%eax, %eax
	je	.L275
	movq	$127, -56(%rbp)
	jmp	.L239
.L275:
	movq	$71, -56(%rbp)
	jmp	.L239
.L145:
	addl	$1, -180(%rbp)
	movq	$7, -56(%rbp)
	jmp	.L239
.L222:
	leaq	.LC20(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	stdin(%rip), %rdx
	leaq	-34(%rbp), %rax
	movl	$3, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	leaq	-34(%rbp), %rax
	movq	%rax, %rdi
	call	flush_input
	leaq	-34(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -64(%rbp)
	movq	$70, -56(%rbp)
	jmp	.L239
.L215:
	movq	stderr(%rip), %rax
	movq	-144(%rbp), %rdx
	leaq	.LC21(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movl	$-1, %edi
	call	exit@PLT
.L224:
	movq	stderr(%rip), %rax
	leaq	-31(%rbp), %rdx
	leaq	.LC22(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movl	$-1, %edi
	call	exit@PLT
.L171:
	addl	$1, -188(%rbp)
	movq	$7, -56(%rbp)
	jmp	.L239
.L207:
	movl	-176(%rbp), %eax
	movslq	%eax, %rdx
	movq	-144(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	movb	%al, -201(%rbp)
	leaq	-201(%rbp), %rdx
	leaq	-31(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcpy@PLT
	movb	$0, -30(%rbp)
	movq	$111, -56(%rbp)
	jmp	.L239
.L188:
	cmpl	$0, -168(%rbp)
	jne	.L277
	movq	$57, -56(%rbp)
	jmp	.L239
.L277:
	movq	$118, -56(%rbp)
	jmp	.L239
.L196:
	movzbl	-201(%rbp), %eax
	cmpb	$42, %al
	jne	.L279
	movq	$13, -56(%rbp)
	jmp	.L239
.L279:
	movq	$56, -56(%rbp)
	jmp	.L239
.L151:
	movq	-104(%rbp), %rax
	movq	(%rax), %rdx
	movl	-176(%rbp), %ecx
	movl	-172(%rbp), %eax
	addl	%ecx, %eax
	movslq	%eax, %rcx
	movq	-144(%rbp), %rax
	addq	%rcx, %rax
	movzbl	(%rax), %eax
	movsbq	%al, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$2048, %eax
	testl	%eax, %eax
	je	.L281
	movq	$109, -56(%rbp)
	jmp	.L239
.L281:
	movq	$92, -56(%rbp)
	jmp	.L239
.L141:
	addl	$1, -188(%rbp)
	movq	$7, -56(%rbp)
	jmp	.L239
.L157:
	movq	-160(%rbp), %rax
	testq	%rax, %rax
	jne	.L283
	movq	$1, -56(%rbp)
	jmp	.L239
.L283:
	movq	$81, -56(%rbp)
	jmp	.L239
.L191:
	movq	-120(%rbp), %rax
	movq	(%rax), %rdx
	movzbl	-29(%rbp), %eax
	movsbq	%al, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$2048, %eax
	testl	%eax, %eax
	je	.L285
	movq	$83, -56(%rbp)
	jmp	.L239
.L285:
	movq	$54, -56(%rbp)
	jmp	.L239
.L192:
	movl	$0, -196(%rbp)
	leaq	-196(%rbp), %rdx
	leaq	-160(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	evaluate_tree
	movq	%xmm0, %rax
	movq	%rax, -80(%rbp)
	movq	$107, -56(%rbp)
	jmp	.L239
.L234:
	movq	stderr(%rip), %rax
	leaq	-31(%rbp), %rdx
	leaq	.LC7(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movl	$-1, %edi
	call	exit@PLT
.L152:
	movq	-224(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -152(%rbp)
	movq	$103, -56(%rbp)
	jmp	.L239
.L217:
	movl	$1, -168(%rbp)
	movq	$67, -56(%rbp)
	jmp	.L239
.L209:
	movzbl	-34(%rbp), %eax
	cmpb	$55, %al
	jne	.L287
	movq	$14, -56(%rbp)
	jmp	.L239
.L287:
	movq	$130, -56(%rbp)
	jmp	.L239
.L173:
	movzbl	-201(%rbp), %eax
	cmpb	$43, %al
	jne	.L289
	movq	$48, -56(%rbp)
	jmp	.L239
.L289:
	movq	$45, -56(%rbp)
	jmp	.L239
.L154:
	movq	stderr(%rip), %rax
	leaq	-31(%rbp), %rdx
	leaq	.LC23(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movl	$-1, %edi
	call	exit@PLT
.L193:
	movzbl	-201(%rbp), %eax
	cmpb	$41, %al
	jne	.L291
	movq	$125, -56(%rbp)
	jmp	.L239
.L291:
	movq	$94, -56(%rbp)
	jmp	.L239
.L213:
	movzbl	-34(%rbp), %eax
	cmpb	$49, %al
	jne	.L293
	movq	$73, -56(%rbp)
	jmp	.L239
.L293:
	movq	$120, -56(%rbp)
	jmp	.L239
.L182:
	addl	$1, -184(%rbp)
	movq	$7, -56(%rbp)
	jmp	.L239
.L155:
	addl	$1, -192(%rbp)
	movl	-176(%rbp), %eax
	movslq	%eax, %rdx
	movq	-144(%rbp), %rax
	leaq	(%rdx,%rax), %rcx
	leaq	-31(%rbp), %rax
	movl	$4, %edx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	strncpy@PLT
	movb	$0, -27(%rbp)
	leaq	-31(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -136(%rbp)
	movq	$88, -56(%rbp)
	jmp	.L239
.L181:
	movzbl	-34(%rbp), %eax
	cmpb	$51, %al
	jne	.L295
	movq	$41, -56(%rbp)
	jmp	.L239
.L295:
	movq	$96, -56(%rbp)
	jmp	.L239
.L146:
	movq	-112(%rbp), %rax
	movq	(%rax), %rdx
	movzbl	-28(%rbp), %eax
	movsbq	%al, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$2048, %eax
	testl	%eax, %eax
	je	.L297
	movq	$99, -56(%rbp)
	jmp	.L239
.L297:
	movq	$6, -56(%rbp)
	jmp	.L239
.L201:
	addl	$1, -188(%rbp)
	movq	$7, -56(%rbp)
	jmp	.L239
.L144:
	leaq	.LC24(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$19, -56(%rbp)
	jmp	.L239
.L185:
	movq	-80(%rbp), %rax
	movq	%rax, %xmm0
	leaq	.LC25(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$19, -56(%rbp)
	jmp	.L239
.L202:
	movq	stderr(%rip), %rax
	leaq	-31(%rbp), %rdx
	leaq	.LC7(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movl	$-1, %edi
	call	exit@PLT
.L183:
	leaq	-160(%rbp), %rax
	movq	%rax, %rdi
	call	display_tree
	movl	$10, %edi
	call	putchar@PLT
	movq	$19, -56(%rbp)
	jmp	.L239
.L204:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$19, -56(%rbp)
	jmp	.L239
.L170:
	movzbl	-20(%rbp), %eax
	cmpb	$120, %al
	je	.L299
	movq	$104, -56(%rbp)
	jmp	.L239
.L299:
	movq	$78, -56(%rbp)
	jmp	.L239
.L148:
	movzbl	-34(%rbp), %eax
	cmpb	$50, %al
	jne	.L301
	movq	$8, -56(%rbp)
	jmp	.L239
.L301:
	movq	$75, -56(%rbp)
	jmp	.L239
.L184:
	movl	-184(%rbp), %eax
	cmpl	-180(%rbp), %eax
	je	.L303
	movq	$85, -56(%rbp)
	jmp	.L239
.L303:
	movq	$0, -56(%rbp)
	jmp	.L239
.L164:
	addl	$3, -176(%rbp)
	movq	$7, -56(%rbp)
	jmp	.L239
.L214:
	movzbl	-201(%rbp), %eax
	cmpb	$120, %al
	jne	.L305
	movq	$95, -56(%rbp)
	jmp	.L239
.L305:
	movq	$87, -56(%rbp)
	jmp	.L239
.L210:
	leaq	.LC26(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-160(%rbp), %rax
	movq	%rax, %rdi
	call	postorder
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$19, -56(%rbp)
	jmp	.L239
.L149:
	call	__ctype_b_loc@PLT
	movq	%rax, -128(%rbp)
	movq	$16, -56(%rbp)
	jmp	.L239
.L206:
	leaq	.LC27(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-160(%rbp), %rax
	movq	%rax, %rdi
	call	inorder
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$19, -56(%rbp)
	jmp	.L239
.L167:
	addl	$1, -192(%rbp)
	movl	$1, -172(%rbp)
	movq	$11, -56(%rbp)
	jmp	.L239
.L169:
	movl	-172(%rbp), %eax
	cltq
	movb	$0, -31(%rbp,%rax)
	leaq	-31(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -96(%rbp)
	movq	$30, -56(%rbp)
	jmp	.L239
.L153:
	movzbl	-201(%rbp), %eax
	movsbl	%al, %edx
	movq	stderr(%rip), %rax
	leaq	.LC28(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movl	$-1, %edi
	call	exit@PLT
.L231:
	leaq	.LC29(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$67, -56(%rbp)
	jmp	.L239
.L238:
	movl	-192(%rbp), %eax
	subl	-188(%rbp), %eax
	cmpl	$1, %eax
	je	.L307
	movq	$32, -56(%rbp)
	jmp	.L239
.L307:
	movq	$35, -56(%rbp)
	jmp	.L239
.L208:
	movq	-224(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	movzbl	(%rax), %eax
	cmpb	$40, %al
	je	.L309
	movq	$4, -56(%rbp)
	jmp	.L239
.L309:
	movq	$116, -56(%rbp)
	jmp	.L239
.L175:
	call	__ctype_b_loc@PLT
	movq	%rax, -112(%rbp)
	movq	$123, -56(%rbp)
	jmp	.L239
.L233:
	addl	$1, -176(%rbp)
	movq	$15, -56(%rbp)
	jmp	.L239
.L172:
	cmpq	$4, -136(%rbp)
	je	.L311
	movq	$47, -56(%rbp)
	jmp	.L239
.L311:
	movq	$119, -56(%rbp)
	jmp	.L239
.L212:
	movq	-144(%rbp), %rdx
	leaq	-160(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	create_tree
	movq	$110, -56(%rbp)
	jmp	.L239
.L205:
	leaq	.LC30(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$19, -56(%rbp)
	jmp	.L239
.L236:
	movzbl	-201(%rbp), %eax
	cmpb	$57, %al
	jg	.L313
	movq	$112, -56(%rbp)
	jmp	.L239
.L313:
	movq	$33, -56(%rbp)
	jmp	.L239
.L221:
	movzbl	-34(%rbp), %eax
	cmpb	$53, %al
	jne	.L315
	movq	$77, -56(%rbp)
	jmp	.L239
.L315:
	movq	$82, -56(%rbp)
	jmp	.L239
.L320:
	nop
.L239:
	jmp	.L317
.L319:
	call	__stack_chk_fail@PLT
.L318:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	main, .-main
	.globl	evaluate_tree
	.type	evaluate_tree, @function
evaluate_tree:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movq	%rdi, -56(%rbp)
	movq	%rsi, -64(%rbp)
	movq	$20, -24(%rbp)
.L361:
	cmpq	$20, -24(%rbp)
	ja	.L363
	movq	-24(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L324(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L324(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L324:
	.long	.L341-.L324
	.long	.L363-.L324
	.long	.L340-.L324
	.long	.L339-.L324
	.long	.L338-.L324
	.long	.L337-.L324
	.long	.L336-.L324
	.long	.L335-.L324
	.long	.L334-.L324
	.long	.L333-.L324
	.long	.L332-.L324
	.long	.L331-.L324
	.long	.L330-.L324
	.long	.L329-.L324
	.long	.L328-.L324
	.long	.L327-.L324
	.long	.L326-.L324
	.long	.L363-.L324
	.long	.L325-.L324
	.long	.L363-.L324
	.long	.L323-.L324
	.text
.L325:
	movq	-56(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, %rdx
	movq	-64(%rbp), %rax
	movq	%rax, %rsi
	movq	%rdx, %rdi
	call	evaluate_tree
	movq	%xmm0, %rax
	movq	%rax, -16(%rbp)
	movsd	-16(%rbp), %xmm0
	movsd	%xmm0, -40(%rbp)
	movq	-56(%rbp), %rax
	movq	(%rax), %rax
	leaq	8(%rax), %rdx
	movq	-64(%rbp), %rax
	movq	%rax, %rsi
	movq	%rdx, %rdi
	call	evaluate_tree
	movq	%xmm0, %rax
	movq	%rax, -8(%rbp)
	movsd	-8(%rbp), %xmm0
	movsd	%xmm0, -32(%rbp)
	movq	$16, -24(%rbp)
	jmp	.L342
.L338:
	movsd	-40(%rbp), %xmm0
	mulsd	-32(%rbp), %xmm0
	jmp	.L343
.L328:
	movq	-64(%rbp), %rax
	movl	$1, (%rax)
	movq	$10, -24(%rbp)
	jmp	.L342
.L327:
	movq	-64(%rbp), %rax
	movl	(%rax), %eax
	testl	%eax, %eax
	je	.L344
	movq	$5, -24(%rbp)
	jmp	.L342
.L344:
	movq	$18, -24(%rbp)
	jmp	.L342
.L330:
	movq	-56(%rbp), %rax
	movq	(%rax), %rax
	movq	8(%rax), %rax
	testq	%rax, %rax
	jne	.L346
	movq	$13, -24(%rbp)
	jmp	.L342
.L346:
	movq	$15, -24(%rbp)
	jmp	.L342
.L334:
	movsd	-40(%rbp), %xmm0
	subsd	-32(%rbp), %xmm0
	jmp	.L343
.L339:
	movq	-56(%rbp), %rax
	movq	(%rax), %rax
	movq	16(%rax), %rax
	movzbl	(%rax), %eax
	cmpb	$42, %al
	jne	.L348
	movq	$4, -24(%rbp)
	jmp	.L342
.L348:
	movq	$7, -24(%rbp)
	jmp	.L342
.L326:
	movq	-56(%rbp), %rax
	movq	(%rax), %rax
	movq	16(%rax), %rax
	movzbl	(%rax), %eax
	cmpb	$43, %al
	jne	.L350
	movq	$2, -24(%rbp)
	jmp	.L342
.L350:
	movq	$6, -24(%rbp)
	jmp	.L342
.L331:
	pxor	%xmm0, %xmm0
	jmp	.L343
.L333:
	movq	-56(%rbp), %rax
	movq	(%rax), %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	jne	.L352
	movq	$12, -24(%rbp)
	jmp	.L342
.L352:
	movq	$15, -24(%rbp)
	jmp	.L342
.L329:
	movq	-56(%rbp), %rax
	movq	(%rax), %rax
	movsd	24(%rax), %xmm0
	jmp	.L343
.L336:
	movq	-56(%rbp), %rax
	movq	(%rax), %rax
	movq	16(%rax), %rax
	movzbl	(%rax), %eax
	cmpb	$45, %al
	jne	.L354
	movq	$8, -24(%rbp)
	jmp	.L342
.L354:
	movq	$3, -24(%rbp)
	jmp	.L342
.L337:
	pxor	%xmm0, %xmm0
	jmp	.L343
.L332:
	pxor	%xmm0, %xmm0
	jmp	.L343
.L341:
	movsd	-40(%rbp), %xmm0
	divsd	-32(%rbp), %xmm0
	jmp	.L343
.L335:
	pxor	%xmm0, %xmm0
	ucomisd	-32(%rbp), %xmm0
	jp	.L356
	pxor	%xmm0, %xmm0
	ucomisd	-32(%rbp), %xmm0
	jne	.L356
	movq	$14, -24(%rbp)
	jmp	.L342
.L356:
	movq	$0, -24(%rbp)
	jmp	.L342
.L340:
	movsd	-40(%rbp), %xmm0
	addsd	-32(%rbp), %xmm0
	jmp	.L343
.L323:
	movq	-56(%rbp), %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	jne	.L359
	movq	$11, -24(%rbp)
	jmp	.L342
.L359:
	movq	$9, -24(%rbp)
	jmp	.L342
.L363:
	nop
.L342:
	jmp	.L361
.L343:
	movq	%xmm0, %rax
	movq	%rax, %xmm0
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	evaluate_tree, .-evaluate_tree
	.globl	max
	.type	max, @function
max:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -20(%rbp)
	movl	%esi, -24(%rbp)
	movq	$0, -8(%rbp)
.L373:
	cmpq	$3, -8(%rbp)
	je	.L365
	cmpq	$3, -8(%rbp)
	ja	.L375
	cmpq	$2, -8(%rbp)
	je	.L367
	cmpq	$2, -8(%rbp)
	ja	.L375
	cmpq	$0, -8(%rbp)
	je	.L368
	cmpq	$1, -8(%rbp)
	jne	.L375
	movl	-24(%rbp), %eax
	movl	%eax, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L369
.L365:
	movl	-20(%rbp), %eax
	movl	%eax, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L369
.L368:
	movl	-20(%rbp), %eax
	cmpl	-24(%rbp), %eax
	jle	.L370
	movq	$3, -8(%rbp)
	jmp	.L369
.L370:
	movq	$1, -8(%rbp)
	jmp	.L369
.L367:
	movl	-12(%rbp), %eax
	jmp	.L374
.L375:
	nop
.L369:
	jmp	.L373
.L374:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	max, .-max
	.globl	free_tree
	.type	free_tree, @function
free_tree:
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
	movq	$1, -8(%rbp)
.L385:
	cmpq	$3, -8(%rbp)
	je	.L386
	cmpq	$3, -8(%rbp)
	ja	.L387
	cmpq	$2, -8(%rbp)
	je	.L379
	cmpq	$2, -8(%rbp)
	ja	.L387
	cmpq	$0, -8(%rbp)
	je	.L388
	cmpq	$1, -8(%rbp)
	jne	.L387
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	jne	.L381
	movq	$0, -8(%rbp)
	jmp	.L383
.L381:
	movq	$2, -8(%rbp)
	jmp	.L383
.L379:
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	free_tree
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	addq	$8, %rax
	movq	%rax, %rdi
	call	free_tree
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	movq	16(%rax), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$3, -8(%rbp)
	jmp	.L383
.L387:
	nop
.L383:
	jmp	.L385
.L386:
	nop
	jmp	.L376
.L388:
	nop
.L376:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	free_tree, .-free_tree
	.globl	preorder
	.type	preorder, @function
preorder:
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
	movq	$3, -8(%rbp)
.L399:
	cmpq	$4, -8(%rbp)
	je	.L400
	cmpq	$4, -8(%rbp)
	ja	.L401
	cmpq	$3, -8(%rbp)
	je	.L392
	cmpq	$3, -8(%rbp)
	ja	.L401
	cmpq	$0, -8(%rbp)
	je	.L393
	cmpq	$2, -8(%rbp)
	je	.L402
	jmp	.L401
.L392:
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	jne	.L396
	movq	$2, -8(%rbp)
	jmp	.L398
.L396:
	movq	$0, -8(%rbp)
	jmp	.L398
.L393:
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	movq	16(%rax), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	preorder
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	addq	$8, %rax
	movq	%rax, %rdi
	call	preorder
	movq	$4, -8(%rbp)
	jmp	.L398
.L401:
	nop
.L398:
	jmp	.L399
.L400:
	nop
	jmp	.L389
.L402:
	nop
.L389:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE11:
	.size	preorder, .-preorder
	.globl	display_tree
	.type	display_tree, @function
display_tree:
.LFB13:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movq	%rdi, -56(%rbp)
	movq	$1, -16(%rbp)
.L416:
	cmpq	$7, -16(%rbp)
	ja	.L417
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L406(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L406(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L406:
	.long	.L411-.L406
	.long	.L410-.L406
	.long	.L409-.L406
	.long	.L408-.L406
	.long	.L417-.L406
	.long	.L407-.L406
	.long	.L417-.L406
	.long	.L418-.L406
	.text
.L410:
	movq	$2, -16(%rbp)
	jmp	.L412
.L408:
	movl	-36(%rbp), %eax
	leal	1(%rax), %edx
	movl	%edx, %eax
	addl	%eax, %eax
	addl	%edx, %eax
	addl	%eax, %eax
	subl	$1, %eax
	cmpl	%eax, -32(%rbp)
	jge	.L413
	movq	$5, -16(%rbp)
	jmp	.L412
.L413:
	movq	$0, -16(%rbp)
	jmp	.L412
.L407:
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movb	$48, (%rax)
	addl	$1, -32(%rbp)
	movq	$3, -16(%rbp)
	jmp	.L412
.L411:
	movq	-24(%rbp), %rdx
	movq	-56(%rbp), %rax
	movl	$0, %ecx
	movl	$0, %esi
	movq	%rax, %rdi
	call	display_tree2
	movq	$7, -16(%rbp)
	jmp	.L412
.L409:
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	get_tree_height
	movl	%eax, -28(%rbp)
	movl	-28(%rbp), %eax
	movl	%eax, -36(%rbp)
	movl	-36(%rbp), %eax
	leal	1(%rax), %edx
	movl	%edx, %eax
	addl	%eax, %eax
	addl	%edx, %eax
	addl	%eax, %eax
	cltq
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -24(%rbp)
	movl	-36(%rbp), %eax
	leal	1(%rax), %edx
	movl	%edx, %eax
	addl	%eax, %eax
	addl	%edx, %eax
	addl	%eax, %eax
	cltq
	leaq	-1(%rax), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movb	$0, (%rax)
	movl	$0, -32(%rbp)
	movq	$3, -16(%rbp)
	jmp	.L412
.L417:
	nop
.L412:
	jmp	.L416
.L418:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE13:
	.size	display_tree, .-display_tree
	.globl	isDouble
	.type	isDouble, @function
isDouble:
.LFB15:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$80, %rsp
	movq	%rdi, -72(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$15, -32(%rbp)
.L449:
	cmpq	$18, -32(%rbp)
	ja	.L452
	movq	-32(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L422(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L422(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L422:
	.long	.L436-.L422
	.long	.L435-.L422
	.long	.L452-.L422
	.long	.L434-.L422
	.long	.L433-.L422
	.long	.L432-.L422
	.long	.L431-.L422
	.long	.L430-.L422
	.long	.L429-.L422
	.long	.L428-.L422
	.long	.L452-.L422
	.long	.L427-.L422
	.long	.L426-.L422
	.long	.L452-.L422
	.long	.L452-.L422
	.long	.L425-.L422
	.long	.L424-.L422
	.long	.L423-.L422
	.long	.L421-.L422
	.text
.L421:
	movq	-72(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -40(%rbp)
	movq	$4, -32(%rbp)
	jmp	.L437
.L433:
	movl	-64(%rbp), %eax
	cltq
	cmpq	%rax, -40(%rbp)
	jbe	.L438
	movq	$8, -32(%rbp)
	jmp	.L437
.L438:
	movq	$0, -32(%rbp)
	jmp	.L437
.L425:
	movl	$0, -64(%rbp)
	movq	$18, -32(%rbp)
	jmp	.L437
.L426:
	movl	$0, %eax
	jmp	.L450
.L429:
	call	__ctype_b_loc@PLT
	movq	%rax, -48(%rbp)
	movq	$9, -32(%rbp)
	jmp	.L437
.L435:
	movl	$1, %eax
	jmp	.L450
.L434:
	movq	-56(%rbp), %rax
	testq	%rax, %rax
	jne	.L441
	movq	$7, -32(%rbp)
	jmp	.L437
.L441:
	movq	$6, -32(%rbp)
	jmp	.L437
.L424:
	cmpl	$0, -60(%rbp)
	jne	.L443
	movq	$1, -32(%rbp)
	jmp	.L437
.L443:
	movq	$17, -32(%rbp)
	jmp	.L437
.L427:
	movl	-64(%rbp), %eax
	movslq	%eax, %rdx
	movq	-72(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	cmpb	$46, %al
	je	.L445
	movq	$12, -32(%rbp)
	jmp	.L437
.L445:
	movq	$5, -32(%rbp)
	jmp	.L437
.L428:
	movq	-48(%rbp), %rax
	movq	(%rax), %rdx
	movl	-64(%rbp), %eax
	movslq	%eax, %rcx
	movq	-72(%rbp), %rax
	addq	%rcx, %rax
	movzbl	(%rax), %eax
	movsbq	%al, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$2048, %eax
	testl	%eax, %eax
	je	.L447
	movq	$5, -32(%rbp)
	jmp	.L437
.L447:
	movq	$11, -32(%rbp)
	jmp	.L437
.L423:
	movl	$0, %eax
	jmp	.L450
.L431:
	movq	-56(%rbp), %rax
	movzbl	(%rax), %eax
	movzbl	%al, %eax
	movl	%eax, -60(%rbp)
	movq	$16, -32(%rbp)
	jmp	.L437
.L432:
	addl	$1, -64(%rbp)
	movq	$18, -32(%rbp)
	jmp	.L437
.L436:
	leaq	-56(%rbp), %rdx
	movq	-72(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strtod@PLT
	movq	%xmm0, %rax
	movq	%rax, -24(%rbp)
	movsd	-24(%rbp), %xmm0
	movsd	%xmm0, -16(%rbp)
	movsd	-16(%rbp), %xmm1
	pxor	%xmm0, %xmm0
	addsd	%xmm1, %xmm0
	movsd	%xmm0, -16(%rbp)
	movq	$3, -32(%rbp)
	jmp	.L437
.L430:
	movl	$1, %eax
	jmp	.L450
.L452:
	nop
.L437:
	jmp	.L449
.L450:
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L451
	call	__stack_chk_fail@PLT
.L451:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE15:
	.size	isDouble, .-isDouble
	.globl	isWhitespace
	.type	isWhitespace, @function
isWhitespace:
.LFB17:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movq	$4, -8(%rbp)
.L470:
	cmpq	$8, -8(%rbp)
	ja	.L471
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L456(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L456(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L456:
	.long	.L463-.L456
	.long	.L462-.L456
	.long	.L461-.L456
	.long	.L471-.L456
	.long	.L460-.L456
	.long	.L459-.L456
	.long	.L458-.L456
	.long	.L457-.L456
	.long	.L455-.L456
	.text
.L460:
	movl	$0, -28(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L464
.L455:
	movq	-24(%rbp), %rax
	movq	(%rax), %rdx
	movl	-28(%rbp), %eax
	movslq	%eax, %rcx
	movq	-40(%rbp), %rax
	addq	%rcx, %rax
	movzbl	(%rax), %eax
	movsbq	%al, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$8192, %eax
	testl	%eax, %eax
	je	.L465
	movq	$2, -8(%rbp)
	jmp	.L464
.L465:
	movq	$1, -8(%rbp)
	jmp	.L464
.L462:
	movl	$0, %eax
	jmp	.L467
.L458:
	movl	-28(%rbp), %eax
	cltq
	cmpq	%rax, -16(%rbp)
	jbe	.L468
	movq	$0, -8(%rbp)
	jmp	.L464
.L468:
	movq	$7, -8(%rbp)
	jmp	.L464
.L459:
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -16(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L464
.L463:
	call	__ctype_b_loc@PLT
	movq	%rax, -24(%rbp)
	movq	$8, -8(%rbp)
	jmp	.L464
.L457:
	movl	$1, %eax
	jmp	.L467
.L461:
	addl	$1, -28(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L464
.L471:
	nop
.L464:
	jmp	.L470
.L467:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE17:
	.size	isWhitespace, .-isWhitespace
	.globl	create_tree
	.type	create_tree, @function
create_tree:
.LFB18:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$184, %rsp
	.cfi_offset 3, -24
	movq	%rdi, -184(%rbp)
	movq	%rsi, -192(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -24(%rbp)
	xorl	%eax, %eax
	movq	$22, -136(%rbp)
.L522:
	cmpq	$31, -136(%rbp)
	ja	.L525
	movq	-136(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L475(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L475(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L475:
	.long	.L499-.L475
	.long	.L498-.L475
	.long	.L497-.L475
	.long	.L496-.L475
	.long	.L525-.L475
	.long	.L495-.L475
	.long	.L494-.L475
	.long	.L493-.L475
	.long	.L492-.L475
	.long	.L526-.L475
	.long	.L490-.L475
	.long	.L489-.L475
	.long	.L488-.L475
	.long	.L487-.L475
	.long	.L526-.L475
	.long	.L485-.L475
	.long	.L484-.L475
	.long	.L483-.L475
	.long	.L525-.L475
	.long	.L482-.L475
	.long	.L525-.L475
	.long	.L525-.L475
	.long	.L481-.L475
	.long	.L480-.L475
	.long	.L479-.L475
	.long	.L478-.L475
	.long	.L477-.L475
	.long	.L525-.L475
	.long	.L525-.L475
	.long	.L476-.L475
	.long	.L525-.L475
	.long	.L474-.L475
	.text
.L478:
	movl	$0, -164(%rbp)
	movl	$0, -160(%rbp)
	movq	$13, -136(%rbp)
	jmp	.L500
.L485:
	movl	-156(%rbp), %eax
	movslq	%eax, %rdx
	movq	-192(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	movb	%al, -165(%rbp)
	movq	$3, -136(%rbp)
	jmp	.L500
.L474:
	cmpb	$41, -165(%rbp)
	jne	.L502
	movq	$5, -136(%rbp)
	jmp	.L500
.L502:
	movq	$8, -136(%rbp)
	jmp	.L500
.L488:
	movl	-156(%rbp), %eax
	cltq
	cmpq	%rax, -144(%rbp)
	jbe	.L504
	movq	$15, -136(%rbp)
	jmp	.L500
.L504:
	movq	$9, -136(%rbp)
	jmp	.L500
.L492:
	movl	-164(%rbp), %eax
	subl	-160(%rbp), %eax
	cmpl	$1, %eax
	jne	.L506
	movq	$1, -136(%rbp)
	jmp	.L500
.L506:
	movq	$2, -136(%rbp)
	jmp	.L500
.L498:
	cmpb	$43, -165(%rbp)
	jne	.L508
	movq	$17, -136(%rbp)
	jmp	.L500
.L508:
	movq	$7, -136(%rbp)
	jmp	.L500
.L480:
	movq	-192(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$120, %al
	jne	.L510
	movq	$11, -136(%rbp)
	jmp	.L500
.L510:
	movq	$16, -136(%rbp)
	jmp	.L500
.L496:
	cmpb	$40, -165(%rbp)
	jne	.L512
	movq	$29, -136(%rbp)
	jmp	.L500
.L512:
	movq	$31, -136(%rbp)
	jmp	.L500
.L484:
	movq	-184(%rbp), %rax
	movq	(%rax), %rbx
	leaq	-152(%rbp), %rdx
	movq	-192(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strtod@PLT
	movq	%xmm0, %rax
	movq	%rax, 24(%rbx)
	movq	$6, -136(%rbp)
	jmp	.L500
.L479:
	movl	$0, -156(%rbp)
	movq	$26, -136(%rbp)
	jmp	.L500
.L477:
	movq	-192(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -144(%rbp)
	movq	$12, -136(%rbp)
	jmp	.L500
.L489:
	movq	-184(%rbp), %rax
	movq	(%rax), %rax
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 24(%rax)
	movq	$6, -136(%rbp)
	jmp	.L500
.L487:
	movq	-192(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$40, %al
	je	.L514
	movq	$0, -136(%rbp)
	jmp	.L500
.L514:
	movq	$24, -136(%rbp)
	jmp	.L500
.L482:
	cmpb	$47, -165(%rbp)
	jne	.L516
	movq	$17, -136(%rbp)
	jmp	.L500
.L516:
	movq	$2, -136(%rbp)
	jmp	.L500
.L483:
	movl	-156(%rbp), %eax
	cltq
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -128(%rbp)
	movq	-128(%rbp), %rax
	movq	%rax, -120(%rbp)
	movl	-156(%rbp), %eax
	subl	$1, %eax
	movslq	%eax, %rdx
	movq	-192(%rbp), %rax
	leaq	1(%rax), %rcx
	movq	-120(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	strncpy@PLT
	movq	-120(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -112(%rbp)
	movq	-120(%rbp), %rdx
	movq	-112(%rbp), %rax
	addq	%rdx, %rax
	movb	$0, (%rax)
	movq	-192(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -104(%rbp)
	movl	-156(%rbp), %eax
	movslq	%eax, %rdx
	movq	-104(%rbp), %rax
	subq	%rdx, %rax
	subq	$1, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -96(%rbp)
	movq	-96(%rbp), %rax
	movq	%rax, -88(%rbp)
	movq	-192(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -80(%rbp)
	movl	-156(%rbp), %eax
	movslq	%eax, %rdx
	movq	-80(%rbp), %rax
	subq	%rdx, %rax
	leaq	-2(%rax), %rdx
	movl	-156(%rbp), %eax
	cltq
	leaq	1(%rax), %rcx
	movq	-192(%rbp), %rax
	addq	%rax, %rcx
	movq	-88(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	strncpy@PLT
	movq	-88(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -72(%rbp)
	movq	-88(%rbp), %rdx
	movq	-72(%rbp), %rax
	addq	%rdx, %rax
	movb	$0, (%rax)
	movl	$32, %edi
	call	malloc@PLT
	movq	%rax, -64(%rbp)
	movq	-184(%rbp), %rax
	movq	-64(%rbp), %rdx
	movq	%rdx, (%rax)
	movl	$2, %edi
	call	malloc@PLT
	movq	%rax, -56(%rbp)
	movq	-184(%rbp), %rax
	movq	(%rax), %rax
	movq	-56(%rbp), %rdx
	movq	%rdx, 16(%rax)
	movq	-184(%rbp), %rax
	movq	(%rax), %rax
	movq	16(%rax), %rax
	movzbl	-165(%rbp), %edx
	movb	%dl, (%rax)
	movq	-184(%rbp), %rax
	movq	(%rax), %rax
	movq	16(%rax), %rax
	addq	$1, %rax
	movb	$0, (%rax)
	movq	-184(%rbp), %rax
	movq	(%rax), %rax
	pxor	%xmm0, %xmm0
	movsd	%xmm0, 24(%rax)
	movq	-184(%rbp), %rax
	movq	(%rax), %rax
	movq	$0, (%rax)
	movq	-184(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, %rdx
	movq	-120(%rbp), %rax
	movq	%rax, %rsi
	movq	%rdx, %rdi
	call	create_tree
	movq	-184(%rbp), %rax
	movq	(%rax), %rax
	movq	$0, 8(%rax)
	movq	-184(%rbp), %rax
	movq	(%rax), %rax
	leaq	8(%rax), %rdx
	movq	-88(%rbp), %rax
	movq	%rax, %rsi
	movq	%rdx, %rdi
	call	create_tree
	movq	$2, -136(%rbp)
	jmp	.L500
.L494:
	movq	-184(%rbp), %rax
	movq	(%rax), %rax
	movq	$0, (%rax)
	movq	-184(%rbp), %rax
	movq	(%rax), %rax
	movq	$0, 8(%rax)
	movq	$14, -136(%rbp)
	jmp	.L500
.L481:
	movq	$25, -136(%rbp)
	jmp	.L500
.L495:
	addl	$1, -160(%rbp)
	movq	$2, -136(%rbp)
	jmp	.L500
.L490:
	cmpb	$42, -165(%rbp)
	jne	.L518
	movq	$17, -136(%rbp)
	jmp	.L500
.L518:
	movq	$19, -136(%rbp)
	jmp	.L500
.L499:
	movl	$32, %edi
	call	malloc@PLT
	movq	%rax, -48(%rbp)
	movq	-184(%rbp), %rax
	movq	-48(%rbp), %rdx
	movq	%rdx, (%rax)
	movq	-192(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -40(%rbp)
	movq	-40(%rbp), %rax
	addq	$1, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -32(%rbp)
	movq	-184(%rbp), %rax
	movq	(%rax), %rax
	movq	-32(%rbp), %rdx
	movq	%rdx, 16(%rax)
	movq	-184(%rbp), %rax
	movq	(%rax), %rax
	movq	-192(%rbp), %rdx
	movq	%rdx, 16(%rax)
	movq	$23, -136(%rbp)
	jmp	.L500
.L493:
	cmpb	$45, -165(%rbp)
	jne	.L520
	movq	$17, -136(%rbp)
	jmp	.L500
.L520:
	movq	$10, -136(%rbp)
	jmp	.L500
.L476:
	addl	$1, -164(%rbp)
	movq	$2, -136(%rbp)
	jmp	.L500
.L497:
	addl	$1, -156(%rbp)
	movq	$26, -136(%rbp)
	jmp	.L500
.L525:
	nop
.L500:
	jmp	.L522
.L526:
	nop
	movq	-24(%rbp), %rax
	subq	%fs:40, %rax
	je	.L524
	call	__stack_chk_fail@PLT
.L524:
	movq	-8(%rbp), %rbx
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE18:
	.size	create_tree, .-create_tree
	.section	.rodata
.LC31:
	.string	"%s"
	.text
	.globl	inorder
	.type	inorder, @function
inorder:
.LFB19:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	$8, -8(%rbp)
.L546:
	cmpq	$8, -8(%rbp)
	ja	.L547
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L530(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L530(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L530:
	.long	.L547-.L530
	.long	.L537-.L530
	.long	.L536-.L530
	.long	.L535-.L530
	.long	.L548-.L530
	.long	.L533-.L530
	.long	.L548-.L530
	.long	.L531-.L530
	.long	.L529-.L530
	.text
.L529:
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	jne	.L539
	movq	$4, -8(%rbp)
	jmp	.L541
.L539:
	movq	$7, -8(%rbp)
	jmp	.L541
.L537:
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	inorder
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	movq	16(%rax), %rax
	movq	%rax, %rsi
	leaq	.LC31(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	addq	$8, %rax
	movq	%rax, %rdi
	call	inorder
	movq	$5, -8(%rbp)
	jmp	.L541
.L535:
	movl	$40, %edi
	call	putchar@PLT
	movq	$1, -8(%rbp)
	jmp	.L541
.L533:
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	movq	8(%rax), %rax
	testq	%rax, %rax
	je	.L542
	movq	$2, -8(%rbp)
	jmp	.L541
.L542:
	movq	$6, -8(%rbp)
	jmp	.L541
.L531:
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	je	.L544
	movq	$3, -8(%rbp)
	jmp	.L541
.L544:
	movq	$1, -8(%rbp)
	jmp	.L541
.L536:
	movl	$41, %edi
	call	putchar@PLT
	movq	$6, -8(%rbp)
	jmp	.L541
.L547:
	nop
.L541:
	jmp	.L546
.L548:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE19:
	.size	inorder, .-inorder
	.globl	flush_input
	.type	flush_input, @function
flush_input:
.LFB20:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movq	%rdi, -56(%rbp)
	movq	$6, -24(%rbp)
.L565:
	cmpq	$10, -24(%rbp)
	ja	.L566
	movq	-24(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L552(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L552(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L552:
	.long	.L558-.L552
	.long	.L557-.L552
	.long	.L556-.L552
	.long	.L566-.L552
	.long	.L555-.L552
	.long	.L566-.L552
	.long	.L554-.L552
	.long	.L567-.L552
	.long	.L566-.L552
	.long	.L566-.L552
	.long	.L551-.L552
	.text
.L555:
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	leaq	-1(%rax), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movb	$0, (%rax)
	movq	$7, -24(%rbp)
	jmp	.L559
.L557:
	cmpl	$10, -36(%rbp)
	je	.L560
	movq	$0, -24(%rbp)
	jmp	.L559
.L560:
	movq	$2, -24(%rbp)
	jmp	.L559
.L554:
	movq	-56(%rbp), %rax
	movl	$10, %esi
	movq	%rax, %rdi
	call	strchr@PLT
	movq	%rax, -32(%rbp)
	movq	$10, -24(%rbp)
	jmp	.L559
.L551:
	cmpq	$0, -32(%rbp)
	jne	.L562
	movq	$0, -24(%rbp)
	jmp	.L559
.L562:
	movq	$4, -24(%rbp)
	jmp	.L559
.L558:
	call	getchar@PLT
	movl	%eax, -36(%rbp)
	movq	$1, -24(%rbp)
	jmp	.L559
.L556:
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -8(%rbp)
	movq	-56(%rbp), %rdx
	movq	-8(%rbp), %rax
	addq	%rdx, %rax
	movb	$0, (%rax)
	movq	$7, -24(%rbp)
	jmp	.L559
.L566:
	nop
.L559:
	jmp	.L565
.L567:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE20:
	.size	flush_input, .-flush_input
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
