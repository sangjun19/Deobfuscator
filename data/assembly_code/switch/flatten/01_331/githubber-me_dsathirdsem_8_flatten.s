	.file	"githubber-me_dsathirdsem_8_flatten.c"
	.text
	.globl	_TIG_IZ_m1uc_argc
	.bss
	.align 4
	.type	_TIG_IZ_m1uc_argc, @object
	.size	_TIG_IZ_m1uc_argc, 4
_TIG_IZ_m1uc_argc:
	.zero	4
	.globl	count
	.align 4
	.type	count, @object
	.size	count, 4
count:
	.zero	4
	.globl	_TIG_IZ_m1uc_envp
	.align 8
	.type	_TIG_IZ_m1uc_envp, @object
	.size	_TIG_IZ_m1uc_envp, 8
_TIG_IZ_m1uc_envp:
	.zero	8
	.globl	_TIG_IZ_m1uc_argv
	.align 8
	.type	_TIG_IZ_m1uc_argv, @object
	.size	_TIG_IZ_m1uc_argv, 8
_TIG_IZ_m1uc_argv:
	.zero	8
	.globl	first
	.align 8
	.type	first, @object
	.size	first, 8
first:
	.zero	8
	.section	.rodata
.LC0:
	.string	"\nDoubly Linked List is empty"
	.align 8
.LC1:
	.string	"\nThe employee node with the ssn:%s is deleted"
	.text
	.globl	deletefront
	.type	deletefront, @function
deletefront:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$8, -16(%rbp)
.L18:
	cmpq	$8, -16(%rbp)
	ja	.L19
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
	.long	.L19-.L4
	.long	.L11-.L4
	.long	.L10-.L4
	.long	.L9-.L4
	.long	.L8-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L8:
	movl	$0, %eax
	jmp	.L12
.L3:
	movq	first(%rip), %rax
	testq	%rax, %rax
	jne	.L13
	movq	$1, -16(%rbp)
	jmp	.L15
.L13:
	movq	$5, -16(%rbp)
	jmp	.L15
.L11:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$4, -16(%rbp)
	jmp	.L15
.L9:
	movq	first(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	first(%rip), %rax
	movq	%rax, %rdi
	call	free@PLT
	movl	count(%rip), %eax
	subl	$1, %eax
	movl	%eax, count(%rip)
	movq	$7, -16(%rbp)
	jmp	.L15
.L6:
	movq	first(%rip), %rax
	movq	%rax, -8(%rbp)
	movq	first(%rip), %rax
	movq	112(%rax), %rax
	movq	%rax, first(%rip)
	movq	-8(%rbp), %rax
	movq	$0, 112(%rax)
	movq	first(%rip), %rax
	movq	$0, 104(%rax)
	movq	-8(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movl	count(%rip), %eax
	subl	$1, %eax
	movl	%eax, count(%rip)
	movq	$2, -16(%rbp)
	jmp	.L15
.L7:
	movq	first(%rip), %rax
	movq	112(%rax), %rax
	testq	%rax, %rax
	jne	.L16
	movq	$3, -16(%rbp)
	jmp	.L15
.L16:
	movq	$6, -16(%rbp)
	jmp	.L15
.L5:
	movl	$0, %eax
	jmp	.L12
.L10:
	movq	first(%rip), %rax
	jmp	.L12
.L19:
	nop
.L15:
	jmp	.L18
.L12:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	deletefront, .-deletefront
	.globl	deleteend
	.type	deleteend, @function
deleteend:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	$5, -8(%rbp)
.L42:
	cmpq	$15, -8(%rbp)
	ja	.L43
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
	.long	.L33-.L23
	.long	.L43-.L23
	.long	.L32-.L23
	.long	.L31-.L23
	.long	.L43-.L23
	.long	.L30-.L23
	.long	.L43-.L23
	.long	.L29-.L23
	.long	.L28-.L23
	.long	.L27-.L23
	.long	.L43-.L23
	.long	.L26-.L23
	.long	.L25-.L23
	.long	.L43-.L23
	.long	.L24-.L23
	.long	.L22-.L23
	.text
.L24:
	movl	$0, %eax
	jmp	.L34
.L22:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$14, -8(%rbp)
	jmp	.L35
.L25:
	movq	first(%rip), %rax
	movq	112(%rax), %rax
	testq	%rax, %rax
	jne	.L36
	movq	$3, -8(%rbp)
	jmp	.L35
.L36:
	movq	$0, -8(%rbp)
	jmp	.L35
.L28:
	movq	first(%rip), %rax
	jmp	.L34
.L31:
	movq	first(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	first(%rip), %rax
	movq	%rax, %rdi
	call	free@PLT
	movl	count(%rip), %eax
	subl	$1, %eax
	movl	%eax, count(%rip)
	movq	$7, -8(%rbp)
	jmp	.L35
.L26:
	movq	-16(%rbp), %rax
	movq	112(%rax), %rax
	testq	%rax, %rax
	je	.L38
	movq	$9, -8(%rbp)
	jmp	.L35
.L38:
	movq	$2, -8(%rbp)
	jmp	.L35
.L27:
	movq	-16(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	-16(%rbp), %rax
	movq	112(%rax), %rax
	movq	%rax, -16(%rbp)
	movq	$11, -8(%rbp)
	jmp	.L35
.L30:
	movq	first(%rip), %rax
	testq	%rax, %rax
	jne	.L40
	movq	$15, -8(%rbp)
	jmp	.L35
.L40:
	movq	$12, -8(%rbp)
	jmp	.L35
.L33:
	movq	$0, -24(%rbp)
	movq	first(%rip), %rax
	movq	%rax, -16(%rbp)
	movq	$11, -8(%rbp)
	jmp	.L35
.L29:
	movl	$0, %eax
	jmp	.L34
.L32:
	movq	-16(%rbp), %rax
	movq	$0, 104(%rax)
	movq	-16(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-16(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	-24(%rbp), %rax
	movq	$0, 112(%rax)
	movl	count(%rip), %eax
	subl	$1, %eax
	movl	%eax, count(%rip)
	movq	$8, -8(%rbp)
	jmp	.L35
.L43:
	nop
.L35:
	jmp	.L42
.L34:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	deleteend, .-deleteend
	.section	.rodata
	.align 8
.LC2:
	.string	"\nDemo Double Ended Queue Operation"
	.align 8
.LC3:
	.string	"\n1:InsertQueueFront\n 2: DeleteQueueFront\n 3:InsertQueueRear\n 4:DeleteQueueRear\n 5:DisplayStatus\n 6: Exit "
.LC4:
	.string	"%d"
	.text
	.globl	deqdemo
	.type	deqdemo, @function
deqdemo:
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
	movq	$1, -16(%rbp)
.L66:
	cmpq	$14, -16(%rbp)
	ja	.L69
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L47(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L47(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L47:
	.long	.L55-.L47
	.long	.L54-.L47
	.long	.L53-.L47
	.long	.L69-.L47
	.long	.L52-.L47
	.long	.L69-.L47
	.long	.L70-.L47
	.long	.L69-.L47
	.long	.L50-.L47
	.long	.L69-.L47
	.long	.L69-.L47
	.long	.L69-.L47
	.long	.L49-.L47
	.long	.L48-.L47
	.long	.L46-.L47
	.text
.L52:
	call	deletefront
	movq	%rax, first(%rip)
	movq	$2, -16(%rbp)
	jmp	.L56
.L46:
	movl	-20(%rbp), %eax
	cmpl	$5, %eax
	ja	.L57
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L59(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L59(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L59:
	.long	.L57-.L59
	.long	.L63-.L59
	.long	.L62-.L59
	.long	.L61-.L59
	.long	.L60-.L59
	.long	.L58-.L59
	.text
.L58:
	movq	$8, -16(%rbp)
	jmp	.L64
.L60:
	movq	$0, -16(%rbp)
	jmp	.L64
.L61:
	movq	$12, -16(%rbp)
	jmp	.L64
.L62:
	movq	$4, -16(%rbp)
	jmp	.L64
.L63:
	movq	$13, -16(%rbp)
	jmp	.L64
.L57:
	movq	$6, -16(%rbp)
	nop
.L64:
	jmp	.L56
.L49:
	call	insertend
	movq	%rax, first(%rip)
	movq	$2, -16(%rbp)
	jmp	.L56
.L50:
	call	display
	movq	$2, -16(%rbp)
	jmp	.L56
.L54:
	movq	$2, -16(%rbp)
	jmp	.L56
.L48:
	call	insertfront
	movq	%rax, first(%rip)
	movq	$2, -16(%rbp)
	jmp	.L56
.L55:
	call	deleteend
	movq	%rax, first(%rip)
	movq	$2, -16(%rbp)
	jmp	.L56
.L53:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-20(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$14, -16(%rbp)
	jmp	.L56
.L69:
	nop
.L56:
	jmp	.L66
.L70:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L68
	call	__stack_chk_fail@PLT
.L68:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	deqdemo, .-deqdemo
	.globl	insertfront
	.type	insertfront, @function
insertfront:
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
.L83:
	cmpq	$5, -8(%rbp)
	ja	.L84
	movq	-8(%rbp), %rax
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
	.long	.L84-.L74
	.long	.L76-.L74
	.long	.L75-.L74
	.long	.L73-.L74
	.text
.L75:
	movq	-16(%rbp), %rax
	jmp	.L79
.L77:
	movq	first(%rip), %rax
	testq	%rax, %rax
	jne	.L80
	movq	$4, -8(%rbp)
	jmp	.L82
.L80:
	movq	$3, -8(%rbp)
	jmp	.L82
.L76:
	movq	first(%rip), %rdx
	movq	-16(%rbp), %rax
	movq	%rdx, 112(%rax)
	movq	first(%rip), %rax
	movq	-16(%rbp), %rdx
	movq	%rdx, 104(%rax)
	movq	$5, -8(%rbp)
	jmp	.L82
.L73:
	movq	-16(%rbp), %rax
	jmp	.L79
.L78:
	call	create
	movq	%rax, -16(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L82
.L84:
	nop
.L82:
	jmp	.L83
.L79:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	insertfront, .-insertfront
	.section	.rodata
.LC5:
	.string	"\n\n~~~Menu~~~"
	.align 8
.LC6:
	.string	"\n1:Create DLL of Employee Nodes"
.LC7:
	.string	"\n2:DisplayStatus"
.LC8:
	.string	"\n3:InsertAtEnd"
.LC9:
	.string	"\n4:DeleteAtEnd"
.LC10:
	.string	"\n5:InsertAtFront"
.LC11:
	.string	"\n6:DeleteAtFront"
	.align 8
.LC12:
	.string	"\n7:Double Ended Queue Demo using DLL"
.LC13:
	.string	"\n8:Exit "
.LC14:
	.string	"\nPlease enter your choice: "
	.align 8
.LC15:
	.string	"\nPlease Enter the valid choice"
	.align 8
.LC16:
	.string	"\nEnter the no of Employees:   "
	.text
	.globl	main
	.type	main, @function
main:
.LFB8:
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
	movl	$0, count(%rip)
	nop
.L86:
	movq	$0, first(%rip)
	nop
.L87:
	movq	$0, _TIG_IZ_m1uc_envp(%rip)
	nop
.L88:
	movq	$0, _TIG_IZ_m1uc_argv(%rip)
	nop
.L89:
	movl	$0, _TIG_IZ_m1uc_argc(%rip)
	nop
	nop
.L90:
.L91:
#APP
# 104 "githubber-me_dsathirdsem_8.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-m1uc--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_m1uc_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_m1uc_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_m1uc_envp(%rip)
	nop
	movq	$10, -16(%rbp)
.L122:
	cmpq	$26, -16(%rbp)
	ja	.L124
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L94(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L94(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L94:
	.long	.L124-.L94
	.long	.L107-.L94
	.long	.L106-.L94
	.long	.L124-.L94
	.long	.L105-.L94
	.long	.L104-.L94
	.long	.L103-.L94
	.long	.L124-.L94
	.long	.L102-.L94
	.long	.L101-.L94
	.long	.L100-.L94
	.long	.L99-.L94
	.long	.L124-.L94
	.long	.L98-.L94
	.long	.L124-.L94
	.long	.L124-.L94
	.long	.L124-.L94
	.long	.L124-.L94
	.long	.L97-.L94
	.long	.L124-.L94
	.long	.L96-.L94
	.long	.L124-.L94
	.long	.L124-.L94
	.long	.L124-.L94
	.long	.L95-.L94
	.long	.L124-.L94
	.long	.L93-.L94
	.text
.L97:
	call	insertfront
	movq	%rax, first(%rip)
	movq	$24, -16(%rbp)
	jmp	.L108
.L105:
	call	display
	movq	$24, -16(%rbp)
	jmp	.L108
.L102:
	call	deleteend
	movq	%rax, first(%rip)
	movq	$24, -16(%rbp)
	jmp	.L108
.L107:
	call	deletefront
	movq	%rax, first(%rip)
	movq	$24, -16(%rbp)
	jmp	.L108
.L95:
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
	call	puts@PLT
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-28(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$26, -16(%rbp)
	jmp	.L108
.L93:
	movl	-28(%rbp), %eax
	cmpl	$8, %eax
	ja	.L109
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L111(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L111(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L111:
	.long	.L109-.L111
	.long	.L118-.L111
	.long	.L117-.L111
	.long	.L116-.L111
	.long	.L115-.L111
	.long	.L114-.L111
	.long	.L113-.L111
	.long	.L112-.L111
	.long	.L110-.L111
	.text
.L110:
	movq	$9, -16(%rbp)
	jmp	.L119
.L112:
	movq	$5, -16(%rbp)
	jmp	.L119
.L113:
	movq	$1, -16(%rbp)
	jmp	.L119
.L114:
	movq	$18, -16(%rbp)
	jmp	.L119
.L115:
	movq	$8, -16(%rbp)
	jmp	.L119
.L116:
	movq	$11, -16(%rbp)
	jmp	.L119
.L117:
	movq	$4, -16(%rbp)
	jmp	.L119
.L118:
	movq	$2, -16(%rbp)
	jmp	.L119
.L109:
	movq	$13, -16(%rbp)
	nop
.L119:
	jmp	.L108
.L99:
	call	insertend
	movq	%rax, first(%rip)
	movq	$24, -16(%rbp)
	jmp	.L108
.L101:
	movl	$0, %edi
	call	exit@PLT
.L98:
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$24, -16(%rbp)
	jmp	.L108
.L103:
	call	insertend
	movq	%rax, first(%rip)
	addl	$1, -20(%rbp)
	movq	$20, -16(%rbp)
	jmp	.L108
.L104:
	call	deqdemo
	movq	$24, -16(%rbp)
	jmp	.L108
.L100:
	movq	$24, -16(%rbp)
	jmp	.L108
.L106:
	leaq	.LC16(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	$1, -20(%rbp)
	movq	$20, -16(%rbp)
	jmp	.L108
.L96:
	movl	-24(%rbp), %eax
	cmpl	%eax, -20(%rbp)
	jg	.L120
	movq	$6, -16(%rbp)
	jmp	.L108
.L120:
	movq	$24, -16(%rbp)
	jmp	.L108
.L124:
	nop
.L108:
	jmp	.L122
	.cfi_endproc
.LFE8:
	.size	main, .-main
	.section	.rodata
	.align 8
.LC17:
	.string	"\nENode:%d||SSN:%s|Name:%s|Department:%s|Designation:%s|Salary:%d|Phone no:%ld"
.LC18:
	.string	"\nNo of employee nodes is %d"
	.align 8
.LC19:
	.string	"\nNo Contents to display in DLL"
	.text
	.globl	display
	.type	display, @function
display:
.LFB11:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	$9, -8(%rbp)
.L142:
	cmpq	$10, -8(%rbp)
	ja	.L143
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L128(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L128(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L128:
	.long	.L143-.L128
	.long	.L135-.L128
	.long	.L143-.L128
	.long	.L134-.L128
	.long	.L133-.L128
	.long	.L132-.L128
	.long	.L131-.L128
	.long	.L144-.L128
	.long	.L143-.L128
	.long	.L129-.L128
	.long	.L127-.L128
	.text
.L133:
	movq	-16(%rbp), %rax
	movq	96(%rax), %rdi
	movq	-16(%rbp), %rax
	movl	88(%rax), %esi
	movq	-16(%rbp), %rax
	leaq	60(%rax), %r9
	movq	-16(%rbp), %rax
	leaq	50(%rax), %r8
	movq	-16(%rbp), %rax
	leaq	25(%rax), %rcx
	movq	-16(%rbp), %rdx
	movl	-20(%rbp), %eax
	pushq	%rdi
	pushq	%rsi
	movl	%eax, %esi
	leaq	.LC17(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addq	$16, %rsp
	movq	-16(%rbp), %rax
	movq	112(%rax), %rax
	movq	%rax, -16(%rbp)
	addl	$1, -20(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L136
.L135:
	movl	$1, -20(%rbp)
	movq	first(%rip), %rax
	movq	%rax, -16(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L136
.L134:
	cmpq	$0, -16(%rbp)
	je	.L137
	movq	$4, -8(%rbp)
	jmp	.L136
.L137:
	movq	$5, -8(%rbp)
	jmp	.L136
.L129:
	movq	$1, -8(%rbp)
	jmp	.L136
.L131:
	cmpq	$0, -16(%rbp)
	jne	.L139
	movq	$10, -8(%rbp)
	jmp	.L136
.L139:
	movq	$3, -8(%rbp)
	jmp	.L136
.L132:
	movl	count(%rip), %eax
	movl	%eax, %esi
	leaq	.LC18(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$7, -8(%rbp)
	jmp	.L136
.L127:
	leaq	.LC19(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -8(%rbp)
	jmp	.L136
.L143:
	nop
.L136:
	jmp	.L142
.L144:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE11:
	.size	display, .-display
	.globl	insertend
	.type	insertend, @function
insertend:
.LFB12:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	$2, -8(%rbp)
.L162:
	cmpq	$10, -8(%rbp)
	ja	.L163
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L148(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L148(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L148:
	.long	.L155-.L148
	.long	.L154-.L148
	.long	.L153-.L148
	.long	.L152-.L148
	.long	.L163-.L148
	.long	.L163-.L148
	.long	.L163-.L148
	.long	.L151-.L148
	.long	.L150-.L148
	.long	.L149-.L148
	.long	.L147-.L148
	.text
.L150:
	movq	first(%rip), %rax
	testq	%rax, %rax
	jne	.L156
	movq	$0, -8(%rbp)
	jmp	.L158
.L156:
	movq	$10, -8(%rbp)
	jmp	.L158
.L154:
	movq	-24(%rbp), %rax
	movq	112(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L158
.L152:
	movq	-24(%rbp), %rax
	movq	112(%rax), %rax
	testq	%rax, %rax
	je	.L159
	movq	$1, -8(%rbp)
	jmp	.L158
.L159:
	movq	$7, -8(%rbp)
	jmp	.L158
.L149:
	movq	first(%rip), %rax
	jmp	.L161
.L147:
	movq	first(%rip), %rax
	movq	%rax, -24(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L158
.L155:
	movq	-16(%rbp), %rax
	jmp	.L161
.L151:
	movq	-24(%rbp), %rax
	movq	-16(%rbp), %rdx
	movq	%rdx, 112(%rax)
	movq	-16(%rbp), %rax
	movq	-24(%rbp), %rdx
	movq	%rdx, 104(%rax)
	movq	$9, -8(%rbp)
	jmp	.L158
.L153:
	call	create
	movq	%rax, -16(%rbp)
	movq	$8, -8(%rbp)
	jmp	.L158
.L163:
	nop
.L158:
	jmp	.L162
.L161:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE12:
	.size	insertend, .-insertend
	.section	.rodata
	.align 8
.LC20:
	.string	"\nEnter the ssn,Name,Department,Designation,Salary,PhoneNo of the employee: "
.LC21:
	.string	"%s %s %s %s %d %ld"
.LC22:
	.string	"\nRunning out of memory"
	.text
	.globl	create
	.type	create, @function
create:
.LFB13:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	$2, -16(%rbp)
.L177:
	cmpq	$7, -16(%rbp)
	ja	.L179
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L167(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L167(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L167:
	.long	.L172-.L167
	.long	.L179-.L167
	.long	.L171-.L167
	.long	.L170-.L167
	.long	.L169-.L167
	.long	.L168-.L167
	.long	.L179-.L167
	.long	.L166-.L167
	.text
.L169:
	leaq	.LC20(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	-24(%rbp), %rax
	leaq	96(%rax), %rsi
	movq	-24(%rbp), %rax
	leaq	88(%rax), %r8
	movq	-24(%rbp), %rax
	leaq	60(%rax), %rdi
	movq	-24(%rbp), %rax
	leaq	50(%rax), %rcx
	movq	-24(%rbp), %rax
	leaq	25(%rax), %rdx
	movq	-24(%rbp), %rax
	subq	$8, %rsp
	pushq	%rsi
	movq	%r8, %r9
	movq	%rdi, %r8
	movq	%rax, %rsi
	leaq	.LC21(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	addq	$16, %rsp
	movq	-24(%rbp), %rax
	movq	$0, 104(%rax)
	movq	-24(%rbp), %rax
	movq	$0, 112(%rax)
	movl	count(%rip), %eax
	addl	$1, %eax
	movl	%eax, count(%rip)
	movq	$5, -16(%rbp)
	jmp	.L173
.L170:
	movl	$120, %edi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	$0, -16(%rbp)
	jmp	.L173
.L168:
	movq	-24(%rbp), %rax
	jmp	.L178
.L172:
	cmpq	$0, -24(%rbp)
	jne	.L175
	movq	$7, -16(%rbp)
	jmp	.L173
.L175:
	movq	$4, -16(%rbp)
	jmp	.L173
.L166:
	leaq	.LC22(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$0, %edi
	call	exit@PLT
.L171:
	movq	$3, -16(%rbp)
	jmp	.L173
.L179:
	nop
.L173:
	jmp	.L177
.L178:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE13:
	.size	create, .-create
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
