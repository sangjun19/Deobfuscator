	.file	"githubber-me_dsathirdsem_7_flatten.c"
	.text
	.globl	_TIG_IZ_Hah2_argv
	.bss
	.align 8
	.type	_TIG_IZ_Hah2_argv, @object
	.size	_TIG_IZ_Hah2_argv, 8
_TIG_IZ_Hah2_argv:
	.zero	8
	.globl	count
	.align 4
	.type	count, @object
	.size	count, 4
count:
	.zero	4
	.globl	start
	.align 8
	.type	start, @object
	.size	start, 8
start:
	.zero	8
	.globl	_TIG_IZ_Hah2_envp
	.align 8
	.type	_TIG_IZ_Hah2_envp, @object
	.size	_TIG_IZ_Hah2_envp, 8
_TIG_IZ_Hah2_envp:
	.zero	8
	.globl	_TIG_IZ_Hah2_argc
	.align 4
	.type	_TIG_IZ_Hah2_argc, @object
	.size	_TIG_IZ_Hah2_argc, 4
_TIG_IZ_Hah2_argc:
	.zero	4
	.section	.rodata
	.align 8
.LC0:
	.string	"\nThe Student node with usn:%s is deleted"
	.align 8
.LC1:
	.string	"\nThe Student node with usn:%s is deleted "
.LC2:
	.string	"\nLinked list is empty"
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
	movq	$1, -16(%rbp)
.L18:
	cmpq	$9, -16(%rbp)
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
	.long	.L19-.L4
	.long	.L8-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L5:
	movq	start(%rip), %rax
	movq	88(%rax), %rax
	testq	%rax, %rax
	jne	.L12
	movq	$7, -16(%rbp)
	jmp	.L14
.L12:
	movq	$5, -16(%rbp)
	jmp	.L14
.L11:
	movq	start(%rip), %rax
	testq	%rax, %rax
	jne	.L15
	movq	$2, -16(%rbp)
	jmp	.L14
.L15:
	movq	$8, -16(%rbp)
	jmp	.L14
.L9:
	movl	$0, %eax
	jmp	.L17
.L3:
	movq	start(%rip), %rax
	jmp	.L17
.L7:
	movl	$0, %eax
	jmp	.L17
.L8:
	movq	start(%rip), %rax
	movq	%rax, -8(%rbp)
	movq	start(%rip), %rax
	movq	88(%rax), %rax
	movq	%rax, start(%rip)
	movq	-8(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	count(%rip), %eax
	subl	$1, %eax
	movl	%eax, count(%rip)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$9, -16(%rbp)
	jmp	.L14
.L6:
	movq	start(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	count(%rip), %eax
	subl	$1, %eax
	movl	%eax, count(%rip)
	movq	start(%rip), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$3, -16(%rbp)
	jmp	.L14
.L10:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$6, -16(%rbp)
	jmp	.L14
.L19:
	nop
.L14:
	jmp	.L18
.L17:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	deletefront, .-deletefront
	.section	.rodata
	.align 8
.LC3:
	.string	"\nThe student node with the usn:%s is deleted"
.LC4:
	.string	"\nLinked List is empty"
	.text
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
	movq	$8, -8(%rbp)
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
	.long	.L30-.L23
	.long	.L43-.L23
	.long	.L29-.L23
	.long	.L28-.L23
	.long	.L27-.L23
	.long	.L26-.L23
	.long	.L25-.L23
	.long	.L43-.L23
	.long	.L43-.L23
	.long	.L43-.L23
	.long	.L24-.L23
	.long	.L22-.L23
	.text
.L30:
	movq	start(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	start(%rip), %rax
	movq	%rax, %rdi
	call	free@PLT
	movl	count(%rip), %eax
	subl	$1, %eax
	movl	%eax, count(%rip)
	movq	$14, -8(%rbp)
	jmp	.L34
.L24:
	movl	$0, %eax
	jmp	.L35
.L22:
	movq	start(%rip), %rax
	movq	88(%rax), %rax
	testq	%rax, %rax
	jne	.L36
	movq	$4, -8(%rbp)
	jmp	.L34
.L36:
	movq	$10, -8(%rbp)
	jmp	.L34
.L27:
	movq	start(%rip), %rax
	testq	%rax, %rax
	jne	.L38
	movq	$6, -8(%rbp)
	jmp	.L34
.L38:
	movq	$15, -8(%rbp)
	jmp	.L34
.L31:
	movq	start(%rip), %rax
	jmp	.L35
.L26:
	movq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	-16(%rbp), %rax
	movq	$0, 88(%rax)
	movl	count(%rip), %eax
	subl	$1, %eax
	movl	%eax, count(%rip)
	movq	$3, -8(%rbp)
	jmp	.L34
.L29:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$7, -8(%rbp)
	jmp	.L34
.L25:
	movq	$0, -16(%rbp)
	movq	start(%rip), %rax
	movq	%rax, -24(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L34
.L33:
	movq	-24(%rbp), %rax
	movq	%rax, -16(%rbp)
	movq	-24(%rbp), %rax
	movq	88(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L34
.L28:
	movl	$0, %eax
	jmp	.L35
.L32:
	movq	-24(%rbp), %rax
	movq	88(%rax), %rax
	testq	%rax, %rax
	je	.L40
	movq	$0, -8(%rbp)
	jmp	.L34
.L40:
	movq	$9, -8(%rbp)
	jmp	.L34
.L43:
	nop
.L34:
	jmp	.L42
.L35:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	deleteend, .-deleteend
	.section	.rodata
.LC5:
	.string	"\nThe contents of SLL: "
.LC6:
	.string	"\n No of student nodes is %d \n"
	.align 8
.LC7:
	.string	"\n||%d|| USN:%s| Name:%s| Branch:%s| Sem:%d| Ph:%ld|"
	.align 8
.LC8:
	.string	"\nNo Contents to display in SLL "
	.text
	.globl	display
	.type	display, @function
display:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	$12, -8(%rbp)
.L62:
	cmpq	$12, -8(%rbp)
	ja	.L63
	movq	-8(%rbp), %rax
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
	.long	.L63-.L47
	.long	.L63-.L47
	.long	.L55-.L47
	.long	.L54-.L47
	.long	.L64-.L47
	.long	.L52-.L47
	.long	.L63-.L47
	.long	.L51-.L47
	.long	.L50-.L47
	.long	.L64-.L47
	.long	.L63-.L47
	.long	.L48-.L47
	.long	.L46-.L47
	.text
.L46:
	movl	$1, -20(%rbp)
	movq	$11, -8(%rbp)
	jmp	.L57
.L50:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	start(%rip), %rax
	movq	%rax, -16(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L57
.L54:
	movl	count(%rip), %eax
	movl	%eax, %esi
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$9, -8(%rbp)
	jmp	.L57
.L48:
	movq	start(%rip), %rax
	testq	%rax, %rax
	jne	.L58
	movq	$2, -8(%rbp)
	jmp	.L57
.L58:
	movq	$8, -8(%rbp)
	jmp	.L57
.L52:
	movq	-16(%rbp), %rax
	movq	80(%rax), %rsi
	movq	-16(%rbp), %rax
	movl	76(%rax), %edi
	movq	-16(%rbp), %rax
	leaq	50(%rax), %r8
	movq	-16(%rbp), %rax
	leaq	25(%rax), %rcx
	movq	-16(%rbp), %rdx
	movl	-20(%rbp), %eax
	subq	$8, %rsp
	pushq	%rsi
	movl	%edi, %r9d
	movl	%eax, %esi
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addq	$16, %rsp
	movq	-16(%rbp), %rax
	movq	88(%rax), %rax
	movq	%rax, -16(%rbp)
	addl	$1, -20(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L57
.L51:
	cmpq	$0, -16(%rbp)
	je	.L60
	movq	$5, -8(%rbp)
	jmp	.L57
.L60:
	movq	$3, -8(%rbp)
	jmp	.L57
.L55:
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$4, -8(%rbp)
	jmp	.L57
.L63:
	nop
.L57:
	jmp	.L62
.L64:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	display, .-display
	.section	.rodata
.LC9:
	.string	"\n~~~Stack Demo using SLL~~~"
	.align 8
.LC10:
	.string	"\n1:Push operation \n2: Pop operation \n3: Display \n4:Exit "
	.align 8
.LC11:
	.string	"\nEnter your choice for stack demo"
.LC12:
	.string	"%d"
	.text
	.globl	stackdemo
	.type	stackdemo, @function
stackdemo:
.LFB7:
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
	movq	$8, -16(%rbp)
.L82:
	cmpq	$11, -16(%rbp)
	ja	.L85
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L68(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L68(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L68:
	.long	.L74-.L68
	.long	.L73-.L68
	.long	.L85-.L68
	.long	.L85-.L68
	.long	.L72-.L68
	.long	.L71-.L68
	.long	.L85-.L68
	.long	.L85-.L68
	.long	.L70-.L68
	.long	.L86-.L68
	.long	.L85-.L68
	.long	.L67-.L68
	.text
.L72:
	call	display
	movq	$0, -16(%rbp)
	jmp	.L75
.L70:
	movq	$0, -16(%rbp)
	jmp	.L75
.L73:
	movl	-20(%rbp), %eax
	cmpl	$3, %eax
	je	.L76
	cmpl	$3, %eax
	jg	.L77
	cmpl	$1, %eax
	je	.L78
	cmpl	$2, %eax
	je	.L79
	jmp	.L77
.L76:
	movq	$4, -16(%rbp)
	jmp	.L80
.L79:
	movq	$5, -16(%rbp)
	jmp	.L80
.L78:
	movq	$11, -16(%rbp)
	jmp	.L80
.L77:
	movq	$9, -16(%rbp)
	nop
.L80:
	jmp	.L75
.L67:
	call	insertfront
	movq	%rax, start(%rip)
	movq	$0, -16(%rbp)
	jmp	.L75
.L71:
	call	deletefront
	movq	%rax, start(%rip)
	movq	$0, -16(%rbp)
	jmp	.L75
.L74:
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
	leaq	-20(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$1, -16(%rbp)
	jmp	.L75
.L85:
	nop
.L75:
	jmp	.L82
.L86:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L84
	call	__stack_chk_fail@PLT
.L84:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	stackdemo, .-stackdemo
	.section	.rodata
.LC13:
	.string	"\n~~~Menu~~~"
	.align 8
.LC14:
	.string	"\nEnter your choice for SLL operation "
	.align 8
.LC15:
	.string	"\n1:Create SLL of Student Nodes"
.LC16:
	.string	"\n2:DisplayStatus"
.LC17:
	.string	"\n3:InsertAtEnd"
.LC18:
	.string	"\n4:DeleteAtEnd"
	.align 8
.LC19:
	.string	"\n5:Stack Demo using SLL(Insertion and Deletion at Front)"
.LC20:
	.string	"\n6:Exit "
.LC21:
	.string	"\nEnter your choice:"
	.align 8
.LC22:
	.string	"\nEnter the no of students:    "
	.align 8
.LC23:
	.string	"\nPlease enter the valid choice"
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
.L88:
	movq	$0, start(%rip)
	nop
.L89:
	movq	$0, _TIG_IZ_Hah2_envp(%rip)
	nop
.L90:
	movq	$0, _TIG_IZ_Hah2_argv(%rip)
	nop
.L91:
	movl	$0, _TIG_IZ_Hah2_argc(%rip)
	nop
	nop
.L92:
.L93:
#APP
# 214 "githubber-me_dsathirdsem_7.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-Hah2--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_Hah2_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_Hah2_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_Hah2_envp(%rip)
	nop
	movq	$14, -16(%rbp)
.L120:
	cmpq	$22, -16(%rbp)
	ja	.L122
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L96(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L96(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L96:
	.long	.L122-.L96
	.long	.L107-.L96
	.long	.L122-.L96
	.long	.L122-.L96
	.long	.L106-.L96
	.long	.L122-.L96
	.long	.L105-.L96
	.long	.L122-.L96
	.long	.L104-.L96
	.long	.L122-.L96
	.long	.L103-.L96
	.long	.L102-.L96
	.long	.L122-.L96
	.long	.L122-.L96
	.long	.L101-.L96
	.long	.L100-.L96
	.long	.L122-.L96
	.long	.L122-.L96
	.long	.L99-.L96
	.long	.L98-.L96
	.long	.L97-.L96
	.long	.L122-.L96
	.long	.L95-.L96
	.text
.L99:
	movl	-24(%rbp), %eax
	cmpl	%eax, -20(%rbp)
	jg	.L108
	movq	$11, -16(%rbp)
	jmp	.L110
.L108:
	movq	$1, -16(%rbp)
	jmp	.L110
.L106:
	call	display
	movq	$1, -16(%rbp)
	jmp	.L110
.L101:
	movq	$1, -16(%rbp)
	jmp	.L110
.L100:
	movl	-28(%rbp), %eax
	cmpl	$6, %eax
	ja	.L111
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L113(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L113(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L113:
	.long	.L111-.L113
	.long	.L118-.L113
	.long	.L117-.L113
	.long	.L116-.L113
	.long	.L115-.L113
	.long	.L114-.L113
	.long	.L112-.L113
	.text
.L112:
	movq	$20, -16(%rbp)
	jmp	.L119
.L114:
	movq	$6, -16(%rbp)
	jmp	.L119
.L115:
	movq	$19, -16(%rbp)
	jmp	.L119
.L116:
	movq	$8, -16(%rbp)
	jmp	.L119
.L117:
	movq	$4, -16(%rbp)
	jmp	.L119
.L118:
	movq	$22, -16(%rbp)
	jmp	.L119
.L111:
	movq	$10, -16(%rbp)
	nop
.L119:
	jmp	.L110
.L104:
	call	insertend
	movq	%rax, start(%rip)
	movq	$1, -16(%rbp)
	jmp	.L110
.L107:
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC16(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC17(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC18(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC19(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC20(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC21(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-28(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$15, -16(%rbp)
	jmp	.L110
.L102:
	call	insertfront
	movq	%rax, start(%rip)
	addl	$1, -20(%rbp)
	movq	$18, -16(%rbp)
	jmp	.L110
.L98:
	call	deleteend
	movq	%rax, start(%rip)
	movq	$1, -16(%rbp)
	jmp	.L110
.L105:
	call	stackdemo
	movq	$1, -16(%rbp)
	jmp	.L110
.L95:
	leaq	.LC22(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	$1, -20(%rbp)
	movq	$18, -16(%rbp)
	jmp	.L110
.L103:
	leaq	.LC23(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -16(%rbp)
	jmp	.L110
.L97:
	movl	$0, %edi
	call	exit@PLT
.L122:
	nop
.L110:
	jmp	.L120
	.cfi_endproc
.LFE8:
	.size	main, .-main
	.globl	insertend
	.type	insertend, @function
insertend:
.LFB11:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	$5, -8(%rbp)
.L140:
	cmpq	$9, -8(%rbp)
	ja	.L141
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L126(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L126(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L126:
	.long	.L133-.L126
	.long	.L132-.L126
	.long	.L141-.L126
	.long	.L131-.L126
	.long	.L130-.L126
	.long	.L129-.L126
	.long	.L128-.L126
	.long	.L127-.L126
	.long	.L141-.L126
	.long	.L125-.L126
	.text
.L130:
	movq	-16(%rbp), %rax
	jmp	.L134
.L132:
	movq	start(%rip), %rax
	jmp	.L134
.L131:
	movq	-24(%rbp), %rax
	movq	-16(%rbp), %rdx
	movq	%rdx, 88(%rax)
	movq	$1, -8(%rbp)
	jmp	.L135
.L125:
	movq	-24(%rbp), %rax
	movq	88(%rax), %rax
	testq	%rax, %rax
	je	.L136
	movq	$7, -8(%rbp)
	jmp	.L135
.L136:
	movq	$3, -8(%rbp)
	jmp	.L135
.L128:
	movq	start(%rip), %rax
	movq	%rax, -24(%rbp)
	movq	$9, -8(%rbp)
	jmp	.L135
.L129:
	call	create
	movq	%rax, -16(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L135
.L133:
	movq	start(%rip), %rax
	testq	%rax, %rax
	jne	.L138
	movq	$4, -8(%rbp)
	jmp	.L135
.L138:
	movq	$6, -8(%rbp)
	jmp	.L135
.L127:
	movq	-24(%rbp), %rax
	movq	88(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	$9, -8(%rbp)
	jmp	.L135
.L141:
	nop
.L135:
	jmp	.L140
.L134:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE11:
	.size	insertend, .-insertend
	.globl	insertfront
	.type	insertfront, @function
insertfront:
.LFB12:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$1, -8(%rbp)
.L154:
	cmpq	$4, -8(%rbp)
	ja	.L155
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L145(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L145(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L145:
	.long	.L149-.L145
	.long	.L148-.L145
	.long	.L147-.L145
	.long	.L146-.L145
	.long	.L144-.L145
	.text
.L144:
	movq	start(%rip), %rdx
	movq	-16(%rbp), %rax
	movq	%rdx, 88(%rax)
	movq	$3, -8(%rbp)
	jmp	.L150
.L148:
	call	create
	movq	%rax, -16(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L150
.L146:
	movq	-16(%rbp), %rax
	jmp	.L151
.L149:
	movq	start(%rip), %rax
	testq	%rax, %rax
	jne	.L152
	movq	$2, -8(%rbp)
	jmp	.L150
.L152:
	movq	$4, -8(%rbp)
	jmp	.L150
.L147:
	movq	-16(%rbp), %rax
	jmp	.L151
.L155:
	nop
.L150:
	jmp	.L154
.L151:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE12:
	.size	insertfront, .-insertfront
	.section	.rodata
.LC24:
	.string	"\nMemory is not available"
	.align 8
.LC25:
	.string	"\nEnter the usn,Name,Branch, sem,PhoneNo of the student:"
.LC26:
	.string	"%s %s %s %d %ld"
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
	movq	$5, -16(%rbp)
.L169:
	cmpq	$7, -16(%rbp)
	ja	.L171
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L159(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L159(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L159:
	.long	.L164-.L159
	.long	.L171-.L159
	.long	.L163-.L159
	.long	.L162-.L159
	.long	.L171-.L159
	.long	.L161-.L159
	.long	.L160-.L159
	.long	.L158-.L159
	.text
.L162:
	movl	$96, %edi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	$6, -16(%rbp)
	jmp	.L165
.L160:
	cmpq	$0, -24(%rbp)
	jne	.L166
	movq	$7, -16(%rbp)
	jmp	.L165
.L166:
	movq	$2, -16(%rbp)
	jmp	.L165
.L161:
	movq	$3, -16(%rbp)
	jmp	.L165
.L164:
	movq	-24(%rbp), %rax
	jmp	.L170
.L158:
	leaq	.LC24(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$1, %edi
	call	exit@PLT
.L163:
	leaq	.LC25(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-24(%rbp), %rax
	leaq	80(%rax), %rdi
	movq	-24(%rbp), %rax
	leaq	76(%rax), %rsi
	movq	-24(%rbp), %rax
	leaq	50(%rax), %rcx
	movq	-24(%rbp), %rax
	leaq	25(%rax), %rdx
	movq	-24(%rbp), %rax
	movq	%rdi, %r9
	movq	%rsi, %r8
	movq	%rax, %rsi
	leaq	.LC26(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	-24(%rbp), %rax
	movq	$0, 88(%rax)
	movl	count(%rip), %eax
	addl	$1, %eax
	movl	%eax, count(%rip)
	movq	$0, -16(%rbp)
	jmp	.L165
.L171:
	nop
.L165:
	jmp	.L169
.L170:
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
