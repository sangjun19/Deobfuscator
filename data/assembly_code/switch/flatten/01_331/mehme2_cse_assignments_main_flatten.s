	.file	"mehme2_cse_assignments_main_flatten.c"
	.text
	.globl	_TIG_IZ_qARH_argv
	.bss
	.align 8
	.type	_TIG_IZ_qARH_argv, @object
	.size	_TIG_IZ_qARH_argv, 8
_TIG_IZ_qARH_argv:
	.zero	8
	.globl	_TIG_IZ_qARH_argc
	.align 4
	.type	_TIG_IZ_qARH_argc, @object
	.size	_TIG_IZ_qARH_argc, 4
_TIG_IZ_qARH_argc:
	.zero	4
	.globl	_TIG_IZ_qARH_envp
	.align 8
	.type	_TIG_IZ_qARH_envp, @object
	.size	_TIG_IZ_qARH_envp, 8
_TIG_IZ_qARH_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"title"
.LC1:
	.string	"publication year"
.LC2:
	.string	"%d"
.LC3:
	.string	"\nInvalid feature name: %s\n"
.LC4:
	.string	"author"
	.align 8
.LC5:
	.string	"\nNo matches found with ISBN: %s\n"
.LC6:
	.string	"year"
	.text
	.globl	updateBook
	.type	updateBook, @function
updateBook:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$80, %rsp
	movq	%rdi, -56(%rbp)
	movq	%rsi, -64(%rbp)
	movq	%rdx, -72(%rbp)
	movq	%rcx, -80(%rbp)
	movq	$23, -8(%rbp)
.L41:
	cmpq	$24, -8(%rbp)
	ja	.L42
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
	.long	.L24-.L4
	.long	.L23-.L4
	.long	.L22-.L4
	.long	.L21-.L4
	.long	.L20-.L4
	.long	.L19-.L4
	.long	.L18-.L4
	.long	.L17-.L4
	.long	.L42-.L4
	.long	.L16-.L4
	.long	.L15-.L4
	.long	.L14-.L4
	.long	.L13-.L4
	.long	.L42-.L4
	.long	.L42-.L4
	.long	.L42-.L4
	.long	.L12-.L4
	.long	.L11-.L4
	.long	.L10-.L4
	.long	.L9-.L4
	.long	.L8-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L43-.L4
	.text
.L10:
	movq	-16(%rbp), %rax
	movq	-64(%rbp), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -36(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L25
.L20:
	cmpl	$0, -36(%rbp)
	je	.L26
	movq	$5, -8(%rbp)
	jmp	.L25
.L26:
	movq	$7, -8(%rbp)
	jmp	.L25
.L13:
	movq	-72(%rbp), %rax
	leaq	.LC0(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -20(%rbp)
	movq	$17, -8(%rbp)
	jmp	.L25
.L23:
	movq	-72(%rbp), %rax
	leaq	.LC1(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -32(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L25
.L5:
	movq	-56(%rbp), %rax
	movq	%rax, -16(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L25
.L21:
	cmpl	$0, -32(%rbp)
	je	.L28
	movq	$10, -8(%rbp)
	jmp	.L25
.L28:
	movq	$20, -8(%rbp)
	jmp	.L25
.L12:
	movq	-16(%rbp), %rax
	leaq	112(%rax), %rdx
	movq	-80(%rbp), %rax
	leaq	.LC2(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_sscanf@PLT
	movq	$24, -8(%rbp)
	jmp	.L25
.L7:
	movq	-72(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$24, -8(%rbp)
	jmp	.L25
.L14:
	movq	-16(%rbp), %rax
	leaq	78(%rax), %rcx
	movq	-80(%rbp), %rax
	movl	$32, %edx
	movq	%rax, %rsi
	movq	%rcx, %rdi
	call	strncpy@PLT
	movq	-16(%rbp), %rax
	movb	$0, 109(%rax)
	movq	$24, -8(%rbp)
	jmp	.L25
.L16:
	movq	-72(%rbp), %rax
	leaq	.LC4(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -24(%rbp)
	movq	$22, -8(%rbp)
	jmp	.L25
.L9:
	movq	-16(%rbp), %rax
	leaq	14(%rax), %rcx
	movq	-80(%rbp), %rax
	movl	$64, %edx
	movq	%rax, %rsi
	movq	%rcx, %rdi
	call	strncpy@PLT
	movq	$24, -8(%rbp)
	jmp	.L25
.L11:
	cmpl	$0, -20(%rbp)
	je	.L31
	movq	$9, -8(%rbp)
	jmp	.L25
.L31:
	movq	$19, -8(%rbp)
	jmp	.L25
.L18:
	movq	-64(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$24, -8(%rbp)
	jmp	.L25
.L6:
	cmpl	$0, -24(%rbp)
	je	.L33
	movq	$1, -8(%rbp)
	jmp	.L25
.L33:
	movq	$11, -8(%rbp)
	jmp	.L25
.L19:
	movq	-16(%rbp), %rax
	movq	120(%rax), %rax
	movq	%rax, -16(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L25
.L15:
	movq	-72(%rbp), %rax
	leaq	.LC6(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -28(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L25
.L24:
	cmpq	$0, -16(%rbp)
	je	.L35
	movq	$18, -8(%rbp)
	jmp	.L25
.L35:
	movq	$7, -8(%rbp)
	jmp	.L25
.L17:
	cmpq	$0, -16(%rbp)
	je	.L37
	movq	$12, -8(%rbp)
	jmp	.L25
.L37:
	movq	$6, -8(%rbp)
	jmp	.L25
.L22:
	cmpl	$0, -28(%rbp)
	je	.L39
	movq	$21, -8(%rbp)
	jmp	.L25
.L39:
	movq	$16, -8(%rbp)
	jmp	.L25
.L8:
	movq	-16(%rbp), %rax
	leaq	112(%rax), %rdx
	movq	-80(%rbp), %rax
	leaq	.LC2(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_sscanf@PLT
	movq	$24, -8(%rbp)
	jmp	.L25
.L42:
	nop
.L25:
	jmp	.L41
.L43:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	updateBook, .-updateBook
	.globl	deleteBook
	.type	deleteBook, @function
deleteBook:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movq	%rsi, -48(%rbp)
	movq	$11, -8(%rbp)
.L64:
	cmpq	$12, -8(%rbp)
	ja	.L66
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
	.long	.L55-.L47
	.long	.L54-.L47
	.long	.L66-.L47
	.long	.L66-.L47
	.long	.L53-.L47
	.long	.L66-.L47
	.long	.L52-.L47
	.long	.L51-.L47
	.long	.L50-.L47
	.long	.L66-.L47
	.long	.L49-.L47
	.long	.L48-.L47
	.long	.L46-.L47
	.text
.L53:
	movq	-16(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, %rdx
	movq	-48(%rbp), %rax
	movq	%rax, %rsi
	movq	%rdx, %rdi
	call	strcmp@PLT
	movl	%eax, -20(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L56
.L46:
	movq	-16(%rbp), %rax
	movq	(%rax), %rax
	addq	$120, %rax
	movq	%rax, -16(%rbp)
	movq	$10, -8(%rbp)
	jmp	.L56
.L50:
	movq	-16(%rbp), %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	je	.L57
	movq	$7, -8(%rbp)
	jmp	.L56
.L57:
	movq	$6, -8(%rbp)
	jmp	.L56
.L54:
	cmpl	$0, -20(%rbp)
	je	.L59
	movq	$12, -8(%rbp)
	jmp	.L56
.L59:
	movq	$8, -8(%rbp)
	jmp	.L56
.L48:
	leaq	-40(%rbp), %rax
	movq	%rax, -16(%rbp)
	movq	$10, -8(%rbp)
	jmp	.L56
.L52:
	movq	-48(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$0, -8(%rbp)
	jmp	.L56
.L49:
	movq	-16(%rbp), %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	je	.L61
	movq	$4, -8(%rbp)
	jmp	.L56
.L61:
	movq	$8, -8(%rbp)
	jmp	.L56
.L55:
	movq	-40(%rbp), %rax
	jmp	.L65
.L51:
	movq	-16(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	-16(%rbp), %rax
	movq	(%rax), %rax
	movq	120(%rax), %rdx
	movq	-16(%rbp), %rax
	movq	%rdx, (%rax)
	movq	$0, -8(%rbp)
	jmp	.L56
.L66:
	nop
.L56:
	jmp	.L64
.L65:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	deleteBook, .-deleteBook
	.globl	filterAndSortBooks
	.type	filterAndSortBooks, @function
filterAndSortBooks:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$264, %rsp
	.cfi_offset 3, -24
	movq	%rdi, -248(%rbp)
	movl	%esi, -252(%rbp)
	movq	%rdx, -264(%rbp)
	movl	%ecx, -256(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -24(%rbp)
	xorl	%eax, %eax
	movq	$11, -176(%rbp)
.L147:
	cmpq	$51, -176(%rbp)
	ja	.L150
	movq	-176(%rbp), %rax
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
	.long	.L150-.L70
	.long	.L113-.L70
	.long	.L112-.L70
	.long	.L111-.L70
	.long	.L110-.L70
	.long	.L109-.L70
	.long	.L108-.L70
	.long	.L107-.L70
	.long	.L150-.L70
	.long	.L106-.L70
	.long	.L105-.L70
	.long	.L104-.L70
	.long	.L103-.L70
	.long	.L102-.L70
	.long	.L101-.L70
	.long	.L100-.L70
	.long	.L99-.L70
	.long	.L98-.L70
	.long	.L150-.L70
	.long	.L97-.L70
	.long	.L96-.L70
	.long	.L95-.L70
	.long	.L94-.L70
	.long	.L93-.L70
	.long	.L92-.L70
	.long	.L91-.L70
	.long	.L90-.L70
	.long	.L89-.L70
	.long	.L88-.L70
	.long	.L87-.L70
	.long	.L86-.L70
	.long	.L85-.L70
	.long	.L84-.L70
	.long	.L83-.L70
	.long	.L150-.L70
	.long	.L150-.L70
	.long	.L150-.L70
	.long	.L82-.L70
	.long	.L81-.L70
	.long	.L80-.L70
	.long	.L150-.L70
	.long	.L79-.L70
	.long	.L78-.L70
	.long	.L77-.L70
	.long	.L150-.L70
	.long	.L76-.L70
	.long	.L75-.L70
	.long	.L74-.L70
	.long	.L73-.L70
	.long	.L72-.L70
	.long	.L71-.L70
	.long	.L69-.L70
	.text
.L71:
	cmpl	$0, -236(%rbp)
	jne	.L114
	movq	$24, -176(%rbp)
	jmp	.L116
.L114:
	movq	$48, -176(%rbp)
	jmp	.L116
.L91:
	movq	-200(%rbp), %rax
	movq	(%rax), %rax
	addq	$120, %rax
	movq	%rax, -200(%rbp)
	movq	$49, -176(%rbp)
	jmp	.L116
.L72:
	movq	-200(%rbp), %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	je	.L117
	movq	$46, -176(%rbp)
	jmp	.L116
.L117:
	movq	$41, -176(%rbp)
	jmp	.L116
.L110:
	leaq	-240(%rbp), %rdx
	movq	-264(%rbp), %rax
	leaq	.LC2(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_sscanf@PLT
	movq	$27, -176(%rbp)
	jmp	.L116
.L86:
	cmpl	$0, -224(%rbp)
	je	.L119
	movq	$17, -176(%rbp)
	jmp	.L116
.L119:
	movq	$19, -176(%rbp)
	jmp	.L116
.L101:
	movl	-212(%rbp), %eax
	movl	%eax, -232(%rbp)
	movq	$6, -176(%rbp)
	jmp	.L116
.L100:
	movl	$1, -204(%rbp)
	movq	$7, -176(%rbp)
	jmp	.L116
.L85:
	movq	-184(%rbp), %rax
	movq	%rax, -192(%rbp)
	movq	$3, -176(%rbp)
	jmp	.L116
.L103:
	cmpl	$0, -228(%rbp)
	jns	.L121
	movq	$31, -176(%rbp)
	jmp	.L116
.L121:
	movq	$37, -176(%rbp)
	jmp	.L116
.L76:
	movq	-192(%rbp), %rax
	leaq	78(%rax), %rdx
	movq	-184(%rbp), %rax
	addq	$78, %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -228(%rbp)
	movq	$12, -176(%rbp)
	jmp	.L116
.L113:
	movq	-192(%rbp), %rax
	movq	120(%rax), %rax
	movq	%rax, -184(%rbp)
	movq	$26, -176(%rbp)
	jmp	.L116
.L93:
	cmpl	$0, -208(%rbp)
	je	.L123
	movq	$39, -176(%rbp)
	jmp	.L116
.L123:
	movq	$15, -176(%rbp)
	jmp	.L116
.L111:
	cmpl	$0, -252(%rbp)
	jne	.L125
	movq	$13, -176(%rbp)
	jmp	.L116
.L125:
	movq	$28, -176(%rbp)
	jmp	.L116
.L99:
	movq	-200(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, %rdx
	movq	-192(%rbp), %rax
	cmpq	%rax, %rdx
	je	.L127
	movq	$47, -176(%rbp)
	jmp	.L116
.L127:
	movq	$25, -176(%rbp)
	jmp	.L116
.L92:
	movl	$-1, -228(%rbp)
	movq	$12, -176(%rbp)
	jmp	.L116
.L95:
	movq	-184(%rbp), %rax
	movl	112(%rax), %edx
	movq	-192(%rbp), %rax
	movl	112(%rax), %ecx
	movl	%edx, %eax
	subl	%ecx, %eax
	movl	%eax, -228(%rbp)
	movq	$12, -176(%rbp)
	jmp	.L116
.L90:
	cmpq	$0, -184(%rbp)
	je	.L129
	movq	$5, -176(%rbp)
	jmp	.L116
.L129:
	movq	$16, -176(%rbp)
	jmp	.L116
.L104:
	cmpl	$1, -252(%rbp)
	jne	.L131
	movq	$4, -176(%rbp)
	jmp	.L116
.L131:
	movq	$27, -176(%rbp)
	jmp	.L116
.L106:
	cmpl	$0, -216(%rbp)
	je	.L133
	movq	$33, -176(%rbp)
	jmp	.L116
.L133:
	movq	$22, -176(%rbp)
	jmp	.L116
.L102:
	movq	-192(%rbp), %rax
	leaq	78(%rax), %rdx
	movq	-264(%rbp), %rax
	movq	%rax, %rsi
	movq	%rdx, %rdi
	call	strcmp@PLT
	movl	%eax, -208(%rbp)
	movq	$23, -176(%rbp)
	jmp	.L116
.L69:
	movq	-192(%rbp), %rax
	movl	112(%rax), %edx
	movl	-240(%rbp), %eax
	cmpl	%eax, %edx
	sete	%al
	movzbl	%al, %eax
	movl	%eax, -236(%rbp)
	movq	$1, -176(%rbp)
	jmp	.L116
.L97:
	movl	$1, -220(%rbp)
	movq	$29, -176(%rbp)
	jmp	.L116
.L84:
	movq	-192(%rbp), %rax
	leaq	14(%rax), %rdx
	movq	-184(%rbp), %rax
	addq	$14, %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -228(%rbp)
	movq	$12, -176(%rbp)
	jmp	.L116
.L98:
	movl	$0, -220(%rbp)
	movq	$29, -176(%rbp)
	jmp	.L116
.L108:
	cmpl	$0, -232(%rbp)
	je	.L135
	movq	$50, -176(%rbp)
	jmp	.L116
.L135:
	movq	$38, -176(%rbp)
	jmp	.L116
.L89:
	leaq	-248(%rbp), %rax
	movq	%rax, -200(%rbp)
	movq	$49, -176(%rbp)
	jmp	.L116
.L81:
	movl	$0, -228(%rbp)
	movq	$12, -176(%rbp)
	jmp	.L116
.L73:
	cmpl	$2, -256(%rbp)
	je	.L137
	cmpl	$2, -256(%rbp)
	jg	.L138
	cmpl	$0, -256(%rbp)
	je	.L139
	cmpl	$1, -256(%rbp)
	je	.L140
	jmp	.L138
.L137:
	movq	$21, -176(%rbp)
	jmp	.L141
.L140:
	movq	$45, -176(%rbp)
	jmp	.L141
.L139:
	movq	$32, -176(%rbp)
	jmp	.L141
.L138:
	movq	$43, -176(%rbp)
	nop
.L141:
	jmp	.L116
.L94:
	movl	$1, -212(%rbp)
	movq	$14, -176(%rbp)
	jmp	.L116
.L88:
	movq	-192(%rbp), %rax
	movl	112(%rax), %edx
	movl	-240(%rbp), %eax
	cmpl	%eax, %edx
	sete	%al
	movzbl	%al, %eax
	movl	%eax, -236(%rbp)
	movq	$37, -176(%rbp)
	jmp	.L116
.L74:
	movq	-200(%rbp), %rax
	movq	(%rax), %rax
	movq	(%rax), %rcx
	movq	8(%rax), %rbx
	movq	%rcx, -160(%rbp)
	movq	%rbx, -152(%rbp)
	movq	16(%rax), %rcx
	movq	24(%rax), %rbx
	movq	%rcx, -144(%rbp)
	movq	%rbx, -136(%rbp)
	movq	32(%rax), %rcx
	movq	40(%rax), %rbx
	movq	%rcx, -128(%rbp)
	movq	%rbx, -120(%rbp)
	movq	48(%rax), %rcx
	movq	56(%rax), %rbx
	movq	%rcx, -112(%rbp)
	movq	%rbx, -104(%rbp)
	movq	64(%rax), %rcx
	movq	72(%rax), %rbx
	movq	%rcx, -96(%rbp)
	movq	%rbx, -88(%rbp)
	movq	80(%rax), %rcx
	movq	88(%rax), %rbx
	movq	%rcx, -80(%rbp)
	movq	%rbx, -72(%rbp)
	movq	96(%rax), %rcx
	movq	104(%rax), %rbx
	movq	%rcx, -64(%rbp)
	movq	%rbx, -56(%rbp)
	movq	120(%rax), %rdx
	movq	112(%rax), %rax
	movq	%rax, -48(%rbp)
	movq	%rdx, -40(%rbp)
	movq	-200(%rbp), %rax
	movq	(%rax), %rax
	movq	120(%rax), %rax
	movq	%rax, -168(%rbp)
	movq	-200(%rbp), %rax
	movq	(%rax), %rax
	movq	-192(%rbp), %rdx
	movq	(%rdx), %rcx
	movq	8(%rdx), %rbx
	movq	%rcx, (%rax)
	movq	%rbx, 8(%rax)
	movq	16(%rdx), %rcx
	movq	24(%rdx), %rbx
	movq	%rcx, 16(%rax)
	movq	%rbx, 24(%rax)
	movq	32(%rdx), %rcx
	movq	40(%rdx), %rbx
	movq	%rcx, 32(%rax)
	movq	%rbx, 40(%rax)
	movq	48(%rdx), %rcx
	movq	56(%rdx), %rbx
	movq	%rcx, 48(%rax)
	movq	%rbx, 56(%rax)
	movq	64(%rdx), %rcx
	movq	72(%rdx), %rbx
	movq	%rcx, 64(%rax)
	movq	%rbx, 72(%rax)
	movq	80(%rdx), %rcx
	movq	88(%rdx), %rbx
	movq	%rcx, 80(%rax)
	movq	%rbx, 88(%rax)
	movq	96(%rdx), %rcx
	movq	104(%rdx), %rbx
	movq	%rcx, 96(%rax)
	movq	%rbx, 104(%rax)
	movq	112(%rdx), %rcx
	movq	120(%rdx), %rbx
	movq	%rcx, 112(%rax)
	movq	%rbx, 120(%rax)
	movq	-192(%rbp), %rax
	movq	-160(%rbp), %rcx
	movq	-152(%rbp), %rbx
	movq	%rcx, (%rax)
	movq	%rbx, 8(%rax)
	movq	-144(%rbp), %rcx
	movq	-136(%rbp), %rbx
	movq	%rcx, 16(%rax)
	movq	%rbx, 24(%rax)
	movq	-128(%rbp), %rcx
	movq	-120(%rbp), %rbx
	movq	%rcx, 32(%rax)
	movq	%rbx, 40(%rax)
	movq	-112(%rbp), %rcx
	movq	-104(%rbp), %rbx
	movq	%rcx, 48(%rax)
	movq	%rbx, 56(%rax)
	movq	-96(%rbp), %rcx
	movq	-88(%rbp), %rbx
	movq	%rcx, 64(%rax)
	movq	%rbx, 72(%rax)
	movq	-80(%rbp), %rcx
	movq	-72(%rbp), %rbx
	movq	%rcx, 80(%rax)
	movq	%rbx, 88(%rax)
	movq	-64(%rbp), %rcx
	movq	-56(%rbp), %rbx
	movq	%rcx, 96(%rax)
	movq	%rbx, 104(%rax)
	movq	-48(%rbp), %rcx
	movq	-40(%rbp), %rbx
	movq	%rcx, 112(%rax)
	movq	%rbx, 120(%rax)
	movq	-200(%rbp), %rax
	movq	(%rax), %rax
	movq	120(%rax), %rdx
	movq	-192(%rbp), %rax
	movq	%rdx, 120(%rax)
	movq	-200(%rbp), %rax
	movq	(%rax), %rax
	movq	-168(%rbp), %rdx
	movq	%rdx, 120(%rax)
	movq	$25, -176(%rbp)
	jmp	.L116
.L109:
	cmpl	$0, -252(%rbp)
	jne	.L142
	movq	$20, -176(%rbp)
	jmp	.L116
.L142:
	movq	$42, -176(%rbp)
	jmp	.L116
.L83:
	movl	$0, -212(%rbp)
	movq	$14, -176(%rbp)
	jmp	.L116
.L82:
	movq	-184(%rbp), %rax
	movq	120(%rax), %rax
	movq	%rax, -184(%rbp)
	movq	$26, -176(%rbp)
	jmp	.L116
.L79:
	movq	-248(%rbp), %rax
	movq	-24(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L148
	jmp	.L149
.L105:
	cmpl	$0, -252(%rbp)
	jne	.L145
	movq	$2, -176(%rbp)
	jmp	.L116
.L145:
	movq	$51, -176(%rbp)
	jmp	.L116
.L78:
	movq	-184(%rbp), %rax
	movl	112(%rax), %edx
	movl	-240(%rbp), %eax
	cmpl	%eax, %edx
	sete	%al
	movzbl	%al, %eax
	movl	%eax, -232(%rbp)
	movq	$6, -176(%rbp)
	jmp	.L116
.L75:
	movq	-200(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, -192(%rbp)
	movq	$10, -176(%rbp)
	jmp	.L116
.L80:
	movl	$0, -204(%rbp)
	movq	$7, -176(%rbp)
	jmp	.L116
.L107:
	movl	-204(%rbp), %eax
	movl	%eax, -236(%rbp)
	movq	$37, -176(%rbp)
	jmp	.L116
.L87:
	movl	-220(%rbp), %eax
	movl	%eax, -236(%rbp)
	movq	$1, -176(%rbp)
	jmp	.L116
.L77:
	movq	$12, -176(%rbp)
	jmp	.L116
.L112:
	movq	-192(%rbp), %rax
	leaq	78(%rax), %rdx
	movq	-264(%rbp), %rax
	movq	%rax, %rsi
	movq	%rdx, %rdi
	call	strcmp@PLT
	movl	%eax, -224(%rbp)
	movq	$30, -176(%rbp)
	jmp	.L116
.L96:
	movq	-184(%rbp), %rax
	leaq	78(%rax), %rdx
	movq	-264(%rbp), %rax
	movq	%rax, %rsi
	movq	%rdx, %rdi
	call	strcmp@PLT
	movl	%eax, -216(%rbp)
	movq	$9, -176(%rbp)
	jmp	.L116
.L150:
	nop
.L116:
	jmp	.L147
.L149:
	call	__stack_chk_fail@PLT
.L148:
	movq	-8(%rbp), %rbx
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	filterAndSortBooks, .-filterAndSortBooks
	.globl	addBook
	.type	addBook, @function
addBook:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$80, %rsp
	movq	%rdi, -40(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	%rcx, -64(%rbp)
	movl	%r8d, -68(%rbp)
	movl	%r9d, -72(%rbp)
	movq	$7, -16(%rbp)
.L169:
	cmpq	$12, -16(%rbp)
	ja	.L171
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L154(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L154(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L154:
	.long	.L162-.L154
	.long	.L161-.L154
	.long	.L160-.L154
	.long	.L171-.L154
	.long	.L171-.L154
	.long	.L171-.L154
	.long	.L159-.L154
	.long	.L158-.L154
	.long	.L157-.L154
	.long	.L156-.L154
	.long	.L155-.L154
	.long	.L171-.L154
	.long	.L153-.L154
	.text
.L153:
	movq	-32(%rbp), %rax
	movq	$0, 120(%rax)
	leaq	-40(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	$10, -16(%rbp)
	jmp	.L163
.L157:
	movq	-24(%rbp), %rax
	movq	-32(%rbp), %rdx
	movq	%rdx, (%rax)
	movq	$1, -16(%rbp)
	jmp	.L163
.L161:
	movq	-40(%rbp), %rax
	jmp	.L170
.L156:
	movq	-40(%rbp), %rdx
	movq	-32(%rbp), %rax
	movq	%rdx, 120(%rax)
	movq	-32(%rbp), %rax
	movq	%rax, -40(%rbp)
	movq	$1, -16(%rbp)
	jmp	.L163
.L159:
	movl	$128, %edi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -32(%rbp)
	movq	-32(%rbp), %rax
	movq	-48(%rbp), %rcx
	movl	$14, %edx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	strncpy@PLT
	movq	-32(%rbp), %rax
	leaq	14(%rax), %rcx
	movq	-56(%rbp), %rax
	movl	$64, %edx
	movq	%rax, %rsi
	movq	%rcx, %rdi
	call	strncpy@PLT
	movq	-32(%rbp), %rax
	leaq	78(%rax), %rcx
	movq	-64(%rbp), %rax
	movl	$32, %edx
	movq	%rax, %rsi
	movq	%rcx, %rdi
	call	strncpy@PLT
	movq	-32(%rbp), %rax
	movl	-68(%rbp), %edx
	movl	%edx, 112(%rax)
	movq	-32(%rbp), %rax
	movl	$0, 116(%rax)
	movq	$0, -16(%rbp)
	jmp	.L163
.L155:
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	je	.L165
	movq	$2, -16(%rbp)
	jmp	.L163
.L165:
	movq	$8, -16(%rbp)
	jmp	.L163
.L162:
	cmpl	$1, -72(%rbp)
	jne	.L167
	movq	$9, -16(%rbp)
	jmp	.L163
.L167:
	movq	$12, -16(%rbp)
	jmp	.L163
.L158:
	movq	$6, -16(%rbp)
	jmp	.L163
.L160:
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	addq	$120, %rax
	movq	%rax, -24(%rbp)
	movq	$10, -16(%rbp)
	jmp	.L163
.L171:
	nop
.L163:
	jmp	.L169
.L170:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	addBook, .-addBook
	.section	.rodata
.LC7:
	.string	"Yes"
.LC8:
	.string	"\nNo results."
.LC9:
	.string	"No"
	.align 8
.LC10:
	.string	"\nISBN: %s | Title: %s | Author: %s | Publication Year: %d | Borrowed: %s\n"
	.text
	.globl	searchBooks
	.type	searchBooks, @function
searchBooks:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$96, %rsp
	movq	%rdi, -72(%rbp)
	movl	%esi, -76(%rbp)
	movq	%rdx, -88(%rbp)
	movq	$35, -8(%rbp)
.L225:
	cmpq	$37, -8(%rbp)
	ja	.L226
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L175(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L175(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L175:
	.long	.L203-.L175
	.long	.L202-.L175
	.long	.L226-.L175
	.long	.L201-.L175
	.long	.L200-.L175
	.long	.L199-.L175
	.long	.L198-.L175
	.long	.L197-.L175
	.long	.L196-.L175
	.long	.L195-.L175
	.long	.L194-.L175
	.long	.L226-.L175
	.long	.L193-.L175
	.long	.L192-.L175
	.long	.L191-.L175
	.long	.L190-.L175
	.long	.L189-.L175
	.long	.L226-.L175
	.long	.L227-.L175
	.long	.L226-.L175
	.long	.L187-.L175
	.long	.L186-.L175
	.long	.L185-.L175
	.long	.L184-.L175
	.long	.L226-.L175
	.long	.L183-.L175
	.long	.L182-.L175
	.long	.L181-.L175
	.long	.L226-.L175
	.long	.L180-.L175
	.long	.L179-.L175
	.long	.L226-.L175
	.long	.L178-.L175
	.long	.L177-.L175
	.long	.L226-.L175
	.long	.L176-.L175
	.long	.L226-.L175
	.long	.L174-.L175
	.text
.L183:
	cmpl	$0, -48(%rbp)
	je	.L205
	movq	$20, -8(%rbp)
	jmp	.L207
.L205:
	movq	$6, -8(%rbp)
	jmp	.L207
.L200:
	cmpl	$2, -76(%rbp)
	je	.L208
	cmpl	$2, -76(%rbp)
	jg	.L209
	cmpl	$0, -76(%rbp)
	je	.L210
	cmpl	$1, -76(%rbp)
	je	.L211
	jmp	.L209
.L208:
	movq	$10, -8(%rbp)
	jmp	.L212
.L211:
	movq	$32, -8(%rbp)
	jmp	.L212
.L210:
	movq	$13, -8(%rbp)
	jmp	.L212
.L209:
	movq	$1, -8(%rbp)
	nop
.L212:
	jmp	.L207
.L179:
	cmpq	$0, -24(%rbp)
	je	.L213
	movq	$4, -8(%rbp)
	jmp	.L207
.L213:
	movq	$23, -8(%rbp)
	jmp	.L207
.L191:
	leaq	.LC7(%rip), %rax
	movq	%rax, -16(%rbp)
	movq	$29, -8(%rbp)
	jmp	.L207
.L190:
	movl	$1, -36(%rbp)
	movq	$21, -8(%rbp)
	jmp	.L207
.L193:
	movl	-44(%rbp), %eax
	movl	%eax, -52(%rbp)
	movl	-52(%rbp), %eax
	addl	%eax, -56(%rbp)
	movq	$27, -8(%rbp)
	jmp	.L207
.L196:
	movl	$0, -56(%rbp)
	movq	-72(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	$30, -8(%rbp)
	jmp	.L207
.L202:
	movl	$0, -52(%rbp)
	movq	$27, -8(%rbp)
	jmp	.L207
.L184:
	cmpl	$0, -56(%rbp)
	jne	.L215
	movq	$9, -8(%rbp)
	jmp	.L207
.L215:
	movq	$18, -8(%rbp)
	jmp	.L207
.L201:
	movl	$1, -28(%rbp)
	movq	$22, -8(%rbp)
	jmp	.L207
.L189:
	movl	$0, -28(%rbp)
	movq	$22, -8(%rbp)
	jmp	.L207
.L186:
	movl	-36(%rbp), %eax
	movl	%eax, -52(%rbp)
	movl	-52(%rbp), %eax
	addl	%eax, -56(%rbp)
	movq	$27, -8(%rbp)
	jmp	.L207
.L182:
	movq	-24(%rbp), %rax
	movq	120(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	$30, -8(%rbp)
	jmp	.L207
.L195:
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$18, -8(%rbp)
	jmp	.L207
.L192:
	movq	-24(%rbp), %rdx
	movq	-88(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -48(%rbp)
	movq	$25, -8(%rbp)
	jmp	.L207
.L178:
	movq	-24(%rbp), %rax
	leaq	78(%rax), %rdx
	movq	-88(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -40(%rbp)
	movq	$37, -8(%rbp)
	jmp	.L207
.L198:
	movl	$1, -44(%rbp)
	movq	$12, -8(%rbp)
	jmp	.L207
.L181:
	cmpl	$0, -52(%rbp)
	je	.L217
	movq	$7, -8(%rbp)
	jmp	.L207
.L217:
	movq	$26, -8(%rbp)
	jmp	.L207
.L185:
	movl	-28(%rbp), %eax
	movl	%eax, -52(%rbp)
	movl	-52(%rbp), %eax
	addl	%eax, -56(%rbp)
	movq	$27, -8(%rbp)
	jmp	.L207
.L199:
	cmpl	$0, -32(%rbp)
	je	.L219
	movq	$16, -8(%rbp)
	jmp	.L207
.L219:
	movq	$3, -8(%rbp)
	jmp	.L207
.L177:
	movl	$0, -36(%rbp)
	movq	$21, -8(%rbp)
	jmp	.L207
.L174:
	cmpl	$0, -40(%rbp)
	je	.L221
	movq	$33, -8(%rbp)
	jmp	.L207
.L221:
	movq	$15, -8(%rbp)
	jmp	.L207
.L194:
	movq	-24(%rbp), %rax
	leaq	14(%rax), %rdx
	movq	-88(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -32(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L207
.L203:
	leaq	.LC9(%rip), %rax
	movq	%rax, -16(%rbp)
	movq	$29, -8(%rbp)
	jmp	.L207
.L197:
	movq	-24(%rbp), %rax
	movl	116(%rax), %eax
	testl	%eax, %eax
	je	.L223
	movq	$14, -8(%rbp)
	jmp	.L207
.L223:
	movq	$0, -8(%rbp)
	jmp	.L207
.L176:
	movq	$8, -8(%rbp)
	jmp	.L207
.L180:
	movq	-24(%rbp), %rax
	movl	112(%rax), %esi
	movq	-24(%rbp), %rax
	leaq	78(%rax), %rcx
	movq	-24(%rbp), %rax
	leaq	14(%rax), %rdx
	movq	-24(%rbp), %rax
	movq	-16(%rbp), %rdi
	movq	%rdi, %r9
	movl	%esi, %r8d
	movq	%rax, %rsi
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$26, -8(%rbp)
	jmp	.L207
.L187:
	movl	$0, -44(%rbp)
	movq	$12, -8(%rbp)
	jmp	.L207
.L226:
	nop
.L207:
	jmp	.L225
.L227:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	searchBooks, .-searchBooks
	.section	.rodata
.LC11:
	.string	"\nNo books borrowed by %s.\n"
.LC12:
	.string	"%[^,]"
	.align 8
.LC13:
	.string	"\nISBN: %s | Title: %s | Author: %s | Publication Year: %d\n"
	.text
	.globl	displayBooks
	.type	displayBooks, @function
displayBooks:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$96, %rsp
	movq	%rdi, -72(%rbp)
	movq	%rsi, -80(%rbp)
	movl	%edx, -84(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$31, -32(%rbp)
.L273:
	cmpq	$31, -32(%rbp)
	ja	.L276
	movq	-32(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L231(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L231(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L231:
	.long	.L276-.L231
	.long	.L250-.L231
	.long	.L249-.L231
	.long	.L248-.L231
	.long	.L247-.L231
	.long	.L276-.L231
	.long	.L246-.L231
	.long	.L276-.L231
	.long	.L276-.L231
	.long	.L245-.L231
	.long	.L244-.L231
	.long	.L276-.L231
	.long	.L276-.L231
	.long	.L276-.L231
	.long	.L276-.L231
	.long	.L243-.L231
	.long	.L242-.L231
	.long	.L276-.L231
	.long	.L277-.L231
	.long	.L240-.L231
	.long	.L239-.L231
	.long	.L238-.L231
	.long	.L276-.L231
	.long	.L237-.L231
	.long	.L236-.L231
	.long	.L235-.L231
	.long	.L276-.L231
	.long	.L234-.L231
	.long	.L233-.L231
	.long	.L276-.L231
	.long	.L232-.L231
	.long	.L230-.L231
	.text
.L235:
	movq	-48(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$18, -32(%rbp)
	jmp	.L252
.L247:
	movq	-48(%rbp), %rax
	movl	32(%rax), %eax
	cmpl	%eax, -84(%rbp)
	je	.L253
	movq	$21, -32(%rbp)
	jmp	.L252
.L253:
	movq	$9, -32(%rbp)
	jmp	.L252
.L232:
	cmpl	$0, -52(%rbp)
	je	.L255
	movq	$23, -32(%rbp)
	jmp	.L252
.L255:
	movq	$10, -32(%rbp)
	jmp	.L252
.L243:
	cmpq	$0, -40(%rbp)
	je	.L257
	movq	$2, -32(%rbp)
	jmp	.L252
.L257:
	movq	$10, -32(%rbp)
	jmp	.L252
.L230:
	movq	-72(%rbp), %rax
	movq	%rax, -48(%rbp)
	movq	$19, -32(%rbp)
	jmp	.L252
.L250:
	movq	-48(%rbp), %rax
	movq	40(%rax), %rdx
	movl	-56(%rbp), %eax
	cltq
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	je	.L259
	movq	$24, -32(%rbp)
	jmp	.L252
.L259:
	movq	$28, -32(%rbp)
	jmp	.L252
.L237:
	movq	-40(%rbp), %rax
	movq	120(%rax), %rax
	movq	%rax, -40(%rbp)
	movq	$15, -32(%rbp)
	jmp	.L252
.L248:
	movq	-48(%rbp), %rax
	movq	40(%rax), %rdx
	movl	-56(%rbp), %eax
	cltq
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	cmpb	$44, %al
	je	.L261
	movq	$1, -32(%rbp)
	jmp	.L252
.L261:
	movq	$28, -32(%rbp)
	jmp	.L252
.L242:
	addl	$1, -56(%rbp)
	movq	-48(%rbp), %rax
	movq	40(%rax), %rdx
	movl	-56(%rbp), %eax
	cltq
	leaq	(%rdx,%rax), %rcx
	leaq	-22(%rbp), %rax
	movq	%rax, %rdx
	leaq	.LC12(%rip), %rax
	movq	%rax, %rsi
	movq	%rcx, %rdi
	movl	$0, %eax
	call	__isoc99_sscanf@PLT
	movq	-80(%rbp), %rax
	movq	%rax, -40(%rbp)
	movq	$15, -32(%rbp)
	jmp	.L252
.L236:
	addl	$1, -56(%rbp)
	movq	$3, -32(%rbp)
	jmp	.L252
.L238:
	movq	-48(%rbp), %rax
	movq	48(%rax), %rax
	movq	%rax, -48(%rbp)
	movq	$19, -32(%rbp)
	jmp	.L252
.L245:
	cmpq	$0, -48(%rbp)
	je	.L263
	movq	$20, -32(%rbp)
	jmp	.L252
.L263:
	movq	$18, -32(%rbp)
	jmp	.L252
.L240:
	cmpq	$0, -48(%rbp)
	je	.L265
	movq	$4, -32(%rbp)
	jmp	.L252
.L265:
	movq	$9, -32(%rbp)
	jmp	.L252
.L246:
	movq	-40(%rbp), %rax
	movl	112(%rax), %esi
	movq	-40(%rbp), %rax
	leaq	78(%rax), %rcx
	movq	-40(%rbp), %rax
	leaq	14(%rax), %rdx
	movq	-40(%rbp), %rax
	movl	%esi, %r8d
	movq	%rax, %rsi
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -32(%rbp)
	jmp	.L252
.L234:
	movl	$0, -56(%rbp)
	movq	$28, -32(%rbp)
	jmp	.L252
.L233:
	movq	-48(%rbp), %rax
	movq	40(%rax), %rdx
	movl	-56(%rbp), %eax
	cltq
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	je	.L267
	movq	$16, -32(%rbp)
	jmp	.L252
.L267:
	movq	$18, -32(%rbp)
	jmp	.L252
.L244:
	cmpq	$0, -40(%rbp)
	je	.L269
	movq	$6, -32(%rbp)
	jmp	.L252
.L269:
	movq	$3, -32(%rbp)
	jmp	.L252
.L249:
	movq	-40(%rbp), %rax
	leaq	-22(%rbp), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -52(%rbp)
	movq	$30, -32(%rbp)
	jmp	.L252
.L239:
	movq	-48(%rbp), %rax
	movq	40(%rax), %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	jne	.L271
	movq	$25, -32(%rbp)
	jmp	.L252
.L271:
	movq	$27, -32(%rbp)
	jmp	.L252
.L276:
	nop
.L252:
	jmp	.L273
.L277:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L275
	call	__stack_chk_fail@PLT
.L275:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	displayBooks, .-displayBooks
	.globl	reverseBookList
	.type	reverseBookList, @function
reverseBookList:
.LFB10:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -40(%rbp)
	movq	$5, -8(%rbp)
.L290:
	cmpq	$5, -8(%rbp)
	ja	.L292
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L281(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L281(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L281:
	.long	.L285-.L281
	.long	.L284-.L281
	.long	.L283-.L281
	.long	.L292-.L281
	.long	.L282-.L281
	.long	.L280-.L281
	.text
.L282:
	cmpq	$0, -24(%rbp)
	je	.L286
	movq	$0, -8(%rbp)
	jmp	.L288
.L286:
	movq	$2, -8(%rbp)
	jmp	.L288
.L284:
	movq	$0, -16(%rbp)
	movq	-40(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L288
.L280:
	movq	$1, -8(%rbp)
	jmp	.L288
.L285:
	movq	-24(%rbp), %rax
	movq	120(%rax), %rax
	movq	%rax, -40(%rbp)
	movq	-24(%rbp), %rax
	movq	-16(%rbp), %rdx
	movq	%rdx, 120(%rax)
	movq	-24(%rbp), %rax
	movq	%rax, -16(%rbp)
	movq	-40(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L288
.L283:
	movq	-16(%rbp), %rax
	jmp	.L291
.L292:
	nop
.L288:
	jmp	.L290
.L291:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	reverseBookList, .-reverseBookList
	.globl	addStudent
	.type	addStudent, @function
addStudent:
.LFB11:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$80, %rsp
	movq	%rdi, -56(%rbp)
	movq	%rsi, -64(%rbp)
	movl	%edx, -68(%rbp)
	movl	%ecx, -72(%rbp)
	movq	$8, -24(%rbp)
.L311:
	cmpq	$11, -24(%rbp)
	ja	.L313
	movq	-24(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L296(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L296(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L296:
	.long	.L313-.L296
	.long	.L313-.L296
	.long	.L304-.L296
	.long	.L303-.L296
	.long	.L313-.L296
	.long	.L302-.L296
	.long	.L301-.L296
	.long	.L300-.L296
	.long	.L299-.L296
	.long	.L298-.L296
	.long	.L297-.L296
	.long	.L295-.L296
	.text
.L299:
	movq	$2, -24(%rbp)
	jmp	.L305
.L303:
	movq	-40(%rbp), %rax
	movq	$0, 48(%rax)
	leaq	-56(%rbp), %rax
	movq	%rax, -32(%rbp)
	movq	$10, -24(%rbp)
	jmp	.L305
.L295:
	cmpl	$1, -72(%rbp)
	jne	.L306
	movq	$5, -24(%rbp)
	jmp	.L305
.L306:
	movq	$3, -24(%rbp)
	jmp	.L305
.L298:
	movq	-32(%rbp), %rax
	movq	-40(%rbp), %rdx
	movq	%rdx, (%rax)
	movq	$7, -24(%rbp)
	jmp	.L305
.L301:
	movq	-32(%rbp), %rax
	movq	(%rax), %rax
	addq	$48, %rax
	movq	%rax, -32(%rbp)
	movq	$10, -24(%rbp)
	jmp	.L305
.L302:
	movq	-56(%rbp), %rdx
	movq	-40(%rbp), %rax
	movq	%rdx, 48(%rax)
	movq	-40(%rbp), %rax
	movq	%rax, -56(%rbp)
	movq	$7, -24(%rbp)
	jmp	.L305
.L297:
	movq	-32(%rbp), %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	je	.L308
	movq	$6, -24(%rbp)
	jmp	.L305
.L308:
	movq	$9, -24(%rbp)
	jmp	.L305
.L300:
	movq	-56(%rbp), %rax
	jmp	.L312
.L304:
	movl	$56, %edi
	call	malloc@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -40(%rbp)
	movq	-40(%rbp), %rax
	movq	-64(%rbp), %rcx
	movl	$32, %edx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	strncpy@PLT
	movq	-40(%rbp), %rax
	movl	-68(%rbp), %edx
	movl	%edx, 32(%rax)
	movl	$1, %edi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-40(%rbp), %rax
	movq	-8(%rbp), %rdx
	movq	%rdx, 40(%rax)
	movq	-40(%rbp), %rax
	movq	40(%rax), %rax
	movb	$0, (%rax)
	movq	$11, -24(%rbp)
	jmp	.L305
.L313:
	nop
.L305:
	jmp	.L311
.L312:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE11:
	.size	addStudent, .-addStudent
	.section	.rodata
	.align 8
.LC14:
	.string	"\nNo matches found with ID: %d\n"
	.text
	.globl	removeStudent
	.type	removeStudent, @function
removeStudent:
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
	movl	%esi, -28(%rbp)
	movq	$3, -8(%rbp)
.L333:
	cmpq	$11, -8(%rbp)
	ja	.L335
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L317(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L317(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L317:
	.long	.L324-.L317
	.long	.L323-.L317
	.long	.L322-.L317
	.long	.L321-.L317
	.long	.L335-.L317
	.long	.L335-.L317
	.long	.L320-.L317
	.long	.L335-.L317
	.long	.L319-.L317
	.long	.L335-.L317
	.long	.L318-.L317
	.long	.L316-.L317
	.text
.L319:
	movq	-16(%rbp), %rax
	movq	(%rax), %rax
	movq	40(%rax), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	-16(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	-16(%rbp), %rax
	movq	(%rax), %rax
	movq	48(%rax), %rdx
	movq	-16(%rbp), %rax
	movq	%rdx, (%rax)
	movq	$11, -8(%rbp)
	jmp	.L325
.L323:
	movq	-16(%rbp), %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	je	.L326
	movq	$8, -8(%rbp)
	jmp	.L325
.L326:
	movq	$6, -8(%rbp)
	jmp	.L325
.L321:
	leaq	-24(%rbp), %rax
	movq	%rax, -16(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L325
.L316:
	movq	-24(%rbp), %rax
	jmp	.L334
.L320:
	movl	-28(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$11, -8(%rbp)
	jmp	.L325
.L318:
	movq	-16(%rbp), %rax
	movq	(%rax), %rax
	addq	$48, %rax
	movq	%rax, -16(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L325
.L324:
	movq	-16(%rbp), %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	je	.L329
	movq	$2, -8(%rbp)
	jmp	.L325
.L329:
	movq	$1, -8(%rbp)
	jmp	.L325
.L322:
	movq	-16(%rbp), %rax
	movq	(%rax), %rax
	movl	32(%rax), %eax
	cmpl	%eax, -28(%rbp)
	je	.L331
	movq	$10, -8(%rbp)
	jmp	.L325
.L331:
	movq	$1, -8(%rbp)
	jmp	.L325
.L335:
	nop
.L325:
	jmp	.L333
.L334:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE12:
	.size	removeStudent, .-removeStudent
	.section	.rodata
	.align 8
.LC15:
	.string	"\nStudent with ID %d did not borrow a book with ISBN %s.\n"
	.align 8
.LC16:
	.string	"\nCouldn't found a student with ID %d.\n"
	.text
	.globl	returnBook
	.type	returnBook, @function
returnBook:
.LFB14:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$80, %rsp
	movq	%rdi, -56(%rbp)
	movq	%rsi, -64(%rbp)
	movl	%edx, -68(%rbp)
	movq	%rcx, -80(%rbp)
	movq	$29, -8(%rbp)
.L400:
	cmpq	$47, -8(%rbp)
	ja	.L401
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L339(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L339(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L339:
	.long	.L367-.L339
	.long	.L366-.L339
	.long	.L401-.L339
	.long	.L365-.L339
	.long	.L401-.L339
	.long	.L401-.L339
	.long	.L401-.L339
	.long	.L401-.L339
	.long	.L401-.L339
	.long	.L364-.L339
	.long	.L401-.L339
	.long	.L401-.L339
	.long	.L363-.L339
	.long	.L401-.L339
	.long	.L362-.L339
	.long	.L361-.L339
	.long	.L401-.L339
	.long	.L360-.L339
	.long	.L359-.L339
	.long	.L358-.L339
	.long	.L357-.L339
	.long	.L401-.L339
	.long	.L356-.L339
	.long	.L401-.L339
	.long	.L355-.L339
	.long	.L354-.L339
	.long	.L401-.L339
	.long	.L401-.L339
	.long	.L401-.L339
	.long	.L353-.L339
	.long	.L401-.L339
	.long	.L352-.L339
	.long	.L351-.L339
	.long	.L350-.L339
	.long	.L349-.L339
	.long	.L401-.L339
	.long	.L348-.L339
	.long	.L347-.L339
	.long	.L346-.L339
	.long	.L401-.L339
	.long	.L345-.L339
	.long	.L401-.L339
	.long	.L344-.L339
	.long	.L343-.L339
	.long	.L402-.L339
	.long	.L341-.L339
	.long	.L340-.L339
	.long	.L338-.L339
	.text
.L359:
	cmpq	$0, -16(%rbp)
	je	.L368
	movq	$17, -8(%rbp)
	jmp	.L370
.L368:
	movq	$37, -8(%rbp)
	jmp	.L370
.L354:
	movl	$0, -40(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L370
.L362:
	cmpq	$0, -24(%rbp)
	je	.L371
	movq	$46, -8(%rbp)
	jmp	.L370
.L371:
	movq	$12, -8(%rbp)
	jmp	.L370
.L361:
	movq	-24(%rbp), %rax
	movq	40(%rax), %rdx
	movl	-40(%rbp), %eax
	cltq
	addq	%rax, %rdx
	movq	-24(%rbp), %rax
	movq	40(%rax), %rcx
	movl	-44(%rbp), %eax
	cltq
	addq	%rcx, %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcpy@PLT
	movq	-64(%rbp), %rax
	movq	%rax, -16(%rbp)
	movq	$18, -8(%rbp)
	jmp	.L370
.L352:
	movq	-16(%rbp), %rax
	movq	120(%rax), %rax
	movq	%rax, -16(%rbp)
	movq	$18, -8(%rbp)
	jmp	.L370
.L363:
	cmpq	$0, -24(%rbp)
	je	.L373
	movq	$25, -8(%rbp)
	jmp	.L370
.L373:
	movq	$33, -8(%rbp)
	jmp	.L370
.L341:
	movq	-24(%rbp), %rax
	movq	48(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	$14, -8(%rbp)
	jmp	.L370
.L366:
	addl	$1, -40(%rbp)
	movq	$24, -8(%rbp)
	jmp	.L370
.L365:
	movq	-24(%rbp), %rax
	movq	40(%rax), %rdx
	movl	-40(%rbp), %eax
	cltq
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	cmpb	$44, %al
	je	.L375
	movq	$1, -8(%rbp)
	jmp	.L370
.L375:
	movq	$0, -8(%rbp)
	jmp	.L370
.L355:
	movq	-24(%rbp), %rax
	movq	40(%rax), %rdx
	movl	-40(%rbp), %eax
	cltq
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	je	.L377
	movq	$3, -8(%rbp)
	jmp	.L370
.L377:
	movq	$0, -8(%rbp)
	jmp	.L370
.L348:
	movl	-36(%rbp), %eax
	movslq	%eax, %rdx
	movq	-80(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	je	.L379
	movq	$38, -8(%rbp)
	jmp	.L370
.L379:
	movq	$42, -8(%rbp)
	jmp	.L370
.L364:
	movl	$-1, -40(%rbp)
	movq	$20, -8(%rbp)
	jmp	.L370
.L358:
	cmpl	$0, -32(%rbp)
	je	.L381
	movq	$31, -8(%rbp)
	jmp	.L370
.L381:
	movq	$37, -8(%rbp)
	jmp	.L370
.L351:
	movl	-36(%rbp), %eax
	movslq	%eax, %rdx
	movq	-80(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	jne	.L383
	movq	$15, -8(%rbp)
	jmp	.L370
.L383:
	movq	$24, -8(%rbp)
	jmp	.L370
.L360:
	movq	-16(%rbp), %rax
	movq	-80(%rbp), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -32(%rbp)
	movq	$19, -8(%rbp)
	jmp	.L370
.L345:
	movq	-80(%rbp), %rdx
	movl	-68(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$44, -8(%rbp)
	jmp	.L370
.L346:
	movl	-36(%rbp), %eax
	movslq	%eax, %rdx
	movq	-80(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %edx
	movq	-24(%rbp), %rax
	movq	40(%rax), %rcx
	movl	-40(%rbp), %eax
	cltq
	addq	%rcx, %rax
	movzbl	(%rax), %eax
	cmpb	%al, %dl
	jne	.L385
	movq	$34, -8(%rbp)
	jmp	.L370
.L385:
	movq	$42, -8(%rbp)
	jmp	.L370
.L349:
	addl	$1, -36(%rbp)
	addl	$1, -40(%rbp)
	movq	$36, -8(%rbp)
	jmp	.L370
.L356:
	movl	-40(%rbp), %eax
	movl	%eax, -28(%rbp)
	addl	$1, -40(%rbp)
	movl	-28(%rbp), %eax
	movl	%eax, -44(%rbp)
	movl	$0, -36(%rbp)
	movq	$36, -8(%rbp)
	jmp	.L370
.L338:
	movq	-16(%rbp), %rax
	movl	$0, 116(%rax)
	movq	$9, -8(%rbp)
	jmp	.L370
.L350:
	movl	-68(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC16(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$44, -8(%rbp)
	jmp	.L370
.L347:
	cmpq	$0, -16(%rbp)
	je	.L388
	movq	$47, -8(%rbp)
	jmp	.L370
.L388:
	movq	$9, -8(%rbp)
	jmp	.L370
.L344:
	movq	-24(%rbp), %rax
	movq	40(%rax), %rdx
	movl	-40(%rbp), %eax
	cltq
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	jne	.L390
	movq	$15, -8(%rbp)
	jmp	.L370
.L390:
	movq	$43, -8(%rbp)
	jmp	.L370
.L367:
	movq	-24(%rbp), %rax
	movq	40(%rax), %rdx
	movl	-40(%rbp), %eax
	cltq
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	je	.L392
	movq	$22, -8(%rbp)
	jmp	.L370
.L392:
	movq	$20, -8(%rbp)
	jmp	.L370
.L340:
	movq	-24(%rbp), %rax
	movl	32(%rax), %eax
	cmpl	%eax, -68(%rbp)
	je	.L394
	movq	$45, -8(%rbp)
	jmp	.L370
.L394:
	movq	$12, -8(%rbp)
	jmp	.L370
.L353:
	movq	-56(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	$14, -8(%rbp)
	jmp	.L370
.L343:
	movq	-24(%rbp), %rax
	movq	40(%rax), %rdx
	movl	-40(%rbp), %eax
	cltq
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	cmpb	$44, %al
	jne	.L396
	movq	$32, -8(%rbp)
	jmp	.L370
.L396:
	movq	$24, -8(%rbp)
	jmp	.L370
.L357:
	cmpl	$-1, -40(%rbp)
	je	.L398
	movq	$40, -8(%rbp)
	jmp	.L370
.L398:
	movq	$44, -8(%rbp)
	jmp	.L370
.L401:
	nop
.L370:
	jmp	.L400
.L402:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE14:
	.size	returnBook, .-returnBook
	.section	.rodata
.LC17:
	.string	"%d,%s"
.LC18:
	.string	"%s,%s,%s,%d,%d\n"
.LC19:
	.string	"w"
.LC20:
	.string	"%s\n"
	.text
	.globl	save
	.type	save, @function
save:
.LFB15:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movq	%rdi, -40(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	%rcx, -64(%rbp)
	movq	$20, -8(%rbp)
.L435:
	cmpq	$21, -8(%rbp)
	ja	.L436
	movq	-8(%rbp), %rax
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
	.long	.L422-.L406
	.long	.L421-.L406
	.long	.L436-.L406
	.long	.L420-.L406
	.long	.L419-.L406
	.long	.L418-.L406
	.long	.L417-.L406
	.long	.L436-.L406
	.long	.L416-.L406
	.long	.L415-.L406
	.long	.L436-.L406
	.long	.L414-.L406
	.long	.L413-.L406
	.long	.L412-.L406
	.long	.L436-.L406
	.long	.L411-.L406
	.long	.L410-.L406
	.long	.L436-.L406
	.long	.L409-.L406
	.long	.L437-.L406
	.long	.L407-.L406
	.long	.L405-.L406
	.text
.L409:
	movq	-16(%rbp), %rax
	movq	48(%rax), %rax
	movq	%rax, -16(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L423
.L419:
	cmpq	$0, -32(%rbp)
	je	.L424
	movq	$0, -8(%rbp)
	jmp	.L423
.L424:
	movq	$19, -8(%rbp)
	jmp	.L423
.L411:
	movq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$21, -8(%rbp)
	jmp	.L423
.L413:
	movq	-16(%rbp), %rcx
	movq	-16(%rbp), %rax
	movl	32(%rax), %edx
	movq	-32(%rbp), %rax
	leaq	.LC17(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	$9, -8(%rbp)
	jmp	.L423
.L416:
	movq	-24(%rbp), %rax
	movl	116(%rax), %esi
	movq	-24(%rbp), %rax
	movl	112(%rax), %edi
	movq	-24(%rbp), %rax
	leaq	78(%rax), %r8
	movq	-24(%rbp), %rax
	leaq	14(%rax), %rcx
	movq	-24(%rbp), %rdx
	movq	-32(%rbp), %rax
	subq	$8, %rsp
	pushq	%rsi
	movl	%edi, %r9d
	leaq	.LC18(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	addq	$16, %rsp
	movq	-24(%rbp), %rax
	movq	120(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	$13, -8(%rbp)
	jmp	.L423
.L421:
	movq	-32(%rbp), %rax
	movq	%rax, %rsi
	movl	$10, %edi
	call	fputc@PLT
	movq	$18, -8(%rbp)
	jmp	.L423
.L420:
	cmpq	$0, -32(%rbp)
	je	.L426
	movq	$16, -8(%rbp)
	jmp	.L423
.L426:
	movq	$21, -8(%rbp)
	jmp	.L423
.L410:
	movq	-56(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	$13, -8(%rbp)
	jmp	.L423
.L405:
	movq	-48(%rbp), %rax
	leaq	.LC19(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -32(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L423
.L414:
	movq	-16(%rbp), %rax
	movq	40(%rax), %rdx
	movq	-32(%rbp), %rax
	leaq	.LC20(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	$18, -8(%rbp)
	jmp	.L423
.L415:
	movq	-16(%rbp), %rax
	movq	40(%rax), %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	je	.L428
	movq	$11, -8(%rbp)
	jmp	.L423
.L428:
	movq	$1, -8(%rbp)
	jmp	.L423
.L412:
	cmpq	$0, -24(%rbp)
	je	.L430
	movq	$8, -8(%rbp)
	jmp	.L423
.L430:
	movq	$15, -8(%rbp)
	jmp	.L423
.L417:
	cmpq	$0, -16(%rbp)
	je	.L433
	movq	$12, -8(%rbp)
	jmp	.L423
.L433:
	movq	$5, -8(%rbp)
	jmp	.L423
.L418:
	movq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$19, -8(%rbp)
	jmp	.L423
.L422:
	movq	-64(%rbp), %rax
	movq	%rax, -16(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L423
.L407:
	movq	-40(%rbp), %rax
	leaq	.LC19(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -32(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L423
.L436:
	nop
.L423:
	jmp	.L435
.L437:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE15:
	.size	save, .-save
	.section	.rodata
.LC21:
	.string	"\nNo book with ISBN %s found.\n"
	.align 8
.LC22:
	.string	"\nThe book with ISBN %s is already borrowed.\n"
	.text
	.globl	borrowBook
	.type	borrowBook, @function
borrowBook:
.LFB16:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$104, %rsp
	.cfi_offset 3, -24
	movq	%rdi, -88(%rbp)
	movq	%rsi, -96(%rbp)
	movl	%edx, -100(%rbp)
	movq	%rcx, -112(%rbp)
	movq	$3, -48(%rbp)
.L474:
	cmpq	$22, -48(%rbp)
	ja	.L475
	movq	-48(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L441(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L441(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L441:
	.long	.L457-.L441
	.long	.L456-.L441
	.long	.L455-.L441
	.long	.L454-.L441
	.long	.L475-.L441
	.long	.L476-.L441
	.long	.L475-.L441
	.long	.L475-.L441
	.long	.L452-.L441
	.long	.L451-.L441
	.long	.L475-.L441
	.long	.L450-.L441
	.long	.L449-.L441
	.long	.L448-.L441
	.long	.L447-.L441
	.long	.L475-.L441
	.long	.L446-.L441
	.long	.L445-.L441
	.long	.L444-.L441
	.long	.L443-.L441
	.long	.L442-.L441
	.long	.L475-.L441
	.long	.L440-.L441
	.text
.L444:
	movq	-112(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC21(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$5, -48(%rbp)
	jmp	.L458
.L447:
	movq	-64(%rbp), %rax
	movl	116(%rax), %eax
	testl	%eax, %eax
	je	.L459
	movq	$17, -48(%rbp)
	jmp	.L458
.L459:
	movq	$20, -48(%rbp)
	jmp	.L458
.L449:
	cmpq	$0, -64(%rbp)
	je	.L461
	movq	$1, -48(%rbp)
	jmp	.L458
.L461:
	movq	$19, -48(%rbp)
	jmp	.L458
.L452:
	movq	-56(%rbp), %rax
	movq	40(%rax), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -40(%rbp)
	movq	-112(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -32(%rbp)
	movq	-40(%rbp), %rdx
	movq	-32(%rbp), %rax
	addq	%rdx, %rax
	leaq	2(%rax), %rdx
	movq	-56(%rbp), %rax
	movq	40(%rax), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -24(%rbp)
	movq	-56(%rbp), %rax
	movq	-24(%rbp), %rdx
	movq	%rdx, 40(%rax)
	movq	-56(%rbp), %rax
	movq	40(%rax), %rbx
	movq	%rbx, %rdi
	call	strlen@PLT
	addq	%rbx, %rax
	movw	$44, (%rax)
	movq	-56(%rbp), %rax
	movq	40(%rax), %rax
	movq	-112(%rbp), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcat@PLT
	movq	-64(%rbp), %rax
	movl	$1, 116(%rax)
	movq	$5, -48(%rbp)
	jmp	.L458
.L456:
	movq	-64(%rbp), %rax
	movq	-112(%rbp), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -68(%rbp)
	movq	$0, -48(%rbp)
	jmp	.L458
.L454:
	movq	-96(%rbp), %rax
	movq	%rax, -64(%rbp)
	movq	$12, -48(%rbp)
	jmp	.L458
.L446:
	movq	-56(%rbp), %rax
	movl	32(%rax), %eax
	cmpl	%eax, -100(%rbp)
	je	.L463
	movq	$22, -48(%rbp)
	jmp	.L458
.L463:
	movq	$2, -48(%rbp)
	jmp	.L458
.L450:
	movl	-100(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC16(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$5, -48(%rbp)
	jmp	.L458
.L451:
	cmpq	$0, -56(%rbp)
	je	.L465
	movq	$16, -48(%rbp)
	jmp	.L458
.L465:
	movq	$2, -48(%rbp)
	jmp	.L458
.L448:
	movq	-64(%rbp), %rax
	movq	120(%rax), %rax
	movq	%rax, -64(%rbp)
	movq	$12, -48(%rbp)
	jmp	.L458
.L443:
	cmpq	$0, -64(%rbp)
	jne	.L467
	movq	$18, -48(%rbp)
	jmp	.L458
.L467:
	movq	$14, -48(%rbp)
	jmp	.L458
.L445:
	movq	-112(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC22(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$5, -48(%rbp)
	jmp	.L458
.L440:
	movq	-56(%rbp), %rax
	movq	48(%rax), %rax
	movq	%rax, -56(%rbp)
	movq	$9, -48(%rbp)
	jmp	.L458
.L457:
	cmpl	$0, -68(%rbp)
	je	.L470
	movq	$13, -48(%rbp)
	jmp	.L458
.L470:
	movq	$19, -48(%rbp)
	jmp	.L458
.L455:
	cmpq	$0, -56(%rbp)
	je	.L472
	movq	$8, -48(%rbp)
	jmp	.L458
.L472:
	movq	$11, -48(%rbp)
	jmp	.L458
.L442:
	movq	-88(%rbp), %rax
	movq	%rax, -56(%rbp)
	movq	$9, -48(%rbp)
	jmp	.L458
.L475:
	nop
.L458:
	jmp	.L474
.L476:
	nop
	movq	-8(%rbp), %rbx
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE16:
	.size	borrowBook, .-borrowBook
	.section	.rodata
.LC23:
	.string	"%d,%31[^,\n]"
.LC24:
	.string	"r"
	.align 8
.LC25:
	.string	"%13[^,],%63[^,],%31[^,],%d,%d "
.LC26:
	.string	"%*[^\n]"
.LC27:
	.string	"%[^\n]"
	.text
	.globl	load
	.type	load, @function
load:
.LFB17:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$344, %rsp
	.cfi_offset 3, -24
	movq	%rdi, -328(%rbp)
	movq	%rsi, -336(%rbp)
	movq	%rdx, -344(%rbp)
	movq	%rcx, -352(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -24(%rbp)
	xorl	%eax, %eax
	movq	$16, -272(%rbp)
.L506:
	cmpq	$20, -272(%rbp)
	ja	.L509
	movq	-272(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L480(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L480(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L480:
	.long	.L509-.L480
	.long	.L495-.L480
	.long	.L510-.L480
	.long	.L493-.L480
	.long	.L492-.L480
	.long	.L491-.L480
	.long	.L490-.L480
	.long	.L489-.L480
	.long	.L488-.L480
	.long	.L487-.L480
	.long	.L486-.L480
	.long	.L485-.L480
	.long	.L509-.L480
	.long	.L484-.L480
	.long	.L509-.L480
	.long	.L509-.L480
	.long	.L483-.L480
	.long	.L509-.L480
	.long	.L482-.L480
	.long	.L481-.L480
	.long	.L479-.L480
	.text
.L482:
	movq	-296(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$2, -272(%rbp)
	jmp	.L496
.L492:
	movq	-352(%rbp), %rax
	movq	%rax, -280(%rbp)
	movq	$8, -272(%rbp)
	jmp	.L496
.L488:
	leaq	-224(%rbp), %rdx
	leaq	-224(%rbp), %rax
	leaq	32(%rax), %rsi
	movq	-296(%rbp), %rax
	movq	%rdx, %rcx
	movq	%rsi, %rdx
	leaq	.LC23(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_fscanf@PLT
	movl	%eax, -308(%rbp)
	movq	$10, -272(%rbp)
	jmp	.L496
.L495:
	movq	-344(%rbp), %rax
	movq	%rax, -288(%rbp)
	movq	$6, -272(%rbp)
	jmp	.L496
.L493:
	movq	-336(%rbp), %rax
	leaq	.LC24(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -296(%rbp)
	movq	$20, -272(%rbp)
	jmp	.L496
.L483:
	movq	$13, -272(%rbp)
	jmp	.L496
.L485:
	movq	$0, -40(%rbp)
	movl	$128, %edi
	call	malloc@PLT
	movq	%rax, -264(%rbp)
	movq	-288(%rbp), %rax
	movq	-264(%rbp), %rdx
	movq	%rdx, (%rax)
	movq	-288(%rbp), %rax
	movq	(%rax), %rax
	movq	-160(%rbp), %rcx
	movq	-152(%rbp), %rbx
	movq	%rcx, (%rax)
	movq	%rbx, 8(%rax)
	movq	-144(%rbp), %rcx
	movq	-136(%rbp), %rbx
	movq	%rcx, 16(%rax)
	movq	%rbx, 24(%rax)
	movq	-128(%rbp), %rcx
	movq	-120(%rbp), %rbx
	movq	%rcx, 32(%rax)
	movq	%rbx, 40(%rax)
	movq	-112(%rbp), %rcx
	movq	-104(%rbp), %rbx
	movq	%rcx, 48(%rax)
	movq	%rbx, 56(%rax)
	movq	-96(%rbp), %rcx
	movq	-88(%rbp), %rbx
	movq	%rcx, 64(%rax)
	movq	%rbx, 72(%rax)
	movq	-80(%rbp), %rcx
	movq	-72(%rbp), %rbx
	movq	%rcx, 80(%rax)
	movq	%rbx, 88(%rax)
	movq	-64(%rbp), %rcx
	movq	-56(%rbp), %rbx
	movq	%rcx, 96(%rax)
	movq	%rbx, 104(%rax)
	movq	-48(%rbp), %rcx
	movq	-40(%rbp), %rbx
	movq	%rcx, 112(%rax)
	movq	%rbx, 120(%rax)
	movq	-288(%rbp), %rax
	movq	(%rax), %rax
	addq	$120, %rax
	movq	%rax, -288(%rbp)
	movq	$6, -272(%rbp)
	jmp	.L496
.L487:
	cmpq	$0, -296(%rbp)
	je	.L497
	movq	$1, -272(%rbp)
	jmp	.L496
.L497:
	movq	$3, -272(%rbp)
	jmp	.L496
.L484:
	movq	-328(%rbp), %rax
	leaq	.LC24(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -296(%rbp)
	movq	-344(%rbp), %rax
	movq	$0, (%rax)
	movq	-352(%rbp), %rax
	movq	$0, (%rax)
	movq	$9, -272(%rbp)
	jmp	.L496
.L481:
	movq	-296(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$3, -272(%rbp)
	jmp	.L496
.L490:
	leaq	-160(%rbp), %rax
	leaq	112(%rax), %r8
	leaq	-160(%rbp), %rax
	leaq	78(%rax), %rdi
	leaq	-160(%rbp), %rax
	leaq	14(%rax), %rcx
	leaq	-160(%rbp), %rdx
	movq	-296(%rbp), %rax
	subq	$8, %rsp
	leaq	-160(%rbp), %rsi
	addq	$116, %rsi
	pushq	%rsi
	movq	%r8, %r9
	movq	%rdi, %r8
	leaq	.LC25(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_fscanf@PLT
	addq	$16, %rsp
	movl	%eax, -312(%rbp)
	movq	$7, -272(%rbp)
	jmp	.L496
.L491:
	movq	-296(%rbp), %rax
	movq	%rax, %rdi
	call	ftell@PLT
	movq	%rax, -256(%rbp)
	movq	-256(%rbp), %rax
	movl	%eax, -304(%rbp)
	movq	-296(%rbp), %rax
	leaq	.LC26(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_fscanf@PLT
	movq	-296(%rbp), %rax
	movq	%rax, %rdi
	call	ftell@PLT
	movq	%rax, -248(%rbp)
	movq	-248(%rbp), %rax
	movl	-304(%rbp), %edx
	subl	%edx, %eax
	movl	%eax, -300(%rbp)
	movl	-300(%rbp), %eax
	addl	$1, %eax
	cltq
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -240(%rbp)
	movq	-240(%rbp), %rax
	movq	%rax, -184(%rbp)
	movl	-304(%rbp), %eax
	movslq	%eax, %rcx
	movq	-296(%rbp), %rax
	movl	$0, %edx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	fseek@PLT
	movq	-184(%rbp), %rdx
	movq	-296(%rbp), %rax
	leaq	.LC27(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_fscanf@PLT
	movq	$0, -176(%rbp)
	movl	$56, %edi
	call	malloc@PLT
	movq	%rax, -232(%rbp)
	movq	-280(%rbp), %rax
	movq	-232(%rbp), %rdx
	movq	%rdx, (%rax)
	movq	-280(%rbp), %rax
	movq	(%rax), %rax
	movq	-224(%rbp), %rcx
	movq	-216(%rbp), %rbx
	movq	%rcx, (%rax)
	movq	%rbx, 8(%rax)
	movq	-208(%rbp), %rcx
	movq	-200(%rbp), %rbx
	movq	%rcx, 16(%rax)
	movq	%rbx, 24(%rax)
	movq	-192(%rbp), %rcx
	movq	-184(%rbp), %rbx
	movq	%rcx, 32(%rax)
	movq	%rbx, 40(%rax)
	movq	-176(%rbp), %rdx
	movq	%rdx, 48(%rax)
	movq	-280(%rbp), %rax
	movq	(%rax), %rax
	addq	$48, %rax
	movq	%rax, -280(%rbp)
	movq	$8, -272(%rbp)
	jmp	.L496
.L486:
	cmpl	$-1, -308(%rbp)
	je	.L499
	movq	$5, -272(%rbp)
	jmp	.L496
.L499:
	movq	$18, -272(%rbp)
	jmp	.L496
.L489:
	cmpl	$-1, -312(%rbp)
	je	.L501
	movq	$11, -272(%rbp)
	jmp	.L496
.L501:
	movq	$19, -272(%rbp)
	jmp	.L496
.L479:
	cmpq	$0, -296(%rbp)
	je	.L504
	movq	$4, -272(%rbp)
	jmp	.L496
.L504:
	movq	$2, -272(%rbp)
	jmp	.L496
.L509:
	nop
.L496:
	jmp	.L506
.L510:
	nop
	movq	-24(%rbp), %rax
	subq	%fs:40, %rax
	je	.L508
	call	__stack_chk_fail@PLT
.L508:
	movq	-8(%rbp), %rbx
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE17:
	.size	load, .-load
	.section	.rodata
	.align 8
.LC28:
	.ascii	"\nWelcome to the library, please select an option:\n 0 "
	.string	"- Exit\n 1 - Add a book\n 2 - Delete a book\n 3 - Update a book\n 4 - Filter and sort books\n 5 - Reverse the book list\n 6 - Search books\n 7 - Borrow a book\n 8 - Return a book\n 9 - Display books borrowed by a student\n10 - Add a student\n11 - Remove a student\n\n> "
.LC29:
	.string	"\nEnter student ID: "
.LC30:
	.string	"\nEnter student name: "
.LC31:
	.string	" %31[^\n]"
.LC32:
	.string	"students.txt"
.LC33:
	.string	"library.txt"
.LC34:
	.string	"\nEnter ISBN: "
.LC35:
	.string	" %13[^\n]"
.LC36:
	.string	"\nEnter feature name: "
.LC37:
	.string	" %63[^\n]"
.LC38:
	.string	"\nEnter new value: "
.LC39:
	.string	"\nInvalid input!"
	.align 8
.LC40:
	.string	"\nEnter search choice (0 for ISBN, 1 for author, 2 for title): "
.LC41:
	.string	"\nEnter search criteria: "
	.align 8
.LC42:
	.string	"\nPlease select the management method (0 for FIFO, 1 for LIFO): "
.LC43:
	.string	"\nEnter the title: "
	.align 8
.LC44:
	.string	"\nEnter the name of the author: "
	.align 8
.LC45:
	.string	"\nEnter the year of publication: "
	.align 8
.LC46:
	.string	"\nEnter filter choice (0 for author, 1 for publication year): "
.LC47:
	.string	"\nEnter filter: "
	.align 8
.LC48:
	.string	"\nEnter sort choice (0 for title, 1 for author, 2 for publication year): "
	.text
	.globl	main
	.type	main, @function
main:
.LFB18:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$288, %rsp
	movl	%edi, -260(%rbp)
	movq	%rsi, -272(%rbp)
	movq	%rdx, -280(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_qARH_envp(%rip)
	nop
.L512:
	movq	$0, _TIG_IZ_qARH_argv(%rip)
	nop
.L513:
	movl	$0, _TIG_IZ_qARH_argc(%rip)
	nop
	nop
.L514:
.L515:
#APP
# 417 "mehme2_cse_assignments_main.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-qARH--0
# 0 "" 2
#NO_APP
	movl	-260(%rbp), %eax
	movl	%eax, _TIG_IZ_qARH_argc(%rip)
	movq	-272(%rbp), %rax
	movq	%rax, _TIG_IZ_qARH_argv(%rip)
	movq	-280(%rbp), %rax
	movq	%rax, _TIG_IZ_qARH_envp(%rip)
	nop
	movq	$18, -224(%rbp)
.L564:
	cmpq	$51, -224(%rbp)
	ja	.L567
	movq	-224(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L518(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L518(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L518:
	.long	.L540-.L518
	.long	.L539-.L518
	.long	.L567-.L518
	.long	.L538-.L518
	.long	.L537-.L518
	.long	.L567-.L518
	.long	.L536-.L518
	.long	.L567-.L518
	.long	.L535-.L518
	.long	.L567-.L518
	.long	.L567-.L518
	.long	.L567-.L518
	.long	.L567-.L518
	.long	.L567-.L518
	.long	.L534-.L518
	.long	.L567-.L518
	.long	.L567-.L518
	.long	.L533-.L518
	.long	.L532-.L518
	.long	.L531-.L518
	.long	.L530-.L518
	.long	.L567-.L518
	.long	.L567-.L518
	.long	.L567-.L518
	.long	.L567-.L518
	.long	.L529-.L518
	.long	.L528-.L518
	.long	.L567-.L518
	.long	.L527-.L518
	.long	.L526-.L518
	.long	.L567-.L518
	.long	.L525-.L518
	.long	.L524-.L518
	.long	.L567-.L518
	.long	.L567-.L518
	.long	.L567-.L518
	.long	.L523-.L518
	.long	.L567-.L518
	.long	.L522-.L518
	.long	.L567-.L518
	.long	.L567-.L518
	.long	.L567-.L518
	.long	.L521-.L518
	.long	.L567-.L518
	.long	.L567-.L518
	.long	.L567-.L518
	.long	.L567-.L518
	.long	.L520-.L518
	.long	.L519-.L518
	.long	.L567-.L518
	.long	.L567-.L518
	.long	.L517-.L518
	.text
.L532:
	movq	$17, -224(%rbp)
	jmp	.L541
.L529:
	leaq	.LC28(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-244(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$29, -224(%rbp)
	jmp	.L541
.L537:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L565
	jmp	.L566
.L534:
	leaq	.LC29(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-216(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-216(%rbp), %edx
	movq	-240(%rbp), %rcx
	movq	-232(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	displayBooks
	movq	$1, -224(%rbp)
	jmp	.L541
.L525:
	movl	-248(%rbp), %eax
	cmpl	$1, %eax
	jne	.L543
	movq	$38, -224(%rbp)
	jmp	.L541
.L543:
	movq	$51, -224(%rbp)
	jmp	.L541
.L535:
	leaq	.LC30(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-208(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC31(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC29(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-216(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-248(%rbp), %ecx
	movl	-216(%rbp), %edx
	movq	-232(%rbp), %rax
	leaq	-208(%rbp), %rsi
	movq	%rax, %rdi
	call	addStudent
	movq	%rax, -232(%rbp)
	movq	$1, -224(%rbp)
	jmp	.L541
.L539:
	movq	-232(%rbp), %rdx
	movq	-240(%rbp), %rax
	movq	%rdx, %rcx
	movq	%rax, %rdx
	leaq	.LC32(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC33(%rip), %rax
	movq	%rax, %rdi
	call	save
	movq	$26, -224(%rbp)
	jmp	.L541
.L538:
	leaq	.LC29(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-216(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC34(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-208(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC35(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-216(%rbp), %edx
	movq	-240(%rbp), %rsi
	movq	-232(%rbp), %rax
	leaq	-208(%rbp), %rcx
	movq	%rax, %rdi
	call	returnBook
	movq	$1, -224(%rbp)
	jmp	.L541
.L523:
	leaq	.LC34(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-208(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC35(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC36(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-208(%rbp), %rax
	addq	$64, %rax
	movq	%rax, %rsi
	leaq	.LC37(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC38(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-208(%rbp), %rax
	subq	$-128, %rax
	movq	%rax, %rsi
	leaq	.LC37(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	-240(%rbp), %rax
	leaq	-208(%rbp), %rdx
	leaq	128(%rdx), %rcx
	leaq	-208(%rbp), %rdx
	addq	$64, %rdx
	leaq	-208(%rbp), %rsi
	movq	%rax, %rdi
	call	updateBook
	movq	$1, -224(%rbp)
	jmp	.L541
.L528:
	movl	-244(%rbp), %eax
	testl	%eax, %eax
	je	.L545
	movq	$25, -224(%rbp)
	jmp	.L541
.L545:
	movq	$4, -224(%rbp)
	jmp	.L541
.L517:
	leaq	.LC39(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$17, -224(%rbp)
	jmp	.L541
.L531:
	leaq	.LC34(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-208(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC35(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	-240(%rbp), %rax
	leaq	-208(%rbp), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	deleteBook
	movq	%rax, -240(%rbp)
	movq	$1, -224(%rbp)
	jmp	.L541
.L524:
	leaq	.LC40(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-216(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC41(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-208(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC37(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-216(%rbp), %ecx
	movq	-240(%rbp), %rax
	leaq	-208(%rbp), %rdx
	movl	%ecx, %esi
	movq	%rax, %rdi
	call	searchBooks
	movq	$1, -224(%rbp)
	jmp	.L541
.L533:
	leaq	.LC42(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-248(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$20, -224(%rbp)
	jmp	.L541
.L536:
	leaq	.LC29(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-216(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC34(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-208(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC35(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-216(%rbp), %edx
	movq	-240(%rbp), %rsi
	movq	-232(%rbp), %rax
	leaq	-208(%rbp), %rcx
	movq	%rax, %rdi
	call	borrowBook
	movq	$1, -224(%rbp)
	jmp	.L541
.L522:
	leaq	-232(%rbp), %rdx
	leaq	-240(%rbp), %rax
	movq	%rdx, %rcx
	movq	%rax, %rdx
	leaq	.LC32(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC33(%rip), %rax
	movq	%rax, %rdi
	call	load
	movq	$25, -224(%rbp)
	jmp	.L541
.L519:
	leaq	.LC39(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$1, -224(%rbp)
	jmp	.L541
.L527:
	leaq	.LC29(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-216(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-216(%rbp), %edx
	movq	-232(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	removeStudent
	movq	%rax, -232(%rbp)
	movq	$1, -224(%rbp)
	jmp	.L541
.L520:
	movq	-240(%rbp), %rax
	movq	%rax, %rdi
	call	reverseBookList
	movq	%rax, -240(%rbp)
	movq	$1, -224(%rbp)
	jmp	.L541
.L521:
	leaq	.LC34(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-208(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC35(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC43(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-208(%rbp), %rax
	addq	$64, %rax
	movq	%rax, %rsi
	leaq	.LC37(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC44(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-208(%rbp), %rax
	subq	$-128, %rax
	movq	%rax, %rsi
	leaq	.LC31(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC45(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-216(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-248(%rbp), %r8d
	movl	-216(%rbp), %edi
	movq	-240(%rbp), %rax
	leaq	-208(%rbp), %rdx
	leaq	128(%rdx), %rcx
	leaq	-208(%rbp), %rdx
	addq	$64, %rdx
	leaq	-208(%rbp), %rsi
	movl	%r8d, %r9d
	movl	%edi, %r8d
	movq	%rax, %rdi
	call	addBook
	movq	%rax, -240(%rbp)
	movq	$1, -224(%rbp)
	jmp	.L541
.L540:
	leaq	.LC46(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-216(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC47(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-208(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC31(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC48(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-216(%rbp), %rax
	addq	$4, %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-212(%rbp), %ecx
	movl	-216(%rbp), %esi
	movq	-240(%rbp), %rax
	leaq	-208(%rbp), %rdx
	movq	%rax, %rdi
	call	filterAndSortBooks
	movq	%rax, -240(%rbp)
	movq	$1, -224(%rbp)
	jmp	.L541
.L526:
	movl	-244(%rbp), %eax
	cmpl	$11, %eax
	ja	.L547
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L549(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L549(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L549:
	.long	.L560-.L549
	.long	.L559-.L549
	.long	.L558-.L549
	.long	.L557-.L549
	.long	.L556-.L549
	.long	.L555-.L549
	.long	.L554-.L549
	.long	.L553-.L549
	.long	.L552-.L549
	.long	.L551-.L549
	.long	.L550-.L549
	.long	.L548-.L549
	.text
.L548:
	movq	$28, -224(%rbp)
	jmp	.L561
.L550:
	movq	$8, -224(%rbp)
	jmp	.L561
.L551:
	movq	$14, -224(%rbp)
	jmp	.L561
.L552:
	movq	$3, -224(%rbp)
	jmp	.L561
.L553:
	movq	$6, -224(%rbp)
	jmp	.L561
.L554:
	movq	$32, -224(%rbp)
	jmp	.L561
.L555:
	movq	$47, -224(%rbp)
	jmp	.L561
.L556:
	movq	$0, -224(%rbp)
	jmp	.L561
.L557:
	movq	$36, -224(%rbp)
	jmp	.L561
.L558:
	movq	$19, -224(%rbp)
	jmp	.L561
.L559:
	movq	$42, -224(%rbp)
	jmp	.L561
.L560:
	movq	$1, -224(%rbp)
	jmp	.L561
.L547:
	movq	$48, -224(%rbp)
	nop
.L561:
	jmp	.L541
.L530:
	movl	-248(%rbp), %eax
	testl	%eax, %eax
	jne	.L562
	movq	$38, -224(%rbp)
	jmp	.L541
.L562:
	movq	$31, -224(%rbp)
	jmp	.L541
.L567:
	nop
.L541:
	jmp	.L564
.L566:
	call	__stack_chk_fail@PLT
.L565:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE18:
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
