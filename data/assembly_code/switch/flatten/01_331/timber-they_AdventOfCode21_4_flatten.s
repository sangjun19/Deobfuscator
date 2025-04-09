	.file	"timber-they_AdventOfCode21_4_flatten.c"
	.text
	.globl	_TIG_IZ_XU3m_argv
	.bss
	.align 8
	.type	_TIG_IZ_XU3m_argv, @object
	.size	_TIG_IZ_XU3m_argv, 8
_TIG_IZ_XU3m_argv:
	.zero	8
	.globl	_TIG_IZ_XU3m_envp
	.align 8
	.type	_TIG_IZ_XU3m_envp, @object
	.size	_TIG_IZ_XU3m_envp, 8
_TIG_IZ_XU3m_envp:
	.zero	8
	.globl	_TIG_IZ_XU3m_argc
	.align 4
	.type	_TIG_IZ_XU3m_argc, @object
	.size	_TIG_IZ_XU3m_argc, 4
_TIG_IZ_XU3m_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"Expected line break!\n"
	.text
	.globl	part1
	.type	part1, @function
part1:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	leaq	-20480(%rsp), %r11
.LPSRL0:
	subq	$4096, %rsp
	orq	$0, (%rsp)
	cmpq	%r11, %rsp
	jne	.LPSRL0
	subq	$2416, %rsp
	movq	%rdi, -22888(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$20, -22840(%rbp)
.L29:
	cmpq	$20, -22840(%rbp)
	ja	.L32
	movq	-22840(%rbp), %rax
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
	.long	.L18-.L4
	.long	.L17-.L4
	.long	.L16-.L4
	.long	.L15-.L4
	.long	.L32-.L4
	.long	.L14-.L4
	.long	.L32-.L4
	.long	.L13-.L4
	.long	.L12-.L4
	.long	.L32-.L4
	.long	.L11-.L4
	.long	.L10-.L4
	.long	.L32-.L4
	.long	.L9-.L4
	.long	.L8-.L4
	.long	.L7-.L4
	.long	.L32-.L4
	.long	.L32-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L6:
	cmpl	$10, -22876(%rbp)
	je	.L19
	movq	$11, -22840(%rbp)
	jmp	.L21
.L19:
	movq	$19, -22840(%rbp)
	jmp	.L21
.L8:
	movl	$-1, %eax
	jmp	.L30
.L7:
	movl	$0, -10016(%rbp)
	movl	$1, -22880(%rbp)
	movq	$8, -22840(%rbp)
	jmp	.L21
.L12:
	cmpl	$2499, -22880(%rbp)
	jbe	.L23
	movq	$7, -22840(%rbp)
	jmp	.L21
.L23:
	movq	$13, -22840(%rbp)
	jmp	.L21
.L17:
	cmpl	$0, -22868(%rbp)
	js	.L25
	movq	$3, -22840(%rbp)
	jmp	.L21
.L25:
	movq	$2, -22840(%rbp)
	jmp	.L21
.L15:
	movl	-22868(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-22848(%rbp), %rax
	addq	%rdx, %rax
	subq	$8, %rsp
	pushq	16(%rax)
	pushq	8(%rax)
	pushq	(%rax)
	call	getScore
	addq	$32, %rsp
	movl	%eax, -22864(%rbp)
	movq	$0, -22840(%rbp)
	jmp	.L21
.L10:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$21, %edx
	movl	$1, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	$19, -22840(%rbp)
	jmp	.L21
.L9:
	movl	-22880(%rbp), %eax
	movl	$0, -10016(%rbp,%rax,4)
	addl	$1, -22880(%rbp)
	movq	$8, -22840(%rbp)
	jmp	.L21
.L5:
	leaq	-10016(%rbp), %rcx
	leaq	-20016(%rbp), %rdx
	leaq	-22416(%rbp), %rsi
	movq	-22888(%rbp), %rax
	movq	%rax, %rdi
	call	readBoards
	movq	%rax, -22832(%rbp)
	movq	-22832(%rbp), %rax
	movq	%rax, -22848(%rbp)
	movl	$0, -22872(%rbp)
	movq	$10, -22840(%rbp)
	jmp	.L21
.L14:
	movl	-22872(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-22856(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %ecx
	movl	-22872(%rbp), %edx
	movq	-22848(%rbp), %rax
	movl	%ecx, %esi
	movq	%rax, %rdi
	call	iterate
	movl	%eax, -22860(%rbp)
	movl	-22860(%rbp), %eax
	movl	%eax, -22868(%rbp)
	movq	$1, -22840(%rbp)
	jmp	.L21
.L11:
	cmpl	$99, -22872(%rbp)
	jg	.L27
	movq	$5, -22840(%rbp)
	jmp	.L21
.L27:
	movq	$14, -22840(%rbp)
	jmp	.L21
.L18:
	movl	-22864(%rbp), %eax
	jmp	.L30
.L13:
	leaq	-22816(%rbp), %rdx
	movq	-22888(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	readNumbers
	movq	%rax, -22824(%rbp)
	movq	-22824(%rbp), %rax
	movq	%rax, -22856(%rbp)
	movq	-22888(%rbp), %rax
	movq	%rax, %rdi
	call	fgetc@PLT
	movl	%eax, -22876(%rbp)
	movq	$18, -22840(%rbp)
	jmp	.L21
.L16:
	addl	$1, -22872(%rbp)
	movq	$10, -22840(%rbp)
	jmp	.L21
.L3:
	movq	$15, -22840(%rbp)
	jmp	.L21
.L32:
	nop
.L21:
	jmp	.L29
.L30:
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L31
	call	__stack_chk_fail@PLT
.L31:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	part1, .-part1
	.globl	hasWon
	.type	hasWon, @function
hasWon:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	$17, -8(%rbp)
.L73:
	cmpq	$28, -8(%rbp)
	ja	.L74
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L36(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L36(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L36:
	.long	.L54-.L36
	.long	.L53-.L36
	.long	.L52-.L36
	.long	.L51-.L36
	.long	.L50-.L36
	.long	.L49-.L36
	.long	.L48-.L36
	.long	.L47-.L36
	.long	.L74-.L36
	.long	.L46-.L36
	.long	.L74-.L36
	.long	.L45-.L36
	.long	.L74-.L36
	.long	.L44-.L36
	.long	.L74-.L36
	.long	.L43-.L36
	.long	.L42-.L36
	.long	.L41-.L36
	.long	.L40-.L36
	.long	.L74-.L36
	.long	.L39-.L36
	.long	.L74-.L36
	.long	.L38-.L36
	.long	.L74-.L36
	.long	.L37-.L36
	.long	.L74-.L36
	.long	.L74-.L36
	.long	.L74-.L36
	.long	.L35-.L36
	.text
.L40:
	addl	$1, -24(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L55
.L50:
	cmpl	$5, -20(%rbp)
	jne	.L56
	movq	$0, -8(%rbp)
	jmp	.L55
.L56:
	movq	$18, -8(%rbp)
	jmp	.L55
.L43:
	movl	$0, -12(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L55
.L53:
	cmpl	$4, -12(%rbp)
	jg	.L58
	movq	$5, -8(%rbp)
	jmp	.L55
.L58:
	movq	$2, -8(%rbp)
	jmp	.L55
.L51:
	movl	$0, -16(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L55
.L42:
	addl	$1, -12(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L55
.L37:
	cmpl	$4, -20(%rbp)
	jg	.L60
	movq	$28, -8(%rbp)
	jmp	.L55
.L60:
	movq	$4, -8(%rbp)
	jmp	.L55
.L45:
	addl	$1, -20(%rbp)
	movq	$24, -8(%rbp)
	jmp	.L55
.L46:
	movl	$2, %eax
	jmp	.L62
.L44:
	movl	$0, %eax
	jmp	.L62
.L41:
	movl	$0, -24(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L55
.L48:
	cmpl	$4, -16(%rbp)
	jg	.L63
	movq	$15, -8(%rbp)
	jmp	.L55
.L63:
	movq	$13, -8(%rbp)
	jmp	.L55
.L38:
	movl	$0, -20(%rbp)
	movq	$24, -8(%rbp)
	jmp	.L55
.L35:
	movq	24(%rbp), %rcx
	movl	-24(%rbp), %edx
	movl	%edx, %eax
	sall	$2, %eax
	addl	%eax, %edx
	movl	-20(%rbp), %eax
	addl	%edx, %eax
	cltq
	salq	$2, %rax
	addq	%rcx, %rax
	movl	(%rax), %eax
	testl	%eax, %eax
	jne	.L65
	movq	$4, -8(%rbp)
	jmp	.L55
.L65:
	movq	$11, -8(%rbp)
	jmp	.L55
.L49:
	movq	24(%rbp), %rcx
	movl	-12(%rbp), %edx
	movl	%edx, %eax
	sall	$2, %eax
	addl	%eax, %edx
	movl	-16(%rbp), %eax
	addl	%edx, %eax
	cltq
	salq	$2, %rax
	addq	%rcx, %rax
	movl	(%rax), %eax
	testl	%eax, %eax
	jne	.L67
	movq	$2, -8(%rbp)
	jmp	.L55
.L67:
	movq	$16, -8(%rbp)
	jmp	.L55
.L54:
	movl	$1, %eax
	jmp	.L62
.L47:
	cmpl	$4, -24(%rbp)
	jg	.L69
	movq	$22, -8(%rbp)
	jmp	.L55
.L69:
	movq	$3, -8(%rbp)
	jmp	.L55
.L52:
	cmpl	$5, -12(%rbp)
	jne	.L71
	movq	$9, -8(%rbp)
	jmp	.L55
.L71:
	movq	$20, -8(%rbp)
	jmp	.L55
.L39:
	addl	$1, -16(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L55
.L74:
	nop
.L55:
	jmp	.L73
.L62:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	hasWon, .-hasWon
	.section	.rodata
	.align 8
.LC1:
	.string	"Board %d didn't win at all! What if the squid chose this one?\n"
	.text
	.globl	part2
	.type	part2, @function
part2:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	leaq	-20480(%rsp), %r11
.LPSRL1:
	subq	$4096, %rsp
	orq	$0, (%rsp)
	cmpq	%r11, %rsp
	jne	.LPSRL1
	subq	$2416, %rsp
	movq	%rdi, -22888(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$28, -22840(%rbp)
.L111:
	cmpq	$28, -22840(%rbp)
	ja	.L114
	movq	-22840(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L78(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L78(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L78:
	.long	.L96-.L78
	.long	.L114-.L78
	.long	.L95-.L78
	.long	.L114-.L78
	.long	.L94-.L78
	.long	.L93-.L78
	.long	.L114-.L78
	.long	.L92-.L78
	.long	.L114-.L78
	.long	.L114-.L78
	.long	.L114-.L78
	.long	.L91-.L78
	.long	.L90-.L78
	.long	.L89-.L78
	.long	.L114-.L78
	.long	.L114-.L78
	.long	.L88-.L78
	.long	.L114-.L78
	.long	.L87-.L78
	.long	.L86-.L78
	.long	.L85-.L78
	.long	.L84-.L78
	.long	.L83-.L78
	.long	.L82-.L78
	.long	.L81-.L78
	.long	.L80-.L78
	.long	.L114-.L78
	.long	.L79-.L78
	.long	.L77-.L78
	.text
.L87:
	addl	$1, -22864(%rbp)
	movq	$23, -22840(%rbp)
	jmp	.L97
.L80:
	leaq	-10016(%rbp), %rcx
	leaq	-20016(%rbp), %rdx
	leaq	-22416(%rbp), %rsi
	movq	-22888(%rbp), %rax
	movq	%rax, %rdi
	call	readBoards
	movq	%rax, -22832(%rbp)
	movq	-22832(%rbp), %rax
	movq	%rax, -22848(%rbp)
	movl	$0, -22872(%rbp)
	movq	$20, -22840(%rbp)
	jmp	.L97
.L94:
	movl	-22872(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-22856(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %ecx
	movl	-22872(%rbp), %edx
	movq	-22848(%rbp), %rax
	movl	%ecx, %esi
	movq	%rax, %rdi
	call	iterate
	addl	$1, -22872(%rbp)
	movq	$20, -22840(%rbp)
	jmp	.L97
.L90:
	movl	-22864(%rbp), %eax
	movl	%eax, -22868(%rbp)
	movq	$18, -22840(%rbp)
	jmp	.L97
.L82:
	cmpl	$99, -22864(%rbp)
	jg	.L98
	movq	$13, -22840(%rbp)
	jmp	.L97
.L98:
	movq	$19, -22840(%rbp)
	jmp	.L97
.L88:
	leaq	-22816(%rbp), %rdx
	movq	-22888(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	readNumbers
	movq	%rax, -22824(%rbp)
	movq	-22824(%rbp), %rax
	movq	%rax, -22856(%rbp)
	movq	-22888(%rbp), %rax
	movq	%rax, %rdi
	call	fgetc@PLT
	movl	%eax, -22876(%rbp)
	movq	$27, -22840(%rbp)
	jmp	.L97
.L81:
	movl	-22864(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-22848(%rbp), %rax
	addq	%rdx, %rax
	movl	20(%rax), %ecx
	movl	-22868(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-22848(%rbp), %rax
	addq	%rdx, %rax
	movl	20(%rax), %eax
	cmpl	%eax, %ecx
	jle	.L100
	movq	$12, -22840(%rbp)
	jmp	.L97
.L100:
	movq	$18, -22840(%rbp)
	jmp	.L97
.L84:
	movq	stderr(%rip), %rax
	movl	-22864(%rbp), %edx
	leaq	.LC1(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	$24, -22840(%rbp)
	jmp	.L97
.L91:
	movl	-22860(%rbp), %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L112
	jmp	.L113
.L89:
	movl	-22864(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-22848(%rbp), %rax
	addq	%rdx, %rax
	movl	20(%rax), %eax
	testl	%eax, %eax
	jne	.L103
	movq	$21, -22840(%rbp)
	jmp	.L97
.L103:
	movq	$24, -22840(%rbp)
	jmp	.L97
.L86:
	movl	-22868(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-22848(%rbp), %rax
	addq	%rdx, %rax
	subq	$8, %rsp
	pushq	16(%rax)
	pushq	8(%rax)
	pushq	(%rax)
	call	getScore
	addq	$32, %rsp
	movl	%eax, -22860(%rbp)
	movq	$11, -22840(%rbp)
	jmp	.L97
.L79:
	cmpl	$10, -22876(%rbp)
	je	.L105
	movq	$0, -22840(%rbp)
	jmp	.L97
.L105:
	movq	$25, -22840(%rbp)
	jmp	.L97
.L83:
	cmpl	$2499, -22880(%rbp)
	jbe	.L107
	movq	$16, -22840(%rbp)
	jmp	.L97
.L107:
	movq	$2, -22840(%rbp)
	jmp	.L97
.L77:
	movq	$5, -22840(%rbp)
	jmp	.L97
.L93:
	movl	$0, -10016(%rbp)
	movl	$1, -22880(%rbp)
	movq	$22, -22840(%rbp)
	jmp	.L97
.L96:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$21, %edx
	movl	$1, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	$25, -22840(%rbp)
	jmp	.L97
.L92:
	movl	$0, -22868(%rbp)
	movl	$1, -22864(%rbp)
	movq	$23, -22840(%rbp)
	jmp	.L97
.L95:
	movl	-22880(%rbp), %eax
	movl	$0, -10016(%rbp,%rax,4)
	addl	$1, -22880(%rbp)
	movq	$22, -22840(%rbp)
	jmp	.L97
.L85:
	cmpl	$99, -22872(%rbp)
	jg	.L109
	movq	$4, -22840(%rbp)
	jmp	.L97
.L109:
	movq	$7, -22840(%rbp)
	jmp	.L97
.L114:
	nop
.L97:
	jmp	.L111
.L113:
	call	__stack_chk_fail@PLT
.L112:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	part2, .-part2
	.globl	iterate
	.type	iterate, @function
iterate:
.LFB6:
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
	movl	%edx, -48(%rbp)
	movq	$5, -8(%rbp)
.L144:
	cmpq	$20, -8(%rbp)
	ja	.L146
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L118(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L118(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L118:
	.long	.L131-.L118
	.long	.L130-.L118
	.long	.L129-.L118
	.long	.L146-.L118
	.long	.L146-.L118
	.long	.L128-.L118
	.long	.L127-.L118
	.long	.L126-.L118
	.long	.L125-.L118
	.long	.L146-.L118
	.long	.L146-.L118
	.long	.L146-.L118
	.long	.L124-.L118
	.long	.L123-.L118
	.long	.L122-.L118
	.long	.L146-.L118
	.long	.L121-.L118
	.long	.L120-.L118
	.long	.L146-.L118
	.long	.L119-.L118
	.long	.L117-.L118
	.text
.L122:
	addl	$1, -16(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L132
.L124:
	movl	-20(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-40(%rbp), %rax
	addq	%rax, %rdx
	movl	-44(%rbp), %eax
	movl	%eax, 16(%rdx)
	movl	-20(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	pushq	16(%rax)
	pushq	8(%rax)
	pushq	(%rax)
	call	hasWon
	addq	$24, %rsp
	movl	%eax, -12(%rbp)
	movq	$20, -8(%rbp)
	jmp	.L132
.L125:
	movl	-20(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movl	20(%rax), %eax
	testl	%eax, %eax
	je	.L133
	movq	$7, -8(%rbp)
	jmp	.L132
.L133:
	movq	$2, -8(%rbp)
	jmp	.L132
.L130:
	movl	-20(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movl	-48(%rbp), %edx
	addl	$1, %edx
	movl	%edx, 20(%rax)
	movl	-20(%rbp), %eax
	movl	%eax, -24(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L132
.L121:
	movl	-24(%rbp), %eax
	jmp	.L145
.L123:
	cmpl	$99, -20(%rbp)
	jg	.L136
	movq	$8, -8(%rbp)
	jmp	.L132
.L136:
	movq	$16, -8(%rbp)
	jmp	.L132
.L119:
	movl	-20(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movq	8(%rax), %rdx
	movl	-16(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	$1, (%rax)
	movq	$14, -8(%rbp)
	jmp	.L132
.L120:
	movl	$-1, -24(%rbp)
	movl	$0, -20(%rbp)
	movq	$13, -8(%rbp)
	jmp	.L132
.L127:
	movl	-20(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rdx
	movl	-16(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	%eax, -44(%rbp)
	jne	.L138
	movq	$19, -8(%rbp)
	jmp	.L132
.L138:
	movq	$14, -8(%rbp)
	jmp	.L132
.L128:
	movq	$17, -8(%rbp)
	jmp	.L132
.L131:
	cmpl	$24, -16(%rbp)
	jg	.L140
	movq	$6, -8(%rbp)
	jmp	.L132
.L140:
	movq	$12, -8(%rbp)
	jmp	.L132
.L126:
	addl	$1, -20(%rbp)
	movq	$13, -8(%rbp)
	jmp	.L132
.L129:
	movl	$0, -16(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L132
.L117:
	cmpl	$0, -12(%rbp)
	je	.L142
	movq	$1, -8(%rbp)
	jmp	.L132
.L142:
	movq	$7, -8(%rbp)
	jmp	.L132
.L146:
	nop
.L132:
	jmp	.L144
.L145:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	iterate, .-iterate
	.globl	readBoards
	.type	readBoards, @function
readBoards:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$144, %rsp
	movq	%rdi, -120(%rbp)
	movq	%rsi, -128(%rbp)
	movq	%rdx, -136(%rbp)
	movq	%rcx, -144(%rbp)
	movq	$30, -72(%rbp)
.L178:
	cmpq	$30, -72(%rbp)
	ja	.L180
	movq	-72(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L150(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L150(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L150:
	.long	.L164-.L150
	.long	.L163-.L150
	.long	.L180-.L150
	.long	.L180-.L150
	.long	.L162-.L150
	.long	.L161-.L150
	.long	.L160-.L150
	.long	.L180-.L150
	.long	.L180-.L150
	.long	.L180-.L150
	.long	.L159-.L150
	.long	.L158-.L150
	.long	.L180-.L150
	.long	.L180-.L150
	.long	.L180-.L150
	.long	.L180-.L150
	.long	.L157-.L150
	.long	.L156-.L150
	.long	.L180-.L150
	.long	.L155-.L150
	.long	.L180-.L150
	.long	.L154-.L150
	.long	.L153-.L150
	.long	.L180-.L150
	.long	.L152-.L150
	.long	.L180-.L150
	.long	.L180-.L150
	.long	.L180-.L150
	.long	.L180-.L150
	.long	.L151-.L150
	.long	.L149-.L150
	.text
.L162:
	movl	-92(%rbp), %edx
	movl	%edx, %eax
	sall	$2, %eax
	addl	%edx, %eax
	addl	%eax, %eax
	movl	%eax, %edx
	movl	-96(%rbp), %eax
	subl	$48, %eax
	addl	%edx, %eax
	movl	%eax, -92(%rbp)
	movq	-120(%rbp), %rax
	movq	%rax, %rdi
	call	fgetc@PLT
	movl	%eax, -96(%rbp)
	movq	$22, -72(%rbp)
	jmp	.L165
.L149:
	movq	$24, -72(%rbp)
	jmp	.L165
.L163:
	movl	-100(%rbp), %eax
	movl	%eax, -80(%rbp)
	addl	$1, -100(%rbp)
	movl	-104(%rbp), %edx
	movl	%edx, %eax
	sall	$2, %eax
	addl	%eax, %edx
	movl	-80(%rbp), %eax
	addl	%edx, %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-136(%rbp), %rax
	addq	%rax, %rdx
	movl	-92(%rbp), %eax
	movl	%eax, (%rdx)
	movl	$0, -92(%rbp)
	movq	$16, -72(%rbp)
	jmp	.L165
.L157:
	movq	-120(%rbp), %rax
	movq	%rax, %rdi
	call	fgetc@PLT
	movl	%eax, -96(%rbp)
	movq	$6, -72(%rbp)
	jmp	.L165
.L152:
	movl	$0, -108(%rbp)
	movl	$0, -104(%rbp)
	movl	$0, -100(%rbp)
	movq	$11, -72(%rbp)
	jmp	.L165
.L154:
	movl	$0, -92(%rbp)
	movq	$22, -72(%rbp)
	jmp	.L165
.L158:
	movq	-120(%rbp), %rax
	movq	%rax, %rdi
	call	fgetc@PLT
	movl	%eax, -96(%rbp)
	movq	$19, -72(%rbp)
	jmp	.L165
.L155:
	cmpl	$32, -96(%rbp)
	jne	.L166
	movq	$11, -72(%rbp)
	jmp	.L165
.L166:
	movq	$21, -72(%rbp)
	jmp	.L165
.L156:
	movl	-108(%rbp), %eax
	movl	%eax, -88(%rbp)
	addl	$1, -108(%rbp)
	movq	-136(%rbp), %rax
	movq	%rax, -64(%rbp)
	movq	-144(%rbp), %rax
	movq	%rax, -56(%rbp)
	movl	$-1, -48(%rbp)
	movl	$0, -44(%rbp)
	movl	-88(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-128(%rbp), %rax
	leaq	(%rdx,%rax), %rcx
	movq	-64(%rbp), %rax
	movq	-56(%rbp), %rdx
	movq	%rax, (%rcx)
	movq	%rdx, 8(%rcx)
	movq	-48(%rbp), %rax
	movq	%rax, 16(%rcx)
	addq	$100, -136(%rbp)
	addq	$100, -144(%rbp)
	movl	$0, -104(%rbp)
	movq	$16, -72(%rbp)
	jmp	.L165
.L160:
	cmpl	$32, -96(%rbp)
	jne	.L168
	movq	$16, -72(%rbp)
	jmp	.L165
.L168:
	movq	$22, -72(%rbp)
	jmp	.L165
.L153:
	cmpl	$32, -96(%rbp)
	je	.L170
	cmpl	$32, -96(%rbp)
	jg	.L171
	cmpl	$-1, -96(%rbp)
	je	.L172
	cmpl	$10, -96(%rbp)
	je	.L173
	jmp	.L171
.L172:
	movq	$0, -72(%rbp)
	jmp	.L174
.L173:
	movq	$10, -72(%rbp)
	jmp	.L174
.L170:
	movq	$1, -72(%rbp)
	jmp	.L174
.L171:
	movq	$4, -72(%rbp)
	nop
.L174:
	jmp	.L165
.L161:
	movl	-104(%rbp), %eax
	movl	%eax, -84(%rbp)
	addl	$1, -104(%rbp)
	movl	-84(%rbp), %edx
	movl	%edx, %eax
	sall	$2, %eax
	addl	%eax, %edx
	movl	-100(%rbp), %eax
	addl	%edx, %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-136(%rbp), %rax
	addq	%rax, %rdx
	movl	-92(%rbp), %eax
	movl	%eax, (%rdx)
	movl	$0, -100(%rbp)
	movl	$0, -92(%rbp)
	movq	$16, -72(%rbp)
	jmp	.L165
.L159:
	cmpl	$0, -100(%rbp)
	jne	.L175
	movq	$17, -72(%rbp)
	jmp	.L165
.L175:
	movq	$5, -72(%rbp)
	jmp	.L165
.L164:
	movl	-108(%rbp), %eax
	movl	%eax, -76(%rbp)
	addl	$1, -108(%rbp)
	movq	-136(%rbp), %rax
	movq	%rax, -32(%rbp)
	movq	-144(%rbp), %rax
	movq	%rax, -24(%rbp)
	movl	$-1, -16(%rbp)
	movl	$0, -12(%rbp)
	movl	-76(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-128(%rbp), %rax
	leaq	(%rdx,%rax), %rcx
	movq	-32(%rbp), %rax
	movq	-24(%rbp), %rdx
	movq	%rax, (%rcx)
	movq	%rdx, 8(%rcx)
	movq	-16(%rbp), %rax
	movq	%rax, 16(%rcx)
	movq	$29, -72(%rbp)
	jmp	.L165
.L151:
	movq	-128(%rbp), %rax
	jmp	.L179
.L180:
	nop
.L165:
	jmp	.L178
.L179:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	readBoards, .-readBoards
	.globl	readNumbers
	.type	readNumbers, @function
readNumbers:
.LFB10:
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
	movq	$1, -8(%rbp)
.L199:
	cmpq	$11, -8(%rbp)
	ja	.L201
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L184(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L184(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L184:
	.long	.L192-.L184
	.long	.L191-.L184
	.long	.L190-.L184
	.long	.L201-.L184
	.long	.L189-.L184
	.long	.L188-.L184
	.long	.L187-.L184
	.long	.L186-.L184
	.long	.L201-.L184
	.long	.L185-.L184
	.long	.L201-.L184
	.long	.L183-.L184
	.text
.L189:
	movl	-28(%rbp), %eax
	movl	%eax, -12(%rbp)
	addl	$1, -28(%rbp)
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-48(%rbp), %rax
	addq	%rax, %rdx
	movl	-24(%rbp), %eax
	movl	%eax, (%rdx)
	movq	$5, -8(%rbp)
	jmp	.L193
.L191:
	movq	$6, -8(%rbp)
	jmp	.L193
.L183:
	movl	-28(%rbp), %eax
	movl	%eax, -16(%rbp)
	addl	$1, -28(%rbp)
	movl	-16(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-48(%rbp), %rax
	addq	%rax, %rdx
	movl	-24(%rbp), %eax
	movl	%eax, (%rdx)
	movl	$0, -24(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L193
.L185:
	cmpl	$44, -20(%rbp)
	jne	.L194
	movq	$11, -8(%rbp)
	jmp	.L193
.L194:
	movq	$7, -8(%rbp)
	jmp	.L193
.L187:
	movl	$0, -28(%rbp)
	movl	$0, -24(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L193
.L188:
	movq	-48(%rbp), %rax
	jmp	.L200
.L192:
	cmpl	$10, -20(%rbp)
	je	.L197
	movq	$9, -8(%rbp)
	jmp	.L193
.L197:
	movq	$4, -8(%rbp)
	jmp	.L193
.L186:
	movl	-24(%rbp), %edx
	movl	%edx, %eax
	sall	$2, %eax
	addl	%edx, %eax
	addl	%eax, %eax
	movl	%eax, %edx
	movl	-20(%rbp), %eax
	subl	$48, %eax
	addl	%edx, %eax
	movl	%eax, -24(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L193
.L190:
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	fgetc@PLT
	movl	%eax, -20(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L193
.L201:
	nop
.L193:
	jmp	.L199
.L200:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	readNumbers, .-readNumbers
	.section	.rodata
.LC2:
	.string	"r"
.LC3:
	.string	"in4"
.LC4:
	.string	"Part1: %d\n"
.LC5:
	.string	"Part2: %d\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB12:
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
	movq	$0, _TIG_IZ_XU3m_envp(%rip)
	nop
.L203:
	movq	$0, _TIG_IZ_XU3m_argv(%rip)
	nop
.L204:
	movl	$0, _TIG_IZ_XU3m_argc(%rip)
	nop
	nop
.L205:
.L206:
#APP
# 172 "timber-they_AdventOfCode21_4.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-XU3m--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_XU3m_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_XU3m_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_XU3m_envp(%rip)
	nop
	movq	$0, -24(%rbp)
.L212:
	cmpq	$2, -24(%rbp)
	je	.L207
	cmpq	$2, -24(%rbp)
	ja	.L214
	cmpq	$0, -24(%rbp)
	je	.L209
	cmpq	$1, -24(%rbp)
	jne	.L214
	leaq	.LC2(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	part1
	movl	%eax, -32(%rbp)
	movl	-32(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	rewind@PLT
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	part2
	movl	%eax, -28(%rbp)
	movl	-28(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$2, -24(%rbp)
	jmp	.L210
.L209:
	movq	$1, -24(%rbp)
	jmp	.L210
.L207:
	movl	$0, %eax
	jmp	.L213
.L214:
	nop
.L210:
	jmp	.L212
.L213:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE12:
	.size	main, .-main
	.globl	getScore
	.type	getScore, @function
getScore:
.LFB13:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	$2, -8(%rbp)
.L231:
	cmpq	$8, -8(%rbp)
	ja	.L233
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L218(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L218(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L218:
	.long	.L224-.L218
	.long	.L223-.L218
	.long	.L222-.L218
	.long	.L233-.L218
	.long	.L233-.L218
	.long	.L221-.L218
	.long	.L220-.L218
	.long	.L219-.L218
	.long	.L217-.L218
	.text
.L217:
	movl	32(%rbp), %eax
	imull	-16(%rbp), %eax
	jmp	.L232
.L223:
	movl	$0, -16(%rbp)
	movl	$0, -12(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L226
.L220:
	addl	$1, -12(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L226
.L221:
	movq	16(%rbp), %rdx
	movl	-12(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	addl	%eax, -16(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L226
.L224:
	movq	24(%rbp), %rdx
	movl	-12(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	testl	%eax, %eax
	jne	.L227
	movq	$5, -8(%rbp)
	jmp	.L226
.L227:
	movq	$6, -8(%rbp)
	jmp	.L226
.L219:
	cmpl	$24, -12(%rbp)
	jg	.L229
	movq	$0, -8(%rbp)
	jmp	.L226
.L229:
	movq	$8, -8(%rbp)
	jmp	.L226
.L222:
	movq	$1, -8(%rbp)
	jmp	.L226
.L233:
	nop
.L226:
	jmp	.L231
.L232:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE13:
	.size	getScore, .-getScore
	.section	.rodata
.LC6:
	.string	"Last: %d\n"
.LC7:
	.string	"\342\226\210"
.LC8:
	.string	" "
.LC9:
	.string	"%s%2d "
	.text
	.globl	printBoard
	.type	printBoard, @function
printBoard:
.LFB14:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	$2, -8(%rbp)
.L256:
	cmpq	$16, -8(%rbp)
	ja	.L257
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L237(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L237(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L237:
	.long	.L257-.L237
	.long	.L257-.L237
	.long	.L247-.L237
	.long	.L246-.L237
	.long	.L245-.L237
	.long	.L244-.L237
	.long	.L243-.L237
	.long	.L257-.L237
	.long	.L257-.L237
	.long	.L242-.L237
	.long	.L257-.L237
	.long	.L241-.L237
	.long	.L257-.L237
	.long	.L240-.L237
	.long	.L239-.L237
	.long	.L258-.L237
	.long	.L236-.L237
	.text
.L245:
	movl	32(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$0, -24(%rbp)
	movq	$13, -8(%rbp)
	jmp	.L248
.L239:
	movl	$0, -20(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L248
.L246:
	leaq	.LC7(%rip), %rax
	movq	%rax, -16(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L248
.L236:
	movq	24(%rbp), %rcx
	movl	-24(%rbp), %edx
	movl	%edx, %eax
	sall	$2, %eax
	addl	%eax, %edx
	movl	-20(%rbp), %eax
	addl	%edx, %eax
	cltq
	salq	$2, %rax
	addq	%rcx, %rax
	movl	(%rax), %eax
	testl	%eax, %eax
	je	.L250
	movq	$3, -8(%rbp)
	jmp	.L248
.L250:
	movq	$9, -8(%rbp)
	jmp	.L248
.L241:
	movl	$10, %edi
	call	putchar@PLT
	addl	$1, -24(%rbp)
	movq	$13, -8(%rbp)
	jmp	.L248
.L242:
	leaq	.LC8(%rip), %rax
	movq	%rax, -16(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L248
.L240:
	cmpl	$4, -24(%rbp)
	jg	.L252
	movq	$14, -8(%rbp)
	jmp	.L248
.L252:
	movq	$15, -8(%rbp)
	jmp	.L248
.L243:
	cmpl	$4, -20(%rbp)
	jg	.L254
	movq	$16, -8(%rbp)
	jmp	.L248
.L254:
	movq	$11, -8(%rbp)
	jmp	.L248
.L244:
	movq	16(%rbp), %rcx
	movl	-24(%rbp), %edx
	movl	%edx, %eax
	sall	$2, %eax
	addl	%eax, %edx
	movl	-20(%rbp), %eax
	addl	%edx, %eax
	cltq
	salq	$2, %rax
	addq	%rcx, %rax
	movl	(%rax), %edx
	movq	-16(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -20(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L248
.L247:
	movq	$4, -8(%rbp)
	jmp	.L248
.L257:
	nop
.L248:
	jmp	.L256
.L258:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE14:
	.size	printBoard, .-printBoard
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
