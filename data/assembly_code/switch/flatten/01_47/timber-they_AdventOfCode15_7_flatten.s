	.file	"timber-they_AdventOfCode15_7_flatten.c"
	.text
	.globl	_TIG_IZ_tBjr_envp
	.bss
	.align 8
	.type	_TIG_IZ_tBjr_envp, @object
	.size	_TIG_IZ_tBjr_envp, 8
_TIG_IZ_tBjr_envp:
	.zero	8
	.globl	known
	.align 32
	.type	known, @object
	.size	known, 2916
known:
	.zero	2916
	.globl	_TIG_IZ_tBjr_argc
	.align 4
	.type	_TIG_IZ_tBjr_argc, @object
	.size	_TIG_IZ_tBjr_argc, 4
_TIG_IZ_tBjr_argc:
	.zero	4
	.globl	knownValues
	.align 32
	.type	knownValues, @object
	.size	knownValues, 1458
knownValues:
	.zero	1458
	.globl	_TIG_IZ_tBjr_argv
	.align 8
	.type	_TIG_IZ_tBjr_argv, @object
	.size	_TIG_IZ_tBjr_argv, 8
_TIG_IZ_tBjr_argv:
	.zero	8
	.text
	.globl	isdigit
	.type	isdigit, @function
isdigit:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -20(%rbp)
	movq	$1, -8(%rbp)
.L16:
	cmpq	$5, -8(%rbp)
	ja	.L18
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
	.long	.L9-.L4
	.long	.L8-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L5:
	movl	$0, -12(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L10
.L8:
	cmpl	$47, -20(%rbp)
	jle	.L11
	movq	$5, -8(%rbp)
	jmp	.L10
.L11:
	movq	$3, -8(%rbp)
	jmp	.L10
.L6:
	movl	$0, -12(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L10
.L3:
	cmpl	$57, -20(%rbp)
	jg	.L13
	movq	$2, -8(%rbp)
	jmp	.L10
.L13:
	movq	$4, -8(%rbp)
	jmp	.L10
.L9:
	movl	-12(%rbp), %eax
	jmp	.L17
.L7:
	movl	$1, -12(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L10
.L18:
	nop
.L10:
	jmp	.L16
.L17:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	isdigit, .-isdigit
	.section	.rodata
.LC0:
	.string	"%s -> %s\n"
.LC1:
	.string	"%s AND %s -> %s\n"
.LC2:
	.string	"Couldn't parse line %s (%d)\n"
.LC3:
	.string	"LSHIFT"
.LC4:
	.string	"%s RSHIFT %s -> %s\n"
.LC5:
	.string	"%s OR %s -> %s\n"
.LC6:
	.string	"NOT %s -> %s\n"
.LC7:
	.string	"OR"
.LC8:
	.string	"RSHIFT"
.LC9:
	.string	"%s LSHIFT %s -> %s\n"
.LC10:
	.string	"AND"
.LC11:
	.string	"NOT"
	.text
	.globl	parse
	.type	parse, @function
parse:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$160, %rsp
	movq	%rdi, -152(%rbp)
	movq	%rsi, -160(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$31, -72(%rbp)
.L117:
	cmpq	$68, -72(%rbp)
	ja	.L120
	movq	-72(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L22(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L22(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L22:
	.long	.L74-.L22
	.long	.L73-.L22
	.long	.L72-.L22
	.long	.L71-.L22
	.long	.L120-.L22
	.long	.L70-.L22
	.long	.L69-.L22
	.long	.L68-.L22
	.long	.L67-.L22
	.long	.L66-.L22
	.long	.L65-.L22
	.long	.L64-.L22
	.long	.L63-.L22
	.long	.L120-.L22
	.long	.L62-.L22
	.long	.L61-.L22
	.long	.L60-.L22
	.long	.L59-.L22
	.long	.L58-.L22
	.long	.L120-.L22
	.long	.L57-.L22
	.long	.L56-.L22
	.long	.L55-.L22
	.long	.L54-.L22
	.long	.L53-.L22
	.long	.L120-.L22
	.long	.L52-.L22
	.long	.L51-.L22
	.long	.L120-.L22
	.long	.L50-.L22
	.long	.L49-.L22
	.long	.L48-.L22
	.long	.L47-.L22
	.long	.L120-.L22
	.long	.L120-.L22
	.long	.L120-.L22
	.long	.L46-.L22
	.long	.L45-.L22
	.long	.L44-.L22
	.long	.L43-.L22
	.long	.L42-.L22
	.long	.L120-.L22
	.long	.L41-.L22
	.long	.L40-.L22
	.long	.L39-.L22
	.long	.L38-.L22
	.long	.L37-.L22
	.long	.L36-.L22
	.long	.L35-.L22
	.long	.L120-.L22
	.long	.L34-.L22
	.long	.L33-.L22
	.long	.L32-.L22
	.long	.L120-.L22
	.long	.L31-.L22
	.long	.L120-.L22
	.long	.L120-.L22
	.long	.L30-.L22
	.long	.L29-.L22
	.long	.L28-.L22
	.long	.L27-.L22
	.long	.L26-.L22
	.long	.L25-.L22
	.long	.L24-.L22
	.long	.L120-.L22
	.long	.L120-.L22
	.long	.L120-.L22
	.long	.L23-.L22
	.long	.L21-.L22
	.text
.L58:
	cmpq	$0, -88(%rbp)
	je	.L75
	movq	$37, -72(%rbp)
	jmp	.L77
.L75:
	movq	$29, -72(%rbp)
	jmp	.L77
.L34:
	movq	-160(%rbp), %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	subl	$48, %eax
	cmpl	$9, %eax
	setbe	%al
	movzbl	%al, %eax
	movl	%eax, -132(%rbp)
	movq	$20, -72(%rbp)
	jmp	.L77
.L32:
	leaq	-18(%rbp), %rcx
	leaq	-38(%rbp), %rdx
	movq	-160(%rbp), %rax
	leaq	.LC0(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_sscanf@PLT
	movl	%eax, -136(%rbp)
	movq	$39, -72(%rbp)
	jmp	.L77
.L49:
	leaq	-18(%rbp), %rsi
	leaq	-28(%rbp), %rcx
	leaq	-38(%rbp), %rdx
	movq	-160(%rbp), %rax
	movq	%rsi, %r8
	leaq	.LC1(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_sscanf@PLT
	movl	%eax, -136(%rbp)
	movq	$47, -72(%rbp)
	jmp	.L77
.L25:
	movl	$0, -64(%rbp)
	movq	$1, -72(%rbp)
	jmp	.L77
.L62:
	cmpl	$3, -136(%rbp)
	je	.L78
	movq	$23, -72(%rbp)
	jmp	.L77
.L78:
	movq	$21, -72(%rbp)
	jmp	.L77
.L61:
	movq	stderr(%rip), %rax
	movl	-136(%rbp), %ecx
	movq	-160(%rbp), %rdx
	leaq	.LC2(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movl	$3, %edi
	call	exit@PLT
.L48:
	movq	$40, -72(%rbp)
	jmp	.L77
.L63:
	cmpl	$0, -124(%rbp)
	je	.L80
	movq	$51, -72(%rbp)
	jmp	.L77
.L80:
	movq	$32, -72(%rbp)
	jmp	.L77
.L67:
	cmpl	$2, -136(%rbp)
	je	.L82
	movq	$15, -72(%rbp)
	jmp	.L77
.L82:
	movq	$21, -72(%rbp)
	jmp	.L77
.L38:
	movl	$1, -64(%rbp)
	movq	$1, -72(%rbp)
	jmp	.L77
.L31:
	movq	$21, -72(%rbp)
	jmp	.L77
.L73:
	movl	-64(%rbp), %eax
	cmpl	$6, %eax
	ja	.L84
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L86(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L86(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L86:
	.long	.L92-.L86
	.long	.L91-.L86
	.long	.L90-.L86
	.long	.L89-.L86
	.long	.L88-.L86
	.long	.L87-.L86
	.long	.L85-.L86
	.text
.L91:
	movq	$52, -72(%rbp)
	jmp	.L93
.L92:
	movq	$59, -72(%rbp)
	jmp	.L93
.L87:
	movq	$38, -72(%rbp)
	jmp	.L93
.L88:
	movq	$36, -72(%rbp)
	jmp	.L93
.L85:
	movq	$63, -72(%rbp)
	jmp	.L93
.L89:
	movq	$30, -72(%rbp)
	jmp	.L93
.L90:
	movq	$68, -72(%rbp)
	jmp	.L93
.L84:
	movq	$54, -72(%rbp)
	nop
.L93:
	jmp	.L77
.L54:
	movq	stderr(%rip), %rax
	movl	-136(%rbp), %ecx
	movq	-160(%rbp), %rdx
	leaq	.LC2(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movl	$3, %edi
	call	exit@PLT
.L71:
	movq	-160(%rbp), %rax
	leaq	.LC3(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strstr@PLT
	movq	%rax, -112(%rbp)
	movq	$44, -72(%rbp)
	jmp	.L77
.L60:
	cmpl	$0, -128(%rbp)
	je	.L94
	movq	$60, -72(%rbp)
	jmp	.L77
.L94:
	movq	$24, -72(%rbp)
	jmp	.L77
.L53:
	leaq	-38(%rbp), %rax
	movq	%rax, %rdi
	call	identify
	movl	%eax, -56(%rbp)
	movq	$17, -72(%rbp)
	jmp	.L77
.L56:
	movzbl	-38(%rbp), %eax
	movsbl	%al, %eax
	subl	$48, %eax
	cmpl	$9, %eax
	setbe	%al
	movzbl	%al, %eax
	movl	%eax, -128(%rbp)
	movq	$16, -72(%rbp)
	jmp	.L77
.L46:
	leaq	-18(%rbp), %rsi
	leaq	-28(%rbp), %rcx
	leaq	-38(%rbp), %rdx
	movq	-160(%rbp), %rax
	movq	%rsi, %r8
	leaq	.LC4(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_sscanf@PLT
	movl	%eax, -136(%rbp)
	movq	$0, -72(%rbp)
	jmp	.L77
.L30:
	movq	-152(%rbp), %rcx
	movq	-64(%rbp), %rax
	movq	-56(%rbp), %rdx
	movq	%rax, (%rcx)
	movq	%rdx, 8(%rcx)
	movl	-48(%rbp), %eax
	movl	%eax, 16(%rcx)
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L118
	jmp	.L119
.L21:
	leaq	-18(%rbp), %rsi
	leaq	-28(%rbp), %rcx
	leaq	-38(%rbp), %rdx
	movq	-160(%rbp), %rax
	movq	%rsi, %r8
	leaq	.LC5(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_sscanf@PLT
	movl	%eax, -136(%rbp)
	movq	$26, -72(%rbp)
	jmp	.L77
.L52:
	cmpl	$3, -136(%rbp)
	je	.L97
	movq	$9, -72(%rbp)
	jmp	.L77
.L97:
	movq	$21, -72(%rbp)
	jmp	.L77
.L64:
	movq	stderr(%rip), %rax
	movl	-136(%rbp), %ecx
	movq	-160(%rbp), %rdx
	leaq	.LC2(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movl	$3, %edi
	call	exit@PLT
.L66:
	movq	stderr(%rip), %rax
	movl	-136(%rbp), %ecx
	movq	-160(%rbp), %rdx
	leaq	.LC2(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movl	$3, %edi
	call	exit@PLT
.L24:
	leaq	-18(%rbp), %rcx
	leaq	-38(%rbp), %rdx
	movq	-160(%rbp), %rax
	leaq	.LC6(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_sscanf@PLT
	movl	%eax, -136(%rbp)
	movq	$10, -72(%rbp)
	jmp	.L77
.L33:
	leaq	-28(%rbp), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movl	%eax, -116(%rbp)
	movl	-116(%rbp), %eax
	movw	%ax, -60(%rbp)
	movq	$61, -72(%rbp)
	jmp	.L77
.L47:
	leaq	-28(%rbp), %rax
	movq	%rax, %rdi
	call	identify
	movl	%eax, -52(%rbp)
	movq	$61, -72(%rbp)
	jmp	.L77
.L59:
	movzbl	-28(%rbp), %eax
	movsbl	%al, %eax
	subl	$48, %eax
	cmpl	$9, %eax
	setbe	%al
	movzbl	%al, %eax
	movl	%eax, -124(%rbp)
	movq	$12, -72(%rbp)
	jmp	.L77
.L42:
	movl	$0, -64(%rbp)
	movw	$0, -60(%rbp)
	movl	$0, -56(%rbp)
	movl	$0, -52(%rbp)
	movl	$0, -48(%rbp)
	movq	-160(%rbp), %rax
	leaq	.LC7(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strstr@PLT
	movq	%rax, -80(%rbp)
	movq	$46, -72(%rbp)
	jmp	.L77
.L23:
	movl	$6, -64(%rbp)
	movq	$1, -72(%rbp)
	jmp	.L77
.L27:
	leaq	-38(%rbp), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movl	%eax, -120(%rbp)
	movl	-120(%rbp), %eax
	movw	%ax, -60(%rbp)
	movq	$17, -72(%rbp)
	jmp	.L77
.L28:
	leaq	-18(%rbp), %rcx
	leaq	-38(%rbp), %rdx
	movq	-160(%rbp), %rax
	leaq	.LC0(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_sscanf@PLT
	movl	%eax, -136(%rbp)
	movq	$8, -72(%rbp)
	jmp	.L77
.L69:
	movq	-160(%rbp), %rax
	leaq	.LC8(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strstr@PLT
	movq	%rax, -104(%rbp)
	movq	$2, -72(%rbp)
	jmp	.L77
.L51:
	cmpq	$0, -96(%rbp)
	je	.L99
	movq	$67, -72(%rbp)
	jmp	.L77
.L99:
	movq	$6, -72(%rbp)
	jmp	.L77
.L44:
	leaq	-18(%rbp), %rsi
	leaq	-28(%rbp), %rcx
	leaq	-38(%rbp), %rdx
	movq	-160(%rbp), %rax
	movq	%rsi, %r8
	leaq	.LC9(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_sscanf@PLT
	movl	%eax, -136(%rbp)
	movq	$14, -72(%rbp)
	jmp	.L77
.L26:
	leaq	-18(%rbp), %rax
	movq	%rax, %rdi
	call	identify
	movl	%eax, -48(%rbp)
	movq	$57, -72(%rbp)
	jmp	.L77
.L29:
	movq	-160(%rbp), %rax
	leaq	.LC10(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strstr@PLT
	movq	%rax, -88(%rbp)
	movq	$18, -72(%rbp)
	jmp	.L77
.L35:
	movq	stderr(%rip), %rax
	movl	-136(%rbp), %ecx
	movq	-160(%rbp), %rdx
	leaq	.LC2(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movl	$3, %edi
	call	exit@PLT
.L55:
	movl	$4, -64(%rbp)
	movq	$1, -72(%rbp)
	jmp	.L77
.L36:
	cmpl	$3, -136(%rbp)
	je	.L101
	movq	$42, -72(%rbp)
	jmp	.L77
.L101:
	movq	$21, -72(%rbp)
	jmp	.L77
.L39:
	cmpq	$0, -112(%rbp)
	je	.L103
	movq	$43, -72(%rbp)
	jmp	.L77
.L103:
	movq	$50, -72(%rbp)
	jmp	.L77
.L70:
	movl	$2, -64(%rbp)
	movq	$1, -72(%rbp)
	jmp	.L77
.L45:
	movl	$3, -64(%rbp)
	movq	$1, -72(%rbp)
	jmp	.L77
.L65:
	cmpl	$2, -136(%rbp)
	je	.L105
	movq	$48, -72(%rbp)
	jmp	.L77
.L105:
	movq	$21, -72(%rbp)
	jmp	.L77
.L41:
	movq	stderr(%rip), %rax
	movl	-136(%rbp), %ecx
	movq	-160(%rbp), %rdx
	leaq	.LC2(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movl	$3, %edi
	call	exit@PLT
.L74:
	cmpl	$3, -136(%rbp)
	je	.L107
	movq	$7, -72(%rbp)
	jmp	.L77
.L107:
	movq	$21, -72(%rbp)
	jmp	.L77
.L37:
	cmpq	$0, -80(%rbp)
	je	.L109
	movq	$5, -72(%rbp)
	jmp	.L77
.L109:
	movq	$58, -72(%rbp)
	jmp	.L77
.L43:
	cmpl	$2, -136(%rbp)
	je	.L111
	movq	$11, -72(%rbp)
	jmp	.L77
.L111:
	movq	$21, -72(%rbp)
	jmp	.L77
.L68:
	movq	stderr(%rip), %rax
	movl	-136(%rbp), %ecx
	movq	-160(%rbp), %rdx
	leaq	.LC2(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movl	$3, %edi
	call	exit@PLT
.L50:
	movq	-160(%rbp), %rax
	leaq	.LC11(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strstr@PLT
	movq	%rax, -96(%rbp)
	movq	$27, -72(%rbp)
	jmp	.L77
.L40:
	movl	$5, -64(%rbp)
	movq	$1, -72(%rbp)
	jmp	.L77
.L72:
	cmpq	$0, -104(%rbp)
	je	.L113
	movq	$22, -72(%rbp)
	jmp	.L77
.L113:
	movq	$3, -72(%rbp)
	jmp	.L77
.L57:
	cmpl	$0, -132(%rbp)
	je	.L115
	movq	$62, -72(%rbp)
	jmp	.L77
.L115:
	movq	$45, -72(%rbp)
	jmp	.L77
.L120:
	nop
.L77:
	jmp	.L117
.L119:
	call	__stack_chk_fail@PLT
.L118:
	movq	-152(%rbp), %rax
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	parse, .-parse
	.section	.rodata
.LC12:
	.string	"Invalid type: %d\n"
	.text
	.globl	getValue
	.type	getValue, @function
getValue:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$112, %rsp
	movl	%edi, -100(%rbp)
	movq	%rsi, -112(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$40, -72(%rbp)
.L168:
	cmpq	$40, -72(%rbp)
	ja	.L171
	movq	-72(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L124(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L124(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L124:
	.long	.L145-.L124
	.long	.L171-.L124
	.long	.L144-.L124
	.long	.L143-.L124
	.long	.L171-.L124
	.long	.L171-.L124
	.long	.L171-.L124
	.long	.L142-.L124
	.long	.L171-.L124
	.long	.L141-.L124
	.long	.L171-.L124
	.long	.L171-.L124
	.long	.L171-.L124
	.long	.L140-.L124
	.long	.L171-.L124
	.long	.L171-.L124
	.long	.L139-.L124
	.long	.L171-.L124
	.long	.L138-.L124
	.long	.L137-.L124
	.long	.L171-.L124
	.long	.L136-.L124
	.long	.L135-.L124
	.long	.L134-.L124
	.long	.L171-.L124
	.long	.L133-.L124
	.long	.L132-.L124
	.long	.L171-.L124
	.long	.L171-.L124
	.long	.L131-.L124
	.long	.L130-.L124
	.long	.L129-.L124
	.long	.L128-.L124
	.long	.L171-.L124
	.long	.L127-.L124
	.long	.L171-.L124
	.long	.L171-.L124
	.long	.L126-.L124
	.long	.L171-.L124
	.long	.L125-.L124
	.long	.L123-.L124
	.text
.L138:
	movl	-52(%rbp), %eax
	movq	-112(%rbp), %rdx
	movq	%rdx, %rsi
	movl	%eax, %edi
	call	getValue
	movw	%ax, -74(%rbp)
	movzwl	-60(%rbp), %eax
	andw	-74(%rbp), %ax
	movw	%ax, -96(%rbp)
	movq	$29, -72(%rbp)
	jmp	.L146
.L133:
	movl	-56(%rbp), %eax
	movq	-112(%rbp), %rdx
	movq	%rdx, %rsi
	movl	%eax, %edi
	call	getValue
	movw	%ax, -88(%rbp)
	movl	-52(%rbp), %eax
	movq	-112(%rbp), %rdx
	movq	%rdx, %rsi
	movl	%eax, %edi
	call	getValue
	movw	%ax, -86(%rbp)
	movzwl	-88(%rbp), %eax
	orw	-86(%rbp), %ax
	movw	%ax, -96(%rbp)
	movq	$29, -72(%rbp)
	jmp	.L146
.L130:
	movl	-56(%rbp), %eax
	movq	-112(%rbp), %rdx
	movq	%rdx, %rsi
	movl	%eax, %edi
	call	getValue
	movw	%ax, -90(%rbp)
	movzwl	-90(%rbp), %edx
	movzwl	-60(%rbp), %eax
	movzwl	%ax, %eax
	movl	%eax, %ecx
	sarl	%cl, %edx
	movl	%edx, %eax
	movw	%ax, -96(%rbp)
	movq	$29, -72(%rbp)
	jmp	.L146
.L129:
	movl	-56(%rbp), %eax
	movq	-112(%rbp), %rdx
	movq	%rdx, %rsi
	movl	%eax, %edi
	call	getValue
	movw	%ax, -96(%rbp)
	movq	$29, -72(%rbp)
	jmp	.L146
.L134:
	movl	-56(%rbp), %eax
	movq	-112(%rbp), %rdx
	movq	%rdx, %rsi
	movl	%eax, %edi
	call	getValue
	movw	%ax, -84(%rbp)
	movzwl	-84(%rbp), %eax
	notl	%eax
	movw	%ax, -96(%rbp)
	movq	$29, -72(%rbp)
	jmp	.L146
.L143:
	movl	-64(%rbp), %eax
	cmpl	$6, %eax
	ja	.L147
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L149(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L149(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L149:
	.long	.L155-.L149
	.long	.L154-.L149
	.long	.L153-.L149
	.long	.L152-.L149
	.long	.L151-.L149
	.long	.L150-.L149
	.long	.L148-.L149
	.text
.L148:
	movq	$23, -72(%rbp)
	jmp	.L156
.L150:
	movq	$39, -72(%rbp)
	jmp	.L156
.L151:
	movq	$30, -72(%rbp)
	jmp	.L156
.L152:
	movq	$7, -72(%rbp)
	jmp	.L156
.L153:
	movq	$2, -72(%rbp)
	jmp	.L156
.L154:
	movq	$31, -72(%rbp)
	jmp	.L156
.L155:
	movq	$32, -72(%rbp)
	jmp	.L156
.L147:
	movq	$26, -72(%rbp)
	nop
.L156:
	jmp	.L146
.L139:
	movl	-56(%rbp), %eax
	testl	%eax, %eax
	je	.L157
	movq	$34, -72(%rbp)
	jmp	.L146
.L157:
	movq	$22, -72(%rbp)
	jmp	.L146
.L136:
	movl	-56(%rbp), %eax
	movq	-112(%rbp), %rdx
	movq	%rdx, %rsi
	movl	%eax, %edi
	call	getValue
	movw	%ax, -80(%rbp)
	movzwl	-60(%rbp), %eax
	andw	-80(%rbp), %ax
	movw	%ax, -96(%rbp)
	movq	$29, -72(%rbp)
	jmp	.L146
.L132:
	movl	-64(%rbp), %edx
	movq	stderr(%rip), %rax
	leaq	.LC12(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movl	$1, %edi
	call	exit@PLT
.L141:
	leaq	-32(%rbp), %rax
	movq	-112(%rbp), %rdx
	movl	-100(%rbp), %ecx
	movl	%ecx, %esi
	movq	%rax, %rdi
	call	get
	movq	-32(%rbp), %rax
	movq	-24(%rbp), %rdx
	movq	%rax, -64(%rbp)
	movq	%rdx, -56(%rbp)
	movl	-16(%rbp), %eax
	movl	%eax, -48(%rbp)
	movq	$3, -72(%rbp)
	jmp	.L146
.L140:
	movzwl	-96(%rbp), %eax
	jmp	.L169
.L137:
	movl	-56(%rbp), %eax
	movq	-112(%rbp), %rdx
	movq	%rdx, %rsi
	movl	%eax, %edi
	call	getValue
	movw	%ax, -78(%rbp)
	movl	-52(%rbp), %eax
	movq	-112(%rbp), %rdx
	movq	%rdx, %rsi
	movl	%eax, %edi
	call	getValue
	movw	%ax, -76(%rbp)
	movzwl	-78(%rbp), %eax
	andw	-76(%rbp), %ax
	movw	%ax, -96(%rbp)
	movq	$29, -72(%rbp)
	jmp	.L146
.L128:
	movzwl	-60(%rbp), %eax
	movw	%ax, -96(%rbp)
	movq	$29, -72(%rbp)
	jmp	.L146
.L123:
	movl	-100(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	known(%rip), %rax
	movl	(%rdx,%rax), %eax
	testl	%eax, %eax
	je	.L160
	movq	$37, -72(%rbp)
	jmp	.L146
.L160:
	movq	$9, -72(%rbp)
	jmp	.L146
.L127:
	movl	-56(%rbp), %eax
	movq	-112(%rbp), %rdx
	movq	%rdx, %rsi
	movl	%eax, %edi
	call	getValue
	movw	%ax, -92(%rbp)
	movzwl	-60(%rbp), %eax
	orw	-92(%rbp), %ax
	movw	%ax, -96(%rbp)
	movq	$29, -72(%rbp)
	jmp	.L146
.L135:
	movl	-52(%rbp), %eax
	movq	-112(%rbp), %rdx
	movq	%rdx, %rsi
	movl	%eax, %edi
	call	getValue
	movw	%ax, -82(%rbp)
	movzwl	-60(%rbp), %eax
	orw	-82(%rbp), %ax
	movw	%ax, -96(%rbp)
	movq	$29, -72(%rbp)
	jmp	.L146
.L126:
	movl	-100(%rbp), %eax
	cltq
	leaq	(%rax,%rax), %rdx
	leaq	knownValues(%rip), %rax
	movzwl	(%rdx,%rax), %eax
	jmp	.L169
.L145:
	movl	-56(%rbp), %eax
	testl	%eax, %eax
	je	.L162
	movq	$21, -72(%rbp)
	jmp	.L146
.L162:
	movq	$18, -72(%rbp)
	jmp	.L146
.L125:
	movl	-56(%rbp), %eax
	movq	-112(%rbp), %rdx
	movq	%rdx, %rsi
	movl	%eax, %edi
	call	getValue
	movw	%ax, -94(%rbp)
	movzwl	-94(%rbp), %edx
	movzwl	-60(%rbp), %eax
	movzwl	%ax, %eax
	movl	%eax, %ecx
	sall	%cl, %edx
	movl	%edx, %eax
	movw	%ax, -96(%rbp)
	movq	$29, -72(%rbp)
	jmp	.L146
.L142:
	movzwl	-60(%rbp), %eax
	testw	%ax, %ax
	je	.L164
	movq	$0, -72(%rbp)
	jmp	.L146
.L164:
	movq	$19, -72(%rbp)
	jmp	.L146
.L131:
	movl	-100(%rbp), %eax
	cltq
	leaq	(%rax,%rax), %rcx
	leaq	knownValues(%rip), %rdx
	movzwl	-96(%rbp), %eax
	movw	%ax, (%rcx,%rdx)
	movl	-100(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	known(%rip), %rax
	movl	$1, (%rdx,%rax)
	movq	$13, -72(%rbp)
	jmp	.L146
.L144:
	movzwl	-60(%rbp), %eax
	testw	%ax, %ax
	je	.L166
	movq	$16, -72(%rbp)
	jmp	.L146
.L166:
	movq	$25, -72(%rbp)
	jmp	.L146
.L171:
	nop
.L146:
	jmp	.L168
.L169:
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L170
	call	__stack_chk_fail@PLT
.L170:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	getValue, .-getValue
	.section	.rodata
.LC13:
	.string	"b"
.LC14:
	.string	"a"
	.text
	.globl	part2
	.type	part2, @function
part2:
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
	movq	$11, -16(%rbp)
.L190:
	cmpq	$12, -16(%rbp)
	ja	.L192
	movq	-16(%rbp), %rax
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
	.long	.L183-.L175
	.long	.L182-.L175
	.long	.L192-.L175
	.long	.L181-.L175
	.long	.L192-.L175
	.long	.L180-.L175
	.long	.L179-.L175
	.long	.L192-.L175
	.long	.L178-.L175
	.long	.L192-.L175
	.long	.L177-.L175
	.long	.L176-.L175
	.long	.L174-.L175
	.text
.L174:
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	call	identify
	movl	%eax, -36(%rbp)
	movq	$0, -16(%rbp)
	jmp	.L184
.L178:
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	getConnections
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -24(%rbp)
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	call	identify
	movl	%eax, -32(%rbp)
	movq	-24(%rbp), %rdx
	movl	-32(%rbp), %eax
	movq	%rdx, %rsi
	movl	%eax, %edi
	call	getValue
	movw	%ax, -42(%rbp)
	movzwl	-42(%rbp), %eax
	movw	%ax, -44(%rbp)
	movl	$2916, %edx
	movl	$0, %esi
	leaq	known(%rip), %rax
	movq	%rax, %rdi
	call	memset@PLT
	movl	$0, -40(%rbp)
	movq	$10, -16(%rbp)
	jmp	.L184
.L182:
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	call	identify
	movl	%eax, -28(%rbp)
	movq	-24(%rbp), %rdx
	movl	-28(%rbp), %eax
	movq	%rdx, %rsi
	movl	%eax, %edi
	call	getValue
	movw	%ax, -44(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$5, -16(%rbp)
	jmp	.L184
.L181:
	addl	$1, -40(%rbp)
	movq	$10, -16(%rbp)
	jmp	.L184
.L176:
	movq	$8, -16(%rbp)
	jmp	.L184
.L179:
	movl	-40(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movl	$0, (%rax)
	movl	-40(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-24(%rbp), %rax
	addq	%rax, %rdx
	movzwl	-44(%rbp), %eax
	movw	%ax, 4(%rdx)
	movq	$3, -16(%rbp)
	jmp	.L184
.L180:
	movzwl	-44(%rbp), %eax
	jmp	.L191
.L177:
	cmpl	$338, -40(%rbp)
	jg	.L186
	movq	$12, -16(%rbp)
	jmp	.L184
.L186:
	movq	$1, -16(%rbp)
	jmp	.L184
.L183:
	movl	-40(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movl	16(%rax), %eax
	cmpl	%eax, -36(%rbp)
	jne	.L188
	movq	$6, -16(%rbp)
	jmp	.L184
.L188:
	movq	$3, -16(%rbp)
	jmp	.L184
.L192:
	nop
.L184:
	jmp	.L190
.L191:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	part2, .-part2
	.section	.rodata
.LC15:
	.string	"Couldn't identify %d\n"
	.text
	.globl	get
	.type	get, @function
get:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -24(%rbp)
	movl	%esi, -28(%rbp)
	movq	%rdx, -40(%rbp)
	movq	$3, -8(%rbp)
.L208:
	cmpq	$7, -8(%rbp)
	ja	.L210
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L196(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L196(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L196:
	.long	.L201-.L196
	.long	.L200-.L196
	.long	.L199-.L196
	.long	.L198-.L196
	.long	.L197-.L196
	.long	.L210-.L196
	.long	.L210-.L196
	.long	.L195-.L196
	.text
.L197:
	addl	$1, -12(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L202
.L200:
	movl	-12(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movl	16(%rax), %eax
	cmpl	%eax, -28(%rbp)
	jne	.L203
	movq	$2, -8(%rbp)
	jmp	.L202
.L203:
	movq	$4, -8(%rbp)
	jmp	.L202
.L198:
	movl	$0, -12(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L202
.L201:
	movq	stderr(%rip), %rax
	movl	-28(%rbp), %edx
	leaq	.LC15(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movl	$2, %edi
	call	exit@PLT
.L195:
	cmpl	$338, -12(%rbp)
	jg	.L205
	movq	$1, -8(%rbp)
	jmp	.L202
.L205:
	movq	$0, -8(%rbp)
	jmp	.L202
.L199:
	movl	-12(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdx
	movq	-40(%rbp), %rax
	leaq	(%rdx,%rax), %rsi
	movq	-24(%rbp), %rcx
	movq	(%rsi), %rax
	movq	8(%rsi), %rdx
	movq	%rax, (%rcx)
	movq	%rdx, 8(%rcx)
	movl	16(%rsi), %eax
	movl	%eax, 16(%rcx)
	jmp	.L209
.L210:
	nop
.L202:
	jmp	.L208
.L209:
	movq	-24(%rbp), %rax
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	get, .-get
	.globl	identify
	.type	identify, @function
identify:
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
	movq	$6, -8(%rbp)
.L231:
	cmpq	$9, -8(%rbp)
	ja	.L232
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L214(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L214(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L214:
	.long	.L222-.L214
	.long	.L221-.L214
	.long	.L220-.L214
	.long	.L219-.L214
	.long	.L218-.L214
	.long	.L232-.L214
	.long	.L217-.L214
	.long	.L216-.L214
	.long	.L215-.L214
	.long	.L213-.L214
	.text
.L218:
	cmpq	$1, -24(%rbp)
	jne	.L223
	movq	$3, -8(%rbp)
	jmp	.L225
.L223:
	movq	$1, -8(%rbp)
	jmp	.L225
.L215:
	cmpq	$0, -16(%rbp)
	jne	.L226
	movq	$7, -8(%rbp)
	jmp	.L225
.L226:
	movq	$0, -8(%rbp)
	jmp	.L225
.L221:
	movq	-40(%rbp), %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	subl	$96, %eax
	imull	$26, %eax, %edx
	movq	-40(%rbp), %rax
	addq	$1, %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	addl	%edx, %eax
	subl	$96, %eax
	jmp	.L228
.L219:
	movq	-40(%rbp), %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	subl	$96, %eax
	jmp	.L228
.L213:
	movl	$0, %eax
	jmp	.L228
.L217:
	cmpq	$0, -40(%rbp)
	jne	.L229
	movq	$9, -8(%rbp)
	jmp	.L225
.L229:
	movq	$2, -8(%rbp)
	jmp	.L225
.L222:
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -24(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L225
.L216:
	movl	$0, %eax
	jmp	.L228
.L220:
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -16(%rbp)
	movq	$8, -8(%rbp)
	jmp	.L225
.L232:
	nop
.L225:
	jmp	.L231
.L228:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	identify, .-identify
	.globl	getConnections
	.type	getConnections, @function
getConnections:
.LFB11:
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
	movq	%rdi, -88(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -24(%rbp)
	xorl	%eax, %eax
	movq	$1, -40(%rbp)
.L250:
	cmpq	$11, -40(%rbp)
	ja	.L253
	movq	-40(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L236(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L236(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L236:
	.long	.L243-.L236
	.long	.L242-.L236
	.long	.L253-.L236
	.long	.L241-.L236
	.long	.L240-.L236
	.long	.L239-.L236
	.long	.L238-.L236
	.long	.L237-.L236
	.long	.L253-.L236
	.long	.L253-.L236
	.long	.L253-.L236
	.long	.L235-.L236
	.text
.L240:
	movq	$0, -72(%rbp)
	movq	$0, -64(%rbp)
	movl	$20, %esi
	movl	$339, %edi
	call	calloc@PLT
	movq	%rax, -32(%rbp)
	movq	-32(%rbp), %rax
	movq	%rax, -56(%rbp)
	movl	$0, -80(%rbp)
	movq	$11, -40(%rbp)
	jmp	.L244
.L242:
	movq	$4, -40(%rbp)
	jmp	.L244
.L241:
	cmpq	$0, -48(%rbp)
	jle	.L245
	movq	$6, -40(%rbp)
	jmp	.L244
.L245:
	movq	$0, -40(%rbp)
	jmp	.L244
.L235:
	movq	-88(%rbp), %rdx
	leaq	-72(%rbp), %rcx
	leaq	-64(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	getline@PLT
	movq	%rax, -48(%rbp)
	movq	$3, -40(%rbp)
	jmp	.L244
.L238:
	movq	-64(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$10, %al
	je	.L247
	movq	$7, -40(%rbp)
	jmp	.L244
.L247:
	movq	$0, -40(%rbp)
	jmp	.L244
.L239:
	movq	-56(%rbp), %rax
	movq	-24(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L251
	jmp	.L252
.L243:
	movq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$5, -40(%rbp)
	jmp	.L244
.L237:
	movl	-80(%rbp), %eax
	movl	%eax, -76(%rbp)
	addl	$1, -80(%rbp)
	movq	-64(%rbp), %rdx
	movl	-76(%rbp), %eax
	movslq	%eax, %rcx
	movq	%rcx, %rax
	salq	$2, %rax
	addq	%rcx, %rax
	salq	$2, %rax
	movq	%rax, %rcx
	movq	-56(%rbp), %rax
	leaq	(%rcx,%rax), %rbx
	leaq	-128(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	parse
	movq	-128(%rbp), %rax
	movq	-120(%rbp), %rdx
	movq	%rax, (%rbx)
	movq	%rdx, 8(%rbx)
	movl	-112(%rbp), %eax
	movl	%eax, 16(%rbx)
	movq	$11, -40(%rbp)
	jmp	.L244
.L253:
	nop
.L244:
	jmp	.L250
.L252:
	call	__stack_chk_fail@PLT
.L251:
	movq	-8(%rbp), %rbx
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE11:
	.size	getConnections, .-getConnections
	.globl	part1
	.type	part1, @function
part1:
.LFB13:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movq	$2, -24(%rbp)
.L260:
	cmpq	$2, -24(%rbp)
	je	.L255
	cmpq	$2, -24(%rbp)
	ja	.L262
	cmpq	$0, -24(%rbp)
	je	.L257
	cmpq	$1, -24(%rbp)
	jne	.L262
	movzwl	-32(%rbp), %eax
	jmp	.L261
.L257:
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	getConnections
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -8(%rbp)
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	call	identify
	movl	%eax, -28(%rbp)
	movq	-8(%rbp), %rdx
	movl	-28(%rbp), %eax
	movq	%rdx, %rsi
	movl	%eax, %edi
	call	getValue
	movw	%ax, -30(%rbp)
	movzwl	-30(%rbp), %eax
	movw	%ax, -32(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$1, -24(%rbp)
	jmp	.L259
.L255:
	movq	$0, -24(%rbp)
	jmp	.L259
.L262:
	nop
.L259:
	jmp	.L260
.L261:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE13:
	.size	part1, .-part1
	.section	.rodata
.LC16:
	.string	"r"
.LC17:
	.string	"in7"
.LC18:
	.string	"Part1: %d\n"
.LC19:
	.string	"Part2: %d\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB14:
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
	movl	$0, -40(%rbp)
	jmp	.L264
.L265:
	movl	-40(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	known(%rip), %rax
	movl	$0, (%rdx,%rax)
	addl	$1, -40(%rbp)
.L264:
	cmpl	$728, -40(%rbp)
	jle	.L265
	nop
.L266:
	movl	$0, -36(%rbp)
	jmp	.L267
.L268:
	movl	-36(%rbp), %eax
	cltq
	leaq	(%rax,%rax), %rdx
	leaq	knownValues(%rip), %rax
	movw	$0, (%rdx,%rax)
	addl	$1, -36(%rbp)
.L267:
	cmpl	$728, -36(%rbp)
	jle	.L268
	nop
.L269:
	movq	$0, _TIG_IZ_tBjr_envp(%rip)
	nop
.L270:
	movq	$0, _TIG_IZ_tBjr_argv(%rip)
	nop
.L271:
	movl	$0, _TIG_IZ_tBjr_argc(%rip)
	nop
	nop
.L272:
.L273:
#APP
# 136 "timber-they_AdventOfCode15_7.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-tBjr--0
# 0 "" 2
#NO_APP
	movl	-52(%rbp), %eax
	movl	%eax, _TIG_IZ_tBjr_argc(%rip)
	movq	-64(%rbp), %rax
	movq	%rax, _TIG_IZ_tBjr_argv(%rip)
	movq	-72(%rbp), %rax
	movq	%rax, _TIG_IZ_tBjr_envp(%rip)
	nop
	movq	$0, -24(%rbp)
.L279:
	cmpq	$2, -24(%rbp)
	je	.L274
	cmpq	$2, -24(%rbp)
	ja	.L281
	cmpq	$0, -24(%rbp)
	je	.L276
	cmpq	$1, -24(%rbp)
	jne	.L281
	movl	$0, %eax
	jmp	.L280
.L276:
	movq	$2, -24(%rbp)
	jmp	.L278
.L274:
	leaq	.LC16(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC17(%rip), %rax
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
	leaq	.LC18(%rip), %rax
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
	leaq	.LC19(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$1, -24(%rbp)
	jmp	.L278
.L281:
	nop
.L278:
	jmp	.L279
.L280:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE14:
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
