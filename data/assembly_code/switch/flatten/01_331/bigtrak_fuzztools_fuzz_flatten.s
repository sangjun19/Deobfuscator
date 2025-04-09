	.file	"bigtrak_fuzztools_fuzz_flatten.c"
	.text
	.globl	flage
	.bss
	.align 4
	.type	flage, @object
	.size	flage, 4
flage:
	.zero	4
	.globl	flago
	.align 4
	.type	flago, @object
	.size	flago, 4
flago:
	.zero	4
	.globl	flag0
	.align 4
	.type	flag0, @object
	.size	flag0, 4
flag0:
	.zero	4
	.globl	outfile
	.align 8
	.type	outfile, @object
	.size	outfile, 8
outfile:
	.zero	8
	.globl	out
	.align 8
	.type	out, @object
	.size	out, 8
out:
	.zero	8
	.globl	flagm
	.align 4
	.type	flagm, @object
	.size	flagm, 4
flagm:
	.zero	4
	.globl	_TIG_IZ_k99G_argc
	.align 4
	.type	_TIG_IZ_k99G_argc, @object
	.size	_TIG_IZ_k99G_argc, 4
_TIG_IZ_k99G_argc:
	.zero	4
	.globl	_TIG_IZ_k99G_envp
	.align 8
	.type	_TIG_IZ_k99G_envp, @object
	.size	_TIG_IZ_k99G_envp, 8
_TIG_IZ_k99G_envp:
	.zero	8
	.globl	flags
	.align 4
	.type	flags, @object
	.size	flags, 4
flags:
	.zero	4
	.globl	seed
	.align 4
	.type	seed, @object
	.size	seed, 4
seed:
	.zero	4
	.globl	_TIG_IZ_k99G_argv
	.align 8
	.type	_TIG_IZ_k99G_argv, @object
	.size	_TIG_IZ_k99G_argv, 8
_TIG_IZ_k99G_argv:
	.zero	8
	.globl	flagl
	.align 4
	.type	flagl, @object
	.size	flagl, 4
flagl:
	.zero	4
	.local	progname
	.comm	progname,8,8
	.globl	flagd
	.align 4
	.type	flagd, @object
	.size	flagd, 4
flagd:
	.zero	4
	.globl	epilog
	.align 32
	.type	epilog, @object
	.size	epilog, 1024
epilog:
	.zero	1024
	.globl	flaga
	.align 4
	.type	flaga, @object
	.size	flaga, 4
flaga:
	.zero	4
	.globl	modulus
	.align 4
	.type	modulus, @object
	.size	modulus, 4
modulus:
	.zero	4
	.globl	length
	.align 4
	.type	length, @object
	.size	length, 4
length:
	.zero	4
	.globl	infile
	.align 8
	.type	infile, @object
	.size	infile, 8
infile:
	.zero	8
	.globl	in
	.align 8
	.type	in, @object
	.size	in, 8
in:
	.zero	8
	.globl	flagr
	.align 4
	.type	flagr, @object
	.size	flagr, 4
flagr:
	.zero	4
	.globl	flagx
	.align 4
	.type	flagx, @object
	.size	flagx, 4
flagx:
	.zero	4
	.globl	flagn
	.align 4
	.type	flagn, @object
	.size	flagn, 4
flagn:
	.zero	4
	.text
	.globl	putch
	.type	putch, @function
putch:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movl	%edi, -52(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$13, -16(%rbp)
.L34:
	cmpq	$16, -16(%rbp)
	ja	.L37
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
	.long	.L18-.L4
	.long	.L17-.L4
	.long	.L16-.L4
	.long	.L15-.L4
	.long	.L14-.L4
	.long	.L13-.L4
	.long	.L12-.L4
	.long	.L11-.L4
	.long	.L10-.L4
	.long	.L9-.L4
	.long	.L37-.L4
	.long	.L8-.L4
	.long	.L7-.L4
	.long	.L38-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L15:
	movq	outfile(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L5:
	cmpq	$1, -24(%rbp)
	je	.L21
	movq	$4, -16(%rbp)
	jmp	.L23
.L21:
	movq	$9, -16(%rbp)
	jmp	.L23
.L8:
	cmpq	$1, -32(%rbp)
	je	.L24
	movq	$10, -16(%rbp)
	jmp	.L23
.L24:
	movq	$7, -16(%rbp)
	jmp	.L23
.L11:
	movq	in(%rip), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$3, -16(%rbp)
	jmp	.L23
.L18:
	movl	$1, %edi
	call	exit@PLT
.L16:
	movl	flago(%rip), %eax
	testl	%eax, %eax
	je	.L26
	movq	$6, -16(%rbp)
	jmp	.L23
.L26:
	movq	$1, -16(%rbp)
	jmp	.L23
.L3:
	movl	-52(%rbp), %eax
	movb	%al, -37(%rbp)
	leaq	-37(%rbp), %rax
	movl	$1, %edx
	movq	%rax, %rsi
	movl	$1, %edi
	call	write@PLT
	movq	%rax, -32(%rbp)
	movq	$12, -16(%rbp)
	jmp	.L23
.L10:
	movl	flagd(%rip), %eax
	testl	%eax, %eax
	je	.L28
	movq	$0, -16(%rbp)
	jmp	.L23
.L28:
	movq	$14, -16(%rbp)
	jmp	.L23
.L7:
	movq	$16, -16(%rbp)
	jmp	.L23
.L13:
	movq	out(%rip), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$1, -16(%rbp)
	jmp	.L23
.L14:
	movl	flagr(%rip), %eax
	testl	%eax, %eax
	je	.L30
	movq	$8, -16(%rbp)
	jmp	.L23
.L30:
	movq	$3, -16(%rbp)
	jmp	.L23
.L9:
	movq	progname(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$5, -16(%rbp)
	jmp	.L23
.L19:
	movl	flagd(%rip), %eax
	movl	%eax, %edi
	call	usleep@PLT
	movq	$14, -16(%rbp)
	jmp	.L23
.L12:
	movl	flago(%rip), %eax
	testl	%eax, %eax
	je	.L32
	movq	$2, -16(%rbp)
	jmp	.L23
.L32:
	movq	$9, -16(%rbp)
	jmp	.L23
.L17:
	movq	out(%rip), %rax
	movq	%rax, %rdi
	call	fileno@PLT
	movl	%eax, -36(%rbp)
	leaq	-37(%rbp), %rcx
	movl	-36(%rbp), %eax
	movl	$1, %edx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	write@PLT
	movq	%rax, -24(%rbp)
	movq	$15, -16(%rbp)
	jmp	.L23
.L37:
	nop
.L23:
	jmp	.L34
.L38:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L36
	call	__stack_chk_fail@PLT
.L36:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	putch, .-putch
	.section	.rodata
	.align 8
.LC0:
	.string	"Usage: fuzz [-x] [-0] [-a] [-l [strlen]] [-p] [-o outfile]"
	.align 8
.LC1:
	.string	"            [-m [modulus] [-r infile] [-d delay] [-s seed]"
	.align 8
.LC2:
	.string	"            [-e \"epilog\"] [len]"
	.text
	.globl	usage
	.type	usage, @function
usage:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$1, -8(%rbp)
.L43:
	cmpq	$0, -8(%rbp)
	je	.L40
	cmpq	$1, -8(%rbp)
	jne	.L44
	movq	$0, -8(%rbp)
	jmp	.L42
.L40:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$1, %edi
	call	exit@PLT
.L44:
	nop
.L42:
	jmp	.L43
	.cfi_endproc
.LFE1:
	.size	usage, .-usage
	.section	.rodata
.LC3:
	.string	"%d\n"
.LC4:
	.string	"rb"
.LC5:
	.string	"wb"
	.text
	.globl	init
	.type	init, @function
init:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$16, -24(%rbp)
.L96:
	cmpq	$31, -24(%rbp)
	ja	.L99
	movq	-24(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L48(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L48(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L48:
	.long	.L71-.L48
	.long	.L70-.L48
	.long	.L69-.L48
	.long	.L68-.L48
	.long	.L67-.L48
	.long	.L66-.L48
	.long	.L99-.L48
	.long	.L65-.L48
	.long	.L64-.L48
	.long	.L63-.L48
	.long	.L62-.L48
	.long	.L99-.L48
	.long	.L99-.L48
	.long	.L99-.L48
	.long	.L61-.L48
	.long	.L60-.L48
	.long	.L59-.L48
	.long	.L58-.L48
	.long	.L57-.L48
	.long	.L56-.L48
	.long	.L99-.L48
	.long	.L99-.L48
	.long	.L55-.L48
	.long	.L99-.L48
	.long	.L54-.L48
	.long	.L53-.L48
	.long	.L100-.L48
	.long	.L51-.L48
	.long	.L99-.L48
	.long	.L50-.L48
	.long	.L49-.L48
	.long	.L47-.L48
	.text
.L57:
	movl	flagr(%rip), %eax
	testl	%eax, %eax
	je	.L72
	movq	$31, -24(%rbp)
	jmp	.L74
.L72:
	movq	$8, -24(%rbp)
	jmp	.L74
.L53:
	movl	seed(%rip), %eax
	movl	%eax, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	stdout(%rip), %rax
	movq	%rax, %rdi
	call	fflush@PLT
	movl	%eax, -44(%rbp)
	movq	$0, -24(%rbp)
	jmp	.L74
.L67:
	movl	seed(%rip), %eax
	movl	modulus(%rip), %ecx
	cltd
	idivl	%ecx
	movl	%edx, %eax
	movl	%eax, seed(%rip)
	movq	$9, -24(%rbp)
	jmp	.L74
.L49:
	cmpl	$-1, -40(%rbp)
	jne	.L75
	movq	$5, -24(%rbp)
	jmp	.L74
.L75:
	movq	$26, -24(%rbp)
	jmp	.L74
.L61:
	movq	out(%rip), %rax
	testq	%rax, %rax
	jne	.L77
	movq	$1, -24(%rbp)
	jmp	.L74
.L77:
	movq	$18, -24(%rbp)
	jmp	.L74
.L60:
	call	rand@PLT
	movl	%eax, -36(%rbp)
	movl	-36(%rbp), %edx
	movslq	%edx, %rax
	imulq	$351843721, %rax, %rax
	shrq	$32, %rax
	sarl	$13, %eax
	movl	%edx, %ecx
	sarl	$31, %ecx
	subl	%ecx, %eax
	imull	$100000, %eax, %ecx
	movl	%edx, %eax
	subl	%ecx, %eax
	movl	%eax, length(%rip)
	movq	$27, -24(%rbp)
	jmp	.L74
.L47:
	movq	infile(%rip), %rax
	leaq	.LC4(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, in(%rip)
	movq	$29, -24(%rbp)
	jmp	.L74
.L64:
	movl	flagx(%rip), %eax
	testl	%eax, %eax
	je	.L79
	movq	$25, -24(%rbp)
	jmp	.L74
.L79:
	movq	$26, -24(%rbp)
	jmp	.L74
.L70:
	movq	outfile(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L68:
	movq	outfile(%rip), %rax
	leaq	.LC5(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, out(%rip)
	movq	$14, -24(%rbp)
	jmp	.L74
.L59:
	movl	flags(%rip), %eax
	testl	%eax, %eax
	jne	.L81
	movq	$2, -24(%rbp)
	jmp	.L74
.L81:
	movq	$22, -24(%rbp)
	jmp	.L74
.L54:
	movl	flago(%rip), %eax
	testl	%eax, %eax
	je	.L83
	movq	$19, -24(%rbp)
	jmp	.L74
.L83:
	movq	$26, -24(%rbp)
	jmp	.L74
.L63:
	movl	seed(%rip), %eax
	movl	%eax, %edi
	call	srand@PLT
	movq	$17, -24(%rbp)
	jmp	.L74
.L56:
	movl	seed(%rip), %edx
	movq	out(%rip), %rax
	leaq	.LC3(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	out(%rip), %rax
	movq	%rax, %rdi
	call	fflush@PLT
	movl	%eax, -40(%rbp)
	movq	$30, -24(%rbp)
	jmp	.L74
.L58:
	movl	flagn(%rip), %eax
	testl	%eax, %eax
	jne	.L86
	movq	$15, -24(%rbp)
	jmp	.L74
.L86:
	movq	$27, -24(%rbp)
	jmp	.L74
.L51:
	movl	flago(%rip), %eax
	testl	%eax, %eax
	je	.L88
	movq	$3, -24(%rbp)
	jmp	.L74
.L88:
	movq	$18, -24(%rbp)
	jmp	.L74
.L55:
	movl	flagm(%rip), %eax
	testl	%eax, %eax
	je	.L90
	movq	$4, -24(%rbp)
	jmp	.L74
.L90:
	movq	$9, -24(%rbp)
	jmp	.L74
.L66:
	movq	outfile(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L62:
	movq	progname(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L71:
	cmpl	$-1, -44(%rbp)
	jne	.L92
	movq	$10, -24(%rbp)
	jmp	.L74
.L92:
	movq	$24, -24(%rbp)
	jmp	.L74
.L65:
	movq	infile(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L50:
	movq	in(%rip), %rax
	testq	%rax, %rax
	jne	.L94
	movq	$7, -24(%rbp)
	jmp	.L74
.L94:
	movq	$26, -24(%rbp)
	jmp	.L74
.L69:
	leaq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	time@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movl	%eax, seed(%rip)
	movq	$22, -24(%rbp)
	jmp	.L74
.L99:
	nop
.L74:
	jmp	.L96
.L100:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L98
	call	__stack_chk_fail@PLT
.L98:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	init, .-init
	.globl	fuzz
	.type	fuzz, @function
fuzz:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$11, -8(%rbp)
.L124:
	cmpq	$11, -8(%rbp)
	ja	.L125
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
	.long	.L113-.L104
	.long	.L112-.L104
	.long	.L111-.L104
	.long	.L110-.L104
	.long	.L109-.L104
	.long	.L108-.L104
	.long	.L107-.L104
	.long	.L125-.L104
	.long	.L126-.L104
	.long	.L105-.L104
	.long	.L125-.L104
	.long	.L103-.L104
	.text
.L109:
	movl	-12(%rbp), %edx
	movl	-16(%rbp), %eax
	movl	%edx, %esi
	movl	%eax, %edi
	call	fuzzchar
	movq	$8, -8(%rbp)
	jmp	.L114
.L112:
	movl	$0, -12(%rbp)
	movl	$256, -16(%rbp)
	movq	$9, -8(%rbp)
	jmp	.L114
.L110:
	movl	-12(%rbp), %edx
	movl	-16(%rbp), %eax
	movl	%edx, %esi
	movl	%eax, %edi
	call	fuzzstr
	movq	$8, -8(%rbp)
	jmp	.L114
.L103:
	movq	$2, -8(%rbp)
	jmp	.L114
.L105:
	movl	flaga(%rip), %eax
	testl	%eax, %eax
	jne	.L116
	movq	$0, -8(%rbp)
	jmp	.L114
.L116:
	movq	$5, -8(%rbp)
	jmp	.L114
.L107:
	movl	flag0(%rip), %eax
	testl	%eax, %eax
	je	.L118
	movq	$1, -8(%rbp)
	jmp	.L114
.L118:
	movq	$9, -8(%rbp)
	jmp	.L114
.L108:
	movl	flagl(%rip), %eax
	testl	%eax, %eax
	je	.L120
	movq	$3, -8(%rbp)
	jmp	.L114
.L120:
	movq	$4, -8(%rbp)
	jmp	.L114
.L113:
	movl	$32, -12(%rbp)
	movl	flag0(%rip), %eax
	testl	%eax, %eax
	je	.L122
	movl	$96, %eax
	jmp	.L123
.L122:
	movl	$95, %eax
.L123:
	movl	%eax, -16(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L114
.L111:
	movl	$1, -12(%rbp)
	movl	$255, -16(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L114
.L125:
	nop
.L114:
	jmp	.L124
.L126:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	fuzz, .-fuzz
	.globl	replay
	.type	replay, @function
replay:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$3, -8(%rbp)
.L139:
	cmpq	$5, -8(%rbp)
	ja	.L140
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L130(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L130(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L130:
	.long	.L134-.L130
	.long	.L140-.L130
	.long	.L133-.L130
	.long	.L132-.L130
	.long	.L131-.L130
	.long	.L141-.L130
	.text
.L131:
	movl	-12(%rbp), %eax
	movl	%eax, %edi
	call	putch
	movq	$0, -8(%rbp)
	jmp	.L135
.L132:
	movq	$0, -8(%rbp)
	jmp	.L135
.L134:
	movq	in(%rip), %rax
	movq	%rax, %rdi
	call	getc@PLT
	movl	%eax, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L135
.L133:
	cmpl	$-1, -12(%rbp)
	je	.L137
	movq	$4, -8(%rbp)
	jmp	.L135
.L137:
	movq	$5, -8(%rbp)
	jmp	.L135
.L140:
	nop
.L135:
	jmp	.L139
.L141:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	replay, .-replay
	.section	.rodata
.LC6:
	.string	"%3d"
.LC7:
	.string	"%2x"
	.text
	.globl	myputs
	.type	myputs, @function
myputs:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movq	%rdi, -56(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$15, -16(%rbp)
.L192:
	cmpq	$38, -16(%rbp)
	ja	.L195
	movq	-16(%rbp), %rax
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
	.long	.L195-.L145
	.long	.L169-.L145
	.long	.L168-.L145
	.long	.L167-.L145
	.long	.L166-.L145
	.long	.L165-.L145
	.long	.L195-.L145
	.long	.L164-.L145
	.long	.L195-.L145
	.long	.L163-.L145
	.long	.L162-.L145
	.long	.L161-.L145
	.long	.L195-.L145
	.long	.L160-.L145
	.long	.L159-.L145
	.long	.L158-.L145
	.long	.L157-.L145
	.long	.L195-.L145
	.long	.L195-.L145
	.long	.L156-.L145
	.long	.L155-.L145
	.long	.L195-.L145
	.long	.L154-.L145
	.long	.L153-.L145
	.long	.L195-.L145
	.long	.L195-.L145
	.long	.L195-.L145
	.long	.L196-.L145
	.long	.L151-.L145
	.long	.L150-.L145
	.long	.L149-.L145
	.long	.L148-.L145
	.long	.L195-.L145
	.long	.L195-.L145
	.long	.L147-.L145
	.long	.L195-.L145
	.long	.L146-.L145
	.long	.L195-.L145
	.long	.L144-.L145
	.text
.L166:
	leaq	-48(%rbp), %rdx
	movq	-56(%rbp), %rax
	leaq	.LC6(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_sscanf@PLT
	movl	-48(%rbp), %eax
	movl	%eax, %edi
	call	oct2dec
	movl	%eax, -44(%rbp)
	movl	-44(%rbp), %eax
	movl	%eax, %edi
	call	putch
	movl	$0, -48(%rbp)
	movq	$34, -16(%rbp)
	jmp	.L170
.L149:
	addq	$1, -56(%rbp)
	movq	$29, -16(%rbp)
	jmp	.L170
.L159:
	movq	-56(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$92, %al
	jne	.L171
	movq	$30, -16(%rbp)
	jmp	.L170
.L171:
	movq	$28, -16(%rbp)
	jmp	.L170
.L158:
	movq	$13, -16(%rbp)
	jmp	.L170
.L148:
	subq	$1, -56(%rbp)
	movq	$22, -16(%rbp)
	jmp	.L170
.L169:
	movl	$8, %edi
	call	putch
	movq	$22, -16(%rbp)
	jmp	.L170
.L153:
	movl	$12, %edi
	call	putch
	movq	$22, -16(%rbp)
	jmp	.L170
.L167:
	movq	-56(%rbp), %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	movl	%eax, %edi
	call	putch
	movq	$22, -16(%rbp)
	jmp	.L170
.L157:
	cmpq	$0, -24(%rbp)
	je	.L173
	movq	$14, -16(%rbp)
	jmp	.L170
.L173:
	movq	$27, -16(%rbp)
	jmp	.L170
.L146:
	movl	$13, %edi
	call	putch
	movq	$22, -16(%rbp)
	jmp	.L170
.L161:
	movl	$10, %edi
	call	putch
	movq	$22, -16(%rbp)
	jmp	.L170
.L163:
	movq	-40(%rbp), %rax
	movq	(%rax), %rdx
	movq	-56(%rbp), %rax
	movzbl	(%rax), %eax
	movsbq	%al, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$2048, %eax
	testl	%eax, %eax
	je	.L175
	movq	$10, -16(%rbp)
	jmp	.L170
.L175:
	movq	$31, -16(%rbp)
	jmp	.L170
.L160:
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -24(%rbp)
	movq	$16, -16(%rbp)
	jmp	.L170
.L156:
	call	__ctype_b_loc@PLT
	movq	%rax, -40(%rbp)
	movq	$9, -16(%rbp)
	jmp	.L170
.L144:
	movl	$13, %edi
	call	putch
	movq	$22, -16(%rbp)
	jmp	.L170
.L147:
	movl	-48(%rbp), %eax
	cmpl	$2, %eax
	jg	.L178
	movq	$19, -16(%rbp)
	jmp	.L170
.L178:
	movq	$31, -16(%rbp)
	jmp	.L170
.L154:
	addq	$1, -56(%rbp)
	movq	$13, -16(%rbp)
	jmp	.L170
.L151:
	movq	-56(%rbp), %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	movl	%eax, %edi
	call	putch
	movq	$22, -16(%rbp)
	jmp	.L170
.L165:
	movq	-32(%rbp), %rax
	movq	(%rax), %rdx
	movq	-56(%rbp), %rax
	movzbl	(%rax), %eax
	movsbq	%al, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$2048, %eax
	testl	%eax, %eax
	je	.L180
	movq	$4, -16(%rbp)
	jmp	.L170
.L180:
	movq	$3, -16(%rbp)
	jmp	.L170
.L162:
	addq	$1, -56(%rbp)
	movl	-48(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -48(%rbp)
	movq	$34, -16(%rbp)
	jmp	.L170
.L164:
	call	__ctype_b_loc@PLT
	movq	%rax, -32(%rbp)
	movq	$5, -16(%rbp)
	jmp	.L170
.L150:
	movq	-56(%rbp), %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	subl	$98, %eax
	cmpl	$22, %eax
	ja	.L182
	movl	%eax, %eax
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
	.long	.L190-.L184
	.long	.L182-.L184
	.long	.L182-.L184
	.long	.L182-.L184
	.long	.L189-.L184
	.long	.L182-.L184
	.long	.L182-.L184
	.long	.L182-.L184
	.long	.L182-.L184
	.long	.L182-.L184
	.long	.L182-.L184
	.long	.L182-.L184
	.long	.L188-.L184
	.long	.L182-.L184
	.long	.L182-.L184
	.long	.L182-.L184
	.long	.L187-.L184
	.long	.L182-.L184
	.long	.L186-.L184
	.long	.L182-.L184
	.long	.L185-.L184
	.long	.L182-.L184
	.long	.L183-.L184
	.text
.L183:
	movq	$2, -16(%rbp)
	jmp	.L191
.L185:
	movq	$20, -16(%rbp)
	jmp	.L191
.L186:
	movq	$36, -16(%rbp)
	jmp	.L191
.L187:
	movq	$38, -16(%rbp)
	jmp	.L191
.L188:
	movq	$11, -16(%rbp)
	jmp	.L191
.L189:
	movq	$23, -16(%rbp)
	jmp	.L191
.L190:
	movq	$1, -16(%rbp)
	jmp	.L191
.L182:
	movq	$7, -16(%rbp)
	nop
.L191:
	jmp	.L170
.L168:
	addq	$1, -56(%rbp)
	leaq	-48(%rbp), %rdx
	movq	-56(%rbp), %rax
	leaq	.LC7(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_sscanf@PLT
	movl	-48(%rbp), %eax
	movl	%eax, %edi
	call	putch
	addq	$1, -56(%rbp)
	movq	$22, -16(%rbp)
	jmp	.L170
.L155:
	movl	$11, %edi
	call	putch
	movq	$22, -16(%rbp)
	jmp	.L170
.L195:
	nop
.L170:
	jmp	.L192
.L196:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L194
	call	__stack_chk_fail@PLT
.L194:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	myputs, .-myputs
	.globl	fuzzstr
	.type	fuzzstr, @function
fuzzstr:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movl	%edi, -36(%rbp)
	movl	%esi, -40(%rbp)
	movq	$4, -8(%rbp)
.L224:
	movq	-8(%rbp), %rax
	subq	$3, %rax
	cmpq	$16, %rax
	ja	.L225
	leaq	0(,%rax,4), %rdx
	leaq	.L200(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L200(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L200:
	.long	.L211-.L200
	.long	.L210-.L200
	.long	.L209-.L200
	.long	.L225-.L200
	.long	.L225-.L200
	.long	.L226-.L200
	.long	.L207-.L200
	.long	.L206-.L200
	.long	.L225-.L200
	.long	.L205-.L200
	.long	.L204-.L200
	.long	.L203-.L200
	.long	.L202-.L200
	.long	.L225-.L200
	.long	.L225-.L200
	.long	.L201-.L200
	.long	.L199-.L200
	.text
.L201:
	movl	flaga(%rip), %eax
	testl	%eax, %eax
	jne	.L212
	movq	$9, -8(%rbp)
	jmp	.L214
.L212:
	movq	$12, -8(%rbp)
	jmp	.L214
.L210:
	movl	$0, -32(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L214
.L203:
	movl	flag0(%rip), %eax
	testl	%eax, %eax
	je	.L215
	movq	$18, -8(%rbp)
	jmp	.L214
.L215:
	movq	$12, -8(%rbp)
	jmp	.L214
.L202:
	movl	$0, -20(%rbp)
	movq	$12, -8(%rbp)
	jmp	.L214
.L205:
	movl	-20(%rbp), %eax
	movl	%eax, %edi
	call	putch
	addl	$1, -28(%rbp)
	movq	$19, -8(%rbp)
	jmp	.L214
.L211:
	movl	length(%rip), %eax
	cmpl	%eax, -32(%rbp)
	jge	.L218
	movq	$5, -8(%rbp)
	jmp	.L214
.L218:
	movq	$8, -8(%rbp)
	jmp	.L214
.L207:
	cmpl	$127, -20(%rbp)
	jne	.L220
	movq	$15, -8(%rbp)
	jmp	.L214
.L220:
	movq	$12, -8(%rbp)
	jmp	.L214
.L204:
	movl	$10, %edi
	call	putch
	addl	$1, -32(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L214
.L199:
	movl	-28(%rbp), %eax
	cmpl	-24(%rbp), %eax
	jge	.L222
	movq	$10, -8(%rbp)
	jmp	.L214
.L222:
	movq	$13, -8(%rbp)
	jmp	.L214
.L209:
	call	rand@PLT
	movl	%eax, -12(%rbp)
	movl	flagl(%rip), %ecx
	movl	-12(%rbp), %eax
	cltd
	idivl	%ecx
	movl	%edx, -24(%rbp)
	movl	$0, -28(%rbp)
	movq	$19, -8(%rbp)
	jmp	.L214
.L206:
	call	rand@PLT
	movl	%eax, -16(%rbp)
	movl	-16(%rbp), %eax
	cltd
	idivl	-36(%rbp)
	movl	-40(%rbp), %eax
	addl	%edx, %eax
	movl	%eax, -20(%rbp)
	movq	$14, -8(%rbp)
	jmp	.L214
.L225:
	nop
.L214:
	jmp	.L224
.L226:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	fuzzstr, .-fuzzstr
	.section	.rodata
.LC8:
	.string	"%d"
	.text
	.globl	oct2dec
	.type	oct2dec, @function
oct2dec:
.LFB12:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movl	%edi, -52(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$5, -24(%rbp)
.L240:
	cmpq	$8, -24(%rbp)
	ja	.L243
	movq	-24(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L230(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L230(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L230:
	.long	.L235-.L230
	.long	.L243-.L230
	.long	.L243-.L230
	.long	.L234-.L230
	.long	.L243-.L230
	.long	.L233-.L230
	.long	.L232-.L230
	.long	.L231-.L230
	.long	.L229-.L230
	.text
.L229:
	movl	-36(%rbp), %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L241
	jmp	.L242
.L234:
	movl	-36(%rbp), %eax
	leal	0(,%rax,8), %edx
	movl	-52(%rbp), %eax
	cltq
	movzbl	-16(%rbp,%rax), %eax
	movsbl	%al, %eax
	subl	$48, %eax
	addl	%edx, %eax
	movl	%eax, -36(%rbp)
	addl	$1, -52(%rbp)
	movq	$7, -24(%rbp)
	jmp	.L237
.L232:
	movl	$0, -36(%rbp)
	movl	-52(%rbp), %edx
	leaq	-16(%rbp), %rax
	leaq	.LC8(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	sprintf@PLT
	movl	$0, -52(%rbp)
	movq	$7, -24(%rbp)
	jmp	.L237
.L233:
	movq	$6, -24(%rbp)
	jmp	.L237
.L235:
	movl	-52(%rbp), %eax
	cltq
	cmpq	%rax, -32(%rbp)
	jbe	.L238
	movq	$3, -24(%rbp)
	jmp	.L237
.L238:
	movq	$8, -24(%rbp)
	jmp	.L237
.L231:
	leaq	-16(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -32(%rbp)
	movq	$0, -24(%rbp)
	jmp	.L237
.L243:
	nop
.L237:
	jmp	.L240
.L242:
	call	__stack_chk_fail@PLT
.L241:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE12:
	.size	oct2dec, .-oct2dec
	.globl	fuzzchar
	.type	fuzzchar, @function
fuzzchar:
.LFB13:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movl	%edi, -36(%rbp)
	movl	%esi, -40(%rbp)
	movq	$11, -8(%rbp)
.L266:
	cmpq	$11, -8(%rbp)
	ja	.L267
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L247(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L247(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L247:
	.long	.L255-.L247
	.long	.L254-.L247
	.long	.L267-.L247
	.long	.L253-.L247
	.long	.L268-.L247
	.long	.L251-.L247
	.long	.L267-.L247
	.long	.L250-.L247
	.long	.L267-.L247
	.long	.L249-.L247
	.long	.L248-.L247
	.long	.L246-.L247
	.text
.L254:
	movl	flag0(%rip), %eax
	testl	%eax, %eax
	je	.L257
	movq	$3, -8(%rbp)
	jmp	.L259
.L257:
	movq	$5, -8(%rbp)
	jmp	.L259
.L253:
	movl	flaga(%rip), %eax
	testl	%eax, %eax
	jne	.L260
	movq	$0, -8(%rbp)
	jmp	.L259
.L260:
	movq	$5, -8(%rbp)
	jmp	.L259
.L246:
	movl	$0, -20(%rbp)
	movq	$9, -8(%rbp)
	jmp	.L259
.L249:
	movl	length(%rip), %eax
	cmpl	%eax, -20(%rbp)
	jge	.L262
	movq	$10, -8(%rbp)
	jmp	.L259
.L262:
	movq	$4, -8(%rbp)
	jmp	.L259
.L251:
	movl	-16(%rbp), %eax
	movl	%eax, %edi
	call	putch
	addl	$1, -20(%rbp)
	movq	$9, -8(%rbp)
	jmp	.L259
.L248:
	call	rand@PLT
	movl	%eax, -12(%rbp)
	movl	-12(%rbp), %eax
	cltd
	idivl	-36(%rbp)
	movl	-40(%rbp), %eax
	addl	%edx, %eax
	movl	%eax, -16(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L259
.L255:
	cmpl	$127, -16(%rbp)
	jne	.L264
	movq	$7, -8(%rbp)
	jmp	.L259
.L264:
	movq	$5, -8(%rbp)
	jmp	.L259
.L250:
	movl	$0, -16(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L259
.L267:
	nop
.L259:
	jmp	.L266
.L268:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE13:
	.size	fuzzchar, .-fuzzchar
	.section	.rodata
.LC9:
	.string	"fuzz"
.LC11:
	.string	"%s"
.LC12:
	.string	"%f"
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
	subq	$96, %rsp
	movl	%edi, -68(%rbp)
	movq	%rsi, -80(%rbp)
	movq	%rdx, -88(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, out(%rip)
	nop
.L270:
	movq	$0, in(%rip)
	nop
.L271:
	movq	$0, outfile(%rip)
	nop
.L272:
	movq	$0, infile(%rip)
	nop
.L273:
	movl	$0, -48(%rbp)
	jmp	.L274
.L275:
	movl	-48(%rbp), %eax
	cltq
	leaq	epilog(%rip), %rdx
	movb	$0, (%rax,%rdx)
	addl	$1, -48(%rbp)
.L274:
	cmpl	$1023, -48(%rbp)
	jle	.L275
	nop
.L276:
	movl	$0, modulus(%rip)
	nop
.L277:
	movl	$0, flagm(%rip)
	nop
.L278:
	movl	$0, length(%rip)
	nop
.L279:
	movl	$0, flagr(%rip)
	nop
.L280:
	movl	$0, flago(%rip)
	nop
.L281:
	movl	$0, flagx(%rip)
	nop
.L282:
	movl	$0, flagn(%rip)
	nop
.L283:
	movl	$0, seed(%rip)
	nop
.L284:
	movl	$0, flage(%rip)
	nop
.L285:
	movl	$0, flags(%rip)
	nop
.L286:
	movl	$0, flagl(%rip)
	nop
.L287:
	movl	$0, flagd(%rip)
	nop
.L288:
	movl	$1, flaga(%rip)
	nop
.L289:
	movl	$0, flag0(%rip)
	nop
.L290:
	leaq	.LC9(%rip), %rax
	movq	%rax, progname(%rip)
	nop
.L291:
	movq	$0, _TIG_IZ_k99G_envp(%rip)
	nop
.L292:
	movq	$0, _TIG_IZ_k99G_argv(%rip)
	nop
.L293:
	movl	$0, _TIG_IZ_k99G_argc(%rip)
	nop
	nop
.L294:
.L295:
#APP
# 546 "bigtrak_fuzztools_fuzz.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-k99G--0
# 0 "" 2
#NO_APP
	movl	-68(%rbp), %eax
	movl	%eax, _TIG_IZ_k99G_argc(%rip)
	movq	-80(%rbp), %rax
	movq	%rax, _TIG_IZ_k99G_argv(%rip)
	movq	-88(%rbp), %rax
	movq	%rax, _TIG_IZ_k99G_envp(%rip)
	nop
	movq	$13, -16(%rbp)
.L398:
	cmpq	$72, -16(%rbp)
	ja	.L401
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L298(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L298(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L298:
	.long	.L401-.L298
	.long	.L349-.L298
	.long	.L348-.L298
	.long	.L347-.L298
	.long	.L401-.L298
	.long	.L401-.L298
	.long	.L346-.L298
	.long	.L345-.L298
	.long	.L344-.L298
	.long	.L343-.L298
	.long	.L401-.L298
	.long	.L342-.L298
	.long	.L401-.L298
	.long	.L341-.L298
	.long	.L340-.L298
	.long	.L339-.L298
	.long	.L338-.L298
	.long	.L401-.L298
	.long	.L401-.L298
	.long	.L337-.L298
	.long	.L336-.L298
	.long	.L401-.L298
	.long	.L335-.L298
	.long	.L401-.L298
	.long	.L334-.L298
	.long	.L333-.L298
	.long	.L332-.L298
	.long	.L331-.L298
	.long	.L330-.L298
	.long	.L329-.L298
	.long	.L328-.L298
	.long	.L401-.L298
	.long	.L401-.L298
	.long	.L401-.L298
	.long	.L327-.L298
	.long	.L326-.L298
	.long	.L325-.L298
	.long	.L324-.L298
	.long	.L323-.L298
	.long	.L322-.L298
	.long	.L401-.L298
	.long	.L321-.L298
	.long	.L401-.L298
	.long	.L320-.L298
	.long	.L319-.L298
	.long	.L401-.L298
	.long	.L401-.L298
	.long	.L318-.L298
	.long	.L401-.L298
	.long	.L401-.L298
	.long	.L317-.L298
	.long	.L316-.L298
	.long	.L315-.L298
	.long	.L314-.L298
	.long	.L401-.L298
	.long	.L313-.L298
	.long	.L312-.L298
	.long	.L311-.L298
	.long	.L310-.L298
	.long	.L401-.L298
	.long	.L309-.L298
	.long	.L308-.L298
	.long	.L307-.L298
	.long	.L306-.L298
	.long	.L305-.L298
	.long	.L304-.L298
	.long	.L401-.L298
	.long	.L303-.L298
	.long	.L302-.L298
	.long	.L301-.L298
	.long	.L300-.L298
	.long	.L299-.L298
	.long	.L297-.L298
	.text
.L317:
	movl	$1, flagn(%rip)
	movq	$24, -16(%rbp)
	jmp	.L350
.L333:
	movq	out(%rip), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movl	%eax, -24(%rbp)
	movq	$19, -16(%rbp)
	jmp	.L350
.L315:
	cmpl	$1, -32(%rbp)
	je	.L351
	movq	$37, -16(%rbp)
	jmp	.L350
.L351:
	movq	$24, -16(%rbp)
	jmp	.L350
.L328:
	movq	-80(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	movzbl	(%rax), %eax
	cmpb	$45, %al
	je	.L353
	movq	$11, -16(%rbp)
	jmp	.L350
.L353:
	movq	$24, -16(%rbp)
	jmp	.L350
.L307:
	movl	$1, flag0(%rip)
	movq	$24, -16(%rbp)
	jmp	.L350
.L340:
	call	fuzz
	movq	$61, -16(%rbp)
	jmp	.L350
.L339:
	cmpl	$-1, -20(%rbp)
	jne	.L355
	movq	$55, -16(%rbp)
	jmp	.L350
.L355:
	movq	$36, -16(%rbp)
	jmp	.L350
.L312:
	call	init
	movq	$43, -16(%rbp)
	jmp	.L350
.L301:
	call	usage
	movq	$24, -16(%rbp)
	jmp	.L350
.L344:
	cmpl	$1, -40(%rbp)
	je	.L357
	movq	$3, -16(%rbp)
	jmp	.L350
.L357:
	movq	$26, -16(%rbp)
	jmp	.L350
.L349:
	movl	$255, flagl(%rip)
	movq	$72, -16(%rbp)
	jmp	.L350
.L300:
	movq	-80(%rbp), %rax
	movq	(%rax), %rax
	movzbl	(%rax), %eax
	cmpb	$45, %al
	je	.L359
	movq	$2, -16(%rbp)
	jmp	.L350
.L359:
	movq	$7, -16(%rbp)
	jmp	.L350
.L347:
	call	usage
	movq	$26, -16(%rbp)
	jmp	.L350
.L338:
	call	usage
	movq	$63, -16(%rbp)
	jmp	.L350
.L334:
	addq	$8, -80(%rbp)
	movq	$60, -16(%rbp)
	jmp	.L350
.L325:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L399
	jmp	.L400
.L311:
	movl	$0, flaga(%rip)
	movq	$24, -16(%rbp)
	jmp	.L350
.L302:
	cmpl	$1, -36(%rbp)
	je	.L362
	movq	$34, -16(%rbp)
	jmp	.L350
.L362:
	movq	$51, -16(%rbp)
	jmp	.L350
.L332:
	movss	-52(%rbp), %xmm0
	pxor	%xmm1, %xmm1
	cvtss2sd	%xmm0, %xmm1
	movsd	.LC10(%rip), %xmm0
	mulsd	%xmm1, %xmm0
	cvttsd2siq	%xmm0, %rax
	movl	%eax, flagd(%rip)
	movq	$24, -16(%rbp)
	jmp	.L350
.L342:
	addq	$8, -80(%rbp)
	movq	-80(%rbp), %rax
	movq	(%rax), %rax
	leaq	flagl(%rip), %rdx
	leaq	.LC8(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_sscanf@PLT
	movl	%eax, -36(%rbp)
	movq	$68, -16(%rbp)
	jmp	.L350
.L343:
	cmpl	$1, -28(%rbp)
	je	.L364
	movq	$69, -16(%rbp)
	jmp	.L350
.L364:
	movq	$24, -16(%rbp)
	jmp	.L350
.L341:
	movq	$24, -16(%rbp)
	jmp	.L350
.L306:
	movq	-80(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, %rdx
	leaq	.LC11(%rip), %rax
	movq	%rax, %rsi
	leaq	epilog(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	sprintf@PLT
	movq	$24, -16(%rbp)
	jmp	.L350
.L316:
	movl	flagl(%rip), %eax
	testl	%eax, %eax
	jg	.L366
	movq	$22, -16(%rbp)
	jmp	.L350
.L366:
	movq	$24, -16(%rbp)
	jmp	.L350
.L337:
	cmpl	$-1, -24(%rbp)
	jne	.L368
	movq	$64, -16(%rbp)
	jmp	.L350
.L368:
	movq	$28, -16(%rbp)
	jmp	.L350
.L303:
	addq	$8, -80(%rbp)
	movl	$1, flags(%rip)
	movq	-80(%rbp), %rax
	movq	(%rax), %rax
	leaq	seed(%rip), %rdx
	leaq	.LC8(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_sscanf@PLT
	movl	%eax, -32(%rbp)
	movq	$52, -16(%rbp)
	jmp	.L350
.L313:
	movq	infile(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L309:
	movq	-80(%rbp), %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	je	.L370
	movq	$70, -16(%rbp)
	jmp	.L350
.L370:
	movq	$56, -16(%rbp)
	jmp	.L350
.L346:
	movl	$1, flagx(%rip)
	movq	$24, -16(%rbp)
	jmp	.L350
.L331:
	movq	-80(%rbp), %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	jne	.L372
	movq	$16, -16(%rbp)
	jmp	.L350
.L372:
	movq	$63, -16(%rbp)
	jmp	.L350
.L323:
	call	usage
	movq	$24, -16(%rbp)
	jmp	.L350
.L308:
	leaq	epilog(%rip), %rax
	movq	%rax, %rdi
	call	myputs
	movq	$58, -16(%rbp)
	jmp	.L350
.L310:
	movl	flago(%rip), %eax
	testl	%eax, %eax
	je	.L374
	movq	$25, -16(%rbp)
	jmp	.L350
.L374:
	movq	$28, -16(%rbp)
	jmp	.L350
.L327:
	call	usage
	movq	$24, -16(%rbp)
	jmp	.L350
.L299:
	addq	$8, -80(%rbp)
	movq	-80(%rbp), %rax
	movq	(%rax), %rax
	leaq	-52(%rbp), %rdx
	leaq	.LC12(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_sscanf@PLT
	movl	%eax, -40(%rbp)
	movq	$8, -16(%rbp)
	jmp	.L350
.L335:
	call	usage
	movq	$24, -16(%rbp)
	jmp	.L350
.L330:
	movl	flagr(%rip), %eax
	testl	%eax, %eax
	je	.L376
	movq	$20, -16(%rbp)
	jmp	.L350
.L376:
	movq	$36, -16(%rbp)
	jmp	.L350
.L314:
	addq	$8, -80(%rbp)
	movl	$1, flage(%rip)
	movq	$27, -16(%rbp)
	jmp	.L350
.L304:
	movl	$1, flagr(%rip)
	addq	$8, -80(%rbp)
	movq	-80(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, infile(%rip)
	movq	$24, -16(%rbp)
	jmp	.L350
.L318:
	movl	$1, flago(%rip)
	addq	$8, -80(%rbp)
	movq	-80(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, outfile(%rip)
	movq	$24, -16(%rbp)
	jmp	.L350
.L319:
	call	usage
	movq	$50, -16(%rbp)
	jmp	.L350
.L297:
	movq	-80(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	je	.L378
	movq	$30, -16(%rbp)
	jmp	.L350
.L378:
	movq	$24, -16(%rbp)
	jmp	.L350
.L324:
	call	usage
	movq	$24, -16(%rbp)
	jmp	.L350
.L305:
	movq	outfile(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L321:
	movl	$1, flaga(%rip)
	movq	$24, -16(%rbp)
	jmp	.L350
.L322:
	addq	$8, -80(%rbp)
	movl	$1, flagm(%rip)
	movq	-80(%rbp), %rax
	movq	(%rax), %rax
	leaq	modulus(%rip), %rdx
	leaq	.LC8(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_sscanf@PLT
	movl	%eax, -28(%rbp)
	movq	$9, -16(%rbp)
	jmp	.L350
.L345:
	movq	-80(%rbp), %rax
	movq	(%rax), %rax
	addq	$1, %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	subl	$48, %eax
	cmpl	$72, %eax
	ja	.L380
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L382(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L382(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L382:
	.long	.L392-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L391-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L390-.L382
	.long	.L389-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L388-.L382
	.long	.L387-.L382
	.long	.L380-.L382
	.long	.L386-.L382
	.long	.L385-.L382
	.long	.L380-.L382
	.long	.L384-.L382
	.long	.L383-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L380-.L382
	.long	.L381-.L382
	.text
.L381:
	movq	$6, -16(%rbp)
	jmp	.L393
.L389:
	movq	$53, -16(%rbp)
	jmp	.L393
.L387:
	movq	$39, -16(%rbp)
	jmp	.L393
.L383:
	movq	$67, -16(%rbp)
	jmp	.L393
.L385:
	movq	$57, -16(%rbp)
	jmp	.L393
.L388:
	movq	$1, -16(%rbp)
	jmp	.L393
.L384:
	movq	$65, -16(%rbp)
	jmp	.L393
.L386:
	movq	$47, -16(%rbp)
	jmp	.L393
.L390:
	movq	$71, -16(%rbp)
	jmp	.L393
.L391:
	movq	$41, -16(%rbp)
	jmp	.L393
.L392:
	movq	$62, -16(%rbp)
	jmp	.L393
.L380:
	movq	$38, -16(%rbp)
	nop
.L393:
	jmp	.L350
.L326:
	cmpl	$1, -44(%rbp)
	je	.L394
	movq	$44, -16(%rbp)
	jmp	.L350
.L394:
	movq	$50, -16(%rbp)
	jmp	.L350
.L329:
	call	replay
	movq	$61, -16(%rbp)
	jmp	.L350
.L320:
	movl	flagr(%rip), %eax
	testl	%eax, %eax
	je	.L396
	movq	$29, -16(%rbp)
	jmp	.L350
.L396:
	movq	$14, -16(%rbp)
	jmp	.L350
.L348:
	movq	-80(%rbp), %rax
	movq	(%rax), %rax
	leaq	length(%rip), %rdx
	leaq	.LC8(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_sscanf@PLT
	movl	%eax, -44(%rbp)
	movq	$35, -16(%rbp)
	jmp	.L350
.L336:
	movq	in(%rip), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movl	%eax, -20(%rbp)
	movq	$15, -16(%rbp)
	jmp	.L350
.L401:
	nop
.L350:
	jmp	.L398
.L400:
	call	__stack_chk_fail@PLT
.L399:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE14:
	.size	main, .-main
	.section	.rodata
	.align 8
.LC10:
	.long	0
	.long	1093567616
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
