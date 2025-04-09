	.file	"timber-they_AdventOfCode20_8_flatten.c"
	.text
	.globl	_TIG_IZ_MghE_argv
	.bss
	.align 8
	.type	_TIG_IZ_MghE_argv, @object
	.size	_TIG_IZ_MghE_argv, 8
_TIG_IZ_MghE_argv:
	.zero	8
	.globl	_TIG_IZ_MghE_argc
	.align 4
	.type	_TIG_IZ_MghE_argc, @object
	.size	_TIG_IZ_MghE_argc, 4
_TIG_IZ_MghE_argc:
	.zero	4
	.globl	_TIG_IZ_MghE_envp
	.align 8
	.type	_TIG_IZ_MghE_envp, @object
	.size	_TIG_IZ_MghE_envp, 8
_TIG_IZ_MghE_envp:
	.zero	8
	.globl	err
	.align 4
	.type	err, @object
	.size	err, 4
err:
	.zero	4
	.section	.rodata
.LC0:
	.string	"r"
.LC1:
	.string	"in8"
	.align 8
.LC2:
	.string	"Acc after unmodified execution: %d\n"
	.align 8
.LC3:
	.string	"Faulty command was at index %d, resulting in acc %d\n"
	.align 8
.LC4:
	.string	"Couldn't switch back at index %d\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB1:
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
	movl	$0, err(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_MghE_envp(%rip)
	nop
.L3:
	movq	$0, _TIG_IZ_MghE_argv(%rip)
	nop
.L4:
	movl	$0, _TIG_IZ_MghE_argc(%rip)
	nop
	nop
.L5:
.L6:
#APP
# 136 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-MghE--0
# 0 "" 2
#NO_APP
	movl	-68(%rbp), %eax
	movl	%eax, _TIG_IZ_MghE_argc(%rip)
	movq	-80(%rbp), %rax
	movq	%rax, _TIG_IZ_MghE_argv(%rip)
	movq	-88(%rbp), %rax
	movq	%rax, _TIG_IZ_MghE_envp(%rip)
	nop
	movq	$16, -24(%rbp)
.L33:
	cmpq	$18, -24(%rbp)
	ja	.L35
	movq	-24(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L9(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L9(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L9:
	.long	.L35-.L9
	.long	.L22-.L9
	.long	.L21-.L9
	.long	.L20-.L9
	.long	.L19-.L9
	.long	.L18-.L9
	.long	.L17-.L9
	.long	.L16-.L9
	.long	.L15-.L9
	.long	.L35-.L9
	.long	.L14-.L9
	.long	.L13-.L9
	.long	.L35-.L9
	.long	.L35-.L9
	.long	.L12-.L9
	.long	.L11-.L9
	.long	.L10-.L9
	.long	.L35-.L9
	.long	.L8-.L9
	.text
.L8:
	movl	$0, %eax
	jmp	.L34
.L19:
	movl	-56(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-32(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, %rdi
	call	switchCmd
	movl	%eax, -52(%rbp)
	movq	$7, -24(%rbp)
	jmp	.L24
.L12:
	addl	$1, -56(%rbp)
	movq	$8, -24(%rbp)
	jmp	.L24
.L11:
	movq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$18, -24(%rbp)
	jmp	.L24
.L15:
	cmpl	$637, -56(%rbp)
	jg	.L25
	movq	$4, -24(%rbp)
	jmp	.L24
.L25:
	movq	$15, -24(%rbp)
	jmp	.L24
.L22:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -40(%rbp)
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	read
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -32(%rbp)
	movq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	runProgram
	movl	%eax, -44(%rbp)
	movl	-44(%rbp), %eax
	movl	%eax, -60(%rbp)
	movl	-60(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$0, -56(%rbp)
	movq	$8, -24(%rbp)
	jmp	.L24
.L20:
	movl	-60(%rbp), %edx
	movl	-56(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$15, -24(%rbp)
	jmp	.L24
.L10:
	movq	$1, -24(%rbp)
	jmp	.L24
.L13:
	movl	err(%rip), %eax
	testl	%eax, %eax
	jne	.L27
	movq	$3, -24(%rbp)
	jmp	.L24
.L27:
	movq	$2, -24(%rbp)
	jmp	.L24
.L17:
	movq	stderr(%rip), %rax
	movl	-56(%rbp), %edx
	leaq	.LC4(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	$14, -24(%rbp)
	jmp	.L24
.L18:
	movq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	runProgram
	movl	%eax, -60(%rbp)
	movq	$11, -24(%rbp)
	jmp	.L24
.L14:
	cmpl	$0, -48(%rbp)
	je	.L29
	movq	$14, -24(%rbp)
	jmp	.L24
.L29:
	movq	$6, -24(%rbp)
	jmp	.L24
.L16:
	cmpl	$0, -52(%rbp)
	je	.L31
	movq	$5, -24(%rbp)
	jmp	.L24
.L31:
	movq	$14, -24(%rbp)
	jmp	.L24
.L21:
	movl	-56(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-32(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, %rdi
	call	switchCmd
	movl	%eax, -48(%rbp)
	movq	$10, -24(%rbp)
	jmp	.L24
.L35:
	nop
.L24:
	jmp	.L33
.L34:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	main, .-main
	.section	.rodata
.LC5:
	.string	"jmp"
.LC6:
	.string	"acc"
.LC7:
	.string	"eof"
.LC8:
	.string	" %d\n"
.LC9:
	.string	"nop"
	.text
	.globl	printProgram
	.type	printProgram, @function
printProgram:
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
	movq	$15, -8(%rbp)
.L59:
	cmpq	$16, -8(%rbp)
	ja	.L60
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L39(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L39(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L39:
	.long	.L60-.L39
	.long	.L48-.L39
	.long	.L47-.L39
	.long	.L61-.L39
	.long	.L60-.L39
	.long	.L60-.L39
	.long	.L45-.L39
	.long	.L60-.L39
	.long	.L44-.L39
	.long	.L43-.L39
	.long	.L60-.L39
	.long	.L60-.L39
	.long	.L42-.L39
	.long	.L60-.L39
	.long	.L41-.L39
	.long	.L40-.L39
	.long	.L38-.L39
	.text
.L41:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$6, -8(%rbp)
	jmp	.L49
.L40:
	movl	$0, -12(%rbp)
	movq	$12, -8(%rbp)
	jmp	.L49
.L42:
	cmpl	$637, -12(%rbp)
	jg	.L50
	movq	$9, -8(%rbp)
	jmp	.L49
.L50:
	movq	$3, -8(%rbp)
	jmp	.L49
.L44:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$6, -8(%rbp)
	jmp	.L49
.L48:
	movq	$6, -8(%rbp)
	jmp	.L49
.L38:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$6, -8(%rbp)
	jmp	.L49
.L43:
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	$3, %eax
	je	.L53
	cmpl	$3, %eax
	ja	.L54
	cmpl	$2, %eax
	je	.L55
	cmpl	$2, %eax
	ja	.L54
	testl	%eax, %eax
	je	.L56
	cmpl	$1, %eax
	je	.L57
	jmp	.L54
.L53:
	movq	$16, -8(%rbp)
	jmp	.L58
.L55:
	movq	$2, -8(%rbp)
	jmp	.L58
.L57:
	movq	$14, -8(%rbp)
	jmp	.L58
.L56:
	movq	$8, -8(%rbp)
	jmp	.L58
.L54:
	movq	$1, -8(%rbp)
	nop
.L58:
	jmp	.L49
.L45:
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movl	4(%rax), %eax
	movl	%eax, %esi
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -12(%rbp)
	movq	$12, -8(%rbp)
	jmp	.L49
.L47:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$6, -8(%rbp)
	jmp	.L49
.L60:
	nop
.L49:
	jmp	.L59
.L61:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	printProgram, .-printProgram
	.section	.rodata
.LC10:
	.string	"Unexpected cmd: %d\n"
	.text
	.globl	switchCmd
	.type	switchCmd, @function
switchCmd:
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
.L81:
	cmpq	$8, -8(%rbp)
	ja	.L82
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L65(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L65(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L65:
	.long	.L73-.L65
	.long	.L72-.L65
	.long	.L71-.L65
	.long	.L70-.L65
	.long	.L69-.L65
	.long	.L68-.L65
	.long	.L67-.L65
	.long	.L66-.L65
	.long	.L64-.L65
	.text
.L69:
	movq	-24(%rbp), %rax
	movl	$1, (%rax)
	movq	$7, -8(%rbp)
	jmp	.L74
.L64:
	movl	$1, %eax
	jmp	.L75
.L72:
	movl	$0, %eax
	jmp	.L75
.L70:
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	cmpl	$3, %eax
	je	.L76
	cmpl	$3, %eax
	ja	.L77
	cmpl	$2, %eax
	je	.L78
	cmpl	$2, %eax
	ja	.L77
	testl	%eax, %eax
	je	.L79
	cmpl	$1, %eax
	jne	.L77
	movq	$2, -8(%rbp)
	jmp	.L80
.L78:
	movq	$4, -8(%rbp)
	jmp	.L80
.L76:
	movq	$5, -8(%rbp)
	jmp	.L80
.L79:
	movq	$1, -8(%rbp)
	jmp	.L80
.L77:
	movq	$6, -8(%rbp)
	nop
.L80:
	jmp	.L74
.L67:
	movq	-24(%rbp), %rax
	movl	(%rax), %edx
	movq	stderr(%rip), %rax
	leaq	.LC10(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	$0, -8(%rbp)
	jmp	.L74
.L68:
	movl	$0, %eax
	jmp	.L75
.L73:
	movl	$-1, %eax
	jmp	.L75
.L66:
	movl	$1, %eax
	jmp	.L75
.L71:
	movq	-24(%rbp), %rax
	movl	$2, (%rax)
	movq	$8, -8(%rbp)
	jmp	.L74
.L82:
	nop
.L74:
	jmp	.L81
.L75:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	switchCmd, .-switchCmd
	.globl	contains
	.type	contains, @function
contains:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movl	%esi, -28(%rbp)
	movq	$4, -8(%rbp)
.L102:
	cmpq	$7, -8(%rbp)
	ja	.L103
	movq	-8(%rbp), %rax
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
	.long	.L93-.L86
	.long	.L92-.L86
	.long	.L91-.L86
	.long	.L90-.L86
	.long	.L89-.L86
	.long	.L88-.L86
	.long	.L87-.L86
	.long	.L85-.L86
	.text
.L89:
	movl	$0, -12(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L94
.L92:
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	%eax, -28(%rbp)
	jne	.L95
	movq	$6, -8(%rbp)
	jmp	.L94
.L95:
	movq	$3, -8(%rbp)
	jmp	.L94
.L90:
	addl	$1, -12(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L94
.L87:
	movl	$1, %eax
	jmp	.L97
.L88:
	cmpl	$637, -12(%rbp)
	jg	.L98
	movq	$7, -8(%rbp)
	jmp	.L94
.L98:
	movq	$2, -8(%rbp)
	jmp	.L94
.L93:
	movl	$0, %eax
	jmp	.L97
.L85:
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	$-1, %eax
	jne	.L100
	movq	$0, -8(%rbp)
	jmp	.L94
.L100:
	movq	$1, -8(%rbp)
	jmp	.L94
.L91:
	movl	$0, %eax
	jmp	.L97
.L103:
	nop
.L94:
	jmp	.L102
.L97:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	contains, .-contains
	.globl	runProgram
	.type	runProgram, @function
runProgram:
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
	movq	$7, -16(%rbp)
.L136:
	cmpq	$25, -16(%rbp)
	ja	.L137
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L107(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L107(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L107:
	.long	.L123-.L107
	.long	.L122-.L107
	.long	.L137-.L107
	.long	.L121-.L107
	.long	.L137-.L107
	.long	.L137-.L107
	.long	.L137-.L107
	.long	.L120-.L107
	.long	.L119-.L107
	.long	.L118-.L107
	.long	.L117-.L107
	.long	.L116-.L107
	.long	.L115-.L107
	.long	.L137-.L107
	.long	.L114-.L107
	.long	.L137-.L107
	.long	.L113-.L107
	.long	.L112-.L107
	.long	.L111-.L107
	.long	.L110-.L107
	.long	.L109-.L107
	.long	.L137-.L107
	.long	.L137-.L107
	.long	.L137-.L107
	.long	.L108-.L107
	.long	.L106-.L107
	.text
.L111:
	movl	-40(%rbp), %eax
	movl	%eax, -28(%rbp)
	addl	$1, -40(%rbp)
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rax, %rdx
	movl	-44(%rbp), %eax
	movl	%eax, (%rdx)
	movq	$9, -16(%rbp)
	jmp	.L124
.L106:
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$12, -16(%rbp)
	jmp	.L124
.L114:
	movl	-36(%rbp), %eax
	jmp	.L125
.L115:
	movl	-36(%rbp), %eax
	jmp	.L125
.L119:
	movl	$1, err(%rip)
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$17, -16(%rbp)
	jmp	.L124
.L122:
	movl	-44(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movl	4(%rax), %eax
	addl	%eax, -44(%rbp)
	movq	$16, -16(%rbp)
	jmp	.L124
.L121:
	addl	$1, -44(%rbp)
	movq	$16, -16(%rbp)
	jmp	.L124
.L113:
	movl	-44(%rbp), %edx
	movq	-24(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	contains
	movl	%eax, -32(%rbp)
	movq	$10, -16(%rbp)
	jmp	.L124
.L108:
	cmpl	$637, -44(%rbp)
	jle	.L126
	movq	$20, -16(%rbp)
	jmp	.L124
.L126:
	movq	$18, -16(%rbp)
	jmp	.L124
.L116:
	movl	-44(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movl	4(%rax), %eax
	addl	%eax, -36(%rbp)
	addl	$1, -44(%rbp)
	movq	$16, -16(%rbp)
	jmp	.L124
.L118:
	movl	-44(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	$3, %eax
	je	.L128
	cmpl	$3, %eax
	ja	.L129
	cmpl	$2, %eax
	je	.L130
	cmpl	$2, %eax
	ja	.L129
	testl	%eax, %eax
	je	.L131
	cmpl	$1, %eax
	je	.L132
	jmp	.L129
.L128:
	movq	$25, -16(%rbp)
	jmp	.L133
.L130:
	movq	$3, -16(%rbp)
	jmp	.L133
.L132:
	movq	$1, -16(%rbp)
	jmp	.L133
.L131:
	movq	$11, -16(%rbp)
	jmp	.L133
.L129:
	movq	$0, -16(%rbp)
	nop
.L133:
	jmp	.L124
.L110:
	movl	$4, %esi
	movl	$638, %edi
	call	calloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movl	$638, %edx
	movl	$-1, %esi
	movq	%rax, %rdi
	call	memset@PLT
	movl	$0, -44(%rbp)
	movl	$0, -40(%rbp)
	movl	$0, -36(%rbp)
	movl	$0, err(%rip)
	movq	$16, -16(%rbp)
	jmp	.L124
.L112:
	movl	-36(%rbp), %eax
	jmp	.L125
.L117:
	cmpl	$0, -32(%rbp)
	je	.L134
	movq	$8, -16(%rbp)
	jmp	.L124
.L134:
	movq	$24, -16(%rbp)
	jmp	.L124
.L123:
	movq	$16, -16(%rbp)
	jmp	.L124
.L120:
	movq	$19, -16(%rbp)
	jmp	.L124
.L109:
	movl	$-1, err(%rip)
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$14, -16(%rbp)
	jmp	.L124
.L137:
	nop
.L124:
	jmp	.L136
.L125:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	runProgram, .-runProgram
	.section	.rodata
	.align 8
.LC11:
	.string	"Command start %c (of line %s) was unexpected.\n"
	.text
	.globl	read
	.type	read, @function
read:
.LFB10:
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
	movq	%rdi, -104(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -24(%rbp)
	xorl	%eax, %eax
	movq	$3, -48(%rbp)
.L176:
	cmpq	$30, -48(%rbp)
	ja	.L179
	movq	-48(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L141(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L141(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L141:
	.long	.L161-.L141
	.long	.L179-.L141
	.long	.L160-.L141
	.long	.L159-.L141
	.long	.L158-.L141
	.long	.L179-.L141
	.long	.L157-.L141
	.long	.L156-.L141
	.long	.L155-.L141
	.long	.L179-.L141
	.long	.L154-.L141
	.long	.L153-.L141
	.long	.L179-.L141
	.long	.L152-.L141
	.long	.L179-.L141
	.long	.L179-.L141
	.long	.L151-.L141
	.long	.L150-.L141
	.long	.L149-.L141
	.long	.L148-.L141
	.long	.L179-.L141
	.long	.L147-.L141
	.long	.L179-.L141
	.long	.L179-.L141
	.long	.L179-.L141
	.long	.L146-.L141
	.long	.L145-.L141
	.long	.L144-.L141
	.long	.L143-.L141
	.long	.L142-.L141
	.long	.L140-.L141
	.text
.L149:
	movq	-80(%rbp), %rax
	addq	$4, %rax
	movzbl	(%rax), %eax
	cmpb	$45, %al
	jne	.L162
	movq	$27, -48(%rbp)
	jmp	.L164
.L162:
	movq	$16, -48(%rbp)
	jmp	.L164
.L146:
	movq	-64(%rbp), %rax
	jmp	.L177
.L158:
	movl	-88(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-64(%rbp), %rax
	addq	%rdx, %rax
	movl	$0, (%rax)
	movq	$21, -48(%rbp)
	jmp	.L164
.L140:
	cmpq	$0, -56(%rbp)
	jns	.L166
	movq	$17, -48(%rbp)
	jmp	.L164
.L166:
	movq	$7, -48(%rbp)
	jmp	.L164
.L155:
	movl	-88(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-64(%rbp), %rax
	addq	%rdx, %rax
	movl	$3, (%rax)
	movq	-80(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$25, -48(%rbp)
	jmp	.L164
.L159:
	movq	$0, -48(%rbp)
	jmp	.L164
.L151:
	movl	$1, -84(%rbp)
	movq	$13, -48(%rbp)
	jmp	.L164
.L147:
	movq	-80(%rbp), %rax
	addq	$5, %rax
	movl	-88(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,8), %rcx
	movq	-64(%rbp), %rdx
	leaq	(%rcx,%rdx), %rbx
	movq	%rax, %rdi
	call	atoi@PLT
	movl	%eax, 4(%rbx)
	movq	$18, -48(%rbp)
	jmp	.L164
.L145:
	cmpl	$637, -88(%rbp)
	jg	.L168
	movq	$19, -48(%rbp)
	jmp	.L164
.L168:
	movq	$6, -48(%rbp)
	jmp	.L164
.L153:
	movq	-64(%rbp), %rax
	jmp	.L177
.L152:
	movl	-88(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-64(%rbp), %rax
	addq	%rdx, %rax
	movl	4(%rax), %eax
	movl	-88(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,8), %rcx
	movq	-64(%rbp), %rdx
	addq	%rcx, %rdx
	imull	-84(%rbp), %eax
	movl	%eax, 4(%rdx)
	addl	$1, -88(%rbp)
	movq	$26, -48(%rbp)
	jmp	.L164
.L148:
	movq	$10, -72(%rbp)
	movq	-104(%rbp), %rdx
	leaq	-72(%rbp), %rcx
	leaq	-80(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	getline@PLT
	movq	%rax, -56(%rbp)
	movq	$30, -48(%rbp)
	jmp	.L164
.L150:
	movl	-88(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-64(%rbp), %rax
	addq	%rdx, %rax
	movl	$3, (%rax)
	movq	-80(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$29, -48(%rbp)
	jmp	.L164
.L157:
	movq	-80(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$11, -48(%rbp)
	jmp	.L164
.L144:
	movl	$-1, -84(%rbp)
	movq	$13, -48(%rbp)
	jmp	.L164
.L143:
	movl	-88(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-64(%rbp), %rax
	addq	%rdx, %rax
	movl	$2, (%rax)
	movq	$21, -48(%rbp)
	jmp	.L164
.L154:
	movl	-88(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-64(%rbp), %rax
	addq	%rdx, %rax
	movl	$1, (%rax)
	movq	$21, -48(%rbp)
	jmp	.L164
.L161:
	movl	$10, %edi
	call	malloc@PLT
	movq	%rax, -40(%rbp)
	movq	-40(%rbp), %rax
	movq	%rax, -80(%rbp)
	movl	$8, %esi
	movl	$638, %edi
	call	calloc@PLT
	movq	%rax, -32(%rbp)
	movq	-32(%rbp), %rax
	movq	%rax, -64(%rbp)
	movl	$0, -88(%rbp)
	movq	$26, -48(%rbp)
	jmp	.L164
.L156:
	movq	-80(%rbp), %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	cmpl	$110, %eax
	je	.L170
	cmpl	$110, %eax
	jg	.L171
	cmpl	$106, %eax
	je	.L172
	cmpl	$106, %eax
	jg	.L171
	testl	%eax, %eax
	je	.L173
	cmpl	$97, %eax
	je	.L174
	jmp	.L171
.L170:
	movq	$28, -48(%rbp)
	jmp	.L175
.L174:
	movq	$4, -48(%rbp)
	jmp	.L175
.L172:
	movq	$10, -48(%rbp)
	jmp	.L175
.L173:
	movq	$8, -48(%rbp)
	jmp	.L175
.L171:
	movq	$2, -48(%rbp)
	nop
.L175:
	jmp	.L164
.L142:
	movq	-64(%rbp), %rax
	jmp	.L177
.L160:
	movq	-80(%rbp), %rcx
	movq	-80(%rbp), %rax
	movzbl	(%rax), %eax
	movsbl	%al, %edx
	movq	stderr(%rip), %rax
	leaq	.LC11(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	$21, -48(%rbp)
	jmp	.L164
.L179:
	nop
.L164:
	jmp	.L176
.L177:
	movq	-24(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L178
	call	__stack_chk_fail@PLT
.L178:
	movq	-8(%rbp), %rbx
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	read, .-read
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
