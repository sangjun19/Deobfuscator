	.file	"lucija-26_Strukture-altf4-_zad7_flatten.c"
	.text
	.globl	_TIG_IZ_IvFK_argv
	.bss
	.align 8
	.type	_TIG_IZ_IvFK_argv, @object
	.size	_TIG_IZ_IvFK_argv, 8
_TIG_IZ_IvFK_argv:
	.zero	8
	.globl	_TIG_IZ_IvFK_envp
	.align 8
	.type	_TIG_IZ_IvFK_envp, @object
	.size	_TIG_IZ_IvFK_envp, 8
_TIG_IZ_IvFK_envp:
	.zero	8
	.globl	_TIG_IZ_IvFK_argc
	.align 4
	.type	_TIG_IZ_IvFK_argc, @object
	.size	_TIG_IZ_IvFK_argc, 4
_TIG_IZ_IvFK_argc:
	.zero	4
	.section	.rodata
	.align 8
.LC0:
	.string	"Memory allocation unsuccessful."
	.text
	.globl	newDir
	.type	newDir, @function
newDir:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	$0, -16(%rbp)
.L15:
	cmpq	$7, -16(%rbp)
	ja	.L16
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
	.long	.L10-.L4
	.long	.L9-.L4
	.long	.L8-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L16-.L4
	.long	.L3-.L4
	.text
.L6:
	movq	$0, -24(%rbp)
	movl	$272, %edi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	$2, -16(%rbp)
	jmp	.L11
.L9:
	movq	-24(%rbp), %rax
	jmp	.L12
.L7:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$5, -16(%rbp)
	jmp	.L11
.L5:
	movl	$0, %eax
	jmp	.L12
.L10:
	movq	$4, -16(%rbp)
	jmp	.L11
.L3:
	movq	-24(%rbp), %rax
	movq	$0, 264(%rax)
	movq	-24(%rbp), %rax
	movq	$0, 256(%rax)
	movq	$1, -16(%rbp)
	jmp	.L11
.L8:
	cmpq	$0, -24(%rbp)
	jne	.L13
	movq	$3, -16(%rbp)
	jmp	.L11
.L13:
	movq	$7, -16(%rbp)
	jmp	.L11
.L16:
	nop
.L11:
	jmp	.L15
.L12:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	newDir, .-newDir
	.globl	delete
	.type	delete, @function
delete:
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
	movq	$0, -8(%rbp)
.L27:
	cmpq	$4, -8(%rbp)
	je	.L18
	cmpq	$4, -8(%rbp)
	ja	.L28
	cmpq	$2, -8(%rbp)
	je	.L20
	cmpq	$2, -8(%rbp)
	ja	.L28
	cmpq	$0, -8(%rbp)
	je	.L21
	cmpq	$1, -8(%rbp)
	je	.L22
	jmp	.L28
.L18:
	movl	$0, %eax
	jmp	.L23
.L22:
	movq	-24(%rbp), %rax
	movq	256(%rax), %rax
	movq	%rax, %rdi
	call	delete
	movq	-24(%rbp), %rdx
	movq	%rax, 256(%rdx)
	movq	-24(%rbp), %rax
	movq	264(%rax), %rax
	movq	%rax, %rdi
	call	delete
	movq	-24(%rbp), %rdx
	movq	%rax, 264(%rdx)
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$4, -8(%rbp)
	jmp	.L24
.L21:
	cmpq	$0, -24(%rbp)
	jne	.L25
	movq	$2, -8(%rbp)
	jmp	.L24
.L25:
	movq	$1, -8(%rbp)
	jmp	.L24
.L20:
	movl	$0, %eax
	jmp	.L23
.L28:
	nop
.L24:
	jmp	.L27
.L23:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	delete, .-delete
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
	subq	$624, %rsp
	movl	%edi, -596(%rbp)
	movq	%rsi, -608(%rbp)
	movq	%rdx, -616(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_IvFK_envp(%rip)
	nop
.L30:
	movq	$0, _TIG_IZ_IvFK_argv(%rip)
	nop
.L31:
	movl	$0, _TIG_IZ_IvFK_argc(%rip)
	nop
	nop
.L32:
.L33:
#APP
# 299 "lucija-26_Strukture-altf4-_zad7.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-IvFK--0
# 0 "" 2
#NO_APP
	movl	-596(%rbp), %eax
	movl	%eax, _TIG_IZ_IvFK_argc(%rip)
	movq	-608(%rbp), %rax
	movq	%rax, _TIG_IZ_IvFK_argv(%rip)
	movq	-616(%rbp), %rax
	movq	%rax, _TIG_IZ_IvFK_envp(%rip)
	nop
	movq	$8, -576(%rbp)
.L51:
	cmpq	$13, -576(%rbp)
	ja	.L54
	movq	-576(%rbp), %rax
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
	.long	.L44-.L36
	.long	.L43-.L36
	.long	.L54-.L36
	.long	.L54-.L36
	.long	.L42-.L36
	.long	.L41-.L36
	.long	.L54-.L36
	.long	.L40-.L36
	.long	.L39-.L36
	.long	.L54-.L36
	.long	.L38-.L36
	.long	.L37-.L36
	.long	.L35-.L36
	.text
.L37:
	movb	$67, -544(%rbp)
	movb	$58, -543(%rbp)
	movb	$0, -542(%rbp)
	movl	$3, -580(%rbp)
	movq	$11, -576(%rbp)
	jmp	.L45
.L40:
	movq	$9, -576(%rbp)
	jmp	.L45
.L44:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L52
	jmp	.L53
.L38:
	cmpl	$255, -580(%rbp)
	jbe	.L47
	movq	$13, -576(%rbp)
	jmp	.L45
.L47:
	movq	$2, -576(%rbp)
	jmp	.L45
.L39:
	movq	$0, -568(%rbp)
	movb	$0, -272(%rbp)
	movl	$1, -584(%rbp)
	movq	$5, -576(%rbp)
	jmp	.L45
.L35:
	movq	$0, -288(%rbp)
	movq	$0, -280(%rbp)
	movq	$0, -560(%rbp)
	movq	$0, -552(%rbp)
	leaq	-544(%rbp), %rdx
	leaq	-560(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	Push
	leaq	-544(%rbp), %rdx
	leaq	-560(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	terminal
	movq	-280(%rbp), %rax
	movq	%rax, %rdi
	call	delete
	movq	$1, -576(%rbp)
	jmp	.L45
.L41:
	movl	-584(%rbp), %eax
	movb	$0, -272(%rbp,%rax)
	addl	$1, -584(%rbp)
	movq	$5, -576(%rbp)
	jmp	.L45
.L42:
	cmpl	$255, -584(%rbp)
	jbe	.L49
	movq	$12, -576(%rbp)
	jmp	.L45
.L49:
	movq	$6, -576(%rbp)
	jmp	.L45
.L43:
	movl	-580(%rbp), %eax
	movb	$0, -544(%rbp,%rax)
	addl	$1, -580(%rbp)
	movq	$11, -576(%rbp)
	jmp	.L45
.L54:
	nop
.L45:
	jmp	.L51
.L53:
	call	__stack_chk_fail@PLT
.L52:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	main, .-main
	.globl	makeDir
	.type	makeDir, @function
makeDir:
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
	movq	%rsi, -64(%rbp)
	movq	$11, -16(%rbp)
.L83:
	cmpq	$18, -16(%rbp)
	ja	.L84
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L58(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L58(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L58:
	.long	.L72-.L58
	.long	.L71-.L58
	.long	.L84-.L58
	.long	.L70-.L58
	.long	.L84-.L58
	.long	.L69-.L58
	.long	.L68-.L58
	.long	.L67-.L58
	.long	.L66-.L58
	.long	.L65-.L58
	.long	.L64-.L58
	.long	.L63-.L58
	.long	.L62-.L58
	.long	.L84-.L58
	.long	.L61-.L58
	.long	.L60-.L58
	.long	.L84-.L58
	.long	.L59-.L58
	.long	.L57-.L58
	.text
.L57:
	movq	-40(%rbp), %rdx
	movq	-32(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -44(%rbp)
	movq	$0, -16(%rbp)
	jmp	.L73
.L61:
	movl	$0, %eax
	jmp	.L74
.L60:
	movq	-56(%rbp), %rax
	movq	-40(%rbp), %rdx
	movq	%rdx, 264(%rax)
	movq	$3, -16(%rbp)
	jmp	.L73
.L62:
	cmpl	$0, -48(%rbp)
	jle	.L75
	movq	$10, -16(%rbp)
	jmp	.L73
.L75:
	movq	$9, -16(%rbp)
	jmp	.L73
.L66:
	movq	-32(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	-32(%rbp), %rax
	movq	256(%rax), %rax
	movq	%rax, -32(%rbp)
	movq	$9, -16(%rbp)
	jmp	.L73
.L71:
	cmpq	$0, -32(%rbp)
	jne	.L77
	movq	$15, -16(%rbp)
	jmp	.L73
.L77:
	movq	$6, -16(%rbp)
	jmp	.L73
.L70:
	movl	$0, %eax
	jmp	.L74
.L63:
	movq	$7, -16(%rbp)
	jmp	.L73
.L65:
	cmpq	$0, -32(%rbp)
	je	.L79
	movq	$18, -16(%rbp)
	jmp	.L73
.L79:
	movq	$5, -16(%rbp)
	jmp	.L73
.L59:
	movl	$0, %eax
	jmp	.L74
.L68:
	movq	-40(%rbp), %rdx
	movq	-32(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -48(%rbp)
	movq	$12, -16(%rbp)
	jmp	.L73
.L69:
	movq	-24(%rbp), %rax
	movq	-40(%rbp), %rdx
	movq	%rdx, 256(%rax)
	movq	-40(%rbp), %rax
	movq	-32(%rbp), %rdx
	movq	%rdx, 256(%rax)
	movq	$14, -16(%rbp)
	jmp	.L73
.L64:
	movq	-40(%rbp), %rax
	movq	-32(%rbp), %rdx
	movq	%rdx, 256(%rax)
	movq	-56(%rbp), %rax
	movq	-40(%rbp), %rdx
	movq	%rdx, 264(%rax)
	movq	$17, -16(%rbp)
	jmp	.L73
.L72:
	cmpl	$0, -44(%rbp)
	jns	.L81
	movq	$8, -16(%rbp)
	jmp	.L73
.L81:
	movq	$5, -16(%rbp)
	jmp	.L73
.L67:
	movq	$0, -40(%rbp)
	movq	-56(%rbp), %rax
	movq	264(%rax), %rax
	movq	%rax, -32(%rbp)
	movq	$0, -8(%rbp)
	movq	$0, -24(%rbp)
	call	newDir
	movq	%rax, -40(%rbp)
	movq	-40(%rbp), %rax
	movq	-64(%rbp), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcpy@PLT
	movq	$1, -16(%rbp)
	jmp	.L73
.L84:
	nop
.L73:
	jmp	.L83
.L74:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	makeDir, .-makeDir
	.globl	Push
	.type	Push, @function
Push:
.LFB9:
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
	movq	$0, -16(%rbp)
.L97:
	cmpq	$7, -16(%rbp)
	ja	.L99
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L88(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L88(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L88:
	.long	.L92-.L88
	.long	.L91-.L88
	.long	.L99-.L88
	.long	.L90-.L88
	.long	.L99-.L88
	.long	.L99-.L88
	.long	.L89-.L88
	.long	.L87-.L88
	.text
.L91:
	movl	$16, %edi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-24(%rbp), %rax
	movq	-8(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movq	-24(%rbp), %rax
	movq	8(%rax), %rax
	movq	$0, 8(%rax)
	movq	-24(%rbp), %rax
	movq	8(%rax), %rax
	movq	-48(%rbp), %rdx
	movq	%rdx, (%rax)
	movq	$7, -16(%rbp)
	jmp	.L93
.L90:
	movq	-24(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	$6, -16(%rbp)
	jmp	.L93
.L89:
	movq	-24(%rbp), %rax
	movq	8(%rax), %rax
	testq	%rax, %rax
	je	.L94
	movq	$3, -16(%rbp)
	jmp	.L93
.L94:
	movq	$1, -16(%rbp)
	jmp	.L93
.L92:
	movq	-40(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	$6, -16(%rbp)
	jmp	.L93
.L87:
	movl	$0, %eax
	jmp	.L98
.L99:
	nop
.L93:
	jmp	.L97
.L98:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	Push, .-Push
	.globl	Pop
	.type	Pop, @function
Pop:
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
	movq	$1, -16(%rbp)
.L117:
	cmpq	$9, -16(%rbp)
	ja	.L118
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L103(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L103(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L103:
	.long	.L118-.L103
	.long	.L110-.L103
	.long	.L109-.L103
	.long	.L108-.L103
	.long	.L107-.L103
	.long	.L106-.L103
	.long	.L105-.L103
	.long	.L104-.L103
	.long	.L118-.L103
	.long	.L102-.L103
	.text
.L107:
	movq	-32(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	-32(%rbp), %rax
	movq	$0, 8(%rax)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$7, -16(%rbp)
	jmp	.L111
.L110:
	movq	$5, -16(%rbp)
	jmp	.L111
.L108:
	movq	-40(%rbp), %rax
	movq	8(%rax), %rax
	testq	%rax, %rax
	jne	.L112
	movq	$6, -16(%rbp)
	jmp	.L111
.L112:
	movq	$2, -16(%rbp)
	jmp	.L111
.L102:
	movq	-32(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, -32(%rbp)
	movq	$2, -16(%rbp)
	jmp	.L111
.L105:
	movl	$0, %eax
	jmp	.L114
.L106:
	movq	-40(%rbp), %rax
	movq	%rax, -32(%rbp)
	movq	$0, -8(%rbp)
	movq	$0, -24(%rbp)
	movq	$3, -16(%rbp)
	jmp	.L111
.L104:
	movq	-24(%rbp), %rax
	jmp	.L114
.L109:
	movq	-32(%rbp), %rax
	movq	8(%rax), %rax
	movq	8(%rax), %rax
	testq	%rax, %rax
	je	.L115
	movq	$9, -16(%rbp)
	jmp	.L111
.L115:
	movq	$4, -16(%rbp)
	jmp	.L111
.L118:
	nop
.L111:
	jmp	.L117
.L114:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	Pop, .-Pop
	.section	.rodata
	.align 8
.LC1:
	.string	"'%s' is not recognized as an internal or external command, operable program or batch file.\n"
.LC2:
	.string	" %s %[^\n]"
.LC3:
	.string	"md"
.LC4:
	.string	"exit"
.LC5:
	.string	"cd.."
.LC6:
	.string	"dir"
.LC7:
	.string	"cd"
.LC8:
	.string	"%s\\"
	.text
	.globl	terminal
	.type	terminal, @function
terminal:
.LFB11:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$864, %rsp
	movq	%rdi, -856(%rbp)
	movq	%rsi, -864(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$41, -792(%rbp)
.L175:
	cmpq	$49, -792(%rbp)
	ja	.L178
	movq	-792(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L122(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L122(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L122:
	.long	.L152-.L122
	.long	.L178-.L122
	.long	.L178-.L122
	.long	.L178-.L122
	.long	.L151-.L122
	.long	.L150-.L122
	.long	.L149-.L122
	.long	.L178-.L122
	.long	.L148-.L122
	.long	.L178-.L122
	.long	.L147-.L122
	.long	.L146-.L122
	.long	.L178-.L122
	.long	.L145-.L122
	.long	.L144-.L122
	.long	.L143-.L122
	.long	.L142-.L122
	.long	.L141-.L122
	.long	.L140-.L122
	.long	.L178-.L122
	.long	.L178-.L122
	.long	.L139-.L122
	.long	.L178-.L122
	.long	.L138-.L122
	.long	.L137-.L122
	.long	.L136-.L122
	.long	.L135-.L122
	.long	.L178-.L122
	.long	.L134-.L122
	.long	.L178-.L122
	.long	.L178-.L122
	.long	.L178-.L122
	.long	.L178-.L122
	.long	.L133-.L122
	.long	.L132-.L122
	.long	.L178-.L122
	.long	.L131-.L122
	.long	.L130-.L122
	.long	.L178-.L122
	.long	.L129-.L122
	.long	.L178-.L122
	.long	.L128-.L122
	.long	.L127-.L122
	.long	.L178-.L122
	.long	.L126-.L122
	.long	.L125-.L122
	.long	.L124-.L122
	.long	.L123-.L122
	.long	.L178-.L122
	.long	.L121-.L122
	.text
.L140:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L176
	jmp	.L177
.L136:
	movq	-856(%rbp), %rdx
	movq	-864(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	changeToParentDir
	movq	%rax, -864(%rbp)
	movq	$24, -792(%rbp)
	jmp	.L154
.L121:
	cmpl	$255, -828(%rbp)
	jbe	.L155
	movq	$46, -792(%rbp)
	jmp	.L154
.L155:
	movq	$8, -792(%rbp)
	jmp	.L154
.L151:
	leaq	-528(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$24, -792(%rbp)
	jmp	.L154
.L144:
	cmpq	$0, -800(%rbp)
	je	.L157
	movq	$44, -792(%rbp)
	jmp	.L154
.L157:
	movq	$45, -792(%rbp)
	jmp	.L154
.L143:
	movl	-832(%rbp), %eax
	movb	$0, -528(%rbp,%rax)
	addl	$1, -832(%rbp)
	movq	$33, -792(%rbp)
	jmp	.L154
.L148:
	movl	-828(%rbp), %eax
	movb	$0, -272(%rbp,%rax)
	addl	$1, -828(%rbp)
	movq	$49, -792(%rbp)
	jmp	.L154
.L125:
	movl	$62, %edi
	call	putchar@PLT
	movq	stdin(%rip), %rdx
	leaq	-784(%rbp), %rax
	movl	$256, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	leaq	-272(%rbp), %rcx
	leaq	-528(%rbp), %rdx
	leaq	-784(%rbp), %rax
	leaq	.LC2(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_sscanf@PLT
	leaq	-528(%rbp), %rax
	leaq	.LC3(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -808(%rbp)
	movq	$36, -792(%rbp)
	jmp	.L154
.L138:
	leaq	-272(%rbp), %rdx
	movq	-864(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	makeDir
	movq	$24, -792(%rbp)
	jmp	.L154
.L142:
	movb	$0, -528(%rbp)
	movl	$1, -832(%rbp)
	movq	$33, -792(%rbp)
	jmp	.L154
.L137:
	movq	-856(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, -800(%rbp)
	movl	$10, %edi
	call	putchar@PLT
	leaq	-784(%rbp), %rax
	leaq	.LC4(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -804(%rbp)
	movq	$39, -792(%rbp)
	jmp	.L154
.L139:
	cmpl	$255, -836(%rbp)
	jbe	.L159
	movq	$16, -792(%rbp)
	jmp	.L154
.L159:
	movq	$5, -792(%rbp)
	jmp	.L154
.L131:
	cmpl	$0, -808(%rbp)
	jne	.L161
	movq	$23, -792(%rbp)
	jmp	.L154
.L161:
	movq	$26, -792(%rbp)
	jmp	.L154
.L135:
	leaq	-528(%rbp), %rax
	leaq	.LC5(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -812(%rbp)
	movq	$0, -792(%rbp)
	jmp	.L154
.L146:
	leaq	-528(%rbp), %rax
	leaq	.LC6(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -820(%rbp)
	movq	$37, -792(%rbp)
	jmp	.L154
.L145:
	movb	$0, -784(%rbp)
	movl	$1, -836(%rbp)
	movq	$21, -792(%rbp)
	jmp	.L154
.L141:
	movq	-864(%rbp), %rax
	movq	%rax, %rdi
	call	printDir
	movq	$24, -792(%rbp)
	jmp	.L154
.L149:
	leaq	-528(%rbp), %rax
	leaq	.LC7(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -816(%rbp)
	movq	$47, -792(%rbp)
	jmp	.L154
.L132:
	movb	$0, -272(%rbp)
	movl	$1, -828(%rbp)
	movq	$49, -792(%rbp)
	jmp	.L154
.L134:
	movq	-856(%rbp), %rdx
	leaq	-272(%rbp), %rcx
	movq	-864(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	changeDir
	movq	%rax, -864(%rbp)
	movq	$24, -792(%rbp)
	jmp	.L154
.L123:
	cmpl	$0, -816(%rbp)
	jne	.L163
	movq	$28, -792(%rbp)
	jmp	.L154
.L163:
	movq	$11, -792(%rbp)
	jmp	.L154
.L126:
	movq	-800(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, %rsi
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-800(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, -800(%rbp)
	movq	$14, -792(%rbp)
	jmp	.L154
.L150:
	movl	-836(%rbp), %eax
	movb	$0, -784(%rbp,%rax)
	addl	$1, -836(%rbp)
	movq	$21, -792(%rbp)
	jmp	.L154
.L133:
	cmpl	$255, -832(%rbp)
	jbe	.L165
	movq	$34, -792(%rbp)
	jmp	.L154
.L165:
	movq	$15, -792(%rbp)
	jmp	.L154
.L130:
	cmpl	$0, -820(%rbp)
	jne	.L167
	movq	$17, -792(%rbp)
	jmp	.L154
.L167:
	movq	$10, -792(%rbp)
	jmp	.L154
.L128:
	movq	$13, -792(%rbp)
	jmp	.L154
.L147:
	leaq	-528(%rbp), %rax
	leaq	.LC4(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -824(%rbp)
	movq	$42, -792(%rbp)
	jmp	.L154
.L127:
	cmpl	$0, -824(%rbp)
	jne	.L169
	movq	$18, -792(%rbp)
	jmp	.L154
.L169:
	movq	$4, -792(%rbp)
	jmp	.L154
.L152:
	cmpl	$0, -812(%rbp)
	jne	.L171
	movq	$25, -792(%rbp)
	jmp	.L154
.L171:
	movq	$6, -792(%rbp)
	jmp	.L154
.L124:
	movq	-856(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, -800(%rbp)
	movq	$14, -792(%rbp)
	jmp	.L154
.L129:
	cmpl	$0, -804(%rbp)
	je	.L173
	movq	$14, -792(%rbp)
	jmp	.L154
.L173:
	movq	$18, -792(%rbp)
	jmp	.L154
.L178:
	nop
.L154:
	jmp	.L175
.L177:
	call	__stack_chk_fail@PLT
.L176:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE11:
	.size	terminal, .-terminal
	.section	.rodata
	.align 8
.LC9:
	.string	"\nThe system cannot find the path specified."
	.text
	.globl	changeDir
	.type	changeDir, @function
changeDir:
.LFB12:
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
	movq	$15, -8(%rbp)
.L205:
	cmpq	$15, -8(%rbp)
	ja	.L206
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L182(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L182(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L182:
	.long	.L206-.L182
	.long	.L194-.L182
	.long	.L193-.L182
	.long	.L192-.L182
	.long	.L191-.L182
	.long	.L206-.L182
	.long	.L190-.L182
	.long	.L189-.L182
	.long	.L188-.L182
	.long	.L187-.L182
	.long	.L186-.L182
	.long	.L185-.L182
	.long	.L206-.L182
	.long	.L184-.L182
	.long	.L183-.L182
	.long	.L181-.L182
	.text
.L191:
	movq	-16(%rbp), %rdx
	movq	-56(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	Push
	movq	$6, -8(%rbp)
	jmp	.L195
.L183:
	cmpl	$0, -20(%rbp)
	je	.L196
	movq	$8, -8(%rbp)
	jmp	.L195
.L196:
	movq	$13, -8(%rbp)
	jmp	.L195
.L181:
	movq	-40(%rbp), %rax
	movq	264(%rax), %rax
	movq	%rax, -16(%rbp)
	movq	$11, -8(%rbp)
	jmp	.L195
.L188:
	movq	-16(%rbp), %rax
	movq	256(%rax), %rax
	movq	%rax, -16(%rbp)
	movq	$9, -8(%rbp)
	jmp	.L195
.L194:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$10, -8(%rbp)
	jmp	.L195
.L192:
	movq	-40(%rbp), %rax
	jmp	.L198
.L185:
	movq	-40(%rbp), %rax
	movq	264(%rax), %rax
	testq	%rax, %rax
	jne	.L199
	movq	$7, -8(%rbp)
	jmp	.L195
.L199:
	movq	$9, -8(%rbp)
	jmp	.L195
.L187:
	cmpq	$0, -16(%rbp)
	je	.L201
	movq	$2, -8(%rbp)
	jmp	.L195
.L201:
	movq	$13, -8(%rbp)
	jmp	.L195
.L184:
	cmpq	$0, -16(%rbp)
	jne	.L203
	movq	$1, -8(%rbp)
	jmp	.L195
.L203:
	movq	$4, -8(%rbp)
	jmp	.L195
.L190:
	movq	-16(%rbp), %rax
	jmp	.L198
.L186:
	movq	-40(%rbp), %rax
	jmp	.L198
.L189:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$3, -8(%rbp)
	jmp	.L195
.L193:
	movq	-16(%rbp), %rax
	movq	-48(%rbp), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -20(%rbp)
	movq	$14, -8(%rbp)
	jmp	.L195
.L206:
	nop
.L195:
	jmp	.L205
.L198:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE12:
	.size	changeDir, .-changeDir
	.globl	changeToParentDir
	.type	changeToParentDir, @function
changeToParentDir:
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
	movq	%rsi, -32(%rbp)
	movq	$2, -8(%rbp)
.L213:
	cmpq	$2, -8(%rbp)
	je	.L208
	cmpq	$2, -8(%rbp)
	ja	.L215
	cmpq	$0, -8(%rbp)
	je	.L210
	cmpq	$1, -8(%rbp)
	jne	.L215
	movq	-24(%rbp), %rax
	jmp	.L214
.L210:
	movq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	Pop
	movq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	Pop
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rdx
	movq	-32(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	Push
	movq	$1, -8(%rbp)
	jmp	.L212
.L208:
	movq	$0, -8(%rbp)
	jmp	.L212
.L215:
	nop
.L212:
	jmp	.L213
.L214:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE13:
	.size	changeToParentDir, .-changeToParentDir
	.section	.rodata
.LC10:
	.string	"Directory is empty!"
.LC11:
	.string	"\t%s\n"
	.text
	.globl	printDir
	.type	printDir, @function
printDir:
.LFB14:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	$4, -8(%rbp)
.L231:
	cmpq	$8, -8(%rbp)
	ja	.L233
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L219(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L219(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L219:
	.long	.L224-.L219
	.long	.L233-.L219
	.long	.L233-.L219
	.long	.L223-.L219
	.long	.L222-.L219
	.long	.L233-.L219
	.long	.L221-.L219
	.long	.L220-.L219
	.long	.L218-.L219
	.text
.L222:
	movq	-24(%rbp), %rax
	movq	264(%rax), %rax
	testq	%rax, %rax
	jne	.L225
	movq	$0, -8(%rbp)
	jmp	.L227
.L225:
	movq	$6, -8(%rbp)
	jmp	.L227
.L218:
	movl	$0, %eax
	jmp	.L232
.L223:
	cmpq	$0, -16(%rbp)
	je	.L229
	movq	$7, -8(%rbp)
	jmp	.L227
.L229:
	movq	$8, -8(%rbp)
	jmp	.L227
.L221:
	movq	-24(%rbp), %rax
	movq	264(%rax), %rax
	movq	%rax, -16(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L227
.L224:
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$8, -8(%rbp)
	jmp	.L227
.L220:
	movq	-16(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-16(%rbp), %rax
	movq	256(%rax), %rax
	movq	%rax, -16(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L227
.L233:
	nop
.L227:
	jmp	.L231
.L232:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE14:
	.size	printDir, .-printDir
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
