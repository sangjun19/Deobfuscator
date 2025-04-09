	.file	"timber-they_AdventOfCode22_9_flatten.c"
	.text
	.globl	_TIG_IZ_zP1K_argc
	.bss
	.align 4
	.type	_TIG_IZ_zP1K_argc, @object
	.size	_TIG_IZ_zP1K_argc, 4
_TIG_IZ_zP1K_argc:
	.zero	4
	.globl	_TIG_IZ_zP1K_argv
	.align 8
	.type	_TIG_IZ_zP1K_argv, @object
	.size	_TIG_IZ_zP1K_argv, 8
_TIG_IZ_zP1K_argv:
	.zero	8
	.globl	_TIG_IZ_zP1K_envp
	.align 8
	.type	_TIG_IZ_zP1K_envp, @object
	.size	_TIG_IZ_zP1K_envp, 8
_TIG_IZ_zP1K_envp:
	.zero	8
	.text
	.globl	enterPosition
	.type	enterPosition, @function
enterPosition:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -40(%rbp)
	movl	%esi, -44(%rbp)
	movl	%edx, -48(%rbp)
	movl	%ecx, -52(%rbp)
	movq	$9, -8(%rbp)
.L20:
	cmpq	$10, -8(%rbp)
	ja	.L21
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
	.long	.L21-.L4
	.long	.L21-.L4
	.long	.L11-.L4
	.long	.L10-.L4
	.long	.L9-.L4
	.long	.L21-.L4
	.long	.L8-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L9:
	movl	-44(%rbp), %eax
	jmp	.L12
.L6:
	addl	$2, -20(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L13
.L10:
	movl	-44(%rbp), %eax
	jmp	.L12
.L5:
	movl	$0, -20(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L13
.L8:
	movl	-20(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	%eax, -48(%rbp)
	jne	.L14
	movq	$10, -8(%rbp)
	jmp	.L13
.L14:
	movq	$8, -8(%rbp)
	jmp	.L13
.L3:
	movl	-20(%rbp), %eax
	cltq
	addq	$1, %rax
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	%eax, -52(%rbp)
	jne	.L16
	movq	$4, -8(%rbp)
	jmp	.L13
.L16:
	movq	$8, -8(%rbp)
	jmp	.L13
.L7:
	movl	-44(%rbp), %eax
	subl	$1, %eax
	cmpl	%eax, -20(%rbp)
	jge	.L18
	movq	$6, -8(%rbp)
	jmp	.L13
.L18:
	movq	$2, -8(%rbp)
	jmp	.L13
.L11:
	movl	-44(%rbp), %eax
	movl	%eax, -16(%rbp)
	addl	$1, -44(%rbp)
	movl	-16(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rax, %rdx
	movl	-48(%rbp), %eax
	movl	%eax, (%rdx)
	movl	-44(%rbp), %eax
	movl	%eax, -12(%rbp)
	addl	$1, -44(%rbp)
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rax, %rdx
	movl	-52(%rbp), %eax
	movl	%eax, (%rdx)
	movq	$3, -8(%rbp)
	jmp	.L13
.L21:
	nop
.L13:
	jmp	.L20
.L12:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	enterPosition, .-enterPosition
	.section	.rodata
.LC0:
	.string	"%c %d\n"
	.text
	.globl	part2
	.type	part2, @function
part2:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	leaq	-397312(%rsp), %r11
.LPSRL0:
	subq	$4096, %rsp
	orq	$0, (%rsp)
	cmpq	%r11, %rsp
	jne	.LPSRL0
	subq	$2848, %rsp
	movq	%rdi, -400152(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$23, -400104(%rbp)
.L48:
	cmpq	$23, -400104(%rbp)
	ja	.L51
	movq	-400104(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L25(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L25(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L25:
	.long	.L37-.L25
	.long	.L51-.L25
	.long	.L36-.L25
	.long	.L35-.L25
	.long	.L34-.L25
	.long	.L33-.L25
	.long	.L51-.L25
	.long	.L51-.L25
	.long	.L32-.L25
	.long	.L31-.L25
	.long	.L51-.L25
	.long	.L51-.L25
	.long	.L51-.L25
	.long	.L51-.L25
	.long	.L30-.L25
	.long	.L51-.L25
	.long	.L29-.L25
	.long	.L51-.L25
	.long	.L28-.L25
	.long	.L27-.L25
	.long	.L26-.L25
	.long	.L51-.L25
	.long	.L51-.L25
	.long	.L24-.L25
	.text
.L28:
	leaq	-400096(%rbp), %rax
	addq	$4, %rax
	movzbl	-400129(%rbp), %edx
	movsbl	%dl, %ecx
	leaq	-400096(%rbp), %rsi
	movq	%rax, %rdx
	movl	%ecx, %edi
	call	move
	movl	$2, -400112(%rbp)
	movq	$3, -400104(%rbp)
	jmp	.L38
.L34:
	movl	$0, -400016(%rbp)
	movl	$0, -400012(%rbp)
	movl	$2, -400124(%rbp)
	movl	$0, -400096(%rbp)
	movl	$1, -400120(%rbp)
	movq	$19, -400104(%rbp)
	jmp	.L38
.L30:
	cmpl	$2, -400108(%rbp)
	jne	.L39
	movq	$2, -400104(%rbp)
	jmp	.L38
.L39:
	movq	$0, -400104(%rbp)
	jmp	.L38
.L32:
	movl	-400020(%rbp), %ecx
	movl	-400024(%rbp), %edx
	movl	-400124(%rbp), %esi
	leaq	-400016(%rbp), %rax
	movq	%rax, %rdi
	call	enterPosition
	movl	%eax, -400124(%rbp)
	addl	$1, -400116(%rbp)
	movq	$16, -400104(%rbp)
	jmp	.L38
.L24:
	movq	$4, -400104(%rbp)
	jmp	.L38
.L35:
	cmpl	$18, -400112(%rbp)
	jg	.L41
	movq	$5, -400104(%rbp)
	jmp	.L38
.L41:
	movq	$8, -400104(%rbp)
	jmp	.L38
.L29:
	movl	-400128(%rbp), %eax
	cmpl	%eax, -400116(%rbp)
	jge	.L43
	movq	$18, -400104(%rbp)
	jmp	.L38
.L43:
	movq	$9, -400104(%rbp)
	jmp	.L38
.L31:
	leaq	-400128(%rbp), %rcx
	leaq	-400129(%rbp), %rdx
	movq	-400152(%rbp), %rax
	leaq	.LC0(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_fscanf@PLT
	movl	%eax, -400108(%rbp)
	movq	$14, -400104(%rbp)
	jmp	.L38
.L27:
	cmpl	$19, -400120(%rbp)
	jbe	.L45
	movq	$9, -400104(%rbp)
	jmp	.L38
.L45:
	movq	$20, -400104(%rbp)
	jmp	.L38
.L33:
	movl	-400112(%rbp), %eax
	cltq
	addq	$1, %rax
	leaq	0(,%rax,4), %rdx
	leaq	-400096(%rbp), %rax
	leaq	(%rax,%rdx), %rcx
	movl	-400112(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	-400096(%rbp), %rax
	addq	%rax, %rdx
	movl	-400112(%rbp), %eax
	subl	$1, %eax
	cltq
	movl	-400096(%rbp,%rax,4), %esi
	movl	-400112(%rbp), %eax
	subl	$2, %eax
	cltq
	movl	-400096(%rbp,%rax,4), %eax
	movl	%eax, %edi
	call	adjustTail
	addl	$2, -400112(%rbp)
	movq	$3, -400104(%rbp)
	jmp	.L38
.L37:
	movl	-400124(%rbp), %eax
	movl	%eax, %edx
	shrl	$31, %edx
	addl	%edx, %eax
	sarl	%eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L49
	jmp	.L50
.L36:
	movl	$0, -400116(%rbp)
	movq	$16, -400104(%rbp)
	jmp	.L38
.L26:
	movl	-400120(%rbp), %eax
	movl	$0, -400096(%rbp,%rax,4)
	addl	$1, -400120(%rbp)
	movq	$19, -400104(%rbp)
	jmp	.L38
.L51:
	nop
.L38:
	jmp	.L48
.L50:
	call	__stack_chk_fail@PLT
.L49:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	part2, .-part2
	.section	.rodata
.LC1:
	.string	"Lost tail!\n"
	.text
	.globl	adjustTail
	.type	adjustTail, @function
adjustTail:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movl	%esi, -40(%rbp)
	movq	%rdx, -48(%rbp)
	movq	%rcx, -56(%rbp)
	movq	$14, -8(%rbp)
.L77:
	cmpq	$14, -8(%rbp)
	ja	.L78
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L55(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L55(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L55:
	.long	.L78-.L55
	.long	.L79-.L55
	.long	.L65-.L55
	.long	.L64-.L55
	.long	.L63-.L55
	.long	.L62-.L55
	.long	.L61-.L55
	.long	.L60-.L55
	.long	.L79-.L55
	.long	.L58-.L55
	.long	.L57-.L55
	.long	.L78-.L55
	.long	.L78-.L55
	.long	.L56-.L55
	.long	.L54-.L55
	.text
.L63:
	movq	-56(%rbp), %rax
	movl	(%rax), %edx
	movl	-40(%rbp), %eax
	subl	%edx, %eax
	movl	%eax, %edx
	negl	%edx
	cmovns	%edx, %eax
	movl	%eax, -28(%rbp)
	movq	$9, -8(%rbp)
	jmp	.L67
.L54:
	movq	-48(%rbp), %rax
	movl	(%rax), %edx
	movl	-36(%rbp), %eax
	subl	%edx, %eax
	movl	%eax, %edx
	negl	%edx
	cmovns	%edx, %eax
	movl	%eax, -32(%rbp)
	movq	$10, -8(%rbp)
	jmp	.L67
.L64:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$11, %edx
	movl	$1, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movl	$1, %edi
	call	exit@PLT
.L58:
	cmpl	$1, -28(%rbp)
	jg	.L69
	movq	$8, -8(%rbp)
	jmp	.L67
.L69:
	movq	$5, -8(%rbp)
	jmp	.L67
.L56:
	cmpl	$1, -20(%rbp)
	jle	.L71
	movq	$7, -8(%rbp)
	jmp	.L67
.L71:
	movq	$1, -8(%rbp)
	jmp	.L67
.L61:
	cmpl	$1, -24(%rbp)
	jle	.L73
	movq	$3, -8(%rbp)
	jmp	.L67
.L73:
	movq	$2, -8(%rbp)
	jmp	.L67
.L62:
	movq	-48(%rbp), %rax
	movl	(%rax), %edx
	movl	-36(%rbp), %eax
	subl	%edx, %eax
	movl	%eax, %edi
	call	sgn
	movl	%eax, -16(%rbp)
	movq	-48(%rbp), %rax
	movl	(%rax), %edx
	movl	-16(%rbp), %eax
	addl	%eax, %edx
	movq	-48(%rbp), %rax
	movl	%edx, (%rax)
	movq	-56(%rbp), %rax
	movl	(%rax), %edx
	movl	-40(%rbp), %eax
	subl	%edx, %eax
	movl	%eax, %edi
	call	sgn
	movl	%eax, -12(%rbp)
	movq	-56(%rbp), %rax
	movl	(%rax), %edx
	movl	-12(%rbp), %eax
	addl	%eax, %edx
	movq	-56(%rbp), %rax
	movl	%edx, (%rax)
	movq	-48(%rbp), %rax
	movl	(%rax), %edx
	movl	-36(%rbp), %eax
	subl	%edx, %eax
	movl	%eax, %edx
	negl	%edx
	cmovns	%edx, %eax
	movl	%eax, -24(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L67
.L57:
	cmpl	$1, -32(%rbp)
	jg	.L75
	movq	$4, -8(%rbp)
	jmp	.L67
.L75:
	movq	$5, -8(%rbp)
	jmp	.L67
.L60:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$11, %edx
	movl	$1, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movl	$1, %edi
	call	exit@PLT
.L65:
	movq	-56(%rbp), %rax
	movl	(%rax), %edx
	movl	-40(%rbp), %eax
	subl	%edx, %eax
	movl	%eax, %edx
	negl	%edx
	cmovns	%edx, %eax
	movl	%eax, -20(%rbp)
	movq	$13, -8(%rbp)
	jmp	.L67
.L78:
	nop
.L67:
	jmp	.L77
.L79:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	adjustTail, .-adjustTail
	.globl	move
	.type	move, @function
move:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, %eax
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movb	%al, -20(%rbp)
	movq	$5, -8(%rbp)
.L98:
	cmpq	$8, -8(%rbp)
	ja	.L99
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L83(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L83(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L83:
	.long	.L89-.L83
	.long	.L100-.L83
	.long	.L99-.L83
	.long	.L87-.L83
	.long	.L86-.L83
	.long	.L85-.L83
	.long	.L99-.L83
	.long	.L84-.L83
	.long	.L82-.L83
	.text
.L86:
	movq	-40(%rbp), %rax
	movl	(%rax), %eax
	leal	1(%rax), %edx
	movq	-40(%rbp), %rax
	movl	%edx, (%rax)
	movq	$1, -8(%rbp)
	jmp	.L90
.L82:
	movq	-32(%rbp), %rax
	movl	(%rax), %eax
	leal	-1(%rax), %edx
	movq	-32(%rbp), %rax
	movl	%edx, (%rax)
	movq	$1, -8(%rbp)
	jmp	.L90
.L87:
	movq	$1, -8(%rbp)
	jmp	.L90
.L85:
	movsbl	-20(%rbp), %eax
	cmpl	$85, %eax
	je	.L92
	cmpl	$85, %eax
	jg	.L93
	cmpl	$82, %eax
	je	.L94
	cmpl	$82, %eax
	jg	.L93
	cmpl	$68, %eax
	je	.L95
	cmpl	$76, %eax
	je	.L96
	jmp	.L93
.L92:
	movq	$0, -8(%rbp)
	jmp	.L97
.L96:
	movq	$8, -8(%rbp)
	jmp	.L97
.L95:
	movq	$4, -8(%rbp)
	jmp	.L97
.L94:
	movq	$7, -8(%rbp)
	jmp	.L97
.L93:
	movq	$3, -8(%rbp)
	nop
.L97:
	jmp	.L90
.L89:
	movq	-40(%rbp), %rax
	movl	(%rax), %eax
	leal	-1(%rax), %edx
	movq	-40(%rbp), %rax
	movl	%edx, (%rax)
	movq	$1, -8(%rbp)
	jmp	.L90
.L84:
	movq	-32(%rbp), %rax
	movl	(%rax), %eax
	leal	1(%rax), %edx
	movq	-32(%rbp), %rax
	movl	%edx, (%rax)
	movq	$1, -8(%rbp)
	jmp	.L90
.L99:
	nop
.L90:
	jmp	.L98
.L100:
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	move, .-move
	.globl	part1
	.type	part1, @function
part1:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	leaq	-397312(%rsp), %r11
.LPSRL1:
	subq	$4096, %rsp
	orq	$0, (%rsp)
	cmpq	%r11, %rsp
	jne	.LPSRL1
	subq	$2768, %rsp
	movq	%rdi, -400072(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$10, -400024(%rbp)
.L118:
	cmpq	$12, -400024(%rbp)
	ja	.L121
	movq	-400024(%rbp), %rax
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
	.long	.L111-.L104
	.long	.L110-.L104
	.long	.L121-.L104
	.long	.L109-.L104
	.long	.L121-.L104
	.long	.L108-.L104
	.long	.L121-.L104
	.long	.L121-.L104
	.long	.L121-.L104
	.long	.L107-.L104
	.long	.L106-.L104
	.long	.L105-.L104
	.long	.L103-.L104
	.text
.L103:
	movzbl	-400057(%rbp), %eax
	movsbl	%al, %eax
	leaq	-400048(%rbp), %rdx
	leaq	-400052(%rbp), %rcx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	move
	movl	-400048(%rbp), %esi
	movl	-400052(%rbp), %eax
	leaq	-400040(%rbp), %rcx
	leaq	-400044(%rbp), %rdx
	movl	%eax, %edi
	call	adjustTail
	movl	-400040(%rbp), %ecx
	movl	-400044(%rbp), %edx
	movl	-400036(%rbp), %esi
	leaq	-400016(%rbp), %rax
	movq	%rax, %rdi
	call	enterPosition
	movl	%eax, -400036(%rbp)
	addl	$1, -400032(%rbp)
	movq	$5, -400024(%rbp)
	jmp	.L112
.L110:
	movl	$0, -400016(%rbp)
	movl	$0, -400012(%rbp)
	movl	$2, -400036(%rbp)
	movl	$0, -400052(%rbp)
	movl	$0, -400048(%rbp)
	movl	$0, -400044(%rbp)
	movl	$0, -400040(%rbp)
	movq	$11, -400024(%rbp)
	jmp	.L112
.L109:
	movl	-400036(%rbp), %eax
	movl	%eax, %edx
	shrl	$31, %edx
	addl	%edx, %eax
	sarl	%eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L119
	jmp	.L120
.L105:
	leaq	-400056(%rbp), %rcx
	leaq	-400057(%rbp), %rdx
	movq	-400072(%rbp), %rax
	leaq	.LC0(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_fscanf@PLT
	movl	%eax, -400028(%rbp)
	movq	$9, -400024(%rbp)
	jmp	.L112
.L107:
	cmpl	$2, -400028(%rbp)
	jne	.L114
	movq	$0, -400024(%rbp)
	jmp	.L112
.L114:
	movq	$3, -400024(%rbp)
	jmp	.L112
.L108:
	movl	-400056(%rbp), %eax
	cmpl	%eax, -400032(%rbp)
	jge	.L116
	movq	$12, -400024(%rbp)
	jmp	.L112
.L116:
	movq	$11, -400024(%rbp)
	jmp	.L112
.L106:
	movq	$1, -400024(%rbp)
	jmp	.L112
.L111:
	movl	$0, -400032(%rbp)
	movq	$5, -400024(%rbp)
	jmp	.L112
.L121:
	nop
.L112:
	jmp	.L118
.L120:
	call	__stack_chk_fail@PLT
.L119:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	part1, .-part1
	.globl	abs
	.type	abs, @function
abs:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -20(%rbp)
	movq	$0, -8(%rbp)
.L131:
	cmpq	$3, -8(%rbp)
	je	.L123
	cmpq	$3, -8(%rbp)
	ja	.L133
	cmpq	$2, -8(%rbp)
	je	.L125
	cmpq	$2, -8(%rbp)
	ja	.L133
	cmpq	$0, -8(%rbp)
	je	.L126
	cmpq	$1, -8(%rbp)
	jne	.L133
	movl	-20(%rbp), %eax
	movl	%eax, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L127
.L123:
	movl	-20(%rbp), %eax
	negl	%eax
	movl	%eax, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L127
.L126:
	cmpl	$0, -20(%rbp)
	jns	.L128
	movq	$3, -8(%rbp)
	jmp	.L127
.L128:
	movq	$1, -8(%rbp)
	jmp	.L127
.L125:
	movl	-12(%rbp), %eax
	jmp	.L132
.L133:
	nop
.L127:
	jmp	.L131
.L132:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	abs, .-abs
	.section	.rodata
.LC2:
	.string	"r"
.LC3:
	.string	"in9"
.LC4:
	.string	"Part1: %d\n"
.LC5:
	.string	"Part2: %d\n"
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
	movq	$0, _TIG_IZ_zP1K_envp(%rip)
	nop
.L135:
	movq	$0, _TIG_IZ_zP1K_argv(%rip)
	nop
.L136:
	movl	$0, _TIG_IZ_zP1K_argc(%rip)
	nop
	nop
.L137:
.L138:
#APP
# 192 "timber-they_AdventOfCode22_9.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-zP1K--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_zP1K_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_zP1K_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_zP1K_envp(%rip)
	nop
	movq	$2, -24(%rbp)
.L144:
	cmpq	$2, -24(%rbp)
	je	.L139
	cmpq	$2, -24(%rbp)
	ja	.L146
	cmpq	$0, -24(%rbp)
	je	.L141
	cmpq	$1, -24(%rbp)
	jne	.L146
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
	movq	$0, -24(%rbp)
	jmp	.L142
.L141:
	movl	$0, %eax
	jmp	.L145
.L139:
	movq	$1, -24(%rbp)
	jmp	.L142
.L146:
	nop
.L142:
	jmp	.L144
.L145:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	main, .-main
	.globl	sgn
	.type	sgn, @function
sgn:
.LFB12:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -20(%rbp)
	movq	$5, -8(%rbp)
.L163:
	cmpq	$6, -8(%rbp)
	ja	.L165
	movq	-8(%rbp), %rax
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
	.long	.L156-.L150
	.long	.L155-.L150
	.long	.L154-.L150
	.long	.L153-.L150
	.long	.L152-.L150
	.long	.L151-.L150
	.long	.L149-.L150
	.text
.L152:
	movl	$1, -16(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L157
.L155:
	movl	-16(%rbp), %eax
	movl	%eax, -12(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L157
.L153:
	cmpl	$0, -20(%rbp)
	jle	.L158
	movq	$4, -8(%rbp)
	jmp	.L157
.L158:
	movq	$6, -8(%rbp)
	jmp	.L157
.L149:
	movl	$0, -16(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L157
.L151:
	cmpl	$0, -20(%rbp)
	jns	.L160
	movq	$2, -8(%rbp)
	jmp	.L157
.L160:
	movq	$3, -8(%rbp)
	jmp	.L157
.L156:
	movl	-12(%rbp), %eax
	jmp	.L164
.L154:
	movl	$-1, -12(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L157
.L165:
	nop
.L157:
	jmp	.L163
.L164:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE12:
	.size	sgn, .-sgn
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
