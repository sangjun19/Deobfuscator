	.file	"Kyawkhaing444_C_exercise_main_flatten.c"
	.text
	.globl	_TIG_IZ_sPbh_argc
	.bss
	.align 4
	.type	_TIG_IZ_sPbh_argc, @object
	.size	_TIG_IZ_sPbh_argc, 4
_TIG_IZ_sPbh_argc:
	.zero	4
	.globl	_TIG_IZ_sPbh_envp
	.align 8
	.type	_TIG_IZ_sPbh_envp, @object
	.size	_TIG_IZ_sPbh_envp, 8
_TIG_IZ_sPbh_envp:
	.zero	8
	.globl	_TIG_IZ_sPbh_argv
	.align 8
	.type	_TIG_IZ_sPbh_argv, @object
	.size	_TIG_IZ_sPbh_argv, 8
_TIG_IZ_sPbh_argv:
	.zero	8
	.text
	.globl	lar
	.type	lar, @function
lar:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movl	%esi, -28(%rbp)
	movl	%edx, -32(%rbp)
	movq	$5, -8(%rbp)
.L17:
	cmpq	$8, -8(%rbp)
	ja	.L19
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
	.long	.L19-.L4
	.long	.L10-.L4
	.long	.L19-.L4
	.long	.L9-.L4
	.long	.L8-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L8:
	movl	-12(%rbp), %eax
	jmp	.L18
.L3:
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, -12(%rbp)
	movl	$0, -32(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L12
.L10:
	addl	$1, -32(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L12
.L9:
	movl	-32(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	%eax, -12(%rbp)
	jge	.L13
	movq	$6, -8(%rbp)
	jmp	.L12
.L13:
	movq	$1, -8(%rbp)
	jmp	.L12
.L6:
	movl	-32(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, -12(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L12
.L7:
	movq	$8, -8(%rbp)
	jmp	.L12
.L5:
	movl	-32(%rbp), %eax
	cmpl	-28(%rbp), %eax
	jge	.L15
	movq	$3, -8(%rbp)
	jmp	.L12
.L15:
	movq	$4, -8(%rbp)
	jmp	.L12
.L19:
	nop
.L12:
	jmp	.L17
.L18:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	lar, .-lar
	.section	.rodata
.LC0:
	.string	"%d\t"
	.text
	.globl	displayarrays
	.type	displayarrays, @function
displayarrays:
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
	movl	%esi, -28(%rbp)
	movl	%edx, -32(%rbp)
	movq	$1, -8(%rbp)
.L28:
	cmpq	$2, -8(%rbp)
	je	.L29
	cmpq	$2, -8(%rbp)
	ja	.L30
	cmpq	$0, -8(%rbp)
	je	.L23
	cmpq	$1, -8(%rbp)
	jne	.L30
	movl	-32(%rbp), %eax
	cmpl	-28(%rbp), %eax
	jge	.L24
	movq	$0, -8(%rbp)
	jmp	.L26
.L24:
	movq	$2, -8(%rbp)
	jmp	.L26
.L23:
	movl	-32(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-32(%rbp), %eax
	leal	1(%rax), %edx
	movl	-28(%rbp), %ecx
	movq	-24(%rbp), %rax
	movl	%ecx, %esi
	movq	%rax, %rdi
	call	displayarrays
	movq	$2, -8(%rbp)
	jmp	.L26
.L30:
	nop
.L26:
	jmp	.L28
.L29:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	displayarrays, .-displayarrays
	.section	.rodata
.LC1:
	.string	"A[%d]="
.LC2:
	.string	"%d"
	.text
	.globl	readarrays
	.type	readarrays, @function
readarrays:
.LFB2:
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
	movl	%edx, -32(%rbp)
	movq	$0, -8(%rbp)
.L40:
	cmpq	$3, -8(%rbp)
	je	.L32
	cmpq	$3, -8(%rbp)
	ja	.L41
	cmpq	$0, -8(%rbp)
	je	.L34
	cmpq	$2, -8(%rbp)
	je	.L42
	jmp	.L41
.L32:
	movl	-32(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-32(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-32(%rbp), %eax
	leal	1(%rax), %edx
	movl	-28(%rbp), %ecx
	movq	-24(%rbp), %rax
	movl	%ecx, %esi
	movq	%rax, %rdi
	call	readarrays
	movq	$2, -8(%rbp)
	jmp	.L36
.L34:
	movl	-32(%rbp), %eax
	cmpl	-28(%rbp), %eax
	jge	.L37
	movq	$3, -8(%rbp)
	jmp	.L36
.L37:
	movq	$2, -8(%rbp)
	jmp	.L36
.L41:
	nop
.L36:
	jmp	.L40
.L42:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	readarrays, .-readarrays
	.section	.rodata
.LC3:
	.string	"The smallest number is %d"
	.text
	.globl	sm
	.type	sm, @function
sm:
.LFB3:
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
	movl	%edx, -32(%rbp)
	movq	$8, -8(%rbp)
.L60:
	cmpq	$9, -8(%rbp)
	ja	.L61
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L46(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L46(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L46:
	.long	.L61-.L46
	.long	.L53-.L46
	.long	.L62-.L46
	.long	.L51-.L46
	.long	.L61-.L46
	.long	.L50-.L46
	.long	.L49-.L46
	.long	.L48-.L46
	.long	.L47-.L46
	.long	.L45-.L46
	.text
.L47:
	movq	$3, -8(%rbp)
	jmp	.L54
.L53:
	movl	-32(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	%eax, -12(%rbp)
	jle	.L55
	movq	$9, -8(%rbp)
	jmp	.L54
.L55:
	movq	$7, -8(%rbp)
	jmp	.L54
.L51:
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, -12(%rbp)
	movl	$0, -32(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L54
.L45:
	movl	-32(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, -12(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L54
.L49:
	movl	-32(%rbp), %eax
	cmpl	-28(%rbp), %eax
	jge	.L57
	movq	$1, -8(%rbp)
	jmp	.L54
.L57:
	movq	$5, -8(%rbp)
	jmp	.L54
.L50:
	movl	-12(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$2, -8(%rbp)
	jmp	.L54
.L48:
	addl	$1, -32(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L54
.L61:
	nop
.L54:
	jmp	.L60
.L62:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	sm, .-sm
	.section	.rodata
.LC4:
	.string	" %d \t"
.LC5:
	.string	"\nThe max number is %d"
.LC6:
	.string	"\n The mode number is %d\n"
.LC7:
	.string	"\n %d count %d"
	.text
	.globl	mode
	.type	mode, @function
mode:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$112, %rsp
	movq	%rdi, -88(%rbp)
	movl	%esi, -92(%rbp)
	movl	%edx, -96(%rbp)
	movl	%ecx, -100(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$16, -56(%rbp)
.L103:
	cmpq	$36, -56(%rbp)
	ja	.L106
	movq	-56(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L66(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L66(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L66:
	.long	.L106-.L66
	.long	.L86-.L66
	.long	.L85-.L66
	.long	.L106-.L66
	.long	.L107-.L66
	.long	.L83-.L66
	.long	.L106-.L66
	.long	.L106-.L66
	.long	.L106-.L66
	.long	.L106-.L66
	.long	.L106-.L66
	.long	.L106-.L66
	.long	.L82-.L66
	.long	.L81-.L66
	.long	.L106-.L66
	.long	.L106-.L66
	.long	.L80-.L66
	.long	.L79-.L66
	.long	.L78-.L66
	.long	.L77-.L66
	.long	.L106-.L66
	.long	.L76-.L66
	.long	.L75-.L66
	.long	.L74-.L66
	.long	.L106-.L66
	.long	.L73-.L66
	.long	.L106-.L66
	.long	.L72-.L66
	.long	.L71-.L66
	.long	.L106-.L66
	.long	.L106-.L66
	.long	.L106-.L66
	.long	.L70-.L66
	.long	.L69-.L66
	.long	.L68-.L66
	.long	.L67-.L66
	.long	.L65-.L66
	.text
.L78:
	movl	-48(%rbp), %eax
	movl	%eax, -64(%rbp)
	movl	$0, -96(%rbp)
	movq	$2, -56(%rbp)
	jmp	.L87
.L73:
	movl	-100(%rbp), %eax
	cmpl	-92(%rbp), %eax
	jge	.L88
	movq	$28, -56(%rbp)
	jmp	.L87
.L88:
	movq	$18, -56(%rbp)
	jmp	.L87
.L82:
	movl	-96(%rbp), %edx
	movl	-92(%rbp), %ecx
	movq	-88(%rbp), %rax
	movl	%ecx, %esi
	movq	%rax, %rdi
	call	lar
	movl	%eax, -60(%rbp)
	movl	-60(%rbp), %eax
	movl	%eax, -72(%rbp)
	movl	$0, -96(%rbp)
	movq	$22, -56(%rbp)
	jmp	.L87
.L86:
	addl	$1, -96(%rbp)
	movq	$2, -56(%rbp)
	jmp	.L87
.L74:
	movl	-96(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$13, -56(%rbp)
	jmp	.L87
.L80:
	movq	$12, -56(%rbp)
	jmp	.L87
.L76:
	movl	-96(%rbp), %eax
	cltq
	movl	-48(%rbp,%rax,4), %eax
	testl	%eax, %eax
	je	.L91
	movq	$23, -56(%rbp)
	jmp	.L87
.L91:
	movq	$13, -56(%rbp)
	jmp	.L87
.L65:
	movl	-96(%rbp), %eax
	cltq
	movl	-48(%rbp,%rax,4), %eax
	cmpl	%eax, -64(%rbp)
	jge	.L93
	movq	$27, -56(%rbp)
	jmp	.L87
.L93:
	movq	$1, -56(%rbp)
	jmp	.L87
.L81:
	addl	$1, -96(%rbp)
	movq	$35, -56(%rbp)
	jmp	.L87
.L77:
	movl	-96(%rbp), %eax
	cmpl	-72(%rbp), %eax
	jg	.L95
	movq	$33, -56(%rbp)
	jmp	.L87
.L95:
	movq	$17, -56(%rbp)
	jmp	.L87
.L70:
	movl	$0, -100(%rbp)
	movq	$25, -56(%rbp)
	jmp	.L87
.L79:
	movl	-68(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-64(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$0, -96(%rbp)
	movq	$35, -56(%rbp)
	jmp	.L87
.L72:
	movl	-96(%rbp), %eax
	cltq
	movl	-48(%rbp,%rax,4), %eax
	movl	%eax, -64(%rbp)
	movl	-96(%rbp), %eax
	movl	%eax, -68(%rbp)
	movq	$1, -56(%rbp)
	jmp	.L87
.L68:
	movl	-96(%rbp), %eax
	cltq
	movl	$0, -48(%rbp,%rax,4)
	addl	$1, -96(%rbp)
	movq	$22, -56(%rbp)
	jmp	.L87
.L75:
	movl	-96(%rbp), %eax
	cmpl	-72(%rbp), %eax
	jg	.L97
	movq	$34, -56(%rbp)
	jmp	.L87
.L97:
	movq	$32, -56(%rbp)
	jmp	.L87
.L71:
	movl	-100(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-88(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movslq	%eax, %rdx
	movl	-48(%rbp,%rdx,4), %edx
	addl	$1, %edx
	cltq
	movl	%edx, -48(%rbp,%rax,4)
	addl	$1, -100(%rbp)
	movq	$25, -56(%rbp)
	jmp	.L87
.L83:
	movl	$0, -96(%rbp)
	movq	$19, -56(%rbp)
	jmp	.L87
.L69:
	movl	-96(%rbp), %eax
	cltq
	movl	-48(%rbp,%rax,4), %edx
	movl	-96(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -96(%rbp)
	movq	$19, -56(%rbp)
	jmp	.L87
.L67:
	movl	-96(%rbp), %eax
	cmpl	-72(%rbp), %eax
	jg	.L99
	movq	$21, -56(%rbp)
	jmp	.L87
.L99:
	movq	$4, -56(%rbp)
	jmp	.L87
.L85:
	movl	-96(%rbp), %eax
	cmpl	-72(%rbp), %eax
	jg	.L101
	movq	$36, -56(%rbp)
	jmp	.L87
.L101:
	movq	$5, -56(%rbp)
	jmp	.L87
.L106:
	nop
.L87:
	jmp	.L103
.L107:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L105
	call	__stack_chk_fail@PLT
.L105:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	mode, .-mode
	.section	.rodata
	.align 8
.LC8:
	.string	"The second smallest number is %d"
	.text
	.globl	sesm
	.type	sesm, @function
sesm:
.LFB7:
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
	movl	%edx, -32(%rbp)
	movq	$15, -8(%rbp)
.L137:
	cmpq	$17, -8(%rbp)
	ja	.L138
	movq	-8(%rbp), %rax
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
	.long	.L124-.L111
	.long	.L123-.L111
	.long	.L138-.L111
	.long	.L122-.L111
	.long	.L121-.L111
	.long	.L120-.L111
	.long	.L138-.L111
	.long	.L119-.L111
	.long	.L118-.L111
	.long	.L139-.L111
	.long	.L116-.L111
	.long	.L115-.L111
	.long	.L138-.L111
	.long	.L138-.L111
	.long	.L114-.L111
	.long	.L113-.L111
	.long	.L112-.L111
	.long	.L110-.L111
	.text
.L121:
	addl	$1, -32(%rbp)
	movq	$16, -8(%rbp)
	jmp	.L125
.L114:
	addl	$1, -32(%rbp)
	movq	$10, -8(%rbp)
	jmp	.L125
.L113:
	movq	$1, -8(%rbp)
	jmp	.L125
.L118:
	movl	-12(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$9, -8(%rbp)
	jmp	.L125
.L123:
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, -16(%rbp)
	movl	$0, -32(%rbp)
	movq	$16, -8(%rbp)
	jmp	.L125
.L122:
	movl	-32(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, -16(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L125
.L112:
	movl	-32(%rbp), %eax
	cmpl	-28(%rbp), %eax
	jge	.L126
	movq	$17, -8(%rbp)
	jmp	.L125
.L126:
	movq	$5, -8(%rbp)
	jmp	.L125
.L115:
	movl	-32(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	%eax, -16(%rbp)
	je	.L128
	movq	$0, -8(%rbp)
	jmp	.L125
.L128:
	movq	$14, -8(%rbp)
	jmp	.L125
.L110:
	movl	-32(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	%eax, -16(%rbp)
	jle	.L131
	movq	$3, -8(%rbp)
	jmp	.L125
.L131:
	movq	$4, -8(%rbp)
	jmp	.L125
.L120:
	movq	-24(%rbp), %rax
	movl	4(%rax), %eax
	movl	%eax, -12(%rbp)
	movl	$0, -32(%rbp)
	movq	$10, -8(%rbp)
	jmp	.L125
.L116:
	movl	-32(%rbp), %eax
	cmpl	-28(%rbp), %eax
	jge	.L133
	movq	$11, -8(%rbp)
	jmp	.L125
.L133:
	movq	$8, -8(%rbp)
	jmp	.L125
.L124:
	movl	-32(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	%eax, -12(%rbp)
	jle	.L135
	movq	$7, -8(%rbp)
	jmp	.L125
.L135:
	movq	$14, -8(%rbp)
	jmp	.L125
.L119:
	movl	-32(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, -12(%rbp)
	movq	$14, -8(%rbp)
	jmp	.L125
.L138:
	nop
.L125:
	jmp	.L137
.L139:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	sesm, .-sesm
	.section	.rodata
.LC9:
	.string	"The element found at %d ."
	.text
	.globl	linearsearch
	.type	linearsearch, @function
linearsearch:
.LFB9:
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
	movl	%edx, -32(%rbp)
	movq	$3, -8(%rbp)
.L149:
	cmpq	$3, -8(%rbp)
	je	.L141
	cmpq	$3, -8(%rbp)
	ja	.L150
	cmpq	$1, -8(%rbp)
	je	.L151
	cmpq	$2, -8(%rbp)
	je	.L144
	jmp	.L150
.L141:
	movl	-28(%rbp), %eax
	cltq
	salq	$2, %rax
	leaq	-4(%rax), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	%eax, -32(%rbp)
	jne	.L146
	movq	$2, -8(%rbp)
	jmp	.L148
.L146:
	movq	$1, -8(%rbp)
	jmp	.L148
.L144:
	movl	-28(%rbp), %eax
	subl	$1, %eax
	movl	%eax, %esi
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-28(%rbp), %eax
	leal	-1(%rax), %ecx
	movl	-32(%rbp), %edx
	movq	-24(%rbp), %rax
	movl	%ecx, %esi
	movq	%rax, %rdi
	call	linearsearch
	movq	$1, -8(%rbp)
	jmp	.L148
.L150:
	nop
.L148:
	jmp	.L149
.L151:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	linearsearch, .-linearsearch
	.section	.rodata
.LC11:
	.string	"The mideum is :\t %d"
	.text
	.globl	mideum
	.type	mideum, @function
mideum:
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
	movl	%esi, -28(%rbp)
	movl	%edx, -32(%rbp)
	movq	$3, -8(%rbp)
.L161:
	cmpq	$3, -8(%rbp)
	je	.L153
	cmpq	$3, -8(%rbp)
	ja	.L163
	cmpq	$2, -8(%rbp)
	je	.L155
	cmpq	$2, -8(%rbp)
	ja	.L163
	cmpq	$0, -8(%rbp)
	je	.L156
	cmpq	$1, -8(%rbp)
	jne	.L163
	jmp	.L162
.L153:
	movl	-28(%rbp), %eax
	andl	$1, %eax
	testl	%eax, %eax
	jne	.L158
	movq	$0, -8(%rbp)
	jmp	.L160
.L158:
	movq	$2, -8(%rbp)
	jmp	.L160
.L156:
	movl	-28(%rbp), %eax
	movl	%eax, %edx
	shrl	$31, %edx
	addl	%edx, %eax
	sarl	%eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	movl	-28(%rbp), %eax
	movl	%eax, %ecx
	shrl	$31, %ecx
	addl	%ecx, %eax
	sarl	%eax
	cltq
	addq	$1, %rax
	leaq	0(,%rax,4), %rcx
	movq	-24(%rbp), %rax
	addq	%rcx, %rax
	movl	(%rax), %eax
	addl	%edx, %eax
	pxor	%xmm0, %xmm0
	cvtsi2ssl	%eax, %xmm0
	movss	.LC10(%rip), %xmm1
	divss	%xmm1, %xmm0
	pxor	%xmm2, %xmm2
	cvtss2sd	%xmm0, %xmm2
	movq	%xmm2, %rax
	movq	%rax, %xmm0
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$1, -8(%rbp)
	jmp	.L160
.L155:
	movl	-28(%rbp), %eax
	movl	%eax, %edx
	shrl	$31, %edx
	addl	%edx, %eax
	sarl	%eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	pxor	%xmm0, %xmm0
	cvtsi2ssl	%eax, %xmm0
	pxor	%xmm3, %xmm3
	cvtss2sd	%xmm0, %xmm3
	movq	%xmm3, %rax
	movq	%rax, %xmm0
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$1, -8(%rbp)
	jmp	.L160
.L163:
	nop
.L160:
	jmp	.L161
.L162:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	mideum, .-mideum
	.section	.rodata
.LC12:
	.string	"The element in arrays is :\t"
	.align 8
.LC13:
	.string	"\n************Main Menu*************"
.LC14:
	.string	"\n 1. Read the arrays"
.LC15:
	.string	"\n 2. Print the arrays"
.LC16:
	.string	"\n 3. Sort the arrays"
.LC17:
	.string	"\n 4. Binary search the arrays"
.LC18:
	.string	"\n 5. Linear search the arrays"
	.align 8
.LC19:
	.string	"\n 6. mean number in the arrays"
	.align 8
.LC20:
	.string	"\n 7. mideum number in the arrays"
	.align 8
.LC21:
	.string	"\n 8. Largest number in the arrays"
	.align 8
.LC22:
	.string	"\n 9. Mode number in the arrays"
	.align 8
.LC23:
	.string	"\n 10. Second largest number in the arrays"
	.align 8
.LC24:
	.string	"\n 11. Smallest number in the array"
	.align 8
.LC25:
	.string	"\n 12, Second Smallest number in the array"
.LC26:
	.string	"\n 13. Merging arrays"
.LC27:
	.string	"\nEnter the option:\t"
	.align 8
.LC28:
	.string	"\nEnter the element to search :\t"
.LC29:
	.string	"The largest number is  %d"
.LC30:
	.string	"Enter number of arrays:\t"
	.text
	.globl	main
	.type	main, @function
main:
.LFB11:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	subq	$384, %rsp
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	movl	%edi, -212(%rbp)
	movq	%rsi, -224(%rbp)
	movq	%rdx, -232(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -40(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_sPbh_envp(%rip)
	nop
.L165:
	movq	$0, _TIG_IZ_sPbh_argv(%rip)
	nop
.L166:
	movl	$0, _TIG_IZ_sPbh_argc(%rip)
	nop
	nop
.L167:
.L168:
#APP
# 200 "Kyawkhaing444_C_exercise_main.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-sPbh--0
# 0 "" 2
#NO_APP
	movl	-212(%rbp), %eax
	movl	%eax, _TIG_IZ_sPbh_argc(%rip)
	movq	-224(%rbp), %rax
	movq	%rax, _TIG_IZ_sPbh_argv(%rip)
	movq	-232(%rbp), %rax
	movq	%rax, _TIG_IZ_sPbh_envp(%rip)
	nop
	movq	$1, -160(%rbp)
.L215:
	cmpq	$41, -160(%rbp)
	ja	.L218
	movq	-160(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L171(%rip), %rax
	movl	(%rdx,%rax), %eax
	movslq	%eax, %rdx
	leaq	.L171(%rip), %rax
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L171:
	.long	.L191-.L171
	.long	.L190-.L171
	.long	.L218-.L171
	.long	.L218-.L171
	.long	.L218-.L171
	.long	.L218-.L171
	.long	.L218-.L171
	.long	.L189-.L171
	.long	.L218-.L171
	.long	.L218-.L171
	.long	.L188-.L171
	.long	.L218-.L171
	.long	.L187-.L171
	.long	.L218-.L171
	.long	.L186-.L171
	.long	.L218-.L171
	.long	.L218-.L171
	.long	.L185-.L171
	.long	.L218-.L171
	.long	.L184-.L171
	.long	.L183-.L171
	.long	.L182-.L171
	.long	.L181-.L171
	.long	.L180-.L171
	.long	.L179-.L171
	.long	.L218-.L171
	.long	.L218-.L171
	.long	.L218-.L171
	.long	.L218-.L171
	.long	.L178-.L171
	.long	.L218-.L171
	.long	.L177-.L171
	.long	.L176-.L171
	.long	.L175-.L171
	.long	.L218-.L171
	.long	.L174-.L171
	.long	.L173-.L171
	.long	.L218-.L171
	.long	.L218-.L171
	.long	.L172-.L171
	.long	.L218-.L171
	.long	.L170-.L171
	.text
.L186:
	movl	-196(%rbp), %esi
	movl	-196(%rbp), %edx
	movslq	%edx, %rax
	subq	$1, %rax
	movq	%rax, -72(%rbp)
	movslq	%edx, %rax
	movq	%rax, -256(%rbp)
	movq	$0, -248(%rbp)
	movl	-180(%rbp), %eax
	movq	-168(%rbp), %rcx
	movl	%eax, %edx
	movq	%rcx, %rdi
	call	selar
	movq	$21, -160(%rbp)
	jmp	.L192
.L177:
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-196(%rbp), %ecx
	movl	-196(%rbp), %edx
	movslq	%edx, %rax
	subq	$1, %rax
	movq	%rax, -120(%rbp)
	movslq	%edx, %rax
	movq	%rax, %r12
	movl	$0, %r13d
	movq	-168(%rbp), %rax
	movl	$0, %edx
	movl	%ecx, %esi
	movq	%rax, %rdi
	call	displayarrays
	movq	$21, -160(%rbp)
	jmp	.L192
.L187:
	movl	-196(%rbp), %esi
	movl	-196(%rbp), %edx
	movslq	%edx, %rax
	subq	$1, %rax
	movq	%rax, -64(%rbp)
	movslq	%edx, %rax
	movq	%rax, -272(%rbp)
	movq	$0, -264(%rbp)
	movl	-180(%rbp), %eax
	movq	-168(%rbp), %rcx
	movl	%eax, %edx
	movq	%rcx, %rdi
	call	mideum
	movq	$21, -160(%rbp)
	jmp	.L192
.L190:
	movq	$35, -160(%rbp)
	jmp	.L192
.L180:
	movl	-196(%rbp), %esi
	movl	-196(%rbp), %edx
	movslq	%edx, %rax
	subq	$1, %rax
	movq	%rax, -96(%rbp)
	movslq	%edx, %rax
	movq	%rax, -288(%rbp)
	movq	$0, -280(%rbp)
	movl	-180(%rbp), %eax
	movq	-168(%rbp), %rcx
	movl	%eax, %edx
	movq	%rcx, %rdi
	call	mean
	movq	$21, -160(%rbp)
	jmp	.L192
.L179:
	movl	-196(%rbp), %ecx
	movl	-196(%rbp), %edx
	movslq	%edx, %rax
	subq	$1, %rax
	movq	%rax, -104(%rbp)
	movslq	%edx, %rax
	movq	%rax, -304(%rbp)
	movq	$0, -296(%rbp)
	movl	-172(%rbp), %edx
	movl	-180(%rbp), %esi
	movq	-168(%rbp), %rax
	movq	%rax, %rdi
	call	marging
	movq	$21, -160(%rbp)
	jmp	.L192
.L182:
	movl	-184(%rbp), %eax
	cmpl	$13, %eax
	jg	.L193
	movq	$19, -160(%rbp)
	jmp	.L192
.L193:
	movq	$10, -160(%rbp)
	jmp	.L192
.L173:
	movl	-196(%rbp), %eax
	cltq
	salq	$5, %rax
	addq	$31, %rax
	shrq	$3, %rax
	movq	%rax, %rdx
	movabsq	$2305843009213693948, %rax
	andq	%rdx, %rax
	movq	%rax, -144(%rbp)
	movq	-144(%rbp), %rax
	leaq	8(%rax), %rdx
	movl	$16, %eax
	subq	$1, %rax
	addq	%rdx, %rax
	movl	$16, %ecx
	movl	$0, %edx
	divq	%rcx
	imulq	$16, %rax, %rdx
	movq	%rdx, %rax
	andq	$-4096, %rax
	movq	%rsp, %rcx
	subq	%rax, %rcx
.L195:
	cmpq	%rcx, %rsp
	je	.L196
	subq	$4096, %rsp
	orq	$0, 4088(%rsp)
	jmp	.L195
.L196:
	movq	%rdx, %rax
	andl	$4095, %eax
	subq	%rax, %rsp
	movq	%rdx, %rax
	andl	$4095, %eax
	testq	%rax, %rax
	je	.L197
	movq	%rdx, %rax
	andl	$4095, %eax
	subq	$8, %rax
	addq	%rsp, %rax
	orq	$0, (%rax)
.L197:
	movq	%rsp, %rax
	addq	$15, %rax
	shrq	$4, %rax
	salq	$4, %rax
	movq	%rax, -168(%rbp)
	movq	$19, -160(%rbp)
	jmp	.L192
.L184:
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
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
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC21(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC22(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC23(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC24(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC25(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC26(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC27(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-184(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$39, -160(%rbp)
	jmp	.L192
.L176:
	leaq	.LC28(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-192(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-192(%rbp), %edi
	movl	-196(%rbp), %eax
	leal	-1(%rax), %ecx
	movl	-196(%rbp), %esi
	movl	-196(%rbp), %edx
	movslq	%edx, %rax
	subq	$1, %rax
	movq	%rax, -128(%rbp)
	movslq	%edx, %rax
	movq	%rax, -320(%rbp)
	movq	$0, -312(%rbp)
	movq	-168(%rbp), %rax
	movl	%edi, %r8d
	movl	$0, %edx
	movq	%rax, %rdi
	call	binarysearch
	movq	$21, -160(%rbp)
	jmp	.L192
.L185:
	movl	-196(%rbp), %esi
	movl	-196(%rbp), %edx
	movslq	%edx, %rax
	subq	$1, %rax
	movq	%rax, -80(%rbp)
	movslq	%edx, %rax
	movq	%rax, -336(%rbp)
	movq	$0, -328(%rbp)
	movl	-172(%rbp), %edx
	movl	-180(%rbp), %eax
	movq	-168(%rbp), %rdi
	movl	%edx, %ecx
	movl	%eax, %edx
	call	mode
	movq	$21, -160(%rbp)
	jmp	.L192
.L181:
	movq	$21, -160(%rbp)
	jmp	.L192
.L175:
	movl	-196(%rbp), %esi
	movl	-196(%rbp), %edx
	movslq	%edx, %rax
	subq	$1, %rax
	movq	%rax, -136(%rbp)
	movslq	%edx, %rax
	movq	%rax, -352(%rbp)
	movq	$0, -344(%rbp)
	movl	-180(%rbp), %eax
	movq	-168(%rbp), %rcx
	movl	%eax, %edx
	movq	%rcx, %rdi
	call	lar
	movl	%eax, -176(%rbp)
	movl	-176(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC29(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$21, -160(%rbp)
	jmp	.L192
.L170:
	leaq	.LC28(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-188(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-188(%rbp), %esi
	movl	-196(%rbp), %ecx
	movl	-196(%rbp), %edx
	movslq	%edx, %rax
	subq	$1, %rax
	movq	%rax, -152(%rbp)
	movslq	%edx, %rax
	movq	%rax, %r14
	movl	$0, %r15d
	movq	-168(%rbp), %rax
	movl	%esi, %edx
	movl	%ecx, %esi
	movq	%rax, %rdi
	call	linearsearch
	movq	$21, -160(%rbp)
	jmp	.L192
.L188:
	movl	$0, %eax
	movq	-40(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L216
	jmp	.L217
.L191:
	movl	-196(%rbp), %esi
	movl	-196(%rbp), %edx
	movslq	%edx, %rax
	subq	$1, %rax
	movq	%rax, -48(%rbp)
	movslq	%edx, %rax
	movq	%rax, -368(%rbp)
	movq	$0, -360(%rbp)
	movl	-172(%rbp), %edx
	movl	-180(%rbp), %eax
	movq	-168(%rbp), %rdi
	movl	%edx, %ecx
	movl	%eax, %edx
	call	sort
	movq	$21, -160(%rbp)
	jmp	.L192
.L172:
	movl	-184(%rbp), %eax
	cmpl	$13, %eax
	ja	.L199
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L201(%rip), %rax
	movl	(%rdx,%rax), %eax
	movslq	%eax, %rdx
	leaq	.L201(%rip), %rax
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L201:
	.long	.L199-.L201
	.long	.L213-.L201
	.long	.L212-.L201
	.long	.L211-.L201
	.long	.L210-.L201
	.long	.L209-.L201
	.long	.L208-.L201
	.long	.L207-.L201
	.long	.L206-.L201
	.long	.L205-.L201
	.long	.L204-.L201
	.long	.L203-.L201
	.long	.L202-.L201
	.long	.L200-.L201
	.text
.L200:
	movq	$24, -160(%rbp)
	jmp	.L214
.L202:
	movq	$20, -160(%rbp)
	jmp	.L214
.L203:
	movq	$29, -160(%rbp)
	jmp	.L214
.L204:
	movq	$14, -160(%rbp)
	jmp	.L214
.L205:
	movq	$17, -160(%rbp)
	jmp	.L214
.L206:
	movq	$33, -160(%rbp)
	jmp	.L214
.L207:
	movq	$12, -160(%rbp)
	jmp	.L214
.L208:
	movq	$23, -160(%rbp)
	jmp	.L214
.L209:
	movq	$41, -160(%rbp)
	jmp	.L214
.L210:
	movq	$32, -160(%rbp)
	jmp	.L214
.L211:
	movq	$0, -160(%rbp)
	jmp	.L214
.L212:
	movq	$31, -160(%rbp)
	jmp	.L214
.L213:
	movq	$7, -160(%rbp)
	jmp	.L214
.L199:
	movq	$22, -160(%rbp)
	nop
.L214:
	jmp	.L192
.L189:
	movl	-196(%rbp), %ecx
	movl	-196(%rbp), %edx
	movslq	%edx, %rax
	subq	$1, %rax
	movq	%rax, -56(%rbp)
	movslq	%edx, %rax
	movq	%rax, -384(%rbp)
	movq	$0, -376(%rbp)
	movq	-168(%rbp), %rax
	movl	$0, %edx
	movl	%ecx, %esi
	movq	%rax, %rdi
	call	readarrays
	movq	$21, -160(%rbp)
	jmp	.L192
.L174:
	leaq	.LC30(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-196(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$36, -160(%rbp)
	jmp	.L192
.L178:
	movl	-196(%rbp), %esi
	movl	-196(%rbp), %edx
	movslq	%edx, %rax
	subq	$1, %rax
	movq	%rax, -112(%rbp)
	movslq	%edx, %rax
	movq	%rax, -400(%rbp)
	movq	$0, -392(%rbp)
	movl	-180(%rbp), %eax
	movq	-168(%rbp), %rcx
	movl	%eax, %edx
	movq	%rcx, %rdi
	call	sm
	movq	$21, -160(%rbp)
	jmp	.L192
.L183:
	movl	-196(%rbp), %esi
	movl	-196(%rbp), %edx
	movslq	%edx, %rax
	subq	$1, %rax
	movq	%rax, -88(%rbp)
	movslq	%edx, %rax
	movq	%rax, -416(%rbp)
	movq	$0, -408(%rbp)
	movl	-180(%rbp), %eax
	movq	-168(%rbp), %rcx
	movl	%eax, %edx
	movq	%rcx, %rdi
	call	sesm
	movq	$21, -160(%rbp)
	jmp	.L192
.L218:
	nop
.L192:
	jmp	.L215
.L217:
	call	__stack_chk_fail@PLT
.L216:
	leaq	-32(%rbp), %rsp
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE11:
	.size	main, .-main
	.section	.rodata
.LC31:
	.string	"\nThe sorted element is :\t"
	.text
	.globl	sort
	.type	sort, @function
sort:
.LFB12:
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
	movl	%edx, -32(%rbp)
	movl	%ecx, -36(%rbp)
	movq	$14, -8(%rbp)
.L241:
	cmpq	$14, -8(%rbp)
	ja	.L242
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L222(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L222(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L222:
	.long	.L232-.L222
	.long	.L231-.L222
	.long	.L242-.L222
	.long	.L230-.L222
	.long	.L242-.L222
	.long	.L229-.L222
	.long	.L228-.L222
	.long	.L242-.L222
	.long	.L242-.L222
	.long	.L227-.L222
	.long	.L226-.L222
	.long	.L225-.L222
	.long	.L224-.L222
	.long	.L243-.L222
	.long	.L221-.L222
	.text
.L221:
	movq	$1, -8(%rbp)
	jmp	.L233
.L224:
	movl	-32(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	movl	-32(%rbp), %eax
	cltq
	addq	$1, %rax
	leaq	0(,%rax,4), %rcx
	movq	-24(%rbp), %rax
	addq	%rcx, %rax
	movl	(%rax), %eax
	cmpl	%eax, %edx
	jle	.L234
	movq	$5, -8(%rbp)
	jmp	.L233
.L234:
	movq	$0, -8(%rbp)
	jmp	.L233
.L231:
	leaq	.LC31(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$0, -36(%rbp)
	movq	$9, -8(%rbp)
	jmp	.L233
.L230:
	movl	-32(%rbp), %eax
	cmpl	-28(%rbp), %eax
	jge	.L236
	movq	$12, -8(%rbp)
	jmp	.L233
.L236:
	movq	$11, -8(%rbp)
	jmp	.L233
.L225:
	addl	$1, -36(%rbp)
	movq	$9, -8(%rbp)
	jmp	.L233
.L227:
	movl	-28(%rbp), %eax
	subl	$1, %eax
	cmpl	%eax, -36(%rbp)
	jge	.L238
	movq	$6, -8(%rbp)
	jmp	.L233
.L238:
	movq	$10, -8(%rbp)
	jmp	.L233
.L228:
	movl	$0, -32(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L233
.L229:
	movl	-32(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, -12(%rbp)
	movl	-32(%rbp), %eax
	cltq
	addq	$1, %rax
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movl	-32(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-24(%rbp), %rdx
	addq	%rcx, %rdx
	movl	(%rax), %eax
	movl	%eax, (%rdx)
	movl	-32(%rbp), %eax
	cltq
	addq	$1, %rax
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rax, %rdx
	movl	-12(%rbp), %eax
	movl	%eax, (%rdx)
	movq	$0, -8(%rbp)
	jmp	.L233
.L226:
	movl	-28(%rbp), %ecx
	movq	-24(%rbp), %rax
	movl	$0, %edx
	movl	%ecx, %esi
	movq	%rax, %rdi
	call	displayarrays
	movq	$13, -8(%rbp)
	jmp	.L233
.L232:
	addl	$1, -32(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L233
.L242:
	nop
.L233:
	jmp	.L241
.L243:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE12:
	.size	sort, .-sort
	.section	.rodata
.LC32:
	.string	"\nThe marging arrays is "
	.align 8
.LC33:
	.string	"\nEnter the number of elements in arrays 2:\t"
	.text
	.globl	marging
	.type	marging, @function
marging:
.LFB14:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	addq	$-128, %rsp
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	movq	%rdi, -120(%rbp)
	movl	%esi, -124(%rbp)
	movl	%edx, -128(%rbp)
	movl	%ecx, -132(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -40(%rbp)
	xorl	%eax, %eax
	movq	$2, -88(%rbp)
.L272:
	cmpq	$21, -88(%rbp)
	ja	.L275
	movq	-88(%rbp), %rax
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
	.long	.L259-.L247
	.long	.L275-.L247
	.long	.L258-.L247
	.long	.L257-.L247
	.long	.L256-.L247
	.long	.L275-.L247
	.long	.L255-.L247
	.long	.L254-.L247
	.long	.L275-.L247
	.long	.L276-.L247
	.long	.L252-.L247
	.long	.L275-.L247
	.long	.L275-.L247
	.long	.L275-.L247
	.long	.L275-.L247
	.long	.L275-.L247
	.long	.L251-.L247
	.long	.L275-.L247
	.long	.L250-.L247
	.long	.L249-.L247
	.long	.L248-.L247
	.long	.L246-.L247
	.text
.L250:
	movl	-112(%rbp), %eax
	cmpl	%eax, -128(%rbp)
	jge	.L260
	movq	$16, -88(%rbp)
	jmp	.L262
.L260:
	movq	$10, -88(%rbp)
	jmp	.L262
.L256:
	movl	$0, -124(%rbp)
	movl	$0, -128(%rbp)
	movq	$3, -88(%rbp)
	jmp	.L262
.L257:
	movl	-128(%rbp), %eax
	cmpl	-132(%rbp), %eax
	jge	.L263
	movq	$19, -88(%rbp)
	jmp	.L262
.L263:
	movq	$21, -88(%rbp)
	jmp	.L262
.L251:
	movl	-128(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-104(%rbp), %rax
	addq	%rdx, %rax
	movl	-124(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-96(%rbp), %rdx
	addq	%rcx, %rdx
	movl	(%rax), %eax
	movl	%eax, (%rdx)
	addl	$1, -124(%rbp)
	addl	$1, -128(%rbp)
	movq	$18, -88(%rbp)
	jmp	.L262
.L246:
	movl	$0, -128(%rbp)
	movq	$18, -88(%rbp)
	jmp	.L262
.L249:
	movl	-128(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-120(%rbp), %rax
	addq	%rdx, %rax
	movl	-124(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-96(%rbp), %rdx
	addq	%rcx, %rdx
	movl	(%rax), %eax
	movl	%eax, (%rdx)
	addl	$1, -124(%rbp)
	addl	$1, -128(%rbp)
	movq	$3, -88(%rbp)
	jmp	.L262
.L255:
	movl	-112(%rbp), %ecx
	movl	-112(%rbp), %eax
	movslq	%eax, %rdx
	subq	$1, %rdx
	movq	%rdx, -56(%rbp)
	cltq
	movq	%rax, %r12
	movl	$0, %r13d
	movq	-104(%rbp), %rax
	movl	$0, %edx
	movl	%ecx, %esi
	movq	%rax, %rdi
	call	readarrays
	movl	-112(%rbp), %ecx
	movl	-112(%rbp), %eax
	movslq	%eax, %rdx
	subq	$1, %rdx
	movq	%rdx, -48(%rbp)
	cltq
	movq	%rax, %r14
	movl	$0, %r15d
	movq	-104(%rbp), %rax
	movl	$0, %edx
	movl	%ecx, %esi
	movq	%rax, %rdi
	call	displayarrays
	movl	-112(%rbp), %edx
	movl	-132(%rbp), %eax
	addl	%edx, %eax
	movl	%eax, -108(%rbp)
	movq	$20, -88(%rbp)
	jmp	.L262
.L252:
	leaq	.LC32(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-108(%rbp), %eax
	movslq	%eax, %rdx
	subq	$1, %rdx
	movq	%rdx, -72(%rbp)
	cltq
	movq	%rax, -160(%rbp)
	movq	$0, -152(%rbp)
	movl	-108(%rbp), %ecx
	movq	-96(%rbp), %rax
	movl	$0, %edx
	movl	%ecx, %esi
	movq	%rax, %rdi
	call	displayarrays
	movq	$9, -88(%rbp)
	jmp	.L262
.L259:
	movl	-124(%rbp), %edx
	movl	-132(%rbp), %ecx
	movq	-120(%rbp), %rax
	movl	%ecx, %esi
	movq	%rax, %rdi
	call	displayarrays
	leaq	.LC33(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-112(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$7, -88(%rbp)
	jmp	.L262
.L254:
	movl	-112(%rbp), %eax
	cltq
	salq	$5, %rax
	addq	$31, %rax
	shrq	$3, %rax
	movq	%rax, %rdx
	movabsq	$2305843009213693948, %rax
	andq	%rdx, %rax
	movq	%rax, -64(%rbp)
	movq	-64(%rbp), %rax
	leaq	8(%rax), %rdx
	movl	$16, %eax
	subq	$1, %rax
	addq	%rdx, %rax
	movl	$16, %esi
	movl	$0, %edx
	divq	%rsi
	imulq	$16, %rax, %rax
	movq	%rax, %rcx
	andq	$-4096, %rcx
	movq	%rsp, %rdx
	subq	%rcx, %rdx
.L266:
	cmpq	%rdx, %rsp
	je	.L267
	subq	$4096, %rsp
	orq	$0, 4088(%rsp)
	jmp	.L266
.L267:
	movq	%rax, %rdx
	andl	$4095, %edx
	subq	%rdx, %rsp
	movq	%rax, %rdx
	andl	$4095, %edx
	testq	%rdx, %rdx
	je	.L268
	andl	$4095, %eax
	subq	$8, %rax
	addq	%rsp, %rax
	orq	$0, (%rax)
.L268:
	movq	%rsp, %rax
	addq	$15, %rax
	shrq	$4, %rax
	salq	$4, %rax
	movq	%rax, -104(%rbp)
	movq	$6, -88(%rbp)
	jmp	.L262
.L258:
	movq	$0, -88(%rbp)
	jmp	.L262
.L248:
	movl	-108(%rbp), %eax
	cltq
	salq	$5, %rax
	addq	$31, %rax
	shrq	$3, %rax
	movq	%rax, %rdx
	movabsq	$2305843009213693948, %rax
	andq	%rdx, %rax
	movq	%rax, -80(%rbp)
	movq	-80(%rbp), %rax
	leaq	8(%rax), %rdx
	movl	$16, %eax
	subq	$1, %rax
	addq	%rdx, %rax
	movl	$16, %esi
	movl	$0, %edx
	divq	%rsi
	imulq	$16, %rax, %rax
	movq	%rax, %rcx
	andq	$-4096, %rcx
	movq	%rsp, %rdx
	subq	%rcx, %rdx
.L269:
	cmpq	%rdx, %rsp
	je	.L270
	subq	$4096, %rsp
	orq	$0, 4088(%rsp)
	jmp	.L269
.L270:
	movq	%rax, %rdx
	andl	$4095, %edx
	subq	%rdx, %rsp
	movq	%rax, %rdx
	andl	$4095, %edx
	testq	%rdx, %rdx
	je	.L271
	andl	$4095, %eax
	subq	$8, %rax
	addq	%rsp, %rax
	orq	$0, (%rax)
.L271:
	movq	%rsp, %rax
	addq	$15, %rax
	shrq	$4, %rax
	salq	$4, %rax
	movq	%rax, -96(%rbp)
	movq	$4, -88(%rbp)
	jmp	.L262
.L275:
	nop
.L262:
	jmp	.L272
.L276:
	nop
	movq	-40(%rbp), %rax
	subq	%fs:40, %rax
	je	.L274
	call	__stack_chk_fail@PLT
.L274:
	leaq	-32(%rbp), %rsp
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE14:
	.size	marging, .-marging
	.section	.rodata
	.align 8
.LC34:
	.string	"The second largest number is %d"
	.text
	.globl	selar
	.type	selar, @function
selar:
.LFB15:
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
	movl	%edx, -32(%rbp)
	movq	$5, -8(%rbp)
.L306:
	cmpq	$17, -8(%rbp)
	ja	.L307
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L280(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L280(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L280:
	.long	.L293-.L280
	.long	.L308-.L280
	.long	.L291-.L280
	.long	.L290-.L280
	.long	.L289-.L280
	.long	.L288-.L280
	.long	.L307-.L280
	.long	.L287-.L280
	.long	.L307-.L280
	.long	.L286-.L280
	.long	.L307-.L280
	.long	.L285-.L280
	.long	.L284-.L280
	.long	.L283-.L280
	.long	.L282-.L280
	.long	.L307-.L280
	.long	.L281-.L280
	.long	.L279-.L280
	.text
.L289:
	movl	-32(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, -12(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L294
.L282:
	movl	-12(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC34(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -8(%rbp)
	jmp	.L294
.L284:
	movl	-32(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	%eax, -16(%rbp)
	jge	.L295
	movq	$7, -8(%rbp)
	jmp	.L294
.L295:
	movq	$9, -8(%rbp)
	jmp	.L294
.L290:
	movq	-24(%rbp), %rax
	movl	4(%rax), %eax
	movl	%eax, -12(%rbp)
	movl	$0, -32(%rbp)
	movq	$17, -8(%rbp)
	jmp	.L294
.L281:
	movl	-32(%rbp), %eax
	cmpl	-28(%rbp), %eax
	jge	.L298
	movq	$12, -8(%rbp)
	jmp	.L294
.L298:
	movq	$3, -8(%rbp)
	jmp	.L294
.L285:
	movl	-32(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	%eax, -16(%rbp)
	je	.L300
	movq	$13, -8(%rbp)
	jmp	.L294
.L300:
	movq	$0, -8(%rbp)
	jmp	.L294
.L286:
	addl	$1, -32(%rbp)
	movq	$16, -8(%rbp)
	jmp	.L294
.L283:
	movl	-32(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	%eax, -12(%rbp)
	jge	.L302
	movq	$4, -8(%rbp)
	jmp	.L294
.L302:
	movq	$0, -8(%rbp)
	jmp	.L294
.L279:
	movl	-32(%rbp), %eax
	cmpl	-28(%rbp), %eax
	jge	.L304
	movq	$11, -8(%rbp)
	jmp	.L294
.L304:
	movq	$14, -8(%rbp)
	jmp	.L294
.L288:
	movq	$2, -8(%rbp)
	jmp	.L294
.L293:
	addl	$1, -32(%rbp)
	movq	$17, -8(%rbp)
	jmp	.L294
.L287:
	movl	-32(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, -16(%rbp)
	movq	$9, -8(%rbp)
	jmp	.L294
.L291:
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, -16(%rbp)
	movl	$0, -32(%rbp)
	movq	$16, -8(%rbp)
	jmp	.L294
.L307:
	nop
.L294:
	jmp	.L306
.L308:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE15:
	.size	selar, .-selar
	.section	.rodata
.LC35:
	.string	"The mean number is %d:\t"
	.text
	.globl	mean
	.type	mean, @function
mean:
.LFB16:
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
	movl	%edx, -32(%rbp)
	movq	$3, -8(%rbp)
.L321:
	cmpq	$4, -8(%rbp)
	ja	.L322
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L312(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L312(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L312:
	.long	.L316-.L312
	.long	.L315-.L312
	.long	.L323-.L312
	.long	.L313-.L312
	.long	.L311-.L312
	.text
.L311:
	movl	-32(%rbp), %eax
	cmpl	-28(%rbp), %eax
	jge	.L317
	movq	$0, -8(%rbp)
	jmp	.L319
.L317:
	movq	$1, -8(%rbp)
	jmp	.L319
.L315:
	pxor	%xmm0, %xmm0
	cvtsi2ssl	-12(%rbp), %xmm0
	pxor	%xmm1, %xmm1
	cvtsi2ssl	-28(%rbp), %xmm1
	divss	%xmm1, %xmm0
	pxor	%xmm2, %xmm2
	cvtss2sd	%xmm0, %xmm2
	movq	%xmm2, %rax
	movq	%rax, %xmm0
	leaq	.LC35(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$2, -8(%rbp)
	jmp	.L319
.L313:
	movl	$0, -12(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L319
.L316:
	movl	-32(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	addl	%eax, -12(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L319
.L322:
	nop
.L319:
	jmp	.L321
.L323:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE16:
	.size	mean, .-mean
	.section	.rodata
	.align 8
.LC36:
	.string	"\nFound element that you search at %d."
	.text
	.globl	binarysearch
	.type	binarysearch, @function
binarysearch:
.LFB18:
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
	movl	%edx, -32(%rbp)
	movl	%ecx, -36(%rbp)
	movl	%r8d, -40(%rbp)
	movq	$3, -8(%rbp)
.L340:
	cmpq	$6, -8(%rbp)
	ja	.L341
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L327(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L327(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L327:
	.long	.L342-.L327
	.long	.L332-.L327
	.long	.L331-.L327
	.long	.L330-.L327
	.long	.L329-.L327
	.long	.L328-.L327
	.long	.L326-.L327
	.text
.L329:
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	%eax, -40(%rbp)
	jne	.L334
	movq	$2, -8(%rbp)
	jmp	.L336
.L334:
	movq	$6, -8(%rbp)
	jmp	.L336
.L332:
	movl	-12(%rbp), %eax
	leal	1(%rax), %edi
	movl	-40(%rbp), %ecx
	movl	-36(%rbp), %edx
	movl	-28(%rbp), %esi
	movq	-24(%rbp), %rax
	movl	%ecx, %r8d
	movl	%edx, %ecx
	movl	%edi, %edx
	movq	%rax, %rdi
	call	binarysearch
	movq	$0, -8(%rbp)
	jmp	.L336
.L330:
	movl	-32(%rbp), %edx
	movl	-36(%rbp), %eax
	addl	%edx, %eax
	movl	%eax, %edx
	shrl	$31, %edx
	addl	%edx, %eax
	sarl	%eax
	movl	%eax, -12(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L336
.L326:
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	%eax, -40(%rbp)
	jle	.L337
	movq	$1, -8(%rbp)
	jmp	.L336
.L337:
	movq	$5, -8(%rbp)
	jmp	.L336
.L328:
	movl	-12(%rbp), %eax
	leal	-1(%rax), %ecx
	movl	-40(%rbp), %edi
	movl	-32(%rbp), %edx
	movl	-28(%rbp), %esi
	movq	-24(%rbp), %rax
	movl	%edi, %r8d
	movq	%rax, %rdi
	call	binarysearch
	movq	$0, -8(%rbp)
	jmp	.L336
.L331:
	movl	-12(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC36(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$0, -8(%rbp)
	jmp	.L336
.L341:
	nop
.L336:
	jmp	.L340
.L342:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE18:
	.size	binarysearch, .-binarysearch
	.section	.rodata
	.align 4
.LC10:
	.long	1073741824
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
