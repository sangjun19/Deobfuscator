	.file	"Harshpathakjnp_CCodings_main_flatten.c"
	.text
	.globl	_TIG_IZ_j6dG_argc
	.bss
	.align 4
	.type	_TIG_IZ_j6dG_argc, @object
	.size	_TIG_IZ_j6dG_argc, 4
_TIG_IZ_j6dG_argc:
	.zero	4
	.globl	_TIG_IZ_j6dG_envp
	.align 8
	.type	_TIG_IZ_j6dG_envp, @object
	.size	_TIG_IZ_j6dG_envp, 8
_TIG_IZ_j6dG_envp:
	.zero	8
	.globl	_TIG_IZ_j6dG_argv
	.align 8
	.type	_TIG_IZ_j6dG_argv, @object
	.size	_TIG_IZ_j6dG_argv, 8
_TIG_IZ_j6dG_argv:
	.zero	8
	.text
	.globl	daysInMonths
	.type	daysInMonths, @function
daysInMonths:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, -20(%rbp)
	movl	%esi, -24(%rbp)
	movq	$4, -8(%rbp)
.L32:
	cmpq	$13, -8(%rbp)
	ja	.L33
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
	.long	.L17-.L4
	.long	.L16-.L4
	.long	.L15-.L4
	.long	.L14-.L4
	.long	.L13-.L4
	.long	.L12-.L4
	.long	.L11-.L4
	.long	.L10-.L4
	.long	.L9-.L4
	.long	.L8-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L13:
	cmpl	$4, -20(%rbp)
	jne	.L18
	movq	$5, -8(%rbp)
	jmp	.L20
.L18:
	movq	$12, -8(%rbp)
	jmp	.L20
.L5:
	cmpl	$6, -20(%rbp)
	jne	.L21
	movq	$8, -8(%rbp)
	jmp	.L20
.L21:
	movq	$2, -8(%rbp)
	jmp	.L20
.L9:
	movl	$30, %eax
	jmp	.L23
.L16:
	movl	$30, %eax
	jmp	.L23
.L14:
	movl	-24(%rbp), %eax
	movl	%eax, %edi
	call	isLeapYear
	movl	%eax, -12(%rbp)
	movq	$13, -8(%rbp)
	jmp	.L20
.L6:
	movl	$28, %eax
	jmp	.L23
.L8:
	movl	$30, %eax
	jmp	.L23
.L3:
	cmpl	$0, -12(%rbp)
	je	.L24
	movq	$7, -8(%rbp)
	jmp	.L20
.L24:
	movq	$11, -8(%rbp)
	jmp	.L20
.L11:
	cmpl	$11, -20(%rbp)
	jne	.L26
	movq	$1, -8(%rbp)
	jmp	.L20
.L26:
	movq	$0, -8(%rbp)
	jmp	.L20
.L12:
	movl	$30, %eax
	jmp	.L23
.L7:
	movl	$31, %eax
	jmp	.L23
.L17:
	cmpl	$2, -20(%rbp)
	jne	.L28
	movq	$3, -8(%rbp)
	jmp	.L20
.L28:
	movq	$10, -8(%rbp)
	jmp	.L20
.L10:
	movl	$29, %eax
	jmp	.L23
.L15:
	cmpl	$9, -20(%rbp)
	jne	.L30
	movq	$9, -8(%rbp)
	jmp	.L20
.L30:
	movq	$6, -8(%rbp)
	jmp	.L20
.L33:
	nop
.L20:
	jmp	.L32
.L23:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	daysInMonths, .-daysInMonths
	.globl	isDateValid
	.type	isDateValid, @function
isDateValid:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, -20(%rbp)
	movl	%esi, -24(%rbp)
	movl	%edx, -28(%rbp)
	movq	$2, -8(%rbp)
.L61:
	cmpq	$12, -8(%rbp)
	ja	.L62
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L37(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L37(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L37:
	.long	.L48-.L37
	.long	.L47-.L37
	.long	.L46-.L37
	.long	.L45-.L37
	.long	.L44-.L37
	.long	.L43-.L37
	.long	.L42-.L37
	.long	.L41-.L37
	.long	.L40-.L37
	.long	.L39-.L37
	.long	.L62-.L37
	.long	.L38-.L37
	.long	.L36-.L37
	.text
.L44:
	cmpl	$0, -20(%rbp)
	jg	.L49
	movq	$6, -8(%rbp)
	jmp	.L51
.L49:
	movq	$9, -8(%rbp)
	jmp	.L51
.L36:
	movl	$0, %eax
	jmp	.L52
.L40:
	movl	$0, %eax
	jmp	.L52
.L47:
	movl	$0, %eax
	jmp	.L52
.L45:
	cmpl	$0, -24(%rbp)
	jg	.L53
	movq	$8, -8(%rbp)
	jmp	.L51
.L53:
	movq	$0, -8(%rbp)
	jmp	.L51
.L38:
	movl	-20(%rbp), %eax
	cmpl	-16(%rbp), %eax
	jle	.L55
	movq	$7, -8(%rbp)
	jmp	.L51
.L55:
	movq	$5, -8(%rbp)
	jmp	.L51
.L39:
	movl	-28(%rbp), %edx
	movl	-24(%rbp), %eax
	movl	%edx, %esi
	movl	%eax, %edi
	call	daysInMonths
	movl	%eax, -12(%rbp)
	movl	-12(%rbp), %eax
	movl	%eax, -16(%rbp)
	movq	$11, -8(%rbp)
	jmp	.L51
.L42:
	movl	$0, %eax
	jmp	.L52
.L43:
	movl	$1, %eax
	jmp	.L52
.L48:
	cmpl	$12, -24(%rbp)
	jle	.L57
	movq	$1, -8(%rbp)
	jmp	.L51
.L57:
	movq	$4, -8(%rbp)
	jmp	.L51
.L41:
	movl	$0, %eax
	jmp	.L52
.L46:
	cmpl	$0, -28(%rbp)
	jg	.L59
	movq	$12, -8(%rbp)
	jmp	.L51
.L59:
	movq	$3, -8(%rbp)
	jmp	.L51
.L62:
	nop
.L51:
	jmp	.L61
.L52:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	isDateValid, .-isDateValid
	.globl	getWeekDayForDate
	.type	getWeekDayForDate, @function
getWeekDayForDate:
.LFB3:
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
	movl	%edx, -44(%rbp)
	movq	$1, -8(%rbp)
.L69:
	cmpq	$2, -8(%rbp)
	je	.L64
	cmpq	$2, -8(%rbp)
	ja	.L71
	cmpq	$0, -8(%rbp)
	je	.L66
	cmpq	$1, -8(%rbp)
	jne	.L71
	movq	$0, -8(%rbp)
	jmp	.L67
.L66:
	movl	$31, -24(%rbp)
	movl	$10, -20(%rbp)
	movl	$2021, -16(%rbp)
	movl	-44(%rbp), %r8d
	movl	-40(%rbp), %edi
	movl	-36(%rbp), %ecx
	movl	-16(%rbp), %edx
	movl	-20(%rbp), %esi
	movl	-24(%rbp), %eax
	movl	%r8d, %r9d
	movl	%edi, %r8d
	movl	%eax, %edi
	call	gapBetweenDates
	movl	%eax, -12(%rbp)
	movl	-12(%rbp), %eax
	movl	%eax, -28(%rbp)
	movl	-28(%rbp), %eax
	movslq	%eax, %rdx
	imulq	$-1840700269, %rdx, %rdx
	shrq	$32, %rdx
	addl	%eax, %edx
	sarl	$2, %edx
	movl	%eax, %esi
	sarl	$31, %esi
	movl	%edx, %ecx
	subl	%esi, %ecx
	movl	%ecx, %edx
	sall	$3, %edx
	subl	%ecx, %edx
	subl	%edx, %eax
	movl	%eax, -28(%rbp)
	addl	$7, -28(%rbp)
	movl	-28(%rbp), %eax
	movslq	%eax, %rdx
	imulq	$-1840700269, %rdx, %rdx
	shrq	$32, %rdx
	addl	%eax, %edx
	sarl	$2, %edx
	movl	%eax, %esi
	sarl	$31, %esi
	movl	%edx, %ecx
	subl	%esi, %ecx
	movl	%ecx, %edx
	sall	$3, %edx
	subl	%ecx, %edx
	subl	%edx, %eax
	movl	%eax, -28(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L67
.L64:
	movl	-28(%rbp), %eax
	jmp	.L70
.L71:
	nop
.L67:
	jmp	.L69
.L70:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	getWeekDayForDate, .-getWeekDayForDate
	.globl	isLeapYear
	.type	isLeapYear, @function
isLeapYear:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -20(%rbp)
	movq	$3, -8(%rbp)
.L90:
	cmpq	$7, -8(%rbp)
	ja	.L91
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L75(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L75(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L75:
	.long	.L81-.L75
	.long	.L80-.L75
	.long	.L79-.L75
	.long	.L78-.L75
	.long	.L77-.L75
	.long	.L76-.L75
	.long	.L91-.L75
	.long	.L74-.L75
	.text
.L77:
	movl	$0, %eax
	jmp	.L82
.L80:
	movl	-20(%rbp), %edx
	movslq	%edx, %rax
	imulq	$1374389535, %rax, %rax
	shrq	$32, %rax
	sarl	$5, %eax
	movl	%edx, %ecx
	sarl	$31, %ecx
	subl	%ecx, %eax
	imull	$100, %eax, %ecx
	movl	%edx, %eax
	subl	%ecx, %eax
	testl	%eax, %eax
	je	.L83
	movq	$2, -8(%rbp)
	jmp	.L85
.L83:
	movq	$4, -8(%rbp)
	jmp	.L85
.L78:
	movl	-20(%rbp), %edx
	movslq	%edx, %rax
	imulq	$1374389535, %rax, %rax
	shrq	$32, %rax
	sarl	$7, %eax
	movl	%edx, %ecx
	sarl	$31, %ecx
	subl	%ecx, %eax
	imull	$400, %eax, %ecx
	movl	%edx, %eax
	subl	%ecx, %eax
	testl	%eax, %eax
	jne	.L86
	movq	$7, -8(%rbp)
	jmp	.L85
.L86:
	movq	$0, -8(%rbp)
	jmp	.L85
.L76:
	movl	$0, %eax
	jmp	.L82
.L81:
	movl	-20(%rbp), %eax
	andl	$3, %eax
	testl	%eax, %eax
	jne	.L88
	movq	$1, -8(%rbp)
	jmp	.L85
.L88:
	movq	$5, -8(%rbp)
	jmp	.L85
.L74:
	movl	$1, %eax
	jmp	.L82
.L79:
	movl	$1, %eax
	jmp	.L82
.L91:
	nop
.L85:
	jmp	.L90
.L82:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	isLeapYear, .-isLeapYear
	.section	.rodata
.LC0:
	.string	"Fri"
.LC1:
	.string	"%6s"
.LC2:
	.string	"Tue"
.LC3:
	.string	"Thu"
.LC4:
	.string	"wrong day"
.LC5:
	.string	"Wed"
.LC6:
	.string	"Sat"
.LC7:
	.string	"Sun"
.LC8:
	.string	"Mon"
	.text
	.globl	dayOfWeek
	.type	dayOfWeek, @function
dayOfWeek:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, -20(%rbp)
	movq	$5, -8(%rbp)
.L117:
	cmpq	$16, -8(%rbp)
	ja	.L119
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L95(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L95(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L95:
	.long	.L119-.L95
	.long	.L104-.L95
	.long	.L103-.L95
	.long	.L119-.L95
	.long	.L102-.L95
	.long	.L101-.L95
	.long	.L119-.L95
	.long	.L119-.L95
	.long	.L100-.L95
	.long	.L119-.L95
	.long	.L99-.L95
	.long	.L98-.L95
	.long	.L119-.L95
	.long	.L97-.L95
	.long	.L119-.L95
	.long	.L96-.L95
	.long	.L94-.L95
	.text
.L102:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$16, -8(%rbp)
	jmp	.L105
.L96:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$16, -8(%rbp)
	jmp	.L105
.L100:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$16, -8(%rbp)
	jmp	.L105
.L104:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$16, -8(%rbp)
	jmp	.L105
.L94:
	movl	$0, %eax
	jmp	.L118
.L98:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$16, -8(%rbp)
	jmp	.L105
.L97:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$16, -8(%rbp)
	jmp	.L105
.L101:
	cmpl	$6, -20(%rbp)
	ja	.L107
	movl	-20(%rbp), %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L109(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L109(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L109:
	.long	.L115-.L109
	.long	.L114-.L109
	.long	.L113-.L109
	.long	.L112-.L109
	.long	.L111-.L109
	.long	.L110-.L109
	.long	.L108-.L109
	.text
.L108:
	movq	$13, -8(%rbp)
	jmp	.L116
.L110:
	movq	$4, -8(%rbp)
	jmp	.L116
.L111:
	movq	$8, -8(%rbp)
	jmp	.L116
.L112:
	movq	$11, -8(%rbp)
	jmp	.L116
.L113:
	movq	$15, -8(%rbp)
	jmp	.L116
.L114:
	movq	$2, -8(%rbp)
	jmp	.L116
.L115:
	movq	$10, -8(%rbp)
	jmp	.L116
.L107:
	movq	$1, -8(%rbp)
	nop
.L116:
	jmp	.L105
.L99:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$16, -8(%rbp)
	jmp	.L105
.L103:
	leaq	.LC8(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$16, -8(%rbp)
	jmp	.L105
.L119:
	nop
.L105:
	jmp	.L117
.L118:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	dayOfWeek, .-dayOfWeek
	.globl	dayDifferenceFromToday
	.type	dayDifferenceFromToday, @function
dayDifferenceFromToday:
.LFB8:
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
	movl	%edx, -44(%rbp)
	movq	$0, -8(%rbp)
.L126:
	cmpq	$2, -8(%rbp)
	je	.L127
	cmpq	$2, -8(%rbp)
	ja	.L128
	cmpq	$0, -8(%rbp)
	je	.L123
	cmpq	$1, -8(%rbp)
	jne	.L128
	movl	$31, -28(%rbp)
	movl	$10, -24(%rbp)
	movl	$2021, -20(%rbp)
	movl	-44(%rbp), %r8d
	movl	-40(%rbp), %edi
	movl	-36(%rbp), %ecx
	movl	-20(%rbp), %edx
	movl	-24(%rbp), %esi
	movl	-28(%rbp), %eax
	movl	%r8d, %r9d
	movl	%edi, %r8d
	movl	%eax, %edi
	call	gapBetweenDates
	movl	%eax, -16(%rbp)
	movl	-16(%rbp), %eax
	movl	%eax, -12(%rbp)
	movl	-12(%rbp), %eax
	movslq	%eax, %rdx
	imulq	$-1840700269, %rdx, %rdx
	shrq	$32, %rdx
	addl	%eax, %edx
	sarl	$2, %edx
	movl	%eax, %esi
	sarl	$31, %esi
	movl	%edx, %ecx
	subl	%esi, %ecx
	movl	%ecx, %edx
	sall	$3, %edx
	subl	%ecx, %edx
	subl	%edx, %eax
	movl	%eax, -12(%rbp)
	addl	$7, -12(%rbp)
	movl	-12(%rbp), %eax
	movslq	%eax, %rdx
	imulq	$-1840700269, %rdx, %rdx
	shrq	$32, %rdx
	addl	%eax, %edx
	sarl	$2, %edx
	movl	%eax, %esi
	sarl	$31, %esi
	movl	%edx, %ecx
	subl	%esi, %ecx
	movl	%ecx, %edx
	sall	$3, %edx
	subl	%ecx, %edx
	subl	%edx, %eax
	movl	%eax, -12(%rbp)
	movl	-12(%rbp), %eax
	movl	%eax, %edi
	call	dayOfWeekByWeekDayNo
	movq	$2, -8(%rbp)
	jmp	.L124
.L123:
	movq	$1, -8(%rbp)
	jmp	.L124
.L128:
	nop
.L124:
	jmp	.L126
.L127:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	dayDifferenceFromToday, .-dayDifferenceFromToday
	.section	.rodata
.LC9:
	.string	" "
.LC10:
	.string	"%6d"
.LC11:
	.string	"%13s"
.LC12:
	.string	" - %d"
	.text
	.globl	printingCalender
	.type	printingCalender, @function
printingCalender:
.LFB10:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movl	%edi, -52(%rbp)
	movl	%esi, -56(%rbp)
	movq	$21, -8(%rbp)
.L156:
	cmpq	$21, -8(%rbp)
	ja	.L158
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L132(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L132(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L132:
	.long	.L145-.L132
	.long	.L144-.L132
	.long	.L158-.L132
	.long	.L158-.L132
	.long	.L143-.L132
	.long	.L142-.L132
	.long	.L141-.L132
	.long	.L140-.L132
	.long	.L158-.L132
	.long	.L139-.L132
	.long	.L138-.L132
	.long	.L158-.L132
	.long	.L158-.L132
	.long	.L158-.L132
	.long	.L137-.L132
	.long	.L136-.L132
	.long	.L135-.L132
	.long	.L158-.L132
	.long	.L158-.L132
	.long	.L134-.L132
	.long	.L133-.L132
	.long	.L131-.L132
	.text
.L143:
	movl	-28(%rbp), %eax
	movl	%eax, %edi
	call	dayOfWeek
	addl	$1, -28(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L146
.L137:
	movl	$1, -20(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L146
.L136:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -24(%rbp)
	movq	$9, -8(%rbp)
	jmp	.L146
.L144:
	movl	-20(%rbp), %eax
	cmpl	-32(%rbp), %eax
	jg	.L147
	movq	$7, -8(%rbp)
	jmp	.L146
.L147:
	movq	$10, -8(%rbp)
	jmp	.L146
.L135:
	addl	$1, -20(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L146
.L131:
	movq	$20, -8(%rbp)
	jmp	.L146
.L139:
	movl	-24(%rbp), %eax
	cmpl	-36(%rbp), %eax
	jg	.L149
	movq	$15, -8(%rbp)
	jmp	.L146
.L149:
	movq	$14, -8(%rbp)
	jmp	.L146
.L134:
	movl	$10, %edi
	call	putchar@PLT
	movl	$1, -24(%rbp)
	movq	$9, -8(%rbp)
	jmp	.L146
.L141:
	movl	-20(%rbp), %edx
	movl	-36(%rbp), %eax
	addl	%eax, %edx
	movslq	%edx, %rax
	imulq	$-1840700269, %rax, %rax
	shrq	$32, %rax
	addl	%edx, %eax
	sarl	$2, %eax
	movl	%edx, %ecx
	sarl	$31, %ecx
	subl	%ecx, %eax
	movl	%eax, %ecx
	sall	$3, %ecx
	subl	%eax, %ecx
	movl	%edx, %eax
	subl	%ecx, %eax
	testl	%eax, %eax
	jne	.L151
	movq	$0, -8(%rbp)
	jmp	.L146
.L151:
	movq	$16, -8(%rbp)
	jmp	.L146
.L142:
	cmpl	$6, -28(%rbp)
	jg	.L153
	movq	$4, -8(%rbp)
	jmp	.L146
.L153:
	movq	$19, -8(%rbp)
	jmp	.L146
.L138:
	movl	$0, %eax
	jmp	.L157
.L145:
	movl	$10, %edi
	call	putchar@PLT
	movq	$16, -8(%rbp)
	jmp	.L146
.L140:
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$6, -8(%rbp)
	jmp	.L146
.L133:
	movl	-56(%rbp), %edx
	movl	-52(%rbp), %eax
	movl	%eax, %esi
	movl	$1, %edi
	call	getWeekDayForDate
	movl	%eax, -16(%rbp)
	movl	-16(%rbp), %eax
	movl	%eax, -36(%rbp)
	movl	-56(%rbp), %edx
	movl	-52(%rbp), %eax
	movl	%edx, %esi
	movl	%eax, %edi
	call	daysInMonths
	movl	%eax, -12(%rbp)
	movl	-12(%rbp), %eax
	movl	%eax, -32(%rbp)
	leaq	.LC9(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-52(%rbp), %eax
	movl	%eax, %edi
	call	monthNames
	movl	-56(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$10, %edi
	call	putchar@PLT
	movl	$0, -28(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L146
.L158:
	nop
.L146:
	jmp	.L156
.L157:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	printingCalender, .-printingCalender
	.section	.rodata
.LC13:
	.string	"October  "
.LC14:
	.string	"July  "
.LC15:
	.string	"May  "
.LC16:
	.string	"March  "
.LC17:
	.string	"August  "
.LC18:
	.string	"December  "
.LC19:
	.string	"January  "
.LC20:
	.string	"November  "
.LC21:
	.string	"September  "
.LC22:
	.string	"April  "
.LC23:
	.string	"February  "
.LC24:
	.string	"June  "
.LC25:
	.string	"wrong Month"
	.text
	.globl	monthNames
	.type	monthNames, @function
monthNames:
.LFB11:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, -20(%rbp)
	movq	$5, -8(%rbp)
.L194:
	cmpq	$25, -8(%rbp)
	ja	.L196
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L162(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L162(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L162:
	.long	.L176-.L162
	.long	.L175-.L162
	.long	.L174-.L162
	.long	.L196-.L162
	.long	.L173-.L162
	.long	.L172-.L162
	.long	.L196-.L162
	.long	.L171-.L162
	.long	.L170-.L162
	.long	.L196-.L162
	.long	.L196-.L162
	.long	.L169-.L162
	.long	.L168-.L162
	.long	.L196-.L162
	.long	.L167-.L162
	.long	.L166-.L162
	.long	.L196-.L162
	.long	.L196-.L162
	.long	.L196-.L162
	.long	.L196-.L162
	.long	.L165-.L162
	.long	.L164-.L162
	.long	.L196-.L162
	.long	.L196-.L162
	.long	.L163-.L162
	.long	.L161-.L162
	.text
.L161:
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$11, -8(%rbp)
	jmp	.L177
.L173:
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$11, -8(%rbp)
	jmp	.L177
.L167:
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$11, -8(%rbp)
	jmp	.L177
.L166:
	leaq	.LC16(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$11, -8(%rbp)
	jmp	.L177
.L168:
	leaq	.LC17(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$11, -8(%rbp)
	jmp	.L177
.L170:
	leaq	.LC18(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$11, -8(%rbp)
	jmp	.L177
.L175:
	leaq	.LC19(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$11, -8(%rbp)
	jmp	.L177
.L163:
	leaq	.LC20(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$11, -8(%rbp)
	jmp	.L177
.L164:
	leaq	.LC21(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$11, -8(%rbp)
	jmp	.L177
.L169:
	movl	$0, %eax
	jmp	.L195
.L172:
	cmpl	$12, -20(%rbp)
	ja	.L179
	movl	-20(%rbp), %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L181(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L181(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L181:
	.long	.L179-.L181
	.long	.L192-.L181
	.long	.L191-.L181
	.long	.L190-.L181
	.long	.L189-.L181
	.long	.L188-.L181
	.long	.L187-.L181
	.long	.L186-.L181
	.long	.L185-.L181
	.long	.L184-.L181
	.long	.L183-.L181
	.long	.L182-.L181
	.long	.L180-.L181
	.text
.L180:
	movq	$8, -8(%rbp)
	jmp	.L193
.L182:
	movq	$24, -8(%rbp)
	jmp	.L193
.L183:
	movq	$25, -8(%rbp)
	jmp	.L193
.L184:
	movq	$21, -8(%rbp)
	jmp	.L193
.L185:
	movq	$12, -8(%rbp)
	jmp	.L193
.L186:
	movq	$4, -8(%rbp)
	jmp	.L193
.L187:
	movq	$2, -8(%rbp)
	jmp	.L193
.L188:
	movq	$14, -8(%rbp)
	jmp	.L193
.L189:
	movq	$0, -8(%rbp)
	jmp	.L193
.L190:
	movq	$15, -8(%rbp)
	jmp	.L193
.L191:
	movq	$7, -8(%rbp)
	jmp	.L193
.L192:
	movq	$1, -8(%rbp)
	jmp	.L193
.L179:
	movq	$20, -8(%rbp)
	nop
.L193:
	jmp	.L177
.L176:
	leaq	.LC22(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$11, -8(%rbp)
	jmp	.L177
.L171:
	leaq	.LC23(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$11, -8(%rbp)
	jmp	.L177
.L174:
	leaq	.LC24(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$11, -8(%rbp)
	jmp	.L177
.L165:
	leaq	.LC25(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$11, -8(%rbp)
	jmp	.L177
.L196:
	nop
.L177:
	jmp	.L194
.L195:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE11:
	.size	monthNames, .-monthNames
	.section	.rodata
.LC26:
	.string	"\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB13:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movl	%edi, -20(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movq	$0, _TIG_IZ_j6dG_envp(%rip)
	nop
.L198:
	movq	$0, _TIG_IZ_j6dG_argv(%rip)
	nop
.L199:
	movl	$0, _TIG_IZ_j6dG_argc(%rip)
	nop
	nop
.L200:
.L201:
#APP
# 122 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-j6dG--0
# 0 "" 2
#NO_APP
	movl	-20(%rbp), %eax
	movl	%eax, _TIG_IZ_j6dG_argc(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, _TIG_IZ_j6dG_argv(%rip)
	movq	-40(%rbp), %rax
	movq	%rax, _TIG_IZ_j6dG_envp(%rip)
	nop
	movq	$3, -8(%rbp)
.L213:
	cmpq	$7, -8(%rbp)
	ja	.L215
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L204(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L204(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L204:
	.long	.L215-.L204
	.long	.L208-.L204
	.long	.L215-.L204
	.long	.L207-.L204
	.long	.L206-.L204
	.long	.L205-.L204
	.long	.L215-.L204
	.long	.L203-.L204
	.text
.L206:
	movl	$2021, -16(%rbp)
	movl	$1, -12(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L209
.L208:
	cmpl	$12, -12(%rbp)
	jg	.L210
	movq	$5, -8(%rbp)
	jmp	.L209
.L210:
	movq	$7, -8(%rbp)
	jmp	.L209
.L207:
	movq	$4, -8(%rbp)
	jmp	.L209
.L205:
	movl	-16(%rbp), %edx
	movl	-12(%rbp), %eax
	movl	%edx, %esi
	movl	%eax, %edi
	call	printingCalender
	leaq	.LC26(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	addl	$1, -12(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L209
.L203:
	movl	$0, %eax
	jmp	.L214
.L215:
	nop
.L209:
	jmp	.L213
.L214:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE13:
	.size	main, .-main
	.section	.rodata
.LC27:
	.string	"Wednesday "
.LC28:
	.string	"Saturday "
.LC29:
	.string	"Monday "
.LC30:
	.string	"Thursday "
.LC31:
	.string	"Sunday "
.LC32:
	.string	"Tuesday "
.LC33:
	.string	"Friday "
	.text
	.globl	dayOfWeekByWeekDayNo
	.type	dayOfWeekByWeekDayNo, @function
dayOfWeekByWeekDayNo:
.LFB14:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, -20(%rbp)
	movq	$11, -8(%rbp)
.L241:
	cmpq	$16, -8(%rbp)
	ja	.L243
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
	.long	.L228-.L219
	.long	.L227-.L219
	.long	.L243-.L219
	.long	.L226-.L219
	.long	.L243-.L219
	.long	.L243-.L219
	.long	.L243-.L219
	.long	.L225-.L219
	.long	.L243-.L219
	.long	.L243-.L219
	.long	.L224-.L219
	.long	.L223-.L219
	.long	.L222-.L219
	.long	.L243-.L219
	.long	.L221-.L219
	.long	.L220-.L219
	.long	.L218-.L219
	.text
.L221:
	leaq	.LC27(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$10, -8(%rbp)
	jmp	.L229
.L220:
	leaq	.LC28(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$10, -8(%rbp)
	jmp	.L229
.L222:
	leaq	.LC29(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$10, -8(%rbp)
	jmp	.L229
.L227:
	leaq	.LC30(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$10, -8(%rbp)
	jmp	.L229
.L226:
	leaq	.LC31(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$10, -8(%rbp)
	jmp	.L229
.L218:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$10, -8(%rbp)
	jmp	.L229
.L223:
	cmpl	$6, -20(%rbp)
	ja	.L230
	movl	-20(%rbp), %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L232(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L232(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L232:
	.long	.L238-.L232
	.long	.L237-.L232
	.long	.L236-.L232
	.long	.L235-.L232
	.long	.L234-.L232
	.long	.L233-.L232
	.long	.L231-.L232
	.text
.L234:
	movq	$1, -8(%rbp)
	jmp	.L239
.L235:
	movq	$14, -8(%rbp)
	jmp	.L239
.L236:
	movq	$0, -8(%rbp)
	jmp	.L239
.L237:
	movq	$12, -8(%rbp)
	jmp	.L239
.L238:
	movq	$3, -8(%rbp)
	jmp	.L239
.L231:
	movq	$15, -8(%rbp)
	jmp	.L239
.L233:
	movq	$7, -8(%rbp)
	jmp	.L239
.L230:
	movq	$16, -8(%rbp)
	nop
.L239:
	jmp	.L229
.L224:
	movl	$0, %eax
	jmp	.L242
.L228:
	leaq	.LC32(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$10, -8(%rbp)
	jmp	.L229
.L225:
	leaq	.LC33(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$10, -8(%rbp)
	jmp	.L229
.L243:
	nop
.L229:
	jmp	.L241
.L242:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE14:
	.size	dayOfWeekByWeekDayNo, .-dayOfWeekByWeekDayNo
	.globl	gapBetweenDates
	.type	gapBetweenDates, @function
gapBetweenDates:
.LFB15:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$112, %rsp
	movl	%edi, -84(%rbp)
	movl	%esi, -88(%rbp)
	movl	%edx, -92(%rbp)
	movl	%ecx, -96(%rbp)
	movl	%r8d, -100(%rbp)
	movl	%r9d, -104(%rbp)
	movq	$15, -8(%rbp)
.L282:
	cmpq	$32, -8(%rbp)
	ja	.L283
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
	.long	.L267-.L247
	.long	.L266-.L247
	.long	.L283-.L247
	.long	.L265-.L247
	.long	.L283-.L247
	.long	.L283-.L247
	.long	.L264-.L247
	.long	.L263-.L247
	.long	.L283-.L247
	.long	.L262-.L247
	.long	.L261-.L247
	.long	.L260-.L247
	.long	.L283-.L247
	.long	.L259-.L247
	.long	.L258-.L247
	.long	.L257-.L247
	.long	.L256-.L247
	.long	.L283-.L247
	.long	.L255-.L247
	.long	.L283-.L247
	.long	.L254-.L247
	.long	.L283-.L247
	.long	.L283-.L247
	.long	.L253-.L247
	.long	.L252-.L247
	.long	.L251-.L247
	.long	.L283-.L247
	.long	.L250-.L247
	.long	.L249-.L247
	.long	.L248-.L247
	.long	.L283-.L247
	.long	.L283-.L247
	.long	.L246-.L247
	.text
.L255:
	movl	-104(%rbp), %edx
	movl	-52(%rbp), %eax
	movl	%edx, %esi
	movl	%eax, %edi
	call	daysInMonths
	movl	%eax, -20(%rbp)
	movl	-20(%rbp), %eax
	addl	%eax, -56(%rbp)
	addl	$1, -52(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L268
.L251:
	movl	-92(%rbp), %edx
	movl	-44(%rbp), %eax
	movl	%edx, %esi
	movl	%eax, %edi
	call	daysInMonths
	movl	%eax, -24(%rbp)
	movl	-24(%rbp), %eax
	addl	%eax, -48(%rbp)
	addl	$1, -44(%rbp)
	movq	$16, -8(%rbp)
	jmp	.L268
.L258:
	movl	-68(%rbp), %eax
	imull	-28(%rbp), %eax
	jmp	.L269
.L257:
	movq	$1, -8(%rbp)
	jmp	.L268
.L266:
	movl	-104(%rbp), %r8d
	movl	-100(%rbp), %edi
	movl	-96(%rbp), %ecx
	movl	-92(%rbp), %edx
	movl	-88(%rbp), %esi
	movl	-84(%rbp), %eax
	movl	%r8d, %r9d
	movl	%edi, %r8d
	movl	%eax, %edi
	call	compareDate
	movl	%eax, -12(%rbp)
	movl	-12(%rbp), %eax
	movl	%eax, -68(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L268
.L253:
	movl	$0, -48(%rbp)
	movl	$1, -44(%rbp)
	movq	$16, -8(%rbp)
	jmp	.L268
.L265:
	cmpl	$0, -68(%rbp)
	jne	.L270
	movq	$10, -8(%rbp)
	jmp	.L268
.L270:
	movq	$20, -8(%rbp)
	jmp	.L268
.L256:
	movl	-88(%rbp), %eax
	cmpl	-44(%rbp), %eax
	jle	.L272
	movq	$25, -8(%rbp)
	jmp	.L268
.L272:
	movq	$6, -8(%rbp)
	jmp	.L268
.L252:
	movl	-64(%rbp), %edx
	movl	-56(%rbp), %eax
	addl	%eax, %edx
	movl	-40(%rbp), %eax
	addl	%edx, %eax
	subl	-48(%rbp), %eax
	subl	-60(%rbp), %eax
	movl	%eax, -28(%rbp)
	movq	$14, -8(%rbp)
	jmp	.L268
.L260:
	movl	-104(%rbp), %eax
	cmpl	-36(%rbp), %eax
	jle	.L274
	movq	$13, -8(%rbp)
	jmp	.L268
.L274:
	movq	$24, -8(%rbp)
	jmp	.L268
.L262:
	cmpl	$0, -32(%rbp)
	je	.L276
	movq	$32, -8(%rbp)
	jmp	.L268
.L276:
	movq	$27, -8(%rbp)
	jmp	.L268
.L259:
	movl	-36(%rbp), %eax
	movl	%eax, %edi
	call	isLeapYear
	movl	%eax, -32(%rbp)
	movq	$9, -8(%rbp)
	jmp	.L268
.L246:
	addl	$366, -40(%rbp)
	movq	$28, -8(%rbp)
	jmp	.L268
.L264:
	movl	$0, -40(%rbp)
	movl	-92(%rbp), %eax
	movl	%eax, -36(%rbp)
	movq	$11, -8(%rbp)
	jmp	.L268
.L250:
	addl	$365, -40(%rbp)
	movq	$28, -8(%rbp)
	jmp	.L268
.L249:
	addl	$1, -36(%rbp)
	movq	$11, -8(%rbp)
	jmp	.L268
.L261:
	movl	$0, %eax
	jmp	.L269
.L267:
	movl	-100(%rbp), %eax
	cmpl	-52(%rbp), %eax
	jle	.L278
	movq	$18, -8(%rbp)
	jmp	.L268
.L278:
	movq	$23, -8(%rbp)
	jmp	.L268
.L263:
	movl	-84(%rbp), %eax
	movl	%eax, -16(%rbp)
	movl	-96(%rbp), %eax
	movl	%eax, -84(%rbp)
	movl	-16(%rbp), %eax
	movl	%eax, -96(%rbp)
	movl	-88(%rbp), %eax
	movl	%eax, -16(%rbp)
	movl	-100(%rbp), %eax
	movl	%eax, -88(%rbp)
	movl	-16(%rbp), %eax
	movl	%eax, -100(%rbp)
	movl	-92(%rbp), %eax
	movl	%eax, -16(%rbp)
	movl	-104(%rbp), %eax
	movl	%eax, -92(%rbp)
	movl	-16(%rbp), %eax
	movl	%eax, -104(%rbp)
	movq	$29, -8(%rbp)
	jmp	.L268
.L248:
	movl	-96(%rbp), %eax
	subl	$1, %eax
	movl	%eax, -64(%rbp)
	movl	-84(%rbp), %eax
	subl	$1, %eax
	movl	%eax, -60(%rbp)
	movl	$0, -56(%rbp)
	movl	$1, -52(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L268
.L254:
	cmpl	$-1, -68(%rbp)
	jne	.L280
	movq	$7, -8(%rbp)
	jmp	.L268
.L280:
	movq	$29, -8(%rbp)
	jmp	.L268
.L283:
	nop
.L268:
	jmp	.L282
.L269:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE15:
	.size	gapBetweenDates, .-gapBetweenDates
	.globl	compareDate
	.type	compareDate, @function
compareDate:
.LFB17:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -20(%rbp)
	movl	%esi, -24(%rbp)
	movl	%edx, -28(%rbp)
	movl	%ecx, -32(%rbp)
	movl	%r8d, -36(%rbp)
	movl	%r9d, -40(%rbp)
	movq	$12, -8(%rbp)
.L314:
	cmpq	$12, -8(%rbp)
	ja	.L315
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L287(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L287(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L287:
	.long	.L299-.L287
	.long	.L298-.L287
	.long	.L297-.L287
	.long	.L296-.L287
	.long	.L295-.L287
	.long	.L294-.L287
	.long	.L293-.L287
	.long	.L292-.L287
	.long	.L291-.L287
	.long	.L290-.L287
	.long	.L289-.L287
	.long	.L288-.L287
	.long	.L286-.L287
	.text
.L295:
	movl	-32(%rbp), %eax
	cmpl	-20(%rbp), %eax
	jge	.L300
	movq	$9, -8(%rbp)
	jmp	.L302
.L300:
	movq	$2, -8(%rbp)
	jmp	.L302
.L286:
	movl	-40(%rbp), %eax
	cmpl	-28(%rbp), %eax
	jle	.L303
	movq	$7, -8(%rbp)
	jmp	.L302
.L303:
	movq	$11, -8(%rbp)
	jmp	.L302
.L291:
	movl	-36(%rbp), %eax
	cmpl	-24(%rbp), %eax
	jge	.L305
	movq	$3, -8(%rbp)
	jmp	.L302
.L305:
	movq	$5, -8(%rbp)
	jmp	.L302
.L298:
	movl	$-1, %eax
	jmp	.L307
.L296:
	movl	$-1, %eax
	jmp	.L307
.L288:
	movl	-40(%rbp), %eax
	cmpl	-28(%rbp), %eax
	jge	.L308
	movq	$1, -8(%rbp)
	jmp	.L302
.L308:
	movq	$10, -8(%rbp)
	jmp	.L302
.L290:
	movl	$-1, %eax
	jmp	.L307
.L293:
	movl	$1, %eax
	jmp	.L307
.L294:
	movl	-32(%rbp), %eax
	cmpl	-20(%rbp), %eax
	jle	.L310
	movq	$0, -8(%rbp)
	jmp	.L302
.L310:
	movq	$4, -8(%rbp)
	jmp	.L302
.L289:
	movl	-36(%rbp), %eax
	cmpl	-24(%rbp), %eax
	jle	.L312
	movq	$6, -8(%rbp)
	jmp	.L302
.L312:
	movq	$8, -8(%rbp)
	jmp	.L302
.L299:
	movl	$1, %eax
	jmp	.L307
.L292:
	movl	$1, %eax
	jmp	.L307
.L297:
	movl	$0, %eax
	jmp	.L307
.L315:
	nop
.L302:
	jmp	.L314
.L307:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE17:
	.size	compareDate, .-compareDate
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
