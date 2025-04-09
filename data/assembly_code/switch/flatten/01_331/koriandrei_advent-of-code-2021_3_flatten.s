	.file	"koriandrei_advent-of-code-2021_3_flatten.c"
	.text
	.globl	_TIG_IZ_eRE2_argv
	.bss
	.align 8
	.type	_TIG_IZ_eRE2_argv, @object
	.size	_TIG_IZ_eRE2_argv, 8
_TIG_IZ_eRE2_argv:
	.zero	8
	.globl	_TIG_IZ_eRE2_envp
	.align 8
	.type	_TIG_IZ_eRE2_envp, @object
	.size	_TIG_IZ_eRE2_envp, 8
_TIG_IZ_eRE2_envp:
	.zero	8
	.globl	_TIG_IZ_eRE2_argc
	.align 4
	.type	_TIG_IZ_eRE2_argc, @object
	.size	_TIG_IZ_eRE2_argc, 4
_TIG_IZ_eRE2_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"Incomplete scores: %lld\n"
.LC1:
	.string	"Total cost of errors %d\n"
.LC2:
	.string	"No errors"
.LC3:
	.string	"%s"
.LC4:
	.string	"Line score %lld\n"
.LC5:
	.string	"Error symbol is %c\n"
.LC6:
	.string	"exit"
	.text
	.globl	main
	.type	main, @function
main:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$4096, %rsp
	orq	$0, (%rsp)
	subq	$4096, %rsp
	orq	$0, (%rsp)
	subq	$2176, %rsp
	movl	%edi, -10340(%rbp)
	movq	%rsi, -10352(%rbp)
	movq	%rdx, -10360(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_eRE2_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_eRE2_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_eRE2_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 123 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-eRE2--0
# 0 "" 2
#NO_APP
	movl	-10340(%rbp), %eax
	movl	%eax, _TIG_IZ_eRE2_argc(%rip)
	movq	-10352(%rbp), %rax
	movq	%rax, _TIG_IZ_eRE2_argv(%rip)
	movq	-10360(%rbp), %rax
	movq	%rax, _TIG_IZ_eRE2_envp(%rip)
	nop
	movq	$40, -10280(%rbp)
.L65:
	cmpq	$48, -10280(%rbp)
	ja	.L68
	movq	-10280(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L8(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L8(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L8:
	.long	.L68-.L8
	.long	.L38-.L8
	.long	.L37-.L8
	.long	.L36-.L8
	.long	.L35-.L8
	.long	.L34-.L8
	.long	.L68-.L8
	.long	.L33-.L8
	.long	.L32-.L8
	.long	.L31-.L8
	.long	.L68-.L8
	.long	.L30-.L8
	.long	.L68-.L8
	.long	.L29-.L8
	.long	.L68-.L8
	.long	.L68-.L8
	.long	.L68-.L8
	.long	.L68-.L8
	.long	.L28-.L8
	.long	.L27-.L8
	.long	.L26-.L8
	.long	.L68-.L8
	.long	.L25-.L8
	.long	.L68-.L8
	.long	.L24-.L8
	.long	.L23-.L8
	.long	.L22-.L8
	.long	.L21-.L8
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L68-.L8
	.long	.L68-.L8
	.long	.L68-.L8
	.long	.L14-.L8
	.long	.L68-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L68-.L8
	.long	.L68-.L8
	.long	.L11-.L8
	.long	.L68-.L8
	.long	.L68-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L28:
	cmpl	$-1, -10296(%rbp)
	jge	.L39
	movq	$32, -10280(%rbp)
	jmp	.L41
.L39:
	movq	$8, -10280(%rbp)
	jmp	.L41
.L23:
	movzbl	-10320(%rbp), %eax
	movb	%al, -10321(%rbp)
	movq	$32, -10280(%rbp)
	jmp	.L41
.L35:
	cmpl	$0, -10308(%rbp)
	jle	.L42
	movq	$30, -10280(%rbp)
	jmp	.L41
.L42:
	movq	$13, -10280(%rbp)
	jmp	.L41
.L18:
	movl	-10308(%rbp), %eax
	movl	%eax, %edx
	shrl	$31, %edx
	addl	%edx, %eax
	sarl	%eax
	cltq
	movq	-10256(%rbp,%rax,8), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$13, -10280(%rbp)
	jmp	.L41
.L17:
	cmpl	$0, -10300(%rbp)
	jne	.L44
	movq	$26, -10280(%rbp)
	jmp	.L41
.L44:
	movq	$33, -10280(%rbp)
	jmp	.L41
.L32:
	addl	$1, -10292(%rbp)
	movq	$1, -10280(%rbp)
	jmp	.L41
.L38:
	cmpl	$1023, -10292(%rbp)
	jg	.L46
	movq	$46, -10280(%rbp)
	jmp	.L41
.L46:
	movq	$32, -10280(%rbp)
	jmp	.L41
.L36:
	cmpb	$0, -10318(%rbp)
	je	.L48
	movq	$24, -10280(%rbp)
	jmp	.L41
.L48:
	movq	$25, -10280(%rbp)
	jmp	.L41
.L24:
	subl	$1, -10296(%rbp)
	movq	$18, -10280(%rbp)
	jmp	.L41
.L22:
	movl	-10316(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-10308(%rbp), %eax
	movslq	%eax, %rsi
	leaq	-10256(%rbp), %rax
	leaq	comparison(%rip), %rdx
	movq	%rdx, %rcx
	movl	$8, %edx
	movq	%rax, %rdi
	call	qsort@PLT
	movq	$4, -10280(%rbp)
	jmp	.L41
.L30:
	cmpl	$0, -10296(%rbp)
	jns	.L50
	movq	$32, -10280(%rbp)
	jmp	.L41
.L50:
	movq	$5, -10280(%rbp)
	jmp	.L41
.L31:
	cmpl	$1, -10304(%rbp)
	je	.L52
	movq	$26, -10280(%rbp)
	jmp	.L41
.L52:
	movq	$20, -10280(%rbp)
	jmp	.L41
.L29:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L66
	jmp	.L67
.L27:
	cmpl	$0, -10296(%rbp)
	jle	.L55
	movq	$47, -10280(%rbp)
	jmp	.L41
.L55:
	movq	$27, -10280(%rbp)
	jmp	.L41
.L16:
	cmpb	$0, -10321(%rbp)
	je	.L57
	movq	$29, -10280(%rbp)
	jmp	.L41
.L57:
	movq	$19, -10280(%rbp)
	jmp	.L41
.L12:
	movq	$48, -10280(%rbp)
	jmp	.L41
.L21:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$28, -10280(%rbp)
	jmp	.L41
.L7:
	movl	$0, -10316(%rbp)
	movq	$0, -10256(%rbp)
	movl	$1, -10312(%rbp)
	movq	$2, -10280(%rbp)
	jmp	.L41
.L25:
	movl	-10292(%rbp), %eax
	cltq
	movzbl	-2064(%rbp,%rax), %eax
	movb	%al, -10320(%rbp)
	movsbl	-10320(%rbp), %eax
	movl	%eax, %edi
	call	is_opening_brace
	movb	%al, -10317(%rbp)
	movzbl	-10317(%rbp), %eax
	movb	%al, -10319(%rbp)
	movq	$39, -10280(%rbp)
	jmp	.L41
.L20:
	leaq	-2064(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	%eax, -10304(%rbp)
	movq	$9, -10280(%rbp)
	jmp	.L41
.L9:
	movl	-10296(%rbp), %edx
	leaq	-1040(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	calculate_incomplete_line_score
	movq	%rax, -10272(%rbp)
	movq	-10272(%rbp), %rax
	movq	%rax, -10264(%rbp)
	movq	-10264(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-10308(%rbp), %eax
	movl	%eax, -10288(%rbp)
	addl	$1, -10308(%rbp)
	movl	-10288(%rbp), %eax
	cltq
	movq	-10264(%rbp), %rdx
	movq	%rdx, -10256(%rbp,%rax,8)
	movq	$28, -10280(%rbp)
	jmp	.L41
.L34:
	movsbl	-10320(%rbp), %edx
	movl	-10296(%rbp), %eax
	cltq
	movzbl	-1040(%rbp,%rax), %eax
	movsbl	%al, %eax
	movl	%edx, %esi
	movl	%eax, %edi
	call	is_matching_closing_brace
	movb	%al, -10318(%rbp)
	movq	$3, -10280(%rbp)
	jmp	.L41
.L15:
	movl	$-1, -10296(%rbp)
	movb	$0, -10321(%rbp)
	movl	$0, -10292(%rbp)
	movq	$1, -10280(%rbp)
	jmp	.L41
.L14:
	movl	$0, -10308(%rbp)
	movq	$28, -10280(%rbp)
	jmp	.L41
.L10:
	movl	-10292(%rbp), %eax
	cltq
	movzbl	-2064(%rbp,%rax), %eax
	testb	%al, %al
	je	.L59
	movq	$22, -10280(%rbp)
	jmp	.L41
.L59:
	movq	$32, -10280(%rbp)
	jmp	.L41
.L13:
	cmpb	$0, -10319(%rbp)
	je	.L61
	movq	$43, -10280(%rbp)
	jmp	.L41
.L61:
	movq	$11, -10280(%rbp)
	jmp	.L41
.L33:
	movl	-10312(%rbp), %eax
	movq	$0, -10256(%rbp,%rax,8)
	addl	$1, -10312(%rbp)
	movq	$2, -10280(%rbp)
	jmp	.L41
.L19:
	movsbl	-10321(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movsbl	-10321(%rbp), %eax
	movl	%eax, %edi
	call	get_error_symbol_cost
	movl	%eax, -10284(%rbp)
	movl	-10284(%rbp), %eax
	addl	%eax, -10316(%rbp)
	movq	$28, -10280(%rbp)
	jmp	.L41
.L11:
	addl	$1, -10296(%rbp)
	movl	-10296(%rbp), %eax
	cltq
	movzbl	-10320(%rbp), %edx
	movb	%dl, -1040(%rbp,%rax)
	movq	$8, -10280(%rbp)
	jmp	.L41
.L37:
	cmpl	$1023, -10312(%rbp)
	jbe	.L63
	movq	$37, -10280(%rbp)
	jmp	.L41
.L63:
	movq	$7, -10280(%rbp)
	jmp	.L41
.L26:
	leaq	-2064(%rbp), %rax
	leaq	.LC6(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -10300(%rbp)
	movq	$31, -10280(%rbp)
	jmp	.L41
.L68:
	nop
.L41:
	jmp	.L65
.L67:
	call	__stack_chk_fail@PLT
.L66:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	main, .-main
	.globl	calculate_incomplete_line_score
	.type	calculate_incomplete_line_score, @function
calculate_incomplete_line_score:
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
	movl	%esi, -44(%rbp)
	movq	$5, -8(%rbp)
.L81:
	cmpq	$6, -8(%rbp)
	ja	.L83
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L72(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L72(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L72:
	.long	.L76-.L72
	.long	.L83-.L72
	.long	.L75-.L72
	.long	.L83-.L72
	.long	.L74-.L72
	.long	.L73-.L72
	.long	.L71-.L72
	.text
.L74:
	movq	-16(%rbp), %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	movq	%rax, -16(%rbp)
	movl	-24(%rbp), %eax
	movslq	%eax, %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	movl	%eax, %edi
	call	get_incomplete_symbol_cost
	movl	%eax, -20(%rbp)
	movl	-20(%rbp), %eax
	cltq
	addq	%rax, -16(%rbp)
	subl	$1, -24(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L77
.L71:
	movq	-16(%rbp), %rax
	jmp	.L82
.L73:
	movq	$0, -8(%rbp)
	jmp	.L77
.L76:
	movq	$0, -16(%rbp)
	movl	-44(%rbp), %eax
	movl	%eax, -24(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L77
.L75:
	cmpl	$0, -24(%rbp)
	js	.L79
	movq	$4, -8(%rbp)
	jmp	.L77
.L79:
	movq	$6, -8(%rbp)
	jmp	.L77
.L83:
	nop
.L77:
	jmp	.L81
.L82:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	calculate_incomplete_line_score, .-calculate_incomplete_line_score
	.globl	get_error_symbol_cost
	.type	get_error_symbol_cost, @function
get_error_symbol_cost:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, %eax
	movb	%al, -20(%rbp)
	movq	$0, -8(%rbp)
.L101:
	cmpq	$6, -8(%rbp)
	ja	.L102
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L87(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L87(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L87:
	.long	.L93-.L87
	.long	.L92-.L87
	.long	.L91-.L87
	.long	.L90-.L87
	.long	.L89-.L87
	.long	.L88-.L87
	.long	.L86-.L87
	.text
.L89:
	movl	$57, %eax
	jmp	.L94
.L92:
	movl	$25137, %eax
	jmp	.L94
.L90:
	movq	$5, -8(%rbp)
	jmp	.L95
.L86:
	movl	$1197, %eax
	jmp	.L94
.L88:
	movl	$-1, %eax
	jmp	.L94
.L93:
	movsbl	-20(%rbp), %eax
	cmpl	$125, %eax
	je	.L96
	cmpl	$125, %eax
	jg	.L97
	cmpl	$93, %eax
	je	.L98
	cmpl	$93, %eax
	jg	.L97
	cmpl	$41, %eax
	je	.L99
	cmpl	$62, %eax
	jne	.L97
	movq	$1, -8(%rbp)
	jmp	.L100
.L96:
	movq	$6, -8(%rbp)
	jmp	.L100
.L98:
	movq	$4, -8(%rbp)
	jmp	.L100
.L99:
	movq	$2, -8(%rbp)
	jmp	.L100
.L97:
	movq	$3, -8(%rbp)
	nop
.L100:
	jmp	.L95
.L91:
	movl	$3, %eax
	jmp	.L94
.L102:
	nop
.L95:
	jmp	.L101
.L94:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	get_error_symbol_cost, .-get_error_symbol_cost
	.globl	comparison
	.type	comparison, @function
comparison:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -40(%rbp)
	movq	%rsi, -48(%rbp)
	movq	$1, -8(%rbp)
.L119:
	cmpq	$6, -8(%rbp)
	ja	.L120
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L106(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L106(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L106:
	.long	.L112-.L106
	.long	.L111-.L106
	.long	.L110-.L106
	.long	.L109-.L106
	.long	.L108-.L106
	.long	.L107-.L106
	.long	.L105-.L106
	.text
.L108:
	movl	$-1, %eax
	jmp	.L113
.L111:
	movq	$6, -8(%rbp)
	jmp	.L114
.L109:
	movq	-24(%rbp), %rax
	cmpq	-16(%rbp), %rax
	jle	.L115
	movq	$0, -8(%rbp)
	jmp	.L114
.L115:
	movq	$5, -8(%rbp)
	jmp	.L114
.L105:
	movq	-40(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	-48(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, -16(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L114
.L107:
	movq	-24(%rbp), %rax
	cmpq	-16(%rbp), %rax
	jge	.L117
	movq	$4, -8(%rbp)
	jmp	.L114
.L117:
	movq	$2, -8(%rbp)
	jmp	.L114
.L112:
	movl	$1, %eax
	jmp	.L113
.L110:
	movl	$0, %eax
	jmp	.L113
.L120:
	nop
.L114:
	jmp	.L119
.L113:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	comparison, .-comparison
	.globl	is_opening_brace
	.type	is_opening_brace, @function
is_opening_brace:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, %eax
	movb	%al, -20(%rbp)
	movq	$3, -8(%rbp)
.L144:
	cmpq	$9, -8(%rbp)
	ja	.L146
	movq	-8(%rbp), %rax
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
	.long	.L133-.L124
	.long	.L132-.L124
	.long	.L131-.L124
	.long	.L130-.L124
	.long	.L129-.L124
	.long	.L128-.L124
	.long	.L127-.L124
	.long	.L126-.L124
	.long	.L125-.L124
	.long	.L123-.L124
	.text
.L129:
	cmpb	$60, -20(%rbp)
	jne	.L134
	movq	$7, -8(%rbp)
	jmp	.L136
.L134:
	movq	$1, -8(%rbp)
	jmp	.L136
.L125:
	movl	-12(%rbp), %eax
	jmp	.L145
.L132:
	movl	$0, -12(%rbp)
	movq	$8, -8(%rbp)
	jmp	.L136
.L130:
	cmpb	$40, -20(%rbp)
	jne	.L138
	movq	$5, -8(%rbp)
	jmp	.L136
.L138:
	movq	$6, -8(%rbp)
	jmp	.L136
.L123:
	cmpb	$123, -20(%rbp)
	jne	.L140
	movq	$2, -8(%rbp)
	jmp	.L136
.L140:
	movq	$4, -8(%rbp)
	jmp	.L136
.L127:
	cmpb	$91, -20(%rbp)
	jne	.L142
	movq	$0, -8(%rbp)
	jmp	.L136
.L142:
	movq	$9, -8(%rbp)
	jmp	.L136
.L128:
	movl	$1, -12(%rbp)
	movq	$8, -8(%rbp)
	jmp	.L136
.L133:
	movl	$1, -12(%rbp)
	movq	$8, -8(%rbp)
	jmp	.L136
.L126:
	movl	$1, -12(%rbp)
	movq	$8, -8(%rbp)
	jmp	.L136
.L131:
	movl	$1, -12(%rbp)
	movq	$8, -8(%rbp)
	jmp	.L136
.L146:
	nop
.L136:
	jmp	.L144
.L145:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	is_opening_brace, .-is_opening_brace
	.globl	is_matching_closing_brace
	.type	is_matching_closing_brace, @function
is_matching_closing_brace:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, %edx
	movl	%esi, %eax
	movb	%dl, -20(%rbp)
	movb	%al, -24(%rbp)
	movq	$14, -8(%rbp)
.L183:
	cmpq	$17, -8(%rbp)
	ja	.L185
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
	.long	.L164-.L150
	.long	.L163-.L150
	.long	.L162-.L150
	.long	.L161-.L150
	.long	.L160-.L150
	.long	.L159-.L150
	.long	.L158-.L150
	.long	.L157-.L150
	.long	.L156-.L150
	.long	.L155-.L150
	.long	.L185-.L150
	.long	.L154-.L150
	.long	.L185-.L150
	.long	.L153-.L150
	.long	.L152-.L150
	.long	.L151-.L150
	.long	.L185-.L150
	.long	.L149-.L150
	.text
.L160:
	cmpb	$41, -24(%rbp)
	jne	.L165
	movq	$8, -8(%rbp)
	jmp	.L167
.L165:
	movq	$17, -8(%rbp)
	jmp	.L167
.L152:
	cmpb	$40, -20(%rbp)
	jne	.L168
	movq	$4, -8(%rbp)
	jmp	.L167
.L168:
	movq	$17, -8(%rbp)
	jmp	.L167
.L151:
	movl	$1, -12(%rbp)
	movq	$11, -8(%rbp)
	jmp	.L167
.L156:
	movl	$1, -12(%rbp)
	movq	$11, -8(%rbp)
	jmp	.L167
.L163:
	cmpb	$125, -24(%rbp)
	jne	.L170
	movq	$15, -8(%rbp)
	jmp	.L167
.L170:
	movq	$3, -8(%rbp)
	jmp	.L167
.L161:
	cmpb	$60, -20(%rbp)
	jne	.L172
	movq	$6, -8(%rbp)
	jmp	.L167
.L172:
	movq	$5, -8(%rbp)
	jmp	.L167
.L154:
	movl	-12(%rbp), %eax
	jmp	.L184
.L155:
	cmpb	$123, -20(%rbp)
	jne	.L175
	movq	$1, -8(%rbp)
	jmp	.L167
.L175:
	movq	$3, -8(%rbp)
	jmp	.L167
.L153:
	cmpb	$93, -24(%rbp)
	jne	.L177
	movq	$7, -8(%rbp)
	jmp	.L167
.L177:
	movq	$9, -8(%rbp)
	jmp	.L167
.L149:
	cmpb	$91, -20(%rbp)
	jne	.L179
	movq	$13, -8(%rbp)
	jmp	.L167
.L179:
	movq	$9, -8(%rbp)
	jmp	.L167
.L158:
	cmpb	$62, -24(%rbp)
	jne	.L181
	movq	$0, -8(%rbp)
	jmp	.L167
.L181:
	movq	$2, -8(%rbp)
	jmp	.L167
.L159:
	movl	$0, -12(%rbp)
	movq	$11, -8(%rbp)
	jmp	.L167
.L164:
	movl	$1, -12(%rbp)
	movq	$11, -8(%rbp)
	jmp	.L167
.L157:
	movl	$1, -12(%rbp)
	movq	$11, -8(%rbp)
	jmp	.L167
.L162:
	movl	$0, -12(%rbp)
	movq	$11, -8(%rbp)
	jmp	.L167
.L185:
	nop
.L167:
	jmp	.L183
.L184:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	is_matching_closing_brace, .-is_matching_closing_brace
	.globl	get_incomplete_symbol_cost
	.type	get_incomplete_symbol_cost, @function
get_incomplete_symbol_cost:
.LFB10:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, %eax
	movb	%al, -20(%rbp)
	movq	$1, -8(%rbp)
.L203:
	cmpq	$6, -8(%rbp)
	ja	.L204
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L189(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L189(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L189:
	.long	.L195-.L189
	.long	.L194-.L189
	.long	.L193-.L189
	.long	.L192-.L189
	.long	.L191-.L189
	.long	.L190-.L189
	.long	.L188-.L189
	.text
.L191:
	movl	$3, %eax
	jmp	.L196
.L194:
	movsbl	-20(%rbp), %eax
	cmpl	$123, %eax
	je	.L197
	cmpl	$123, %eax
	jg	.L198
	cmpl	$91, %eax
	je	.L199
	cmpl	$91, %eax
	jg	.L198
	cmpl	$40, %eax
	je	.L200
	cmpl	$60, %eax
	jne	.L198
	movq	$6, -8(%rbp)
	jmp	.L201
.L197:
	movq	$4, -8(%rbp)
	jmp	.L201
.L199:
	movq	$5, -8(%rbp)
	jmp	.L201
.L200:
	movq	$3, -8(%rbp)
	jmp	.L201
.L198:
	movq	$2, -8(%rbp)
	nop
.L201:
	jmp	.L202
.L192:
	movl	$1, %eax
	jmp	.L196
.L188:
	movl	$4, %eax
	jmp	.L196
.L190:
	movl	$2, %eax
	jmp	.L196
.L195:
	movl	$-1, %eax
	jmp	.L196
.L193:
	movq	$0, -8(%rbp)
	jmp	.L202
.L204:
	nop
.L202:
	jmp	.L203
.L196:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	get_incomplete_symbol_cost, .-get_incomplete_symbol_cost
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
