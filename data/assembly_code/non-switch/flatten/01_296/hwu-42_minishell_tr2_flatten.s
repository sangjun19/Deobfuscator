	.file	"hwu-42_minishell_tr2_flatten.c"
	.text
	.globl	_TIG_IZ_if8Q_argv
	.bss
	.align 8
	.type	_TIG_IZ_if8Q_argv, @object
	.size	_TIG_IZ_if8Q_argv, 8
_TIG_IZ_if8Q_argv:
	.zero	8
	.globl	last_exit_status
	.align 4
	.type	last_exit_status, @object
	.size	last_exit_status, 4
last_exit_status:
	.zero	4
	.globl	_TIG_IZ_if8Q_envp
	.align 8
	.type	_TIG_IZ_if8Q_envp, @object
	.size	_TIG_IZ_if8Q_envp, 8
_TIG_IZ_if8Q_envp:
	.zero	8
	.globl	_TIG_IZ_if8Q_argc
	.align 4
	.type	_TIG_IZ_if8Q_argc, @object
	.size	_TIG_IZ_if8Q_argc, 4
_TIG_IZ_if8Q_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"execlp"
.LC1:
	.string	"fork"
	.text
	.globl	execute_command
	.type	execute_command, @function
execute_command:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, -16(%rbp)
.L26:
	cmpq	$13, -16(%rbp)
	ja	.L29
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
	.long	.L15-.L4
	.long	.L14-.L4
	.long	.L13-.L4
	.long	.L12-.L4
	.long	.L11-.L4
	.long	.L10-.L4
	.long	.L9-.L4
	.long	.L29-.L4
	.long	.L29-.L4
	.long	.L30-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L11:
	movl	-28(%rbp), %eax
	andl	$127, %eax
	addl	$1, %eax
	sarb	%al
	testb	%al, %al
	jle	.L16
	movq	$3, -16(%rbp)
	jmp	.L18
.L16:
	movq	$9, -16(%rbp)
	jmp	.L18
.L5:
	movl	-28(%rbp), %eax
	sarl	$8, %eax
	movzbl	%al, %eax
	movl	%eax, last_exit_status(%rip)
	movq	$9, -16(%rbp)
	jmp	.L18
.L14:
	call	fork@PLT
	movl	%eax, -20(%rbp)
	movl	-20(%rbp), %eax
	movl	%eax, -24(%rbp)
	movq	$6, -16(%rbp)
	jmp	.L18
.L12:
	movl	-28(%rbp), %eax
	andl	$127, %eax
	subl	$-128, %eax
	movl	%eax, last_exit_status(%rip)
	movq	$9, -16(%rbp)
	jmp	.L18
.L6:
	movl	-28(%rbp), %eax
	andl	$127, %eax
	testl	%eax, %eax
	jne	.L19
	movq	$12, -16(%rbp)
	jmp	.L18
.L19:
	movq	$4, -16(%rbp)
	jmp	.L18
.L3:
	movq	-40(%rbp), %rcx
	movq	-40(%rbp), %rax
	movl	$0, %edx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	execlp@PLT
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L9:
	cmpl	$-1, -24(%rbp)
	jne	.L22
	movq	$2, -16(%rbp)
	jmp	.L18
.L22:
	movq	$5, -16(%rbp)
	jmp	.L18
.L10:
	cmpl	$0, -24(%rbp)
	jne	.L24
	movq	$13, -16(%rbp)
	jmp	.L18
.L24:
	movq	$10, -16(%rbp)
	jmp	.L18
.L7:
	leaq	-28(%rbp), %rcx
	movl	-24(%rbp), %eax
	movl	$0, %edx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	waitpid@PLT
	movq	$11, -16(%rbp)
	jmp	.L18
.L15:
	movq	$1, -16(%rbp)
	jmp	.L18
.L13:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L29:
	nop
.L18:
	jmp	.L26
.L30:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L28
	call	__stack_chk_fail@PLT
.L28:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	execute_command, .-execute_command
	.section	.rodata
.LC2:
	.string	"%d"
	.text
	.globl	expand_dollar_question
	.type	expand_dollar_question, @function
expand_dollar_question:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$1104, %rsp
	movq	%rdi, -1096(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$3, -1072(%rbp)
.L52:
	cmpq	$14, -1072(%rbp)
	ja	.L55
	movq	-1072(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L34(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L34(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L34:
	.long	.L43-.L34
	.long	.L42-.L34
	.long	.L41-.L34
	.long	.L40-.L34
	.long	.L56-.L34
	.long	.L55-.L34
	.long	.L55-.L34
	.long	.L55-.L34
	.long	.L38-.L34
	.long	.L37-.L34
	.long	.L55-.L34
	.long	.L36-.L34
	.long	.L55-.L34
	.long	.L35-.L34
	.long	.L33-.L34
	.text
.L33:
	movl	last_exit_status(%rip), %edx
	leaq	-1040(%rbp), %rcx
	movq	-1080(%rbp), %rax
	subq	%rcx, %rax
	movq	%rax, %rsi
	movl	$1024, %eax
	subq	%rsi, %rax
	movq	%rax, %rsi
	movq	-1080(%rbp), %rax
	movl	%edx, %ecx
	leaq	.LC2(%rip), %rdx
	movq	%rax, %rdi
	movl	$0, %eax
	call	snprintf@PLT
	movq	-1080(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -1064(%rbp)
	movq	-1064(%rbp), %rax
	addq	%rax, -1080(%rbp)
	addq	$2, -1088(%rbp)
	movq	$0, -1072(%rbp)
	jmp	.L45
.L38:
	movq	-1080(%rbp), %rax
	movq	%rax, -1056(%rbp)
	addq	$1, -1080(%rbp)
	movq	-1088(%rbp), %rax
	movq	%rax, -1048(%rbp)
	addq	$1, -1088(%rbp)
	movq	-1048(%rbp), %rax
	movzbl	(%rax), %edx
	movq	-1056(%rbp), %rax
	movb	%dl, (%rax)
	movq	$0, -1072(%rbp)
	jmp	.L45
.L42:
	movq	-1088(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$36, %al
	jne	.L46
	movq	$11, -1072(%rbp)
	jmp	.L45
.L46:
	movq	$8, -1072(%rbp)
	jmp	.L45
.L40:
	movq	$2, -1072(%rbp)
	jmp	.L45
.L36:
	movq	-1088(%rbp), %rax
	addq	$1, %rax
	movzbl	(%rax), %eax
	cmpb	$63, %al
	jne	.L48
	movq	$14, -1072(%rbp)
	jmp	.L45
.L48:
	movq	$9, -1072(%rbp)
	jmp	.L45
.L37:
	movq	-1080(%rbp), %rax
	movq	%rax, -1056(%rbp)
	addq	$1, -1080(%rbp)
	movq	-1088(%rbp), %rax
	movq	%rax, -1048(%rbp)
	addq	$1, -1088(%rbp)
	movq	-1048(%rbp), %rax
	movzbl	(%rax), %edx
	movq	-1056(%rbp), %rax
	movb	%dl, (%rax)
	movq	$0, -1072(%rbp)
	jmp	.L45
.L35:
	movq	-1080(%rbp), %rax
	movb	$0, (%rax)
	leaq	-1040(%rbp), %rdx
	movq	-1096(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcpy@PLT
	movq	$4, -1072(%rbp)
	jmp	.L45
.L43:
	movq	-1088(%rbp), %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	je	.L50
	movq	$1, -1072(%rbp)
	jmp	.L45
.L50:
	movq	$13, -1072(%rbp)
	jmp	.L45
.L41:
	movq	-1096(%rbp), %rax
	movq	%rax, -1088(%rbp)
	leaq	-1040(%rbp), %rax
	movq	%rax, -1080(%rbp)
	movq	$0, -1072(%rbp)
	jmp	.L45
.L55:
	nop
.L45:
	jmp	.L52
.L56:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L54
	call	__stack_chk_fail@PLT
.L54:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	expand_dollar_question, .-expand_dollar_question
	.section	.rodata
.LC3:
	.string	"\n"
.LC4:
	.string	"fgets"
.LC5:
	.string	"minishell> "
	.text
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
	subq	$1104, %rsp
	movl	%edi, -1076(%rbp)
	movq	%rsi, -1088(%rbp)
	movq	%rdx, -1096(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movl	$0, last_exit_status(%rip)
	nop
.L58:
	movq	$0, _TIG_IZ_if8Q_envp(%rip)
	nop
.L59:
	movq	$0, _TIG_IZ_if8Q_argv(%rip)
	nop
.L60:
	movl	$0, _TIG_IZ_if8Q_argc(%rip)
	nop
	nop
.L61:
.L62:
#APP
# 131 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-if8Q--0
# 0 "" 2
#NO_APP
	movl	-1076(%rbp), %eax
	movl	%eax, _TIG_IZ_if8Q_argc(%rip)
	movq	-1088(%rbp), %rax
	movq	%rax, _TIG_IZ_if8Q_argv(%rip)
	movq	-1096(%rbp), %rax
	movq	%rax, _TIG_IZ_if8Q_envp(%rip)
	nop
	movq	$10, -1056(%rbp)
.L77:
	cmpq	$10, -1056(%rbp)
	ja	.L79
	movq	-1056(%rbp), %rax
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
	.long	.L79-.L65
	.long	.L71-.L65
	.long	.L70-.L65
	.long	.L79-.L65
	.long	.L79-.L65
	.long	.L69-.L65
	.long	.L79-.L65
	.long	.L68-.L65
	.long	.L67-.L65
	.long	.L66-.L65
	.long	.L64-.L65
	.text
.L67:
	leaq	-1040(%rbp), %rax
	leaq	.LC3(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcspn@PLT
	movq	%rax, -1048(%rbp)
	leaq	-1040(%rbp), %rdx
	movq	-1048(%rbp), %rax
	addq	%rdx, %rax
	movb	$0, (%rax)
	leaq	-1040(%rbp), %rax
	movq	%rax, %rdi
	call	expand_dollar_question
	leaq	-1040(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -1064(%rbp)
	movq	$1, -1056(%rbp)
	jmp	.L72
.L71:
	cmpq	$0, -1064(%rbp)
	je	.L73
	movq	$2, -1056(%rbp)
	jmp	.L72
.L73:
	movq	$7, -1056(%rbp)
	jmp	.L72
.L66:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L69:
	cmpq	$0, -1072(%rbp)
	jne	.L75
	movq	$9, -1056(%rbp)
	jmp	.L72
.L75:
	movq	$8, -1056(%rbp)
	jmp	.L72
.L64:
	movq	$7, -1056(%rbp)
	jmp	.L72
.L68:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	stdin(%rip), %rdx
	leaq	-1040(%rbp), %rax
	movl	$1024, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	movq	%rax, -1072(%rbp)
	movq	$5, -1056(%rbp)
	jmp	.L72
.L70:
	leaq	-1040(%rbp), %rax
	movq	%rax, %rdi
	call	execute_command
	movq	$7, -1056(%rbp)
	jmp	.L72
.L79:
	nop
.L72:
	jmp	.L77
	.cfi_endproc
.LFE6:
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
