	.file	"JingMing2019_cs5008-sp2022-assignments-JingMing_hw7_flatten.c"
	.text
	.globl	_TIG_IZ_XhHV_argv
	.bss
	.align 8
	.type	_TIG_IZ_XhHV_argv, @object
	.size	_TIG_IZ_XhHV_argv, 8
_TIG_IZ_XhHV_argv:
	.zero	8
	.globl	_TIG_IZ_XhHV_envp
	.align 8
	.type	_TIG_IZ_XhHV_envp, @object
	.size	_TIG_IZ_XhHV_envp, 8
_TIG_IZ_XhHV_envp:
	.zero	8
	.globl	_TIG_IZ_XhHV_argc
	.align 4
	.type	_TIG_IZ_XhHV_argc, @object
	.size	_TIG_IZ_XhHV_argc, 4
_TIG_IZ_XhHV_argc:
	.zero	4
	.text
	.globl	appendChar
	.type	appendChar, @function
appendChar:
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
	movl	%esi, %eax
	movb	%al, -44(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, -24(%rbp)
.L7:
	cmpq	$2, -24(%rbp)
	je	.L10
	cmpq	$2, -24(%rbp)
	ja	.L11
	cmpq	$0, -24(%rbp)
	je	.L4
	cmpq	$1, -24(%rbp)
	jne	.L11
	movzbl	-44(%rbp), %eax
	movb	%al, -10(%rbp)
	movb	$0, -9(%rbp)
	leaq	-10(%rbp), %rdx
	movq	-40(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcat@PLT
	movq	$2, -24(%rbp)
	jmp	.L5
.L4:
	movq	$1, -24(%rbp)
	jmp	.L5
.L11:
	nop
.L5:
	jmp	.L7
.L10:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L9
	call	__stack_chk_fail@PLT
.L9:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	appendChar, .-appendChar
	.section	.rodata
.LC0:
	.string	"r"
.LC1:
	.string	"pop.csv"
.LC2:
	.string	"File not found!"
.LC3:
	.string	"> %.60s\n"
.LC4:
	.string	"%d, [%.30s]: %d\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$720, %rsp
	movl	%edi, -692(%rbp)
	movq	%rsi, -704(%rbp)
	movq	%rdx, -712(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_XhHV_envp(%rip)
	nop
.L13:
	movq	$0, _TIG_IZ_XhHV_argv(%rip)
	nop
.L14:
	movl	$0, _TIG_IZ_XhHV_argc(%rip)
	nop
	nop
.L15:
.L16:
#APP
# 114 "JingMing2019_cs5008-sp2022-assignments-JingMing_hw7.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-XhHV--0
# 0 "" 2
#NO_APP
	movl	-692(%rbp), %eax
	movl	%eax, _TIG_IZ_XhHV_argc(%rip)
	movq	-704(%rbp), %rax
	movq	%rax, _TIG_IZ_XhHV_argv(%rip)
	movq	-712(%rbp), %rax
	movq	%rax, _TIG_IZ_XhHV_envp(%rip)
	nop
	movq	$59, -632(%rbp)
.L113:
	cmpq	$72, -632(%rbp)
	ja	.L116
	movq	-632(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L19(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L19(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L19:
	.long	.L67-.L19
	.long	.L66-.L19
	.long	.L65-.L19
	.long	.L116-.L19
	.long	.L64-.L19
	.long	.L116-.L19
	.long	.L63-.L19
	.long	.L62-.L19
	.long	.L61-.L19
	.long	.L60-.L19
	.long	.L59-.L19
	.long	.L116-.L19
	.long	.L116-.L19
	.long	.L116-.L19
	.long	.L58-.L19
	.long	.L57-.L19
	.long	.L116-.L19
	.long	.L116-.L19
	.long	.L56-.L19
	.long	.L55-.L19
	.long	.L54-.L19
	.long	.L116-.L19
	.long	.L53-.L19
	.long	.L52-.L19
	.long	.L116-.L19
	.long	.L116-.L19
	.long	.L51-.L19
	.long	.L116-.L19
	.long	.L50-.L19
	.long	.L49-.L19
	.long	.L48-.L19
	.long	.L47-.L19
	.long	.L46-.L19
	.long	.L116-.L19
	.long	.L45-.L19
	.long	.L44-.L19
	.long	.L43-.L19
	.long	.L116-.L19
	.long	.L42-.L19
	.long	.L41-.L19
	.long	.L40-.L19
	.long	.L116-.L19
	.long	.L39-.L19
	.long	.L116-.L19
	.long	.L38-.L19
	.long	.L37-.L19
	.long	.L36-.L19
	.long	.L116-.L19
	.long	.L35-.L19
	.long	.L34-.L19
	.long	.L116-.L19
	.long	.L116-.L19
	.long	.L33-.L19
	.long	.L116-.L19
	.long	.L32-.L19
	.long	.L31-.L19
	.long	.L116-.L19
	.long	.L30-.L19
	.long	.L29-.L19
	.long	.L28-.L19
	.long	.L27-.L19
	.long	.L116-.L19
	.long	.L116-.L19
	.long	.L26-.L19
	.long	.L25-.L19
	.long	.L24-.L19
	.long	.L23-.L19
	.long	.L22-.L19
	.long	.L21-.L19
	.long	.L20-.L19
	.long	.L116-.L19
	.long	.L116-.L19
	.long	.L18-.L19
	.text
.L56:
	movl	$10, -668(%rbp)
	movq	$36, -632(%rbp)
	jmp	.L68
.L34:
	movl	-664(%rbp), %eax
	cltq
	movzbl	-624(%rbp,%rax), %eax
	cmpb	$34, %al
	jne	.L69
	movq	$18, -632(%rbp)
	jmp	.L68
.L69:
	movq	$26, -632(%rbp)
	jmp	.L68
.L33:
	movl	$6, -668(%rbp)
	movl	$0, -672(%rbp)
	movq	$36, -632(%rbp)
	jmp	.L68
.L64:
	movl	$1, -668(%rbp)
	movl	-664(%rbp), %eax
	cltq
	movzbl	-624(%rbp,%rax), %eax
	movsbl	%al, %eax
	subl	$48, %eax
	movl	%eax, -676(%rbp)
	movq	$36, -632(%rbp)
	jmp	.L68
.L48:
	movl	$3, -668(%rbp)
	movl	-664(%rbp), %eax
	cltq
	movzbl	-624(%rbp,%rax), %eax
	movsbl	%al, %edx
	leaq	-208(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	appendChar
	movq	$36, -632(%rbp)
	jmp	.L68
.L58:
	movl	-664(%rbp), %eax
	cltq
	movzbl	-624(%rbp,%rax), %eax
	movsbl	%al, %eax
	movl	%eax, %edi
	call	isDigit
	movb	%al, -678(%rbp)
	movq	$22, -632(%rbp)
	jmp	.L68
.L57:
	movl	$11, -668(%rbp)
	movq	$36, -632(%rbp)
	jmp	.L68
.L47:
	movl	$11, -668(%rbp)
	movq	$36, -632(%rbp)
	jmp	.L68
.L20:
	movl	-664(%rbp), %eax
	cltq
	movzbl	-624(%rbp,%rax), %eax
	movsbl	%al, %eax
	movl	%eax, %edi
	call	isDigit
	movb	%al, -677(%rbp)
	movq	$19, -632(%rbp)
	jmp	.L68
.L61:
	movq	-648(%rbp), %rdx
	leaq	-624(%rbp), %rax
	movl	$200, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	movq	$54, -632(%rbp)
	jmp	.L68
.L37:
	movl	$1, -668(%rbp)
	movl	-664(%rbp), %eax
	cltq
	movzbl	-624(%rbp,%rax), %eax
	movsbl	%al, %eax
	subl	$48, %eax
	movl	%eax, -656(%rbp)
	movl	-676(%rbp), %edx
	movl	%edx, %eax
	sall	$2, %eax
	addl	%edx, %eax
	addl	%eax, %eax
	movl	%eax, %edx
	movl	-656(%rbp), %eax
	addl	%edx, %eax
	movl	%eax, -676(%rbp)
	movq	$36, -632(%rbp)
	jmp	.L68
.L32:
	movq	-648(%rbp), %rax
	movq	%rax, %rdi
	call	feof@PLT
	movl	%eax, -660(%rbp)
	movq	$66, -632(%rbp)
	jmp	.L68
.L66:
	movq	-648(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$6, -632(%rbp)
	jmp	.L68
.L52:
	movl	-664(%rbp), %eax
	cltq
	movzbl	-624(%rbp,%rax), %eax
	movsbl	%al, %eax
	movl	%eax, %edi
	call	isDigit
	movb	%al, -679(%rbp)
	movq	$46, -632(%rbp)
	jmp	.L68
.L43:
	addl	$1, -664(%rbp)
	movq	$0, -632(%rbp)
	jmp	.L68
.L30:
	movl	$2, -668(%rbp)
	movb	$0, -208(%rbp)
	movq	$36, -632(%rbp)
	jmp	.L68
.L21:
	movl	-664(%rbp), %eax
	cltq
	movzbl	-624(%rbp,%rax), %eax
	cmpb	$44, %al
	jne	.L71
	movq	$10, -632(%rbp)
	jmp	.L68
.L71:
	movq	$49, -632(%rbp)
	jmp	.L68
.L51:
	movl	$11, -668(%rbp)
	movq	$36, -632(%rbp)
	jmp	.L68
.L60:
	movl	-664(%rbp), %eax
	cltq
	cmpq	%rax, -640(%rbp)
	ja	.L73
	movq	$60, -632(%rbp)
	jmp	.L68
.L73:
	movq	$0, -632(%rbp)
	jmp	.L68
.L26:
	cmpq	$0, -648(%rbp)
	je	.L75
	movq	$8, -632(%rbp)
	jmp	.L68
.L75:
	movq	$42, -632(%rbp)
	jmp	.L68
.L55:
	cmpb	$0, -677(%rbp)
	je	.L77
	movq	$40, -632(%rbp)
	jmp	.L68
.L77:
	movq	$68, -632(%rbp)
	jmp	.L68
.L46:
	movl	$11, -668(%rbp)
	movq	$36, -632(%rbp)
	jmp	.L68
.L40:
	movl	$6, -668(%rbp)
	movl	-664(%rbp), %eax
	cltq
	movzbl	-624(%rbp,%rax), %eax
	movsbl	%al, %eax
	subl	$48, %eax
	movl	%eax, -652(%rbp)
	movl	-672(%rbp), %edx
	movl	%edx, %eax
	sall	$2, %eax
	addl	%edx, %eax
	addl	%eax, %eax
	movl	%eax, %edx
	movl	-652(%rbp), %eax
	addl	%edx, %eax
	movl	%eax, -672(%rbp)
	movq	$36, -632(%rbp)
	jmp	.L68
.L22:
	movl	-664(%rbp), %eax
	cltq
	movzbl	-624(%rbp,%rax), %eax
	cmpb	$44, %al
	jne	.L79
	movq	$58, -632(%rbp)
	jmp	.L68
.L79:
	movq	$32, -632(%rbp)
	jmp	.L68
.L31:
	cmpl	$10, -668(%rbp)
	ja	.L81
	movl	-668(%rbp), %eax
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
	.long	.L90-.L83
	.long	.L89-.L83
	.long	.L88-.L83
	.long	.L87-.L83
	.long	.L86-.L83
	.long	.L85-.L83
	.long	.L84-.L83
	.long	.L81-.L83
	.long	.L81-.L83
	.long	.L81-.L83
	.long	.L82-.L83
	.text
.L82:
	movq	$36, -632(%rbp)
	jmp	.L91
.L84:
	movq	$69, -632(%rbp)
	jmp	.L91
.L85:
	movq	$39, -632(%rbp)
	jmp	.L91
.L86:
	movq	$67, -632(%rbp)
	jmp	.L91
.L87:
	movq	$2, -632(%rbp)
	jmp	.L91
.L88:
	movq	$44, -632(%rbp)
	jmp	.L91
.L89:
	movq	$14, -632(%rbp)
	jmp	.L91
.L90:
	movq	$23, -632(%rbp)
	jmp	.L91
.L81:
	movq	$29, -632(%rbp)
	nop
.L91:
	jmp	.L68
.L27:
	movl	$11, -668(%rbp)
	movq	$0, -632(%rbp)
	jmp	.L68
.L28:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -648(%rbp)
	movq	$63, -632(%rbp)
	jmp	.L68
.L63:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L114
	jmp	.L115
.L42:
	cmpl	$10, -668(%rbp)
	je	.L93
	movq	$55, -632(%rbp)
	jmp	.L68
.L93:
	movq	$20, -632(%rbp)
	jmp	.L68
.L29:
	movl	$5, -668(%rbp)
	movq	$36, -632(%rbp)
	jmp	.L68
.L45:
	movl	$4, -668(%rbp)
	leaq	-208(%rbp), %rdx
	leaq	-416(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcpy@PLT
	movb	$0, -208(%rbp)
	movq	$36, -632(%rbp)
	jmp	.L68
.L35:
	movl	$11, -668(%rbp)
	movq	$36, -632(%rbp)
	jmp	.L68
.L53:
	cmpb	$0, -678(%rbp)
	je	.L95
	movq	$45, -632(%rbp)
	jmp	.L68
.L95:
	movq	$65, -632(%rbp)
	jmp	.L68
.L50:
	movl	$11, -668(%rbp)
	movq	$36, -632(%rbp)
	jmp	.L68
.L24:
	movl	-664(%rbp), %eax
	cltq
	movzbl	-624(%rbp,%rax), %eax
	cmpb	$44, %al
	jne	.L97
	movq	$57, -632(%rbp)
	jmp	.L68
.L97:
	movq	$28, -632(%rbp)
	jmp	.L68
.L38:
	movl	-664(%rbp), %eax
	cltq
	movzbl	-624(%rbp,%rax), %eax
	cmpb	$34, %al
	jne	.L99
	movq	$72, -632(%rbp)
	jmp	.L68
.L99:
	movq	$31, -632(%rbp)
	jmp	.L68
.L18:
	movl	$3, -668(%rbp)
	movq	$36, -632(%rbp)
	jmp	.L68
.L25:
	movl	$0, -664(%rbp)
	movl	$0, -668(%rbp)
	movb	$0, -208(%rbp)
	leaq	-624(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -640(%rbp)
	movq	$9, -632(%rbp)
	jmp	.L68
.L59:
	movl	$6, -668(%rbp)
	movq	$36, -632(%rbp)
	jmp	.L68
.L39:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$6, -632(%rbp)
	jmp	.L68
.L67:
	cmpl	$11, -668(%rbp)
	je	.L101
	movq	$38, -632(%rbp)
	jmp	.L68
.L101:
	movq	$20, -632(%rbp)
	jmp	.L68
.L36:
	cmpb	$0, -679(%rbp)
	je	.L103
	movq	$4, -632(%rbp)
	jmp	.L68
.L103:
	movq	$15, -632(%rbp)
	jmp	.L68
.L41:
	movl	-664(%rbp), %eax
	cltq
	movzbl	-624(%rbp,%rax), %eax
	cmpb	$34, %al
	jne	.L105
	movq	$52, -632(%rbp)
	jmp	.L68
.L105:
	movq	$35, -632(%rbp)
	jmp	.L68
.L23:
	cmpl	$0, -660(%rbp)
	jne	.L107
	movq	$64, -632(%rbp)
	jmp	.L68
.L107:
	movq	$1, -632(%rbp)
	jmp	.L68
.L62:
	movl	$10, -668(%rbp)
	movl	$0, -672(%rbp)
	movq	$36, -632(%rbp)
	jmp	.L68
.L44:
	movl	-664(%rbp), %eax
	cltq
	movzbl	-624(%rbp,%rax), %eax
	cmpb	$40, %al
	jne	.L109
	movq	$7, -632(%rbp)
	jmp	.L68
.L109:
	movq	$48, -632(%rbp)
	jmp	.L68
.L49:
	movl	$11, -668(%rbp)
	movq	$36, -632(%rbp)
	jmp	.L68
.L65:
	movl	-664(%rbp), %eax
	cltq
	movzbl	-624(%rbp,%rax), %eax
	cmpb	$34, %al
	je	.L111
	movq	$30, -632(%rbp)
	jmp	.L68
.L111:
	movq	$34, -632(%rbp)
	jmp	.L68
.L54:
	leaq	-624(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-672(%rbp), %ecx
	leaq	-416(%rbp), %rdx
	movl	-676(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-648(%rbp), %rdx
	leaq	-624(%rbp), %rax
	movl	$200, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	movq	$54, -632(%rbp)
	jmp	.L68
.L116:
	nop
.L68:
	jmp	.L113
.L115:
	call	__stack_chk_fail@PLT
.L114:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	main, .-main
	.globl	isDigit
	.type	isDigit, @function
isDigit:
.LFB6:
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
.L131:
	cmpq	$5, -8(%rbp)
	ja	.L132
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L120(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L120(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L120:
	.long	.L124-.L120
	.long	.L123-.L120
	.long	.L122-.L120
	.long	.L132-.L120
	.long	.L121-.L120
	.long	.L119-.L120
	.text
.L121:
	cmpb	$57, -20(%rbp)
	jg	.L125
	movq	$5, -8(%rbp)
	jmp	.L127
.L125:
	movq	$0, -8(%rbp)
	jmp	.L127
.L123:
	cmpb	$47, -20(%rbp)
	jle	.L128
	movq	$4, -8(%rbp)
	jmp	.L127
.L128:
	movq	$2, -8(%rbp)
	jmp	.L127
.L119:
	movl	$1, %eax
	jmp	.L130
.L124:
	movl	$0, %eax
	jmp	.L130
.L122:
	movl	$0, %eax
	jmp	.L130
.L132:
	nop
.L127:
	jmp	.L131
.L130:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	isDigit, .-isDigit
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
