	.file	"AlhuwaidiNora_K2_microprocessor-project_assm_flatten.c"
	.text
	.globl	_TIG_IZ_KwEz_envp
	.bss
	.align 8
	.type	_TIG_IZ_KwEz_envp, @object
	.size	_TIG_IZ_KwEz_envp, 8
_TIG_IZ_KwEz_envp:
	.zero	8
	.globl	_TIG_IZ_KwEz_argv
	.align 8
	.type	_TIG_IZ_KwEz_argv, @object
	.size	_TIG_IZ_KwEz_argv, 8
_TIG_IZ_KwEz_argv:
	.zero	8
	.globl	_TIG_IZ_KwEz_argc
	.align 4
	.type	_TIG_IZ_KwEz_argc, @object
	.size	_TIG_IZ_KwEz_argc, 4
_TIG_IZ_KwEz_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"RO=%d\n"
	.text
	.globl	execute_instruction
	.type	execute_instruction, @function
execute_instruction:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movl	%esi, %eax
	movb	%al, -28(%rbp)
	movq	$36, -8(%rbp)
.L67:
	cmpq	$40, -8(%rbp)
	ja	.L68
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
	.long	.L38-.L4
	.long	.L37-.L4
	.long	.L36-.L4
	.long	.L35-.L4
	.long	.L34-.L4
	.long	.L33-.L4
	.long	.L32-.L4
	.long	.L31-.L4
	.long	.L30-.L4
	.long	.L29-.L4
	.long	.L28-.L4
	.long	.L69-.L4
	.long	.L26-.L4
	.long	.L68-.L4
	.long	.L25-.L4
	.long	.L24-.L4
	.long	.L23-.L4
	.long	.L22-.L4
	.long	.L21-.L4
	.long	.L68-.L4
	.long	.L20-.L4
	.long	.L19-.L4
	.long	.L68-.L4
	.long	.L18-.L4
	.long	.L68-.L4
	.long	.L68-.L4
	.long	.L17-.L4
	.long	.L16-.L4
	.long	.L15-.L4
	.long	.L68-.L4
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
.L21:
	movq	-24(%rbp), %rax
	movzbl	-11(%rbp), %edx
	movb	%dl, 3(%rax)
	movq	$11, -8(%rbp)
	jmp	.L39
.L34:
	movq	-24(%rbp), %rax
	movb	$0, 6(%rax)
	movq	$34, -8(%rbp)
	jmp	.L39
.L14:
	movq	-24(%rbp), %rax
	movzbl	6(%rax), %eax
	testb	%al, %al
	jne	.L40
	movq	$1, -8(%rbp)
	jmp	.L39
.L40:
	movq	$11, -8(%rbp)
	jmp	.L39
.L25:
	movq	-24(%rbp), %rax
	movb	$0, 6(%rax)
	movq	$16, -8(%rbp)
	jmp	.L39
.L24:
	movq	-24(%rbp), %rax
	movzbl	(%rax), %eax
	movzbl	%al, %edx
	movq	-24(%rbp), %rax
	movzbl	1(%rax), %eax
	movzbl	%al, %ecx
	movl	%edx, %eax
	subl	%ecx, %eax
	movw	%ax, -10(%rbp)
	movq	$26, -8(%rbp)
	jmp	.L39
.L13:
	movzbl	-28(%rbp), %eax
	andl	$15, %eax
	testl	%eax, %eax
	jne	.L42
	movq	$39, -8(%rbp)
	jmp	.L39
.L42:
	movq	$35, -8(%rbp)
	jmp	.L39
.L26:
	movzbl	-28(%rbp), %eax
	andl	$15, %eax
	cmpl	$4, %eax
	jne	.L44
	movq	$15, -8(%rbp)
	jmp	.L39
.L44:
	movq	$17, -8(%rbp)
	jmp	.L39
.L30:
	movzwl	-10(%rbp), %eax
	movl	%eax, %edx
	movq	-24(%rbp), %rax
	movb	%dl, 1(%rax)
	movq	$11, -8(%rbp)
	jmp	.L39
.L37:
	movq	-24(%rbp), %rax
	movzbl	-11(%rbp), %edx
	movb	%dl, 3(%rax)
	movq	$11, -8(%rbp)
	jmp	.L39
.L18:
	movq	-24(%rbp), %rax
	movb	$1, 6(%rax)
	movq	$16, -8(%rbp)
	jmp	.L39
.L35:
	cmpw	$255, -10(%rbp)
	jbe	.L46
	movq	$9, -8(%rbp)
	jmp	.L39
.L46:
	movq	$20, -8(%rbp)
	jmp	.L39
.L23:
	movzwl	-10(%rbp), %eax
	movl	%eax, %edx
	movq	-24(%rbp), %rax
	movb	%dl, (%rax)
	movq	$11, -8(%rbp)
	jmp	.L39
.L19:
	movq	$11, -8(%rbp)
	jmp	.L39
.L8:
	movq	$5, -8(%rbp)
	jmp	.L39
.L17:
	movq	-24(%rbp), %rax
	movzbl	(%rax), %edx
	movq	-24(%rbp), %rax
	movzbl	1(%rax), %eax
	cmpb	%al, %dl
	jnb	.L48
	movq	$38, -8(%rbp)
	jmp	.L39
.L48:
	movq	$4, -8(%rbp)
	jmp	.L39
.L29:
	movq	-24(%rbp), %rax
	movb	$1, 6(%rax)
	movq	$8, -8(%rbp)
	jmp	.L39
.L12:
	movq	-24(%rbp), %rax
	movzbl	(%rax), %eax
	movzbl	%al, %edx
	movq	-24(%rbp), %rax
	movzbl	1(%rax), %eax
	movzbl	%al, %ecx
	movl	%edx, %eax
	subl	%ecx, %eax
	movw	%ax, -10(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L39
.L22:
	movq	-24(%rbp), %rax
	movzbl	-11(%rbp), %edx
	movb	%dl, 1(%rax)
	movq	$11, -8(%rbp)
	jmp	.L39
.L3:
	movq	-24(%rbp), %rax
	movzbl	(%rax), %edx
	movq	-24(%rbp), %rax
	movb	%dl, 2(%rax)
	movq	-24(%rbp), %rax
	movzbl	2(%rax), %eax
	movzbl	%al, %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$11, -8(%rbp)
	jmp	.L39
.L32:
	movzwl	-10(%rbp), %eax
	movl	%eax, %edx
	movq	-24(%rbp), %rax
	movb	%dl, (%rax)
	movq	$11, -8(%rbp)
	jmp	.L39
.L16:
	movzbl	-28(%rbp), %eax
	andl	$15, %eax
	testl	%eax, %eax
	jne	.L51
	movq	$2, -8(%rbp)
	jmp	.L39
.L51:
	movq	$12, -8(%rbp)
	jmp	.L39
.L6:
	movq	-24(%rbp), %rax
	movb	$1, 6(%rax)
	movq	$34, -8(%rbp)
	jmp	.L39
.L10:
	movzwl	-10(%rbp), %eax
	movl	%eax, %edx
	movq	-24(%rbp), %rax
	movb	%dl, 1(%rax)
	movq	$11, -8(%rbp)
	jmp	.L39
.L15:
	movq	-24(%rbp), %rax
	movb	$1, 6(%rax)
	movq	$6, -8(%rbp)
	jmp	.L39
.L33:
	movzbl	-28(%rbp), %eax
	shrb	$4, %al
	movb	%al, -12(%rbp)
	movzbl	-28(%rbp), %eax
	andl	$15, %eax
	movb	%al, -11(%rbp)
	movq	$10, -8(%rbp)
	jmp	.L39
.L11:
	movq	-24(%rbp), %rax
	movzbl	-11(%rbp), %edx
	movb	%dl, (%rax)
	movq	$11, -8(%rbp)
	jmp	.L39
.L7:
	movq	-24(%rbp), %rax
	movb	$0, 6(%rax)
	movq	$6, -8(%rbp)
	jmp	.L39
.L28:
	movzbl	-12(%rbp), %eax
	cmpl	$11, %eax
	ja	.L53
	movl	%eax, %eax
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
	.long	.L59-.L55
	.long	.L58-.L55
	.long	.L57-.L55
	.long	.L53-.L55
	.long	.L53-.L55
	.long	.L53-.L55
	.long	.L53-.L55
	.long	.L56-.L55
	.long	.L53-.L55
	.long	.L53-.L55
	.long	.L53-.L55
	.long	.L54-.L55
	.text
.L54:
	movq	$18, -8(%rbp)
	jmp	.L60
.L56:
	movq	$30, -8(%rbp)
	jmp	.L60
.L57:
	movq	$40, -8(%rbp)
	jmp	.L60
.L58:
	movq	$27, -8(%rbp)
	jmp	.L60
.L59:
	movq	$31, -8(%rbp)
	jmp	.L60
.L53:
	movq	$21, -8(%rbp)
	nop
.L60:
	jmp	.L39
.L38:
	movq	-24(%rbp), %rax
	movzbl	(%rax), %edx
	movq	-24(%rbp), %rax
	movzbl	1(%rax), %eax
	cmpb	%al, %dl
	jnb	.L61
	movq	$23, -8(%rbp)
	jmp	.L39
.L61:
	movq	$14, -8(%rbp)
	jmp	.L39
.L5:
	movq	-24(%rbp), %rax
	movzbl	(%rax), %eax
	movzbl	%al, %edx
	movq	-24(%rbp), %rax
	movzbl	1(%rax), %eax
	movzbl	%al, %eax
	addl	%edx, %eax
	movw	%ax, -10(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L39
.L31:
	cmpw	$255, -10(%rbp)
	jbe	.L63
	movq	$28, -8(%rbp)
	jmp	.L39
.L63:
	movq	$37, -8(%rbp)
	jmp	.L39
.L9:
	movzbl	-28(%rbp), %eax
	andl	$15, %eax
	cmpl	$4, %eax
	jne	.L65
	movq	$32, -8(%rbp)
	jmp	.L39
.L65:
	movq	$33, -8(%rbp)
	jmp	.L39
.L36:
	movq	-24(%rbp), %rax
	movzbl	(%rax), %eax
	movzbl	%al, %edx
	movq	-24(%rbp), %rax
	movzbl	1(%rax), %eax
	movzbl	%al, %eax
	addl	%edx, %eax
	movw	%ax, -10(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L39
.L20:
	movq	-24(%rbp), %rax
	movb	$0, 6(%rax)
	movq	$8, -8(%rbp)
	jmp	.L39
.L68:
	nop
.L39:
	jmp	.L67
.L69:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	execute_instruction, .-execute_instruction
	.section	.rodata
.LC1:
	.string	"Invalid mode selected"
.LC2:
	.string	"Usage: %s <binary_file>\n"
	.align 8
.LC3:
	.string	"Select one of the following mode"
.LC4:
	.string	"R - Run in continuous mode"
.LC5:
	.string	"S - Run step-by-step"
.LC6:
	.string	"Select mode:"
.LC7:
	.string	" %c"
	.text
	.globl	main
	.type	main, @function
main:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$336, %rsp
	movl	%edi, -308(%rbp)
	movq	%rsi, -320(%rbp)
	movq	%rdx, -328(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_KwEz_envp(%rip)
	nop
.L71:
	movq	$0, _TIG_IZ_KwEz_argv(%rip)
	nop
.L72:
	movl	$0, _TIG_IZ_KwEz_argc(%rip)
	nop
	nop
.L73:
.L74:
#APP
# 88 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-KwEz--0
# 0 "" 2
#NO_APP
	movl	-308(%rbp), %eax
	movl	%eax, _TIG_IZ_KwEz_argc(%rip)
	movq	-320(%rbp), %rax
	movq	%rax, _TIG_IZ_KwEz_argv(%rip)
	movq	-328(%rbp), %rax
	movq	%rax, _TIG_IZ_KwEz_envp(%rip)
	nop
	movq	$8, -280(%rbp)
.L99:
	cmpq	$13, -280(%rbp)
	ja	.L102
	movq	-280(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L77(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L77(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L77:
	.long	.L102-.L77
	.long	.L88-.L77
	.long	.L87-.L77
	.long	.L86-.L77
	.long	.L85-.L77
	.long	.L84-.L77
	.long	.L83-.L77
	.long	.L82-.L77
	.long	.L81-.L77
	.long	.L80-.L77
	.long	.L79-.L77
	.long	.L78-.L77
	.long	.L102-.L77
	.long	.L76-.L77
	.text
.L85:
	movl	$0, %eax
	jmp	.L100
.L81:
	cmpl	$2, -308(%rbp)
	je	.L90
	movq	$5, -280(%rbp)
	jmp	.L92
.L90:
	movq	$13, -280(%rbp)
	jmp	.L92
.L88:
	movzbl	-289(%rbp), %eax
	cmpb	$83, %al
	jne	.L93
	movq	$11, -280(%rbp)
	jmp	.L92
.L93:
	movq	$9, -280(%rbp)
	jmp	.L92
.L86:
	movl	$1, %eax
	jmp	.L100
.L78:
	movzbl	-289(%rbp), %eax
	movsbl	%al, %edx
	leaq	-272(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	run_simulation
	movq	$4, -280(%rbp)
	jmp	.L92
.L80:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$4, -280(%rbp)
	jmp	.L92
.L76:
	leaq	-272(%rbp), %rax
	movq	%rax, %rdi
	call	init_processor
	movq	-320(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rdx
	leaq	-272(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	load_program
	movl	%eax, -284(%rbp)
	movl	-284(%rbp), %eax
	movl	%eax, -288(%rbp)
	movq	$7, -280(%rbp)
	jmp	.L92
.L83:
	movzbl	-289(%rbp), %eax
	movsbl	%al, %edx
	leaq	-272(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	run_simulation
	movq	$4, -280(%rbp)
	jmp	.L92
.L84:
	movq	-320(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -280(%rbp)
	jmp	.L92
.L79:
	movzbl	-289(%rbp), %eax
	cmpb	$82, %al
	jne	.L95
	movq	$6, -280(%rbp)
	jmp	.L92
.L95:
	movq	$1, -280(%rbp)
	jmp	.L92
.L82:
	cmpl	$0, -288(%rbp)
	jle	.L97
	movq	$2, -280(%rbp)
	jmp	.L92
.L97:
	movq	$4, -280(%rbp)
	jmp	.L92
.L87:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-289(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	call	getchar@PLT
	movq	$10, -280(%rbp)
	jmp	.L92
.L102:
	nop
.L92:
	jmp	.L99
.L100:
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L101
	call	__stack_chk_fail@PLT
.L101:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	main, .-main
	.section	.rodata
.LC8:
	.string	"Error: Cannot open file %s\n"
.LC9:
	.string	"\n\r"
.LC10:
	.string	"r"
.LC11:
	.string	"Loading binary file: %s\n"
	.text
	.globl	load_program
	.type	load_program, @function
load_program:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$368, %rsp
	movq	%rdi, -360(%rbp)
	movq	%rsi, -368(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$4, -304(%rbp)
.L142:
	cmpq	$29, -304(%rbp)
	ja	.L145
	movq	-304(%rbp), %rax
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
	.long	.L145-.L106
	.long	.L145-.L106
	.long	.L125-.L106
	.long	.L124-.L106
	.long	.L123-.L106
	.long	.L122-.L106
	.long	.L121-.L106
	.long	.L120-.L106
	.long	.L119-.L106
	.long	.L145-.L106
	.long	.L118-.L106
	.long	.L117-.L106
	.long	.L116-.L106
	.long	.L145-.L106
	.long	.L115-.L106
	.long	.L114-.L106
	.long	.L145-.L106
	.long	.L113-.L106
	.long	.L112-.L106
	.long	.L111-.L106
	.long	.L110-.L106
	.long	.L109-.L106
	.long	.L145-.L106
	.long	.L145-.L106
	.long	.L108-.L106
	.long	.L107-.L106
	.long	.L145-.L106
	.long	.L145-.L106
	.long	.L145-.L106
	.long	.L105-.L106
	.text
.L112:
	movl	$0, %eax
	jmp	.L143
.L107:
	movq	-344(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$13, %al
	jne	.L127
	movq	$6, -304(%rbp)
	jmp	.L129
.L127:
	movq	$5, -304(%rbp)
	jmp	.L129
.L123:
	movq	$17, -304(%rbp)
	jmp	.L129
.L115:
	movl	-352(%rbp), %eax
	movl	%eax, -348(%rbp)
	addl	$1, -352(%rbp)
	movq	-320(%rbp), %rax
	movl	%eax, %ecx
	movq	-360(%rbp), %rdx
	movl	-348(%rbp), %eax
	cltq
	movb	%cl, 7(%rdx,%rax)
	movq	$5, -304(%rbp)
	jmp	.L129
.L114:
	cmpq	$7, -328(%rbp)
	ja	.L130
	movq	$5, -304(%rbp)
	jmp	.L129
.L130:
	movq	$20, -304(%rbp)
	jmp	.L129
.L116:
	cmpq	$0, -336(%rbp)
	jne	.L132
	movq	$8, -304(%rbp)
	jmp	.L129
.L132:
	movq	$2, -304(%rbp)
	jmp	.L129
.L119:
	movq	-368(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$18, -304(%rbp)
	jmp	.L129
.L124:
	leaq	-272(%rbp), %rax
	leaq	.LC9(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcspn@PLT
	movq	%rax, -280(%rbp)
	leaq	-272(%rbp), %rdx
	movq	-280(%rbp), %rax
	addq	%rdx, %rax
	movb	$0, (%rax)
	leaq	-272(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -328(%rbp)
	movq	$15, -304(%rbp)
	jmp	.L129
.L108:
	movl	-352(%rbp), %eax
	jmp	.L143
.L109:
	movl	-352(%rbp), %eax
	movl	%eax, -348(%rbp)
	addl	$1, -352(%rbp)
	movq	-320(%rbp), %rax
	movl	%eax, %ecx
	movq	-360(%rbp), %rdx
	movl	-348(%rbp), %eax
	cltq
	movb	%cl, 7(%rdx,%rax)
	movq	$5, -304(%rbp)
	jmp	.L129
.L117:
	movq	-336(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$24, -304(%rbp)
	jmp	.L129
.L111:
	cmpl	$255, -352(%rbp)
	jg	.L134
	movq	$3, -304(%rbp)
	jmp	.L129
.L134:
	movq	$11, -304(%rbp)
	jmp	.L129
.L113:
	movq	-368(%rbp), %rax
	leaq	.LC10(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -288(%rbp)
	movq	-288(%rbp), %rax
	movq	%rax, -336(%rbp)
	movq	$12, -304(%rbp)
	jmp	.L129
.L121:
	movl	-352(%rbp), %eax
	movl	%eax, -348(%rbp)
	addl	$1, -352(%rbp)
	movq	-320(%rbp), %rax
	movl	%eax, %ecx
	movq	-360(%rbp), %rdx
	movl	-348(%rbp), %eax
	cltq
	movb	%cl, 7(%rdx,%rax)
	movq	$5, -304(%rbp)
	jmp	.L129
.L122:
	movq	-336(%rbp), %rdx
	leaq	-272(%rbp), %rax
	movl	$256, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	movq	%rax, -312(%rbp)
	movq	$10, -304(%rbp)
	jmp	.L129
.L118:
	cmpq	$0, -312(%rbp)
	je	.L136
	movq	$19, -304(%rbp)
	jmp	.L129
.L136:
	movq	$11, -304(%rbp)
	jmp	.L129
.L120:
	movq	-344(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$10, %al
	jne	.L138
	movq	$21, -304(%rbp)
	jmp	.L129
.L138:
	movq	$25, -304(%rbp)
	jmp	.L129
.L105:
	movq	-344(%rbp), %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	jne	.L140
	movq	$14, -304(%rbp)
	jmp	.L129
.L140:
	movq	$7, -304(%rbp)
	jmp	.L129
.L125:
	movq	-368(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$0, -352(%rbp)
	movq	$5, -304(%rbp)
	jmp	.L129
.L110:
	leaq	-344(%rbp), %rcx
	leaq	-272(%rbp), %rax
	movl	$2, %edx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	strtol@PLT
	movq	%rax, -296(%rbp)
	movq	-296(%rbp), %rax
	movq	%rax, -320(%rbp)
	movq	$29, -304(%rbp)
	jmp	.L129
.L145:
	nop
.L129:
	jmp	.L142
.L143:
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L144
	call	__stack_chk_fail@PLT
.L144:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	load_program, .-load_program
	.section	.rodata
.LC12:
	.string	"continuous"
.LC13:
	.string	"step-by-step"
	.align 8
.LC14:
	.string	"Starting Simulator in %s mode...\n"
	.align 8
.LC15:
	.string	"Execution (Register RO output):"
	.text
	.globl	run_simulation
	.type	run_simulation, @function
run_simulation:
.LFB6:
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
	movq	$12, -8(%rbp)
.L176:
	cmpq	$17, -8(%rbp)
	ja	.L178
	movq	-8(%rbp), %rax
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
	.long	.L163-.L149
	.long	.L162-.L149
	.long	.L161-.L149
	.long	.L160-.L149
	.long	.L159-.L149
	.long	.L178-.L149
	.long	.L178-.L149
	.long	.L158-.L149
	.long	.L177-.L149
	.long	.L156-.L149
	.long	.L155-.L149
	.long	.L179-.L149
	.long	.L153-.L149
	.long	.L152-.L149
	.long	.L178-.L149
	.long	.L151-.L149
	.long	.L150-.L149
	.long	.L148-.L149
	.text
.L159:
	leaq	.LC12(%rip), %rax
	movq	%rax, -16(%rbp)
	movq	$9, -8(%rbp)
	jmp	.L164
.L151:
	cmpb	$83, -44(%rbp)
	jne	.L165
	movq	$0, -8(%rbp)
	jmp	.L164
.L165:
	movq	$2, -8(%rbp)
	jmp	.L164
.L153:
	cmpb	$83, -44(%rbp)
	jne	.L167
	movq	$3, -8(%rbp)
	jmp	.L164
.L167:
	movq	$4, -8(%rbp)
	jmp	.L164
.L177:
	movq	$1, -8(%rbp)
	jmp	.L164
.L162:
	movq	-40(%rbp), %rax
	movzbl	3(%rax), %eax
	movb	%al, -17(%rbp)
	movq	-40(%rbp), %rax
	movzbl	3(%rax), %eax
	leal	1(%rax), %edx
	movq	-40(%rbp), %rax
	movb	%dl, 3(%rax)
	movzbl	-17(%rbp), %eax
	movq	-40(%rbp), %rdx
	cltq
	movzbl	7(%rdx,%rax), %eax
	movb	%al, -18(%rbp)
	movq	$16, -8(%rbp)
	jmp	.L164
.L160:
	leaq	.LC13(%rip), %rax
	movq	%rax, -16(%rbp)
	movq	$9, -8(%rbp)
	jmp	.L164
.L150:
	cmpb	$0, -18(%rbp)
	jne	.L169
	movq	$8, -8(%rbp)
	jmp	.L164
.L169:
	movq	$15, -8(%rbp)
	jmp	.L164
.L156:
	movq	-16(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$17, -8(%rbp)
	jmp	.L164
.L152:
	movl	$100000, %edi
	call	usleep@PLT
	movq	$8, -8(%rbp)
	jmp	.L164
.L148:
	cmpb	$82, -44(%rbp)
	jne	.L172
	movq	$10, -8(%rbp)
	jmp	.L164
.L172:
	movq	$8, -8(%rbp)
	jmp	.L164
.L155:
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$8, -8(%rbp)
	jmp	.L164
.L163:
	movzbl	-18(%rbp), %edx
	movq	-40(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	print_instruction
	call	getchar@PLT
	movq	$2, -8(%rbp)
	jmp	.L164
.L158:
	cmpb	$82, -44(%rbp)
	jne	.L174
	movq	$13, -8(%rbp)
	jmp	.L164
.L174:
	movq	$8, -8(%rbp)
	jmp	.L164
.L161:
	movzbl	-18(%rbp), %edx
	movq	-40(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	execute_instruction
	movq	$7, -8(%rbp)
	jmp	.L164
.L178:
	nop
.L164:
	jmp	.L176
.L179:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	run_simulation, .-run_simulation
	.section	.rodata
.LC16:
	.string	"RA=RA+RB"
.LC17:
	.string	" [Press Enter to continue]"
.LC18:
	.string	"J=%d (Jump to Instruction %d)"
.LC19:
	.string	"No Jump"
.LC20:
	.string	"RO=RA -> RO=%d"
.LC21:
	.string	"RA=%d"
.LC22:
	.string	"Instruction %d: "
.LC23:
	.string	"RB=%d"
.LC24:
	.string	"JC=%d (%s)"
.LC25:
	.string	"RA=RA-RB"
.LC26:
	.string	"Jump"
.LC27:
	.string	"RB=RA+RB"
.LC28:
	.string	"RB=RA-RB"
	.text
	.globl	print_instruction
	.type	print_instruction, @function
print_instruction:
.LFB7:
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
	movq	$1, -8(%rbp)
.L225:
	cmpq	$26, -8(%rbp)
	ja	.L226
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L183(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L183(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L183:
	.long	.L204-.L183
	.long	.L203-.L183
	.long	.L226-.L183
	.long	.L202-.L183
	.long	.L201-.L183
	.long	.L200-.L183
	.long	.L199-.L183
	.long	.L198-.L183
	.long	.L197-.L183
	.long	.L226-.L183
	.long	.L196-.L183
	.long	.L195-.L183
	.long	.L194-.L183
	.long	.L226-.L183
	.long	.L193-.L183
	.long	.L192-.L183
	.long	.L191-.L183
	.long	.L190-.L183
	.long	.L189-.L183
	.long	.L227-.L183
	.long	.L187-.L183
	.long	.L186-.L183
	.long	.L185-.L183
	.long	.L226-.L183
	.long	.L184-.L183
	.long	.L226-.L183
	.long	.L182-.L183
	.text
.L189:
	leaq	.LC16(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$14, -8(%rbp)
	jmp	.L205
.L201:
	movq	$14, -8(%rbp)
	jmp	.L205
.L193:
	leaq	.LC17(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$19, -8(%rbp)
	jmp	.L205
.L192:
	movzbl	-44(%rbp), %eax
	andl	$15, %eax
	cmpl	$4, %eax
	jne	.L206
	movq	$20, -8(%rbp)
	jmp	.L205
.L206:
	movq	$17, -8(%rbp)
	jmp	.L205
.L194:
	movzbl	-17(%rbp), %edx
	movzbl	-17(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC18(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$14, -8(%rbp)
	jmp	.L205
.L197:
	leaq	.LC19(%rip), %rax
	movq	%rax, -16(%rbp)
	movq	$22, -8(%rbp)
	jmp	.L205
.L203:
	movq	$26, -8(%rbp)
	jmp	.L205
.L202:
	movq	-40(%rbp), %rax
	movzbl	(%rax), %eax
	movzbl	%al, %eax
	movl	%eax, %esi
	leaq	.LC20(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$14, -8(%rbp)
	jmp	.L205
.L191:
	movzbl	-18(%rbp), %eax
	cmpl	$11, %eax
	ja	.L208
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L210(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L210(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L210:
	.long	.L214-.L210
	.long	.L213-.L210
	.long	.L212-.L210
	.long	.L208-.L210
	.long	.L208-.L210
	.long	.L208-.L210
	.long	.L208-.L210
	.long	.L211-.L210
	.long	.L208-.L210
	.long	.L208-.L210
	.long	.L208-.L210
	.long	.L209-.L210
	.text
.L209:
	movq	$12, -8(%rbp)
	jmp	.L215
.L211:
	movq	$21, -8(%rbp)
	jmp	.L215
.L212:
	movq	$3, -8(%rbp)
	jmp	.L215
.L213:
	movq	$0, -8(%rbp)
	jmp	.L215
.L214:
	movq	$6, -8(%rbp)
	jmp	.L215
.L208:
	movq	$4, -8(%rbp)
	nop
.L215:
	jmp	.L205
.L184:
	movzbl	-17(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC21(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$14, -8(%rbp)
	jmp	.L205
.L186:
	movq	-40(%rbp), %rax
	movzbl	6(%rax), %eax
	testb	%al, %al
	jne	.L216
	movq	$10, -8(%rbp)
	jmp	.L205
.L216:
	movq	$8, -8(%rbp)
	jmp	.L205
.L182:
	movzbl	-44(%rbp), %eax
	shrb	$4, %al
	movb	%al, -18(%rbp)
	movzbl	-44(%rbp), %eax
	andl	$15, %eax
	movb	%al, -17(%rbp)
	movq	-40(%rbp), %rax
	movzbl	3(%rax), %eax
	movzbl	%al, %eax
	subl	$1, %eax
	movl	%eax, %esi
	leaq	.LC22(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$16, -8(%rbp)
	jmp	.L205
.L195:
	movzbl	-44(%rbp), %eax
	andl	$15, %eax
	cmpl	$4, %eax
	jne	.L218
	movq	$5, -8(%rbp)
	jmp	.L205
.L218:
	movq	$24, -8(%rbp)
	jmp	.L205
.L190:
	movzbl	-17(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC23(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$14, -8(%rbp)
	jmp	.L205
.L199:
	movzbl	-44(%rbp), %eax
	andl	$15, %eax
	testl	%eax, %eax
	jne	.L221
	movq	$18, -8(%rbp)
	jmp	.L205
.L221:
	movq	$11, -8(%rbp)
	jmp	.L205
.L185:
	movzbl	-17(%rbp), %eax
	movq	-16(%rbp), %rdx
	movl	%eax, %esi
	leaq	.LC24(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$14, -8(%rbp)
	jmp	.L205
.L200:
	leaq	.LC25(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$14, -8(%rbp)
	jmp	.L205
.L196:
	leaq	.LC26(%rip), %rax
	movq	%rax, -16(%rbp)
	movq	$22, -8(%rbp)
	jmp	.L205
.L204:
	movzbl	-44(%rbp), %eax
	andl	$15, %eax
	testl	%eax, %eax
	jne	.L223
	movq	$7, -8(%rbp)
	jmp	.L205
.L223:
	movq	$15, -8(%rbp)
	jmp	.L205
.L198:
	leaq	.LC27(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$14, -8(%rbp)
	jmp	.L205
.L187:
	leaq	.LC28(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$14, -8(%rbp)
	jmp	.L205
.L226:
	nop
.L205:
	jmp	.L225
.L227:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	print_instruction, .-print_instruction
	.globl	init_processor
	.type	init_processor, @function
init_processor:
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
	movq	$1, -8(%rbp)
.L233:
	cmpq	$0, -8(%rbp)
	je	.L234
	cmpq	$1, -8(%rbp)
	jne	.L235
	movq	-24(%rbp), %rax
	movl	$263, %edx
	movl	$0, %esi
	movq	%rax, %rdi
	call	memset@PLT
	movq	$0, -8(%rbp)
	jmp	.L231
.L235:
	nop
.L231:
	jmp	.L233
.L234:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	init_processor, .-init_processor
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
