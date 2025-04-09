	.file	"rubengill_Comp2510Assignments_a1_flatten.c"
	.text
	.globl	_TIG_IZ_Deet_envp
	.bss
	.align 8
	.type	_TIG_IZ_Deet_envp, @object
	.size	_TIG_IZ_Deet_envp, 8
_TIG_IZ_Deet_envp:
	.zero	8
	.globl	_TIG_IZ_Deet_argv
	.align 8
	.type	_TIG_IZ_Deet_argv, @object
	.size	_TIG_IZ_Deet_argv, 8
_TIG_IZ_Deet_argv:
	.zero	8
	.globl	_TIG_IZ_Deet_argc
	.align 4
	.type	_TIG_IZ_Deet_argc, @object
	.size	_TIG_IZ_Deet_argc, 4
_TIG_IZ_Deet_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"r"
.LC1:
	.string	""
.LC2:
	.string	"w"
	.align 8
.LC3:
	.string	"Usage: %s <line_length> <input_file.txt>\n"
	.align 8
.LC4:
	.string	"Failed to create the output file."
.LC5:
	.string	"Malloc failed ! "
.LC6:
	.string	"Failed to open input file."
	.text
	.globl	main
	.type	main, @function
main:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$160, %rsp
	movl	%edi, -132(%rbp)
	movq	%rsi, -144(%rbp)
	movq	%rdx, -152(%rbp)
	movq	$0, _TIG_IZ_Deet_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_Deet_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_Deet_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 140 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-Deet--0
# 0 "" 2
#NO_APP
	movl	-132(%rbp), %eax
	movl	%eax, _TIG_IZ_Deet_argc(%rip)
	movq	-144(%rbp), %rax
	movq	%rax, _TIG_IZ_Deet_argv(%rip)
	movq	-152(%rbp), %rax
	movq	%rax, _TIG_IZ_Deet_envp(%rip)
	nop
	movq	$21, -64(%rbp)
.L40:
	cmpq	$26, -64(%rbp)
	ja	.L41
	movq	-64(%rbp), %rax
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
	.long	.L41-.L8
	.long	.L27-.L8
	.long	.L41-.L8
	.long	.L41-.L8
	.long	.L26-.L8
	.long	.L25-.L8
	.long	.L24-.L8
	.long	.L23-.L8
	.long	.L22-.L8
	.long	.L41-.L8
	.long	.L41-.L8
	.long	.L21-.L8
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L41-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L41-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L15:
	movl	$1, %eax
	jmp	.L28
.L9:
	movq	-144(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movl	%eax, -104(%rbp)
	movl	-104(%rbp), %eax
	movl	%eax, -120(%rbp)
	movq	-144(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, -56(%rbp)
	movq	-56(%rbp), %rax
	leaq	.LC0(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -48(%rbp)
	movq	-48(%rbp), %rax
	movq	%rax, -88(%rbp)
	movq	$14, -64(%rbp)
	jmp	.L29
.L26:
	movl	-108(%rbp), %eax
	cmpl	-112(%rbp), %eax
	jge	.L30
	movq	$5, -64(%rbp)
	jmp	.L29
.L30:
	movq	$12, -64(%rbp)
	jmp	.L29
.L18:
	cmpq	$0, -88(%rbp)
	jne	.L32
	movq	$7, -64(%rbp)
	jmp	.L29
.L32:
	movq	$8, -64(%rbp)
	jmp	.L29
.L20:
	movq	-80(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	-96(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	leaq	.LC1(%rip), %rax
	movq	%rax, -32(%rbp)
	movq	-32(%rbp), %rax
	leaq	.LC2(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, -72(%rbp)
	movq	$17, -64(%rbp)
	jmp	.L29
.L22:
	movq	-88(%rbp), %rax
	movl	$2, %edx
	movl	$0, %esi
	movq	%rax, %rdi
	call	fseek@PLT
	movq	-88(%rbp), %rax
	movq	%rax, %rdi
	call	ftell@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movl	%eax, -116(%rbp)
	movq	-88(%rbp), %rax
	movl	$0, %edx
	movl	$0, %esi
	movq	%rax, %rdi
	call	fseek@PLT
	movl	-116(%rbp), %eax
	cltq
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -96(%rbp)
	movq	$16, -64(%rbp)
	jmp	.L29
.L27:
	movq	-144(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$26, -64(%rbp)
	jmp	.L29
.L11:
	movl	$1, %eax
	jmp	.L28
.L17:
	cmpq	$0, -96(%rbp)
	jne	.L34
	movq	$22, -64(%rbp)
	jmp	.L29
.L34:
	movq	$13, -64(%rbp)
	jmp	.L29
.L10:
	movq	-72(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$11, -64(%rbp)
	jmp	.L29
.L13:
	cmpl	$3, -132(%rbp)
	je	.L36
	movq	$1, -64(%rbp)
	jmp	.L29
.L36:
	movq	$25, -64(%rbp)
	jmp	.L29
.L7:
	movl	$1, %eax
	jmp	.L28
.L21:
	movl	$0, %eax
	jmp	.L28
.L19:
	movl	-116(%rbp), %eax
	movslq	%eax, %rdx
	movq	-88(%rbp), %rcx
	movq	-96(%rbp), %rax
	movl	$1, %esi
	movq	%rax, %rdi
	call	fread@PLT
	movq	-88(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movl	-116(%rbp), %edx
	movl	-120(%rbp), %ecx
	movq	-96(%rbp), %rax
	movl	%ecx, %esi
	movq	%rax, %rdi
	call	count_lines
	movl	%eax, -100(%rbp)
	movl	-100(%rbp), %eax
	movl	%eax, -112(%rbp)
	movl	-116(%rbp), %ecx
	movl	-112(%rbp), %edx
	movl	-120(%rbp), %esi
	movq	-96(%rbp), %rax
	movq	%rax, %rdi
	call	divide_input
	movq	%rax, -40(%rbp)
	movq	-40(%rbp), %rax
	movq	%rax, -80(%rbp)
	movl	-120(%rbp), %edx
	movl	-112(%rbp), %ecx
	movq	-80(%rbp), %rax
	movl	%ecx, %esi
	movq	%rax, %rdi
	call	justify_rows
	movl	$0, -108(%rbp)
	movq	$4, -64(%rbp)
	jmp	.L29
.L16:
	cmpq	$0, -72(%rbp)
	jne	.L38
	movq	$6, -64(%rbp)
	jmp	.L29
.L38:
	movq	$24, -64(%rbp)
	jmp	.L29
.L24:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$20, -64(%rbp)
	jmp	.L29
.L12:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$18, -64(%rbp)
	jmp	.L29
.L25:
	movl	-108(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-80(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	free@PLT
	addl	$1, -108(%rbp)
	movq	$4, -64(%rbp)
	jmp	.L29
.L23:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$23, -64(%rbp)
	jmp	.L29
.L14:
	movl	$1, %eax
	jmp	.L28
.L41:
	nop
.L29:
	jmp	.L40
.L28:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	main, .-main
	.section	.rodata
	.align 8
.LC7:
	.string	"Error. The word processor can't display the output."
	.text
	.globl	count_lines
	.type	count_lines, @function
count_lines:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movq	%rdi, -56(%rbp)
	movl	%esi, -60(%rbp)
	movl	%edx, -64(%rbp)
	movq	$9, -8(%rbp)
.L101:
	cmpq	$41, -8(%rbp)
	ja	.L103
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L45(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L45(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L45:
	.long	.L70-.L45
	.long	.L69-.L45
	.long	.L68-.L45
	.long	.L67-.L45
	.long	.L66-.L45
	.long	.L65-.L45
	.long	.L64-.L45
	.long	.L63-.L45
	.long	.L103-.L45
	.long	.L62-.L45
	.long	.L103-.L45
	.long	.L61-.L45
	.long	.L60-.L45
	.long	.L103-.L45
	.long	.L103-.L45
	.long	.L103-.L45
	.long	.L59-.L45
	.long	.L103-.L45
	.long	.L103-.L45
	.long	.L58-.L45
	.long	.L57-.L45
	.long	.L56-.L45
	.long	.L55-.L45
	.long	.L103-.L45
	.long	.L54-.L45
	.long	.L103-.L45
	.long	.L103-.L45
	.long	.L53-.L45
	.long	.L103-.L45
	.long	.L52-.L45
	.long	.L51-.L45
	.long	.L50-.L45
	.long	.L49-.L45
	.long	.L48-.L45
	.long	.L47-.L45
	.long	.L46-.L45
	.long	.L103-.L45
	.long	.L103-.L45
	.long	.L103-.L45
	.long	.L103-.L45
	.long	.L103-.L45
	.long	.L44-.L45
	.text
.L66:
	subq	$1, -16(%rbp)
	movq	$20, -8(%rbp)
	jmp	.L71
.L51:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$1, %edi
	call	exit@PLT
.L50:
	movl	$0, -40(%rbp)
	movq	-56(%rbp), %rax
	movq	%rax, -32(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L71
.L60:
	movl	-36(%rbp), %eax
	cmpl	-60(%rbp), %eax
	jne	.L72
	movq	$27, -8(%rbp)
	jmp	.L71
.L72:
	movq	$29, -8(%rbp)
	jmp	.L71
.L69:
	movq	-16(%rbp), %rax
	movq	%rax, -32(%rbp)
	movq	$29, -8(%rbp)
	jmp	.L71
.L67:
	movq	-32(%rbp), %rax
	movq	%rax, -16(%rbp)
	movq	$20, -8(%rbp)
	jmp	.L71
.L59:
	movq	-16(%rbp), %rdx
	movq	-24(%rbp), %rax
	cmpq	%rax, %rdx
	jbe	.L74
	movq	$22, -8(%rbp)
	jmp	.L71
.L74:
	movq	$41, -8(%rbp)
	jmp	.L71
.L54:
	movl	-64(%rbp), %eax
	movslq	%eax, %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, %rdx
	movq	-32(%rbp), %rax
	cmpq	%rax, %rdx
	jbe	.L76
	movq	$21, -8(%rbp)
	jmp	.L71
.L76:
	movq	$6, -8(%rbp)
	jmp	.L71
.L56:
	movq	-32(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$32, %al
	jne	.L78
	movq	$35, -8(%rbp)
	jmp	.L71
.L78:
	movq	$6, -8(%rbp)
	jmp	.L71
.L61:
	movq	-16(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$45, %al
	je	.L80
	movq	$4, -8(%rbp)
	jmp	.L71
.L80:
	movq	$34, -8(%rbp)
	jmp	.L71
.L62:
	movq	$31, -8(%rbp)
	jmp	.L71
.L58:
	movl	-40(%rbp), %eax
	jmp	.L102
.L49:
	movl	-36(%rbp), %eax
	cmpl	-60(%rbp), %eax
	jge	.L83
	movq	$0, -8(%rbp)
	jmp	.L71
.L83:
	movq	$12, -8(%rbp)
	jmp	.L71
.L64:
	movl	-64(%rbp), %eax
	movslq	%eax, %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, %rdx
	movq	-32(%rbp), %rax
	cmpq	%rax, %rdx
	jbe	.L85
	movq	$5, -8(%rbp)
	jmp	.L71
.L85:
	movq	$19, -8(%rbp)
	jmp	.L71
.L53:
	movq	-32(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$32, %al
	je	.L87
	movq	$2, -8(%rbp)
	jmp	.L71
.L87:
	movq	$29, -8(%rbp)
	jmp	.L71
.L47:
	movq	-16(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$45, %al
	jne	.L89
	movq	$16, -8(%rbp)
	jmp	.L71
.L89:
	movq	$41, -8(%rbp)
	jmp	.L71
.L55:
	movq	-16(%rbp), %rax
	addq	$1, %rax
	movq	%rax, -32(%rbp)
	movq	$29, -8(%rbp)
	jmp	.L71
.L65:
	movl	$0, -36(%rbp)
	movq	-32(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	$32, -8(%rbp)
	jmp	.L71
.L48:
	addl	$1, -36(%rbp)
	addq	$1, -32(%rbp)
	movq	$32, -8(%rbp)
	jmp	.L71
.L44:
	movq	-16(%rbp), %rdx
	movq	-24(%rbp), %rax
	cmpq	%rax, %rdx
	jne	.L91
	movq	$30, -8(%rbp)
	jmp	.L71
.L91:
	movq	$1, -8(%rbp)
	jmp	.L71
.L70:
	movl	-64(%rbp), %eax
	movslq	%eax, %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, %rdx
	movq	-32(%rbp), %rax
	cmpq	%rax, %rdx
	jbe	.L93
	movq	$33, -8(%rbp)
	jmp	.L71
.L93:
	movq	$12, -8(%rbp)
	jmp	.L71
.L63:
	movq	-16(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$32, %al
	je	.L95
	movq	$11, -8(%rbp)
	jmp	.L71
.L95:
	movq	$34, -8(%rbp)
	jmp	.L71
.L46:
	addq	$1, -32(%rbp)
	movq	$24, -8(%rbp)
	jmp	.L71
.L52:
	addl	$1, -40(%rbp)
	movq	$24, -8(%rbp)
	jmp	.L71
.L68:
	movq	-32(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$45, %al
	je	.L97
	movq	$3, -8(%rbp)
	jmp	.L71
.L97:
	movq	$29, -8(%rbp)
	jmp	.L71
.L57:
	movq	-16(%rbp), %rdx
	movq	-24(%rbp), %rax
	cmpq	%rax, %rdx
	jbe	.L99
	movq	$7, -8(%rbp)
	jmp	.L71
.L99:
	movq	$34, -8(%rbp)
	jmp	.L71
.L103:
	nop
.L71:
	jmp	.L101
.L102:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	count_lines, .-count_lines
	.globl	divide_input
	.type	divide_input, @function
divide_input:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$144, %rsp
	movq	%rdi, -120(%rbp)
	movl	%esi, -124(%rbp)
	movl	%edx, -128(%rbp)
	movl	%ecx, -132(%rbp)
	movq	$18, -48(%rbp)
.L167:
	cmpq	$50, -48(%rbp)
	ja	.L169
	movq	-48(%rbp), %rax
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
	.long	.L169-.L107
	.long	.L169-.L107
	.long	.L136-.L107
	.long	.L135-.L107
	.long	.L134-.L107
	.long	.L133-.L107
	.long	.L132-.L107
	.long	.L169-.L107
	.long	.L169-.L107
	.long	.L131-.L107
	.long	.L169-.L107
	.long	.L169-.L107
	.long	.L130-.L107
	.long	.L129-.L107
	.long	.L128-.L107
	.long	.L169-.L107
	.long	.L127-.L107
	.long	.L169-.L107
	.long	.L126-.L107
	.long	.L125-.L107
	.long	.L124-.L107
	.long	.L169-.L107
	.long	.L169-.L107
	.long	.L123-.L107
	.long	.L169-.L107
	.long	.L169-.L107
	.long	.L122-.L107
	.long	.L169-.L107
	.long	.L121-.L107
	.long	.L120-.L107
	.long	.L169-.L107
	.long	.L119-.L107
	.long	.L118-.L107
	.long	.L169-.L107
	.long	.L117-.L107
	.long	.L116-.L107
	.long	.L115-.L107
	.long	.L114-.L107
	.long	.L169-.L107
	.long	.L113-.L107
	.long	.L169-.L107
	.long	.L169-.L107
	.long	.L112-.L107
	.long	.L111-.L107
	.long	.L169-.L107
	.long	.L169-.L107
	.long	.L169-.L107
	.long	.L110-.L107
	.long	.L109-.L107
	.long	.L108-.L107
	.long	.L106-.L107
	.text
.L126:
	movq	$37, -48(%rbp)
	jmp	.L137
.L106:
	addq	$8, -88(%rbp)
	addl	$1, -112(%rbp)
	movq	$31, -48(%rbp)
	jmp	.L137
.L108:
	movq	-96(%rbp), %rax
	subq	$1, %rax
	movq	%rax, -64(%rbp)
	movq	$29, -48(%rbp)
	jmp	.L137
.L134:
	movl	-132(%rbp), %eax
	movslq	%eax, %rdx
	movq	-120(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, %rdx
	movq	-96(%rbp), %rax
	cmpq	%rax, %rdx
	jbe	.L138
	movq	$20, -48(%rbp)
	jmp	.L137
.L138:
	movq	$50, -48(%rbp)
	jmp	.L137
.L128:
	movq	-96(%rbp), %rax
	movq	%rax, -72(%rbp)
	movl	-124(%rbp), %eax
	cltq
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-88(%rbp), %rax
	movq	-8(%rbp), %rdx
	movq	%rdx, (%rax)
	movq	-88(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, -56(%rbp)
	movq	$26, -48(%rbp)
	jmp	.L137
.L119:
	movl	-112(%rbp), %eax
	cmpl	-128(%rbp), %eax
	jge	.L140
	movq	$36, -48(%rbp)
	jmp	.L137
.L140:
	movq	$16, -48(%rbp)
	jmp	.L137
.L130:
	subq	$1, -64(%rbp)
	movq	$29, -48(%rbp)
	jmp	.L137
.L123:
	movq	-56(%rbp), %rax
	movq	%rax, -16(%rbp)
	addq	$1, -56(%rbp)
	movq	-16(%rbp), %rax
	movb	$32, (%rax)
	movq	$13, -48(%rbp)
	jmp	.L137
.L135:
	addl	$1, -108(%rbp)
	addq	$1, -96(%rbp)
	movq	$2, -48(%rbp)
	jmp	.L137
.L127:
	movq	-104(%rbp), %rax
	jmp	.L168
.L115:
	movl	$0, -108(%rbp)
	movq	-96(%rbp), %rax
	movq	%rax, -80(%rbp)
	movq	$2, -48(%rbp)
	jmp	.L137
.L122:
	movq	-80(%rbp), %rdx
	movq	-72(%rbp), %rax
	cmpq	%rax, %rdx
	jnb	.L143
	movq	$39, -48(%rbp)
	jmp	.L137
.L143:
	movq	$13, -48(%rbp)
	jmp	.L137
.L131:
	movl	-132(%rbp), %eax
	cltq
	leaq	-1(%rax), %rdx
	movq	-120(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	cmpb	$10, %al
	jne	.L145
	movq	$42, -48(%rbp)
	jmp	.L137
.L145:
	movq	$32, -48(%rbp)
	jmp	.L137
.L129:
	movq	-88(%rbp), %rax
	movq	(%rax), %rdx
	movl	-124(%rbp), %eax
	cltq
	addq	%rdx, %rax
	movq	%rax, %rdx
	movq	-56(%rbp), %rax
	cmpq	%rax, %rdx
	jbe	.L147
	movq	$23, -48(%rbp)
	jmp	.L137
.L147:
	movq	$4, -48(%rbp)
	jmp	.L137
.L125:
	movq	-64(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$32, %al
	je	.L149
	movq	$5, -48(%rbp)
	jmp	.L137
.L149:
	movq	$35, -48(%rbp)
	jmp	.L137
.L118:
	movl	$0, -112(%rbp)
	movq	$31, -48(%rbp)
	jmp	.L137
.L132:
	movq	-64(%rbp), %rax
	movq	%rax, -96(%rbp)
	movq	$14, -48(%rbp)
	jmp	.L137
.L117:
	movq	-64(%rbp), %rax
	addq	$1, %rax
	movq	%rax, -96(%rbp)
	movq	$14, -48(%rbp)
	jmp	.L137
.L109:
	movl	-132(%rbp), %eax
	movslq	%eax, %rdx
	movq	-120(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, %rdx
	movq	-96(%rbp), %rax
	cmpq	%rax, %rdx
	jbe	.L151
	movq	$3, -48(%rbp)
	jmp	.L137
.L151:
	movq	$47, -48(%rbp)
	jmp	.L137
.L121:
	addq	$1, -96(%rbp)
	movq	$4, -48(%rbp)
	jmp	.L137
.L110:
	movl	-132(%rbp), %eax
	movslq	%eax, %rdx
	movq	-120(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, %rdx
	movq	-96(%rbp), %rax
	cmpq	%rax, %rdx
	jbe	.L153
	movq	$43, -48(%rbp)
	jmp	.L137
.L153:
	movq	$14, -48(%rbp)
	jmp	.L137
.L133:
	movq	-64(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$45, %al
	je	.L155
	movq	$12, -48(%rbp)
	jmp	.L137
.L155:
	movq	$35, -48(%rbp)
	jmp	.L137
.L114:
	movl	-128(%rbp), %eax
	cltq
	salq	$3, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, -104(%rbp)
	movq	-120(%rbp), %rax
	movq	%rax, -96(%rbp)
	movq	-104(%rbp), %rax
	movq	%rax, -88(%rbp)
	movq	$9, -48(%rbp)
	jmp	.L137
.L112:
	subl	$1, -132(%rbp)
	movq	$32, -48(%rbp)
	jmp	.L137
.L113:
	movq	-56(%rbp), %rax
	movq	%rax, -40(%rbp)
	addq	$1, -56(%rbp)
	movq	-80(%rbp), %rax
	movq	%rax, -32(%rbp)
	addq	$1, -80(%rbp)
	movq	-32(%rbp), %rax
	movzbl	(%rax), %edx
	movq	-40(%rbp), %rax
	movb	%dl, (%rax)
	movq	$26, -48(%rbp)
	jmp	.L137
.L116:
	movq	-64(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$45, %al
	jne	.L157
	movq	$34, -48(%rbp)
	jmp	.L137
.L157:
	movq	$6, -48(%rbp)
	jmp	.L137
.L120:
	movq	-64(%rbp), %rdx
	movq	-80(%rbp), %rax
	cmpq	%rax, %rdx
	jbe	.L159
	movq	$19, -48(%rbp)
	jmp	.L137
.L159:
	movq	$35, -48(%rbp)
	jmp	.L137
.L111:
	movq	-96(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$32, %al
	je	.L161
	movq	$49, -48(%rbp)
	jmp	.L137
.L161:
	movq	$14, -48(%rbp)
	jmp	.L137
.L136:
	movl	-108(%rbp), %eax
	cmpl	-124(%rbp), %eax
	jge	.L163
	movq	$48, -48(%rbp)
	jmp	.L137
.L163:
	movq	$47, -48(%rbp)
	jmp	.L137
.L124:
	movq	-96(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$32, %al
	jne	.L165
	movq	$28, -48(%rbp)
	jmp	.L137
.L165:
	movq	$50, -48(%rbp)
	jmp	.L137
.L169:
	nop
.L137:
	jmp	.L167
.L168:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	divide_input, .-divide_input
	.globl	justify_rows
	.type	justify_rows, @function
justify_rows:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$112, %rsp
	movq	%rdi, -104(%rbp)
	movl	%esi, -108(%rbp)
	movl	%edx, -112(%rbp)
	movq	$29, -16(%rbp)
.L242:
	cmpq	$65, -16(%rbp)
	ja	.L243
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L173(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L173(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L173:
	.long	.L243-.L173
	.long	.L209-.L173
	.long	.L243-.L173
	.long	.L243-.L173
	.long	.L208-.L173
	.long	.L243-.L173
	.long	.L207-.L173
	.long	.L243-.L173
	.long	.L206-.L173
	.long	.L205-.L173
	.long	.L204-.L173
	.long	.L203-.L173
	.long	.L202-.L173
	.long	.L243-.L173
	.long	.L243-.L173
	.long	.L243-.L173
	.long	.L201-.L173
	.long	.L200-.L173
	.long	.L243-.L173
	.long	.L199-.L173
	.long	.L198-.L173
	.long	.L197-.L173
	.long	.L196-.L173
	.long	.L244-.L173
	.long	.L243-.L173
	.long	.L243-.L173
	.long	.L243-.L173
	.long	.L243-.L173
	.long	.L243-.L173
	.long	.L194-.L173
	.long	.L243-.L173
	.long	.L243-.L173
	.long	.L193-.L173
	.long	.L192-.L173
	.long	.L191-.L173
	.long	.L243-.L173
	.long	.L243-.L173
	.long	.L190-.L173
	.long	.L189-.L173
	.long	.L188-.L173
	.long	.L187-.L173
	.long	.L243-.L173
	.long	.L186-.L173
	.long	.L185-.L173
	.long	.L184-.L173
	.long	.L183-.L173
	.long	.L243-.L173
	.long	.L243-.L173
	.long	.L182-.L173
	.long	.L243-.L173
	.long	.L181-.L173
	.long	.L243-.L173
	.long	.L243-.L173
	.long	.L243-.L173
	.long	.L243-.L173
	.long	.L243-.L173
	.long	.L180-.L173
	.long	.L179-.L173
	.long	.L178-.L173
	.long	.L177-.L173
	.long	.L243-.L173
	.long	.L176-.L173
	.long	.L175-.L173
	.long	.L243-.L173
	.long	.L174-.L173
	.long	.L172-.L173
	.text
.L181:
	movq	-24(%rbp), %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	movl	%eax, %edi
	call	putchar@PLT
	movq	$43, -16(%rbp)
	jmp	.L210
.L208:
	movl	-40(%rbp), %eax
	cmpl	-112(%rbp), %eax
	jge	.L211
	movq	$33, -16(%rbp)
	jmp	.L210
.L211:
	movq	$8, -16(%rbp)
	jmp	.L210
.L175:
	cmpl	$1, -80(%rbp)
	jne	.L213
	movq	$45, -16(%rbp)
	jmp	.L210
.L213:
	movq	$56, -16(%rbp)
	jmp	.L210
.L180:
	movl	-112(%rbp), %eax
	subl	-76(%rbp), %eax
	movl	%eax, -32(%rbp)
	movl	-80(%rbp), %eax
	leal	-1(%rax), %esi
	movl	-32(%rbp), %eax
	cltd
	idivl	%esi
	movl	%eax, -48(%rbp)
	movl	-80(%rbp), %eax
	leal	-1(%rax), %ecx
	movl	-32(%rbp), %eax
	cltd
	idivl	%ecx
	movl	%edx, -44(%rbp)
	movl	$0, -40(%rbp)
	movq	$4, -16(%rbp)
	jmp	.L210
.L202:
	movl	-112(%rbp), %eax
	subl	$1, %eax
	cmpl	%eax, -40(%rbp)
	jge	.L215
	movq	$58, -16(%rbp)
	jmp	.L210
.L215:
	movq	$43, -16(%rbp)
	jmp	.L210
.L206:
	movl	$10, %edi
	call	putchar@PLT
	addl	$1, -84(%rbp)
	movq	$64, -16(%rbp)
	jmp	.L210
.L183:
	movl	-112(%rbp), %eax
	subl	-76(%rbp), %eax
	movl	%eax, -28(%rbp)
	movl	-28(%rbp), %eax
	movl	%eax, %edx
	shrl	$31, %edx
	addl	%edx, %eax
	sarl	%eax
	movl	%eax, %ecx
	movl	-28(%rbp), %eax
	cltd
	shrl	$31, %edx
	addl	%edx, %eax
	andl	$1, %eax
	subl	%edx, %eax
	addl	%ecx, %eax
	movl	%eax, -68(%rbp)
	movl	-28(%rbp), %eax
	movl	%eax, %edx
	shrl	$31, %edx
	addl	%edx, %eax
	sarl	%eax
	movl	%eax, -64(%rbp)
	movl	$0, -60(%rbp)
	movq	$40, -16(%rbp)
	jmp	.L210
.L209:
	movq	-24(%rbp), %rax
	subq	$1, %rax
	movzbl	(%rax), %eax
	cmpb	$32, %al
	jne	.L217
	movq	$65, -16(%rbp)
	jmp	.L210
.L217:
	movq	$39, -16(%rbp)
	jmp	.L210
.L201:
	cmpl	$0, -44(%rbp)
	jle	.L220
	movq	$57, -16(%rbp)
	jmp	.L210
.L220:
	movq	$17, -16(%rbp)
	jmp	.L210
.L197:
	movl	-84(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-104(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movq	%rax, -24(%rbp)
	movl	$0, -80(%rbp)
	movl	$0, -76(%rbp)
	movl	$0, -72(%rbp)
	movq	$42, -16(%rbp)
	jmp	.L210
.L179:
	movl	$32, %edi
	call	putchar@PLT
	subl	$1, -44(%rbp)
	movq	$17, -16(%rbp)
	jmp	.L210
.L203:
	movl	-36(%rbp), %eax
	cmpl	-48(%rbp), %eax
	jge	.L222
	movq	$20, -16(%rbp)
	jmp	.L210
.L222:
	movq	$16, -16(%rbp)
	jmp	.L210
.L205:
	movl	-84(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-104(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	$62, -16(%rbp)
	jmp	.L210
.L199:
	addl	$1, -76(%rbp)
	movq	$59, -16(%rbp)
	jmp	.L210
.L193:
	movl	-56(%rbp), %eax
	cmpl	-76(%rbp), %eax
	jge	.L224
	movq	$34, -16(%rbp)
	jmp	.L210
.L224:
	movq	$6, -16(%rbp)
	jmp	.L210
.L200:
	movq	-24(%rbp), %rax
	addq	$1, %rax
	movzbl	(%rax), %eax
	cmpb	$32, %al
	jne	.L226
	movq	$12, -16(%rbp)
	jmp	.L210
.L226:
	movq	$43, -16(%rbp)
	jmp	.L210
.L187:
	movl	-60(%rbp), %eax
	cmpl	-68(%rbp), %eax
	jge	.L228
	movq	$10, -16(%rbp)
	jmp	.L210
.L228:
	movq	$61, -16(%rbp)
	jmp	.L210
.L177:
	cmpl	$0, -72(%rbp)
	jne	.L230
	movq	$44, -16(%rbp)
	jmp	.L210
.L230:
	movq	$1, -16(%rbp)
	jmp	.L210
.L207:
	movl	$0, -52(%rbp)
	movq	$37, -16(%rbp)
	jmp	.L210
.L189:
	movl	$0, -36(%rbp)
	movq	$11, -16(%rbp)
	jmp	.L210
.L176:
	movl	$0, -56(%rbp)
	movq	$32, -16(%rbp)
	jmp	.L210
.L178:
	addq	$1, -24(%rbp)
	addl	$1, -40(%rbp)
	movq	$17, -16(%rbp)
	jmp	.L210
.L191:
	movq	-24(%rbp), %rax
	movq	%rax, -8(%rbp)
	addq	$1, -24(%rbp)
	movq	-8(%rbp), %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	movl	%eax, %edi
	call	putchar@PLT
	addl	$1, -56(%rbp)
	movq	$32, -16(%rbp)
	jmp	.L210
.L182:
	movq	-24(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$32, %al
	je	.L232
	movq	$19, -16(%rbp)
	jmp	.L210
.L232:
	movq	$39, -16(%rbp)
	jmp	.L210
.L196:
	movl	$32, %edi
	call	putchar@PLT
	addl	$1, -52(%rbp)
	movq	$37, -16(%rbp)
	jmp	.L210
.L172:
	addl	$1, -80(%rbp)
	movq	$39, -16(%rbp)
	jmp	.L210
.L184:
	addl	$1, -80(%rbp)
	movq	$39, -16(%rbp)
	jmp	.L210
.L192:
	movq	-24(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$32, %al
	je	.L234
	movq	$50, -16(%rbp)
	jmp	.L210
.L234:
	movq	$38, -16(%rbp)
	jmp	.L210
.L190:
	movl	-52(%rbp), %eax
	cmpl	-64(%rbp), %eax
	jge	.L236
	movq	$22, -16(%rbp)
	jmp	.L210
.L236:
	movq	$8, -16(%rbp)
	jmp	.L210
.L174:
	movl	-84(%rbp), %eax
	cmpl	-108(%rbp), %eax
	jge	.L238
	movq	$21, -16(%rbp)
	jmp	.L210
.L238:
	movq	$23, -16(%rbp)
	jmp	.L210
.L204:
	movl	$32, %edi
	call	putchar@PLT
	addl	$1, -60(%rbp)
	movq	$40, -16(%rbp)
	jmp	.L210
.L186:
	movl	-72(%rbp), %eax
	cmpl	-112(%rbp), %eax
	jge	.L240
	movq	$48, -16(%rbp)
	jmp	.L210
.L240:
	movq	$9, -16(%rbp)
	jmp	.L210
.L188:
	addl	$1, -72(%rbp)
	addq	$1, -24(%rbp)
	movq	$42, -16(%rbp)
	jmp	.L210
.L194:
	movl	$0, -84(%rbp)
	movq	$64, -16(%rbp)
	jmp	.L210
.L185:
	addl	$1, -40(%rbp)
	addq	$1, -24(%rbp)
	movq	$4, -16(%rbp)
	jmp	.L210
.L198:
	movl	$32, %edi
	call	putchar@PLT
	addl	$1, -36(%rbp)
	movq	$11, -16(%rbp)
	jmp	.L210
.L243:
	nop
.L210:
	jmp	.L242
.L244:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	justify_rows, .-justify_rows
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
