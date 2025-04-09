	.file	"TheProjecter_qvm_q_flatten.c"
	.text
	.local	types
	.comm	types,48,32
	.globl	_TIG_IZ_ld3M_argv
	.bss
	.align 8
	.type	_TIG_IZ_ld3M_argv, @object
	.size	_TIG_IZ_ld3M_argv, 8
_TIG_IZ_ld3M_argv:
	.zero	8
	.local	serr
	.comm	serr,8,8
	.local	hex_tab
	.comm	hex_tab,448,32
	.local	loaded_modules
	.comm	loaded_modules,16,16
	.globl	_TIG_IZ_ld3M_argc
	.align 4
	.type	_TIG_IZ_ld3M_argc, @object
	.size	_TIG_IZ_ld3M_argc, 4
_TIG_IZ_ld3M_argc:
	.zero	4
	.local	nil
	.comm	nil,64,32
	.local	one_char_tokens
	.comm	one_char_tokens,8,8
	.local	rerr
	.comm	rerr,8,8
	.local	opcodes
	.comm	opcodes,96,32
	.globl	_TIG_IZ_ld3M_envp
	.align 8
	.type	_TIG_IZ_ld3M_envp, @object
	.size	_TIG_IZ_ld3M_envp, 8
_TIG_IZ_ld3M_envp:
	.zero	8
	.local	long_tokens
	.comm	long_tokens,120,32
	.local	op_tab
	.comm	op_tab,1024,32
	.section	.rodata
.LC0:
	.string	"r"
.LC1:
	.string	"fopen(%s): %s"
.LC2:
	.string	"compile"
.LC3:
	.string	"TheProjecter_qvm_q.c"
.LC4:
	.string	"q->sc == 1"
.LC5:
	.string	"fread(%s): %s"
.LC6:
	.string	"%s: file is too large"
	.text
	.type	compile, @function
compile:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	leaq	-32768(%rsp), %r11
.LPSRL0:
	subq	$4096, %rsp
	orq	$0, (%rsp)
	cmpq	%r11, %rsp
	jne	.LPSRL0
	subq	$112, %rsp
	movq	%rdi, -32872(%rbp)
	movq	%rsi, -32880(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$9, -32840(%rbp)
.L29:
	cmpq	$22, -32840(%rbp)
	ja	.L32
	movq	-32840(%rbp), %rax
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
	.long	.L33-.L4
	.long	.L32-.L4
	.long	.L32-.L4
	.long	.L32-.L4
	.long	.L17-.L4
	.long	.L16-.L4
	.long	.L15-.L4
	.long	.L14-.L4
	.long	.L13-.L4
	.long	.L12-.L4
	.long	.L11-.L4
	.long	.L10-.L4
	.long	.L32-.L4
	.long	.L9-.L4
	.long	.L8-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L32-.L4
	.long	.L32-.L4
	.long	.L32-.L4
	.long	.L32-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L17:
	movq	-32848(%rbp), %rdx
	leaq	-32784(%rbp), %rax
	movq	%rdx, %rcx
	movl	$32768, %edx
	movl	$1, %esi
	movq	%rax, %rdi
	call	fread@PLT
	movq	%rax, -32792(%rbp)
	movq	-32792(%rbp), %rax
	movl	%eax, -32852(%rbp)
	movq	$13, -32840(%rbp)
	jmp	.L19
.L8:
	movq	-32872(%rbp), %rax
	movq	%rax, %rdi
	call	import_builtin_objects
	movq	-32872(%rbp), %rax
	movq	%rax, %rdi
	call	disasm
	movq	-32872(%rbp), %rax
	movl	$0, 48(%rax)
	movq	-32872(%rbp), %rax
	movq	%rax, %rdi
	call	execute
	movq	$5, -32840(%rbp)
	jmp	.L19
.L7:
	movl	$0, -32852(%rbp)
	movq	-32880(%rbp), %rax
	leaq	.LC0(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -32848(%rbp)
	movq	$8, -32840(%rbp)
	jmp	.L19
.L13:
	cmpq	$0, -32848(%rbp)
	jne	.L20
	movq	$16, -32840(%rbp)
	jmp	.L19
.L20:
	movq	$4, -32840(%rbp)
	jmp	.L19
.L6:
	call	__errno_location@PLT
	movq	%rax, -32824(%rbp)
	movq	-32824(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, %edi
	call	strerror@PLT
	movq	%rax, -32816(%rbp)
	movq	-32816(%rbp), %rcx
	movq	-32880(%rbp), %rdx
	movq	-32872(%rbp), %rax
	leaq	.LC1(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	die
	movq	$21, -32840(%rbp)
	jmp	.L19
.L5:
	movl	-32852(%rbp), %eax
	cltq
	movb	$0, -32784(%rbp,%rax)
	movq	-32848(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	leaq	-32784(%rbp), %rdx
	movq	-32872(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	q_exec_string
	movq	-32872(%rbp), %rax
	movq	%rax, %rdi
	call	push
	movq	%rax, -32832(%rbp)
	movq	-32832(%rbp), %rax
	movl	$0, %esi
	movq	%rax, %rdi
	call	init_hash_obj
	movq	$7, -32840(%rbp)
	jmp	.L19
.L10:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rcx
	movl	$1208, %edx
	leaq	.LC3(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L12:
	movq	$15, -32840(%rbp)
	jmp	.L19
.L9:
	cmpl	$0, -32852(%rbp)
	jns	.L22
	movq	$6, -32840(%rbp)
	jmp	.L19
.L22:
	movq	$10, -32840(%rbp)
	jmp	.L19
.L15:
	call	__errno_location@PLT
	movq	%rax, -32808(%rbp)
	movq	-32808(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, %edi
	call	strerror@PLT
	movq	%rax, -32800(%rbp)
	movq	-32800(%rbp), %rcx
	movq	-32880(%rbp), %rdx
	movq	-32872(%rbp), %rax
	leaq	.LC5(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	die
	movq	$21, -32840(%rbp)
	jmp	.L19
.L3:
	movq	-32880(%rbp), %rdx
	movq	-32872(%rbp), %rax
	leaq	.LC6(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	die
	movq	$21, -32840(%rbp)
	jmp	.L19
.L16:
	movq	loaded_modules(%rip), %rax
	movq	-32872(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movq	loaded_modules(%rip), %rdx
	movq	-32872(%rbp), %rax
	movq	%rdx, (%rax)
	movq	-32872(%rbp), %rax
	leaq	loaded_modules(%rip), %rdx
	movq	%rdx, 8(%rax)
	movq	-32872(%rbp), %rax
	movq	%rax, loaded_modules(%rip)
	movq	$0, -32840(%rbp)
	jmp	.L19
.L11:
	cmpl	$32768, -32852(%rbp)
	jne	.L24
	movq	$22, -32840(%rbp)
	jmp	.L19
.L24:
	movq	$21, -32840(%rbp)
	jmp	.L19
.L14:
	movq	-32872(%rbp), %rax
	movl	40(%rax), %eax
	cmpl	$1, %eax
	jne	.L27
	movq	$14, -32840(%rbp)
	jmp	.L19
.L27:
	movq	$11, -32840(%rbp)
	jmp	.L19
.L32:
	nop
.L19:
	jmp	.L29
.L33:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L31
	call	__stack_chk_fail@PLT
.L31:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	compile, .-compile
	.section	.rodata
	.align 8
.LC7:
	.string	"%s: line %d: badly encoded string"
	.text
	.type	hex_ascii_to_int, @function
hex_ascii_to_int:
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
	movl	%esi, %eax
	movl	%edx, -32(%rbp)
	movb	%al, -28(%rbp)
	movq	$3, -8(%rbp)
.L48:
	cmpq	$4, -8(%rbp)
	ja	.L50
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
	.long	.L41-.L37
	.long	.L40-.L37
	.long	.L39-.L37
	.long	.L38-.L37
	.long	.L36-.L37
	.text
.L36:
	movzbl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	hex_tab(%rip), %rax
	movl	(%rdx,%rax), %eax
	cmpl	$-1, %eax
	jne	.L42
	movq	$2, -8(%rbp)
	jmp	.L44
.L42:
	movq	$0, -8(%rbp)
	jmp	.L44
.L40:
	movq	serr(%rip), %rdx
	movl	-32(%rbp), %ecx
	movq	-24(%rbp), %rax
	leaq	.LC7(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	die
	movq	$0, -8(%rbp)
	jmp	.L44
.L38:
	cmpb	$111, -28(%rbp)
	jbe	.L45
	movq	$1, -8(%rbp)
	jmp	.L44
.L45:
	movq	$4, -8(%rbp)
	jmp	.L44
.L41:
	movzbl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	hex_tab(%rip), %rax
	movl	(%rdx,%rax), %eax
	jmp	.L49
.L39:
	movq	serr(%rip), %rdx
	movl	-32(%rbp), %ecx
	movq	-24(%rbp), %rax
	leaq	.LC7(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	die
	movq	$0, -8(%rbp)
	jmp	.L44
.L50:
	nop
.L44:
	jmp	.L48
.L49:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	hex_ascii_to_int, .-hex_ascii_to_int
	.type	hash_str, @function
hash_str:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	$2, -8(%rbp)
.L60:
	cmpq	$6, -8(%rbp)
	je	.L52
	cmpq	$6, -8(%rbp)
	ja	.L62
	cmpq	$5, -8(%rbp)
	je	.L54
	cmpq	$5, -8(%rbp)
	ja	.L62
	cmpq	$2, -8(%rbp)
	je	.L55
	cmpq	$4, -8(%rbp)
	jne	.L62
	movq	-32(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, -12(%rbp)
	jge	.L56
	movq	$6, -8(%rbp)
	jmp	.L58
.L56:
	movq	$5, -8(%rbp)
	jmp	.L58
.L52:
	movq	-24(%rbp), %rax
	salq	$5, %rax
	subq	-24(%rbp), %rax
	movq	%rax, %rdx
	movq	-32(%rbp), %rax
	movq	8(%rax), %rcx
	movl	-12(%rbp), %eax
	cltq
	addq	%rcx, %rax
	movzbl	(%rax), %eax
	movsbq	%al, %rax
	addq	%rdx, %rax
	movq	%rax, -24(%rbp)
	addl	$1, -12(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L58
.L54:
	movq	-24(%rbp), %rax
	jmp	.L61
.L55:
	movl	$0, -12(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L58
.L62:
	nop
.L58:
	jmp	.L60
.L61:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	hash_str, .-hash_str
	.type	q_exec_string, @function
q_exec_string:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	leaq	-2097152(%rsp), %r11
.LPSRL1:
	subq	$4096, %rsp
	orq	$0, (%rsp)
	cmpq	%r11, %rsp
	jne	.LPSRL1
	subq	$64, %rsp
	movq	%rdi, -2097208(%rbp)
	movq	%rsi, -2097216(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$3, -2097176(%rbp)
.L76:
	cmpq	$8, -2097176(%rbp)
	ja	.L79
	movq	-2097176(%rbp), %rax
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
	.long	.L80-.L66
	.long	.L70-.L66
	.long	.L79-.L66
	.long	.L69-.L66
	.long	.L79-.L66
	.long	.L68-.L66
	.long	.L79-.L66
	.long	.L67-.L66
	.long	.L65-.L66
	.text
.L65:
	movl	-2097192(%rbp), %eax
	cltq
	salq	$6, %rax
	movq	%rax, %rdx
	leaq	-2097168(%rbp), %rax
	leaq	(%rax,%rdx), %rsi
	leaq	-2097184(%rbp), %rdx
	movq	-2097208(%rbp), %rax
	movq	%rdx, %rcx
	movl	$59, %edx
	movq	%rax, %rdi
	call	expr
	movl	%eax, -2097188(%rbp)
	movl	-2097188(%rbp), %eax
	addl	%eax, -2097192(%rbp)
	movq	-2097184(%rbp), %rdx
	movq	-2097208(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	emit_expr
	movq	-2097208(%rbp), %rax
	movl	$7, %esi
	movq	%rax, %rdi
	call	emit_byte
	movq	$5, -2097176(%rbp)
	jmp	.L72
.L70:
	movq	-2097208(%rbp), %rax
	movl	$0, %esi
	movq	%rax, %rdi
	call	emit_byte
	movq	$0, -2097176(%rbp)
	jmp	.L72
.L69:
	movq	$7, -2097176(%rbp)
	jmp	.L72
.L68:
	movl	-2097192(%rbp), %eax
	cltq
	salq	$6, %rax
	addq	%rbp, %rax
	subq	$2097136, %rax
	movl	(%rax), %eax
	cmpl	$111, %eax
	je	.L73
	movq	$8, -2097176(%rbp)
	jmp	.L72
.L73:
	movq	$1, -2097176(%rbp)
	jmp	.L72
.L67:
	movl	$0, -2097192(%rbp)
	leaq	-2097168(%rbp), %rdx
	movq	-2097216(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	tokenize
	movq	$5, -2097176(%rbp)
	jmp	.L72
.L79:
	nop
.L72:
	jmp	.L76
.L80:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L78
	call	__stack_chk_fail@PLT
.L78:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	q_exec_string, .-q_exec_string
	.type	is_op, @function
is_op:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -20(%rbp)
	movq	$0, -8(%rbp)
.L84:
	cmpq	$0, -8(%rbp)
	jne	.L87
	movl	-20(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	op_tab(%rip), %rax
	movl	(%rdx,%rax), %eax
	jmp	.L86
.L87:
	nop
	jmp	.L84
.L86:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	is_op, .-is_op
	.type	is_leftmost, @function
is_leftmost:
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
	movq	$7, -8(%rbp)
.L118:
	cmpq	$13, -8(%rbp)
	ja	.L119
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L91(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L91(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L91:
	.long	.L111-.L91
	.long	.L101-.L91
	.long	.L100-.L91
	.long	.L99-.L91
	.long	.L98-.L91
	.long	.L97-.L91
	.long	.L96-.L91
	.long	.L95-.L91
	.long	.L94-.L91
	.long	.L93-.L91
	.long	.L119-.L91
	.long	.L92-.L91
	.long	.L119-.L91
	.long	.L90-.L91
	.text
.L98:
	movl	-12(%rbp), %eax
	jmp	.L103
.L94:
	movl	$0, %eax
	jmp	.L103
.L101:
	movl	$1, %eax
	jmp	.L103
.L99:
	movl	$0, %eax
	jmp	.L103
.L92:
	movq	-24(%rbp), %rax
	movq	56(%rax), %rax
	movl	32(%rax), %eax
	cmpl	$61, %eax
	jne	.L104
	movq	$1, -8(%rbp)
	jmp	.L106
.L104:
	movq	$0, -8(%rbp)
	jmp	.L106
.L93:
	movq	-24(%rbp), %rax
	movq	56(%rax), %rax
	movl	32(%rax), %eax
	cmpl	$46, %eax
	jne	.L107
	movq	$13, -8(%rbp)
	jmp	.L106
.L107:
	movq	$6, -8(%rbp)
	jmp	.L106
.L90:
	movq	-24(%rbp), %rax
	movq	56(%rax), %rax
	movq	%rax, %rdi
	call	is_leftmost
	movl	%eax, -12(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L106
.L96:
	movl	$0, %eax
	jmp	.L103
.L97:
	movq	-24(%rbp), %rax
	movq	56(%rax), %rax
	movq	40(%rax), %rax
	movq	%rax, %rdx
	movq	-24(%rbp), %rax
	cmpq	%rax, %rdx
	jne	.L109
	movq	$11, -8(%rbp)
	jmp	.L106
.L109:
	movq	$0, -8(%rbp)
	jmp	.L106
.L102:
.L111:
	movq	-24(%rbp), %rax
	movq	56(%rax), %rax
	testq	%rax, %rax
	je	.L112
	movq	$2, -8(%rbp)
	jmp	.L106
.L112:
	movq	$8, -8(%rbp)
	jmp	.L106
.L95:
	movq	-24(%rbp), %rax
	movq	56(%rax), %rax
	testq	%rax, %rax
	je	.L114
	movq	$5, -8(%rbp)
	jmp	.L106
.L114:
	movq	$0, -8(%rbp)
	jmp	.L106
.L100:
	movq	-24(%rbp), %rax
	movq	56(%rax), %rax
	movq	40(%rax), %rax
	movq	%rax, %rdx
	movq	-24(%rbp), %rax
	cmpq	%rax, %rdx
	jne	.L116
	movq	$9, -8(%rbp)
	jmp	.L106
.L116:
	movq	$3, -8(%rbp)
	jmp	.L106
.L119:
	nop
.L106:
	jmp	.L118
.L103:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	is_leftmost, .-is_leftmost
	.type	init_hash_obj, @function
init_hash_obj:
.LFB6:
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
	movq	$3, -32(%rbp)
.L132:
	cmpq	$5, -32(%rbp)
	ja	.L133
	movq	-32(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L123(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L123(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L123:
	.long	.L127-.L123
	.long	.L134-.L123
	.long	.L133-.L123
	.long	.L125-.L123
	.long	.L124-.L123
	.long	.L122-.L123
	.text
.L124:
	cmpq	$12, -40(%rbp)
	ja	.L128
	movq	$0, -32(%rbp)
	jmp	.L130
.L128:
	movq	$1, -32(%rbp)
	jmp	.L130
.L125:
	movq	$5, -32(%rbp)
	jmp	.L130
.L122:
	movq	-56(%rbp), %rax
	movb	$3, (%rax)
	movl	$232, %edi
	call	malloc@PLT
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, -48(%rbp)
	movq	-56(%rbp), %rax
	movq	-48(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movq	-48(%rbp), %rax
	movq	-64(%rbp), %rdx
	movq	%rdx, (%rax)
	movq	-48(%rbp), %rax
	addq	$8, %rax
	movq	%rax, -16(%rbp)
	movq	-48(%rbp), %rax
	movq	-16(%rbp), %rdx
	movq	%rdx, 16(%rax)
	movq	-48(%rbp), %rax
	movq	-16(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movq	$0, -40(%rbp)
	movq	$4, -32(%rbp)
	jmp	.L130
.L127:
	movq	-48(%rbp), %rax
	leaq	24(%rax), %rdx
	movq	-40(%rbp), %rax
	salq	$4, %rax
	addq	%rdx, %rax
	movq	%rax, -8(%rbp)
	movq	-48(%rbp), %rax
	leaq	24(%rax), %rdx
	movq	-40(%rbp), %rax
	salq	$4, %rax
	addq	%rax, %rdx
	movq	-8(%rbp), %rax
	movq	%rax, 8(%rdx)
	movq	-48(%rbp), %rax
	leaq	24(%rax), %rdx
	movq	-40(%rbp), %rax
	salq	$4, %rax
	addq	%rax, %rdx
	movq	-8(%rbp), %rax
	movq	%rax, (%rdx)
	addq	$1, -40(%rbp)
	movq	$4, -32(%rbp)
	jmp	.L130
.L133:
	nop
.L130:
	jmp	.L132
.L134:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	init_hash_obj, .-init_hash_obj
	.type	lookup_ins, @function
lookup_ins:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$80, %rsp
	movq	%rdi, -56(%rbp)
	movq	%rsi, -64(%rbp)
	movq	%rdx, -72(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$4, -24(%rbp)
.L152:
	movq	-24(%rbp), %rax
	subq	$3, %rax
	cmpq	$8, %rax
	ja	.L155
	leaq	0(,%rax,4), %rdx
	leaq	.L138(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L138(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L138:
	.long	.L145-.L138
	.long	.L144-.L138
	.long	.L155-.L138
	.long	.L143-.L138
	.long	.L142-.L138
	.long	.L141-.L138
	.long	.L140-.L138
	.long	.L139-.L138
	.long	.L137-.L138
	.text
.L144:
	movq	$10, -24(%rbp)
	jmp	.L146
.L141:
	movq	-40(%rbp), %rcx
	movq	-72(%rbp), %rsi
	movq	(%rsi), %rax
	movq	8(%rsi), %rdx
	movq	%rax, 40(%rcx)
	movq	%rdx, 48(%rcx)
	movq	16(%rsi), %rax
	movq	%rax, 56(%rcx)
	movq	$3, -24(%rbp)
	jmp	.L146
.L145:
	movq	-32(%rbp), %rax
	leaq	24(%rax), %rdx
	movq	-48(%rbp), %rax
	salq	$4, %rax
	addq	%rdx, %rax
	movq	8(%rax), %rax
	movq	-40(%rbp), %rdx
	movq	%rdx, (%rax)
	movq	-32(%rbp), %rax
	leaq	24(%rax), %rdx
	movq	-48(%rbp), %rax
	salq	$4, %rax
	addq	%rdx, %rax
	movq	8(%rax), %rdx
	movq	-40(%rbp), %rax
	movq	%rdx, 8(%rax)
	movq	-32(%rbp), %rax
	leaq	24(%rax), %rdx
	movq	-48(%rbp), %rax
	salq	$4, %rax
	addq	%rax, %rdx
	movq	-40(%rbp), %rax
	movq	%rdx, (%rax)
	movq	-32(%rbp), %rax
	leaq	24(%rax), %rdx
	movq	-48(%rbp), %rax
	salq	$4, %rax
	addq	%rax, %rdx
	movq	-40(%rbp), %rax
	movq	%rax, 8(%rdx)
	movq	$11, -24(%rbp)
	jmp	.L146
.L137:
	movq	-40(%rbp), %rax
	addq	$40, %rax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L153
	jmp	.L154
.L140:
	cmpq	$0, -40(%rbp)
	jne	.L148
	movq	$7, -24(%rbp)
	jmp	.L146
.L148:
	movq	$11, -24(%rbp)
	jmp	.L146
.L143:
	cmpq	$0, -72(%rbp)
	je	.L150
	movq	$8, -24(%rbp)
	jmp	.L146
.L150:
	movq	$3, -24(%rbp)
	jmp	.L146
.L139:
	movq	-56(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, -32(%rbp)
	leaq	-48(%rbp), %rdx
	movq	-64(%rbp), %rcx
	movq	-56(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	lookup
	movq	%rax, -40(%rbp)
	movq	$9, -24(%rbp)
	jmp	.L146
.L142:
	movl	$64, %esi
	movl	$1, %edi
	call	calloc@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -40(%rbp)
	movq	-40(%rbp), %rcx
	movq	-64(%rbp), %rsi
	movq	(%rsi), %rax
	movq	8(%rsi), %rdx
	movq	%rax, 16(%rcx)
	movq	%rdx, 24(%rcx)
	movq	16(%rsi), %rax
	movq	%rax, 32(%rcx)
	movq	$6, -24(%rbp)
	jmp	.L146
.L155:
	nop
.L146:
	jmp	.L152
.L154:
	call	__stack_chk_fail@PLT
.L153:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	lookup_ins, .-lookup_ins
	.type	bubble_sort_ops_by_priority, @function
bubble_sort_ops_by_priority:
.LFB9:
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
	movq	$1, -16(%rbp)
.L192:
	cmpq	$23, -16(%rbp)
	ja	.L193
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L159(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L159(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L159:
	.long	.L175-.L159
	.long	.L174-.L159
	.long	.L173-.L159
	.long	.L172-.L159
	.long	.L193-.L159
	.long	.L193-.L159
	.long	.L171-.L159
	.long	.L193-.L159
	.long	.L170-.L159
	.long	.L169-.L159
	.long	.L193-.L159
	.long	.L193-.L159
	.long	.L194-.L159
	.long	.L167-.L159
	.long	.L166-.L159
	.long	.L165-.L159
	.long	.L164-.L159
	.long	.L163-.L159
	.long	.L162-.L159
	.long	.L161-.L159
	.long	.L160-.L159
	.long	.L193-.L159
	.long	.L193-.L159
	.long	.L158-.L159
	.text
.L162:
	movl	-48(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movq	%rax, -8(%rbp)
	movl	-48(%rbp), %eax
	cltq
	addq	$1, %rax
	leaq	0(,%rax,8), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movl	-48(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,8), %rcx
	movq	-56(%rbp), %rdx
	addq	%rcx, %rdx
	movq	(%rax), %rax
	movq	%rax, (%rdx)
	movl	-48(%rbp), %eax
	cltq
	addq	$1, %rax
	leaq	0(,%rax,8), %rdx
	movq	-56(%rbp), %rax
	addq	%rax, %rdx
	movq	-8(%rbp), %rax
	movq	%rax, (%rdx)
	movl	$1, -44(%rbp)
	movq	$13, -16(%rbp)
	jmp	.L176
.L166:
	movl	-24(%rbp), %eax
	cmpl	-20(%rbp), %eax
	jge	.L177
	movq	$2, -16(%rbp)
	jmp	.L176
.L177:
	movq	$13, -16(%rbp)
	jmp	.L176
.L165:
	movl	-48(%rbp), %eax
	cltq
	addq	$1, %rax
	leaq	0(,%rax,8), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movl	32(%rax), %eax
	movl	%eax, %edi
	call	is_rtl
	movl	%eax, -36(%rbp)
	movq	$16, -16(%rbp)
	jmp	.L176
.L170:
	movl	-48(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movl	32(%rax), %eax
	movl	%eax, %edi
	call	prio
	movl	%eax, -24(%rbp)
	movl	-48(%rbp), %eax
	cltq
	addq	$1, %rax
	leaq	0(,%rax,8), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movl	32(%rax), %eax
	movl	%eax, %edi
	call	prio
	movl	%eax, -20(%rbp)
	movq	$14, -16(%rbp)
	jmp	.L176
.L174:
	movq	$6, -16(%rbp)
	jmp	.L176
.L158:
	movl	-48(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movq	%rax, %rdx
	movl	-48(%rbp), %eax
	cltq
	addq	$1, %rax
	leaq	0(,%rax,8), %rcx
	movq	-56(%rbp), %rax
	addq	%rcx, %rax
	movq	(%rax), %rax
	cmpq	%rax, %rdx
	jnb	.L180
	movq	$18, -16(%rbp)
	jmp	.L176
.L180:
	movq	$8, -16(%rbp)
	jmp	.L176
.L172:
	cmpl	$0, -40(%rbp)
	je	.L182
	movq	$15, -16(%rbp)
	jmp	.L176
.L182:
	movq	$8, -16(%rbp)
	jmp	.L176
.L164:
	cmpl	$0, -36(%rbp)
	je	.L184
	movq	$0, -16(%rbp)
	jmp	.L176
.L184:
	movq	$8, -16(%rbp)
	jmp	.L176
.L169:
	movl	-48(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movl	32(%rax), %eax
	movl	%eax, %edi
	call	is_rtl
	movl	%eax, -40(%rbp)
	movq	$3, -16(%rbp)
	jmp	.L176
.L167:
	addl	$1, -48(%rbp)
	movq	$20, -16(%rbp)
	jmp	.L176
.L161:
	cmpl	$1, -44(%rbp)
	jne	.L186
	movq	$6, -16(%rbp)
	jmp	.L176
.L186:
	movq	$12, -16(%rbp)
	jmp	.L176
.L163:
	movl	-32(%rbp), %eax
	cmpl	-28(%rbp), %eax
	jne	.L188
	movq	$23, -16(%rbp)
	jmp	.L176
.L188:
	movq	$8, -16(%rbp)
	jmp	.L176
.L171:
	movl	$0, -44(%rbp)
	movl	$0, -48(%rbp)
	movq	$20, -16(%rbp)
	jmp	.L176
.L175:
	movl	-48(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movl	32(%rax), %eax
	movl	%eax, %edi
	call	prio
	movl	%eax, -32(%rbp)
	movl	-48(%rbp), %eax
	cltq
	addq	$1, %rax
	leaq	0(,%rax,8), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movl	32(%rax), %eax
	movl	%eax, %edi
	call	prio
	movl	%eax, -28(%rbp)
	movq	$17, -16(%rbp)
	jmp	.L176
.L173:
	movl	-48(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movq	%rax, -8(%rbp)
	movl	-48(%rbp), %eax
	cltq
	addq	$1, %rax
	leaq	0(,%rax,8), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movl	-48(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,8), %rcx
	movq	-56(%rbp), %rdx
	addq	%rcx, %rdx
	movq	(%rax), %rax
	movq	%rax, (%rdx)
	movl	-48(%rbp), %eax
	cltq
	addq	$1, %rax
	leaq	0(,%rax,8), %rdx
	movq	-56(%rbp), %rax
	addq	%rax, %rdx
	movq	-8(%rbp), %rax
	movq	%rax, (%rdx)
	movl	$1, -44(%rbp)
	movq	$13, -16(%rbp)
	jmp	.L176
.L160:
	movl	-60(%rbp), %eax
	subl	$1, %eax
	cmpl	%eax, -48(%rbp)
	jge	.L190
	movq	$9, -16(%rbp)
	jmp	.L176
.L190:
	movq	$19, -16(%rbp)
	jmp	.L176
.L193:
	nop
.L176:
	jmp	.L192
.L194:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	bubble_sort_ops_by_priority, .-bubble_sort_ops_by_priority
	.type	tokenize, @function
tokenize:
.LFB10:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$80, %rsp
	movq	%rdi, -72(%rbp)
	movq	%rsi, -80(%rbp)
	movq	$21, -16(%rbp)
.L248:
	cmpq	$39, -16(%rbp)
	ja	.L250
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L198(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L198(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L198:
	.long	.L223-.L198
	.long	.L222-.L198
	.long	.L250-.L198
	.long	.L221-.L198
	.long	.L220-.L198
	.long	.L219-.L198
	.long	.L218-.L198
	.long	.L217-.L198
	.long	.L250-.L198
	.long	.L250-.L198
	.long	.L250-.L198
	.long	.L216-.L198
	.long	.L250-.L198
	.long	.L215-.L198
	.long	.L214-.L198
	.long	.L213-.L198
	.long	.L250-.L198
	.long	.L212-.L198
	.long	.L211-.L198
	.long	.L210-.L198
	.long	.L250-.L198
	.long	.L209-.L198
	.long	.L208-.L198
	.long	.L250-.L198
	.long	.L207-.L198
	.long	.L250-.L198
	.long	.L206-.L198
	.long	.L250-.L198
	.long	.L250-.L198
	.long	.L205-.L198
	.long	.L204-.L198
	.long	.L203-.L198
	.long	.L202-.L198
	.long	.L250-.L198
	.long	.L201-.L198
	.long	.L250-.L198
	.long	.L250-.L198
	.long	.L200-.L198
	.long	.L199-.L198
	.long	.L197-.L198
	.text
.L211:
	movl	-52(%rbp), %eax
	jmp	.L249
.L220:
	movl	-60(%rbp), %eax
	leal	97(%rax), %edx
	movq	-40(%rbp), %rax
	movl	%edx, 32(%rax)
	movq	$31, -16(%rbp)
	jmp	.L225
.L204:
	movl	-60(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	leaq	long_tokens(%rip), %rax
	movq	(%rdx,%rax), %rax
	testq	%rax, %rax
	je	.L226
	movq	$37, -16(%rbp)
	jmp	.L225
.L226:
	movq	$31, -16(%rbp)
	jmp	.L225
.L214:
	movq	-32(%rbp), %rax
	movzbl	(%rax), %eax
	movsbl	%al, %edx
	movq	-40(%rbp), %rax
	movl	%edx, 32(%rax)
	movq	-40(%rbp), %rax
	movl	$1, 16(%rax)
	movq	$32, -16(%rbp)
	jmp	.L225
.L213:
	movq	-72(%rbp), %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	je	.L228
	movq	$3, -16(%rbp)
	jmp	.L225
.L228:
	movq	$32, -16(%rbp)
	jmp	.L225
.L203:
	movq	-40(%rbp), %rax
	movl	16(%rax), %eax
	testl	%eax, %eax
	jne	.L230
	movq	$15, -16(%rbp)
	jmp	.L225
.L230:
	movq	$32, -16(%rbp)
	jmp	.L225
.L222:
	movq	-24(%rbp), %rax
	movq	(%rax), %rdx
	movq	-72(%rbp), %rax
	movzbl	(%rax), %eax
	movzbl	%al, %eax
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$8192, %eax
	testl	%eax, %eax
	je	.L232
	movq	$29, -16(%rbp)
	jmp	.L225
.L232:
	movq	$39, -16(%rbp)
	jmp	.L225
.L221:
	movq	-72(%rbp), %rax
	movzbl	(%rax), %eax
	movsbl	%al, %edx
	movq	one_char_tokens(%rip), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	strchr@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -32(%rbp)
	movq	$0, -16(%rbp)
	jmp	.L225
.L207:
	addq	$1, -72(%rbp)
	movq	$6, -16(%rbp)
	jmp	.L225
.L209:
	movq	$26, -16(%rbp)
	jmp	.L225
.L206:
	movl	$1, -56(%rbp)
	movl	$0, -52(%rbp)
	movq	$38, -16(%rbp)
	jmp	.L225
.L216:
	movl	-52(%rbp), %eax
	movl	%eax, -44(%rbp)
	addl	$1, -52(%rbp)
	movl	-44(%rbp), %eax
	cltq
	salq	$6, %rax
	movq	%rax, %rdx
	movq	-80(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, -40(%rbp)
	movq	-40(%rbp), %rax
	movl	$64, %edx
	movl	$0, %esi
	movq	%rax, %rdi
	call	memset@PLT
	movq	-40(%rbp), %rax
	movq	-72(%rbp), %rdx
	movq	%rdx, 24(%rax)
	movq	-40(%rbp), %rax
	movl	-56(%rbp), %edx
	movl	%edx, 36(%rax)
	movq	-40(%rbp), %rax
	movl	$111, 32(%rax)
	movl	$0, -60(%rbp)
	movq	$30, -16(%rbp)
	jmp	.L225
.L215:
	movq	-72(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$10, %al
	jne	.L234
	movq	$38, -16(%rbp)
	jmp	.L225
.L234:
	movq	$24, -16(%rbp)
	jmp	.L225
.L210:
	addl	$1, -60(%rbp)
	movq	$30, -16(%rbp)
	jmp	.L225
.L202:
	movq	-40(%rbp), %rax
	movl	16(%rax), %eax
	cltq
	addq	%rax, -72(%rbp)
	movq	$22, -16(%rbp)
	jmp	.L225
.L212:
	addq	$1, -72(%rbp)
	movq	$38, -16(%rbp)
	jmp	.L225
.L218:
	movq	-72(%rbp), %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	je	.L236
	movq	$13, -16(%rbp)
	jmp	.L225
.L236:
	movq	$11, -16(%rbp)
	jmp	.L225
.L199:
	call	__ctype_b_loc@PLT
	movq	%rax, -24(%rbp)
	movq	$1, -16(%rbp)
	jmp	.L225
.L201:
	addq	$1, -72(%rbp)
	movq	$6, -16(%rbp)
	jmp	.L225
.L208:
	movq	-40(%rbp), %rax
	movl	32(%rax), %eax
	cmpl	$111, %eax
	je	.L238
	movq	$38, -16(%rbp)
	jmp	.L225
.L238:
	movq	$18, -16(%rbp)
	jmp	.L225
.L219:
	addl	$1, -56(%rbp)
	movq	$17, -16(%rbp)
	jmp	.L225
.L200:
	movl	-60(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	leaq	long_tokens(%rip), %rax
	movq	(%rdx,%rax), %rax
	movq	-72(%rbp), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	match_long_token
	movl	%eax, -48(%rbp)
	movq	-40(%rbp), %rax
	movl	-48(%rbp), %edx
	movl	%edx, 16(%rax)
	movq	$7, -16(%rbp)
	jmp	.L225
.L223:
	cmpq	$0, -32(%rbp)
	je	.L240
	movq	$14, -16(%rbp)
	jmp	.L225
.L240:
	movq	$32, -16(%rbp)
	jmp	.L225
.L197:
	movq	-72(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$35, %al
	jne	.L242
	movq	$34, -16(%rbp)
	jmp	.L225
.L242:
	movq	$11, -16(%rbp)
	jmp	.L225
.L217:
	cmpl	$0, -48(%rbp)
	jle	.L244
	movq	$4, -16(%rbp)
	jmp	.L225
.L244:
	movq	$19, -16(%rbp)
	jmp	.L225
.L205:
	movq	-72(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$10, %al
	jne	.L246
	movq	$5, -16(%rbp)
	jmp	.L225
.L246:
	movq	$17, -16(%rbp)
	jmp	.L225
.L250:
	nop
.L225:
	jmp	.L248
.L249:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	tokenize, .-tokenize
	.type	emit_ident, @function
emit_ident:
.LFB11:
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
	movq	$0, -8(%rbp)
.L257:
	cmpq	$2, -8(%rbp)
	je	.L252
	cmpq	$2, -8(%rbp)
	ja	.L259
	cmpq	$0, -8(%rbp)
	je	.L254
	cmpq	$1, -8(%rbp)
	jne	.L259
	jmp	.L258
.L254:
	movq	$2, -8(%rbp)
	jmp	.L256
.L252:
	movq	-24(%rbp), %rax
	movl	$5, %esi
	movq	%rax, %rdi
	call	emit_byte
	movq	-24(%rbp), %rax
	movl	$2, %esi
	movq	%rax, %rdi
	call	emit_byte
	movq	-32(%rbp), %rax
	leaq	16(%rax), %rcx
	movq	-24(%rbp), %rax
	movl	$4, %edx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	emit
	movq	-32(%rbp), %rax
	movl	16(%rax), %edx
	movq	-32(%rbp), %rax
	movq	24(%rax), %rcx
	movq	-24(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	emit
	movq	$1, -8(%rbp)
	jmp	.L256
.L259:
	nop
.L256:
	jmp	.L257
.L258:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE11:
	.size	emit_ident, .-emit_ident
	.type	q_free, @function
q_free:
.LFB12:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	$7, -8(%rbp)
.L282:
	cmpq	$8, -8(%rbp)
	ja	.L283
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L263(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L263(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L263:
	.long	.L271-.L263
	.long	.L284-.L263
	.long	.L269-.L263
	.long	.L268-.L263
	.long	.L267-.L263
	.long	.L266-.L263
	.long	.L265-.L263
	.long	.L264-.L263
	.long	.L262-.L263
	.text
.L267:
	movq	-24(%rbp), %rax
	movq	16(%rax), %rax
	testq	%rax, %rax
	je	.L272
	movq	$5, -8(%rbp)
	jmp	.L274
.L272:
	movq	$0, -8(%rbp)
	jmp	.L274
.L262:
	movq	-24(%rbp), %rax
	movq	32(%rax), %rax
	testq	%rax, %rax
	je	.L275
	movq	$2, -8(%rbp)
	jmp	.L274
.L275:
	movq	$4, -8(%rbp)
	jmp	.L274
.L268:
	movq	-24(%rbp), %rax
	movq	24(%rax), %rax
	testq	%rax, %rax
	je	.L278
	movq	$6, -8(%rbp)
	jmp	.L274
.L278:
	movq	$8, -8(%rbp)
	jmp	.L274
.L265:
	movq	-24(%rbp), %rax
	movq	24(%rax), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$8, -8(%rbp)
	jmp	.L274
.L266:
	movq	-24(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$0, -8(%rbp)
	jmp	.L274
.L271:
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$1, -8(%rbp)
	jmp	.L274
.L264:
	cmpq	$0, -24(%rbp)
	je	.L280
	movq	$3, -8(%rbp)
	jmp	.L274
.L280:
	movq	$1, -8(%rbp)
	jmp	.L274
.L269:
	movq	-24(%rbp), %rax
	movq	32(%rax), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$4, -8(%rbp)
	jmp	.L274
.L283:
	nop
.L274:
	jmp	.L282
.L284:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE12:
	.size	q_free, .-q_free
	.section	.rodata
.LC8:
	.string	"internal parser error"
.LC9:
	.string	"%s: %d: %s [%d]"
.LC10:
	.string	"use ';', not ','"
.LC11:
	.string	"%s: %d: %s"
.LC12:
	.string	"emit_expr"
.LC13:
	.string	"node->right != NULL"
.LC14:
	.string	"invalid assignment"
	.text
	.type	emit_expr, @function
emit_expr:
.LFB13:
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
	movq	$16, -16(%rbp)
.L363:
	cmpq	$54, -16(%rbp)
	ja	.L364
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L288(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L288(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L288:
	.long	.L364-.L288
	.long	.L323-.L288
	.long	.L322-.L288
	.long	.L364-.L288
	.long	.L364-.L288
	.long	.L321-.L288
	.long	.L320-.L288
	.long	.L364-.L288
	.long	.L364-.L288
	.long	.L319-.L288
	.long	.L318-.L288
	.long	.L317-.L288
	.long	.L364-.L288
	.long	.L316-.L288
	.long	.L315-.L288
	.long	.L364-.L288
	.long	.L314-.L288
	.long	.L313-.L288
	.long	.L312-.L288
	.long	.L311-.L288
	.long	.L310-.L288
	.long	.L364-.L288
	.long	.L364-.L288
	.long	.L364-.L288
	.long	.L309-.L288
	.long	.L365-.L288
	.long	.L307-.L288
	.long	.L306-.L288
	.long	.L364-.L288
	.long	.L305-.L288
	.long	.L304-.L288
	.long	.L303-.L288
	.long	.L364-.L288
	.long	.L364-.L288
	.long	.L302-.L288
	.long	.L364-.L288
	.long	.L364-.L288
	.long	.L364-.L288
	.long	.L301-.L288
	.long	.L300-.L288
	.long	.L364-.L288
	.long	.L299-.L288
	.long	.L298-.L288
	.long	.L297-.L288
	.long	.L296-.L288
	.long	.L295-.L288
	.long	.L294-.L288
	.long	.L293-.L288
	.long	.L364-.L288
	.long	.L364-.L288
	.long	.L292-.L288
	.long	.L291-.L288
	.long	.L290-.L288
	.long	.L289-.L288
	.long	.L287-.L288
	.text
.L312:
	movq	-48(%rbp), %rax
	movq	48(%rax), %rax
	testq	%rax, %rax
	je	.L324
	movq	$2, -16(%rbp)
	jmp	.L326
.L324:
	movq	$47, -16(%rbp)
	jmp	.L326
.L292:
	movq	-40(%rbp), %rax
	movl	$4, %esi
	movq	%rax, %rdi
	call	emit_byte
	movq	$25, -16(%rbp)
	jmp	.L326
.L290:
	movq	-40(%rbp), %rax
	movl	$4, %esi
	movq	%rax, %rdi
	call	emit_byte
	movq	$25, -16(%rbp)
	jmp	.L326
.L304:
	movq	-48(%rbp), %rax
	movq	48(%rax), %rcx
	movq	-40(%rbp), %rax
	movl	$0, %edx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	emit_hash_definition
	movq	-40(%rbp), %rax
	movl	$7, %esi
	movq	%rax, %rdi
	call	emit_byte
	movq	$25, -16(%rbp)
	jmp	.L326
.L315:
	movq	-48(%rbp), %rax
	movq	56(%rax), %rax
	testq	%rax, %rax
	je	.L328
	movq	$11, -16(%rbp)
	jmp	.L326
.L328:
	movq	$52, -16(%rbp)
	jmp	.L326
.L303:
	movq	-48(%rbp), %rax
	movl	32(%rax), %esi
	movq	-48(%rbp), %rax
	movl	36(%rax), %ecx
	movq	serr(%rip), %rdx
	movq	-40(%rbp), %rax
	movl	%esi, %r9d
	leaq	.LC8(%rip), %r8
	leaq	.LC9(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	die
	movq	$25, -16(%rbp)
	jmp	.L326
.L295:
	movq	-40(%rbp), %rax
	movl	$10, %esi
	movq	%rax, %rdi
	call	emit_byte
	movq	$51, -16(%rbp)
	jmp	.L326
.L287:
	movq	-48(%rbp), %rax
	movq	40(%rax), %rax
	movl	32(%rax), %eax
	cmpl	$97, %eax
	jne	.L330
	movq	$39, -16(%rbp)
	jmp	.L326
.L330:
	movq	$42, -16(%rbp)
	jmp	.L326
.L323:
	movq	-40(%rbp), %rax
	movl	$3, %esi
	movq	%rax, %rdi
	call	emit_byte
	movq	$25, -16(%rbp)
	jmp	.L326
.L314:
	movq	-48(%rbp), %rax
	movl	32(%rax), %eax
	subl	$37, %eax
	cmpl	$70, %eax
	ja	.L332
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L334(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L334(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L334:
	.long	.L340-.L334
	.long	.L332-.L334
	.long	.L332-.L334
	.long	.L343-.L334
	.long	.L332-.L334
	.long	.L340-.L334
	.long	.L340-.L334
	.long	.L342-.L334
	.long	.L340-.L334
	.long	.L341-.L334
	.long	.L340-.L334
	.long	.L332-.L334
	.long	.L332-.L334
	.long	.L332-.L334
	.long	.L332-.L334
	.long	.L332-.L334
	.long	.L332-.L334
	.long	.L332-.L334
	.long	.L332-.L334
	.long	.L332-.L334
	.long	.L332-.L334
	.long	.L332-.L334
	.long	.L332-.L334
	.long	.L332-.L334
	.long	.L339-.L334
	.long	.L332-.L334
	.long	.L332-.L334
	.long	.L332-.L334
	.long	.L332-.L334
	.long	.L332-.L334
	.long	.L332-.L334
	.long	.L332-.L334
	.long	.L332-.L334
	.long	.L332-.L334
	.long	.L332-.L334
	.long	.L332-.L334
	.long	.L332-.L334
	.long	.L332-.L334
	.long	.L332-.L334
	.long	.L332-.L334
	.long	.L332-.L334
	.long	.L332-.L334
	.long	.L332-.L334
	.long	.L332-.L334
	.long	.L332-.L334
	.long	.L332-.L334
	.long	.L332-.L334
	.long	.L332-.L334
	.long	.L332-.L334
	.long	.L332-.L334
	.long	.L332-.L334
	.long	.L332-.L334
	.long	.L332-.L334
	.long	.L332-.L334
	.long	.L338-.L334
	.long	.L332-.L334
	.long	.L332-.L334
	.long	.L332-.L334
	.long	.L332-.L334
	.long	.L332-.L334
	.long	.L337-.L334
	.long	.L336-.L334
	.long	.L335-.L334
	.long	.L332-.L334
	.long	.L332-.L334
	.long	.L332-.L334
	.long	.L332-.L334
	.long	.L332-.L334
	.long	.L332-.L334
	.long	.L332-.L334
	.long	.L333-.L334
	.text
.L338:
	movq	$29, -16(%rbp)
	jmp	.L344
.L336:
	movq	$53, -16(%rbp)
	jmp	.L344
.L335:
	movq	$5, -16(%rbp)
	jmp	.L344
.L337:
	movq	$41, -16(%rbp)
	jmp	.L344
.L333:
	movq	$19, -16(%rbp)
	jmp	.L344
.L343:
	movq	$9, -16(%rbp)
	jmp	.L344
.L342:
	movq	$24, -16(%rbp)
	jmp	.L344
.L339:
	movq	$13, -16(%rbp)
	jmp	.L344
.L340:
	movq	$34, -16(%rbp)
	jmp	.L344
.L341:
	movq	$54, -16(%rbp)
	jmp	.L344
.L332:
	movq	$31, -16(%rbp)
	nop
.L344:
	jmp	.L326
.L309:
	movq	-48(%rbp), %rax
	movq	40(%rax), %rdx
	movq	-40(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	emit_expr
	movq	-48(%rbp), %rax
	movq	48(%rax), %rdx
	movq	-40(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	emit_expr
	movq	-48(%rbp), %rax
	movl	36(%rax), %ecx
	movq	serr(%rip), %rdx
	movq	-40(%rbp), %rax
	leaq	.LC10(%rip), %r8
	leaq	.LC11(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	die
	movq	$25, -16(%rbp)
	jmp	.L326
.L307:
	movq	-48(%rbp), %rax
	movq	56(%rax), %rax
	movl	32(%rax), %eax
	cmpl	$46, %eax
	je	.L345
	movq	$45, -16(%rbp)
	jmp	.L326
.L345:
	movq	$51, -16(%rbp)
	jmp	.L326
.L317:
	movq	-48(%rbp), %rax
	movq	56(%rax), %rax
	movl	32(%rax), %eax
	cmpl	$46, %eax
	jne	.L347
	movq	$20, -16(%rbp)
	jmp	.L326
.L347:
	movq	$50, -16(%rbp)
	jmp	.L326
.L319:
	movq	-48(%rbp), %rdx
	movq	-40(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	emit_function_call
	movq	$25, -16(%rbp)
	jmp	.L326
.L316:
	movq	-48(%rbp), %rax
	movq	48(%rax), %rdx
	movq	-40(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	emit_expr
	movq	-48(%rbp), %rax
	movq	40(%rax), %rdx
	movq	-40(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	emit_expr
	movq	$46, -16(%rbp)
	jmp	.L326
.L291:
	movq	-48(%rbp), %rdx
	movq	-40(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	emit_ident
	movq	-48(%rbp), %rax
	movq	%rax, %rdi
	call	is_lvalue
	movl	%eax, -24(%rbp)
	movq	$17, -16(%rbp)
	jmp	.L326
.L311:
	movq	-40(%rbp), %rax
	movl	$5, %esi
	movq	%rax, %rdi
	call	emit_byte
	movq	-40(%rbp), %rax
	movl	$0, %esi
	movq	%rax, %rdi
	call	emit_byte
	movq	$25, -16(%rbp)
	jmp	.L326
.L313:
	cmpl	$0, -24(%rbp)
	je	.L349
	movq	$25, -16(%rbp)
	jmp	.L326
.L349:
	movq	$14, -16(%rbp)
	jmp	.L326
.L320:
	movq	-48(%rbp), %rax
	movq	40(%rax), %rax
	movl	32(%rax), %eax
	cmpl	$46, %eax
	je	.L351
	movq	$10, -16(%rbp)
	jmp	.L326
.L351:
	movq	$1, -16(%rbp)
	jmp	.L326
.L306:
	cmpl	$0, -20(%rbp)
	je	.L353
	movq	$25, -16(%rbp)
	jmp	.L326
.L353:
	movq	$38, -16(%rbp)
	jmp	.L326
.L301:
	movq	-40(%rbp), %rax
	movl	$4, %esi
	movq	%rax, %rdi
	call	emit_byte
	movq	$25, -16(%rbp)
	jmp	.L326
.L302:
	movq	-48(%rbp), %rax
	movq	40(%rax), %rdx
	movq	-40(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	emit_expr
	movq	-48(%rbp), %rax
	movq	48(%rax), %rdx
	movq	-40(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	emit_expr
	movq	-40(%rbp), %rax
	movl	$6, %esi
	movq	%rax, %rdi
	call	emit_byte
	movq	-48(%rbp), %rax
	movl	32(%rax), %eax
	movzbl	%al, %edx
	movq	-40(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	emit_byte
	movq	$25, -16(%rbp)
	jmp	.L326
.L289:
	movq	-48(%rbp), %rax
	movq	24(%rax), %rax
	movl	$0, %esi
	movq	%rax, %rdi
	call	strtod@PLT
	movq	%xmm0, %rax
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rdx
	movq	-40(%rbp), %rax
	movq	%rdx, %xmm0
	movq	%rax, %rdi
	call	emit_num
	movq	$25, -16(%rbp)
	jmp	.L326
.L293:
	leaq	.LC12(%rip), %rax
	movq	%rax, %rcx
	movl	$497, %edx
	leaq	.LC3(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L296:
	movq	-48(%rbp), %rax
	movl	36(%rax), %ecx
	movq	serr(%rip), %rdx
	movq	-40(%rbp), %rax
	leaq	.LC14(%rip), %r8
	leaq	.LC11(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	die
	movq	$1, -16(%rbp)
	jmp	.L326
.L321:
	movq	-48(%rbp), %rdx
	movq	-40(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	emit_string_constant
	movq	$25, -16(%rbp)
	jmp	.L326
.L299:
	movq	-48(%rbp), %rax
	movq	56(%rax), %rax
	testq	%rax, %rax
	jne	.L355
	movq	$43, -16(%rbp)
	jmp	.L326
.L355:
	movq	$26, -16(%rbp)
	jmp	.L326
.L318:
	movq	-48(%rbp), %rax
	movq	40(%rax), %rax
	movl	32(%rax), %eax
	cmpl	$61, %eax
	je	.L357
	movq	$44, -16(%rbp)
	jmp	.L326
.L357:
	movq	$1, -16(%rbp)
	jmp	.L326
.L298:
	movq	-48(%rbp), %rax
	movq	40(%rax), %rdx
	movq	-40(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	emit_expr
	movq	-48(%rbp), %rax
	movq	48(%rax), %rdx
	movq	-40(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	emit_expr
	movq	-40(%rbp), %rax
	movl	$4, %esi
	movq	%rax, %rdi
	call	emit_byte
	movq	$25, -16(%rbp)
	jmp	.L326
.L294:
	movq	-48(%rbp), %rax
	movq	40(%rax), %rax
	movl	32(%rax), %eax
	cmpl	$97, %eax
	je	.L359
	movq	$6, -16(%rbp)
	jmp	.L326
.L359:
	movq	$1, -16(%rbp)
	jmp	.L326
.L300:
	movq	-40(%rbp), %rax
	movl	$10, %esi
	movq	%rax, %rdi
	call	emit_byte
	movq	$42, -16(%rbp)
	jmp	.L326
.L305:
	movq	-40(%rbp), %rax
	movl	$11, %esi
	movq	%rax, %rdi
	call	emit_byte
	movq	$18, -16(%rbp)
	jmp	.L326
.L297:
	movq	-40(%rbp), %rax
	movl	$10, %esi
	movq	%rax, %rdi
	call	emit_byte
	movq	$51, -16(%rbp)
	jmp	.L326
.L322:
	movq	-48(%rbp), %rax
	movq	48(%rax), %rax
	movl	32(%rax), %eax
	cmpl	$107, %eax
	je	.L361
	movq	$30, -16(%rbp)
	jmp	.L326
.L361:
	movq	$25, -16(%rbp)
	jmp	.L326
.L310:
	movq	-48(%rbp), %rax
	movq	56(%rax), %rax
	movq	%rax, %rdi
	call	is_lvalue
	movl	%eax, -20(%rbp)
	movq	$27, -16(%rbp)
	jmp	.L326
.L364:
	nop
.L326:
	jmp	.L363
.L365:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE13:
	.size	emit_expr, .-emit_expr
	.section	.rodata
	.align 8
.LC15:
	.string	"%s: cannot grow code to %u bytes"
	.text
	.type	expand_code, @function
expand_code:
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
	movl	%esi, -28(%rbp)
	movq	$3, -16(%rbp)
.L378:
	cmpq	$4, -16(%rbp)
	ja	.L379
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L369(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L369(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L369:
	.long	.L373-.L369
	.long	.L372-.L369
	.long	.L371-.L369
	.long	.L370-.L369
	.long	.L380-.L369
	.text
.L372:
	movq	-24(%rbp), %rax
	movq	24(%rax), %rax
	testq	%rax, %rax
	jne	.L375
	movq	$2, -16(%rbp)
	jmp	.L377
.L375:
	movq	$4, -16(%rbp)
	jmp	.L377
.L370:
	movq	$0, -16(%rbp)
	jmp	.L377
.L373:
	movq	-24(%rbp), %rax
	movl	52(%rax), %edx
	movl	-28(%rbp), %eax
	addl	%eax, %edx
	movq	-24(%rbp), %rax
	movl	%edx, 52(%rax)
	movq	-24(%rbp), %rax
	movl	52(%rax), %eax
	movslq	%eax, %rdx
	movq	-24(%rbp), %rax
	movq	24(%rax), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -8(%rbp)
	movq	-24(%rbp), %rax
	movq	-8(%rbp), %rdx
	movq	%rdx, 24(%rax)
	movq	$1, -16(%rbp)
	jmp	.L377
.L371:
	movq	-24(%rbp), %rax
	movl	52(%rax), %ecx
	movq	serr(%rip), %rdx
	movq	-24(%rbp), %rax
	leaq	.LC15(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	die
	movq	$4, -16(%rbp)
	jmp	.L377
.L379:
	nop
.L377:
	jmp	.L378
.L380:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE14:
	.size	expand_code, .-expand_code
	.type	die, @function
die:
.LFB15:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$240, %rsp
	movq	%rdi, -232(%rbp)
	movq	%rsi, -240(%rbp)
	movq	%rdx, -160(%rbp)
	movq	%rcx, -152(%rbp)
	movq	%r8, -144(%rbp)
	movq	%r9, -136(%rbp)
	testb	%al, %al
	je	.L382
	movaps	%xmm0, -128(%rbp)
	movaps	%xmm1, -112(%rbp)
	movaps	%xmm2, -96(%rbp)
	movaps	%xmm3, -80(%rbp)
	movaps	%xmm4, -64(%rbp)
	movaps	%xmm5, -48(%rbp)
	movaps	%xmm6, -32(%rbp)
	movaps	%xmm7, -16(%rbp)
.L382:
	movq	%fs:40, %rax
	movq	%rax, -184(%rbp)
	xorl	%eax, %eax
	movq	$1, -216(%rbp)
.L387:
	cmpq	$1, -216(%rbp)
	je	.L383
	cmpq	$2, -216(%rbp)
	je	.L384
	jmp	.L386
.L383:
	movq	$2, -216(%rbp)
	jmp	.L386
.L384:
	movl	$16, -208(%rbp)
	movl	$48, -204(%rbp)
	leaq	16(%rbp), %rax
	movq	%rax, -200(%rbp)
	leaq	-176(%rbp), %rax
	movq	%rax, -192(%rbp)
	movq	-232(%rbp), %rax
	movq	264(%rax), %rax
	leaq	-208(%rbp), %rcx
	movq	-240(%rbp), %rdx
	movl	$256, %esi
	movq	%rax, %rdi
	call	vsnprintf@PLT
	movq	-232(%rbp), %rax
	addq	$64, %rax
	movl	$1, %esi
	movq	%rax, %rdi
	call	longjmp@PLT
.L386:
	jmp	.L387
	.cfi_endproc
.LFE15:
	.size	die, .-die
	.section	.rodata
.LC16:
	.string	"%v"
.LC17:
	.string	"%d"
.LC18:
	.string	"%s"
.LC19:
	.string	"function"
.LC20:
	.string	"if"
.LC21:
	.string	"else"
.LC22:
	.string	"+="
.LC23:
	.string	"-="
.LC24:
	.string	"*="
.LC25:
	.string	"/="
.LC26:
	.string	"nil"
.LC27:
	.string	"not"
.LC28:
	.string	"and"
.LC29:
	.string	"or"
.LC30:
	.string	"=[](){}+-*/^,;.\""
.LC31:
	.string	"runtime error"
.LC32:
	.string	"syntax error"
.LC33:
	.string	"xEND"
.LC34:
	.string	"xCALL"
.LC35:
	.string	"xRETURN"
.LC36:
	.string	"xSET"
.LC37:
	.string	"xGET"
.LC38:
	.string	"tPUSH"
.LC39:
	.string	"aMATH"
.LC40:
	.string	"xPOP"
.LC41:
	.string	"xJUMP"
.LC42:
	.string	"nDUP"
.LC43:
	.string	"xPUSHNS"
.LC44:
	.string	"xNEWMAP"
.LC45:
	.string	"num"
.LC46:
	.string	"str"
.LC47:
	.string	"hash"
.LC48:
	.string	"func"
.LC49:
	.string	"c-func"
.LC50:
	.string	"test.txt"
	.text
	.globl	main
	.type	main, @function
main:
.LFB17:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$352, %rsp
	movl	%edi, -324(%rbp)
	movq	%rsi, -336(%rbp)
	movq	%rdx, -344(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, nil(%rip)
	movq	$0, 8+nil(%rip)
	movl	$0, 16+nil(%rip)
	movq	$0, 24+nil(%rip)
	movl	$0, 32+nil(%rip)
	movl	$0, 36+nil(%rip)
	movq	$0, 40+nil(%rip)
	movq	$0, 48+nil(%rip)
	movq	$0, 56+nil(%rip)
	nop
.L390:
	movl	$0, -308(%rbp)
	jmp	.L391
.L392:
	movl	-308(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	op_tab(%rip), %rax
	movl	$0, (%rdx,%rax)
	addl	$1, -308(%rbp)
.L391:
	cmpl	$255, -308(%rbp)
	jle	.L392
	nop
.L393:
	movl	$-1, hex_tab(%rip)
	movl	$-1, 4+hex_tab(%rip)
	movl	$-1, 8+hex_tab(%rip)
	movl	$-1, 12+hex_tab(%rip)
	movl	$-1, 16+hex_tab(%rip)
	movl	$-1, 20+hex_tab(%rip)
	movl	$-1, 24+hex_tab(%rip)
	movl	$-1, 28+hex_tab(%rip)
	movl	$-1, 32+hex_tab(%rip)
	movl	$-1, 36+hex_tab(%rip)
	movl	$-1, 40+hex_tab(%rip)
	movl	$-1, 44+hex_tab(%rip)
	movl	$-1, 48+hex_tab(%rip)
	movl	$-1, 52+hex_tab(%rip)
	movl	$-1, 56+hex_tab(%rip)
	movl	$-1, 60+hex_tab(%rip)
	movl	$-1, 64+hex_tab(%rip)
	movl	$-1, 68+hex_tab(%rip)
	movl	$-1, 72+hex_tab(%rip)
	movl	$-1, 76+hex_tab(%rip)
	movl	$-1, 80+hex_tab(%rip)
	movl	$-1, 84+hex_tab(%rip)
	movl	$-1, 88+hex_tab(%rip)
	movl	$-1, 92+hex_tab(%rip)
	movl	$-1, 96+hex_tab(%rip)
	movl	$-1, 100+hex_tab(%rip)
	movl	$-1, 104+hex_tab(%rip)
	movl	$-1, 108+hex_tab(%rip)
	movl	$-1, 112+hex_tab(%rip)
	movl	$-1, 116+hex_tab(%rip)
	movl	$-1, 120+hex_tab(%rip)
	movl	$-1, 124+hex_tab(%rip)
	movl	$-1, 128+hex_tab(%rip)
	movl	$-1, 132+hex_tab(%rip)
	movl	$-1, 136+hex_tab(%rip)
	movl	$-1, 140+hex_tab(%rip)
	movl	$-1, 144+hex_tab(%rip)
	movl	$-1, 148+hex_tab(%rip)
	movl	$-1, 152+hex_tab(%rip)
	movl	$-1, 156+hex_tab(%rip)
	movl	$-1, 160+hex_tab(%rip)
	movl	$-1, 164+hex_tab(%rip)
	movl	$-1, 168+hex_tab(%rip)
	movl	$-1, 172+hex_tab(%rip)
	movl	$-1, 176+hex_tab(%rip)
	movl	$-1, 180+hex_tab(%rip)
	movl	$-1, 184+hex_tab(%rip)
	movl	$-1, 188+hex_tab(%rip)
	movl	$0, 192+hex_tab(%rip)
	movl	$1, 196+hex_tab(%rip)
	movl	$2, 200+hex_tab(%rip)
	movl	$3, 204+hex_tab(%rip)
	movl	$4, 208+hex_tab(%rip)
	movl	$5, 212+hex_tab(%rip)
	movl	$6, 216+hex_tab(%rip)
	movl	$7, 220+hex_tab(%rip)
	movl	$8, 224+hex_tab(%rip)
	movl	$9, 228+hex_tab(%rip)
	movl	$-1, 232+hex_tab(%rip)
	movl	$-1, 236+hex_tab(%rip)
	movl	$-1, 240+hex_tab(%rip)
	movl	$-1, 244+hex_tab(%rip)
	movl	$-1, 248+hex_tab(%rip)
	movl	$-1, 252+hex_tab(%rip)
	movl	$-1, 256+hex_tab(%rip)
	movl	$10, 260+hex_tab(%rip)
	movl	$11, 264+hex_tab(%rip)
	movl	$12, 268+hex_tab(%rip)
	movl	$13, 272+hex_tab(%rip)
	movl	$14, 276+hex_tab(%rip)
	movl	$15, 280+hex_tab(%rip)
	movl	$-1, 284+hex_tab(%rip)
	movl	$-1, 288+hex_tab(%rip)
	movl	$-1, 292+hex_tab(%rip)
	movl	$-1, 296+hex_tab(%rip)
	movl	$-1, 300+hex_tab(%rip)
	movl	$-1, 304+hex_tab(%rip)
	movl	$-1, 308+hex_tab(%rip)
	movl	$-1, 312+hex_tab(%rip)
	movl	$-1, 316+hex_tab(%rip)
	movl	$-1, 320+hex_tab(%rip)
	movl	$-1, 324+hex_tab(%rip)
	movl	$-1, 328+hex_tab(%rip)
	movl	$-1, 332+hex_tab(%rip)
	movl	$-1, 336+hex_tab(%rip)
	movl	$-1, 340+hex_tab(%rip)
	movl	$-1, 344+hex_tab(%rip)
	movl	$-1, 348+hex_tab(%rip)
	movl	$-1, 352+hex_tab(%rip)
	movl	$-1, 356+hex_tab(%rip)
	movl	$-1, 360+hex_tab(%rip)
	movl	$-1, 364+hex_tab(%rip)
	movl	$-1, 368+hex_tab(%rip)
	movl	$-1, 372+hex_tab(%rip)
	movl	$-1, 376+hex_tab(%rip)
	movl	$-1, 380+hex_tab(%rip)
	movl	$-1, 384+hex_tab(%rip)
	movl	$10, 388+hex_tab(%rip)
	movl	$11, 392+hex_tab(%rip)
	movl	$12, 396+hex_tab(%rip)
	movl	$13, 400+hex_tab(%rip)
	movl	$14, 404+hex_tab(%rip)
	movl	$15, 408+hex_tab(%rip)
	movl	$-1, 412+hex_tab(%rip)
	movl	$-1, 416+hex_tab(%rip)
	movl	$-1, 420+hex_tab(%rip)
	movl	$-1, 424+hex_tab(%rip)
	movl	$-1, 428+hex_tab(%rip)
	movl	$-1, 432+hex_tab(%rip)
	movl	$-1, 436+hex_tab(%rip)
	movl	$-1, 440+hex_tab(%rip)
	movl	$-1, 444+hex_tab(%rip)
	nop
.L394:
	leaq	loaded_modules(%rip), %rax
	movq	%rax, loaded_modules(%rip)
	leaq	loaded_modules(%rip), %rax
	movq	%rax, 8+loaded_modules(%rip)
	nop
.L395:
	leaq	.LC16(%rip), %rax
	movq	%rax, long_tokens(%rip)
	leaq	.LC17(%rip), %rax
	movq	%rax, 8+long_tokens(%rip)
	leaq	.LC18(%rip), %rax
	movq	%rax, 16+long_tokens(%rip)
	leaq	.LC19(%rip), %rax
	movq	%rax, 24+long_tokens(%rip)
	leaq	.LC20(%rip), %rax
	movq	%rax, 32+long_tokens(%rip)
	leaq	.LC21(%rip), %rax
	movq	%rax, 40+long_tokens(%rip)
	leaq	.LC22(%rip), %rax
	movq	%rax, 48+long_tokens(%rip)
	leaq	.LC23(%rip), %rax
	movq	%rax, 56+long_tokens(%rip)
	leaq	.LC24(%rip), %rax
	movq	%rax, 64+long_tokens(%rip)
	leaq	.LC25(%rip), %rax
	movq	%rax, 72+long_tokens(%rip)
	leaq	.LC26(%rip), %rax
	movq	%rax, 80+long_tokens(%rip)
	leaq	.LC27(%rip), %rax
	movq	%rax, 88+long_tokens(%rip)
	leaq	.LC28(%rip), %rax
	movq	%rax, 96+long_tokens(%rip)
	leaq	.LC29(%rip), %rax
	movq	%rax, 104+long_tokens(%rip)
	movq	$0, 112+long_tokens(%rip)
	nop
.L396:
	leaq	.LC30(%rip), %rax
	movq	%rax, one_char_tokens(%rip)
	nop
.L397:
	leaq	.LC31(%rip), %rax
	movq	%rax, rerr(%rip)
	nop
.L398:
	leaq	.LC32(%rip), %rax
	movq	%rax, serr(%rip)
	nop
.L399:
	leaq	.LC33(%rip), %rax
	movq	%rax, opcodes(%rip)
	leaq	.LC34(%rip), %rax
	movq	%rax, 8+opcodes(%rip)
	leaq	.LC35(%rip), %rax
	movq	%rax, 16+opcodes(%rip)
	leaq	.LC36(%rip), %rax
	movq	%rax, 24+opcodes(%rip)
	leaq	.LC37(%rip), %rax
	movq	%rax, 32+opcodes(%rip)
	leaq	.LC38(%rip), %rax
	movq	%rax, 40+opcodes(%rip)
	leaq	.LC39(%rip), %rax
	movq	%rax, 48+opcodes(%rip)
	leaq	.LC40(%rip), %rax
	movq	%rax, 56+opcodes(%rip)
	leaq	.LC41(%rip), %rax
	movq	%rax, 64+opcodes(%rip)
	leaq	.LC42(%rip), %rax
	movq	%rax, 72+opcodes(%rip)
	leaq	.LC43(%rip), %rax
	movq	%rax, 80+opcodes(%rip)
	leaq	.LC44(%rip), %rax
	movq	%rax, 88+opcodes(%rip)
	nop
.L400:
	leaq	.LC26(%rip), %rax
	movq	%rax, types(%rip)
	leaq	.LC45(%rip), %rax
	movq	%rax, 8+types(%rip)
	leaq	.LC46(%rip), %rax
	movq	%rax, 16+types(%rip)
	leaq	.LC47(%rip), %rax
	movq	%rax, 24+types(%rip)
	leaq	.LC48(%rip), %rax
	movq	%rax, 32+types(%rip)
	leaq	.LC49(%rip), %rax
	movq	%rax, 40+types(%rip)
	nop
.L401:
	movq	$0, _TIG_IZ_ld3M_envp(%rip)
	nop
.L402:
	movq	$0, _TIG_IZ_ld3M_argv(%rip)
	nop
.L403:
	movl	$0, _TIG_IZ_ld3M_argc(%rip)
	nop
	nop
.L404:
.L405:
#APP
# 510 "TheProjecter_qvm_q.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-ld3M--0
# 0 "" 2
#NO_APP
	movl	-324(%rbp), %eax
	movl	%eax, _TIG_IZ_ld3M_argc(%rip)
	movq	-336(%rbp), %rax
	movq	%rax, _TIG_IZ_ld3M_argv(%rip)
	movq	-344(%rbp), %rax
	movq	%rax, _TIG_IZ_ld3M_envp(%rip)
	nop
	movq	$15, -280(%rbp)
.L441:
	cmpq	$21, -280(%rbp)
	ja	.L444
	movq	-280(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L408(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L408(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L408:
	.long	.L424-.L408
	.long	.L423-.L408
	.long	.L422-.L408
	.long	.L444-.L408
	.long	.L421-.L408
	.long	.L444-.L408
	.long	.L420-.L408
	.long	.L444-.L408
	.long	.L419-.L408
	.long	.L444-.L408
	.long	.L418-.L408
	.long	.L417-.L408
	.long	.L416-.L408
	.long	.L415-.L408
	.long	.L414-.L408
	.long	.L413-.L408
	.long	.L444-.L408
	.long	.L412-.L408
	.long	.L411-.L408
	.long	.L410-.L408
	.long	.L409-.L408
	.long	.L407-.L408
	.text
.L411:
	movl	-324(%rbp), %eax
	cmpl	$1, %eax
	jle	.L425
	movq	$10, -280(%rbp)
	jmp	.L427
.L425:
	movq	$13, -280(%rbp)
	jmp	.L427
.L421:
	leaq	-272(%rbp), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$20, -280(%rbp)
	jmp	.L427
.L414:
	leaq	-272(%rbp), %rdx
	movq	-288(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	q_load
	movq	%rax, -296(%rbp)
	movq	$0, -280(%rbp)
	jmp	.L427
.L413:
	movq	$0, -304(%rbp)
	movq	$18, -280(%rbp)
	jmp	.L427
.L416:
	movq	-336(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, -288(%rbp)
	movq	$14, -280(%rbp)
	jmp	.L427
.L419:
	leaq	-336(%rbp), %rdx
	leaq	-324(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	get_opt
	movq	%rax, -304(%rbp)
	movq	$18, -280(%rbp)
	jmp	.L427
.L423:
	movq	-336(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	addq	$1, %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	je	.L428
	movq	$19, -280(%rbp)
	jmp	.L427
.L428:
	movq	$13, -280(%rbp)
	jmp	.L427
.L407:
	leaq	-272(%rbp), %rdx
	movq	-296(%rbp), %rcx
	movq	-304(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	q_save
	movq	$20, -280(%rbp)
	jmp	.L427
.L417:
	movl	-324(%rbp), %eax
	cmpl	$1, %eax
	jle	.L430
	movq	$12, -280(%rbp)
	jmp	.L427
.L430:
	movq	$6, -280(%rbp)
	jmp	.L427
.L415:
	call	init_op_tab
	movq	$11, -280(%rbp)
	jmp	.L427
.L410:
	movq	-336(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	addq	$1, %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	cmpl	$99, %eax
	jne	.L432
	movq	$8, -280(%rbp)
	jmp	.L433
.L432:
	movq	$17, -280(%rbp)
	nop
.L433:
	jmp	.L427
.L412:
	call	usage
	movq	$18, -280(%rbp)
	jmp	.L427
.L420:
	leaq	.LC50(%rip), %rax
	movq	%rax, -288(%rbp)
	movq	$14, -280(%rbp)
	jmp	.L427
.L418:
	movq	-336(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	movzbl	(%rax), %eax
	cmpb	$45, %al
	jne	.L434
	movq	$1, -280(%rbp)
	jmp	.L427
.L434:
	movq	$13, -280(%rbp)
	jmp	.L427
.L424:
	cmpq	$0, -296(%rbp)
	jne	.L436
	movq	$4, -280(%rbp)
	jmp	.L427
.L436:
	movq	$2, -280(%rbp)
	jmp	.L427
.L422:
	cmpq	$0, -304(%rbp)
	je	.L438
	movq	$21, -280(%rbp)
	jmp	.L427
.L438:
	movq	$20, -280(%rbp)
	jmp	.L427
.L409:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L442
	jmp	.L443
.L444:
	nop
.L427:
	jmp	.L441
.L443:
	call	__stack_chk_fail@PLT
.L442:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE17:
	.size	main, .-main
	.type	emit, @function
emit:
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
	movq	%rsi, -32(%rbp)
	movl	%edx, -36(%rbp)
	movq	$2, -8(%rbp)
.L460:
	cmpq	$6, -8(%rbp)
	ja	.L461
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L448(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L448(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L448:
	.long	.L453-.L448
	.long	.L452-.L448
	.long	.L451-.L448
	.long	.L450-.L448
	.long	.L449-.L448
	.long	.L461-.L448
	.long	.L462-.L448
	.text
.L449:
	movl	-36(%rbp), %eax
	movslq	%eax, %rdx
	movq	-24(%rbp), %rax
	movq	24(%rax), %rcx
	movq	-24(%rbp), %rax
	movl	48(%rax), %eax
	cltq
	addq	%rax, %rcx
	movq	-32(%rbp), %rax
	movq	%rax, %rsi
	movq	%rcx, %rdi
	call	memcpy@PLT
	movq	-24(%rbp), %rax
	movl	48(%rax), %edx
	movl	-36(%rbp), %eax
	addl	%eax, %edx
	movq	-24(%rbp), %rax
	movl	%edx, 48(%rax)
	movq	$6, -8(%rbp)
	jmp	.L454
.L452:
	movq	-24(%rbp), %rax
	movl	$64, %esi
	movq	%rax, %rdi
	call	expand_code
	movq	$4, -8(%rbp)
	jmp	.L454
.L450:
	movq	-24(%rbp), %rax
	movl	$64, %esi
	movq	%rax, %rdi
	call	expand_code
	movq	$4, -8(%rbp)
	jmp	.L454
.L453:
	movq	-24(%rbp), %rax
	movl	48(%rax), %edx
	movl	-36(%rbp), %eax
	addl	%eax, %edx
	movq	-24(%rbp), %rax
	movl	52(%rax), %eax
	cmpl	%eax, %edx
	jl	.L456
	movq	$3, -8(%rbp)
	jmp	.L454
.L456:
	movq	$4, -8(%rbp)
	jmp	.L454
.L451:
	movq	-24(%rbp), %rax
	movq	24(%rax), %rax
	testq	%rax, %rax
	jne	.L458
	movq	$1, -8(%rbp)
	jmp	.L454
.L458:
	movq	$0, -8(%rbp)
	jmp	.L454
.L461:
	nop
.L454:
	jmp	.L460
.L462:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE18:
	.size	emit, .-emit
	.type	prio, @function
prio:
.LFB19:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -20(%rbp)
	movq	$0, -8(%rbp)
.L466:
	cmpq	$0, -8(%rbp)
	jne	.L469
	movl	-20(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	op_tab(%rip), %rax
	movl	(%rdx,%rax), %eax
	movzbl	%al, %eax
	jmp	.L468
.L469:
	nop
	jmp	.L466
.L468:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE19:
	.size	prio, .-prio
	.type	is_rtl, @function
is_rtl:
.LFB21:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -20(%rbp)
	movq	$0, -8(%rbp)
.L473:
	cmpq	$0, -8(%rbp)
	jne	.L476
	movl	-20(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	op_tab(%rip), %rax
	movl	(%rdx,%rax), %eax
	sarl	$16, %eax
	movzbl	%al, %eax
	jmp	.L475
.L476:
	nop
	jmp	.L473
.L475:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE21:
	.size	is_rtl, .-is_rtl
	.section	.rodata
.LC51:
	.string	"line %d: unexpected EOF"
.LC52:
	.string	"bad expr"
.LC53:
	.string	"%s: line %d: %s"
	.text
	.type	expr, @function
expr:
.LFB22:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$4096, %rsp
	orq	$0, (%rsp)
	subq	$144, %rsp
	movq	%rdi, -4216(%rbp)
	movq	%rsi, -4224(%rbp)
	movl	%edx, -4228(%rbp)
	movq	%rcx, -4240(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$55, -4144(%rbp)
.L548:
	cmpq	$59, -4144(%rbp)
	ja	.L551
	movq	-4144(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L480(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L480(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L480:
	.long	.L517-.L480
	.long	.L551-.L480
	.long	.L551-.L480
	.long	.L516-.L480
	.long	.L515-.L480
	.long	.L514-.L480
	.long	.L513-.L480
	.long	.L512-.L480
	.long	.L551-.L480
	.long	.L511-.L480
	.long	.L551-.L480
	.long	.L510-.L480
	.long	.L509-.L480
	.long	.L508-.L480
	.long	.L507-.L480
	.long	.L506-.L480
	.long	.L551-.L480
	.long	.L505-.L480
	.long	.L504-.L480
	.long	.L503-.L480
	.long	.L502-.L480
	.long	.L551-.L480
	.long	.L501-.L480
	.long	.L500-.L480
	.long	.L499-.L480
	.long	.L551-.L480
	.long	.L498-.L480
	.long	.L497-.L480
	.long	.L496-.L480
	.long	.L551-.L480
	.long	.L495-.L480
	.long	.L551-.L480
	.long	.L551-.L480
	.long	.L551-.L480
	.long	.L551-.L480
	.long	.L551-.L480
	.long	.L551-.L480
	.long	.L494-.L480
	.long	.L493-.L480
	.long	.L551-.L480
	.long	.L492-.L480
	.long	.L551-.L480
	.long	.L551-.L480
	.long	.L491-.L480
	.long	.L490-.L480
	.long	.L551-.L480
	.long	.L489-.L480
	.long	.L488-.L480
	.long	.L487-.L480
	.long	.L551-.L480
	.long	.L486-.L480
	.long	.L485-.L480
	.long	.L484-.L480
	.long	.L551-.L480
	.long	.L551-.L480
	.long	.L483-.L480
	.long	.L482-.L480
	.long	.L481-.L480
	.long	.L551-.L480
	.long	.L479-.L480
	.text
.L504:
	movq	-4120(%rbp), %rax
	movq	-4152(%rbp), %rdx
	movq	%rdx, (%rax)
	movq	-4120(%rbp), %rdx
	movq	-4152(%rbp), %rax
	movq	%rdx, 8(%rax)
	movq	-4152(%rbp), %rax
	leaq	-4128(%rbp), %rdx
	movq	%rdx, (%rax)
	movq	-4152(%rbp), %rax
	movq	%rax, -4120(%rbp)
	movq	$43, -4144(%rbp)
	jmp	.L518
.L486:
	movl	-4208(%rbp), %eax
	cmpl	-4196(%rbp), %eax
	jge	.L519
	movq	$40, -4144(%rbp)
	jmp	.L518
.L519:
	movq	$47, -4144(%rbp)
	jmp	.L518
.L484:
	movq	-4152(%rbp), %rax
	movl	32(%rax), %eax
	cmpl	$40, %eax
	jne	.L521
	movq	$30, -4144(%rbp)
	jmp	.L518
.L521:
	movq	$13, -4144(%rbp)
	jmp	.L518
.L515:
	movl	-4204(%rbp), %eax
	cmpl	-4176(%rbp), %eax
	jge	.L523
	movq	$44, -4144(%rbp)
	jmp	.L518
.L523:
	movq	$51, -4144(%rbp)
	jmp	.L518
.L495:
	movq	-4152(%rbp), %rax
	movl	32(%rax), %eax
	cmpl	$40, %eax
	jne	.L525
	movq	$19, -4144(%rbp)
	jmp	.L518
.L525:
	movq	$56, -4144(%rbp)
	jmp	.L518
.L507:
	movl	-4200(%rbp), %eax
	cltq
	salq	$6, %rax
	movq	%rax, %rdx
	movq	-4224(%rbp), %rax
	addq	%rdx, %rax
	movl	32(%rax), %eax
	cmpl	%eax, -4228(%rbp)
	je	.L527
	movq	$26, -4144(%rbp)
	jmp	.L518
.L527:
	movq	$3, -4144(%rbp)
	jmp	.L518
.L506:
	movl	-4200(%rbp), %eax
	addl	$1, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L549
	jmp	.L550
.L482:
	movq	-4152(%rbp), %rax
	movl	32(%rax), %eax
	cmpl	$91, %eax
	jne	.L530
	movq	$5, -4144(%rbp)
	jmp	.L518
.L530:
	movq	$9, -4144(%rbp)
	jmp	.L518
.L509:
	movl	$107, 32+nil(%rip)
	leaq	-4128(%rbp), %rax
	movq	%rax, -4120(%rbp)
	movq	-4120(%rbp), %rax
	movq	%rax, -4128(%rbp)
	movl	$0, -4196(%rbp)
	movl	-4196(%rbp), %eax
	movl	%eax, -4200(%rbp)
	movq	$14, -4144(%rbp)
	jmp	.L518
.L500:
	cmpl	$0, -4200(%rbp)
	jne	.L532
	movq	$57, -4144(%rbp)
	jmp	.L518
.L532:
	movq	$6, -4144(%rbp)
	jmp	.L518
.L516:
	movl	-4196(%rbp), %edx
	leaq	-4112(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	bubble_sort_ops_by_priority
	movl	$0, -4208(%rbp)
	movq	$50, -4144(%rbp)
	jmp	.L518
.L499:
	movq	-4152(%rbp), %rax
	movl	32(%rax), %eax
	movl	%eax, %edi
	call	is_op
	movl	%eax, -4180(%rbp)
	movq	$17, -4144(%rbp)
	jmp	.L518
.L481:
	movq	-4240(%rbp), %rax
	leaq	nil(%rip), %rdx
	movq	%rdx, (%rax)
	movq	$15, -4144(%rbp)
	jmp	.L518
.L498:
	movl	-4200(%rbp), %eax
	movl	%eax, -4168(%rbp)
	addl	$1, -4200(%rbp)
	movl	-4168(%rbp), %eax
	cltq
	salq	$6, %rax
	movq	%rax, %rdx
	movq	-4224(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, -4152(%rbp)
	movq	$18, -4144(%rbp)
	jmp	.L518
.L510:
	movl	-4208(%rbp), %eax
	cltq
	movq	-4112(%rbp,%rax,8), %rax
	movl	32(%rax), %eax
	movl	%eax, %edi
	call	nargs
	movl	%eax, -4176(%rbp)
	movq	$4, -4144(%rbp)
	jmp	.L518
.L511:
	movq	-4152(%rbp), %rax
	movl	32(%rax), %eax
	cmpl	$40, %eax
	jne	.L534
	movq	$37, -4144(%rbp)
	jmp	.L518
.L534:
	movq	$38, -4144(%rbp)
	jmp	.L518
.L508:
	movq	-4152(%rbp), %rax
	movl	32(%rax), %eax
	cmpl	$91, %eax
	jne	.L536
	movq	$30, -4144(%rbp)
	jmp	.L518
.L536:
	movq	$24, -4144(%rbp)
	jmp	.L518
.L485:
	addl	$1, -4208(%rbp)
	movq	$50, -4144(%rbp)
	jmp	.L518
.L503:
	movl	-4200(%rbp), %edx
	movq	-4152(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	qaz
	movl	%eax, -4188(%rbp)
	movq	$7, -4144(%rbp)
	jmp	.L518
.L505:
	cmpl	$0, -4180(%rbp)
	je	.L538
	movq	$0, -4144(%rbp)
	jmp	.L518
.L538:
	movq	$14, -4144(%rbp)
	jmp	.L518
.L492:
	movl	$0, -4204(%rbp)
	movq	$11, -4144(%rbp)
	jmp	.L518
.L483:
	movq	$12, -4144(%rbp)
	jmp	.L518
.L479:
	movq	-4152(%rbp), %rax
	movq	(%rax), %rax
	movq	-4152(%rbp), %rdx
	movq	8(%rdx), %rdx
	movq	%rdx, 8(%rax)
	movq	-4152(%rbp), %rax
	movq	8(%rax), %rax
	movq	-4152(%rbp), %rdx
	movq	(%rdx), %rdx
	movq	%rdx, (%rax)
	movq	-4152(%rbp), %rax
	movq	%rax, -4136(%rbp)
	movq	-4152(%rbp), %rax
	movq	-4136(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movq	-4152(%rbp), %rax
	movq	-4136(%rbp), %rdx
	movq	%rdx, (%rax)
	movq	$9, -4144(%rbp)
	jmp	.L518
.L513:
	movq	-4128(%rbp), %rdx
	movq	-4240(%rbp), %rax
	movq	%rdx, (%rax)
	movq	$15, -4144(%rbp)
	jmp	.L518
.L497:
	movq	-4152(%rbp), %rax
	movl	36(%rax), %edx
	movq	-4216(%rbp), %rax
	leaq	.LC51(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	die
	movq	$24, -4144(%rbp)
	jmp	.L518
.L493:
	movl	$93, -4184(%rbp)
	movq	$46, -4144(%rbp)
	jmp	.L518
.L487:
	movq	-4152(%rbp), %rax
	movl	36(%rax), %ecx
	movq	serr(%rip), %rdx
	movq	-4216(%rbp), %rax
	leaq	.LC52(%rip), %r8
	leaq	.LC53(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	die
	movq	$23, -4144(%rbp)
	jmp	.L518
.L501:
	movq	-4160(%rbp), %rdx
	movq	-4120(%rbp), %rax
	movq	%rdx, (%rax)
	movq	-4160(%rbp), %rax
	movq	-4120(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movq	-4160(%rbp), %rax
	leaq	-4128(%rbp), %rdx
	movq	%rdx, (%rax)
	movq	-4160(%rbp), %rax
	movq	%rax, -4120(%rbp)
	movq	$24, -4144(%rbp)
	jmp	.L518
.L496:
	movq	-4152(%rbp), %rax
	movl	$46, 32(%rax)
	movq	$9, -4144(%rbp)
	jmp	.L518
.L488:
	movq	-4128(%rbp), %rax
	movq	%rax, %rdx
	movq	-4120(%rbp), %rax
	cmpq	%rax, %rdx
	je	.L540
	movq	$48, -4144(%rbp)
	jmp	.L518
.L540:
	movq	$23, -4144(%rbp)
	jmp	.L518
.L490:
	movl	-4208(%rbp), %eax
	cltq
	movq	-4112(%rbp,%rax,8), %rsi
	leaq	-4128(%rbp), %rcx
	movl	-4204(%rbp), %edx
	movq	-4216(%rbp), %rax
	movq	%rax, %rdi
	call	reduce
	addl	$1, -4204(%rbp)
	movq	$11, -4144(%rbp)
	jmp	.L518
.L514:
	movl	-4200(%rbp), %edx
	movq	-4152(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	qaz
	movl	%eax, -4192(%rbp)
	movq	$20, -4144(%rbp)
	jmp	.L518
.L494:
	movl	$41, -4184(%rbp)
	movq	$46, -4144(%rbp)
	jmp	.L518
.L517:
	movl	-4196(%rbp), %eax
	movl	%eax, -4164(%rbp)
	addl	$1, -4196(%rbp)
	movl	-4164(%rbp), %eax
	cltq
	movq	-4152(%rbp), %rdx
	movq	%rdx, -4112(%rbp,%rax,8)
	movq	$14, -4144(%rbp)
	jmp	.L518
.L489:
	movq	-4152(%rbp), %rax
	leaq	64(%rax), %rsi
	leaq	-4160(%rbp), %rcx
	movl	-4184(%rbp), %edx
	movq	-4216(%rbp), %rax
	movq	%rax, %rdi
	call	expr
	movl	%eax, -4172(%rbp)
	movl	-4172(%rbp), %eax
	addl	%eax, -4200(%rbp)
	movq	$22, -4144(%rbp)
	jmp	.L518
.L512:
	cmpl	$0, -4188(%rbp)
	je	.L542
	movq	$56, -4144(%rbp)
	jmp	.L518
.L542:
	movq	$59, -4144(%rbp)
	jmp	.L518
.L491:
	movq	-4152(%rbp), %rax
	movl	32(%rax), %eax
	cmpl	$111, %eax
	jne	.L544
	movq	$27, -4144(%rbp)
	jmp	.L518
.L544:
	movq	$52, -4144(%rbp)
	jmp	.L518
.L502:
	cmpl	$0, -4192(%rbp)
	je	.L546
	movq	$28, -4144(%rbp)
	jmp	.L518
.L546:
	movq	$9, -4144(%rbp)
	jmp	.L518
.L551:
	nop
.L518:
	jmp	.L548
.L550:
	call	__stack_chk_fail@PLT
.L549:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE22:
	.size	expr, .-expr
	.type	qaz, @function
qaz:
.LFB23:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movl	%esi, -28(%rbp)
	movq	$9, -8(%rbp)
.L576:
	cmpq	$10, -8(%rbp)
	ja	.L578
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L555(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L555(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L555:
	.long	.L565-.L555
	.long	.L564-.L555
	.long	.L563-.L555
	.long	.L562-.L555
	.long	.L561-.L555
	.long	.L560-.L555
	.long	.L559-.L555
	.long	.L558-.L555
	.long	.L557-.L555
	.long	.L556-.L555
	.long	.L554-.L555
	.text
.L561:
	cmpl	$41, -16(%rbp)
	jne	.L566
	movq	$3, -8(%rbp)
	jmp	.L568
.L566:
	movq	$6, -8(%rbp)
	jmp	.L568
.L557:
	movl	-12(%rbp), %eax
	jmp	.L577
.L564:
	movl	$1, -12(%rbp)
	movq	$8, -8(%rbp)
	jmp	.L568
.L562:
	movl	$1, -12(%rbp)
	movq	$8, -8(%rbp)
	jmp	.L568
.L556:
	movq	-24(%rbp), %rax
	subq	$64, %rax
	movl	32(%rax), %eax
	movl	%eax, -16(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L568
.L559:
	movl	$0, -12(%rbp)
	movq	$8, -8(%rbp)
	jmp	.L568
.L560:
	movl	$0, -12(%rbp)
	movq	$8, -8(%rbp)
	jmp	.L568
.L554:
	movl	$1, -12(%rbp)
	movq	$8, -8(%rbp)
	jmp	.L568
.L565:
	cmpl	$1, -28(%rbp)
	jle	.L570
	movq	$7, -8(%rbp)
	jmp	.L568
.L570:
	movq	$5, -8(%rbp)
	jmp	.L568
.L558:
	cmpl	$97, -16(%rbp)
	jne	.L572
	movq	$10, -8(%rbp)
	jmp	.L568
.L572:
	movq	$2, -8(%rbp)
	jmp	.L568
.L563:
	cmpl	$93, -16(%rbp)
	jne	.L574
	movq	$1, -8(%rbp)
	jmp	.L568
.L574:
	movq	$4, -8(%rbp)
	jmp	.L568
.L578:
	nop
.L568:
	jmp	.L576
.L577:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE23:
	.size	qaz, .-qaz
	.section	.rodata
.LC54:
	.string	"cmp"
.LC55:
	.string	"a->type == b->type"
	.text
	.type	cmp, @function
cmp:
.LFB24:
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
	movq	$0, -8(%rbp)
.L613:
	cmpq	$20, -8(%rbp)
	ja	.L615
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L582(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L582(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L582:
	.long	.L595-.L582
	.long	.L615-.L582
	.long	.L615-.L582
	.long	.L615-.L582
	.long	.L594-.L582
	.long	.L593-.L582
	.long	.L615-.L582
	.long	.L592-.L582
	.long	.L591-.L582
	.long	.L590-.L582
	.long	.L589-.L582
	.long	.L615-.L582
	.long	.L588-.L582
	.long	.L615-.L582
	.long	.L587-.L582
	.long	.L586-.L582
	.long	.L585-.L582
	.long	.L584-.L582
	.long	.L583-.L582
	.long	.L615-.L582
	.long	.L581-.L582
	.text
.L583:
	movl	$1, -12(%rbp)
	movq	$16, -8(%rbp)
	jmp	.L596
.L594:
	movq	-24(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, %rdx
	movq	-32(%rbp), %rax
	movq	8(%rax), %rax
	cmpq	%rax, %rdx
	jne	.L597
	movq	$10, -8(%rbp)
	jmp	.L596
.L597:
	movq	$5, -8(%rbp)
	jmp	.L596
.L587:
	movq	-24(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, %rdx
	movq	-32(%rbp), %rax
	movq	8(%rax), %rax
	cmpq	%rax, %rdx
	jne	.L599
	movq	$9, -8(%rbp)
	jmp	.L596
.L599:
	movq	$18, -8(%rbp)
	jmp	.L596
.L586:
	leaq	.LC54(%rip), %rax
	movq	%rax, %rcx
	movl	$529, %edx
	leaq	.LC3(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC55(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L588:
	movq	-24(%rbp), %rax
	movzbl	(%rax), %eax
	movzbl	%al, %eax
	cmpl	$5, %eax
	ja	.L601
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L603(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L603(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L603:
	.long	.L608-.L603
	.long	.L607-.L603
	.long	.L606-.L603
	.long	.L605-.L603
	.long	.L604-.L603
	.long	.L602-.L603
	.text
.L605:
	movq	$4, -8(%rbp)
	jmp	.L609
.L604:
	movq	$14, -8(%rbp)
	jmp	.L609
.L602:
	movq	$16, -8(%rbp)
	jmp	.L609
.L606:
	movq	$17, -8(%rbp)
	jmp	.L609
.L607:
	movq	$20, -8(%rbp)
	jmp	.L609
.L608:
	movq	$16, -8(%rbp)
	jmp	.L609
.L601:
	movq	$7, -8(%rbp)
	nop
.L609:
	jmp	.L596
.L591:
	movq	-24(%rbp), %rax
	movzbl	(%rax), %edx
	movq	-32(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	%al, %dl
	jne	.L610
	movq	$12, -8(%rbp)
	jmp	.L596
.L610:
	movq	$15, -8(%rbp)
	jmp	.L596
.L585:
	movl	-12(%rbp), %eax
	jmp	.L614
.L590:
	movl	$0, -12(%rbp)
	movq	$16, -8(%rbp)
	jmp	.L596
.L584:
	movq	-32(%rbp), %rax
	leaq	8(%rax), %rdx
	movq	-24(%rbp), %rax
	addq	$8, %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	cmp_vec
	movl	%eax, -12(%rbp)
	movq	$16, -8(%rbp)
	jmp	.L596
.L593:
	movl	$1, -12(%rbp)
	movq	$16, -8(%rbp)
	jmp	.L596
.L589:
	movl	$0, -12(%rbp)
	movq	$16, -8(%rbp)
	jmp	.L596
.L595:
	movl	$0, -12(%rbp)
	movq	$8, -8(%rbp)
	jmp	.L596
.L592:
	call	abort@PLT
.L581:
	movq	-24(%rbp), %rax
	movsd	8(%rax), %xmm0
	movq	-32(%rbp), %rax
	movsd	8(%rax), %xmm1
	subsd	%xmm1, %xmm0
	cvttsd2sil	%xmm0, %eax
	movl	%eax, -12(%rbp)
	movq	$16, -8(%rbp)
	jmp	.L596
.L615:
	nop
.L596:
	jmp	.L613
.L614:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE24:
	.size	cmp, .-cmp
	.section	.rodata
.LC56:
	.string	"stack underflow"
	.text
	.type	pop, @function
pop:
.LFB25:
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
.L625:
	cmpq	$3, -8(%rbp)
	je	.L617
	cmpq	$3, -8(%rbp)
	ja	.L627
	cmpq	$2, -8(%rbp)
	je	.L619
	cmpq	$2, -8(%rbp)
	ja	.L627
	cmpq	$0, -8(%rbp)
	je	.L620
	cmpq	$1, -8(%rbp)
	jne	.L627
	movq	-24(%rbp), %rax
	movl	40(%rax), %eax
	leal	-1(%rax), %edx
	movq	-24(%rbp), %rax
	movl	%edx, 40(%rax)
	movq	$2, -8(%rbp)
	jmp	.L621
.L617:
	movq	-24(%rbp), %rax
	movq	32(%rax), %rcx
	movq	-24(%rbp), %rax
	movl	40(%rax), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	jmp	.L626
.L620:
	movq	-24(%rbp), %rax
	movl	48(%rax), %ecx
	movq	rerr(%rip), %rdx
	movq	-24(%rbp), %rax
	leaq	.LC56(%rip), %r8
	leaq	.LC11(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	die
	movq	$3, -8(%rbp)
	jmp	.L621
.L619:
	movq	-24(%rbp), %rax
	movl	40(%rax), %eax
	testl	%eax, %eax
	jns	.L623
	movq	$0, -8(%rbp)
	jmp	.L621
.L623:
	movq	$3, -8(%rbp)
	jmp	.L621
.L627:
	nop
.L621:
	jmp	.L625
.L626:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE25:
	.size	pop, .-pop
	.type	calc_hash, @function
calc_hash:
.LFB26:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	$2, -8(%rbp)
.L652:
	cmpq	$14, -8(%rbp)
	ja	.L654
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L631(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L631(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L631:
	.long	.L654-.L631
	.long	.L638-.L631
	.long	.L637-.L631
	.long	.L636-.L631
	.long	.L654-.L631
	.long	.L654-.L631
	.long	.L654-.L631
	.long	.L635-.L631
	.long	.L634-.L631
	.long	.L654-.L631
	.long	.L633-.L631
	.long	.L654-.L631
	.long	.L654-.L631
	.long	.L632-.L631
	.long	.L630-.L631
	.text
.L630:
	movq	-24(%rbp), %rax
	movsd	8(%rax), %xmm0
	comisd	.LC57(%rip), %xmm0
	jnb	.L639
	cvttsd2siq	%xmm0, %rax
	jmp	.L640
.L639:
	movsd	.LC57(%rip), %xmm1
	subsd	%xmm1, %xmm0
	cvttsd2siq	%xmm0, %rax
	movabsq	$-9223372036854775808, %rdx
	xorq	%rdx, %rax
.L640:
	addq	%rax, -16(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L641
.L634:
	movq	-24(%rbp), %rax
	movq	8(%rax), %rax
	addq	%rax, -16(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L641
.L638:
	movq	-24(%rbp), %rax
	movzbl	(%rax), %eax
	movzbl	%al, %eax
	cmpl	$5, %eax
	ja	.L642
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L644(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L644(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L644:
	.long	.L649-.L644
	.long	.L648-.L644
	.long	.L647-.L644
	.long	.L646-.L644
	.long	.L645-.L644
	.long	.L643-.L644
	.text
.L646:
	movq	$7, -8(%rbp)
	jmp	.L650
.L645:
	movq	$8, -8(%rbp)
	jmp	.L650
.L643:
	movq	$3, -8(%rbp)
	jmp	.L650
.L647:
	movq	$10, -8(%rbp)
	jmp	.L650
.L648:
	movq	$14, -8(%rbp)
	jmp	.L650
.L649:
	movq	$7, -8(%rbp)
	jmp	.L650
.L642:
	movq	$13, -8(%rbp)
	nop
.L650:
	jmp	.L641
.L636:
	movq	-24(%rbp), %rax
	movl	8(%rax), %eax
	cltq
	addq	%rax, -16(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L641
.L632:
	call	abort@PLT
.L633:
	movq	-24(%rbp), %rax
	leaq	8(%rax), %rdx
	movq	-16(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	hash_str
	movq	%rax, -16(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L641
.L635:
	movq	-16(%rbp), %rax
	jmp	.L653
.L637:
	movq	-24(%rbp), %rax
	movzbl	(%rax), %eax
	movzbl	%al, %eax
	movq	%rax, -16(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L641
.L654:
	nop
.L641:
	jmp	.L652
.L653:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE26:
	.size	calc_hash, .-calc_hash
	.type	calc_number_of_params, @function
calc_number_of_params:
.LFB27:
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
	movq	$0, -8(%rbp)
.L673:
	cmpq	$7, -8(%rbp)
	ja	.L675
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L658(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L658(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L658:
	.long	.L664-.L658
	.long	.L663-.L658
	.long	.L662-.L658
	.long	.L661-.L658
	.long	.L660-.L658
	.long	.L659-.L658
	.long	.L675-.L658
	.long	.L657-.L658
	.text
.L660:
	movq	-24(%rbp), %rax
	movq	48(%rax), %rax
	movl	$0, %esi
	movq	%rax, %rdi
	call	calc_number_of_params
	movl	%eax, -16(%rbp)
	movl	-16(%rbp), %eax
	addl	%eax, -28(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L665
.L663:
	movq	-24(%rbp), %rax
	movq	48(%rax), %rax
	testq	%rax, %rax
	je	.L666
	movq	$4, -8(%rbp)
	jmp	.L665
.L666:
	movq	$5, -8(%rbp)
	jmp	.L665
.L661:
	movq	-24(%rbp), %rax
	movq	40(%rax), %rax
	movl	$0, %esi
	movq	%rax, %rdi
	call	calc_number_of_params
	movl	%eax, -12(%rbp)
	movl	-12(%rbp), %eax
	addl	%eax, -28(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L665
.L659:
	movq	-24(%rbp), %rax
	movl	32(%rax), %eax
	cmpl	$44, %eax
	jne	.L668
	movq	$7, -8(%rbp)
	jmp	.L665
.L668:
	movq	$2, -8(%rbp)
	jmp	.L665
.L664:
	movq	-24(%rbp), %rax
	movq	40(%rax), %rax
	testq	%rax, %rax
	je	.L670
	movq	$3, -8(%rbp)
	jmp	.L665
.L670:
	movq	$1, -8(%rbp)
	jmp	.L665
.L657:
	addl	$1, -28(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L665
.L662:
	movl	-28(%rbp), %eax
	jmp	.L674
.L675:
	nop
.L665:
	jmp	.L673
.L674:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE27:
	.size	calc_number_of_params, .-calc_number_of_params
	.section	.rodata
.LC58:
	.string	" <%s> "
.LC59:
	.string	" %d"
.LC60:
	.string	"disasm"
	.align 8
.LC61:
	.string	"NUM_INSTTRUCTIONS == ARRAY_SIZE(opcodes)"
.LC62:
	.string	"%.2f"
.LC63:
	.string	"\"%.*s\""
.LC64:
	.string	"%-4d %s"
.LC65:
	.string	" %c"
.LC66:
	.string	"*c < NUM_INSTTRUCTIONS"
	.text
	.type	disasm, @function
disasm:
.LFB28:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movq	$12, -8(%rbp)
.L714:
	cmpq	$29, -8(%rbp)
	ja	.L716
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L679(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L679(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L679:
	.long	.L715-.L679
	.long	.L716-.L679
	.long	.L716-.L679
	.long	.L716-.L679
	.long	.L695-.L679
	.long	.L716-.L679
	.long	.L716-.L679
	.long	.L716-.L679
	.long	.L694-.L679
	.long	.L716-.L679
	.long	.L716-.L679
	.long	.L716-.L679
	.long	.L693-.L679
	.long	.L692-.L679
	.long	.L717-.L679
	.long	.L690-.L679
	.long	.L716-.L679
	.long	.L689-.L679
	.long	.L688-.L679
	.long	.L687-.L679
	.long	.L686-.L679
	.long	.L685-.L679
	.long	.L684-.L679
	.long	.L683-.L679
	.long	.L682-.L679
	.long	.L716-.L679
	.long	.L681-.L679
	.long	.L716-.L679
	.long	.L680-.L679
	.long	.L678-.L679
	.text
.L688:
	addq	$1, -16(%rbp)
	movq	-16(%rbp), %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	cltq
	leaq	0(,%rax,8), %rdx
	leaq	types(%rip), %rax
	movq	(%rdx,%rax), %rax
	movq	%rax, %rsi
	leaq	.LC58(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$8, -8(%rbp)
	jmp	.L697
.L695:
	addq	$1, -16(%rbp)
	movq	-16(%rbp), %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	movl	%eax, %esi
	leaq	.LC59(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$29, -8(%rbp)
	jmp	.L697
.L690:
	movq	-16(%rbp), %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	je	.L699
	movq	$0, -8(%rbp)
	jmp	.L697
.L699:
	movq	$14, -8(%rbp)
	jmp	.L697
.L693:
	movq	-40(%rbp), %rax
	movq	24(%rax), %rax
	movq	%rax, -16(%rbp)
	movq	$15, -8(%rbp)
	jmp	.L697
.L694:
	movq	-16(%rbp), %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	cmpl	$2, %eax
	je	.L701
	cmpl	$2, %eax
	jg	.L702
	testl	%eax, %eax
	je	.L703
	cmpl	$1, %eax
	je	.L704
	jmp	.L702
.L701:
	movq	$13, -8(%rbp)
	jmp	.L705
.L704:
	movq	$26, -8(%rbp)
	jmp	.L705
.L703:
	movq	$29, -8(%rbp)
	jmp	.L705
.L702:
	movq	$29, -8(%rbp)
	nop
.L705:
	jmp	.L697
.L683:
	leaq	.LC60(%rip), %rax
	movq	%rax, %rcx
	movl	$857, %edx
	leaq	.LC3(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC61(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L682:
	movq	-24(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$110, %al
	jne	.L706
	movq	$4, -8(%rbp)
	jmp	.L697
.L706:
	movq	$19, -8(%rbp)
	jmp	.L697
.L685:
	movq	-24(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$97, %al
	jne	.L708
	movq	$22, -8(%rbp)
	jmp	.L697
.L708:
	movq	$24, -8(%rbp)
	jmp	.L697
.L681:
	movq	-16(%rbp), %rax
	addq	$1, %rax
	movq	(%rax), %rax
	movq	%rax, %xmm0
	leaq	.LC62(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	addq	$8, -16(%rbp)
	movq	$29, -8(%rbp)
	jmp	.L697
.L692:
	movq	-16(%rbp), %rax
	addq	$1, %rax
	movq	(%rax), %rax
	movl	%eax, -28(%rbp)
	addq	$8, -16(%rbp)
	movq	-16(%rbp), %rax
	leaq	1(%rax), %rdx
	movl	-28(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC63(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-28(%rbp), %eax
	cltq
	addq	%rax, -16(%rbp)
	movq	$29, -8(%rbp)
	jmp	.L697
.L687:
	movq	-24(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$116, %al
	jne	.L710
	movq	$18, -8(%rbp)
	jmp	.L697
.L710:
	movq	$29, -8(%rbp)
	jmp	.L697
.L689:
	movq	-16(%rbp), %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	cltq
	leaq	0(,%rax,8), %rdx
	leaq	opcodes(%rip), %rax
	movq	(%rdx,%rax), %rax
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	leaq	1(%rax), %rdx
	movq	-40(%rbp), %rax
	movq	24(%rax), %rcx
	movq	-16(%rbp), %rax
	subq	%rcx, %rax
	movq	%rax, %rsi
	leaq	.LC64(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$21, -8(%rbp)
	jmp	.L697
.L684:
	addq	$1, -16(%rbp)
	movq	-16(%rbp), %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	movl	%eax, %esi
	leaq	.LC65(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$29, -8(%rbp)
	jmp	.L697
.L680:
	leaq	.LC60(%rip), %rax
	movq	%rax, %rcx
	movl	$858, %edx
	leaq	.LC3(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC66(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L715:
	movq	$20, -8(%rbp)
	jmp	.L697
.L678:
	movl	$10, %edi
	call	putchar@PLT
	addq	$1, -16(%rbp)
	movq	$15, -8(%rbp)
	jmp	.L697
.L686:
	movq	-16(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$11, %al
	jg	.L712
	movq	$17, -8(%rbp)
	jmp	.L697
.L712:
	movq	$28, -8(%rbp)
	jmp	.L697
.L716:
	nop
.L697:
	jmp	.L714
.L717:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE28:
	.size	disasm, .-disasm
	.section	.rodata
.LC67:
	.string	"cannot allocate memory for q"
	.text
	.type	q_load, @function
q_load:
.LFB29:
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
	movq	$8, -16(%rbp)
.L746:
	cmpq	$18, -16(%rbp)
	ja	.L747
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L721(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L721(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L721:
	.long	.L734-.L721
	.long	.L733-.L721
	.long	.L732-.L721
	.long	.L747-.L721
	.long	.L731-.L721
	.long	.L747-.L721
	.long	.L730-.L721
	.long	.L729-.L721
	.long	.L728-.L721
	.long	.L747-.L721
	.long	.L727-.L721
	.long	.L726-.L721
	.long	.L725-.L721
	.long	.L724-.L721
	.long	.L723-.L721
	.long	.L722-.L721
	.long	.L747-.L721
	.long	.L747-.L721
	.long	.L720-.L721
	.text
.L720:
	cmpl	$0, -40(%rbp)
	jne	.L735
	movq	$4, -16(%rbp)
	jmp	.L737
.L735:
	movq	$15, -16(%rbp)
	jmp	.L737
.L731:
	movq	-24(%rbp), %rax
	jmp	.L738
.L723:
	movq	-64(%rbp), %rax
	leaq	.LC67(%rip), %rdx
	movq	%rdx, %rcx
	leaq	.LC18(%rip), %rdx
	movl	$256, %esi
	movq	%rax, %rdi
	movl	$0, %eax
	call	snprintf@PLT
	movq	$7, -16(%rbp)
	jmp	.L737
.L722:
	movq	-32(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, -32(%rbp)
	movq	$11, -16(%rbp)
	jmp	.L737
.L725:
	movl	$272, %esi
	movl	$1, %edi
	call	calloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	$2, -16(%rbp)
	jmp	.L737
.L728:
	movq	loaded_modules(%rip), %rax
	movq	%rax, -32(%rbp)
	movq	$11, -16(%rbp)
	jmp	.L737
.L733:
	movq	-24(%rbp), %rax
	addq	$64, %rax
	movq	%rax, %rdi
	call	_setjmp@PLT
	endbr64
	movl	%eax, -36(%rbp)
	movq	$0, -16(%rbp)
	jmp	.L737
.L726:
	movq	-32(%rbp), %rax
	leaq	loaded_modules(%rip), %rdx
	cmpq	%rdx, %rax
	je	.L740
	movq	$13, -16(%rbp)
	jmp	.L737
.L740:
	movq	$12, -16(%rbp)
	jmp	.L737
.L724:
	movq	-32(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	16(%rax), %rax
	movq	-56(%rbp), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -40(%rbp)
	movq	$18, -16(%rbp)
	jmp	.L737
.L730:
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	q_free
	movq	$0, -24(%rbp)
	movq	$7, -16(%rbp)
	jmp	.L737
.L727:
	movq	-24(%rbp), %rax
	movq	-64(%rbp), %rdx
	movq	%rdx, 264(%rax)
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	strdup@PLT
	movq	%rax, %rdx
	movq	-24(%rbp), %rax
	movq	%rdx, 16(%rax)
	movq	-56(%rbp), %rdx
	movq	-24(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	compile
	movq	$7, -16(%rbp)
	jmp	.L737
.L734:
	cmpl	$0, -36(%rbp)
	je	.L742
	movq	$6, -16(%rbp)
	jmp	.L737
.L742:
	movq	$10, -16(%rbp)
	jmp	.L737
.L729:
	movq	-24(%rbp), %rax
	jmp	.L738
.L732:
	cmpq	$0, -24(%rbp)
	jne	.L744
	movq	$14, -16(%rbp)
	jmp	.L737
.L744:
	movq	$1, -16(%rbp)
	jmp	.L737
.L747:
	nop
.L737:
	jmp	.L746
.L738:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE29:
	.size	q_load, .-q_load
	.section	.rodata
.LC68:
	.string	"cannot grow stack to %u"
.LC69:
	.string	"%s: %d: %s %u"
	.text
	.type	push, @function
push:
.LFB30:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movq	$8, -16(%rbp)
.L767:
	cmpq	$10, -16(%rbp)
	ja	.L769
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L751(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L751(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L751:
	.long	.L758-.L751
	.long	.L769-.L751
	.long	.L757-.L751
	.long	.L756-.L751
	.long	.L755-.L751
	.long	.L769-.L751
	.long	.L754-.L751
	.long	.L769-.L751
	.long	.L753-.L751
	.long	.L752-.L751
	.long	.L750-.L751
	.text
.L755:
	movq	-40(%rbp), %rax
	movl	56(%rax), %eax
	leal	64(%rax), %edx
	movq	-40(%rbp), %rax
	movl	%edx, 56(%rax)
	movq	-40(%rbp), %rax
	movl	56(%rax), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-40(%rbp), %rax
	movq	32(%rax), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -8(%rbp)
	movq	-40(%rbp), %rax
	movq	-8(%rbp), %rdx
	movq	%rdx, 32(%rax)
	movq	$10, -16(%rbp)
	jmp	.L759
.L753:
	movq	-40(%rbp), %rax
	movq	32(%rax), %rax
	testq	%rax, %rax
	jne	.L760
	movq	$6, -16(%rbp)
	jmp	.L759
.L760:
	movq	$9, -16(%rbp)
	jmp	.L759
.L756:
	movq	-40(%rbp), %rax
	movq	32(%rax), %rcx
	movl	-20(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	jmp	.L768
.L752:
	movq	-40(%rbp), %rax
	movl	40(%rax), %edx
	movq	-40(%rbp), %rax
	movl	56(%rax), %eax
	cmpl	%eax, %edx
	jl	.L763
	movq	$4, -16(%rbp)
	jmp	.L759
.L763:
	movq	$10, -16(%rbp)
	jmp	.L759
.L754:
	movq	-40(%rbp), %rax
	movl	56(%rax), %eax
	leal	64(%rax), %edx
	movq	-40(%rbp), %rax
	movl	%edx, 56(%rax)
	movq	-40(%rbp), %rax
	movl	56(%rax), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdx
	movq	-40(%rbp), %rax
	movq	32(%rax), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -8(%rbp)
	movq	-40(%rbp), %rax
	movq	-8(%rbp), %rdx
	movq	%rdx, 32(%rax)
	movq	$10, -16(%rbp)
	jmp	.L759
.L750:
	movq	-40(%rbp), %rax
	movq	32(%rax), %rax
	testq	%rax, %rax
	jne	.L765
	movq	$2, -16(%rbp)
	jmp	.L759
.L765:
	movq	$0, -16(%rbp)
	jmp	.L759
.L758:
	movq	-40(%rbp), %rax
	movl	40(%rax), %eax
	movl	%eax, -20(%rbp)
	movq	-40(%rbp), %rax
	movl	40(%rax), %eax
	leal	1(%rax), %edx
	movq	-40(%rbp), %rax
	movl	%edx, 40(%rax)
	movq	$3, -16(%rbp)
	jmp	.L759
.L757:
	movq	-40(%rbp), %rax
	movl	56(%rax), %esi
	movq	-40(%rbp), %rax
	movl	48(%rax), %ecx
	movq	rerr(%rip), %rdx
	movq	-40(%rbp), %rax
	movl	%esi, %r9d
	leaq	.LC68(%rip), %r8
	leaq	.LC69(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	die
	movq	$0, -16(%rbp)
	jmp	.L759
.L769:
	nop
.L759:
	jmp	.L767
.L768:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE30:
	.size	push, .-push
	.type	nargs, @function
nargs:
.LFB31:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -20(%rbp)
	movq	$0, -8(%rbp)
.L773:
	cmpq	$0, -8(%rbp)
	jne	.L776
	movl	-20(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	op_tab(%rip), %rax
	movl	(%rdx,%rax), %eax
	sarl	$8, %eax
	movzbl	%al, %eax
	jmp	.L775
.L776:
	nop
	jmp	.L773
.L775:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE31:
	.size	nargs, .-nargs
	.type	get_opt, @function
get_opt:
.LFB32:
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
	movq	$0, -8(%rbp)
.L792:
	cmpq	$7, -8(%rbp)
	ja	.L794
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L780(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L780(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L780:
	.long	.L785-.L780
	.long	.L784-.L780
	.long	.L794-.L780
	.long	.L783-.L780
	.long	.L782-.L780
	.long	.L794-.L780
	.long	.L781-.L780
	.long	.L779-.L780
	.text
.L782:
	call	usage
	movq	$6, -8(%rbp)
	jmp	.L786
.L784:
	cmpq	$0, -16(%rbp)
	jne	.L787
	movq	$4, -8(%rbp)
	jmp	.L786
.L787:
	movq	$6, -8(%rbp)
	jmp	.L786
.L783:
	movq	-32(%rbp), %rax
	movq	(%rax), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	addq	$2, %rax
	movq	%rax, -16(%rbp)
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	leal	-1(%rax), %edx
	movq	-24(%rbp), %rax
	movl	%edx, (%rax)
	movq	-32(%rbp), %rax
	movq	(%rax), %rax
	leaq	8(%rax), %rdx
	movq	-32(%rbp), %rax
	movq	%rdx, (%rax)
	movq	$1, -8(%rbp)
	jmp	.L786
.L781:
	movq	-16(%rbp), %rax
	jmp	.L793
.L785:
	movq	-32(%rbp), %rax
	movq	(%rax), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	addq	$2, %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	je	.L790
	movq	$3, -8(%rbp)
	jmp	.L786
.L790:
	movq	$7, -8(%rbp)
	jmp	.L786
.L779:
	movq	-32(%rbp), %rax
	movq	(%rax), %rax
	movq	16(%rax), %rax
	movq	%rax, -16(%rbp)
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	leal	-2(%rax), %edx
	movq	-24(%rbp), %rax
	movl	%edx, (%rax)
	movq	-32(%rbp), %rax
	movq	(%rax), %rax
	leaq	16(%rax), %rdx
	movq	-32(%rbp), %rax
	movq	%rdx, (%rax)
	movq	$1, -8(%rbp)
	jmp	.L786
.L794:
	nop
.L786:
	jmp	.L792
.L793:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE32:
	.size	get_opt, .-get_opt
	.section	.rodata
.LC70:
	.string	"lookup"
.LC71:
	.string	"obj->type == TYPE_HASH"
	.text
	.type	lookup, @function
lookup:
.LFB33:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$96, %rsp
	movq	%rdi, -72(%rbp)
	movq	%rsi, -80(%rbp)
	movq	%rdx, -88(%rbp)
	movq	$11, -24(%rbp)
.L824:
	cmpq	$16, -24(%rbp)
	ja	.L825
	movq	-24(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L798(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L798(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L798:
	.long	.L825-.L798
	.long	.L811-.L798
	.long	.L810-.L798
	.long	.L809-.L798
	.long	.L808-.L798
	.long	.L825-.L798
	.long	.L807-.L798
	.long	.L806-.L798
	.long	.L805-.L798
	.long	.L804-.L798
	.long	.L825-.L798
	.long	.L803-.L798
	.long	.L802-.L798
	.long	.L801-.L798
	.long	.L800-.L798
	.long	.L799-.L798
	.long	.L797-.L798
	.text
.L808:
	cmpl	$0, -60(%rbp)
	jne	.L812
	movq	$13, -24(%rbp)
	jmp	.L814
.L812:
	movq	$15, -24(%rbp)
	jmp	.L814
.L800:
	movl	$0, %eax
	jmp	.L815
.L799:
	movq	-56(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, -56(%rbp)
	movq	$12, -24(%rbp)
	jmp	.L814
.L802:
	movq	-56(%rbp), %rdx
	movq	-48(%rbp), %rax
	cmpq	%rax, %rdx
	je	.L816
	movq	$1, -24(%rbp)
	jmp	.L814
.L816:
	movq	$14, -24(%rbp)
	jmp	.L814
.L805:
	movq	-88(%rbp), %rax
	movq	-32(%rbp), %rdx
	movq	%rdx, (%rax)
	movq	$2, -24(%rbp)
	jmp	.L814
.L811:
	movq	-56(%rbp), %rax
	movq	%rax, -40(%rbp)
	movq	$16, -24(%rbp)
	jmp	.L814
.L809:
	movq	-72(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, -16(%rbp)
	movq	-80(%rbp), %rax
	movq	%rax, %rdi
	call	calc_hash
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rcx
	movabsq	$5675921253449092805, %rdx
	movq	%rcx, %rax
	mulq	%rdx
	shrq	$2, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	subq	%rax, %rcx
	movq	%rcx, %rdx
	movq	%rdx, -32(%rbp)
	movq	-16(%rbp), %rax
	leaq	24(%rax), %rdx
	movq	-32(%rbp), %rax
	salq	$4, %rax
	addq	%rdx, %rax
	movq	%rax, -48(%rbp)
	movq	$7, -24(%rbp)
	jmp	.L814
.L797:
	movq	-40(%rbp), %rax
	movzbl	16(%rax), %edx
	movq	-80(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	%al, %dl
	jne	.L818
	movq	$6, -24(%rbp)
	jmp	.L814
.L818:
	movq	$15, -24(%rbp)
	jmp	.L814
.L803:
	movq	-72(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$3, %al
	jne	.L820
	movq	$3, -24(%rbp)
	jmp	.L814
.L820:
	movq	$9, -24(%rbp)
	jmp	.L814
.L804:
	leaq	.LC70(%rip), %rax
	movq	%rax, %rcx
	movl	$602, %edx
	leaq	.LC3(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC71(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L801:
	movq	-40(%rbp), %rax
	jmp	.L815
.L807:
	movq	-40(%rbp), %rax
	leaq	16(%rax), %rdx
	movq	-80(%rbp), %rax
	movq	%rax, %rsi
	movq	%rdx, %rdi
	call	cmp
	movl	%eax, -60(%rbp)
	movq	$4, -24(%rbp)
	jmp	.L814
.L806:
	cmpq	$0, -88(%rbp)
	je	.L822
	movq	$8, -24(%rbp)
	jmp	.L814
.L822:
	movq	$2, -24(%rbp)
	jmp	.L814
.L810:
	movq	-48(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, -56(%rbp)
	movq	$12, -24(%rbp)
	jmp	.L814
.L825:
	nop
.L814:
	jmp	.L824
.L815:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE33:
	.size	lookup, .-lookup
	.section	.rodata
	.align 8
.LC72:
	.string	"%s: %d: attempt to get an attribute from the <%s> object"
	.text
	.type	lookup_rec, @function
lookup_rec:
.LFB34:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movq	$8, -8(%rbp)
.L846:
	cmpq	$9, -8(%rbp)
	ja	.L847
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L829(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L829(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L829:
	.long	.L837-.L829
	.long	.L836-.L829
	.long	.L847-.L829
	.long	.L835-.L829
	.long	.L834-.L829
	.long	.L833-.L829
	.long	.L832-.L829
	.long	.L831-.L829
	.long	.L830-.L829
	.long	.L828-.L829
	.text
.L834:
	movq	-40(%rbp), %rcx
	movq	-32(%rbp), %rax
	movl	$0, %edx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	lookup
	movq	%rax, -16(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L838
.L830:
	movq	$3, -8(%rbp)
	jmp	.L838
.L836:
	movq	-32(%rbp), %rax
	movzbl	(%rax), %eax
	movzbl	%al, %eax
	cltq
	leaq	0(,%rax,8), %rdx
	leaq	types(%rip), %rax
	movq	(%rdx,%rax), %rsi
	movq	-24(%rbp), %rax
	movl	48(%rax), %ecx
	movq	rerr(%rip), %rdx
	movq	-24(%rbp), %rax
	movq	%rsi, %r8
	leaq	.LC72(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	die
	movq	$6, -8(%rbp)
	jmp	.L838
.L835:
	cmpq	$0, -32(%rbp)
	je	.L839
	movq	$9, -8(%rbp)
	jmp	.L838
.L839:
	movq	$5, -8(%rbp)
	jmp	.L838
.L828:
	movq	-32(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$3, %al
	je	.L841
	movq	$1, -8(%rbp)
	jmp	.L838
.L841:
	movq	$4, -8(%rbp)
	jmp	.L838
.L832:
	movq	-32(%rbp), %rax
	movq	8(%rax), %rax
	movq	(%rax), %rax
	movq	%rax, -32(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L838
.L833:
	movl	$0, %eax
	jmp	.L843
.L837:
	cmpq	$0, -16(%rbp)
	je	.L844
	movq	$7, -8(%rbp)
	jmp	.L838
.L844:
	movq	$6, -8(%rbp)
	jmp	.L838
.L831:
	movq	-16(%rbp), %rax
	addq	$40, %rax
	jmp	.L843
.L847:
	nop
.L838:
	jmp	.L846
.L843:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE34:
	.size	lookup_rec, .-lookup_rec
	.type	setattr, @function
setattr:
.LFB35:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$96, %rsp
	movq	%rdi, -72(%rbp)
	movq	%rsi, -80(%rbp)
	movq	%rdx, -88(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, -48(%rbp)
.L854:
	cmpq	$2, -48(%rbp)
	je	.L849
	cmpq	$2, -48(%rbp)
	ja	.L857
	cmpq	$0, -48(%rbp)
	je	.L851
	cmpq	$1, -48(%rbp)
	jne	.L857
	movb	$2, -32(%rbp)
	movq	-80(%rbp), %rax
	movq	%rax, -16(%rbp)
	movq	-80(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -40(%rbp)
	movq	-40(%rbp), %rax
	movl	%eax, -24(%rbp)
	movq	-88(%rbp), %rdx
	leaq	-32(%rbp), %rcx
	movq	-72(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	lookup_ins
	movq	%rax, -56(%rbp)
	movq	$2, -48(%rbp)
	jmp	.L852
.L851:
	movq	$1, -48(%rbp)
	jmp	.L852
.L849:
	movq	-56(%rbp), %rax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L855
	jmp	.L856
.L857:
	nop
.L852:
	jmp	.L854
.L856:
	call	__stack_chk_fail@PLT
.L855:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE35:
	.size	setattr, .-setattr
	.type	emit_byte, @function
emit_byte:
.LFB36:
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
	movq	$1, -8(%rbp)
.L863:
	cmpq	$0, -8(%rbp)
	je	.L864
	cmpq	$1, -8(%rbp)
	jne	.L865
	leaq	-28(%rbp), %rcx
	movq	-24(%rbp), %rax
	movl	$1, %edx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	emit
	movq	$0, -8(%rbp)
	jmp	.L861
.L865:
	nop
.L861:
	jmp	.L863
.L864:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE36:
	.size	emit_byte, .-emit_byte
	.section	.rodata
.LC74:
	.string	"%s: stack underflow"
.LC75:
	.string	"corrupter code"
	.align 8
.LC76:
	.string	"attempt to PUSH an unknown shit"
.LC77:
	.string	"division by zero"
	.align 8
.LC78:
	.string	"%s: %d: unexpected arith operator: %u"
	.align 8
.LC79:
	.string	"%s: %d: <%s> found, <num> expected"
	.align 8
.LC80:
	.string	"%s: %d: attemt to set an attribute to <%s> object"
.LC82:
	.string	"corrupt code"
	.align 8
.LC83:
	.string	"%s: %d: attempt to call <%s> object"
	.text
	.type	execute, @function
execute:
.LFB38:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$112, %rsp
	movq	%rdi, -104(%rbp)
	movq	$35, -24(%rbp)
.L968:
	cmpq	$86, -24(%rbp)
	ja	.L970
	movq	-24(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L869(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L869(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L869:
	.long	.L918-.L869
	.long	.L970-.L869
	.long	.L917-.L869
	.long	.L970-.L869
	.long	.L916-.L869
	.long	.L915-.L869
	.long	.L970-.L869
	.long	.L970-.L869
	.long	.L970-.L869
	.long	.L914-.L869
	.long	.L970-.L869
	.long	.L913-.L869
	.long	.L912-.L869
	.long	.L970-.L869
	.long	.L911-.L869
	.long	.L910-.L869
	.long	.L909-.L869
	.long	.L970-.L869
	.long	.L971-.L869
	.long	.L907-.L869
	.long	.L970-.L869
	.long	.L970-.L869
	.long	.L906-.L869
	.long	.L905-.L869
	.long	.L970-.L869
	.long	.L904-.L869
	.long	.L903-.L869
	.long	.L902-.L869
	.long	.L970-.L869
	.long	.L901-.L869
	.long	.L970-.L869
	.long	.L900-.L869
	.long	.L899-.L869
	.long	.L898-.L869
	.long	.L897-.L869
	.long	.L896-.L869
	.long	.L970-.L869
	.long	.L970-.L869
	.long	.L970-.L869
	.long	.L970-.L869
	.long	.L895-.L869
	.long	.L894-.L869
	.long	.L893-.L869
	.long	.L892-.L869
	.long	.L970-.L869
	.long	.L891-.L869
	.long	.L890-.L869
	.long	.L970-.L869
	.long	.L889-.L869
	.long	.L970-.L869
	.long	.L888-.L869
	.long	.L970-.L869
	.long	.L887-.L869
	.long	.L970-.L869
	.long	.L886-.L869
	.long	.L970-.L869
	.long	.L970-.L869
	.long	.L885-.L869
	.long	.L884-.L869
	.long	.L970-.L869
	.long	.L883-.L869
	.long	.L970-.L869
	.long	.L970-.L869
	.long	.L882-.L869
	.long	.L881-.L869
	.long	.L970-.L869
	.long	.L970-.L869
	.long	.L880-.L869
	.long	.L879-.L869
	.long	.L970-.L869
	.long	.L878-.L869
	.long	.L877-.L869
	.long	.L876-.L869
	.long	.L970-.L869
	.long	.L970-.L869
	.long	.L875-.L869
	.long	.L874-.L869
	.long	.L970-.L869
	.long	.L970-.L869
	.long	.L873-.L869
	.long	.L970-.L869
	.long	.L970-.L869
	.long	.L970-.L869
	.long	.L872-.L869
	.long	.L871-.L869
	.long	.L870-.L869
	.long	.L868-.L869
	.text
.L888:
	movq	-88(%rbp), %rax
	movq	%rax, -16(%rbp)
	addq	$1, -88(%rbp)
	movq	-104(%rbp), %rax
	movq	32(%rax), %rcx
	movq	-104(%rbp), %rax
	movl	40(%rax), %eax
	movslq	%eax, %rdx
	movq	-16(%rbp), %rax
	movzbl	(%rax), %eax
	movsbq	%al, %rax
	subq	%rax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	subq	$24, %rax
	addq	%rcx, %rax
	movq	%rax, -80(%rbp)
	movq	-104(%rbp), %rax
	movq	%rax, %rdi
	call	push
	movq	%rax, -72(%rbp)
	movq	-72(%rbp), %rcx
	movq	-80(%rbp), %rsi
	movq	(%rsi), %rax
	movq	8(%rsi), %rdx
	movq	%rax, (%rcx)
	movq	%rdx, 8(%rcx)
	movq	16(%rsi), %rax
	movq	%rax, 16(%rcx)
	movq	$85, -24(%rbp)
	jmp	.L920
.L904:
	movq	-80(%rbp), %rax
	movl	$0, %esi
	movq	%rax, %rdi
	call	init_hash_obj
	movq	$85, -24(%rbp)
	jmp	.L920
.L887:
	movq	-104(%rbp), %rax
	movl	48(%rax), %edx
	movq	-104(%rbp), %rax
	movl	52(%rax), %eax
	cmpl	%eax, %edx
	jl	.L921
	movq	$42, -24(%rbp)
	jmp	.L920
.L921:
	movq	$67, -24(%rbp)
	jmp	.L920
.L916:
	movq	-48(%rbp), %rax
	movsd	8(%rax), %xmm0
	pxor	%xmm1, %xmm1
	ucomisd	%xmm1, %xmm0
	jp	.L923
	pxor	%xmm1, %xmm1
	ucomisd	%xmm1, %xmm0
	jne	.L923
	movq	$54, -24(%rbp)
	jmp	.L920
.L923:
	movq	$9, -24(%rbp)
	jmp	.L920
.L911:
	movq	-80(%rbp), %rcx
	movq	-72(%rbp), %rsi
	movq	(%rsi), %rax
	movq	8(%rsi), %rdx
	movq	%rax, (%rcx)
	movq	%rdx, 8(%rcx)
	movq	16(%rsi), %rax
	movq	%rax, 16(%rcx)
	movq	$0, -24(%rbp)
	jmp	.L920
.L910:
	movq	rerr(%rip), %rdx
	movq	-104(%rbp), %rax
	leaq	.LC74(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	die
	movq	$50, -24(%rbp)
	jmp	.L920
.L873:
	movq	-104(%rbp), %rax
	movl	48(%rax), %ecx
	movq	rerr(%rip), %rdx
	movq	-104(%rbp), %rax
	leaq	.LC75(%rip), %r8
	leaq	.LC11(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	die
	movq	$85, -24(%rbp)
	jmp	.L920
.L900:
	movq	-80(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$4, %al
	jne	.L926
	movq	$85, -24(%rbp)
	jmp	.L920
.L926:
	movq	$83, -24(%rbp)
	jmp	.L920
.L912:
	movq	-104(%rbp), %rax
	movl	48(%rax), %ecx
	movq	rerr(%rip), %rdx
	movq	-104(%rbp), %rax
	leaq	.LC76(%rip), %r8
	leaq	.LC11(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	die
	call	abort@PLT
.L891:
	movq	-72(%rbp), %rdx
	movq	-64(%rbp), %rcx
	movq	-80(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	lookup_ins
	movq	-104(%rbp), %rax
	movl	40(%rax), %eax
	leal	1(%rax), %edx
	movq	-104(%rbp), %rax
	movl	%edx, 40(%rax)
	movq	$85, -24(%rbp)
	jmp	.L920
.L886:
	movq	-104(%rbp), %rax
	movl	48(%rax), %ecx
	movq	rerr(%rip), %rdx
	movq	-104(%rbp), %rax
	leaq	.LC77(%rip), %r8
	leaq	.LC11(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	die
	movq	$9, -24(%rbp)
	jmp	.L920
.L905:
	movq	-104(%rbp), %rax
	movq	24(%rax), %rdx
	movq	-88(%rbp), %rax
	subq	%rdx, %rax
	movl	%eax, %edx
	movq	-104(%rbp), %rax
	movl	%edx, 48(%rax)
	movq	$52, -24(%rbp)
	jmp	.L920
.L878:
	movq	-56(%rbp), %rax
	movsd	8(%rax), %xmm0
	movq	-48(%rbp), %rax
	movsd	8(%rax), %xmm1
	subsd	%xmm1, %xmm0
	movq	-56(%rbp), %rax
	movsd	%xmm0, 8(%rax)
	movq	$40, -24(%rbp)
	jmp	.L920
.L909:
	movq	-80(%rbp), %rax
	movzbl	(%rax), %eax
	movzbl	%al, %eax
	cmpl	$3, %eax
	je	.L928
	cmpl	$3, %eax
	jg	.L929
	cmpl	$2, %eax
	je	.L930
	cmpl	$2, %eax
	jg	.L929
	testl	%eax, %eax
	je	.L931
	cmpl	$1, %eax
	je	.L932
	jmp	.L929
.L931:
	movq	$85, -24(%rbp)
	jmp	.L933
.L928:
	movq	$25, -24(%rbp)
	jmp	.L933
.L930:
	movq	$32, -24(%rbp)
	jmp	.L933
.L932:
	movq	$34, -24(%rbp)
	jmp	.L933
.L929:
	movq	$12, -24(%rbp)
	nop
.L933:
	jmp	.L920
.L874:
	movq	-104(%rbp), %rax
	movq	%rax, %rdi
	call	pop
	movq	%rax, -48(%rbp)
	movq	-104(%rbp), %rax
	movq	%rax, %rdi
	call	pop
	movq	%rax, -56(%rbp)
	movq	$48, -24(%rbp)
	jmp	.L920
.L885:
	movq	-88(%rbp), %rax
	subq	$1, %rax
	movzbl	(%rax), %eax
	movsbl	%al, %esi
	movq	-104(%rbp), %rax
	movl	48(%rax), %ecx
	movq	rerr(%rip), %rdx
	movq	-104(%rbp), %rax
	movl	%esi, %r8d
	leaq	.LC78(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	die
	movq	$40, -24(%rbp)
	jmp	.L920
.L879:
	movq	-56(%rbp), %rax
	movzbl	(%rax), %eax
	movzbl	%al, %eax
	cltq
	leaq	0(,%rax,8), %rdx
	leaq	types(%rip), %rax
	movq	(%rdx,%rax), %rsi
	movq	-104(%rbp), %rax
	movl	48(%rax), %ecx
	movq	rerr(%rip), %rdx
	movq	-104(%rbp), %rax
	movq	%rsi, %r8
	leaq	.LC79(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	die
	movq	$29, -24(%rbp)
	jmp	.L920
.L870:
	movq	-88(%rbp), %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	je	.L934
	movq	$23, -24(%rbp)
	jmp	.L920
.L934:
	movq	$18, -24(%rbp)
	jmp	.L920
.L903:
	movq	-56(%rbp), %rax
	movzbl	(%rax), %eax
	movzbl	%al, %eax
	cltq
	leaq	0(,%rax,8), %rdx
	leaq	types(%rip), %rax
	movq	(%rdx,%rax), %rsi
	movq	-104(%rbp), %rax
	movl	48(%rax), %ecx
	movq	rerr(%rip), %rdx
	movq	-104(%rbp), %rax
	movq	%rsi, %r8
	leaq	.LC79(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	die
	movq	$29, -24(%rbp)
	jmp	.L920
.L913:
	movq	-104(%rbp), %rax
	movq	%rax, %rdi
	call	pop
	movq	%rax, -64(%rbp)
	movq	-104(%rbp), %rax
	movq	%rax, %rdi
	call	pop
	movq	%rax, -80(%rbp)
	movq	-104(%rbp), %rax
	movq	%rax, %rdi
	call	pop
	movq	%rax, -72(%rbp)
	movq	$33, -24(%rbp)
	jmp	.L920
.L914:
	movq	-56(%rbp), %rax
	movsd	8(%rax), %xmm0
	movq	-48(%rbp), %rax
	movsd	8(%rax), %xmm1
	divsd	%xmm1, %xmm0
	movq	-56(%rbp), %rax
	movsd	%xmm0, 8(%rax)
	movq	$40, -24(%rbp)
	jmp	.L920
.L882:
	cmpq	$0, -72(%rbp)
	je	.L936
	movq	$14, -24(%rbp)
	jmp	.L920
.L936:
	movq	$27, -24(%rbp)
	jmp	.L920
.L907:
	movq	-32(%rbp), %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	cmpl	$47, %eax
	je	.L938
	cmpl	$47, %eax
	jg	.L939
	cmpl	$45, %eax
	je	.L940
	cmpl	$45, %eax
	jg	.L939
	cmpl	$42, %eax
	je	.L941
	cmpl	$43, %eax
	je	.L942
	jmp	.L939
.L938:
	movq	$4, -24(%rbp)
	jmp	.L943
.L941:
	movq	$43, -24(%rbp)
	jmp	.L943
.L940:
	movq	$70, -24(%rbp)
	jmp	.L943
.L942:
	movq	$60, -24(%rbp)
	jmp	.L943
.L939:
	movq	$57, -24(%rbp)
	nop
.L943:
	jmp	.L920
.L899:
	movq	-88(%rbp), %rax
	movq	(%rax), %rax
	movl	%eax, %edx
	movq	-80(%rbp), %rax
	movl	%edx, 8(%rax)
	addq	$8, -88(%rbp)
	movq	-80(%rbp), %rax
	movq	-88(%rbp), %rdx
	movq	%rdx, 16(%rax)
	movq	-80(%rbp), %rax
	movl	8(%rax), %eax
	cltq
	addq	%rax, -88(%rbp)
	movq	$85, -24(%rbp)
	jmp	.L920
.L895:
	movq	-104(%rbp), %rax
	movl	40(%rax), %eax
	leal	1(%rax), %edx
	movq	-104(%rbp), %rax
	movl	%edx, 40(%rax)
	movq	$85, -24(%rbp)
	jmp	.L920
.L880:
	movq	-88(%rbp), %rax
	movq	%rax, -40(%rbp)
	addq	$1, -88(%rbp)
	movq	$22, -24(%rbp)
	jmp	.L920
.L883:
	movq	-56(%rbp), %rax
	movsd	8(%rax), %xmm1
	movq	-48(%rbp), %rax
	movsd	8(%rax), %xmm0
	addsd	%xmm1, %xmm0
	movq	-56(%rbp), %rax
	movsd	%xmm0, 8(%rax)
	movq	$40, -24(%rbp)
	jmp	.L920
.L902:
	movq	-80(%rbp), %rax
	movb	$0, (%rax)
	movq	$0, -24(%rbp)
	jmp	.L920
.L884:
	movq	-104(%rbp), %rax
	movq	%rax, %rdi
	call	push
	movq	%rax, -80(%rbp)
	movq	-104(%rbp), %rax
	movq	32(%rax), %rcx
	movq	-104(%rbp), %rax
	movl	44(%rax), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	leaq	(%rcx,%rax), %rdx
	movq	-80(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	init_hash_obj
	movq	$85, -24(%rbp)
	jmp	.L920
.L871:
	movq	-104(%rbp), %rax
	movq	%rax, %rdi
	call	push
	movq	%rax, -80(%rbp)
	movq	-80(%rbp), %rax
	movb	$3, (%rax)
	movq	-104(%rbp), %rax
	movq	32(%rax), %rcx
	movq	-104(%rbp), %rax
	movl	44(%rax), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movq	8(%rax), %rdx
	movq	-80(%rbp), %rax
	movq	%rdx, 8(%rax)
	movq	$85, -24(%rbp)
	jmp	.L920
.L897:
	movq	-88(%rbp), %rax
	movsd	(%rax), %xmm0
	movq	-80(%rbp), %rax
	movsd	%xmm0, 8(%rax)
	addq	$8, -88(%rbp)
	movq	$85, -24(%rbp)
	jmp	.L920
.L875:
	movq	-48(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$1, %al
	je	.L944
	movq	$68, -24(%rbp)
	jmp	.L920
.L944:
	movq	$29, -24(%rbp)
	jmp	.L920
.L889:
	movq	-56(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$1, %al
	je	.L946
	movq	$26, -24(%rbp)
	jmp	.L920
.L946:
	movq	$75, -24(%rbp)
	jmp	.L920
.L877:
	movq	-80(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$5, %al
	jne	.L948
	movq	$72, -24(%rbp)
	jmp	.L920
.L948:
	movq	$31, -24(%rbp)
	jmp	.L920
.L906:
	movq	-40(%rbp), %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	cmpl	$11, %eax
	ja	.L950
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L952(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L952(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L952:
	.long	.L950-.L952
	.long	.L962-.L952
	.long	.L961-.L952
	.long	.L960-.L952
	.long	.L959-.L952
	.long	.L958-.L952
	.long	.L957-.L952
	.long	.L956-.L952
	.long	.L955-.L952
	.long	.L954-.L952
	.long	.L953-.L952
	.long	.L951-.L952
	.text
.L951:
	movq	$58, -24(%rbp)
	jmp	.L963
.L953:
	movq	$84, -24(%rbp)
	jmp	.L963
.L955:
	movq	$85, -24(%rbp)
	jmp	.L963
.L954:
	movq	$41, -24(%rbp)
	jmp	.L963
.L956:
	movq	$86, -24(%rbp)
	jmp	.L963
.L957:
	movq	$76, -24(%rbp)
	jmp	.L963
.L958:
	movq	$46, -24(%rbp)
	jmp	.L963
.L959:
	movq	$64, -24(%rbp)
	jmp	.L963
.L960:
	movq	$11, -24(%rbp)
	jmp	.L963
.L961:
	movq	$85, -24(%rbp)
	jmp	.L963
.L962:
	movq	$2, -24(%rbp)
	jmp	.L963
.L950:
	movq	$79, -24(%rbp)
	nop
.L963:
	jmp	.L920
.L915:
	movq	-80(%rbp), %rax
	movzbl	(%rax), %eax
	movzbl	%al, %eax
	cltq
	leaq	0(,%rax,8), %rdx
	leaq	types(%rip), %rax
	movq	(%rdx,%rax), %rsi
	movq	-104(%rbp), %rax
	movl	48(%rax), %ecx
	movq	rerr(%rip), %rdx
	movq	-104(%rbp), %rax
	movq	%rsi, %r8
	leaq	.LC80(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	die
	movq	$45, -24(%rbp)
	jmp	.L920
.L876:
	movq	-80(%rbp), %rax
	movq	8(%rax), %rdx
	movq	-104(%rbp), %rax
	movq	%rax, %rdi
	call	*%rdx
	movq	-104(%rbp), %rax
	movl	40(%rax), %eax
	pxor	%xmm0, %xmm0
	cvtsi2sdl	%eax, %xmm0
	movq	-80(%rbp), %rax
	subq	$24, %rax
	movsd	8(%rax), %xmm2
	movsd	.LC81(%rip), %xmm1
	addsd	%xmm2, %xmm1
	subsd	%xmm1, %xmm0
	cvttsd2sil	%xmm0, %edx
	movq	-104(%rbp), %rax
	movl	%edx, 40(%rax)
	movq	$85, -24(%rbp)
	jmp	.L920
.L898:
	movq	-80(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$3, %al
	je	.L964
	movq	$5, -24(%rbp)
	jmp	.L920
.L964:
	movq	$45, -24(%rbp)
	jmp	.L920
.L881:
	movq	-104(%rbp), %rax
	movq	%rax, %rdi
	call	pop
	movq	%rax, -64(%rbp)
	movq	-104(%rbp), %rax
	movq	%rax, %rdi
	call	pop
	movq	%rax, -80(%rbp)
	movq	-64(%rbp), %rdx
	movq	-80(%rbp), %rcx
	movq	-104(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	lookup_rec
	movq	%rax, -72(%rbp)
	movq	$63, -24(%rbp)
	jmp	.L920
.L894:
	movq	-104(%rbp), %rax
	movl	40(%rax), %edx
	movq	-88(%rbp), %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	cmpl	%eax, %edx
	jg	.L966
	movq	$15, -24(%rbp)
	jmp	.L920
.L966:
	movq	$50, -24(%rbp)
	jmp	.L920
.L893:
	movq	-104(%rbp), %rax
	leaq	.LC82(%rip), %rdx
	leaq	.LC18(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	die
	movq	$67, -24(%rbp)
	jmp	.L920
.L918:
	movq	-104(%rbp), %rax
	movl	40(%rax), %eax
	leal	1(%rax), %edx
	movq	-104(%rbp), %rax
	movl	%edx, 40(%rax)
	movq	$85, -24(%rbp)
	jmp	.L920
.L890:
	movq	-104(%rbp), %rax
	movq	%rax, %rdi
	call	push
	movq	%rax, -80(%rbp)
	movq	-88(%rbp), %rax
	movq	%rax, -8(%rbp)
	addq	$1, -88(%rbp)
	movq	-8(%rbp), %rax
	movzbl	(%rax), %eax
	movl	%eax, %edx
	movq	-80(%rbp), %rax
	movb	%dl, (%rax)
	movq	$16, -24(%rbp)
	jmp	.L920
.L872:
	movq	-80(%rbp), %rax
	movzbl	(%rax), %eax
	movzbl	%al, %eax
	cltq
	leaq	0(,%rax,8), %rdx
	leaq	types(%rip), %rax
	movq	(%rdx,%rax), %rsi
	movq	-104(%rbp), %rax
	movl	48(%rax), %ecx
	movq	rerr(%rip), %rdx
	movq	-104(%rbp), %rax
	movq	%rsi, %r8
	leaq	.LC83(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	die
	movq	$85, -24(%rbp)
	jmp	.L920
.L896:
	movq	-104(%rbp), %rax
	movq	24(%rax), %rdx
	movq	-104(%rbp), %rax
	movl	48(%rax), %eax
	cltq
	addq	%rdx, %rax
	movq	%rax, -88(%rbp)
	movq	$85, -24(%rbp)
	jmp	.L920
.L901:
	movq	-88(%rbp), %rax
	movq	%rax, -32(%rbp)
	addq	$1, -88(%rbp)
	movq	$19, -24(%rbp)
	jmp	.L920
.L892:
	movq	-56(%rbp), %rax
	movsd	8(%rax), %xmm1
	movq	-48(%rbp), %rax
	movsd	8(%rax), %xmm0
	mulsd	%xmm1, %xmm0
	movq	-56(%rbp), %rax
	movsd	%xmm0, 8(%rax)
	movq	$40, -24(%rbp)
	jmp	.L920
.L868:
	movq	-104(%rbp), %rax
	movq	%rax, %rdi
	call	pop
	movq	$85, -24(%rbp)
	jmp	.L920
.L917:
	movq	-104(%rbp), %rax
	movq	%rax, %rdi
	call	pop
	movq	%rax, -80(%rbp)
	movq	$71, -24(%rbp)
	jmp	.L920
.L970:
	nop
.L920:
	jmp	.L968
.L971:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE38:
	.size	execute, .-execute
	.section	.rodata
	.align 8
.LC84:
	.string	"%s: %d: expected number on stack"
.LC85:
	.string	"%s: %d: stack underflow"
	.text
	.type	gsi, @function
gsi:
.LFB39:
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
	movq	$5, -8(%rbp)
.L987:
	cmpq	$5, -8(%rbp)
	ja	.L989
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L975(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L975(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L975:
	.long	.L980-.L975
	.long	.L979-.L975
	.long	.L978-.L975
	.long	.L977-.L975
	.long	.L976-.L975
	.long	.L974-.L975
	.text
.L976:
	movq	-24(%rbp), %rax
	movl	48(%rax), %ecx
	movq	rerr(%rip), %rdx
	movq	-24(%rbp), %rax
	leaq	.LC84(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	die
	movq	$3, -8(%rbp)
	jmp	.L981
.L979:
	movq	-24(%rbp), %rax
	movl	48(%rax), %ecx
	movq	rerr(%rip), %rdx
	movq	-24(%rbp), %rax
	leaq	.LC85(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	die
	movq	$3, -8(%rbp)
	jmp	.L981
.L977:
	movq	-16(%rbp), %rax
	movsd	8(%rax), %xmm0
	cvttsd2sil	%xmm0, %eax
	jmp	.L988
.L974:
	movq	-24(%rbp), %rax
	movq	32(%rax), %rcx
	movq	-24(%rbp), %rax
	movl	40(%rax), %eax
	cltq
	movl	-28(%rbp), %edx
	movslq	%edx, %rsi
	subq	%rsi, %rax
	movq	%rax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	subq	$24, %rax
	addq	%rcx, %rax
	movq	%rax, -16(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L981
.L980:
	movq	-16(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$1, %al
	je	.L983
	movq	$4, -8(%rbp)
	jmp	.L981
.L983:
	movq	$3, -8(%rbp)
	jmp	.L981
.L978:
	movq	-24(%rbp), %rax
	movq	32(%rax), %rax
	movq	%rax, %rdx
	movq	-16(%rbp), %rax
	cmpq	%rax, %rdx
	jbe	.L985
	movq	$1, -8(%rbp)
	jmp	.L981
.L985:
	movq	$0, -8(%rbp)
	jmp	.L981
.L989:
	nop
.L981:
	jmp	.L987
.L988:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE39:
	.size	gsi, .-gsi
	.section	.rodata
.LC86:
	.string	"ABC1"
.LC87:
	.string	"w+"
	.text
	.type	q_save, @function
q_save:
.LFB40:
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
	movq	$6, -24(%rbp)
.L1002:
	cmpq	$6, -24(%rbp)
	ja	.L1003
	movq	-24(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L993(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L993(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L993:
	.long	.L1003-.L993
	.long	.L997-.L993
	.long	.L1003-.L993
	.long	.L996-.L993
	.long	.L995-.L993
	.long	.L1004-.L993
	.long	.L992-.L993
	.text
.L995:
	cmpq	$0, -32(%rbp)
	jne	.L998
	movq	$3, -24(%rbp)
	jmp	.L1000
.L998:
	movq	$1, -24(%rbp)
	jmp	.L1000
.L997:
	movq	-32(%rbp), %rax
	movq	%rax, %rcx
	movl	$4, %edx
	movl	$1, %esi
	leaq	.LC86(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	-48(%rbp), %rax
	movl	52(%rax), %eax
	movslq	%eax, %rdx
	movq	-48(%rbp), %rax
	movq	24(%rax), %rax
	movq	-32(%rbp), %rcx
	movl	$1, %esi
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$5, -24(%rbp)
	jmp	.L1000
.L996:
	call	__errno_location@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, %edi
	call	strerror@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rcx
	movq	-40(%rbp), %rdx
	movq	-56(%rbp), %rax
	movq	%rcx, %r8
	movq	%rdx, %rcx
	leaq	.LC1(%rip), %rdx
	movl	$256, %esi
	movq	%rax, %rdi
	movl	$0, %eax
	call	snprintf@PLT
	movq	$5, -24(%rbp)
	jmp	.L1000
.L992:
	movq	-40(%rbp), %rax
	leaq	.LC87(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -32(%rbp)
	movq	$4, -24(%rbp)
	jmp	.L1000
.L1003:
	nop
.L1000:
	jmp	.L1002
.L1004:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE40:
	.size	q_save, .-q_save
	.section	.rodata
.LC88:
	.string	"%.*s"
.LC89:
	.string	"<%s>"
	.text
	.type	print_obj, @function
print_obj:
.LFB42:
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
.L1019:
	cmpq	$7, -8(%rbp)
	ja	.L1021
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L1008(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L1008(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L1008:
	.long	.L1012-.L1008
	.long	.L1021-.L1008
	.long	.L1011-.L1008
	.long	.L1022-.L1008
	.long	.L1009-.L1008
	.long	.L1021-.L1008
	.long	.L1021-.L1008
	.long	.L1007-.L1008
	.text
.L1009:
	movq	-24(%rbp), %rax
	movzbl	(%rax), %eax
	movzbl	%al, %eax
	cmpl	$1, %eax
	je	.L1013
	cmpl	$2, %eax
	je	.L1014
	jmp	.L1020
.L1013:
	movq	$7, -8(%rbp)
	jmp	.L1016
.L1014:
	movq	$0, -8(%rbp)
	jmp	.L1016
.L1020:
	movq	$2, -8(%rbp)
	nop
.L1016:
	jmp	.L1017
.L1012:
	movq	-24(%rbp), %rax
	movq	16(%rax), %rdx
	movq	-24(%rbp), %rax
	movl	8(%rax), %eax
	movl	%eax, %esi
	leaq	.LC88(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -8(%rbp)
	jmp	.L1017
.L1007:
	movq	-24(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, %xmm0
	leaq	.LC62(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$3, -8(%rbp)
	jmp	.L1017
.L1011:
	movq	-24(%rbp), %rax
	movzbl	(%rax), %eax
	movzbl	%al, %eax
	cltq
	leaq	0(,%rax,8), %rdx
	leaq	types(%rip), %rax
	movq	(%rdx,%rax), %rax
	movq	%rax, %rsi
	leaq	.LC89(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -8(%rbp)
	jmp	.L1017
.L1021:
	nop
.L1017:
	jmp	.L1019
.L1022:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE42:
	.size	print_obj, .-print_obj
	.type	match_long_token, @function
match_long_token:
.LFB43:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$80, %rsp
	movq	%rdi, -72(%rbp)
	movq	%rsi, -80(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$7, -32(%rbp)
.L1077:
	cmpq	$31, -32(%rbp)
	ja	.L1080
	movq	-32(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L1026(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L1026(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L1026:
	.long	.L1048-.L1026
	.long	.L1080-.L1026
	.long	.L1047-.L1026
	.long	.L1046-.L1026
	.long	.L1080-.L1026
	.long	.L1045-.L1026
	.long	.L1080-.L1026
	.long	.L1044-.L1026
	.long	.L1043-.L1026
	.long	.L1042-.L1026
	.long	.L1041-.L1026
	.long	.L1040-.L1026
	.long	.L1039-.L1026
	.long	.L1038-.L1026
	.long	.L1037-.L1026
	.long	.L1080-.L1026
	.long	.L1036-.L1026
	.long	.L1035-.L1026
	.long	.L1080-.L1026
	.long	.L1034-.L1026
	.long	.L1033-.L1026
	.long	.L1032-.L1026
	.long	.L1031-.L1026
	.long	.L1030-.L1026
	.long	.L1080-.L1026
	.long	.L1029-.L1026
	.long	.L1028-.L1026
	.long	.L1080-.L1026
	.long	.L1080-.L1026
	.long	.L1027-.L1026
	.long	.L1080-.L1026
	.long	.L1025-.L1026
	.text
.L1029:
	movq	-72(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rdx
	movq	-72(%rbp), %rcx
	movq	-80(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	strncmp@PLT
	movl	%eax, -60(%rbp)
	movq	$16, -32(%rbp)
	jmp	.L1049
.L1037:
	movl	-64(%rbp), %eax
	movslq	%eax, %rdx
	movq	-80(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	cmpb	$95, %al
	jne	.L1050
	movq	$17, -32(%rbp)
	jmp	.L1049
.L1050:
	movq	$31, -32(%rbp)
	jmp	.L1049
.L1025:
	call	__ctype_b_loc@PLT
	movq	%rax, -48(%rbp)
	movq	$29, -32(%rbp)
	jmp	.L1049
.L1039:
	movq	-72(%rbp), %rax
	addq	$1, %rax
	movzbl	(%rax), %eax
	cmpb	$118, %al
	jne	.L1052
	movq	$26, -32(%rbp)
	jmp	.L1049
.L1052:
	movq	$13, -32(%rbp)
	jmp	.L1049
.L1043:
	movq	-80(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$34, %al
	jne	.L1054
	movq	$11, -32(%rbp)
	jmp	.L1049
.L1054:
	movq	$9, -32(%rbp)
	jmp	.L1049
.L1030:
	call	__ctype_b_loc@PLT
	movq	%rax, -40(%rbp)
	movq	$19, -32(%rbp)
	jmp	.L1049
.L1046:
	movq	-56(%rbp), %rax
	subq	-80(%rbp), %rax
	addl	$1, %eax
	movl	%eax, -64(%rbp)
	movq	$9, -32(%rbp)
	jmp	.L1049
.L1036:
	cmpl	$0, -60(%rbp)
	jne	.L1056
	movq	$21, -32(%rbp)
	jmp	.L1049
.L1056:
	movq	$9, -32(%rbp)
	jmp	.L1049
.L1032:
	movq	-72(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movl	%eax, -64(%rbp)
	movq	$9, -32(%rbp)
	jmp	.L1049
.L1028:
	movq	-80(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$95, %al
	jne	.L1058
	movq	$14, -32(%rbp)
	jmp	.L1049
.L1058:
	movq	$23, -32(%rbp)
	jmp	.L1049
.L1040:
	movq	-80(%rbp), %rax
	addq	$1, %rax
	movl	$34, %esi
	movq	%rax, %rdi
	call	strchr@PLT
	movq	%rax, -56(%rbp)
	movq	$10, -32(%rbp)
	jmp	.L1049
.L1042:
	movl	-64(%rbp), %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L1078
	jmp	.L1079
.L1038:
	movq	-72(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$37, %al
	jne	.L1061
	movq	$5, -32(%rbp)
	jmp	.L1049
.L1061:
	movq	$22, -32(%rbp)
	jmp	.L1049
.L1034:
	movq	-40(%rbp), %rax
	movq	(%rax), %rdx
	movq	-80(%rbp), %rax
	movzbl	(%rax), %eax
	movzbl	%al, %eax
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$1024, %eax
	testl	%eax, %eax
	je	.L1063
	movq	$14, -32(%rbp)
	jmp	.L1049
.L1063:
	movq	$9, -32(%rbp)
	jmp	.L1049
.L1035:
	addl	$1, -64(%rbp)
	movq	$14, -32(%rbp)
	jmp	.L1049
.L1031:
	movq	-72(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$37, %al
	jne	.L1065
	movq	$2, -32(%rbp)
	jmp	.L1049
.L1065:
	movq	$25, -32(%rbp)
	jmp	.L1049
.L1045:
	movq	-72(%rbp), %rax
	addq	$1, %rax
	movzbl	(%rax), %eax
	cmpb	$100, %al
	jne	.L1067
	movq	$0, -32(%rbp)
	jmp	.L1049
.L1067:
	movq	$22, -32(%rbp)
	jmp	.L1049
.L1041:
	movq	-56(%rbp), %rax
	testq	%rax, %rax
	je	.L1069
	movq	$3, -32(%rbp)
	jmp	.L1049
.L1069:
	movq	$9, -32(%rbp)
	jmp	.L1049
.L1048:
	leaq	-56(%rbp), %rdx
	movq	-80(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strtod@PLT
	movq	-56(%rbp), %rax
	subq	-80(%rbp), %rax
	movl	%eax, -64(%rbp)
	movq	$9, -32(%rbp)
	jmp	.L1049
.L1044:
	movl	$0, -64(%rbp)
	movq	$20, -32(%rbp)
	jmp	.L1049
.L1027:
	movq	-48(%rbp), %rax
	movq	(%rax), %rdx
	movl	-64(%rbp), %eax
	movslq	%eax, %rcx
	movq	-80(%rbp), %rax
	addq	%rcx, %rax
	movzbl	(%rax), %eax
	movzbl	%al, %eax
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$8, %eax
	testl	%eax, %eax
	je	.L1071
	movq	$17, -32(%rbp)
	jmp	.L1049
.L1071:
	movq	$9, -32(%rbp)
	jmp	.L1049
.L1047:
	movq	-72(%rbp), %rax
	addq	$1, %rax
	movzbl	(%rax), %eax
	cmpb	$115, %al
	jne	.L1073
	movq	$8, -32(%rbp)
	jmp	.L1049
.L1073:
	movq	$25, -32(%rbp)
	jmp	.L1049
.L1033:
	movq	-72(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$37, %al
	jne	.L1075
	movq	$12, -32(%rbp)
	jmp	.L1049
.L1075:
	movq	$13, -32(%rbp)
	jmp	.L1049
.L1080:
	nop
.L1049:
	jmp	.L1077
.L1079:
	call	__stack_chk_fail@PLT
.L1078:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE43:
	.size	match_long_token, .-match_long_token
	.section	.rodata
.LC90:
	.string	"reduce"
.LC91:
	.string	"is_op(parent->tok)"
.LC92:
	.string	"%s: line %d"
	.text
	.type	reduce, @function
reduce:
.LFB44:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$80, %rsp
	movq	%rdi, -56(%rbp)
	movq	%rsi, -64(%rbp)
	movl	%edx, -68(%rbp)
	movq	%rcx, -80(%rbp)
	movq	$10, -16(%rbp)
.L1110:
	cmpq	$19, -16(%rbp)
	ja	.L1111
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L1084(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L1084(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L1084:
	.long	.L1099-.L1084
	.long	.L1098-.L1084
	.long	.L1111-.L1084
	.long	.L1097-.L1084
	.long	.L1112-.L1084
	.long	.L1095-.L1084
	.long	.L1094-.L1084
	.long	.L1093-.L1084
	.long	.L1092-.L1084
	.long	.L1111-.L1084
	.long	.L1091-.L1084
	.long	.L1090-.L1084
	.long	.L1111-.L1084
	.long	.L1089-.L1084
	.long	.L1088-.L1084
	.long	.L1087-.L1084
	.long	.L1086-.L1084
	.long	.L1085-.L1084
	.long	.L1111-.L1084
	.long	.L1083-.L1084
	.text
.L1088:
	movq	-64(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, -32(%rbp)
	movq	$17, -16(%rbp)
	jmp	.L1101
.L1087:
	cmpl	$0, -44(%rbp)
	je	.L1102
	movq	$16, -16(%rbp)
	jmp	.L1101
.L1102:
	movq	$6, -16(%rbp)
	jmp	.L1101
.L1092:
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	movq	-24(%rbp), %rdx
	movq	8(%rdx), %rdx
	movq	%rdx, 8(%rax)
	movq	-24(%rbp), %rax
	movq	8(%rax), %rax
	movq	-24(%rbp), %rdx
	movq	(%rdx), %rdx
	movq	%rdx, (%rax)
	movq	-24(%rbp), %rax
	movq	%rax, -8(%rbp)
	movq	-24(%rbp), %rax
	movq	-8(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movq	-24(%rbp), %rax
	movq	-8(%rbp), %rdx
	movq	%rdx, (%rax)
	movq	$15, -16(%rbp)
	jmp	.L1101
.L1098:
	movq	-32(%rbp), %rdx
	movq	-80(%rbp), %rax
	cmpq	%rax, %rdx
	jne	.L1104
	movq	$19, -16(%rbp)
	jmp	.L1101
.L1104:
	movq	$8, -16(%rbp)
	jmp	.L1101
.L1097:
	movq	-24(%rbp), %rax
	movq	-64(%rbp), %rdx
	movq	%rdx, 56(%rax)
	movq	$4, -16(%rbp)
	jmp	.L1101
.L1086:
	movq	-64(%rbp), %rax
	movq	-24(%rbp), %rdx
	movq	%rdx, 40(%rax)
	movq	$3, -16(%rbp)
	jmp	.L1101
.L1090:
	movq	-64(%rbp), %rax
	movl	32(%rax), %eax
	movl	%eax, %edi
	call	is_rtl
	movl	%eax, -36(%rbp)
	cmpl	$0, -68(%rbp)
	sete	%al
	movzbl	%al, %eax
	xorl	-36(%rbp), %eax
	movl	%eax, -44(%rbp)
	movq	$7, -16(%rbp)
	jmp	.L1101
.L1089:
	leaq	.LC90(%rip), %rax
	movq	%rax, %rcx
	movl	$1071, %edx
	leaq	.LC3(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC91(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L1083:
	movq	-24(%rbp), %rax
	movl	36(%rax), %ecx
	movq	serr(%rip), %rdx
	movq	-56(%rbp), %rax
	leaq	.LC92(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	die
	movq	$8, -16(%rbp)
	jmp	.L1101
.L1085:
	movq	-32(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	$1, -16(%rbp)
	jmp	.L1101
.L1094:
	movq	-64(%rbp), %rax
	movq	-24(%rbp), %rdx
	movq	%rdx, 48(%rax)
	movq	$3, -16(%rbp)
	jmp	.L1101
.L1095:
	movq	-64(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, -32(%rbp)
	movq	$17, -16(%rbp)
	jmp	.L1101
.L1091:
	movq	-64(%rbp), %rax
	movl	32(%rax), %eax
	movl	%eax, %edi
	call	is_op
	movl	%eax, -40(%rbp)
	movq	$0, -16(%rbp)
	jmp	.L1101
.L1099:
	cmpl	$0, -40(%rbp)
	je	.L1106
	movq	$11, -16(%rbp)
	jmp	.L1101
.L1106:
	movq	$13, -16(%rbp)
	jmp	.L1101
.L1093:
	cmpl	$0, -44(%rbp)
	je	.L1108
	movq	$5, -16(%rbp)
	jmp	.L1101
.L1108:
	movq	$14, -16(%rbp)
	jmp	.L1101
.L1111:
	nop
.L1101:
	jmp	.L1110
.L1112:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE44:
	.size	reduce, .-reduce
	.section	.rodata
.LC93:
	.string	"%s: line %d: bad hash def"
	.text
	.type	emit_hash_definition, @function
emit_hash_definition:
.LFB45:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movl	%edx, -36(%rbp)
	movq	$12, -8(%rbp)
.L1136:
	cmpq	$12, -8(%rbp)
	ja	.L1137
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L1116(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L1116(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L1116:
	.long	.L1137-.L1116
	.long	.L1137-.L1116
	.long	.L1125-.L1116
	.long	.L1137-.L1116
	.long	.L1124-.L1116
	.long	.L1138-.L1116
	.long	.L1122-.L1116
	.long	.L1121-.L1116
	.long	.L1120-.L1116
	.long	.L1119-.L1116
	.long	.L1118-.L1116
	.long	.L1117-.L1116
	.long	.L1115-.L1116
	.text
.L1124:
	movq	-32(%rbp), %rax
	movl	36(%rax), %ecx
	movq	serr(%rip), %rdx
	movq	-24(%rbp), %rax
	leaq	.LC93(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	die
	movq	$5, -8(%rbp)
	jmp	.L1126
.L1115:
	cmpq	$0, -32(%rbp)
	jne	.L1127
	movq	$4, -8(%rbp)
	jmp	.L1126
.L1127:
	movq	$8, -8(%rbp)
	jmp	.L1126
.L1120:
	movq	-32(%rbp), %rax
	movl	32(%rax), %eax
	cmpl	$44, %eax
	jne	.L1129
	movq	$7, -8(%rbp)
	jmp	.L1126
.L1129:
	movq	$6, -8(%rbp)
	jmp	.L1126
.L1117:
	movq	-32(%rbp), %rax
	movl	36(%rax), %ecx
	movq	serr(%rip), %rdx
	movq	-24(%rbp), %rax
	leaq	.LC93(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	die
	movq	$5, -8(%rbp)
	jmp	.L1126
.L1119:
	movq	-32(%rbp), %rax
	movq	40(%rax), %rax
	movl	32(%rax), %eax
	cmpl	$97, %eax
	jne	.L1131
	movq	$10, -8(%rbp)
	jmp	.L1126
.L1131:
	movq	$11, -8(%rbp)
	jmp	.L1126
.L1122:
	movq	-32(%rbp), %rax
	movl	32(%rax), %eax
	cmpl	$61, %eax
	jne	.L1133
	movq	$9, -8(%rbp)
	jmp	.L1126
.L1133:
	movq	$2, -8(%rbp)
	jmp	.L1126
.L1118:
	movq	-32(%rbp), %rax
	movq	48(%rax), %rdx
	movq	-24(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	emit_expr
	movq	-24(%rbp), %rax
	movl	$9, %esi
	movq	%rax, %rdi
	call	emit_byte
	movq	-24(%rbp), %rax
	movl	$1, %esi
	movq	%rax, %rdi
	call	emit_byte
	movq	-32(%rbp), %rax
	movq	40(%rax), %rdx
	movq	-24(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	emit_ident
	movq	-24(%rbp), %rax
	movl	$3, %esi
	movq	%rax, %rdi
	call	emit_byte
	movq	$5, -8(%rbp)
	jmp	.L1126
.L1121:
	movq	-32(%rbp), %rax
	movq	40(%rax), %rcx
	movl	-36(%rbp), %edx
	movq	-24(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	emit_hash_definition
	movq	-24(%rbp), %rax
	movl	$7, %esi
	movq	%rax, %rdi
	call	emit_byte
	movl	-36(%rbp), %eax
	leal	1(%rax), %edx
	movq	-32(%rbp), %rax
	movq	48(%rax), %rcx
	movq	-24(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	emit_hash_definition
	movq	$5, -8(%rbp)
	jmp	.L1126
.L1125:
	movq	-32(%rbp), %rdx
	movq	-24(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	emit_expr
	movq	-24(%rbp), %rax
	movl	$9, %esi
	movq	%rax, %rdi
	call	emit_byte
	movq	-24(%rbp), %rax
	movl	$1, %esi
	movq	%rax, %rdi
	call	emit_byte
	pxor	%xmm1, %xmm1
	cvtsi2sdl	-36(%rbp), %xmm1
	movq	%xmm1, %rdx
	movq	-24(%rbp), %rax
	movq	%rdx, %xmm0
	movq	%rax, %rdi
	call	emit_num
	movq	-24(%rbp), %rax
	movl	$3, %esi
	movq	%rax, %rdi
	call	emit_byte
	movq	$5, -8(%rbp)
	jmp	.L1126
.L1137:
	nop
.L1126:
	jmp	.L1136
.L1138:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE45:
	.size	emit_hash_definition, .-emit_hash_definition
	.type	print, @function
print:
.LFB46:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movq	$4, -8(%rbp)
.L1151:
	cmpq	$7, -8(%rbp)
	ja	.L1152
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L1142(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L1142(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L1142:
	.long	.L1146-.L1142
	.long	.L1152-.L1142
	.long	.L1145-.L1142
	.long	.L1152-.L1142
	.long	.L1144-.L1142
	.long	.L1153-.L1142
	.long	.L1152-.L1142
	.long	.L1141-.L1142
	.text
.L1144:
	movq	$2, -8(%rbp)
	jmp	.L1147
.L1146:
	movq	-16(%rbp), %rax
	movq	%rax, %rdi
	call	print_obj
	addl	$1, -28(%rbp)
	addq	$24, -16(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L1147
.L1141:
	movl	-28(%rbp), %eax
	cmpl	-24(%rbp), %eax
	jge	.L1149
	movq	$0, -8(%rbp)
	jmp	.L1147
.L1149:
	movq	$5, -8(%rbp)
	jmp	.L1147
.L1145:
	movq	-40(%rbp), %rax
	movl	$0, %esi
	movq	%rax, %rdi
	call	gsi
	movl	%eax, -20(%rbp)
	movl	-20(%rbp), %eax
	movl	%eax, -24(%rbp)
	movq	-40(%rbp), %rax
	movq	32(%rax), %rcx
	movq	-40(%rbp), %rax
	movl	40(%rax), %eax
	cltq
	movl	-24(%rbp), %edx
	movslq	%edx, %rsi
	subq	%rsi, %rax
	movq	%rax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	subq	$24, %rax
	addq	%rcx, %rax
	movq	%rax, -16(%rbp)
	movl	$0, -28(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L1147
.L1152:
	nop
.L1147:
	jmp	.L1151
.L1153:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE46:
	.size	print, .-print
	.type	cmp_vec, @function
cmp_vec:
.LFB47:
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
	movq	$1, -8(%rbp)
.L1170:
	cmpq	$6, -8(%rbp)
	ja	.L1171
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L1157(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L1157(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L1157:
	.long	.L1163-.L1157
	.long	.L1162-.L1157
	.long	.L1161-.L1157
	.long	.L1160-.L1157
	.long	.L1159-.L1157
	.long	.L1158-.L1157
	.long	.L1156-.L1157
	.text
.L1159:
	movq	-24(%rbp), %rax
	movl	(%rax), %edx
	movq	-32(%rbp), %rax
	movl	(%rax), %ecx
	movl	%edx, %eax
	subl	%ecx, %eax
	movl	%eax, -16(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L1164
.L1162:
	movq	-24(%rbp), %rax
	movl	(%rax), %edx
	movq	-32(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, %edx
	jge	.L1165
	movq	$6, -8(%rbp)
	jmp	.L1164
.L1165:
	movq	$0, -8(%rbp)
	jmp	.L1164
.L1160:
	movl	-16(%rbp), %eax
	jmp	.L1167
.L1156:
	movq	-24(%rbp), %rdx
	movq	-32(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	cmp_vec
	movl	%eax, -12(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L1164
.L1158:
	movl	-12(%rbp), %eax
	negl	%eax
	jmp	.L1167
.L1163:
	movq	-32(%rbp), %rax
	movl	(%rax), %eax
	movslq	%eax, %rdx
	movq	-32(%rbp), %rax
	movq	8(%rax), %rcx
	movq	-24(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	memcmp@PLT
	movl	%eax, -16(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L1164
.L1161:
	cmpl	$0, -16(%rbp)
	jne	.L1168
	movq	$4, -8(%rbp)
	jmp	.L1164
.L1168:
	movq	$3, -8(%rbp)
	jmp	.L1164
.L1171:
	nop
.L1164:
	jmp	.L1170
.L1167:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE47:
	.size	cmp_vec, .-cmp_vec
	.section	.rodata
.LC94:
	.string	"emit_string_constant"
.LC95:
	.string	"node->vec.ptr[0] == '\"'"
	.align 8
.LC96:
	.string	"node->vec.ptr[node->vec.len - 1] == '\"'"
	.align 8
.LC97:
	.string	"dst + node->vec.len < q->code + q->cs"
	.text
	.type	emit_string_constant, @function
emit_string_constant:
.LFB48:
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
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$14, -16(%rbp)
.L1206:
	cmpq	$22, -16(%rbp)
	ja	.L1209
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L1175(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L1175(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L1175:
	.long	.L1209-.L1175
	.long	.L1191-.L1175
	.long	.L1190-.L1175
	.long	.L1189-.L1175
	.long	.L1188-.L1175
	.long	.L1187-.L1175
	.long	.L1186-.L1175
	.long	.L1209-.L1175
	.long	.L1185-.L1175
	.long	.L1184-.L1175
	.long	.L1183-.L1175
	.long	.L1182-.L1175
	.long	.L1209-.L1175
	.long	.L1209-.L1175
	.long	.L1181-.L1175
	.long	.L1209-.L1175
	.long	.L1180-.L1175
	.long	.L1179-.L1175
	.long	.L1178-.L1175
	.long	.L1177-.L1175
	.long	.L1176-.L1175
	.long	.L1209-.L1175
	.long	.L1210-.L1175
	.text
.L1178:
	movq	-56(%rbp), %rax
	movq	24(%rax), %rdx
	movq	-56(%rbp), %rax
	movl	48(%rax), %eax
	movslq	%eax, %rcx
	movq	-64(%rbp), %rax
	movl	16(%rax), %eax
	cltq
	addq	%rcx, %rax
	addq	$10, %rax
	addq	%rdx, %rax
	movq	%rax, %rcx
	movq	-56(%rbp), %rax
	movq	24(%rax), %rdx
	movq	-56(%rbp), %rax
	movl	52(%rax), %eax
	cltq
	addq	%rdx, %rax
	cmpq	%rax, %rcx
	jbe	.L1192
	movq	$2, -16(%rbp)
	jmp	.L1194
.L1192:
	movq	$6, -16(%rbp)
	jmp	.L1194
.L1188:
	movq	-64(%rbp), %rax
	movl	16(%rax), %eax
	movslq	%eax, %rdx
	movq	-32(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, %rcx
	movq	-56(%rbp), %rax
	movq	24(%rax), %rdx
	movq	-56(%rbp), %rax
	movl	52(%rax), %eax
	cltq
	addq	%rdx, %rax
	cmpq	%rax, %rcx
	jnb	.L1195
	movq	$10, -16(%rbp)
	jmp	.L1194
.L1195:
	movq	$5, -16(%rbp)
	jmp	.L1194
.L1181:
	movq	$0, -40(%rbp)
	movq	$18, -16(%rbp)
	jmp	.L1194
.L1185:
	movq	-64(%rbp), %rax
	movl	36(%rax), %edx
	movq	-24(%rbp), %rax
	addq	$1, %rax
	movzbl	(%rax), %eax
	movzbl	%al, %ecx
	movq	-56(%rbp), %rax
	movl	%ecx, %esi
	movq	%rax, %rdi
	call	hex_ascii_to_int
	movl	%eax, -48(%rbp)
	movl	-48(%rbp), %eax
	sall	$4, %eax
	movl	%eax, %ecx
	movq	-40(%rbp), %rdx
	movq	-32(%rbp), %rax
	addq	%rdx, %rax
	movl	%ecx, %edx
	movb	%dl, (%rax)
	movq	-64(%rbp), %rax
	movl	36(%rax), %edx
	movq	-24(%rbp), %rax
	addq	$2, %rax
	movzbl	(%rax), %eax
	movzbl	%al, %ecx
	movq	-56(%rbp), %rax
	movl	%ecx, %esi
	movq	%rax, %rdi
	call	hex_ascii_to_int
	movl	%eax, -44(%rbp)
	movq	-40(%rbp), %rdx
	movq	-32(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %edx
	movl	-44(%rbp), %eax
	movl	%eax, %esi
	movq	-40(%rbp), %rcx
	movq	-32(%rbp), %rax
	addq	%rcx, %rax
	orl	%esi, %edx
	movb	%dl, (%rax)
	addq	$2, -24(%rbp)
	movq	$9, -16(%rbp)
	jmp	.L1194
.L1191:
	movq	-64(%rbp), %rax
	movq	24(%rax), %rdx
	movq	-64(%rbp), %rax
	movl	16(%rax), %eax
	cltq
	subq	$1, %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	cmpb	$34, %al
	jne	.L1197
	movq	$4, -16(%rbp)
	jmp	.L1194
.L1197:
	movq	$11, -16(%rbp)
	jmp	.L1194
.L1189:
	leaq	.LC94(%rip), %rax
	movq	%rax, %rcx
	movl	$275, %edx
	leaq	.LC3(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC95(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L1180:
	movq	-40(%rbp), %rdx
	movq	-32(%rbp), %rax
	addq	%rax, %rdx
	movq	-24(%rbp), %rax
	movzbl	(%rax), %eax
	movb	%al, (%rdx)
	movq	$9, -16(%rbp)
	jmp	.L1194
.L1182:
	leaq	.LC94(%rip), %rax
	movq	%rax, %rcx
	movl	$276, %edx
	leaq	.LC3(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC96(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L1184:
	addq	$1, -24(%rbp)
	movq	-40(%rbp), %rax
	addq	$1, %rax
	movq	%rax, -40(%rbp)
	movq	$10, -16(%rbp)
	jmp	.L1194
.L1177:
	movq	-56(%rbp), %rax
	movl	$5, %esi
	movq	%rax, %rdi
	call	emit_byte
	movq	-56(%rbp), %rax
	movl	$2, %esi
	movq	%rax, %rdi
	call	emit_byte
	leaq	-40(%rbp), %rcx
	movq	-56(%rbp), %rax
	movl	$8, %edx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	emit
	movq	-56(%rbp), %rax
	movl	48(%rax), %eax
	movl	%eax, %edx
	movq	-40(%rbp), %rax
	addl	%edx, %eax
	movl	%eax, %edx
	movq	-56(%rbp), %rax
	movl	%edx, 48(%rax)
	movq	$22, -16(%rbp)
	jmp	.L1194
.L1179:
	movq	-64(%rbp), %rax
	movq	24(%rax), %rax
	movzbl	(%rax), %eax
	cmpb	$34, %al
	jne	.L1199
	movq	$1, -16(%rbp)
	jmp	.L1194
.L1199:
	movq	$3, -16(%rbp)
	jmp	.L1194
.L1186:
	movq	-56(%rbp), %rax
	movq	24(%rax), %rdx
	movq	-56(%rbp), %rax
	movl	48(%rax), %eax
	cltq
	addq	$10, %rax
	addq	%rdx, %rax
	movq	%rax, -32(%rbp)
	movq	-64(%rbp), %rax
	movq	24(%rax), %rax
	addq	$1, %rax
	movq	%rax, -24(%rbp)
	movq	$17, -16(%rbp)
	jmp	.L1194
.L1187:
	leaq	.LC94(%rip), %rax
	movq	%rax, %rcx
	movl	$277, %edx
	leaq	.LC3(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC97(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L1183:
	movq	-64(%rbp), %rax
	movq	24(%rax), %rdx
	movq	-64(%rbp), %rax
	movl	16(%rax), %eax
	cltq
	subq	$1, %rax
	addq	%rdx, %rax
	movq	%rax, %rdx
	movq	-24(%rbp), %rax
	cmpq	%rax, %rdx
	jbe	.L1202
	movq	$20, -16(%rbp)
	jmp	.L1194
.L1202:
	movq	$19, -16(%rbp)
	jmp	.L1194
.L1190:
	movq	-64(%rbp), %rax
	movl	16(%rax), %eax
	addl	$11, %eax
	movl	%eax, %edx
	movq	-56(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	expand_code
	movq	$6, -16(%rbp)
	jmp	.L1194
.L1176:
	movq	-24(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$37, %al
	jne	.L1204
	movq	$8, -16(%rbp)
	jmp	.L1194
.L1204:
	movq	$16, -16(%rbp)
	jmp	.L1194
.L1209:
	nop
.L1194:
	jmp	.L1206
.L1210:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L1208
	call	__stack_chk_fail@PLT
.L1208:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE48:
	.size	emit_string_constant, .-emit_string_constant
	.type	is_lvalue, @function
is_lvalue:
.LFB49:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movq	$5, -8(%rbp)
.L1230:
	cmpq	$7, -8(%rbp)
	ja	.L1232
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L1214(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L1214(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L1214:
	.long	.L1221-.L1214
	.long	.L1220-.L1214
	.long	.L1219-.L1214
	.long	.L1218-.L1214
	.long	.L1217-.L1214
	.long	.L1216-.L1214
	.long	.L1215-.L1214
	.long	.L1213-.L1214
	.text
.L1217:
	movq	-24(%rbp), %rax
	movq	56(%rax), %rax
	movq	40(%rax), %rax
	movq	%rax, %rdx
	movq	-24(%rbp), %rax
	cmpq	%rax, %rdx
	jne	.L1222
	movq	$0, -8(%rbp)
	jmp	.L1224
.L1222:
	movq	$7, -8(%rbp)
	jmp	.L1224
.L1220:
	movl	$0, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L1224
.L1218:
	movq	-24(%rbp), %rax
	movq	56(%rax), %rax
	movl	32(%rax), %eax
	cmpl	$61, %eax
	jne	.L1225
	movq	$4, -8(%rbp)
	jmp	.L1224
.L1225:
	movq	$1, -8(%rbp)
	jmp	.L1224
.L1215:
	movl	$0, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L1224
.L1216:
	movq	-24(%rbp), %rax
	movq	56(%rax), %rax
	testq	%rax, %rax
	je	.L1227
	movq	$3, -8(%rbp)
	jmp	.L1224
.L1227:
	movq	$6, -8(%rbp)
	jmp	.L1224
.L1221:
	movl	$1, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L1224
.L1213:
	movl	$0, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L1224
.L1219:
	movl	-12(%rbp), %eax
	jmp	.L1231
.L1232:
	nop
.L1224:
	jmp	.L1230
.L1231:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE49:
	.size	is_lvalue, .-is_lvalue
	.type	init_op_tab, @function
init_op_tab:
.LFB50:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$0, -8(%rbp)
.L1239:
	cmpq	$2, -8(%rbp)
	je	.L1234
	cmpq	$2, -8(%rbp)
	ja	.L1241
	cmpq	$0, -8(%rbp)
	je	.L1236
	cmpq	$1, -8(%rbp)
	jne	.L1241
	jmp	.L1240
.L1236:
	movq	$2, -8(%rbp)
	jmp	.L1238
.L1234:
	movl	$0, %edx
	movl	$2, %esi
	movl	$100, %edi
	call	set_op_attributes
	movl	%eax, 160+op_tab(%rip)
	movl	160+op_tab(%rip), %eax
	movl	%eax, 184+op_tab(%rip)
	movl	$1, %edx
	movl	$1, %esi
	movl	$100, %edi
	call	set_op_attributes
	movl	%eax, 364+op_tab(%rip)
	movl	$0, %edx
	movl	$2, %esi
	movl	$80, %edi
	call	set_op_attributes
	movl	%eax, 148+op_tab(%rip)
	movl	148+op_tab(%rip), %eax
	movl	%eax, 188+op_tab(%rip)
	movl	188+op_tab(%rip), %eax
	movl	%eax, 168+op_tab(%rip)
	movl	$0, %edx
	movl	$2, %esi
	movl	$60, %edi
	call	set_op_attributes
	movl	%eax, 180+op_tab(%rip)
	movl	180+op_tab(%rip), %eax
	movl	%eax, 172+op_tab(%rip)
	movl	$1, %edx
	movl	$2, %esi
	movl	$20, %edi
	call	set_op_attributes
	movl	%eax, 424+op_tab(%rip)
	movl	424+op_tab(%rip), %eax
	movl	%eax, 420+op_tab(%rip)
	movl	420+op_tab(%rip), %eax
	movl	%eax, 416+op_tab(%rip)
	movl	416+op_tab(%rip), %eax
	movl	%eax, 412+op_tab(%rip)
	movl	412+op_tab(%rip), %eax
	movl	%eax, 244+op_tab(%rip)
	movl	$0, %edx
	movl	$2, %esi
	movl	$10, %edi
	call	set_op_attributes
	movl	%eax, 176+op_tab(%rip)
	movq	$1, -8(%rbp)
	jmp	.L1238
.L1241:
	nop
.L1238:
	jmp	.L1239
.L1240:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE50:
	.size	init_op_tab, .-init_op_tab
	.section	.rodata
.LC98:
	.string	"print"
	.text
	.type	import_builtin_objects, @function
import_builtin_objects:
.LFB51:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movq	%rdi, -56(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, -48(%rbp)
.L1248:
	cmpq	$2, -48(%rbp)
	je	.L1243
	cmpq	$2, -48(%rbp)
	ja	.L1252
	cmpq	$0, -48(%rbp)
	je	.L1245
	cmpq	$1, -48(%rbp)
	jne	.L1252
	jmp	.L1251
.L1245:
	movq	$2, -48(%rbp)
	jmp	.L1247
.L1243:
	movq	-56(%rbp), %rax
	movq	32(%rax), %rax
	movq	%rax, -40(%rbp)
	movb	$5, -32(%rbp)
	leaq	print(%rip), %rax
	movq	%rax, -24(%rbp)
	leaq	-32(%rbp), %rdx
	movq	-40(%rbp), %rax
	leaq	.LC98(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	setattr
	leaq	disasm(%rip), %rax
	movq	%rax, -24(%rbp)
	leaq	-32(%rbp), %rdx
	movq	-40(%rbp), %rax
	leaq	.LC60(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	setattr
	movq	$1, -48(%rbp)
	jmp	.L1247
.L1252:
	nop
.L1247:
	jmp	.L1248
.L1251:
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L1250
	call	__stack_chk_fail@PLT
.L1250:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE51:
	.size	import_builtin_objects, .-import_builtin_objects
	.type	emit_function_call, @function
emit_function_call:
.LFB52:
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
.L1259:
	cmpq	$2, -8(%rbp)
	je	.L1254
	cmpq	$2, -8(%rbp)
	ja	.L1260
	cmpq	$0, -8(%rbp)
	je	.L1261
	cmpq	$1, -8(%rbp)
	jne	.L1260
	movq	-32(%rbp), %rax
	movq	48(%rax), %rcx
	movq	-24(%rbp), %rax
	movl	$0, %edx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	emit_params
	movl	%eax, -12(%rbp)
	pxor	%xmm1, %xmm1
	cvtsi2sdl	-12(%rbp), %xmm1
	movq	%xmm1, %rdx
	movq	-24(%rbp), %rax
	movq	%rdx, %xmm0
	movq	%rax, %rdi
	call	emit_num
	movq	-32(%rbp), %rax
	movq	40(%rax), %rdx
	movq	-24(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	emit_expr
	movq	-24(%rbp), %rax
	movl	$1, %esi
	movq	%rax, %rdi
	call	emit_byte
	movq	$0, -8(%rbp)
	jmp	.L1257
.L1254:
	movq	$1, -8(%rbp)
	jmp	.L1257
.L1260:
	nop
.L1257:
	jmp	.L1259
.L1261:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE52:
	.size	emit_function_call, .-emit_function_call
	.section	.rodata
.LC99:
	.string	"usage: qq [-c <file>] file\n"
	.text
	.type	usage, @function
usage:
.LFB53:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$1, -8(%rbp)
.L1267:
	cmpq	$1, -8(%rbp)
	je	.L1263
	cmpq	$2, -8(%rbp)
	je	.L1264
	jmp	.L1266
.L1263:
	movq	$2, -8(%rbp)
	jmp	.L1266
.L1264:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$27, %edx
	movl	$1, %esi
	leaq	.LC99(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movl	$1, %edi
	call	exit@PLT
.L1266:
	jmp	.L1267
	.cfi_endproc
.LFE53:
	.size	usage, .-usage
	.type	set_op_attributes, @function
set_op_attributes:
.LFB54:
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
	movq	$0, -8(%rbp)
.L1271:
	cmpq	$0, -8(%rbp)
	jne	.L1274
	movl	-24(%rbp), %eax
	sall	$8, %eax
	orl	-20(%rbp), %eax
	movl	%eax, %edx
	movl	-28(%rbp), %eax
	sall	$16, %eax
	orl	%edx, %eax
	jmp	.L1273
.L1274:
	nop
	jmp	.L1271
.L1273:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE54:
	.size	set_op_attributes, .-set_op_attributes
	.type	emit_num, @function
emit_num:
.LFB56:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movsd	%xmm0, -32(%rbp)
	movq	$1, -8(%rbp)
.L1281:
	cmpq	$2, -8(%rbp)
	je	.L1276
	cmpq	$2, -8(%rbp)
	ja	.L1282
	cmpq	$0, -8(%rbp)
	je	.L1283
	cmpq	$1, -8(%rbp)
	jne	.L1282
	movq	$2, -8(%rbp)
	jmp	.L1279
.L1276:
	movq	-24(%rbp), %rax
	movl	$5, %esi
	movq	%rax, %rdi
	call	emit_byte
	movq	-24(%rbp), %rax
	movl	$1, %esi
	movq	%rax, %rdi
	call	emit_byte
	leaq	-32(%rbp), %rcx
	movq	-24(%rbp), %rax
	movl	$8, %edx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	emit
	movq	$0, -8(%rbp)
	jmp	.L1279
.L1282:
	nop
.L1279:
	jmp	.L1281
.L1283:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE56:
	.size	emit_num, .-emit_num
	.type	emit_params, @function
emit_params:
.LFB57:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movl	%edx, -36(%rbp)
	movq	$2, -8(%rbp)
.L1298:
	cmpq	$5, -8(%rbp)
	ja	.L1300
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L1287(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L1287(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L1287:
	.long	.L1300-.L1287
	.long	.L1291-.L1287
	.long	.L1290-.L1287
	.long	.L1289-.L1287
	.long	.L1288-.L1287
	.long	.L1286-.L1287
	.text
.L1288:
	movq	-32(%rbp), %rax
	movq	40(%rax), %rcx
	movl	-36(%rbp), %edx
	movq	-24(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	emit_params
	movl	%eax, -36(%rbp)
	movq	-32(%rbp), %rax
	movq	48(%rax), %rcx
	movl	-36(%rbp), %edx
	movq	-24(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	emit_params
	movl	%eax, -36(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L1292
.L1291:
	movq	-32(%rbp), %rax
	movl	32(%rax), %eax
	cmpl	$44, %eax
	jne	.L1293
	movq	$4, -8(%rbp)
	jmp	.L1292
.L1293:
	movq	$5, -8(%rbp)
	jmp	.L1292
.L1289:
	movl	-36(%rbp), %eax
	jmp	.L1299
.L1286:
	movq	-32(%rbp), %rdx
	movq	-24(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	emit_expr
	addl	$1, -36(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L1292
.L1290:
	movq	-32(%rbp), %rax
	movl	32(%rax), %eax
	cmpl	$107, %eax
	jne	.L1296
	movq	$3, -8(%rbp)
	jmp	.L1292
.L1296:
	movq	$1, -8(%rbp)
	jmp	.L1292
.L1300:
	nop
.L1292:
	jmp	.L1298
.L1299:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE57:
	.size	emit_params, .-emit_params
	.section	.rodata
	.align 8
.LC57:
	.long	0
	.long	1138753536
	.align 8
.LC81:
	.long	0
	.long	1072693248
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
