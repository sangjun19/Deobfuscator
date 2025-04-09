	.file	"ntrupin_l.c_l_flatten.c"
	.text
	.globl	_TIG_IZ_HOyo_argv
	.bss
	.align 8
	.type	_TIG_IZ_HOyo_argv, @object
	.size	_TIG_IZ_HOyo_argv, 8
_TIG_IZ_HOyo_argv:
	.zero	8
	.globl	_TIG_IZ_HOyo_envp
	.align 8
	.type	_TIG_IZ_HOyo_envp, @object
	.size	_TIG_IZ_HOyo_envp, 8
_TIG_IZ_HOyo_envp:
	.zero	8
	.local	node_type
	.comm	node_type,24,16
	.globl	_TIG_IZ_HOyo_argc
	.align 4
	.type	_TIG_IZ_HOyo_argc, @object
	.size	_TIG_IZ_HOyo_argc, 4
_TIG_IZ_HOyo_argc:
	.zero	4
	.text
	.globl	create_var
	.type	create_var, @function
create_var:
.LFB0:
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
	movq	$1, -40(%rbp)
.L19:
	cmpq	$11, -40(%rbp)
	ja	.L22
	movq	-40(%rbp), %rax
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
	.long	.L12-.L4
	.long	.L11-.L4
	.long	.L22-.L4
	.long	.L10-.L4
	.long	.L9-.L4
	.long	.L8-.L4
	.long	.L22-.L4
	.long	.L22-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L9:
	movq	-48(%rbp), %rax
	movl	(%rax), %eax
	movslq	%eax, %rdx
	movq	-48(%rbp), %rax
	movq	8(%rax), %rax
	movq	-56(%rbp), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	strncpy@PLT
	movq	$11, -40(%rbp)
	jmp	.L13
.L7:
	leaq	-48(%rbp), %rax
	movq	%rax, %rdi
	call	destroy_var
	movq	$0, -40(%rbp)
	jmp	.L13
.L11:
	movq	$10, -40(%rbp)
	jmp	.L13
.L10:
	movq	-48(%rbp), %rax
	testq	%rax, %rax
	jne	.L14
	movq	$8, -40(%rbp)
	jmp	.L13
.L14:
	movq	$9, -40(%rbp)
	jmp	.L13
.L3:
	movq	-48(%rbp), %rax
	jmp	.L20
.L6:
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -24(%rbp)
	movq	-48(%rbp), %rax
	movq	-24(%rbp), %rdx
	movl	%edx, (%rax)
	movq	-48(%rbp), %rax
	movl	(%rax), %eax
	addl	$1, %eax
	cltq
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -16(%rbp)
	movq	-48(%rbp), %rax
	movq	-16(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movq	$5, -40(%rbp)
	jmp	.L13
.L8:
	movq	-48(%rbp), %rax
	movq	8(%rax), %rax
	testq	%rax, %rax
	jne	.L17
	movq	$8, -40(%rbp)
	jmp	.L13
.L17:
	movq	$4, -40(%rbp)
	jmp	.L13
.L5:
	movl	$16, %edi
	call	malloc@PLT
	movq	%rax, -32(%rbp)
	movq	-32(%rbp), %rax
	movq	%rax, -48(%rbp)
	movq	$3, -40(%rbp)
	jmp	.L13
.L12:
	movl	$0, %eax
	jmp	.L20
.L22:
	nop
.L13:
	jmp	.L19
.L20:
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L21
	call	__stack_chk_fail@PLT
.L21:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	create_var, .-create_var
	.section	.rodata
.LC0:
	.string	"var"
.LC1:
	.string	"abstr"
.LC2:
	.string	"appl"
.LC3:
	.string	"not enough arguments\n"
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
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	leaq	.LC0(%rip), %rax
	movq	%rax, node_type(%rip)
	leaq	.LC1(%rip), %rax
	movq	%rax, 8+node_type(%rip)
	leaq	.LC2(%rip), %rax
	movq	%rax, 16+node_type(%rip)
	nop
.L24:
	movq	$0, _TIG_IZ_HOyo_envp(%rip)
	nop
.L25:
	movq	$0, _TIG_IZ_HOyo_argv(%rip)
	nop
.L26:
	movl	$0, _TIG_IZ_HOyo_argc(%rip)
	nop
	nop
.L27:
.L28:
#APP
# 168 "ntrupin_l.c_l.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-HOyo--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_HOyo_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_HOyo_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_HOyo_envp(%rip)
	nop
	movq	$3, -24(%rbp)
.L40:
	cmpq	$5, -24(%rbp)
	ja	.L43
	movq	-24(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L31(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L31(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L31:
	.long	.L35-.L31
	.long	.L43-.L31
	.long	.L34-.L31
	.long	.L33-.L31
	.long	.L32-.L31
	.long	.L30-.L31
	.text
.L32:
	movq	-48(%rbp), %rax
	addq	$8, %rax
	movq	%rax, %rdi
	call	parse_expr
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -32(%rbp)
	movq	-32(%rbp), %rax
	movl	$0, %esi
	movq	%rax, %rdi
	call	_debug_node
	leaq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	eval
	movq	-32(%rbp), %rax
	movl	$0, %esi
	movq	%rax, %rdi
	call	_debug_node
	leaq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	destroy_node
	movq	$2, -24(%rbp)
	jmp	.L36
.L33:
	cmpl	$1, -36(%rbp)
	jg	.L37
	movq	$5, -24(%rbp)
	jmp	.L36
.L37:
	movq	$4, -24(%rbp)
	jmp	.L36
.L30:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$21, %edx
	movl	$1, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	$0, -24(%rbp)
	jmp	.L36
.L35:
	movl	$1, %eax
	jmp	.L41
.L34:
	movl	$0, %eax
	jmp	.L41
.L43:
	nop
.L36:
	jmp	.L40
.L41:
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L42
	call	__stack_chk_fail@PLT
.L42:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	main, .-main
	.globl	destroy_node
	.type	destroy_node, @function
destroy_node:
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
	movq	$3, -8(%rbp)
.L64:
	cmpq	$12, -8(%rbp)
	ja	.L65
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L47(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L47(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L47:
	.long	.L55-.L47
	.long	.L65-.L47
	.long	.L54-.L47
	.long	.L53-.L47
	.long	.L52-.L47
	.long	.L66-.L47
	.long	.L65-.L47
	.long	.L50-.L47
	.long	.L66-.L47
	.long	.L65-.L47
	.long	.L65-.L47
	.long	.L48-.L47
	.long	.L46-.L47
	.text
.L52:
	movq	$0, -8(%rbp)
	jmp	.L56
.L46:
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	addq	$8, %rax
	movq	%rax, %rdi
	call	destroy_abstr
	movq	$0, -8(%rbp)
	jmp	.L56
.L53:
	cmpq	$0, -24(%rbp)
	jne	.L58
	movq	$8, -8(%rbp)
	jmp	.L56
.L58:
	movq	$2, -8(%rbp)
	jmp	.L56
.L48:
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	addq	$8, %rax
	movq	%rax, %rdi
	call	destroy_appl
	movq	$0, -8(%rbp)
	jmp	.L56
.L55:
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	-24(%rbp), %rax
	movq	$0, (%rax)
	movq	$5, -8(%rbp)
	jmp	.L56
.L50:
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	addq	$8, %rax
	movq	%rax, %rdi
	call	destroy_var
	movq	$0, -8(%rbp)
	jmp	.L56
.L54:
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	cmpl	$2, %eax
	je	.L60
	cmpl	$2, %eax
	jg	.L61
	testl	%eax, %eax
	je	.L62
	cmpl	$1, %eax
	jne	.L61
	movq	$12, -8(%rbp)
	jmp	.L63
.L60:
	movq	$11, -8(%rbp)
	jmp	.L63
.L62:
	movq	$7, -8(%rbp)
	jmp	.L63
.L61:
	movq	$4, -8(%rbp)
	nop
.L63:
	jmp	.L56
.L65:
	nop
.L56:
	jmp	.L64
.L66:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	destroy_node, .-destroy_node
	.globl	reduce
	.type	reduce, @function
reduce:
.LFB4:
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
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$7, -16(%rbp)
.L95:
	cmpq	$14, -16(%rbp)
	ja	.L98
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L70(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L70(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L70:
	.long	.L99-.L70
	.long	.L80-.L70
	.long	.L79-.L70
	.long	.L98-.L70
	.long	.L78-.L70
	.long	.L77-.L70
	.long	.L76-.L70
	.long	.L75-.L70
	.long	.L99-.L70
	.long	.L73-.L70
	.long	.L98-.L70
	.long	.L72-.L70
	.long	.L71-.L70
	.long	.L98-.L70
	.long	.L69-.L70
	.text
.L78:
	movq	-48(%rbp), %rax
	movl	(%rax), %eax
	movslq	%eax, %rdx
	movq	-40(%rbp), %rax
	movq	(%rax), %rax
	movq	8(%rax), %rax
	movq	8(%rax), %rcx
	movq	-48(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	strncmp@PLT
	movl	%eax, -32(%rbp)
	movq	$9, -16(%rbp)
	jmp	.L82
.L69:
	movq	-48(%rbp), %rax
	movl	(%rax), %edx
	movq	-40(%rbp), %rax
	movq	(%rax), %rax
	movq	8(%rax), %rax
	movl	(%rax), %eax
	cmpl	%eax, %edx
	jne	.L83
	movq	$5, -16(%rbp)
	jmp	.L82
.L83:
	movq	$0, -16(%rbp)
	jmp	.L82
.L71:
	movq	-48(%rbp), %rax
	movl	(%rax), %edx
	movq	-40(%rbp), %rax
	movq	(%rax), %rax
	movq	8(%rax), %rax
	movq	(%rax), %rax
	movq	8(%rax), %rax
	movl	(%rax), %eax
	cmpl	%eax, %edx
	jne	.L85
	movq	$8, -16(%rbp)
	jmp	.L82
.L85:
	movq	$11, -16(%rbp)
	jmp	.L82
.L80:
	movq	$0, -16(%rbp)
	jmp	.L82
.L72:
	movq	-40(%rbp), %rax
	movq	(%rax), %rax
	movq	8(%rax), %rax
	leaq	8(%rax), %rcx
	movq	-56(%rbp), %rdx
	movq	-48(%rbp), %rax
	movq	%rax, %rsi
	movq	%rcx, %rdi
	call	reduce
	movq	$0, -16(%rbp)
	jmp	.L82
.L73:
	cmpl	$0, -32(%rbp)
	jne	.L88
	movq	$14, -16(%rbp)
	jmp	.L82
.L88:
	movq	$0, -16(%rbp)
	jmp	.L82
.L76:
	cmpl	$0, -28(%rbp)
	jne	.L90
	movq	$12, -16(%rbp)
	jmp	.L82
.L90:
	movq	$11, -16(%rbp)
	jmp	.L82
.L77:
	movq	-40(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	-40(%rbp), %rax
	movq	-56(%rbp), %rdx
	movq	%rdx, (%rax)
	leaq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	destroy_node
	movq	$0, -16(%rbp)
	jmp	.L82
.L75:
	movq	-40(%rbp), %rax
	movq	(%rax), %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	testl	%eax, %eax
	je	.L92
	cmpl	$1, %eax
	jne	.L93
	movq	$2, -16(%rbp)
	jmp	.L94
.L92:
	movq	$4, -16(%rbp)
	jmp	.L94
.L93:
	movq	$1, -16(%rbp)
	nop
.L94:
	jmp	.L82
.L79:
	movq	-48(%rbp), %rax
	movl	(%rax), %eax
	movslq	%eax, %rdx
	movq	-40(%rbp), %rax
	movq	(%rax), %rax
	movq	8(%rax), %rax
	movq	(%rax), %rax
	movq	8(%rax), %rax
	movq	8(%rax), %rcx
	movq	-48(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	strncmp@PLT
	movl	%eax, -28(%rbp)
	movq	$6, -16(%rbp)
	jmp	.L82
.L98:
	nop
.L82:
	jmp	.L95
.L99:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L97
	call	__stack_chk_fail@PLT
.L97:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	reduce, .-reduce
	.section	.rodata
.LC4:
	.string	"create_appl"
.LC5:
	.string	"ntrupin_l.c_l.c"
.LC6:
	.string	"fn != NULL"
.LC7:
	.string	"arg != NULL"
	.text
	.globl	create_appl
	.type	create_appl, @function
create_appl:
.LFB8:
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
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$8, -24(%rbp)
.L121:
	cmpq	$12, -24(%rbp)
	ja	.L124
	movq	-24(%rbp), %rax
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
	.long	.L112-.L103
	.long	.L124-.L103
	.long	.L111-.L103
	.long	.L110-.L103
	.long	.L109-.L103
	.long	.L108-.L103
	.long	.L107-.L103
	.long	.L124-.L103
	.long	.L106-.L103
	.long	.L105-.L103
	.long	.L124-.L103
	.long	.L104-.L103
	.long	.L102-.L103
	.text
.L109:
	movq	-32(%rbp), %rax
	movq	-40(%rbp), %rdx
	movq	%rdx, (%rax)
	movq	-32(%rbp), %rax
	movq	-48(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movq	$6, -24(%rbp)
	jmp	.L113
.L102:
	movq	-32(%rbp), %rax
	testq	%rax, %rax
	jne	.L114
	movq	$5, -24(%rbp)
	jmp	.L113
.L114:
	movq	$4, -24(%rbp)
	jmp	.L113
.L106:
	cmpq	$0, -40(%rbp)
	je	.L116
	movq	$9, -24(%rbp)
	jmp	.L113
.L116:
	movq	$3, -24(%rbp)
	jmp	.L113
.L110:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rcx
	movl	$140, %edx
	leaq	.LC5(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L104:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rcx
	movl	$141, %edx
	leaq	.LC5(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L105:
	cmpq	$0, -48(%rbp)
	je	.L118
	movq	$0, -24(%rbp)
	jmp	.L113
.L118:
	movq	$11, -24(%rbp)
	jmp	.L113
.L107:
	movq	-32(%rbp), %rax
	jmp	.L122
.L108:
	leaq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	destroy_appl
	movq	$2, -24(%rbp)
	jmp	.L113
.L112:
	movl	$16, %edi
	call	malloc@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -32(%rbp)
	movq	$12, -24(%rbp)
	jmp	.L113
.L111:
	movl	$0, %eax
	jmp	.L122
.L124:
	nop
.L113:
	jmp	.L121
.L122:
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L123
	call	__stack_chk_fail@PLT
.L123:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	create_appl, .-create_appl
	.section	.rodata
.LC8:
	.string	"destroy_abstr"
.LC9:
	.string	"abstr != NULL"
	.text
	.globl	destroy_abstr
	.type	destroy_abstr, @function
destroy_abstr:
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
	movq	$5, -8(%rbp)
.L140:
	cmpq	$6, -8(%rbp)
	ja	.L141
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L128(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L128(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L128:
	.long	.L133-.L128
	.long	.L142-.L128
	.long	.L131-.L128
	.long	.L130-.L128
	.long	.L141-.L128
	.long	.L129-.L128
	.long	.L142-.L128
	.text
.L130:
	leaq	.LC8(%rip), %rax
	movq	%rax, %rcx
	movl	$102, %edx
	leaq	.LC5(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L129:
	cmpq	$0, -24(%rbp)
	je	.L135
	movq	$0, -8(%rbp)
	jmp	.L137
.L135:
	movq	$3, -8(%rbp)
	jmp	.L137
.L133:
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	jne	.L138
	movq	$6, -8(%rbp)
	jmp	.L137
.L138:
	movq	$2, -8(%rbp)
	jmp	.L137
.L131:
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	destroy_node
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	addq	$8, %rax
	movq	%rax, %rdi
	call	destroy_node
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	-24(%rbp), %rax
	movq	$0, (%rax)
	movq	$1, -8(%rbp)
	jmp	.L137
.L141:
	nop
.L137:
	jmp	.L140
.L142:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	destroy_abstr, .-destroy_abstr
	.globl	eval
	.type	eval, @function
eval:
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
	movq	$2, -8(%rbp)
.L164:
	cmpq	$9, -8(%rbp)
	ja	.L165
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L146(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L146(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L146:
	.long	.L154-.L146
	.long	.L153-.L146
	.long	.L152-.L146
	.long	.L166-.L146
	.long	.L150-.L146
	.long	.L166-.L146
	.long	.L166-.L146
	.long	.L166-.L146
	.long	.L165-.L146
	.long	.L166-.L146
	.text
.L150:
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	movq	8(%rax), %rax
	addq	$8, %rax
	movq	%rax, %rdi
	call	eval
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	movq	8(%rax), %rax
	movq	8(%rax), %rdx
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	movq	8(%rax), %rax
	movq	(%rax), %rax
	movq	8(%rax), %rax
	movq	(%rax), %rax
	movq	8(%rax), %rax
	movq	-24(%rbp), %rcx
	movq	(%rcx), %rcx
	movq	8(%rcx), %rcx
	movq	(%rcx), %rcx
	movq	8(%rcx), %rcx
	addq	$8, %rcx
	movq	%rax, %rsi
	movq	%rcx, %rdi
	call	reduce
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	movq	8(%rax), %rax
	movq	(%rax), %rax
	movq	8(%rax), %rax
	movq	8(%rax), %rdx
	movq	-24(%rbp), %rax
	movq	%rdx, (%rax)
	movq	$7, -8(%rbp)
	jmp	.L155
.L153:
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	movq	8(%rax), %rax
	movq	(%rax), %rax
	movzbl	(%rax), %eax
	cmpb	$1, %al
	je	.L156
	movq	$6, -8(%rbp)
	jmp	.L155
.L156:
	movq	$4, -8(%rbp)
	jmp	.L155
.L154:
	movq	$5, -8(%rbp)
	jmp	.L155
.L152:
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	cmpl	$2, %eax
	je	.L159
	cmpl	$2, %eax
	jg	.L160
	testl	%eax, %eax
	je	.L161
	cmpl	$1, %eax
	je	.L162
	jmp	.L160
.L159:
	movq	$1, -8(%rbp)
	jmp	.L163
.L162:
	movq	$3, -8(%rbp)
	jmp	.L163
.L161:
	movq	$9, -8(%rbp)
	jmp	.L163
.L160:
	movq	$0, -8(%rbp)
	nop
.L163:
	jmp	.L155
.L165:
	nop
.L155:
	jmp	.L164
.L166:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	eval, .-eval
	.section	.rodata
.LC10:
	.string	"expected )\n"
.LC11:
	.string	"expected .\n"
	.text
	.globl	parse_expr
	.type	parse_expr, @function
parse_expr:
.LFB12:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$160, %rsp
	movq	%rdi, -152(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$8, -112(%rbp)
.L198:
	cmpq	$24, -112(%rbp)
	ja	.L201
	movq	-112(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L170(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L170(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L170:
	.long	.L186-.L170
	.long	.L185-.L170
	.long	.L184-.L170
	.long	.L183-.L170
	.long	.L201-.L170
	.long	.L182-.L170
	.long	.L181-.L170
	.long	.L201-.L170
	.long	.L180-.L170
	.long	.L179-.L170
	.long	.L178-.L170
	.long	.L177-.L170
	.long	.L201-.L170
	.long	.L201-.L170
	.long	.L201-.L170
	.long	.L201-.L170
	.long	.L176-.L170
	.long	.L175-.L170
	.long	.L201-.L170
	.long	.L201-.L170
	.long	.L174-.L170
	.long	.L173-.L170
	.long	.L172-.L170
	.long	.L171-.L170
	.long	.L169-.L170
	.text
.L180:
	movq	$0, -136(%rbp)
	movq	$2, -112(%rbp)
	jmp	.L187
.L185:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$11, %edx
	movl	$1, %esi
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	$24, -112(%rbp)
	jmp	.L187
.L171:
	movq	-152(%rbp), %rax
	movq	%rax, %rdi
	call	parse_expr
	movq	%rax, -104(%rbp)
	movq	-104(%rbp), %rax
	movq	%rax, -96(%rbp)
	movq	-128(%rbp), %rax
	movq	%rax, -48(%rbp)
	movq	-96(%rbp), %rax
	movq	%rax, -40(%rbp)
	leaq	-48(%rbp), %rax
	movl	$2, %edx
	movq	%rax, %rsi
	movl	$2, %edi
	call	create_node
	movq	%rax, -136(%rbp)
	movq	$6, -112(%rbp)
	jmp	.L187
.L183:
	movq	-152(%rbp), %rax
	movq	%rax, %rdi
	call	parse_expr
	movq	%rax, -72(%rbp)
	movq	-72(%rbp), %rax
	movq	%rax, -64(%rbp)
	movq	-120(%rbp), %rax
	movq	%rax, -32(%rbp)
	movq	-64(%rbp), %rax
	movq	%rax, -24(%rbp)
	leaq	-32(%rbp), %rax
	movl	$2, %edx
	movq	%rax, %rsi
	movl	$1, %edi
	call	create_node
	movq	%rax, -136(%rbp)
	movq	$6, -112(%rbp)
	jmp	.L187
.L176:
	movq	-152(%rbp), %rax
	movq	(%rax), %rax
	movzbl	(%rax), %eax
	cmpb	$46, %al
	jne	.L188
	movq	$9, -112(%rbp)
	jmp	.L187
.L188:
	movq	$17, -112(%rbp)
	jmp	.L187
.L169:
	movl	$0, %eax
	jmp	.L199
.L173:
	movl	$0, %eax
	jmp	.L199
.L177:
	movq	-152(%rbp), %rax
	movq	(%rax), %rax
	movzbl	(%rax), %eax
	movb	%al, -10(%rbp)
	movb	$0, -9(%rbp)
	movq	-152(%rbp), %rax
	movq	(%rax), %rax
	leaq	1(%rax), %rdx
	movq	-152(%rbp), %rax
	movq	%rdx, (%rax)
	leaq	-10(%rbp), %rax
	movq	%rax, -56(%rbp)
	leaq	-56(%rbp), %rax
	movl	$1, %edx
	movq	%rax, %rsi
	movl	$0, %edi
	call	create_node
	movq	%rax, -136(%rbp)
	movq	$6, -112(%rbp)
	jmp	.L187
.L179:
	movq	-152(%rbp), %rax
	movq	(%rax), %rax
	leaq	1(%rax), %rdx
	movq	-152(%rbp), %rax
	movq	%rdx, (%rax)
	movq	$3, -112(%rbp)
	jmp	.L187
.L175:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$11, %edx
	movl	$1, %esi
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	$21, -112(%rbp)
	jmp	.L187
.L181:
	movq	-136(%rbp), %rax
	jmp	.L199
.L172:
	movq	-152(%rbp), %rax
	movq	(%rax), %rax
	leaq	1(%rax), %rdx
	movq	-152(%rbp), %rax
	movq	%rdx, (%rax)
	movq	-152(%rbp), %rax
	movq	%rax, %rdi
	call	parse_expr
	movq	%rax, -88(%rbp)
	movq	-88(%rbp), %rax
	movq	%rax, -120(%rbp)
	movq	$16, -112(%rbp)
	jmp	.L187
.L182:
	movq	-152(%rbp), %rax
	movq	(%rax), %rax
	movzbl	(%rax), %eax
	cmpb	$41, %al
	jne	.L191
	movq	$20, -112(%rbp)
	jmp	.L187
.L191:
	movq	$1, -112(%rbp)
	jmp	.L187
.L178:
	movq	-152(%rbp), %rax
	movq	(%rax), %rax
	leaq	1(%rax), %rdx
	movq	-152(%rbp), %rax
	movq	%rdx, (%rax)
	movq	-152(%rbp), %rax
	movq	%rax, %rdi
	call	parse_expr
	movq	%rax, -80(%rbp)
	movq	-80(%rbp), %rax
	movq	%rax, -128(%rbp)
	movq	$5, -112(%rbp)
	jmp	.L187
.L186:
	movl	$0, %eax
	jmp	.L199
.L184:
	movq	-152(%rbp), %rax
	movq	(%rax), %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	cmpl	$92, %eax
	je	.L193
	cmpl	$92, %eax
	jg	.L194
	testl	%eax, %eax
	je	.L195
	cmpl	$40, %eax
	je	.L196
	jmp	.L194
.L195:
	movq	$0, -112(%rbp)
	jmp	.L197
.L193:
	movq	$22, -112(%rbp)
	jmp	.L197
.L196:
	movq	$10, -112(%rbp)
	jmp	.L197
.L194:
	movq	$11, -112(%rbp)
	nop
.L197:
	jmp	.L187
.L174:
	movq	-152(%rbp), %rax
	movq	(%rax), %rax
	leaq	1(%rax), %rdx
	movq	-152(%rbp), %rax
	movq	%rdx, (%rax)
	movq	$23, -112(%rbp)
	jmp	.L187
.L201:
	nop
.L187:
	jmp	.L198
.L199:
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L200
	call	__stack_chk_fail@PLT
.L200:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE12:
	.size	parse_expr, .-parse_expr
	.section	.rodata
.LC12:
	.string	"destroy_appl"
.LC13:
	.string	"appl != NULL"
	.text
	.globl	destroy_appl
	.type	destroy_appl, @function
destroy_appl:
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
	movq	$2, -8(%rbp)
.L217:
	cmpq	$6, -8(%rbp)
	ja	.L218
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L205(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L205(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L205:
	.long	.L219-.L205
	.long	.L219-.L205
	.long	.L208-.L205
	.long	.L218-.L205
	.long	.L207-.L205
	.long	.L206-.L205
	.long	.L204-.L205
	.text
.L207:
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	destroy_node
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	addq	$8, %rax
	movq	%rax, %rdi
	call	destroy_node
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	-24(%rbp), %rax
	movq	$0, (%rax)
	movq	$0, -8(%rbp)
	jmp	.L211
.L204:
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	jne	.L213
	movq	$1, -8(%rbp)
	jmp	.L211
.L213:
	movq	$4, -8(%rbp)
	jmp	.L211
.L206:
	leaq	.LC12(%rip), %rax
	movq	%rax, %rcx
	movl	$128, %edx
	leaq	.LC5(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L208:
	cmpq	$0, -24(%rbp)
	je	.L215
	movq	$6, -8(%rbp)
	jmp	.L211
.L215:
	movq	$5, -8(%rbp)
	jmp	.L211
.L218:
	nop
.L211:
	jmp	.L217
.L219:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE13:
	.size	destroy_appl, .-destroy_appl
	.globl	create_node
	.type	create_node, @function
create_node:
.LFB15:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$56, %rsp
	.cfi_offset 3, -24
	movl	%edi, %eax
	movq	%rsi, -64(%rbp)
	movl	%edx, -56(%rbp)
	movb	%al, -52(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -24(%rbp)
	xorl	%eax, %eax
	movq	$16, -40(%rbp)
.L261:
	cmpq	$27, -40(%rbp)
	ja	.L264
	movq	-40(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L223(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L223(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L223:
	.long	.L240-.L223
	.long	.L264-.L223
	.long	.L239-.L223
	.long	.L264-.L223
	.long	.L238-.L223
	.long	.L264-.L223
	.long	.L237-.L223
	.long	.L236-.L223
	.long	.L235-.L223
	.long	.L264-.L223
	.long	.L264-.L223
	.long	.L264-.L223
	.long	.L234-.L223
	.long	.L233-.L223
	.long	.L264-.L223
	.long	.L232-.L223
	.long	.L231-.L223
	.long	.L230-.L223
	.long	.L264-.L223
	.long	.L229-.L223
	.long	.L228-.L223
	.long	.L227-.L223
	.long	.L226-.L223
	.long	.L225-.L223
	.long	.L264-.L223
	.long	.L264-.L223
	.long	.L224-.L223
	.long	.L222-.L223
	.text
.L238:
	movq	-48(%rbp), %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	cmpl	$2, %eax
	je	.L241
	cmpl	$2, %eax
	jg	.L242
	testl	%eax, %eax
	je	.L243
	cmpl	$1, %eax
	jne	.L242
	movq	$15, -40(%rbp)
	jmp	.L244
.L241:
	movq	$6, -40(%rbp)
	jmp	.L244
.L243:
	movq	$17, -40(%rbp)
	jmp	.L244
.L242:
	movq	$8, -40(%rbp)
	nop
.L244:
	jmp	.L245
.L232:
	cmpl	$2, -56(%rbp)
	je	.L246
	movq	$2, -40(%rbp)
	jmp	.L245
.L246:
	movq	$13, -40(%rbp)
	jmp	.L245
.L234:
	movq	-48(%rbp), %rax
	movq	8(%rax), %rax
	testq	%rax, %rax
	jne	.L248
	movq	$2, -40(%rbp)
	jmp	.L245
.L248:
	movq	$21, -40(%rbp)
	jmp	.L245
.L235:
	movq	$21, -40(%rbp)
	jmp	.L245
.L225:
	movl	$0, %eax
	jmp	.L262
.L231:
	movq	$27, -40(%rbp)
	jmp	.L245
.L227:
	movq	-48(%rbp), %rax
	jmp	.L262
.L224:
	movq	-48(%rbp), %rax
	movzbl	-52(%rbp), %edx
	movb	%dl, (%rax)
	movq	$4, -40(%rbp)
	jmp	.L245
.L233:
	movq	-64(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rdx
	movq	-64(%rbp), %rax
	movq	(%rax), %rax
	movq	-48(%rbp), %rbx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	create_abstr
	movq	%rax, 8(%rbx)
	movq	$12, -40(%rbp)
	jmp	.L245
.L229:
	movq	-48(%rbp), %rax
	testq	%rax, %rax
	jne	.L251
	movq	$2, -40(%rbp)
	jmp	.L245
.L251:
	movq	$26, -40(%rbp)
	jmp	.L245
.L230:
	cmpl	$1, -56(%rbp)
	je	.L253
	movq	$2, -40(%rbp)
	jmp	.L245
.L253:
	movq	$22, -40(%rbp)
	jmp	.L245
.L237:
	cmpl	$2, -56(%rbp)
	je	.L255
	movq	$2, -40(%rbp)
	jmp	.L245
.L255:
	movq	$20, -40(%rbp)
	jmp	.L245
.L222:
	movl	$16, %edi
	call	malloc@PLT
	movq	%rax, -32(%rbp)
	movq	-32(%rbp), %rax
	movq	%rax, -48(%rbp)
	movq	$19, -40(%rbp)
	jmp	.L245
.L226:
	movq	-64(%rbp), %rax
	movq	(%rax), %rax
	movq	-48(%rbp), %rbx
	movq	%rax, %rdi
	call	create_var
	movq	%rax, 8(%rbx)
	movq	$7, -40(%rbp)
	jmp	.L245
.L240:
	movq	-48(%rbp), %rax
	movq	8(%rax), %rax
	testq	%rax, %rax
	jne	.L257
	movq	$2, -40(%rbp)
	jmp	.L245
.L257:
	movq	$21, -40(%rbp)
	jmp	.L245
.L236:
	movq	-48(%rbp), %rax
	movq	8(%rax), %rax
	testq	%rax, %rax
	jne	.L259
	movq	$2, -40(%rbp)
	jmp	.L245
.L259:
	movq	$21, -40(%rbp)
	jmp	.L245
.L239:
	leaq	-48(%rbp), %rax
	movq	%rax, %rdi
	call	destroy_node
	movq	$23, -40(%rbp)
	jmp	.L245
.L228:
	movq	-64(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rdx
	movq	-64(%rbp), %rax
	movq	(%rax), %rax
	movq	-48(%rbp), %rbx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	create_appl
	movq	%rax, 8(%rbx)
	movq	$0, -40(%rbp)
	jmp	.L245
.L264:
	nop
.L245:
	jmp	.L261
.L262:
	movq	-24(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L263
	call	__stack_chk_fail@PLT
.L263:
	movq	-8(%rbp), %rbx
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE15:
	.size	create_node, .-create_node
	.section	.rodata
.LC14:
	.string	"destroy_var"
.LC15:
	.string	"var != NULL"
	.text
	.globl	destroy_var
	.type	destroy_var, @function
destroy_var:
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
	movq	$9, -8(%rbp)
.L284:
	cmpq	$9, -8(%rbp)
	ja	.L285
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L268(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L268(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L268:
	.long	.L275-.L268
	.long	.L285-.L268
	.long	.L274-.L268
	.long	.L273-.L268
	.long	.L286-.L268
	.long	.L271-.L268
	.long	.L270-.L268
	.long	.L286-.L268
	.long	.L285-.L268
	.long	.L267-.L268
	.text
.L273:
	leaq	.LC14(%rip), %rax
	movq	%rax, %rcx
	movl	$71, %edx
	leaq	.LC5(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L267:
	cmpq	$0, -24(%rbp)
	je	.L277
	movq	$2, -8(%rbp)
	jmp	.L279
.L277:
	movq	$3, -8(%rbp)
	jmp	.L279
.L270:
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	movq	8(%rax), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	movq	$0, 8(%rax)
	movq	$0, -8(%rbp)
	jmp	.L279
.L271:
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	movq	8(%rax), %rax
	testq	%rax, %rax
	jne	.L280
	movq	$6, -8(%rbp)
	jmp	.L279
.L280:
	movq	$0, -8(%rbp)
	jmp	.L279
.L275:
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	-24(%rbp), %rax
	movq	$0, (%rax)
	movq	$7, -8(%rbp)
	jmp	.L279
.L274:
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	jne	.L282
	movq	$4, -8(%rbp)
	jmp	.L279
.L282:
	movq	$5, -8(%rbp)
	jmp	.L279
.L285:
	nop
.L279:
	jmp	.L284
.L286:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE16:
	.size	destroy_var, .-destroy_var
	.globl	create_abstr
	.type	create_abstr, @function
create_abstr:
.LFB17:
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
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$7, -24(%rbp)
.L301:
	cmpq	$8, -24(%rbp)
	ja	.L304
	movq	-24(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L290(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L290(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L290:
	.long	.L296-.L290
	.long	.L295-.L290
	.long	.L294-.L290
	.long	.L304-.L290
	.long	.L293-.L290
	.long	.L292-.L290
	.long	.L304-.L290
	.long	.L291-.L290
	.long	.L289-.L290
	.text
.L293:
	movl	$16, %edi
	call	malloc@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -32(%rbp)
	movq	$8, -24(%rbp)
	jmp	.L297
.L289:
	movq	-32(%rbp), %rax
	testq	%rax, %rax
	jne	.L298
	movq	$5, -24(%rbp)
	jmp	.L297
.L298:
	movq	$2, -24(%rbp)
	jmp	.L297
.L295:
	movl	$0, %eax
	jmp	.L302
.L292:
	leaq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	destroy_abstr
	movq	$1, -24(%rbp)
	jmp	.L297
.L296:
	movq	-32(%rbp), %rax
	jmp	.L302
.L291:
	movq	$4, -24(%rbp)
	jmp	.L297
.L294:
	movq	-32(%rbp), %rax
	movq	-40(%rbp), %rdx
	movq	%rdx, (%rax)
	movq	-32(%rbp), %rax
	movq	-48(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movq	$0, -24(%rbp)
	jmp	.L297
.L304:
	nop
.L297:
	jmp	.L301
.L302:
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L303
	call	__stack_chk_fail@PLT
.L303:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE17:
	.size	create_abstr, .-create_abstr
	.section	.rodata
.LC16:
	.string	"%*c| %s"
.LC17:
	.string	" '%*s'\n"
	.text
	.globl	_debug_node
	.type	_debug_node, @function
_debug_node:
.LFB18:
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
	movq	$11, -8(%rbp)
.L326:
	cmpq	$13, -8(%rbp)
	ja	.L327
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L308(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L308(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L308:
	.long	.L327-.L308
	.long	.L328-.L308
	.long	.L315-.L308
	.long	.L327-.L308
	.long	.L327-.L308
	.long	.L327-.L308
	.long	.L314-.L308
	.long	.L313-.L308
	.long	.L312-.L308
	.long	.L327-.L308
	.long	.L311-.L308
	.long	.L310-.L308
	.long	.L328-.L308
	.long	.L307-.L308
	.text
.L312:
	movl	$10, %edi
	call	putchar@PLT
	movl	-28(%rbp), %eax
	leal	1(%rax), %edx
	movq	-24(%rbp), %rax
	movq	8(%rax), %rax
	movq	(%rax), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	_debug_node
	movl	-28(%rbp), %eax
	leal	1(%rax), %edx
	movq	-24(%rbp), %rax
	movq	8(%rax), %rax
	movq	8(%rax), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	_debug_node
	movq	$1, -8(%rbp)
	jmp	.L318
.L310:
	cmpq	$0, -24(%rbp)
	jne	.L319
	movq	$12, -8(%rbp)
	jmp	.L318
.L319:
	movq	$7, -8(%rbp)
	jmp	.L318
.L307:
	movl	$10, %edi
	call	putchar@PLT
	movl	-28(%rbp), %eax
	leal	1(%rax), %edx
	movq	-24(%rbp), %rax
	movq	8(%rax), %rax
	movq	(%rax), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	_debug_node
	movl	-28(%rbp), %eax
	leal	1(%rax), %edx
	movq	-24(%rbp), %rax
	movq	8(%rax), %rax
	movq	8(%rax), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	_debug_node
	movq	$1, -8(%rbp)
	jmp	.L318
.L314:
	movq	-24(%rbp), %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	cmpl	$2, %eax
	je	.L321
	cmpl	$2, %eax
	jg	.L322
	testl	%eax, %eax
	je	.L323
	cmpl	$1, %eax
	je	.L324
	jmp	.L322
.L321:
	movq	$13, -8(%rbp)
	jmp	.L325
.L324:
	movq	$8, -8(%rbp)
	jmp	.L325
.L323:
	movq	$2, -8(%rbp)
	jmp	.L325
.L322:
	movq	$10, -8(%rbp)
	nop
.L325:
	jmp	.L318
.L311:
	movq	$1, -8(%rbp)
	jmp	.L318
.L313:
	movq	-24(%rbp), %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	cltq
	leaq	0(,%rax,8), %rdx
	leaq	node_type(%rip), %rax
	movq	(%rdx,%rax), %rax
	movl	-28(%rbp), %edx
	leal	0(,%rdx,4), %esi
	movq	%rax, %rcx
	movl	$32, %edx
	leaq	.LC16(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$6, -8(%rbp)
	jmp	.L318
.L315:
	movq	-24(%rbp), %rax
	movq	8(%rax), %rax
	movq	8(%rax), %rdx
	movq	-24(%rbp), %rax
	movq	8(%rax), %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC17(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -8(%rbp)
	jmp	.L318
.L327:
	nop
.L318:
	jmp	.L326
.L328:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE18:
	.size	_debug_node, .-_debug_node
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
