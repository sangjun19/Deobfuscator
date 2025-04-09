	.file	"scs_uclinux_subst_flatten.c"
	.text
	.local	replace_name
	.comm	replace_name,128,32
	.globl	_TIG_IZ_LFXr_argc
	.bss
	.align 4
	.type	_TIG_IZ_LFXr_argc, @object
	.size	_TIG_IZ_LFXr_argc, 4
_TIG_IZ_LFXr_argc:
	.zero	4
	.globl	_TIG_IZ_LFXr_argv
	.align 8
	.type	_TIG_IZ_LFXr_argv, @object
	.size	_TIG_IZ_LFXr_argv, 8
_TIG_IZ_LFXr_argv:
	.zero	8
	.globl	subst_table
	.align 8
	.type	subst_table, @object
	.size	subst_table, 8
subst_table:
	.zero	8
	.globl	_TIG_IZ_LFXr_envp
	.align 8
	.type	_TIG_IZ_LFXr_envp, @object
	.size	_TIG_IZ_LFXr_envp, 8
_TIG_IZ_LFXr_envp:
	.zero	8
	.text
	.type	replace_string, @function
replace_string:
.LFB1:
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
	movq	$1, -32(%rbp)
.L18:
	cmpq	$9, -32(%rbp)
	ja	.L19
	movq	-32(%rbp), %rax
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
	.long	.L11-.L4
	.long	.L10-.L4
	.long	.L19-.L4
	.long	.L9-.L4
	.long	.L20-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L9:
	cmpl	$0, -40(%rbp)
	jne	.L12
	movq	$2, -32(%rbp)
	jmp	.L14
.L12:
	movq	$6, -32(%rbp)
	jmp	.L14
.L5:
	movq	-72(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movl	%eax, -40(%rbp)
	movq	-64(%rbp), %rax
	subq	-56(%rbp), %rax
	movl	%eax, -36(%rbp)
	movq	$4, -32(%rbp)
	jmp	.L14
.L11:
	movq	$8, -32(%rbp)
	jmp	.L14
.L3:
	movq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	leaq	1(%rax), %rdx
	movl	-40(%rbp), %eax
	subl	-36(%rbp), %eax
	cltq
	leaq	-1(%rax), %rcx
	movq	-64(%rbp), %rax
	addq	%rax, %rcx
	movq	-64(%rbp), %rax
	movq	%rax, %rsi
	movq	%rcx, %rdi
	call	memmove@PLT
	movq	$7, -32(%rbp)
	jmp	.L14
.L7:
	movl	-36(%rbp), %eax
	addl	$1, %eax
	cmpl	%eax, -40(%rbp)
	je	.L15
	movq	$9, -32(%rbp)
	jmp	.L14
.L15:
	movq	$7, -32(%rbp)
	jmp	.L14
.L6:
	movl	-40(%rbp), %eax
	movslq	%eax, %rdx
	movq	-72(%rbp), %rcx
	movq	-56(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	memcpy@PLT
	movq	$5, -32(%rbp)
	jmp	.L14
.L10:
	movq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	leaq	1(%rax), %rdx
	movq	-64(%rbp), %rax
	leaq	1(%rax), %rcx
	movq	-56(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	memmove@PLT
	movq	$7, -32(%rbp)
	jmp	.L14
.L19:
	nop
.L14:
	jmp	.L18
.L20:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	replace_string, .-replace_string
	.type	parse_config_file, @function
parse_config_file:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$2160, %rsp
	movq	%rdi, -2152(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$36, -2080(%rbp)
.L94:
	movq	-2080(%rbp), %rax
	subq	$4, %rax
	cmpq	$52, %rax
	ja	.L97
	leaq	0(,%rax,4), %rdx
	leaq	.L24(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L24(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L24:
	.long	.L61-.L24
	.long	.L60-.L24
	.long	.L97-.L24
	.long	.L97-.L24
	.long	.L97-.L24
	.long	.L59-.L24
	.long	.L58-.L24
	.long	.L57-.L24
	.long	.L56-.L24
	.long	.L97-.L24
	.long	.L55-.L24
	.long	.L54-.L24
	.long	.L53-.L24
	.long	.L97-.L24
	.long	.L52-.L24
	.long	.L97-.L24
	.long	.L51-.L24
	.long	.L97-.L24
	.long	.L97-.L24
	.long	.L97-.L24
	.long	.L50-.L24
	.long	.L97-.L24
	.long	.L49-.L24
	.long	.L97-.L24
	.long	.L97-.L24
	.long	.L48-.L24
	.long	.L47-.L24
	.long	.L46-.L24
	.long	.L45-.L24
	.long	.L44-.L24
	.long	.L98-.L24
	.long	.L42-.L24
	.long	.L41-.L24
	.long	.L40-.L24
	.long	.L39-.L24
	.long	.L97-.L24
	.long	.L38-.L24
	.long	.L37-.L24
	.long	.L36-.L24
	.long	.L35-.L24
	.long	.L97-.L24
	.long	.L34-.L24
	.long	.L33-.L24
	.long	.L32-.L24
	.long	.L31-.L24
	.long	.L30-.L24
	.long	.L29-.L24
	.long	.L28-.L24
	.long	.L27-.L24
	.long	.L26-.L24
	.long	.L25-.L24
	.long	.L97-.L24
	.long	.L23-.L24
	.text
.L52:
	movq	-2088(%rbp), %rax
	movq	(%rax), %rdx
	movq	-2128(%rbp), %rax
	movzbl	(%rax), %eax
	movsbq	%al, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$8192, %eax
	testl	%eax, %eax
	je	.L62
	movq	$33, -2080(%rbp)
	jmp	.L64
.L62:
	movq	$12, -2080(%rbp)
	jmp	.L64
.L29:
	movq	-2128(%rbp), %rax
	movb	$0, (%rax)
	movq	$31, -2080(%rbp)
	jmp	.L64
.L30:
	cmpq	$0, -2128(%rbp)
	je	.L65
	movq	$50, -2080(%rbp)
	jmp	.L64
.L65:
	movq	$31, -2080(%rbp)
	jmp	.L64
.L27:
	call	__ctype_b_loc@PLT
	movq	%rax, -2088(%rbp)
	movq	$18, -2080(%rbp)
	jmp	.L64
.L61:
	movq	-2120(%rbp), %rax
	movq	%rax, -2128(%rbp)
	movq	$56, -2080(%rbp)
	jmp	.L64
.L47:
	movq	-2128(%rbp), %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	je	.L67
	movq	$41, -2080(%rbp)
	jmp	.L64
.L67:
	movq	$51, -2080(%rbp)
	jmp	.L64
.L55:
	movq	-2128(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$9, %al
	jne	.L69
	movq	$38, -2080(%rbp)
	jmp	.L64
.L69:
	movq	$16, -2080(%rbp)
	jmp	.L64
.L54:
	movq	-2104(%rbp), %rax
	movq	(%rax), %rdx
	movq	-2128(%rbp), %rax
	movzbl	(%rax), %eax
	movsbq	%al, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$8192, %eax
	testl	%eax, %eax
	je	.L71
	movq	$43, -2080(%rbp)
	jmp	.L64
.L71:
	movq	$51, -2080(%rbp)
	jmp	.L64
.L23:
	movq	-2128(%rbp), %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	je	.L73
	movq	$11, -2080(%rbp)
	jmp	.L64
.L73:
	movq	$20, -2080(%rbp)
	jmp	.L64
.L46:
	leaq	-2064(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -2072(%rbp)
	movq	-2072(%rbp), %rax
	leaq	-1(%rax), %rdx
	leaq	-2064(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, -2128(%rbp)
	movq	$24, -2080(%rbp)
	jmp	.L64
.L56:
	movq	-2128(%rbp), %rdx
	movq	-2120(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	add_subst
	movq	$45, -2080(%rbp)
	jmp	.L64
.L34:
	movq	-2152(%rbp), %rax
	movq	%rax, %rdi
	call	feof@PLT
	movl	%eax, -2132(%rbp)
	movq	$46, -2080(%rbp)
	jmp	.L64
.L25:
	movq	-2128(%rbp), %rax
	movb	$0, (%rax)
	movq	$42, -2080(%rbp)
	jmp	.L64
.L53:
	leaq	-2064(%rbp), %rax
	movq	%rax, -2128(%rbp)
	movq	$30, -2080(%rbp)
	jmp	.L64
.L50:
	leaq	-2064(%rbp), %rdx
	movq	-2128(%rbp), %rax
	cmpq	%rax, %rdx
	ja	.L75
	movq	$9, -2080(%rbp)
	jmp	.L64
.L75:
	movq	$16, -2080(%rbp)
	jmp	.L64
.L41:
	movq	$45, -2080(%rbp)
	jmp	.L64
.L49:
	cmpq	$0, -2128(%rbp)
	je	.L77
	movq	$54, -2080(%rbp)
	jmp	.L64
.L77:
	movq	$42, -2080(%rbp)
	jmp	.L64
.L57:
	call	__ctype_b_loc@PLT
	movq	%rax, -2096(%rbp)
	movq	$10, -2080(%rbp)
	jmp	.L64
.L59:
	movq	-2128(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$32, %al
	jne	.L79
	movq	$32, -2080(%rbp)
	jmp	.L64
.L79:
	movq	$14, -2080(%rbp)
	jmp	.L64
.L28:
	movq	-2128(%rbp), %rax
	movq	%rax, -2120(%rbp)
	movq	$29, -2080(%rbp)
	jmp	.L64
.L45:
	movq	-2128(%rbp), %rax
	movb	$0, (%rax)
	movq	$48, -2080(%rbp)
	jmp	.L64
.L38:
	cmpq	$0, -2112(%rbp)
	jne	.L81
	movq	$34, -2080(%rbp)
	jmp	.L64
.L81:
	movq	$5, -2080(%rbp)
	jmp	.L64
.L39:
	movq	-2128(%rbp), %rax
	movb	$0, (%rax)
	movq	$48, -2080(%rbp)
	jmp	.L64
.L31:
	subq	$1, -2128(%rbp)
	movq	$24, -2080(%rbp)
	jmp	.L64
.L26:
	movq	-2120(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$64, %al
	jne	.L84
	movq	$45, -2080(%rbp)
	jmp	.L64
.L84:
	movq	$4, -2080(%rbp)
	jmp	.L64
.L32:
	addq	$1, -2128(%rbp)
	movq	$56, -2080(%rbp)
	jmp	.L64
.L60:
	leaq	-2064(%rbp), %rax
	movl	$10, %esi
	movq	%rax, %rdi
	call	strchr@PLT
	movq	%rax, -2128(%rbp)
	movq	$26, -2080(%rbp)
	jmp	.L64
.L44:
	addq	$1, -2128(%rbp)
	movq	$35, -2080(%rbp)
	jmp	.L64
.L40:
	leaq	-2064(%rbp), %rax
	movl	$2048, %edx
	movl	$0, %esi
	movq	%rax, %rdi
	call	memset@PLT
	movq	-2152(%rbp), %rdx
	leaq	-2064(%rbp), %rax
	movl	$2048, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	movq	%rax, -2112(%rbp)
	movq	$40, -2080(%rbp)
	jmp	.L64
.L37:
	call	__ctype_b_loc@PLT
	movq	%rax, -2104(%rbp)
	movq	$15, -2080(%rbp)
	jmp	.L64
.L58:
	movq	-2096(%rbp), %rax
	movq	(%rax), %rdx
	movq	-2128(%rbp), %rax
	movzbl	(%rax), %eax
	movsbq	%al, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	movzwl	(%rax), %eax
	movzwl	%ax, %eax
	andl	$8192, %eax
	testl	%eax, %eax
	je	.L86
	movq	$20, -2080(%rbp)
	jmp	.L64
.L86:
	movq	$47, -2080(%rbp)
	jmp	.L64
.L36:
	leaq	-2064(%rbp), %rax
	movl	$35, %esi
	movq	%rax, %rdi
	call	strchr@PLT
	movq	%rax, -2128(%rbp)
	movq	$49, -2080(%rbp)
	jmp	.L64
.L33:
	cmpl	$0, -2132(%rbp)
	je	.L88
	movq	$34, -2080(%rbp)
	jmp	.L64
.L88:
	movq	$37, -2080(%rbp)
	jmp	.L64
.L42:
	movq	-2128(%rbp), %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	je	.L90
	movq	$52, -2080(%rbp)
	jmp	.L64
.L90:
	movq	$12, -2080(%rbp)
	jmp	.L64
.L48:
	movq	-2120(%rbp), %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	jne	.L92
	movq	$45, -2080(%rbp)
	jmp	.L64
.L92:
	movq	$53, -2080(%rbp)
	jmp	.L64
.L35:
	addq	$1, -2128(%rbp)
	movq	$30, -2080(%rbp)
	jmp	.L64
.L51:
	movq	-2128(%rbp), %rax
	movb	$0, (%rax)
	addq	$1, -2128(%rbp)
	movq	$35, -2080(%rbp)
	jmp	.L64
.L97:
	nop
.L64:
	jmp	.L94
.L98:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L96
	call	__stack_chk_fail@PLT
.L96:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	parse_config_file, .-parse_config_file
	.section	.rodata
.LC0:
	.string	"utime"
.LC1:
	.string	"Memory error!  Exiting.\n"
.LC2:
	.string	"f:tv"
.LC3:
	.string	"r"
.LC4:
	.string	"w"
.LC5:
	.string	"Creating or replacing %s.\n"
.LC6:
	.string	"%s: [-f config-file] [file]\n"
.LC7:
	.string	"No change, keeping %s.\n"
.LC8:
	.string	"Updating modtime for %s\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$2512, %rsp
	movl	%edi, -2484(%rbp)
	movq	%rsi, -2496(%rbp)
	movq	%rdx, -2504(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movl	$0, -2468(%rbp)
	jmp	.L100
.L101:
	movl	-2468(%rbp), %eax
	cltq
	leaq	replace_name(%rip), %rdx
	movb	$0, (%rax,%rdx)
	addl	$1, -2468(%rbp)
.L100:
	cmpl	$127, -2468(%rbp)
	jle	.L101
	nop
.L102:
	movq	$0, subst_table(%rip)
	nop
.L103:
	movq	$0, _TIG_IZ_LFXr_envp(%rip)
	nop
.L104:
	movq	$0, _TIG_IZ_LFXr_argv(%rip)
	nop
.L105:
	movl	$0, _TIG_IZ_LFXr_argc(%rip)
	nop
	nop
.L106:
.L107:
#APP
# 137 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-LFXr--0
# 0 "" 2
#NO_APP
	movl	-2484(%rbp), %eax
	movl	%eax, _TIG_IZ_LFXr_argc(%rip)
	movq	-2496(%rbp), %rax
	movq	%rax, _TIG_IZ_LFXr_argv(%rip)
	movq	-2504(%rbp), %rax
	movq	%rax, _TIG_IZ_LFXr_envp(%rip)
	nop
	movq	$27, -2392(%rbp)
.L206:
	cmpq	$72, -2392(%rbp)
	ja	.L209
	movq	-2392(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L110(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L110(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L110:
	.long	.L162-.L110
	.long	.L161-.L110
	.long	.L160-.L110
	.long	.L159-.L110
	.long	.L158-.L110
	.long	.L209-.L110
	.long	.L157-.L110
	.long	.L156-.L110
	.long	.L155-.L110
	.long	.L154-.L110
	.long	.L153-.L110
	.long	.L152-.L110
	.long	.L209-.L110
	.long	.L151-.L110
	.long	.L150-.L110
	.long	.L149-.L110
	.long	.L148-.L110
	.long	.L209-.L110
	.long	.L209-.L110
	.long	.L209-.L110
	.long	.L147-.L110
	.long	.L146-.L110
	.long	.L145-.L110
	.long	.L144-.L110
	.long	.L209-.L110
	.long	.L143-.L110
	.long	.L142-.L110
	.long	.L141-.L110
	.long	.L209-.L110
	.long	.L140-.L110
	.long	.L139-.L110
	.long	.L138-.L110
	.long	.L209-.L110
	.long	.L137-.L110
	.long	.L209-.L110
	.long	.L136-.L110
	.long	.L209-.L110
	.long	.L135-.L110
	.long	.L209-.L110
	.long	.L209-.L110
	.long	.L134-.L110
	.long	.L133-.L110
	.long	.L209-.L110
	.long	.L132-.L110
	.long	.L131-.L110
	.long	.L130-.L110
	.long	.L129-.L110
	.long	.L128-.L110
	.long	.L127-.L110
	.long	.L209-.L110
	.long	.L126-.L110
	.long	.L125-.L110
	.long	.L124-.L110
	.long	.L123-.L110
	.long	.L122-.L110
	.long	.L121-.L110
	.long	.L120-.L110
	.long	.L119-.L110
	.long	.L118-.L110
	.long	.L117-.L110
	.long	.L116-.L110
	.long	.L209-.L110
	.long	.L209-.L110
	.long	.L115-.L110
	.long	.L114-.L110
	.long	.L209-.L110
	.long	.L113-.L110
	.long	.L209-.L110
	.long	.L209-.L110
	.long	.L209-.L110
	.long	.L112-.L110
	.long	.L111-.L110
	.long	.L109-.L110
	.text
.L126:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$30, -2392(%rbp)
	jmp	.L163
.L143:
	movq	stdin(%rip), %rax
	movq	%rax, -2432(%rbp)
	movq	$54, -2392(%rbp)
	jmp	.L163
.L124:
	cmpl	$0, -2436(%rbp)
	jne	.L164
	movq	$22, -2392(%rbp)
	jmp	.L163
.L164:
	movq	$70, -2392(%rbp)
	jmp	.L163
.L158:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$24, %edx
	movl	$1, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movl	$1, %edi
	call	exit@PLT
.L139:
	movq	-2408(%rbp), %rax
	movq	%rax, %rdi
	call	unlink@PLT
	movq	$48, -2392(%rbp)
	jmp	.L163
.L150:
	movq	-2496(%rbp), %rcx
	movl	-2484(%rbp), %eax
	leaq	.LC2(%rip), %rdx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	getopt@PLT
	movl	%eax, -2464(%rbp)
	movq	$31, -2392(%rbp)
	jmp	.L163
.L149:
	cmpl	$0, -2440(%rbp)
	je	.L166
	movq	$41, -2392(%rbp)
	jmp	.L163
.L166:
	movq	$10, -2392(%rbp)
	jmp	.L163
.L120:
	addl	$1, -2460(%rbp)
	movq	$14, -2392(%rbp)
	jmp	.L163
.L138:
	cmpl	$-1, -2464(%rbp)
	je	.L168
	movq	$21, -2392(%rbp)
	jmp	.L163
.L168:
	movq	$58, -2392(%rbp)
	jmp	.L163
.L155:
	movq	optarg(%rip), %rax
	leaq	.LC3(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -2432(%rbp)
	movq	$55, -2392(%rbp)
	jmp	.L163
.L130:
	cmpq	$0, -2424(%rbp)
	jne	.L170
	movq	$9, -2392(%rbp)
	jmp	.L163
.L170:
	movq	$26, -2392(%rbp)
	jmp	.L163
.L122:
	movl	optind(%rip), %eax
	cmpl	%eax, -2484(%rbp)
	jle	.L172
	movq	$1, -2392(%rbp)
	jmp	.L163
.L172:
	movq	$43, -2392(%rbp)
	jmp	.L163
.L161:
	movl	optind(%rip), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-2496(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movq	%rax, -2416(%rbp)
	movq	-2416(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -2384(%rbp)
	movq	-2384(%rbp), %rax
	addq	$20, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -2376(%rbp)
	movq	-2376(%rbp), %rax
	movq	%rax, -2408(%rbp)
	movq	$60, -2392(%rbp)
	jmp	.L163
.L144:
	cmpq	$0, -2432(%rbp)
	jne	.L174
	movq	$11, -2392(%rbp)
	jmp	.L163
.L174:
	movq	$16, -2392(%rbp)
	jmp	.L163
.L112:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L207
	jmp	.L208
.L159:
	movq	-2416(%rbp), %rdx
	movq	-2408(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcpy@PLT
	movq	-2408(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, %rdx
	movq	-2408(%rbp), %rax
	addq	%rdx, %rax
	movl	$2003136046, (%rax)
	movb	$0, 4(%rax)
	movq	-2408(%rbp), %rax
	leaq	.LC4(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -2424(%rbp)
	movq	$45, -2392(%rbp)
	jmp	.L163
.L148:
	movl	optind(%rip), %eax
	addl	$1, %eax
	movl	%eax, optind(%rip)
	movq	$54, -2392(%rbp)
	jmp	.L163
.L146:
	cmpl	$118, -2464(%rbp)
	je	.L177
	cmpl	$118, -2464(%rbp)
	jg	.L178
	cmpl	$102, -2464(%rbp)
	je	.L179
	cmpl	$116, -2464(%rbp)
	je	.L180
	jmp	.L178
.L177:
	movq	$56, -2392(%rbp)
	jmp	.L181
.L180:
	movq	$53, -2392(%rbp)
	jmp	.L181
.L179:
	movq	$8, -2392(%rbp)
	jmp	.L181
.L178:
	movq	$6, -2392(%rbp)
	nop
.L181:
	jmp	.L163
.L119:
	movq	-2416(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$37, -2392(%rbp)
	jmp	.L163
.L142:
	movq	-2432(%rbp), %rax
	movq	%rax, %rdi
	call	feof@PLT
	movl	%eax, -2452(%rbp)
	movq	$20, -2392(%rbp)
	jmp	.L163
.L152:
	movl	optind(%rip), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-2496(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L154:
	movq	-2408(%rbp), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L151:
	movq	-2432(%rbp), %rdx
	leaq	-2064(%rbp), %rax
	movl	$2048, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	movq	%rax, -2400(%rbp)
	movq	$51, -2392(%rbp)
	jmp	.L163
.L115:
	leaq	-2064(%rbp), %rax
	movq	%rax, %rdi
	call	substitute_line
	movq	-2424(%rbp), %rdx
	leaq	-2064(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fputs@PLT
	movq	$26, -2392(%rbp)
	jmp	.L163
.L125:
	cmpq	$0, -2400(%rbp)
	jne	.L182
	movq	$46, -2392(%rbp)
	jmp	.L163
.L182:
	movq	$63, -2392(%rbp)
	jmp	.L163
.L134:
	cmpl	$0, -2460(%rbp)
	je	.L184
	movq	$66, -2392(%rbp)
	jmp	.L163
.L184:
	movq	$35, -2392(%rbp)
	jmp	.L163
.L121:
	cmpq	$0, -2432(%rbp)
	jne	.L186
	movq	$72, -2392(%rbp)
	jmp	.L163
.L186:
	movq	$33, -2392(%rbp)
	jmp	.L163
.L116:
	cmpq	$0, -2408(%rbp)
	jne	.L188
	movq	$4, -2392(%rbp)
	jmp	.L163
.L188:
	movq	$3, -2392(%rbp)
	jmp	.L163
.L117:
	leaq	-2352(%rbp), %rdx
	movq	-2416(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	stat@PLT
	movl	%eax, -2444(%rbp)
	movq	$47, -2392(%rbp)
	jmp	.L163
.L157:
	movq	-2496(%rbp), %rax
	movq	(%rax), %rdx
	movq	stderr(%rip), %rax
	leaq	.LC6(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	$14, -2392(%rbp)
	jmp	.L163
.L141:
	movq	$0, -2392(%rbp)
	jmp	.L163
.L118:
	movl	optind(%rip), %eax
	cmpl	%eax, -2484(%rbp)
	jle	.L190
	movq	$7, -2392(%rbp)
	jmp	.L163
.L190:
	movq	$25, -2392(%rbp)
	jmp	.L163
.L127:
	leaq	-2208(%rbp), %rdx
	movq	-2416(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	stat@PLT
	movl	%eax, -2436(%rbp)
	movq	$52, -2392(%rbp)
	jmp	.L163
.L111:
	movq	-2408(%rbp), %rdx
	movq	-2416(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	compare_file
	movl	%eax, -2440(%rbp)
	movq	$15, -2392(%rbp)
	jmp	.L163
.L145:
	movl	-2184(%rbp), %eax
	andb	$109, %al
	movl	%eax, %edx
	movq	-2416(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	chmod@PLT
	movq	$70, -2392(%rbp)
	jmp	.L163
.L123:
	addl	$1, -2456(%rbp)
	movq	$14, -2392(%rbp)
	jmp	.L163
.L128:
	cmpl	$0, -2444(%rbp)
	jne	.L192
	movq	$40, -2392(%rbp)
	jmp	.L163
.L192:
	movq	$30, -2392(%rbp)
	jmp	.L163
.L131:
	movq	-2416(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$29, -2392(%rbp)
	jmp	.L163
.L109:
	movq	optarg(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L137:
	movq	-2432(%rbp), %rax
	movq	%rax, %rdi
	call	parse_config_file
	movq	-2432(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$14, -2392(%rbp)
	jmp	.L163
.L135:
	movq	-2416(%rbp), %rdx
	movq	-2408(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	rename@PLT
	movq	$48, -2392(%rbp)
	jmp	.L163
.L114:
	cmpq	$0, -2416(%rbp)
	je	.L194
	movq	$71, -2392(%rbp)
	jmp	.L163
.L194:
	movq	$70, -2392(%rbp)
	jmp	.L163
.L133:
	cmpl	$0, -2460(%rbp)
	je	.L196
	movq	$44, -2392(%rbp)
	jmp	.L163
.L196:
	movq	$29, -2392(%rbp)
	jmp	.L163
.L153:
	cmpl	$0, -2460(%rbp)
	je	.L198
	movq	$57, -2392(%rbp)
	jmp	.L163
.L198:
	movq	$37, -2392(%rbp)
	jmp	.L163
.L162:
	movq	$0, -2416(%rbp)
	movq	$0, -2408(%rbp)
	movl	$0, -2460(%rbp)
	movl	$0, -2456(%rbp)
	movq	$14, -2392(%rbp)
	jmp	.L163
.L129:
	movq	-2432(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	-2424(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$64, -2392(%rbp)
	jmp	.L163
.L113:
	movq	-2416(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$35, -2392(%rbp)
	jmp	.L163
.L156:
	movl	optind(%rip), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-2496(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	leaq	.LC3(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -2432(%rbp)
	movq	$23, -2392(%rbp)
	jmp	.L163
.L136:
	movq	-2280(%rbp), %rax
	movq	%rax, -2368(%rbp)
	movl	$0, %edi
	call	time@PLT
	movq	%rax, -2360(%rbp)
	leaq	-2368(%rbp), %rdx
	movq	-2416(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	utime@PLT
	movl	%eax, -2448(%rbp)
	movq	$2, -2392(%rbp)
	jmp	.L163
.L140:
	cmpl	$0, -2456(%rbp)
	je	.L200
	movq	$59, -2392(%rbp)
	jmp	.L163
.L200:
	movq	$30, -2392(%rbp)
	jmp	.L163
.L132:
	movq	stdout(%rip), %rax
	movq	%rax, -2424(%rbp)
	movq	$0, -2416(%rbp)
	movq	$26, -2392(%rbp)
	jmp	.L163
.L160:
	cmpl	$0, -2448(%rbp)
	jns	.L202
	movq	$50, -2392(%rbp)
	jmp	.L163
.L202:
	movq	$30, -2392(%rbp)
	jmp	.L163
.L147:
	cmpl	$0, -2452(%rbp)
	je	.L204
	movq	$46, -2392(%rbp)
	jmp	.L163
.L204:
	movq	$13, -2392(%rbp)
	jmp	.L163
.L209:
	nop
.L163:
	jmp	.L206
.L208:
	call	__stack_chk_fail@PLT
.L207:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	main, .-main
	.type	fetch_subst_entry, @function
fetch_subst_entry:
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
	movq	$1, -8(%rbp)
.L225:
	cmpq	$8, -8(%rbp)
	ja	.L227
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L213(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L213(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L213:
	.long	.L227-.L213
	.long	.L218-.L213
	.long	.L217-.L213
	.long	.L216-.L213
	.long	.L215-.L213
	.long	.L227-.L213
	.long	.L227-.L213
	.long	.L214-.L213
	.long	.L212-.L213
	.text
.L215:
	movq	-16(%rbp), %rax
	jmp	.L226
.L212:
	movq	-16(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, -16(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L220
.L218:
	movq	subst_table(%rip), %rax
	movq	%rax, -16(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L220
.L216:
	movq	-16(%rbp), %rax
	movq	(%rax), %rdx
	movq	-40(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -20(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L220
.L214:
	cmpl	$0, -20(%rbp)
	jne	.L221
	movq	$4, -8(%rbp)
	jmp	.L220
.L221:
	movq	$8, -8(%rbp)
	jmp	.L220
.L217:
	cmpq	$0, -16(%rbp)
	je	.L223
	movq	$3, -8(%rbp)
	jmp	.L220
.L223:
	movq	$4, -8(%rbp)
	jmp	.L220
.L227:
	nop
.L220:
	jmp	.L225
.L226:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	fetch_subst_entry, .-fetch_subst_entry
	.section	.rodata
.LC9:
	.string	"Unfound expansion: '%s'\n"
	.text
	.type	substitute_line, @function
substitute_line:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	addq	$-128, %rsp
	movq	%rdi, -120(%rbp)
	movq	$2, -40(%rbp)
.L302:
	cmpq	$62, -40(%rbp)
	ja	.L303
	movq	-40(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L231(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L231(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L231:
	.long	.L303-.L231
	.long	.L303-.L231
	.long	.L269-.L231
	.long	.L303-.L231
	.long	.L268-.L231
	.long	.L303-.L231
	.long	.L267-.L231
	.long	.L266-.L231
	.long	.L303-.L231
	.long	.L265-.L231
	.long	.L264-.L231
	.long	.L263-.L231
	.long	.L303-.L231
	.long	.L262-.L231
	.long	.L261-.L231
	.long	.L260-.L231
	.long	.L303-.L231
	.long	.L303-.L231
	.long	.L259-.L231
	.long	.L258-.L231
	.long	.L303-.L231
	.long	.L257-.L231
	.long	.L256-.L231
	.long	.L255-.L231
	.long	.L254-.L231
	.long	.L253-.L231
	.long	.L303-.L231
	.long	.L303-.L231
	.long	.L303-.L231
	.long	.L252-.L231
	.long	.L251-.L231
	.long	.L250-.L231
	.long	.L303-.L231
	.long	.L249-.L231
	.long	.L248-.L231
	.long	.L247-.L231
	.long	.L304-.L231
	.long	.L245-.L231
	.long	.L244-.L231
	.long	.L303-.L231
	.long	.L243-.L231
	.long	.L303-.L231
	.long	.L242-.L231
	.long	.L241-.L231
	.long	.L303-.L231
	.long	.L303-.L231
	.long	.L240-.L231
	.long	.L303-.L231
	.long	.L239-.L231
	.long	.L238-.L231
	.long	.L237-.L231
	.long	.L303-.L231
	.long	.L236-.L231
	.long	.L303-.L231
	.long	.L303-.L231
	.long	.L303-.L231
	.long	.L303-.L231
	.long	.L235-.L231
	.long	.L303-.L231
	.long	.L234-.L231
	.long	.L233-.L231
	.long	.L232-.L231
	.long	.L230-.L231
	.text
.L259:
	cmpq	$0, -64(%rbp)
	jne	.L270
	movq	$38, -40(%rbp)
	jmp	.L272
.L270:
	movq	$13, -40(%rbp)
	jmp	.L272
.L237:
	movq	-48(%rbp), %rax
	subq	$2, %rax
	cmpq	%rax, -56(%rbp)
	jne	.L273
	movq	$10, -40(%rbp)
	jmp	.L272
.L273:
	movq	$52, -40(%rbp)
	jmp	.L272
.L253:
	cmpq	$0, -88(%rbp)
	jne	.L275
	movq	$43, -40(%rbp)
	jmp	.L272
.L275:
	movq	$9, -40(%rbp)
	jmp	.L272
.L238:
	movq	-72(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -32(%rbp)
	movq	-32(%rbp), %rax
	addq	%rax, -96(%rbp)
	movq	$52, -40(%rbp)
	jmp	.L272
.L236:
	cmpq	$0, -96(%rbp)
	je	.L277
	movq	$29, -40(%rbp)
	jmp	.L272
.L277:
	movq	$43, -40(%rbp)
	jmp	.L272
.L268:
	movq	-80(%rbp), %rax
	subq	-88(%rbp), %rax
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rcx
	movq	-88(%rbp), %rax
	movl	$0, %edx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	get_subst_symbol
	movq	%rax, -64(%rbp)
	movq	$59, -40(%rbp)
	jmp	.L272
.L251:
	movq	-88(%rbp), %rax
	movq	%rax, -96(%rbp)
	movq	$52, -40(%rbp)
	jmp	.L272
.L230:
	addq	$1, -88(%rbp)
	movq	-88(%rbp), %rax
	movl	$125, %esi
	movq	%rax, %rdi
	call	strchr@PLT
	movq	%rax, -80(%rbp)
	movq	$57, -40(%rbp)
	jmp	.L272
.L261:
	movq	-88(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$64, %al
	jne	.L279
	movq	$48, -40(%rbp)
	jmp	.L272
.L279:
	movq	$33, -40(%rbp)
	jmp	.L272
.L260:
	cmpl	$0, -100(%rbp)
	je	.L281
	movq	$52, -40(%rbp)
	jmp	.L272
.L281:
	movq	$49, -40(%rbp)
	jmp	.L272
.L250:
	movq	-88(%rbp), %rax
	movq	%rax, -96(%rbp)
	movq	$23, -40(%rbp)
	jmp	.L272
.L255:
	cmpq	$0, -96(%rbp)
	je	.L283
	movq	$7, -40(%rbp)
	jmp	.L272
.L283:
	movq	$36, -40(%rbp)
	jmp	.L272
.L254:
	addq	$1, -88(%rbp)
	movq	$42, -40(%rbp)
	jmp	.L272
.L257:
	movq	stderr(%rip), %rax
	movq	-64(%rbp), %rdx
	leaq	.LC9(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	-80(%rbp), %rax
	addq	$1, %rax
	movq	%rax, -96(%rbp)
	movq	$52, -40(%rbp)
	jmp	.L272
.L235:
	cmpq	$0, -80(%rbp)
	jne	.L286
	movq	$36, -40(%rbp)
	jmp	.L272
.L286:
	movq	$34, -40(%rbp)
	jmp	.L272
.L263:
	movq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	fetch_subst_entry
	movq	%rax, -72(%rbp)
	movq	$46, -40(%rbp)
	jmp	.L272
.L265:
	addq	$1, -88(%rbp)
	movq	$14, -40(%rbp)
	jmp	.L272
.L262:
	movq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	fetch_subst_entry
	movq	%rax, -72(%rbp)
	movq	$19, -40(%rbp)
	jmp	.L272
.L258:
	cmpq	$0, -72(%rbp)
	jne	.L288
	movq	$22, -40(%rbp)
	jmp	.L272
.L288:
	movq	$35, -40(%rbp)
	jmp	.L272
.L243:
	cmpq	$0, -88(%rbp)
	jne	.L290
	movq	$36, -40(%rbp)
	jmp	.L272
.L290:
	movq	$24, -40(%rbp)
	jmp	.L272
.L233:
	movq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -56(%rbp)
	movq	-72(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -48(%rbp)
	movq	$50, -40(%rbp)
	jmp	.L272
.L234:
	cmpq	$0, -64(%rbp)
	jne	.L292
	movq	$30, -40(%rbp)
	jmp	.L272
.L292:
	movq	$11, -40(%rbp)
	jmp	.L272
.L267:
	movq	-72(%rbp), %rax
	movq	8(%rax), %rax
	movzbl	(%rax), %eax
	cmpb	$64, %al
	jne	.L294
	movq	$60, -40(%rbp)
	jmp	.L272
.L294:
	movq	$52, -40(%rbp)
	jmp	.L272
.L244:
	movq	-88(%rbp), %rax
	movq	%rax, -96(%rbp)
	movq	$23, -40(%rbp)
	jmp	.L272
.L232:
	movq	-88(%rbp), %rax
	subq	$1, %rax
	movq	%rax, -96(%rbp)
	movq	-72(%rbp), %rax
	movq	8(%rax), %rdx
	movq	-80(%rbp), %rcx
	movq	-96(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	replace_string
	movq	$6, -40(%rbp)
	jmp	.L272
.L248:
	movq	-80(%rbp), %rax
	subq	-88(%rbp), %rax
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rcx
	movq	-88(%rbp), %rax
	movl	$36, %edx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	get_subst_symbol
	movq	%rax, -64(%rbp)
	movq	$18, -40(%rbp)
	jmp	.L272
.L239:
	movq	-88(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	leaq	1(%rax), %rdx
	movq	-88(%rbp), %rax
	leaq	-1(%rax), %rcx
	movq	-88(%rbp), %rax
	movq	%rax, %rsi
	movq	%rcx, %rdi
	call	memmove@PLT
	movq	-88(%rbp), %rax
	addq	$1, %rax
	movq	%rax, -96(%rbp)
	movq	$52, -40(%rbp)
	jmp	.L272
.L256:
	movq	-80(%rbp), %rax
	addq	$1, %rax
	movq	%rax, -96(%rbp)
	movq	$23, -40(%rbp)
	jmp	.L272
.L249:
	movq	-88(%rbp), %rax
	movl	$64, %esi
	movq	%rax, %rdi
	call	strchr@PLT
	movq	%rax, -80(%rbp)
	movq	$37, -40(%rbp)
	jmp	.L272
.L245:
	cmpq	$0, -80(%rbp)
	jne	.L296
	movq	$43, -40(%rbp)
	jmp	.L272
.L296:
	movq	$4, -40(%rbp)
	jmp	.L272
.L264:
	movq	-72(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	leaq	-2(%rax), %rdx
	movq	-72(%rbp), %rax
	movq	8(%rax), %rax
	leaq	1(%rax), %rcx
	movq	-64(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	strncmp@PLT
	movl	%eax, -100(%rbp)
	movq	$15, -40(%rbp)
	jmp	.L272
.L242:
	movq	-88(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$123, %al
	je	.L298
	movq	$31, -40(%rbp)
	jmp	.L272
.L298:
	movq	$62, -40(%rbp)
	jmp	.L272
.L240:
	cmpq	$0, -72(%rbp)
	jne	.L300
	movq	$21, -40(%rbp)
	jmp	.L272
.L300:
	movq	$61, -40(%rbp)
	jmp	.L272
.L266:
	movq	-96(%rbp), %rax
	movl	$36, %esi
	movq	%rax, %rdi
	call	strchr@PLT
	movq	%rax, -88(%rbp)
	movq	$40, -40(%rbp)
	jmp	.L272
.L247:
	movq	-88(%rbp), %rax
	subq	$2, %rax
	movq	%rax, -96(%rbp)
	movq	-72(%rbp), %rax
	movq	8(%rax), %rdx
	movq	-80(%rbp), %rcx
	movq	-96(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	replace_string
	movq	$23, -40(%rbp)
	jmp	.L272
.L252:
	movq	-96(%rbp), %rax
	movl	$64, %esi
	movq	%rax, %rdi
	call	strchr@PLT
	movq	%rax, -88(%rbp)
	movq	$25, -40(%rbp)
	jmp	.L272
.L241:
	movq	-120(%rbp), %rax
	movq	%rax, -96(%rbp)
	movq	$23, -40(%rbp)
	jmp	.L272
.L269:
	movq	-120(%rbp), %rax
	movq	%rax, -96(%rbp)
	movq	$52, -40(%rbp)
	jmp	.L272
.L303:
	nop
.L272:
	jmp	.L302
.L304:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	substitute_line, .-substitute_line
	.type	compare_file, @function
compare_file:
.LFB10:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$4096, %rsp
	orq	$0, (%rsp)
	subq	$80, %rsp
	movq	%rdi, -4168(%rbp)
	movq	%rsi, -4176(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$6, -4120(%rbp)
.L344:
	cmpq	$26, -4120(%rbp)
	ja	.L347
	movq	-4120(%rbp), %rax
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
	.long	.L326-.L308
	.long	.L325-.L308
	.long	.L324-.L308
	.long	.L323-.L308
	.long	.L322-.L308
	.long	.L321-.L308
	.long	.L320-.L308
	.long	.L319-.L308
	.long	.L318-.L308
	.long	.L317-.L308
	.long	.L316-.L308
	.long	.L347-.L308
	.long	.L315-.L308
	.long	.L314-.L308
	.long	.L347-.L308
	.long	.L313-.L308
	.long	.L347-.L308
	.long	.L312-.L308
	.long	.L347-.L308
	.long	.L311-.L308
	.long	.L347-.L308
	.long	.L310-.L308
	.long	.L347-.L308
	.long	.L347-.L308
	.long	.L309-.L308
	.long	.L307-.L308
	.text
.L312:
	movl	$1, -4160(%rbp)
	movq	$2, -4120(%rbp)
	jmp	.L328
.L309:
	cmpq	$0, -4136(%rbp)
	jne	.L329
	movq	$9, -4120(%rbp)
	jmp	.L328
.L329:
	movq	$10, -4120(%rbp)
	jmp	.L328
.L323:
	cmpq	$0, -4152(%rbp)
	jne	.L331
	movq	$8, -4120(%rbp)
	jmp	.L328
.L331:
	movq	$20, -4120(%rbp)
	jmp	.L328
.L314:
	movl	-4160(%rbp), %eax
	jmp	.L345
.L319:
	movl	$0, %eax
	jmp	.L345
.L326:
	movq	-4152(%rbp), %rdx
	leaq	-4112(%rbp), %rax
	movl	$2048, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	movq	%rax, -4136(%rbp)
	movq	-4144(%rbp), %rdx
	leaq	-2064(%rbp), %rax
	movl	$2048, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	movq	%rax, -4128(%rbp)
	movq	$16, -4120(%rbp)
	jmp	.L328
.L324:
	cmpq	$0, -4144(%rbp)
	jne	.L334
	movq	$0, -4120(%rbp)
	jmp	.L328
.L334:
	movq	$1, -4120(%rbp)
	jmp	.L328
.L313:
	cmpq	$0, -4136(%rbp)
	jne	.L336
	movq	$26, -4120(%rbp)
	jmp	.L328
.L336:
	movq	$25, -4120(%rbp)
	jmp	.L328
.L307:
	cmpq	$0, -4128(%rbp)
	jne	.L338
	movq	$18, -4120(%rbp)
	jmp	.L328
.L338:
	movq	$25, -4120(%rbp)
	jmp	.L328
.L316:
	movl	$0, -4160(%rbp)
	movq	$2, -4120(%rbp)
	jmp	.L328
.L318:
	movl	$0, -4160(%rbp)
	movq	$2, -4120(%rbp)
	jmp	.L328
.L315:
	leaq	-2064(%rbp), %rdx
	leaq	-4112(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -4156(%rbp)
	movq	$5, -4120(%rbp)
	jmp	.L328
.L321:
	movq	-4168(%rbp), %rax
	leaq	.LC3(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -4152(%rbp)
	movq	$4, -4120(%rbp)
	jmp	.L328
.L310:
	movl	$0, -4160(%rbp)
	movq	$2, -4120(%rbp)
	jmp	.L328
.L322:
	cmpl	$0, -4156(%rbp)
	je	.L340
	movq	$11, -4120(%rbp)
	jmp	.L328
.L340:
	movq	$1, -4120(%rbp)
	jmp	.L328
.L317:
	cmpq	$0, -4128(%rbp)
	jne	.L342
	movq	$22, -4120(%rbp)
	jmp	.L328
.L342:
	movq	$13, -4120(%rbp)
	jmp	.L328
.L327:
	movq	-4152(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$7, -4120(%rbp)
	jmp	.L328
.L320:
	movl	$0, %eax
	jmp	.L345
.L325:
	movq	-4152(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	-4144(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$14, -4120(%rbp)
	jmp	.L328
.L311:
	movq	-4176(%rbp), %rax
	leaq	.LC3(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -4144(%rbp)
	movq	$3, -4120(%rbp)
	jmp	.L328
.L347:
	nop
.L328:
	jmp	.L344
.L345:
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L346
	call	__stack_chk_fail@PLT
.L346:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	compare_file, .-compare_file
	.type	get_subst_symbol, @function
get_subst_symbol:
.LFB11:
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
	movl	%edx, %eax
	movb	%al, -52(%rbp)
	movq	$7, -16(%rbp)
.L390:
	cmpq	$23, -16(%rbp)
	ja	.L391
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L351(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L351(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L351:
	.long	.L367-.L351
	.long	.L366-.L351
	.long	.L391-.L351
	.long	.L365-.L351
	.long	.L391-.L351
	.long	.L391-.L351
	.long	.L364-.L351
	.long	.L363-.L351
	.long	.L362-.L351
	.long	.L361-.L351
	.long	.L391-.L351
	.long	.L391-.L351
	.long	.L360-.L351
	.long	.L359-.L351
	.long	.L358-.L351
	.long	.L391-.L351
	.long	.L357-.L351
	.long	.L356-.L351
	.long	.L355-.L351
	.long	.L354-.L351
	.long	.L353-.L351
	.long	.L391-.L351
	.long	.L352-.L351
	.long	.L350-.L351
	.text
.L355:
	movq	-32(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$47, %al
	jle	.L368
	movq	$16, -16(%rbp)
	jmp	.L370
.L368:
	movq	$9, -16(%rbp)
	jmp	.L370
.L358:
	movq	-32(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$96, %al
	jle	.L371
	movq	$0, -16(%rbp)
	jmp	.L370
.L371:
	movq	$17, -16(%rbp)
	jmp	.L370
.L360:
	movq	-48(%rbp), %rdx
	movq	-40(%rbp), %rcx
	movq	-24(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	memcpy@PLT
	movq	-24(%rbp), %rdx
	movq	-48(%rbp), %rax
	addq	%rdx, %rax
	movb	$0, (%rax)
	movq	-24(%rbp), %rax
	movq	%rax, -32(%rbp)
	movq	$20, -16(%rbp)
	jmp	.L370
.L362:
	movl	$0, %eax
	jmp	.L373
.L366:
	movq	-24(%rbp), %rax
	movq	%rax, -8(%rbp)
	addq	$1, -24(%rbp)
	movq	-8(%rbp), %rax
	movzbl	-52(%rbp), %edx
	movb	%dl, (%rax)
	movq	$3, -16(%rbp)
	jmp	.L370
.L350:
	cmpb	$0, -52(%rbp)
	je	.L374
	movq	$1, -16(%rbp)
	jmp	.L370
.L374:
	movq	$3, -16(%rbp)
	jmp	.L370
.L365:
	cmpq	$126, -48(%rbp)
	jbe	.L376
	movq	$8, -16(%rbp)
	jmp	.L370
.L376:
	movq	$12, -16(%rbp)
	jmp	.L370
.L357:
	movq	-32(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$57, %al
	jg	.L378
	movq	$19, -16(%rbp)
	jmp	.L370
.L378:
	movq	$9, -16(%rbp)
	jmp	.L370
.L361:
	movq	-32(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$95, %al
	je	.L380
	movq	$6, -16(%rbp)
	jmp	.L370
.L380:
	movq	$19, -16(%rbp)
	jmp	.L370
.L359:
	movq	-32(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$90, %al
	jg	.L382
	movq	$19, -16(%rbp)
	jmp	.L370
.L382:
	movq	$18, -16(%rbp)
	jmp	.L370
.L354:
	addq	$1, -32(%rbp)
	movq	$20, -16(%rbp)
	jmp	.L370
.L356:
	movq	-32(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$64, %al
	jle	.L384
	movq	$13, -16(%rbp)
	jmp	.L370
.L384:
	movq	$18, -16(%rbp)
	jmp	.L370
.L364:
	movl	$0, %eax
	jmp	.L373
.L352:
	leaq	replace_name(%rip), %rax
	jmp	.L373
.L367:
	movq	-32(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$122, %al
	jg	.L386
	movq	$19, -16(%rbp)
	jmp	.L370
.L386:
	movq	$17, -16(%rbp)
	jmp	.L370
.L363:
	leaq	replace_name(%rip), %rax
	movq	%rax, -24(%rbp)
	movq	$23, -16(%rbp)
	jmp	.L370
.L353:
	movq	-32(%rbp), %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	je	.L388
	movq	$14, -16(%rbp)
	jmp	.L370
.L388:
	movq	$22, -16(%rbp)
	jmp	.L370
.L391:
	nop
.L370:
	jmp	.L390
.L373:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE11:
	.size	get_subst_symbol, .-get_subst_symbol
	.type	add_subst, @function
add_subst:
.LFB12:
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
	movq	$14, -48(%rbp)
.L425:
	cmpq	$19, -48(%rbp)
	ja	.L426
	movq	-48(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L395(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L395(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L395:
	.long	.L426-.L395
	.long	.L410-.L395
	.long	.L409-.L395
	.long	.L408-.L395
	.long	.L426-.L395
	.long	.L407-.L395
	.long	.L406-.L395
	.long	.L405-.L395
	.long	.L404-.L395
	.long	.L403-.L395
	.long	.L402-.L395
	.long	.L401-.L395
	.long	.L426-.L395
	.long	.L426-.L395
	.long	.L400-.L395
	.long	.L399-.L395
	.long	.L398-.L395
	.long	.L397-.L395
	.long	.L396-.L395
	.long	.L394-.L395
	.text
.L396:
	movq	-56(%rbp), %rax
	movq	(%rax), %rax
	movq	-72(%rbp), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcpy@PLT
	movq	-56(%rbp), %rax
	movq	8(%rax), %rax
	movq	-80(%rbp), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcpy@PLT
	movq	subst_table(%rip), %rdx
	movq	-56(%rbp), %rax
	movq	%rdx, 16(%rax)
	movq	-56(%rbp), %rax
	movq	%rax, subst_table(%rip)
	movq	$3, -48(%rbp)
	jmp	.L411
.L400:
	movq	$6, -48(%rbp)
	jmp	.L411
.L399:
	movq	-56(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$16, -48(%rbp)
	jmp	.L411
.L404:
	cmpq	$0, -56(%rbp)
	jne	.L412
	movq	$5, -48(%rbp)
	jmp	.L411
.L412:
	movq	$7, -48(%rbp)
	jmp	.L411
.L410:
	movl	-60(%rbp), %eax
	jmp	.L414
.L408:
	movl	$0, %eax
	jmp	.L414
.L398:
	movq	-56(%rbp), %rax
	movq	8(%rax), %rax
	testq	%rax, %rax
	je	.L415
	movq	$10, -48(%rbp)
	jmp	.L411
.L415:
	movq	$9, -48(%rbp)
	jmp	.L411
.L401:
	movq	-80(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -40(%rbp)
	movq	-40(%rbp), %rax
	addq	$1, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -32(%rbp)
	movq	-56(%rbp), %rax
	movq	-32(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movq	$17, -48(%rbp)
	jmp	.L411
.L403:
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$1, -48(%rbp)
	jmp	.L411
.L394:
	movq	-56(%rbp), %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	jne	.L417
	movq	$5, -48(%rbp)
	jmp	.L411
.L417:
	movq	$11, -48(%rbp)
	jmp	.L411
.L397:
	movq	-56(%rbp), %rax
	movq	8(%rax), %rax
	testq	%rax, %rax
	jne	.L419
	movq	$5, -48(%rbp)
	jmp	.L411
.L419:
	movq	$18, -48(%rbp)
	jmp	.L411
.L406:
	movq	$0, -56(%rbp)
	movl	$12, -60(%rbp)
	movl	$24, %edi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -56(%rbp)
	movq	$8, -48(%rbp)
	jmp	.L411
.L407:
	cmpq	$0, -56(%rbp)
	je	.L421
	movq	$2, -48(%rbp)
	jmp	.L411
.L421:
	movq	$1, -48(%rbp)
	jmp	.L411
.L402:
	movq	-56(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$9, -48(%rbp)
	jmp	.L411
.L405:
	movq	-72(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	addq	$1, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -16(%rbp)
	movq	-56(%rbp), %rax
	movq	-16(%rbp), %rdx
	movq	%rdx, (%rax)
	movq	$19, -48(%rbp)
	jmp	.L411
.L409:
	movq	-56(%rbp), %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	je	.L423
	movq	$15, -48(%rbp)
	jmp	.L411
.L423:
	movq	$16, -48(%rbp)
	jmp	.L411
.L426:
	nop
.L411:
	jmp	.L425
.L414:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE12:
	.size	add_subst, .-add_subst
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
