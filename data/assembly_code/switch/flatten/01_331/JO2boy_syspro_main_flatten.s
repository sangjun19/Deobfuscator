	.file	"JO2boy_syspro_main_flatten.c"
	.text
	.globl	_TIG_IZ_eWe5_argc
	.bss
	.align 4
	.type	_TIG_IZ_eWe5_argc, @object
	.size	_TIG_IZ_eWe5_argc, 4
_TIG_IZ_eWe5_argc:
	.zero	4
	.globl	_TIG_IZ_eWe5_argv
	.align 8
	.type	_TIG_IZ_eWe5_argv, @object
	.size	_TIG_IZ_eWe5_argv, 8
_TIG_IZ_eWe5_argv:
	.zero	8
	.globl	_TIG_IZ_eWe5_envp
	.align 8
	.type	_TIG_IZ_eWe5_envp, @object
	.size	_TIG_IZ_eWe5_envp, 8
_TIG_IZ_eWe5_envp:
	.zero	8
	.section	.rodata
	.align 8
.LC0:
	.string	"my realistic group id : %d\nmy valid group id : %d\n"
	.text
	.globl	print_group_ids
	.type	print_group_ids, @function
print_group_ids:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	$2, -8(%rbp)
.L7:
	cmpq	$2, -8(%rbp)
	je	.L2
	cmpq	$2, -8(%rbp)
	ja	.L9
	cmpq	$0, -8(%rbp)
	je	.L4
	cmpq	$1, -8(%rbp)
	jne	.L9
	jmp	.L8
.L4:
	call	getgid@PLT
	movl	%eax, -24(%rbp)
	movl	-24(%rbp), %eax
	movl	%eax, -20(%rbp)
	call	getegid@PLT
	movl	%eax, -16(%rbp)
	movl	-16(%rbp), %eax
	movl	%eax, -12(%rbp)
	movl	-12(%rbp), %edx
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -8(%rbp)
	jmp	.L6
.L2:
	movq	$0, -8(%rbp)
	jmp	.L6
.L9:
	nop
.L6:
	jmp	.L7
.L8:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	print_group_ids, .-print_group_ids
	.section	.rodata
.LC1:
	.string	"my process number: %d\n"
	.text
	.globl	print_process_id
	.type	print_process_id, @function
print_process_id:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$0, -8(%rbp)
.L16:
	cmpq	$2, -8(%rbp)
	je	.L11
	cmpq	$2, -8(%rbp)
	ja	.L18
	cmpq	$0, -8(%rbp)
	je	.L13
	cmpq	$1, -8(%rbp)
	jne	.L18
	jmp	.L17
.L13:
	movq	$2, -8(%rbp)
	jmp	.L15
.L11:
	call	getpid@PLT
	movl	%eax, -16(%rbp)
	movl	-16(%rbp), %eax
	movl	%eax, -12(%rbp)
	movl	-12(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -8(%rbp)
	jmp	.L15
.L18:
	nop
.L15:
	jmp	.L16
.L17:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	print_process_id, .-print_process_id
	.section	.rodata
.LC2:
	.string	"%s=%s\n"
	.align 8
.LC3:
	.string	"Environment variable %s not found.\n"
	.text
	.globl	print_environment_variable
	.type	print_environment_variable, @function
print_environment_variable:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movq	$4, -16(%rbp)
.L32:
	cmpq	$5, -16(%rbp)
	ja	.L33
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L22(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L22(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L22:
	.long	.L27-.L22
	.long	.L26-.L22
	.long	.L25-.L22
	.long	.L34-.L22
	.long	.L23-.L22
	.long	.L21-.L22
	.text
.L23:
	movq	$2, -16(%rbp)
	jmp	.L28
.L26:
	movq	-24(%rbp), %rdx
	movq	-40(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -16(%rbp)
	jmp	.L28
.L21:
	cmpq	$0, -24(%rbp)
	je	.L30
	movq	$1, -16(%rbp)
	jmp	.L28
.L30:
	movq	$0, -16(%rbp)
	jmp	.L28
.L27:
	movq	-40(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -16(%rbp)
	jmp	.L28
.L25:
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	getenv@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	$5, -16(%rbp)
	jmp	.L28
.L33:
	nop
.L28:
	jmp	.L32
.L34:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	print_environment_variable, .-print_environment_variable
	.globl	print_all_environment_variables
	.type	print_all_environment_variables, @function
print_all_environment_variables:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$2, -8(%rbp)
.L45:
	cmpq	$6, -8(%rbp)
	je	.L36
	cmpq	$6, -8(%rbp)
	ja	.L46
	cmpq	$4, -8(%rbp)
	je	.L47
	cmpq	$4, -8(%rbp)
	ja	.L46
	cmpq	$0, -8(%rbp)
	je	.L39
	cmpq	$2, -8(%rbp)
	je	.L40
	jmp	.L46
.L36:
	movq	-16(%rbp), %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	je	.L42
	movq	$0, -8(%rbp)
	jmp	.L44
.L42:
	movq	$4, -8(%rbp)
	jmp	.L44
.L39:
	movq	-16(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	puts@PLT
	addq	$8, -16(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L44
.L40:
	movq	environ(%rip), %rax
	movq	%rax, -16(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L44
.L46:
	nop
.L44:
	jmp	.L45
.L47:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	print_all_environment_variables, .-print_all_environment_variables
	.section	.rodata
	.align 8
.LC4:
	.string	"my realistic user id : %d\nmy valid user id : %d\n"
	.text
	.globl	print_user_ids
	.type	print_user_ids, @function
print_user_ids:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	$1, -8(%rbp)
.L54:
	cmpq	$2, -8(%rbp)
	je	.L49
	cmpq	$2, -8(%rbp)
	ja	.L55
	cmpq	$0, -8(%rbp)
	je	.L56
	cmpq	$1, -8(%rbp)
	jne	.L55
	movq	$2, -8(%rbp)
	jmp	.L52
.L49:
	call	getuid@PLT
	movl	%eax, -24(%rbp)
	movl	-24(%rbp), %eax
	movl	%eax, -20(%rbp)
	call	geteuid@PLT
	movl	%eax, -16(%rbp)
	movl	-16(%rbp), %eax
	movl	%eax, -12(%rbp)
	movl	-12(%rbp), %edx
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$0, -8(%rbp)
	jmp	.L52
.L55:
	nop
.L52:
	jmp	.L54
.L56:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	print_user_ids, .-print_user_ids
	.section	.rodata
.LC5:
	.string	"Invalid argument: %s\n"
.LC6:
	.string	"Unknown option: %s\n"
	.align 8
.LC7:
	.string	"Usage: %s [-e ENV_VAR | -u | -g | -i | -p]\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB9:
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
	movq	$0, _TIG_IZ_eWe5_envp(%rip)
	nop
.L58:
	movq	$0, _TIG_IZ_eWe5_argv(%rip)
	nop
.L59:
	movl	$0, _TIG_IZ_eWe5_argc(%rip)
	nop
	nop
.L60:
.L61:
#APP
# 77 "JO2boy_syspro_main.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-eWe5--0
# 0 "" 2
#NO_APP
	movl	-20(%rbp), %eax
	movl	%eax, _TIG_IZ_eWe5_argc(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, _TIG_IZ_eWe5_argv(%rip)
	movq	-40(%rbp), %rax
	movq	%rax, _TIG_IZ_eWe5_envp(%rip)
	nop
	movq	$24, -8(%rbp)
.L104:
	cmpq	$25, -8(%rbp)
	ja	.L86
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L64(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L64(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L64:
	.long	.L83-.L64
	.long	.L82-.L64
	.long	.L81-.L64
	.long	.L86-.L64
	.long	.L80-.L64
	.long	.L79-.L64
	.long	.L78-.L64
	.long	.L77-.L64
	.long	.L76-.L64
	.long	.L86-.L64
	.long	.L86-.L64
	.long	.L75-.L64
	.long	.L74-.L64
	.long	.L73-.L64
	.long	.L72-.L64
	.long	.L86-.L64
	.long	.L71-.L64
	.long	.L86-.L64
	.long	.L70-.L64
	.long	.L69-.L64
	.long	.L86-.L64
	.long	.L68-.L64
	.long	.L67-.L64
	.long	.L66-.L64
	.long	.L65-.L64
	.long	.L63-.L64
	.text
.L70:
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-32(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	addq	$1, %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	je	.L84
	movq	$13, -8(%rbp)
	jmp	.L86
.L84:
	movq	$2, -8(%rbp)
	jmp	.L86
.L63:
	movl	-12(%rbp), %eax
	cltq
	addq	$1, %rax
	leaq	0(,%rax,8), %rdx
	movq	-32(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	print_environment_variable
	addl	$1, -12(%rbp)
	movq	$8, -8(%rbp)
	jmp	.L86
.L80:
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-32(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$8, -8(%rbp)
	jmp	.L86
.L72:
	movl	$1, %eax
	jmp	.L87
.L74:
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-32(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movq	%rax, %rsi
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$8, -8(%rbp)
	jmp	.L86
.L76:
	addl	$1, -12(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L86
.L82:
	movl	$1, -12(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L86
.L66:
	call	print_parent_process_id
	movq	$8, -8(%rbp)
	jmp	.L86
.L71:
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-32(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movzbl	(%rax), %eax
	cmpb	$45, %al
	jne	.L88
	movq	$18, -8(%rbp)
	jmp	.L86
.L88:
	movq	$4, -8(%rbp)
	jmp	.L86
.L65:
	cmpl	$1, -20(%rbp)
	jne	.L90
	movq	$6, -8(%rbp)
	jmp	.L86
.L90:
	movq	$1, -8(%rbp)
	jmp	.L86
.L68:
	movl	-12(%rbp), %eax
	addl	$1, %eax
	cmpl	%eax, -20(%rbp)
	jle	.L92
	movq	$25, -8(%rbp)
	jmp	.L86
.L92:
	movq	$11, -8(%rbp)
	jmp	.L86
.L75:
	call	print_all_environment_variables
	movq	$8, -8(%rbp)
	jmp	.L86
.L73:
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-32(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	addq	$1, %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	subl	$101, %eax
	cmpl	$16, %eax
	ja	.L94
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L96(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L96(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L96:
	.long	.L100-.L96
	.long	.L94-.L96
	.long	.L99-.L96
	.long	.L94-.L96
	.long	.L98-.L96
	.long	.L94-.L96
	.long	.L94-.L96
	.long	.L94-.L96
	.long	.L94-.L96
	.long	.L94-.L96
	.long	.L94-.L96
	.long	.L97-.L96
	.long	.L94-.L96
	.long	.L94-.L96
	.long	.L94-.L96
	.long	.L94-.L96
	.long	.L95-.L96
	.text
.L97:
	movq	$23, -8(%rbp)
	jmp	.L101
.L98:
	movq	$0, -8(%rbp)
	jmp	.L101
.L99:
	movq	$5, -8(%rbp)
	jmp	.L101
.L95:
	movq	$19, -8(%rbp)
	jmp	.L101
.L100:
	movq	$21, -8(%rbp)
	jmp	.L101
.L94:
	movq	$12, -8(%rbp)
	nop
.L101:
	jmp	.L86
.L69:
	call	print_user_ids
	movq	$8, -8(%rbp)
	jmp	.L86
.L78:
	movq	-32(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, %rsi
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$14, -8(%rbp)
	jmp	.L86
.L67:
	movl	$0, %eax
	jmp	.L87
.L79:
	call	print_group_ids
	movq	$8, -8(%rbp)
	jmp	.L86
.L83:
	call	print_process_id
	movq	$8, -8(%rbp)
	jmp	.L86
.L77:
	movl	-12(%rbp), %eax
	cmpl	-20(%rbp), %eax
	jge	.L102
	movq	$16, -8(%rbp)
	jmp	.L86
.L102:
	movq	$22, -8(%rbp)
	jmp	.L86
.L81:
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-32(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$8, -8(%rbp)
	nop
.L86:
	jmp	.L104
.L87:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	main, .-main
	.section	.rodata
	.align 8
.LC8:
	.string	"my paret's process number : %d\n"
	.text
	.globl	print_parent_process_id
	.type	print_parent_process_id, @function
print_parent_process_id:
.LFB10:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$2, -8(%rbp)
.L111:
	cmpq	$2, -8(%rbp)
	je	.L106
	cmpq	$2, -8(%rbp)
	ja	.L113
	cmpq	$0, -8(%rbp)
	je	.L108
	cmpq	$1, -8(%rbp)
	jne	.L113
	jmp	.L112
.L108:
	call	getppid@PLT
	movl	%eax, -16(%rbp)
	movl	-16(%rbp), %eax
	movl	%eax, -12(%rbp)
	movl	-12(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -8(%rbp)
	jmp	.L110
.L106:
	movq	$0, -8(%rbp)
	jmp	.L110
.L113:
	nop
.L110:
	jmp	.L111
.L112:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	print_parent_process_id, .-print_parent_process_id
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
