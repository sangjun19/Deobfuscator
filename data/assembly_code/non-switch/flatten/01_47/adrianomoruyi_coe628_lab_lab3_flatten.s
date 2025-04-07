	.file	"adrianomoruyi_coe628_lab_lab3_flatten.c"
	.text
	.globl	_TIG_IZ_7SXy_argv
	.bss
	.align 8
	.type	_TIG_IZ_7SXy_argv, @object
	.size	_TIG_IZ_7SXy_argv, 8
_TIG_IZ_7SXy_argv:
	.zero	8
	.globl	_TIG_IZ_7SXy_argc
	.align 4
	.type	_TIG_IZ_7SXy_argc, @object
	.size	_TIG_IZ_7SXy_argc, 4
_TIG_IZ_7SXy_argc:
	.zero	4
	.globl	_TIG_IZ_7SXy_envp
	.align 8
	.type	_TIG_IZ_7SXy_envp, @object
	.size	_TIG_IZ_7SXy_envp, 8
_TIG_IZ_7SXy_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"Fork failed!"
	.align 8
.LC1:
	.string	"Command excecution unsucessful"
	.text
	.globl	execute_command
	.type	execute_command, @function
execute_command:
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
	movl	%esi, -44(%rbp)
	movq	$3, -8(%rbp)
.L25:
	cmpq	$13, -8(%rbp)
	ja	.L26
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
	.long	.L26-.L4
	.long	.L14-.L4
	.long	.L13-.L4
	.long	.L12-.L4
	.long	.L11-.L4
	.long	.L26-.L4
	.long	.L10-.L4
	.long	.L9-.L4
	.long	.L8-.L4
	.long	.L26-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L27-.L4
	.long	.L3-.L4
	.text
.L11:
	cmpl	$0, -44(%rbp)
	jne	.L15
	movq	$11, -8(%rbp)
	jmp	.L17
.L15:
	movq	$12, -8(%rbp)
	jmp	.L17
.L8:
	cmpl	$0, -20(%rbp)
	jns	.L19
	movq	$10, -8(%rbp)
	jmp	.L17
.L19:
	movq	$2, -8(%rbp)
	jmp	.L17
.L14:
	movq	-40(%rbp), %rax
	movq	(%rax), %rax
	movq	-40(%rbp), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	execvp@PLT
	movl	%eax, -16(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L17
.L12:
	movq	$13, -8(%rbp)
	jmp	.L17
.L6:
	movl	-20(%rbp), %eax
	movl	$0, %edx
	movl	$0, %esi
	movl	%eax, %edi
	call	waitpid@PLT
	movq	$12, -8(%rbp)
	jmp	.L17
.L3:
	call	fork@PLT
	movl	%eax, -12(%rbp)
	movl	-12(%rbp), %eax
	movl	%eax, -20(%rbp)
	movq	$8, -8(%rbp)
	jmp	.L17
.L10:
	cmpl	$0, -16(%rbp)
	jns	.L21
	movq	$7, -8(%rbp)
	jmp	.L17
.L21:
	movq	$4, -8(%rbp)
	jmp	.L17
.L7:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L9:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L13:
	cmpl	$0, -20(%rbp)
	jne	.L23
	movq	$1, -8(%rbp)
	jmp	.L17
.L23:
	movq	$4, -8(%rbp)
	jmp	.L17
.L26:
	nop
.L17:
	jmp	.L25
.L27:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	execute_command, .-execute_command
	.section	.rodata
.LC2:
	.string	" \n"
.LC3:
	.string	"&"
	.text
	.globl	parse_input
	.type	parse_input, @function
parse_input:
.LFB5:
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
	movq	$6, -16(%rbp)
.L50:
	cmpq	$13, -16(%rbp)
	ja	.L51
	movq	-16(%rbp), %rax
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
	.long	.L51-.L31
	.long	.L41-.L31
	.long	.L40-.L31
	.long	.L39-.L31
	.long	.L51-.L31
	.long	.L38-.L31
	.long	.L37-.L31
	.long	.L36-.L31
	.long	.L35-.L31
	.long	.L51-.L31
	.long	.L52-.L31
	.long	.L33-.L31
	.long	.L32-.L31
	.long	.L30-.L31
	.text
.L32:
	movl	$0, -36(%rbp)
	movq	-72(%rbp), %rax
	movl	$0, (%rax)
	movq	-56(%rbp), %rax
	leaq	.LC2(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strtok@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	$1, -16(%rbp)
	jmp	.L42
.L35:
	movl	-36(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-64(%rbp), %rax
	addq	%rdx, %rax
	movq	$0, (%rax)
	movq	$10, -16(%rbp)
	jmp	.L42
.L41:
	cmpq	$0, -24(%rbp)
	je	.L43
	movq	$3, -16(%rbp)
	jmp	.L42
.L43:
	movq	$5, -16(%rbp)
	jmp	.L42
.L39:
	movl	-36(%rbp), %eax
	movl	%eax, -28(%rbp)
	addl	$1, -36(%rbp)
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-64(%rbp), %rax
	addq	%rax, %rdx
	movq	-24(%rbp), %rax
	movq	%rax, (%rdx)
	leaq	.LC2(%rip), %rax
	movq	%rax, %rsi
	movl	$0, %edi
	call	strtok@PLT
	movq	%rax, -24(%rbp)
	movq	$1, -16(%rbp)
	jmp	.L42
.L33:
	cmpl	$0, -32(%rbp)
	jne	.L45
	movq	$2, -16(%rbp)
	jmp	.L42
.L45:
	movq	$13, -16(%rbp)
	jmp	.L42
.L30:
	movl	-36(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-64(%rbp), %rax
	addq	%rdx, %rax
	movq	$0, (%rax)
	movq	$10, -16(%rbp)
	jmp	.L42
.L37:
	movq	$12, -16(%rbp)
	jmp	.L42
.L38:
	cmpl	$0, -36(%rbp)
	jle	.L47
	movq	$7, -16(%rbp)
	jmp	.L42
.L47:
	movq	$8, -16(%rbp)
	jmp	.L42
.L36:
	movl	-36(%rbp), %eax
	cltq
	salq	$3, %rax
	leaq	-8(%rax), %rdx
	movq	-64(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	leaq	.LC3(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -32(%rbp)
	movq	$11, -16(%rbp)
	jmp	.L42
.L40:
	movq	-72(%rbp), %rax
	movl	$1, (%rax)
	movl	-36(%rbp), %eax
	cltq
	salq	$3, %rax
	leaq	-8(%rax), %rdx
	movq	-64(%rbp), %rax
	addq	%rdx, %rax
	movq	$0, (%rax)
	movq	$10, -16(%rbp)
	jmp	.L42
.L51:
	nop
.L42:
	jmp	.L50
.L52:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	parse_input, .-parse_input
	.section	.rodata
.LC4:
	.string	"Your command> "
.LC5:
	.string	"Exiting shell..."
.LC6:
	.string	"exit"
	.text
	.globl	main
	.type	main, @function
main:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$256, %rsp
	movl	%edi, -228(%rbp)
	movq	%rsi, -240(%rbp)
	movq	%rdx, -248(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_7SXy_envp(%rip)
	nop
.L54:
	movq	$0, _TIG_IZ_7SXy_argv(%rip)
	nop
.L55:
	movl	$0, _TIG_IZ_7SXy_argc(%rip)
	nop
	nop
.L56:
.L57:
#APP
# 125 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-7SXy--0
# 0 "" 2
#NO_APP
	movl	-228(%rbp), %eax
	movl	%eax, _TIG_IZ_7SXy_argc(%rip)
	movq	-240(%rbp), %rax
	movq	%rax, _TIG_IZ_7SXy_argv(%rip)
	movq	-248(%rbp), %rax
	movq	%rax, _TIG_IZ_7SXy_envp(%rip)
	nop
	movq	$3, -200(%rbp)
.L78:
	cmpq	$13, -200(%rbp)
	ja	.L81
	movq	-200(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L60(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L60(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L60:
	.long	.L81-.L60
	.long	.L69-.L60
	.long	.L68-.L60
	.long	.L67-.L60
	.long	.L81-.L60
	.long	.L66-.L60
	.long	.L65-.L60
	.long	.L64-.L60
	.long	.L63-.L60
	.long	.L81-.L60
	.long	.L62-.L60
	.long	.L61-.L60
	.long	.L81-.L60
	.long	.L59-.L60
	.text
.L63:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	stdout(%rip), %rax
	movq	%rax, %rdi
	call	fflush@PLT
	movq	stdin(%rip), %rdx
	leaq	-112(%rbp), %rax
	movl	$10, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	movq	%rax, -208(%rbp)
	movq	$7, -200(%rbp)
	jmp	.L70
.L69:
	movl	-216(%rbp), %edx
	leaq	-192(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	execute_command
	movq	$8, -200(%rbp)
	jmp	.L70
.L67:
	movq	$8, -200(%rbp)
	jmp	.L70
.L61:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$10, -200(%rbp)
	jmp	.L70
.L59:
	cmpl	$0, -212(%rbp)
	jne	.L71
	movq	$11, -200(%rbp)
	jmp	.L70
.L71:
	movq	$1, -200(%rbp)
	jmp	.L70
.L65:
	movq	-184(%rbp), %rax
	testq	%rax, %rax
	je	.L73
	movq	$8, -200(%rbp)
	jmp	.L70
.L73:
	movq	$5, -200(%rbp)
	jmp	.L70
.L66:
	movq	-192(%rbp), %rax
	leaq	.LC6(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -212(%rbp)
	movq	$13, -200(%rbp)
	jmp	.L70
.L62:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L79
	jmp	.L80
.L64:
	cmpq	$0, -208(%rbp)
	jne	.L76
	movq	$11, -200(%rbp)
	jmp	.L70
.L76:
	movq	$2, -200(%rbp)
	jmp	.L70
.L68:
	leaq	-216(%rbp), %rdx
	leaq	-192(%rbp), %rcx
	leaq	-112(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	parse_input
	movq	$6, -200(%rbp)
	jmp	.L70
.L81:
	nop
.L70:
	jmp	.L78
.L80:
	call	__stack_chk_fail@PLT
.L79:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
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
