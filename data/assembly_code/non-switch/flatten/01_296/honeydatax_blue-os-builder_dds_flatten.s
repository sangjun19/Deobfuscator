	.file	"honeydatax_blue-os-builder_dds_flatten.c"
	.text
	.globl	_TIG_IZ_rpcx_envp
	.bss
	.align 8
	.type	_TIG_IZ_rpcx_envp, @object
	.size	_TIG_IZ_rpcx_envp, 8
_TIG_IZ_rpcx_envp:
	.zero	8
	.globl	_TIG_IZ_rpcx_argc
	.align 4
	.type	_TIG_IZ_rpcx_argc, @object
	.size	_TIG_IZ_rpcx_argc, 4
_TIG_IZ_rpcx_argc:
	.zero	4
	.globl	_TIG_IZ_rpcx_argv
	.align 8
	.type	_TIG_IZ_rpcx_argv, @object
	.size	_TIG_IZ_rpcx_argv, 8
_TIG_IZ_rpcx_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"r"
.LC1:
	.string	"r+"
.LC2:
	.string	"%s %s"
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
	subq	$592, %rsp
	movl	%edi, -564(%rbp)
	movq	%rsi, -576(%rbp)
	movq	%rdx, -584(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_rpcx_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_rpcx_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_rpcx_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 126 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-rpcx--0
# 0 "" 2
#NO_APP
	movl	-564(%rbp), %eax
	movl	%eax, _TIG_IZ_rpcx_argc(%rip)
	movq	-576(%rbp), %rax
	movq	%rax, _TIG_IZ_rpcx_argv(%rip)
	movq	-584(%rbp), %rax
	movq	%rax, _TIG_IZ_rpcx_envp(%rip)
	nop
	movq	$1, -552(%rbp)
.L13:
	cmpq	$2, -552(%rbp)
	je	.L6
	cmpq	$2, -552(%rbp)
	ja	.L16
	cmpq	$0, -552(%rbp)
	je	.L8
	cmpq	$1, -552(%rbp)
	jne	.L16
	cmpl	$2, -564(%rbp)
	jle	.L9
	movq	$0, -552(%rbp)
	jmp	.L11
.L9:
	movq	$2, -552(%rbp)
	jmp	.L11
.L8:
	movq	-576(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	leaq	.LC0(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -544(%rbp)
	movq	-576(%rbp), %rax
	addq	$16, %rax
	movq	(%rax), %rax
	leaq	.LC1(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -536(%rbp)
	movq	-544(%rbp), %rdx
	leaq	-528(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$2, %esi
	movq	%rax, %rdi
	call	fread@PLT
	movq	-536(%rbp), %rdx
	leaq	-528(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$2, %esi
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	-544(%rbp), %rdx
	leaq	-528(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$57, %esi
	movq	%rax, %rdi
	call	fread@PLT
	movq	-536(%rbp), %rdx
	leaq	-528(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$57, %esi
	movq	%rax, %rdi
	call	fread@PLT
	movq	-544(%rbp), %rdx
	leaq	-528(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$450, %esi
	movq	%rax, %rdi
	call	fread@PLT
	movq	-536(%rbp), %rdx
	leaq	-528(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$450, %esi
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	-544(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	-536(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	-576(%rbp), %rax
	addq	$16, %rax
	movq	(%rax), %rdx
	movq	-576(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$2, -552(%rbp)
	jmp	.L11
.L6:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L14
	jmp	.L15
.L16:
	nop
.L11:
	jmp	.L13
.L15:
	call	__stack_chk_fail@PLT
.L14:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
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
