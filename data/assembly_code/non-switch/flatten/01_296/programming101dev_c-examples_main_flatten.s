	.file	"programming101dev_c-examples_main_flatten.c"
	.text
	.globl	_TIG_IZ_jIUQ_argc
	.bss
	.align 4
	.type	_TIG_IZ_jIUQ_argc, @object
	.size	_TIG_IZ_jIUQ_argc, 4
_TIG_IZ_jIUQ_argc:
	.zero	4
	.globl	_TIG_IZ_jIUQ_envp
	.align 8
	.type	_TIG_IZ_jIUQ_envp, @object
	.size	_TIG_IZ_jIUQ_envp, 8
_TIG_IZ_jIUQ_envp:
	.zero	8
	.globl	_TIG_IZ_jIUQ_argv
	.align 8
	.type	_TIG_IZ_jIUQ_argv, @object
	.size	_TIG_IZ_jIUQ_argv, 8
_TIG_IZ_jIUQ_argv:
	.zero	8
	.section	.rodata
	.align 8
.LC0:
	.string	"File descriptor for stdin: %d\n"
	.align 8
.LC1:
	.string	"File descriptor for stdout: %d\n"
	.align 8
.LC2:
	.string	"File descriptor for stderr: %d\n"
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
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	$0, _TIG_IZ_jIUQ_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_jIUQ_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_jIUQ_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 109 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-jIUQ--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_jIUQ_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_jIUQ_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_jIUQ_envp(%rip)
	nop
	movq	$2, -8(%rbp)
.L11:
	cmpq	$2, -8(%rbp)
	je	.L6
	cmpq	$2, -8(%rbp)
	ja	.L13
	cmpq	$0, -8(%rbp)
	je	.L8
	cmpq	$1, -8(%rbp)
	jne	.L13
	movq	stdin(%rip), %rax
	movq	%rax, %rdi
	call	fileno@PLT
	movl	%eax, -20(%rbp)
	movq	stdout(%rip), %rax
	movq	%rax, %rdi
	call	fileno@PLT
	movl	%eax, -16(%rbp)
	movq	stderr(%rip), %rax
	movq	%rax, %rdi
	call	fileno@PLT
	movl	%eax, -12(%rbp)
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-16(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-12(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$0, -8(%rbp)
	jmp	.L9
.L8:
	movl	$0, %eax
	jmp	.L12
.L6:
	movq	$1, -8(%rbp)
	jmp	.L9
.L13:
	nop
.L9:
	jmp	.L11
.L12:
	leave
	.cfi_def_cfa 7, 8
	ret
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
