	.file	"nazmusweb-coding_C-language_read_and_write_flatten.c"
	.text
	.globl	_TIG_IZ_HGfs_argc
	.bss
	.align 4
	.type	_TIG_IZ_HGfs_argc, @object
	.size	_TIG_IZ_HGfs_argc, 4
_TIG_IZ_HGfs_argc:
	.zero	4
	.globl	_TIG_IZ_HGfs_envp
	.align 8
	.type	_TIG_IZ_HGfs_envp, @object
	.size	_TIG_IZ_HGfs_envp, 8
_TIG_IZ_HGfs_envp:
	.zero	8
	.globl	_TIG_IZ_HGfs_argv
	.align 8
	.type	_TIG_IZ_HGfs_argv, @object
	.size	_TIG_IZ_HGfs_argv, 8
_TIG_IZ_HGfs_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"r"
.LC1:
	.string	"test.txt"
.LC2:
	.string	"%c\n"
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
	subq	$80, %rsp
	movl	%edi, -52(%rbp)
	movq	%rsi, -64(%rbp)
	movq	%rdx, -72(%rbp)
	movq	$0, _TIG_IZ_HGfs_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_HGfs_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_HGfs_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 112 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-HGfs--0
# 0 "" 2
#NO_APP
	movl	-52(%rbp), %eax
	movl	%eax, _TIG_IZ_HGfs_argc(%rip)
	movq	-64(%rbp), %rax
	movq	%rax, _TIG_IZ_HGfs_argv(%rip)
	movq	-72(%rbp), %rax
	movq	%rax, _TIG_IZ_HGfs_envp(%rip)
	nop
	movq	$2, -16(%rbp)
.L11:
	cmpq	$2, -16(%rbp)
	je	.L6
	cmpq	$2, -16(%rbp)
	ja	.L13
	cmpq	$0, -16(%rbp)
	je	.L8
	cmpq	$1, -16(%rbp)
	jne	.L13
	leaq	.LC0(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	fgetc@PLT
	movl	%eax, -36(%rbp)
	movl	-36(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	fgetc@PLT
	movl	%eax, -32(%rbp)
	movl	-32(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	fgetc@PLT
	movl	%eax, -28(%rbp)
	movl	-28(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	fgetc@PLT
	movl	%eax, -24(%rbp)
	movl	-24(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	fgetc@PLT
	movl	%eax, -20(%rbp)
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-8(%rbp), %rax
	movq	%rax, %rsi
	movl	$77, %edi
	call	fputc@PLT
	movq	-8(%rbp), %rax
	movq	%rax, %rsi
	movl	$65, %edi
	call	fputc@PLT
	movq	-8(%rbp), %rax
	movq	%rax, %rsi
	movl	$78, %edi
	call	fputc@PLT
	movq	-8(%rbp), %rax
	movq	%rax, %rsi
	movl	$71, %edi
	call	fputc@PLT
	movq	-8(%rbp), %rax
	movq	%rax, %rsi
	movl	$79, %edi
	call	fputc@PLT
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$0, -16(%rbp)
	jmp	.L9
.L8:
	movl	$0, %eax
	jmp	.L12
.L6:
	movq	$1, -16(%rbp)
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
