	.file	"AvTe_Hacktoberfest-2023-new-student_filehandeling_flatten.c"
	.text
	.globl	_TIG_IZ_L04n_envp
	.bss
	.align 8
	.type	_TIG_IZ_L04n_envp, @object
	.size	_TIG_IZ_L04n_envp, 8
_TIG_IZ_L04n_envp:
	.zero	8
	.globl	_TIG_IZ_L04n_argc
	.align 4
	.type	_TIG_IZ_L04n_argc, @object
	.size	_TIG_IZ_L04n_argc, 4
_TIG_IZ_L04n_argc:
	.zero	4
	.globl	_TIG_IZ_L04n_argv
	.align 8
	.type	_TIG_IZ_L04n_argv, @object
	.size	_TIG_IZ_L04n_argv, 8
_TIG_IZ_L04n_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"Hello, File!\n"
.LC1:
	.string	"w"
.LC2:
	.string	"sample.txt"
.LC3:
	.string	"File could not be opened."
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
	subq	$48, %rsp
	movl	%edi, -20(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movq	$0, _TIG_IZ_L04n_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_L04n_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_L04n_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 121 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-L04n--0
# 0 "" 2
#NO_APP
	movl	-20(%rbp), %eax
	movl	%eax, _TIG_IZ_L04n_argc(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, _TIG_IZ_L04n_argv(%rip)
	movq	-40(%rbp), %rax
	movq	%rax, _TIG_IZ_L04n_envp(%rip)
	nop
	movq	$5, -8(%rbp)
.L18:
	cmpq	$6, -8(%rbp)
	ja	.L19
	movq	-8(%rbp), %rax
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
	.long	.L19-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L10:
	cmpq	$0, -16(%rbp)
	jne	.L14
	movq	$2, -8(%rbp)
	jmp	.L16
.L14:
	movq	$6, -8(%rbp)
	jmp	.L16
.L13:
	movl	$1, %eax
	jmp	.L17
.L11:
	movl	$0, %eax
	jmp	.L17
.L7:
	movq	-16(%rbp), %rax
	movq	%rax, %rcx
	movl	$13, %edx
	movl	$1, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	-16(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$3, -8(%rbp)
	jmp	.L16
.L9:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -16(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L16
.L12:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -8(%rbp)
	jmp	.L16
.L19:
	nop
.L16:
	jmp	.L18
.L17:
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
