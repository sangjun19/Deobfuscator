	.file	"jeffmuthoki_alx-low_level_programming_100-main_opcodes_flatten.c"
	.text
	.globl	_TIG_IZ_EBy6_argv
	.bss
	.align 8
	.type	_TIG_IZ_EBy6_argv, @object
	.size	_TIG_IZ_EBy6_argv, 8
_TIG_IZ_EBy6_argv:
	.zero	8
	.globl	_TIG_IZ_EBy6_argc
	.align 4
	.type	_TIG_IZ_EBy6_argc, @object
	.size	_TIG_IZ_EBy6_argc, 4
_TIG_IZ_EBy6_argc:
	.zero	4
	.globl	_TIG_IZ_EBy6_envp
	.align 8
	.type	_TIG_IZ_EBy6_envp, @object
	.size	_TIG_IZ_EBy6_envp, 8
_TIG_IZ_EBy6_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"Error"
.LC1:
	.string	"%02x"
	.text
	.globl	main
	.type	main, @function
main:
.LFB3:
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
	movq	$0, _TIG_IZ_EBy6_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_EBy6_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_EBy6_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 102 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-EBy6--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_EBy6_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_EBy6_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_EBy6_envp(%rip)
	nop
	movq	$7, -8(%rbp)
.L33:
	cmpq	$17, -8(%rbp)
	ja	.L34
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
	.long	.L22-.L8
	.long	.L34-.L8
	.long	.L21-.L8
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L34-.L8
	.long	.L34-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L19:
	movl	$2, %eax
	jmp	.L23
.L11:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$4, -8(%rbp)
	jmp	.L24
.L10:
	movl	$0, %eax
	jmp	.L23
.L13:
	movl	-12(%rbp), %eax
	movl	%eax, %edi
	call	putchar@PLT
	addl	$1, -20(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L24
.L15:
	movl	$32, -12(%rbp)
	movq	$12, -8(%rbp)
	jmp	.L24
.L20:
	movl	$1, %eax
	jmp	.L23
.L9:
	movl	$10, -12(%rbp)
	movq	$12, -8(%rbp)
	jmp	.L24
.L14:
	movq	-48(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movl	%eax, -16(%rbp)
	movq	$17, -8(%rbp)
	jmp	.L24
.L12:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$3, -8(%rbp)
	jmp	.L24
.L7:
	cmpl	$0, -16(%rbp)
	jns	.L25
	movq	$14, -8(%rbp)
	jmp	.L24
.L25:
	movq	$0, -8(%rbp)
	jmp	.L24
.L17:
	movl	-20(%rbp), %eax
	cmpl	-16(%rbp), %eax
	jge	.L27
	movq	$2, -8(%rbp)
	jmp	.L24
.L27:
	movq	$15, -8(%rbp)
	jmp	.L24
.L18:
	movl	-16(%rbp), %eax
	subl	$1, %eax
	cmpl	%eax, -20(%rbp)
	jge	.L29
	movq	$8, -8(%rbp)
	jmp	.L24
.L29:
	movq	$16, -8(%rbp)
	jmp	.L24
.L22:
	movl	$0, -20(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L24
.L16:
	cmpl	$2, -36(%rbp)
	je	.L31
	movq	$13, -8(%rbp)
	jmp	.L24
.L31:
	movq	$9, -8(%rbp)
	jmp	.L24
.L21:
	movl	-20(%rbp), %eax
	cltq
	leaq	main(%rip), %rdx
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	movzbl	%al, %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$5, -8(%rbp)
	jmp	.L24
.L34:
	nop
.L24:
	jmp	.L33
.L23:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
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
