	.file	"derrell_LearnCS_t040-case-enum_flatten.c"
	.text
	.globl	_TIG_IZ_EeXx_argv
	.bss
	.align 8
	.type	_TIG_IZ_EeXx_argv, @object
	.size	_TIG_IZ_EeXx_argv, 8
_TIG_IZ_EeXx_argv:
	.zero	8
	.globl	_TIG_IZ_EeXx_envp
	.align 8
	.type	_TIG_IZ_EeXx_envp, @object
	.size	_TIG_IZ_EeXx_envp, 8
_TIG_IZ_EeXx_envp:
	.zero	8
	.globl	_TIG_IZ_EeXx_argc
	.align 4
	.type	_TIG_IZ_EeXx_argc, @object
	.size	_TIG_IZ_EeXx_argc, 4
_TIG_IZ_EeXx_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"hello world"
.LC1:
	.string	"hi there"
.LC2:
	.string	"case 0"
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
	subq	$48, %rsp
	movl	%edi, -20(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movq	$0, _TIG_IZ_EeXx_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_EeXx_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_EeXx_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 99 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-EeXx--0
# 0 "" 2
#NO_APP
	movl	-20(%rbp), %eax
	movl	%eax, _TIG_IZ_EeXx_argc(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, _TIG_IZ_EeXx_argv(%rip)
	movq	-40(%rbp), %rax
	movq	%rax, _TIG_IZ_EeXx_envp(%rip)
	nop
	movq	$3, -8(%rbp)
.L15:
	cmpq	$7, -8(%rbp)
	ja	.L17
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
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L17-.L8
	.long	.L10-.L8
	.long	.L17-.L8
	.long	.L9-.L8
	.long	.L17-.L8
	.long	.L7-.L8
	.text
.L11:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$7, -8(%rbp)
	jmp	.L13
.L10:
	movq	$1, -8(%rbp)
	nop
	jmp	.L13
.L9:
	movq	$7, -8(%rbp)
	jmp	.L13
.L12:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$7, -8(%rbp)
	jmp	.L13
.L7:
	movl	$0, %eax
	jmp	.L16
.L17:
	nop
.L13:
	jmp	.L15
.L16:
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
