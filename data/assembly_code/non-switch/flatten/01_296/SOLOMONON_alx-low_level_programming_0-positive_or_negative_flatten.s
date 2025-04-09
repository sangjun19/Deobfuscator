	.file	"SOLOMONON_alx-low_level_programming_0-positive_or_negative_flatten.c"
	.text
	.globl	_TIG_IZ_QZim_argv
	.bss
	.align 8
	.type	_TIG_IZ_QZim_argv, @object
	.size	_TIG_IZ_QZim_argv, 8
_TIG_IZ_QZim_argv:
	.zero	8
	.globl	_TIG_IZ_QZim_envp
	.align 8
	.type	_TIG_IZ_QZim_envp, @object
	.size	_TIG_IZ_QZim_envp, 8
_TIG_IZ_QZim_envp:
	.zero	8
	.globl	_TIG_IZ_QZim_argc
	.align 4
	.type	_TIG_IZ_QZim_argc, @object
	.size	_TIG_IZ_QZim_argc, 4
_TIG_IZ_QZim_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"%d is positive\n"
.LC1:
	.string	"%d is zero\n"
.LC2:
	.string	"%d is negative\n"
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
	movq	$0, _TIG_IZ_QZim_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_QZim_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_QZim_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 109 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-QZim--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_QZim_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_QZim_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_QZim_envp(%rip)
	nop
	movq	$3, -16(%rbp)
.L22:
	cmpq	$7, -16(%rbp)
	ja	.L24
	movq	-16(%rbp), %rax
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
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L11:
	cmpl	$0, -24(%rbp)
	jns	.L16
	movq	$7, -16(%rbp)
	jmp	.L18
.L16:
	movq	$0, -16(%rbp)
	jmp	.L18
.L14:
	cmpl	$0, -24(%rbp)
	jle	.L19
	movq	$6, -16(%rbp)
	jmp	.L18
.L19:
	movq	$4, -16(%rbp)
	jmp	.L18
.L12:
	movq	$2, -16(%rbp)
	jmp	.L18
.L9:
	movl	-24(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$5, -16(%rbp)
	jmp	.L18
.L10:
	movl	$0, %eax
	jmp	.L23
.L15:
	movl	-24(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$5, -16(%rbp)
	jmp	.L18
.L7:
	movl	-24(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$5, -16(%rbp)
	jmp	.L18
.L13:
	movl	$0, %edi
	call	time@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movl	%eax, %edi
	call	srand@PLT
	call	rand@PLT
	movl	%eax, -20(%rbp)
	movl	-20(%rbp), %eax
	subl	$1073741823, %eax
	movl	%eax, -24(%rbp)
	movq	$1, -16(%rbp)
	jmp	.L18
.L24:
	nop
.L18:
	jmp	.L22
.L23:
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
