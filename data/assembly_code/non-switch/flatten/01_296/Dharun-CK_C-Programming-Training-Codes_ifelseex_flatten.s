	.file	"Dharun-CK_C-Programming-Training-Codes_ifelseex_flatten.c"
	.text
	.globl	_TIG_IZ_70jc_argv
	.bss
	.align 8
	.type	_TIG_IZ_70jc_argv, @object
	.size	_TIG_IZ_70jc_argv, 8
_TIG_IZ_70jc_argv:
	.zero	8
	.globl	_TIG_IZ_70jc_envp
	.align 8
	.type	_TIG_IZ_70jc_envp, @object
	.size	_TIG_IZ_70jc_envp, 8
_TIG_IZ_70jc_envp:
	.zero	8
	.globl	_TIG_IZ_70jc_argc
	.align 4
	.type	_TIG_IZ_70jc_argc, @object
	.size	_TIG_IZ_70jc_argc, 4
_TIG_IZ_70jc_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"your mark is %d"
.LC1:
	.string	"ENTER THE Mark: "
.LC2:
	.string	"%d"
	.align 8
.LC3:
	.string	"ENTER THE Handwriting Condition: "
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
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_70jc_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_70jc_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_70jc_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 125 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-70jc--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_70jc_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_70jc_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_70jc_envp(%rip)
	nop
	movq	$5, -16(%rbp)
.L18:
	cmpq	$6, -16(%rbp)
	ja	.L21
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
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L21-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L10:
	movl	-20(%rbp), %eax
	cmpl	$1, %eax
	jne	.L14
	movq	$2, -16(%rbp)
	jmp	.L16
.L14:
	movq	$1, -16(%rbp)
	jmp	.L16
.L12:
	movl	-24(%rbp), %eax
	subl	$10, %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$0, -16(%rbp)
	jmp	.L16
.L7:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-20(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$4, -16(%rbp)
	jmp	.L16
.L9:
	movq	$6, -16(%rbp)
	jmp	.L16
.L13:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L19
	jmp	.L20
.L11:
	movl	-24(%rbp), %eax
	addl	$10, %eax
	movl	%eax, -24(%rbp)
	movl	-24(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$0, -16(%rbp)
	jmp	.L16
.L21:
	nop
.L16:
	jmp	.L18
.L20:
	call	__stack_chk_fail@PLT
.L19:
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
