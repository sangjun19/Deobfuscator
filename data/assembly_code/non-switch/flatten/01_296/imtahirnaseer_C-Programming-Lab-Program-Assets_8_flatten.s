	.file	"imtahirnaseer_C-Programming-Lab-Program-Assets_8_flatten.c"
	.text
	.globl	_TIG_IZ_b3iY_envp
	.bss
	.align 8
	.type	_TIG_IZ_b3iY_envp, @object
	.size	_TIG_IZ_b3iY_envp, 8
_TIG_IZ_b3iY_envp:
	.zero	8
	.globl	_TIG_IZ_b3iY_argc
	.align 4
	.type	_TIG_IZ_b3iY_argc, @object
	.size	_TIG_IZ_b3iY_argc, 4
_TIG_IZ_b3iY_argc:
	.zero	4
	.globl	_TIG_IZ_b3iY_argv
	.align 8
	.type	_TIG_IZ_b3iY_argv, @object
	.size	_TIG_IZ_b3iY_argv, 8
_TIG_IZ_b3iY_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"%d is the largest number\n"
.LC1:
	.string	"Enter three numbers: "
.LC2:
	.string	"%d %d %d"
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
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_b3iY_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_b3iY_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_b3iY_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 105 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-b3iY--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_b3iY_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_b3iY_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_b3iY_envp(%rip)
	nop
	movq	$4, -16(%rbp)
.L29:
	cmpq	$10, -16(%rbp)
	ja	.L32
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
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L14:
	movq	$0, -16(%rbp)
	jmp	.L19
.L10:
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -16(%rbp)
	jmp	.L19
.L17:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L30
	jmp	.L31
.L15:
	movl	-24(%rbp), %edx
	movl	-20(%rbp), %eax
	cmpl	%eax, %edx
	jl	.L21
	movq	$2, -16(%rbp)
	jmp	.L19
.L21:
	movq	$10, -16(%rbp)
	jmp	.L19
.L9:
	movl	-28(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -16(%rbp)
	jmp	.L19
.L12:
	movl	-28(%rbp), %edx
	movl	-20(%rbp), %eax
	cmpl	%eax, %edx
	jl	.L23
	movq	$9, -16(%rbp)
	jmp	.L19
.L23:
	movq	$5, -16(%rbp)
	jmp	.L19
.L13:
	movl	-24(%rbp), %edx
	movl	-28(%rbp), %eax
	cmpl	%eax, %edx
	jl	.L25
	movq	$3, -16(%rbp)
	jmp	.L19
.L25:
	movq	$8, -16(%rbp)
	jmp	.L19
.L7:
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -16(%rbp)
	jmp	.L19
.L18:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-20(%rbp), %rcx
	leaq	-24(%rbp), %rdx
	leaq	-28(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$7, -16(%rbp)
	jmp	.L19
.L11:
	movl	-28(%rbp), %edx
	movl	-24(%rbp), %eax
	cmpl	%eax, %edx
	jl	.L27
	movq	$6, -16(%rbp)
	jmp	.L19
.L27:
	movq	$5, -16(%rbp)
	jmp	.L19
.L16:
	movl	-24(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -16(%rbp)
	jmp	.L19
.L32:
	nop
.L19:
	jmp	.L29
.L31:
	call	__stack_chk_fail@PLT
.L30:
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
