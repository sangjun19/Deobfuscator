	.file	"varun-al_CPS-Programming-Assigment_6_flatten.c"
	.text
	.globl	_TIG_IZ_IU3O_envp
	.bss
	.align 8
	.type	_TIG_IZ_IU3O_envp, @object
	.size	_TIG_IZ_IU3O_envp, 8
_TIG_IZ_IU3O_envp:
	.zero	8
	.globl	_TIG_IZ_IU3O_argc
	.align 4
	.type	_TIG_IZ_IU3O_argc, @object
	.size	_TIG_IZ_IU3O_argc, 4
_TIG_IZ_IU3O_argc:
	.zero	4
	.globl	_TIG_IZ_IU3O_argv
	.align 8
	.type	_TIG_IZ_IU3O_argv, @object
	.size	_TIG_IZ_IU3O_argv, 8
_TIG_IZ_IU3O_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"Enter the number: "
.LC1:
	.string	"%d"
.LC2:
	.string	"%d*%d=%d\n"
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
	movq	$0, _TIG_IZ_IU3O_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_IU3O_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_IU3O_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 122 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-IU3O--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_IU3O_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_IU3O_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_IU3O_envp(%rip)
	nop
	movq	$5, -16(%rbp)
.L17:
	cmpq	$7, -16(%rbp)
	ja	.L20
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
	.long	.L20-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L20-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L20-.L8
	.long	.L21-.L8
	.text
.L10:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	$1, -20(%rbp)
	movq	$1, -16(%rbp)
	jmp	.L13
.L12:
	cmpl	$10, -20(%rbp)
	jg	.L14
	movq	$2, -16(%rbp)
	jmp	.L13
.L14:
	movq	$7, -16(%rbp)
	jmp	.L13
.L9:
	movq	$4, -16(%rbp)
	jmp	.L13
.L11:
	movl	-24(%rbp), %eax
	imull	-20(%rbp), %eax
	movl	%eax, %ecx
	movl	-24(%rbp), %eax
	movl	-20(%rbp), %edx
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -20(%rbp)
	movq	$1, -16(%rbp)
	jmp	.L13
.L20:
	nop
.L13:
	jmp	.L17
.L21:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L19
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
