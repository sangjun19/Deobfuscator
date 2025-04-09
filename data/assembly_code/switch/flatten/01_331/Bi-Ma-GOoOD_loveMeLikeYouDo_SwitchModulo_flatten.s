	.file	"Bi-Ma-GOoOD_loveMeLikeYouDo_SwitchModulo_flatten.c"
	.text
	.globl	_TIG_IZ_yaIe_argv
	.bss
	.align 8
	.type	_TIG_IZ_yaIe_argv, @object
	.size	_TIG_IZ_yaIe_argv, 8
_TIG_IZ_yaIe_argv:
	.zero	8
	.globl	_TIG_IZ_yaIe_argc
	.align 4
	.type	_TIG_IZ_yaIe_argc, @object
	.size	_TIG_IZ_yaIe_argc, 4
_TIG_IZ_yaIe_argc:
	.zero	4
	.globl	_TIG_IZ_yaIe_envp
	.align 8
	.type	_TIG_IZ_yaIe_envp, @object
	.size	_TIG_IZ_yaIe_envp, 8
_TIG_IZ_yaIe_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"%d is even number.\n"
.LC1:
	.string	"%d is odd number.\n"
	.align 8
.LC2:
	.string	"This program will find what is your kind of number.\nsuch as\n\t1 is odd number.\n\t2 is even number."
.LC3:
	.string	"Enter your number : "
.LC4:
	.string	"%d"
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
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_yaIe_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_yaIe_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_yaIe_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 108 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-yaIe--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_yaIe_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_yaIe_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_yaIe_envp(%rip)
	nop
	movq	$3, -16(%rbp)
.L18:
	cmpq	$7, -16(%rbp)
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
	.long	.L21-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L21-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L13:
	cmpl	$0, -20(%rbp)
	jne	.L14
	movq	$6, -16(%rbp)
	jmp	.L15
.L14:
	movq	$7, -16(%rbp)
	nop
.L15:
	jmp	.L16
.L11:
	movq	$2, -16(%rbp)
	jmp	.L16
.L9:
	movl	-24(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$5, -16(%rbp)
	jmp	.L16
.L10:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L19
	jmp	.L20
.L7:
	movl	-24(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$5, -16(%rbp)
	jmp	.L16
.L12:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-24(%rbp), %eax
	cltd
	shrl	$31, %edx
	addl	%edx, %eax
	andl	$1, %eax
	subl	%edx, %eax
	movl	%eax, -20(%rbp)
	movq	$1, -16(%rbp)
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
