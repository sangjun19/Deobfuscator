	.file	"wwegiel111_Elementary_Tasks_C_main3_flatten.c"
	.text
	.globl	_TIG_IZ_G04e_argc
	.bss
	.align 4
	.type	_TIG_IZ_G04e_argc, @object
	.size	_TIG_IZ_G04e_argc, 4
_TIG_IZ_G04e_argc:
	.zero	4
	.globl	_TIG_IZ_G04e_argv
	.align 8
	.type	_TIG_IZ_G04e_argv, @object
	.size	_TIG_IZ_G04e_argv, 8
_TIG_IZ_G04e_argv:
	.zero	8
	.globl	_TIG_IZ_G04e_envp
	.align 8
	.type	_TIG_IZ_G04e_envp, @object
	.size	_TIG_IZ_G04e_envp, 8
_TIG_IZ_G04e_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"Enter two numbers: "
.LC1:
	.string	"%lf %lf"
	.align 8
.LC3:
	.string	"The average of %.2lf and %.2lf is %.2lf\n"
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
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_G04e_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_G04e_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_G04e_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 109 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-G04e--0
# 0 "" 2
#NO_APP
	movl	-52(%rbp), %eax
	movl	%eax, _TIG_IZ_G04e_argc(%rip)
	movq	-64(%rbp), %rax
	movq	%rax, _TIG_IZ_G04e_argv(%rip)
	movq	-72(%rbp), %rax
	movq	%rax, _TIG_IZ_G04e_envp(%rip)
	nop
	movq	$2, -24(%rbp)
.L11:
	cmpq	$2, -24(%rbp)
	je	.L6
	cmpq	$2, -24(%rbp)
	ja	.L14
	cmpq	$0, -24(%rbp)
	je	.L8
	cmpq	$1, -24(%rbp)
	jne	.L14
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-32(%rbp), %rdx
	leaq	-40(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movsd	-40(%rbp), %xmm1
	movsd	-32(%rbp), %xmm0
	addsd	%xmm1, %xmm0
	movsd	.LC2(%rip), %xmm1
	divsd	%xmm1, %xmm0
	movsd	%xmm0, -16(%rbp)
	movsd	-32(%rbp), %xmm0
	movq	-40(%rbp), %rax
	movsd	-16(%rbp), %xmm1
	movapd	%xmm1, %xmm2
	movapd	%xmm0, %xmm1
	movq	%rax, %xmm0
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$3, %eax
	call	printf@PLT
	movq	$0, -24(%rbp)
	jmp	.L9
.L8:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L12
	jmp	.L13
.L6:
	movq	$1, -24(%rbp)
	jmp	.L9
.L14:
	nop
.L9:
	jmp	.L11
.L13:
	call	__stack_chk_fail@PLT
.L12:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	main, .-main
	.section	.rodata
	.align 8
.LC2:
	.long	0
	.long	1073741824
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
