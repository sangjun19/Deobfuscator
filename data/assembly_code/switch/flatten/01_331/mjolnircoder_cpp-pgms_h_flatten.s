	.file	"mjolnircoder_cpp-pgms_h_flatten.c"
	.text
	.globl	_TIG_IZ_0UlY_argc
	.bss
	.align 4
	.type	_TIG_IZ_0UlY_argc, @object
	.size	_TIG_IZ_0UlY_argc, 4
_TIG_IZ_0UlY_argc:
	.zero	4
	.globl	_TIG_IZ_0UlY_envp
	.align 8
	.type	_TIG_IZ_0UlY_envp, @object
	.size	_TIG_IZ_0UlY_envp, 8
_TIG_IZ_0UlY_envp:
	.zero	8
	.globl	_TIG_IZ_0UlY_argv
	.align 8
	.type	_TIG_IZ_0UlY_argv, @object
	.size	_TIG_IZ_0UlY_argv, 8
_TIG_IZ_0UlY_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"Enter the string:"
.LC1:
	.string	"% [^\n] s"
.LC2:
	.string	"*"
.LC3:
	.string	""
	.string	""
.LC4:
	.string	"+"
.LC5:
	.string	"-"
	.align 8
.LC6:
	.string	"Number of Operators in a given string are : %d \n"
.LC7:
	.string	"/"
	.text
	.globl	main
	.type	main, @function
main:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$256, %rsp
	movl	%edi, -228(%rbp)
	movq	%rsi, -240(%rbp)
	movq	%rdx, -248(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_0UlY_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_0UlY_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_0UlY_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 107 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-0UlY--0
# 0 "" 2
#NO_APP
	movl	-228(%rbp), %eax
	movl	%eax, _TIG_IZ_0UlY_argc(%rip)
	movq	-240(%rbp), %rax
	movq	%rax, _TIG_IZ_0UlY_argv(%rip)
	movq	-248(%rbp), %rax
	movq	%rax, _TIG_IZ_0UlY_envp(%rip)
	nop
	movq	$5, -216(%rbp)
.L34:
	cmpq	$15, -216(%rbp)
	ja	.L37
	movq	-216(%rbp), %rax
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
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L37-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L37-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L17:
	addl	$1, -224(%rbp)
	movq	$7, -216(%rbp)
	jmp	.L22
.L9:
	movl	$0, -224(%rbp)
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-208(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	$0, -220(%rbp)
	movq	$3, -216(%rbp)
	jmp	.L22
.L7:
	addl	$1, -224(%rbp)
	movq	$7, -216(%rbp)
	jmp	.L22
.L10:
	movl	-220(%rbp), %eax
	cltq
	movzbl	-208(%rbp,%rax), %eax
	movsbq	%al, %rax
	leaq	.LC2(%rip), %rdx
	cmpq	%rdx, %rax
	jne	.L23
	movq	$6, -216(%rbp)
	jmp	.L22
.L23:
	movq	$0, -216(%rbp)
	jmp	.L22
.L20:
	addl	$1, -224(%rbp)
	movq	$7, -216(%rbp)
	jmp	.L22
.L18:
	movl	-220(%rbp), %eax
	cltq
	movzbl	-208(%rbp,%rax), %eax
	movsbq	%al, %rax
	leaq	.LC3(%rip), %rdx
	cmpq	%rdx, %rax
	je	.L25
	movq	$11, -216(%rbp)
	jmp	.L22
.L25:
	movq	$10, -216(%rbp)
	jmp	.L22
.L11:
	movl	-220(%rbp), %eax
	cltq
	movzbl	-208(%rbp,%rax), %eax
	movsbq	%al, %rax
	leaq	.LC4(%rip), %rdx
	cmpq	%rdx, %rax
	jne	.L27
	movq	$4, -216(%rbp)
	jmp	.L22
.L27:
	movq	$9, -216(%rbp)
	jmp	.L22
.L13:
	movl	-220(%rbp), %eax
	cltq
	movzbl	-208(%rbp,%rax), %eax
	movsbq	%al, %rax
	leaq	.LC5(%rip), %rdx
	cmpq	%rdx, %rax
	jne	.L29
	movq	$15, -216(%rbp)
	jmp	.L22
.L29:
	movq	$12, -216(%rbp)
	jmp	.L22
.L15:
	addl	$1, -224(%rbp)
	movq	$7, -216(%rbp)
	jmp	.L22
.L16:
	movq	$14, -216(%rbp)
	jmp	.L22
.L12:
	movl	-224(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$2, -216(%rbp)
	jmp	.L22
.L21:
	movl	-220(%rbp), %eax
	cltq
	movzbl	-208(%rbp,%rax), %eax
	movsbq	%al, %rax
	leaq	.LC7(%rip), %rdx
	cmpq	%rdx, %rax
	jne	.L31
	movq	$1, -216(%rbp)
	jmp	.L22
.L31:
	movq	$7, -216(%rbp)
	jmp	.L22
.L14:
	addl	$1, -220(%rbp)
	movq	$3, -216(%rbp)
	jmp	.L22
.L19:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L35
	jmp	.L36
.L37:
	nop
.L22:
	jmp	.L34
.L36:
	call	__stack_chk_fail@PLT
.L35:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
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
