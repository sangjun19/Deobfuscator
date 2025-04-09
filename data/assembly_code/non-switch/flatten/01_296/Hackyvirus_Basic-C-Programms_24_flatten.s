	.file	"Hackyvirus_Basic-C-Programms_24_flatten.c"
	.text
	.globl	_TIG_IZ_GRUW_argc
	.bss
	.align 4
	.type	_TIG_IZ_GRUW_argc, @object
	.size	_TIG_IZ_GRUW_argc, 4
_TIG_IZ_GRUW_argc:
	.zero	4
	.globl	_TIG_IZ_GRUW_argv
	.align 8
	.type	_TIG_IZ_GRUW_argv, @object
	.size	_TIG_IZ_GRUW_argv, 8
_TIG_IZ_GRUW_argv:
	.zero	8
	.globl	_TIG_IZ_GRUW_envp
	.align 8
	.type	_TIG_IZ_GRUW_envp, @object
	.size	_TIG_IZ_GRUW_envp, 8
_TIG_IZ_GRUW_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"%d * %d = %d\n"
.LC1:
	.string	"Enter the number: "
.LC2:
	.string	"%d"
	.align 8
.LC3:
	.string	"Multiplication table using for loop"
	.align 8
.LC4:
	.string	"Multiplication table using do-while loop"
	.align 8
.LC5:
	.string	"Multiplication table using while loop"
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
	movq	$0, _TIG_IZ_GRUW_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_GRUW_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_GRUW_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 106 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-GRUW--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_GRUW_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_GRUW_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_GRUW_envp(%rip)
	nop
	movq	$19, -16(%rbp)
.L27:
	cmpq	$21, -16(%rbp)
	ja	.L30
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
	.long	.L30-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L30-.L8
	.long	.L30-.L8
	.long	.L14-.L8
	.long	.L30-.L8
	.long	.L30-.L8
	.long	.L30-.L8
	.long	.L13-.L8
	.long	.L30-.L8
	.long	.L30-.L8
	.long	.L30-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L30-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L30-.L8
	.long	.L7-.L8
	.text
.L10:
	cmpl	$10, -20(%rbp)
	jg	.L19
	movq	$15, -16(%rbp)
	jmp	.L21
.L19:
	movq	$4, -16(%rbp)
	jmp	.L21
.L15:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L28
	jmp	.L29
.L12:
	movl	-32(%rbp), %eax
	imull	-20(%rbp), %eax
	movl	%eax, %ecx
	movl	-32(%rbp), %eax
	movl	-20(%rbp), %edx
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -20(%rbp)
	movq	$18, -16(%rbp)
	jmp	.L21
.L18:
	cmpl	$10, -24(%rbp)
	jg	.L23
	movq	$2, -16(%rbp)
	jmp	.L21
.L23:
	movq	$11, -16(%rbp)
	jmp	.L21
.L16:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-32(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$1, -28(%rbp)
	movq	$16, -16(%rbp)
	jmp	.L21
.L11:
	cmpl	$10, -28(%rbp)
	jg	.L25
	movq	$21, -16(%rbp)
	jmp	.L21
.L25:
	movq	$7, -16(%rbp)
	jmp	.L21
.L7:
	movl	-32(%rbp), %eax
	imull	-28(%rbp), %eax
	movl	%eax, %ecx
	movl	-32(%rbp), %eax
	movl	-28(%rbp), %edx
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -28(%rbp)
	movq	$16, -16(%rbp)
	jmp	.L21
.L13:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$1, -20(%rbp)
	movq	$15, -16(%rbp)
	jmp	.L21
.L9:
	movq	$3, -16(%rbp)
	jmp	.L21
.L14:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$1, -24(%rbp)
	movq	$1, -16(%rbp)
	jmp	.L21
.L17:
	movl	-32(%rbp), %eax
	imull	-24(%rbp), %eax
	movl	%eax, %ecx
	movl	-32(%rbp), %eax
	movl	-24(%rbp), %edx
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -24(%rbp)
	movq	$1, -16(%rbp)
	jmp	.L21
.L30:
	nop
.L21:
	jmp	.L27
.L29:
	call	__stack_chk_fail@PLT
.L28:
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
