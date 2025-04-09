	.file	"Hackyvirus_Basic-C-Programms_27_flatten.c"
	.text
	.globl	_TIG_IZ_8495_argc
	.bss
	.align 4
	.type	_TIG_IZ_8495_argc, @object
	.size	_TIG_IZ_8495_argc, 4
_TIG_IZ_8495_argc:
	.zero	4
	.globl	_TIG_IZ_8495_argv
	.align 8
	.type	_TIG_IZ_8495_argv, @object
	.size	_TIG_IZ_8495_argv, 8
_TIG_IZ_8495_argv:
	.zero	8
	.globl	_TIG_IZ_8495_envp
	.align 8
	.type	_TIG_IZ_8495_envp, @object
	.size	_TIG_IZ_8495_envp, 8
_TIG_IZ_8495_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"Using for loop"
	.align 8
.LC1:
	.string	"The sum of first 10 natural Numbers: %d\n"
.LC2:
	.string	"Using while loop"
.LC3:
	.string	"Using do-while loop"
.LC4:
	.string	"%d + %d = %d\n"
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
	movq	$0, _TIG_IZ_8495_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_8495_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_8495_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 105 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-8495--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_8495_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_8495_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_8495_envp(%rip)
	nop
	movq	$9, -8(%rbp)
.L28:
	cmpq	$18, -8(%rbp)
	ja	.L30
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
	.long	.L30-.L8
	.long	.L19-.L8
	.long	.L30-.L8
	.long	.L18-.L8
	.long	.L30-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L30-.L8
	.long	.L30-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L30-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L30-.L8
	.long	.L7-.L8
	.text
.L7:
	movl	$0, -32(%rbp)
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$1, -28(%rbp)
	movq	$11, -8(%rbp)
	jmp	.L20
.L10:
	movl	$0, %eax
	jmp	.L29
.L12:
	movl	-12(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$15, -8(%rbp)
	jmp	.L20
.L19:
	movl	-32(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$1, -24(%rbp)
	movl	$0, -20(%rbp)
	movq	$13, -8(%rbp)
	jmp	.L20
.L18:
	cmpl	$10, -16(%rbp)
	jg	.L22
	movq	$5, -8(%rbp)
	jmp	.L20
.L22:
	movq	$12, -8(%rbp)
	jmp	.L20
.L9:
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$1, -16(%rbp)
	movl	$0, -12(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L20
.L13:
	cmpl	$10, -28(%rbp)
	jg	.L24
	movq	$6, -8(%rbp)
	jmp	.L20
.L24:
	movq	$1, -8(%rbp)
	jmp	.L20
.L15:
	movq	$18, -8(%rbp)
	jmp	.L20
.L11:
	cmpl	$10, -24(%rbp)
	jg	.L26
	movq	$10, -8(%rbp)
	jmp	.L20
.L26:
	movq	$16, -8(%rbp)
	jmp	.L20
.L16:
	movl	-28(%rbp), %eax
	addl	%eax, -32(%rbp)
	movl	-32(%rbp), %ecx
	movl	-32(%rbp), %edx
	movl	-28(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -28(%rbp)
	movq	$11, -8(%rbp)
	jmp	.L20
.L17:
	movl	-16(%rbp), %eax
	addl	%eax, -12(%rbp)
	movl	-12(%rbp), %ecx
	movl	-12(%rbp), %edx
	movl	-16(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -16(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L20
.L14:
	movl	-24(%rbp), %eax
	addl	%eax, -20(%rbp)
	movl	-20(%rbp), %ecx
	movl	-20(%rbp), %edx
	movl	-24(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -24(%rbp)
	movq	$13, -8(%rbp)
	jmp	.L20
.L30:
	nop
.L20:
	jmp	.L28
.L29:
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
