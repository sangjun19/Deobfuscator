	.file	"rarimuk_alx-low_level_programming_100-prime_factor_flatten.c"
	.text
	.globl	_TIG_IZ_You0_argc
	.bss
	.align 4
	.type	_TIG_IZ_You0_argc, @object
	.size	_TIG_IZ_You0_argc, 4
_TIG_IZ_You0_argc:
	.zero	4
	.globl	_TIG_IZ_You0_envp
	.align 8
	.type	_TIG_IZ_You0_envp, @object
	.size	_TIG_IZ_You0_envp, 8
_TIG_IZ_You0_envp:
	.zero	8
	.globl	_TIG_IZ_You0_argv
	.align 8
	.type	_TIG_IZ_You0_argv, @object
	.size	_TIG_IZ_You0_argv, 8
_TIG_IZ_You0_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"%ld\n"
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
	subq	$80, %rsp
	movl	%edi, -52(%rbp)
	movq	%rsi, -64(%rbp)
	movq	%rdx, -72(%rbp)
	movq	$0, _TIG_IZ_You0_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_You0_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_You0_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 110 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-You0--0
# 0 "" 2
#NO_APP
	movl	-52(%rbp), %eax
	movl	%eax, _TIG_IZ_You0_argc(%rip)
	movq	-64(%rbp), %rax
	movq	%rax, _TIG_IZ_You0_argv(%rip)
	movq	-72(%rbp), %rax
	movq	%rax, _TIG_IZ_You0_envp(%rip)
	nop
	movq	$21, -8(%rbp)
.L33:
	cmpq	$21, -8(%rbp)
	ja	.L37
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
	.long	.L21-.L8
	.long	.L37-.L8
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L37-.L8
	.long	.L37-.L8
	.long	.L37-.L8
	.long	.L37-.L8
	.long	.L37-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L37-.L8
	.long	.L37-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L11:
	movq	-40(%rbp), %rax
	cqto
	idivq	-24(%rbp)
	movq	%rdx, %rax
	testq	%rax, %rax
	jne	.L22
	movq	$13, -8(%rbp)
	jmp	.L24
.L22:
	movq	$11, -8(%rbp)
	jmp	.L24
.L13:
	cmpq	$2, -40(%rbp)
	jle	.L25
	movq	$2, -8(%rbp)
	jmp	.L24
.L25:
	movq	$10, -8(%rbp)
	jmp	.L24
.L15:
	movq	$2, -32(%rbp)
	movq	-40(%rbp), %rax
	movq	%rax, %rdx
	shrq	$63, %rdx
	addq	%rdx, %rax
	sarq	%rax
	movq	%rax, -40(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L24
.L19:
	movq	$3, -24(%rbp)
	movq	$9, -8(%rbp)
	jmp	.L24
.L7:
	movq	$17, -8(%rbp)
	jmp	.L24
.L16:
	addq	$2, -24(%rbp)
	movq	$9, -8(%rbp)
	jmp	.L24
.L18:
	pxor	%xmm2, %xmm2
	cvtsi2sdq	-40(%rbp), %xmm2
	movq	%xmm2, %rax
	movq	%rax, %xmm0
	call	sqrt@PLT
	movq	%xmm0, %rax
	movq	%rax, -16(%rbp)
	movq	$20, -8(%rbp)
	jmp	.L24
.L14:
	movq	-24(%rbp), %rax
	movq	%rax, -32(%rbp)
	movq	-40(%rbp), %rax
	cqto
	idivq	-24(%rbp)
	movq	%rax, -40(%rbp)
	movq	$18, -8(%rbp)
	jmp	.L24
.L10:
	movl	$0, %eax
	jmp	.L35
.L12:
	movabsq	$612852475143, %rax
	movq	%rax, -40(%rbp)
	movq	$-1, -32(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L24
.L17:
	movq	-32(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$19, -8(%rbp)
	jmp	.L24
.L21:
	movq	-40(%rbp), %rax
	andl	$1, %eax
	testq	%rax, %rax
	jne	.L28
	movq	$12, -8(%rbp)
	jmp	.L24
.L28:
	movq	$3, -8(%rbp)
	jmp	.L24
.L20:
	movq	-40(%rbp), %rax
	movq	%rax, -32(%rbp)
	movq	$10, -8(%rbp)
	jmp	.L24
.L9:
	pxor	%xmm1, %xmm1
	cvtsi2sdq	-24(%rbp), %xmm1
	movsd	-16(%rbp), %xmm0
	comisd	%xmm1, %xmm0
	jb	.L36
	movq	$18, -8(%rbp)
	jmp	.L24
.L36:
	movq	$14, -8(%rbp)
	jmp	.L24
.L37:
	nop
.L24:
	jmp	.L33
.L35:
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
