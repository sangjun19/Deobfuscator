	.file	"EGABENADETTE10_alx-low_level_programming_101-print_comb4_flatten.c"
	.text
	.globl	_TIG_IZ_XrQh_envp
	.bss
	.align 8
	.type	_TIG_IZ_XrQh_envp, @object
	.size	_TIG_IZ_XrQh_envp, 8
_TIG_IZ_XrQh_envp:
	.zero	8
	.globl	_TIG_IZ_XrQh_argv
	.align 8
	.type	_TIG_IZ_XrQh_argv, @object
	.size	_TIG_IZ_XrQh_argv, 8
_TIG_IZ_XrQh_argv:
	.zero	8
	.globl	_TIG_IZ_XrQh_argc
	.align 4
	.type	_TIG_IZ_XrQh_argc, @object
	.size	_TIG_IZ_XrQh_argc, 4
_TIG_IZ_XrQh_argc:
	.zero	4
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
	movq	$0, _TIG_IZ_XrQh_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_XrQh_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_XrQh_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 109 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-XrQh--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_XrQh_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_XrQh_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_XrQh_envp(%rip)
	nop
	movq	$25, -8(%rbp)
.L42:
	cmpq	$25, -8(%rbp)
	ja	.L44
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
	.long	.L44-.L8
	.long	.L44-.L8
	.long	.L25-.L8
	.long	.L24-.L8
	.long	.L23-.L8
	.long	.L22-.L8
	.long	.L44-.L8
	.long	.L21-.L8
	.long	.L44-.L8
	.long	.L44-.L8
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L44-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L44-.L8
	.long	.L44-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L13:
	addl	$1, -12(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L26
.L7:
	movl	$48, -20(%rbp)
	movq	$19, -8(%rbp)
	jmp	.L26
.L23:
	movl	$49, -16(%rbp)
	movq	$12, -8(%rbp)
	jmp	.L26
.L16:
	movl	$44, %edi
	call	putchar@PLT
	movl	$32, %edi
	call	putchar@PLT
	movq	$18, -8(%rbp)
	jmp	.L26
.L18:
	cmpl	$57, -16(%rbp)
	jg	.L27
	movq	$13, -8(%rbp)
	jmp	.L26
.L27:
	movq	$3, -8(%rbp)
	jmp	.L26
.L10:
	movl	$0, %eax
	jmp	.L43
.L24:
	addl	$1, -20(%rbp)
	movq	$19, -8(%rbp)
	jmp	.L26
.L15:
	movl	-20(%rbp), %eax
	movl	%eax, %edi
	call	putchar@PLT
	movl	-16(%rbp), %eax
	movl	%eax, %edi
	call	putchar@PLT
	movl	-12(%rbp), %eax
	movl	%eax, %edi
	call	putchar@PLT
	movq	$2, -8(%rbp)
	jmp	.L26
.L9:
	movl	-16(%rbp), %eax
	cmpl	-20(%rbp), %eax
	jle	.L30
	movq	$16, -8(%rbp)
	jmp	.L26
.L30:
	movq	$18, -8(%rbp)
	jmp	.L26
.L19:
	cmpl	$56, -16(%rbp)
	je	.L32
	movq	$17, -8(%rbp)
	jmp	.L26
.L32:
	movq	$18, -8(%rbp)
	jmp	.L26
.L17:
	movl	$50, -12(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L26
.L12:
	cmpl	$57, -20(%rbp)
	jg	.L34
	movq	$4, -8(%rbp)
	jmp	.L26
.L34:
	movq	$10, -8(%rbp)
	jmp	.L26
.L14:
	movl	$44, %edi
	call	putchar@PLT
	movl	$32, %edi
	call	putchar@PLT
	movq	$18, -8(%rbp)
	jmp	.L26
.L11:
	movl	-12(%rbp), %eax
	cmpl	-16(%rbp), %eax
	jle	.L36
	movq	$24, -8(%rbp)
	jmp	.L26
.L36:
	movq	$18, -8(%rbp)
	jmp	.L26
.L22:
	addl	$1, -16(%rbp)
	movq	$12, -8(%rbp)
	jmp	.L26
.L20:
	movl	$10, %edi
	call	putchar@PLT
	movq	$23, -8(%rbp)
	jmp	.L26
.L21:
	cmpl	$57, -12(%rbp)
	jg	.L38
	movq	$22, -8(%rbp)
	jmp	.L26
.L38:
	movq	$5, -8(%rbp)
	jmp	.L26
.L25:
	cmpl	$55, -20(%rbp)
	je	.L40
	movq	$15, -8(%rbp)
	jmp	.L26
.L40:
	movq	$11, -8(%rbp)
	jmp	.L26
.L44:
	nop
.L26:
	jmp	.L42
.L43:
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
