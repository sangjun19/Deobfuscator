	.file	"DagnachewA_alx-low_level_programming_101-print_comb4_flatten.c"
	.text
	.globl	_TIG_IZ_gOiX_argv
	.bss
	.align 8
	.type	_TIG_IZ_gOiX_argv, @object
	.size	_TIG_IZ_gOiX_argv, 8
_TIG_IZ_gOiX_argv:
	.zero	8
	.globl	_TIG_IZ_gOiX_argc
	.align 4
	.type	_TIG_IZ_gOiX_argc, @object
	.size	_TIG_IZ_gOiX_argc, 4
_TIG_IZ_gOiX_argc:
	.zero	4
	.globl	_TIG_IZ_gOiX_envp
	.align 8
	.type	_TIG_IZ_gOiX_envp, @object
	.size	_TIG_IZ_gOiX_envp, 8
_TIG_IZ_gOiX_envp:
	.zero	8
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
	movq	$0, _TIG_IZ_gOiX_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_gOiX_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_gOiX_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 105 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-gOiX--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_gOiX_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_gOiX_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_gOiX_envp(%rip)
	nop
	movq	$4, -8(%rbp)
.L38:
	movq	-8(%rbp), %rax
	subq	$3, %rax
	cmpq	$21, %rax
	ja	.L40
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
	.long	.L23-.L8
	.long	.L22-.L8
	.long	.L21-.L8
	.long	.L40-.L8
	.long	.L40-.L8
	.long	.L40-.L8
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L40-.L8
	.long	.L16-.L8
	.long	.L40-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L40-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L13:
	movl	-20(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -16(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L24
.L22:
	movl	$0, -20(%rbp)
	movq	$23, -8(%rbp)
	jmp	.L24
.L16:
	addl	$1, -12(%rbp)
	movq	$12, -8(%rbp)
	jmp	.L24
.L17:
	cmpl	$9, -12(%rbp)
	jg	.L25
	movq	$19, -8(%rbp)
	jmp	.L24
.L25:
	movq	$10, -8(%rbp)
	jmp	.L24
.L9:
	cmpl	$8, -20(%rbp)
	jg	.L27
	movq	$18, -8(%rbp)
	jmp	.L24
.L27:
	movq	$16, -8(%rbp)
	jmp	.L24
.L23:
	cmpl	$9, -16(%rbp)
	jg	.L29
	movq	$11, -8(%rbp)
	jmp	.L24
.L29:
	movq	$5, -8(%rbp)
	jmp	.L24
.L15:
	movl	$10, %edi
	call	putchar@PLT
	movq	$17, -8(%rbp)
	jmp	.L24
.L7:
	movl	$44, %edi
	call	putchar@PLT
	movl	$32, %edi
	call	putchar@PLT
	movq	$14, -8(%rbp)
	jmp	.L24
.L11:
	cmpl	$7, -20(%rbp)
	jne	.L31
	movq	$22, -8(%rbp)
	jmp	.L24
.L31:
	movq	$24, -8(%rbp)
	jmp	.L24
.L18:
	movl	-16(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -12(%rbp)
	movq	$12, -8(%rbp)
	jmp	.L24
.L20:
	cmpl	$9, -12(%rbp)
	jne	.L33
	movq	$14, -8(%rbp)
	jmp	.L24
.L33:
	movq	$24, -8(%rbp)
	jmp	.L24
.L12:
	movl	-20(%rbp), %ecx
	movslq	%ecx, %rax
	imulq	$1717986919, %rax, %rax
	shrq	$32, %rax
	sarl	$2, %eax
	movl	%ecx, %esi
	sarl	$31, %esi
	subl	%esi, %eax
	movl	%eax, %edx
	movl	%edx, %eax
	sall	$2, %eax
	addl	%edx, %eax
	addl	%eax, %eax
	subl	%eax, %ecx
	movl	%ecx, %edx
	leal	48(%rdx), %eax
	movl	%eax, %edi
	call	putchar@PLT
	movl	-16(%rbp), %ecx
	movslq	%ecx, %rax
	imulq	$1717986919, %rax, %rax
	shrq	$32, %rax
	sarl	$2, %eax
	movl	%ecx, %esi
	sarl	$31, %esi
	subl	%esi, %eax
	movl	%eax, %edx
	movl	%edx, %eax
	sall	$2, %eax
	addl	%edx, %eax
	addl	%eax, %eax
	subl	%eax, %ecx
	movl	%ecx, %edx
	leal	48(%rdx), %eax
	movl	%eax, %edi
	call	putchar@PLT
	movl	-12(%rbp), %ecx
	movslq	%ecx, %rax
	imulq	$1717986919, %rax, %rax
	shrq	$32, %rax
	sarl	$2, %eax
	movl	%ecx, %esi
	sarl	$31, %esi
	subl	%esi, %eax
	movl	%eax, %edx
	movl	%edx, %eax
	sall	$2, %eax
	addl	%edx, %eax
	addl	%eax, %eax
	subl	%eax, %ecx
	movl	%ecx, %edx
	leal	48(%rdx), %eax
	movl	%eax, %edi
	call	putchar@PLT
	movq	$21, -8(%rbp)
	jmp	.L24
.L14:
	movl	$0, %eax
	jmp	.L39
.L10:
	cmpl	$8, -16(%rbp)
	jne	.L36
	movq	$9, -8(%rbp)
	jmp	.L24
.L36:
	movq	$24, -8(%rbp)
	jmp	.L24
.L21:
	addl	$1, -20(%rbp)
	movq	$23, -8(%rbp)
	jmp	.L24
.L19:
	addl	$1, -16(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L24
.L40:
	nop
.L24:
	jmp	.L38
.L39:
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
