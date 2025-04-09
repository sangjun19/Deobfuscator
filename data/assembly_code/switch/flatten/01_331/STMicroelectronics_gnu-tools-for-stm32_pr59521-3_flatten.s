	.file	"STMicroelectronics_gnu-tools-for-stm32_pr59521-3_flatten.c"
	.text
	.globl	_TIG_IZ_TGlX_argv
	.bss
	.align 8
	.type	_TIG_IZ_TGlX_argv, @object
	.size	_TIG_IZ_TGlX_argv, 8
_TIG_IZ_TGlX_argv:
	.zero	8
	.globl	_TIG_IZ_TGlX_argc
	.align 4
	.type	_TIG_IZ_TGlX_argc, @object
	.size	_TIG_IZ_TGlX_argc, 4
_TIG_IZ_TGlX_argc:
	.zero	4
	.globl	_TIG_IZ_TGlX_envp
	.align 8
	.type	_TIG_IZ_TGlX_envp, @object
	.size	_TIG_IZ_TGlX_envp, 8
_TIG_IZ_TGlX_envp:
	.zero	8
	.text
	.globl	sink
	.type	sink, @function
sink:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movq	$1, -8(%rbp)
.L6:
	cmpq	$0, -8(%rbp)
	je	.L7
	cmpq	$1, -8(%rbp)
	jne	.L8
	movq	-24(%rbp), %rax
	movq	$0, -8(%rbp)
	jmp	.L4
.L8:
	nop
.L4:
	jmp	.L6
.L7:
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	sink, .-sink
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
	subq	$48, %rsp
	movl	%edi, -20(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movq	$0, _TIG_IZ_TGlX_envp(%rip)
	nop
.L10:
	movq	$0, _TIG_IZ_TGlX_argv(%rip)
	nop
.L11:
	movl	$0, _TIG_IZ_TGlX_argc(%rip)
	nop
	nop
.L12:
.L13:
#APP
# 126 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-TGlX--0
# 0 "" 2
#NO_APP
	movl	-20(%rbp), %eax
	movl	%eax, _TIG_IZ_TGlX_argc(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, _TIG_IZ_TGlX_argv(%rip)
	movq	-40(%rbp), %rax
	movq	%rax, _TIG_IZ_TGlX_envp(%rip)
	nop
	movq	$1, -8(%rbp)
.L33:
	cmpq	$11, -8(%rbp)
	ja	.L35
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L16(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L16(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L16:
	.long	.L35-.L16
	.long	.L24-.L16
	.long	.L23-.L16
	.long	.L22-.L16
	.long	.L21-.L16
	.long	.L35-.L16
	.long	.L20-.L16
	.long	.L19-.L16
	.long	.L18-.L16
	.long	.L17-.L16
	.long	.L35-.L16
	.long	.L15-.L16
	.text
.L21:
	movl	$100, -12(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L25
.L18:
	cmpl	$9999, -16(%rbp)
	jg	.L26
	movq	$2, -8(%rbp)
	jmp	.L25
.L26:
	movq	$9, -8(%rbp)
	jmp	.L25
.L24:
	movl	$0, -16(%rbp)
	movq	$8, -8(%rbp)
	jmp	.L25
.L22:
	movl	-12(%rbp), %eax
	movl	%eax, %edi
	call	foo
	addl	$1, -16(%rbp)
	movq	$8, -8(%rbp)
	jmp	.L25
.L15:
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
	testl	%edx, %edx
	jne	.L28
	movq	$6, -8(%rbp)
	jmp	.L25
.L28:
	movq	$7, -8(%rbp)
	jmp	.L25
.L17:
	movl	$0, %eax
	jmp	.L34
.L20:
	movl	$10, -12(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L25
.L19:
	movl	$1, -12(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L25
.L23:
	movl	-16(%rbp), %edx
	movslq	%edx, %rax
	imulq	$1374389535, %rax, %rax
	shrq	$32, %rax
	sarl	$5, %eax
	movl	%edx, %ecx
	sarl	$31, %ecx
	subl	%ecx, %eax
	imull	$100, %eax, %ecx
	movl	%edx, %eax
	subl	%ecx, %eax
	testl	%eax, %eax
	jne	.L31
	movq	$4, -8(%rbp)
	jmp	.L25
.L31:
	movq	$11, -8(%rbp)
	jmp	.L25
.L35:
	nop
.L25:
	jmp	.L33
.L34:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	main, .-main
	.section	.rodata
.LC0:
	.string	"1"
.LC1:
	.string	"10"
.LC2:
	.string	"100"
	.text
	.globl	foo
	.type	foo, @function
foo:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$24, %rsp
	movl	%edi, -20(%rbp)
	movq	$8, -8(%rbp)
.L52:
	cmpq	$8, -8(%rbp)
	ja	.L53
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L39(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L39(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L39:
	.long	.L44-.L39
	.long	.L53-.L39
	.long	.L43-.L39
	.long	.L53-.L39
	.long	.L54-.L39
	.long	.L41-.L39
	.long	.L53-.L39
	.long	.L40-.L39
	.long	.L38-.L39
	.text
.L38:
	cmpl	$100, -20(%rbp)
	je	.L46
	cmpl	$100, -20(%rbp)
	jg	.L47
	cmpl	$1, -20(%rbp)
	je	.L48
	cmpl	$10, -20(%rbp)
	je	.L49
	jmp	.L47
.L48:
	movq	$5, -8(%rbp)
	jmp	.L50
.L49:
	movq	$0, -8(%rbp)
	jmp	.L50
.L46:
	movq	$7, -8(%rbp)
	jmp	.L50
.L47:
	movq	$2, -8(%rbp)
	nop
.L50:
	jmp	.L51
.L41:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	sink
	movq	$4, -8(%rbp)
	jmp	.L51
.L44:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	sink
	movq	$4, -8(%rbp)
	jmp	.L51
.L40:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	sink
	movq	$4, -8(%rbp)
	jmp	.L51
.L43:
	movq	$4, -8(%rbp)
	jmp	.L51
.L53:
	nop
.L51:
	jmp	.L52
.L54:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	foo, .-foo
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
