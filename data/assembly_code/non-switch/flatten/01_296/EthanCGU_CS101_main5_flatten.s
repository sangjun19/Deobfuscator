	.file	"EthanCGU_CS101_main5_flatten.c"
	.text
	.globl	_TIG_IZ_iKPW_argc
	.bss
	.align 4
	.type	_TIG_IZ_iKPW_argc, @object
	.size	_TIG_IZ_iKPW_argc, 4
_TIG_IZ_iKPW_argc:
	.zero	4
	.globl	_TIG_IZ_iKPW_argv
	.align 8
	.type	_TIG_IZ_iKPW_argv, @object
	.size	_TIG_IZ_iKPW_argv, 8
_TIG_IZ_iKPW_argv:
	.zero	8
	.globl	_TIG_IZ_iKPW_envp
	.align 8
	.type	_TIG_IZ_iKPW_envp, @object
	.size	_TIG_IZ_iKPW_envp, 8
_TIG_IZ_iKPW_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"%d \345\205\203\n"
.LC1:
	.string	"240\345\205\203"
.LC2:
	.string	"\345\205\215\350\262\273"
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
	subq	$48, %rsp
	movl	%edi, -20(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movq	$0, _TIG_IZ_iKPW_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_iKPW_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_iKPW_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 104 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-iKPW--0
# 0 "" 2
#NO_APP
	movl	-20(%rbp), %eax
	movl	%eax, _TIG_IZ_iKPW_argc(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, _TIG_IZ_iKPW_argv(%rip)
	movq	-40(%rbp), %rax
	movq	%rax, _TIG_IZ_iKPW_envp(%rip)
	nop
	movq	$1, -8(%rbp)
.L25:
	cmpq	$9, -8(%rbp)
	ja	.L27
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
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L27-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L12:
	cmpl	$30, -16(%rbp)
	jg	.L17
	movq	$0, -8(%rbp)
	jmp	.L19
.L17:
	movq	$3, -8(%rbp)
	jmp	.L19
.L9:
	movl	-16(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$9, -8(%rbp)
	jmp	.L19
.L15:
	movl	$119, -16(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L19
.L13:
	cmpl	$239, -16(%rbp)
	jle	.L20
	movq	$5, -8(%rbp)
	jmp	.L19
.L20:
	movq	$2, -8(%rbp)
	jmp	.L19
.L7:
	movl	$0, %eax
	jmp	.L26
.L10:
	movl	-16(%rbp), %eax
	movslq	%eax, %rdx
	imulq	$-2004318071, %rdx, %rdx
	shrq	$32, %rdx
	addl	%eax, %edx
	sarl	$4, %edx
	sarl	$31, %eax
	movl	%eax, %ecx
	movl	%edx, %eax
	subl	%ecx, %eax
	addl	$1, %eax
	imull	$30, %eax, %eax
	movl	%eax, -12(%rbp)
	movl	-12(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$9, -8(%rbp)
	jmp	.L19
.L11:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$9, -8(%rbp)
	jmp	.L19
.L16:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$9, -8(%rbp)
	jmp	.L19
.L14:
	movl	-16(%rbp), %eax
	movslq	%eax, %rdx
	imulq	$-2004318071, %rdx, %rdx
	shrq	$32, %rdx
	addl	%eax, %edx
	sarl	$4, %edx
	movl	%eax, %ecx
	sarl	$31, %ecx
	subl	%ecx, %edx
	imull	$30, %edx, %ecx
	subl	%ecx, %eax
	movl	%eax, %edx
	testl	%edx, %edx
	je	.L23
	movq	$6, -8(%rbp)
	jmp	.L19
.L23:
	movq	$8, -8(%rbp)
	jmp	.L19
.L27:
	nop
.L19:
	jmp	.L25
.L26:
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
