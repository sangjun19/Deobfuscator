	.file	"Jihun22_JavaTest_ex4_flatten.c"
	.text
	.globl	_TIG_IZ_fkbZ_argc
	.bss
	.align 4
	.type	_TIG_IZ_fkbZ_argc, @object
	.size	_TIG_IZ_fkbZ_argc, 4
_TIG_IZ_fkbZ_argc:
	.zero	4
	.globl	_TIG_IZ_fkbZ_argv
	.align 8
	.type	_TIG_IZ_fkbZ_argv, @object
	.size	_TIG_IZ_fkbZ_argv, 8
_TIG_IZ_fkbZ_argv:
	.zero	8
	.globl	_TIG_IZ_fkbZ_envp
	.align 8
	.type	_TIG_IZ_fkbZ_envp, @object
	.size	_TIG_IZ_fkbZ_envp, 8
_TIG_IZ_fkbZ_envp:
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
	subq	$80, %rsp
	movl	%edi, -52(%rbp)
	movq	%rsi, -64(%rbp)
	movq	%rdx, -72(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_fkbZ_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_fkbZ_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_fkbZ_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 126 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-fkbZ--0
# 0 "" 2
#NO_APP
	movl	-52(%rbp), %eax
	movl	%eax, _TIG_IZ_fkbZ_argc(%rip)
	movq	-64(%rbp), %rax
	movq	%rax, _TIG_IZ_fkbZ_argv(%rip)
	movq	-72(%rbp), %rax
	movq	%rax, _TIG_IZ_fkbZ_envp(%rip)
	nop
	movq	$5, -32(%rbp)
.L28:
	cmpq	$11, -32(%rbp)
	ja	.L31
	movq	-32(%rbp), %rax
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
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L31-.L8
	.long	.L9-.L8
	.long	.L31-.L8
	.long	.L7-.L8
	.text
.L13:
	movl	-36(%rbp), %eax
	movslq	%eax, %rdx
	imulq	$-2004318071, %rdx, %rdx
	shrq	$32, %rdx
	addl	%eax, %edx
	sarl	$4, %edx
	sarl	$31, %eax
	movl	%eax, %ecx
	movl	%edx, %eax
	subl	%ecx, %eax
	cmpl	$10, %eax
	jg	.L18
	cmpl	$9, %eax
	jge	.L19
	cmpl	$7, %eax
	jg	.L20
	cmpl	$6, %eax
	jge	.L21
	jmp	.L18
.L20:
	cmpl	$8, %eax
	je	.L22
	jmp	.L18
.L21:
	movq	$9, -32(%rbp)
	jmp	.L23
.L22:
	movq	$11, -32(%rbp)
	jmp	.L23
.L19:
	movq	$2, -32(%rbp)
	jmp	.L23
.L18:
	movq	$1, -32(%rbp)
	nop
.L23:
	jmp	.L24
.L16:
	movl	$68, %edi
	call	putchar@PLT
	movq	$3, -32(%rbp)
	jmp	.L24
.L14:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L29
	jmp	.L30
.L7:
	movl	$66, %edi
	call	putchar@PLT
	movq	$9, -32(%rbp)
	jmp	.L24
.L9:
	movl	$67, %edi
	call	putchar@PLT
	movq	$1, -32(%rbp)
	jmp	.L24
.L11:
	movl	-40(%rbp), %eax
	cltq
	movl	-20(%rbp,%rax,4), %eax
	addl	%eax, -36(%rbp)
	addl	$1, -40(%rbp)
	movq	$7, -32(%rbp)
	jmp	.L24
.L12:
	movq	$0, -32(%rbp)
	jmp	.L24
.L17:
	movl	$73, -20(%rbp)
	movl	$95, -16(%rbp)
	movl	$82, -12(%rbp)
	movl	$0, -36(%rbp)
	movl	$0, -40(%rbp)
	movq	$7, -32(%rbp)
	jmp	.L24
.L10:
	cmpl	$2, -40(%rbp)
	jg	.L26
	movq	$6, -32(%rbp)
	jmp	.L24
.L26:
	movq	$4, -32(%rbp)
	jmp	.L24
.L15:
	movl	$65, %edi
	call	putchar@PLT
	movq	$11, -32(%rbp)
	jmp	.L24
.L31:
	nop
.L24:
	jmp	.L28
.L30:
	call	__stack_chk_fail@PLT
.L29:
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
