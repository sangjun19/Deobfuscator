	.file	"vaibhavdostuff_vaibhav-s-work__flatten.c"
	.text
	.globl	_TIG_IZ_o5Rd_argv
	.bss
	.align 8
	.type	_TIG_IZ_o5Rd_argv, @object
	.size	_TIG_IZ_o5Rd_argv, 8
_TIG_IZ_o5Rd_argv:
	.zero	8
	.globl	_TIG_IZ_o5Rd_envp
	.align 8
	.type	_TIG_IZ_o5Rd_envp, @object
	.size	_TIG_IZ_o5Rd_envp, 8
_TIG_IZ_o5Rd_envp:
	.zero	8
	.globl	_TIG_IZ_o5Rd_argc
	.align 4
	.type	_TIG_IZ_o5Rd_argc, @object
	.size	_TIG_IZ_o5Rd_argc, 4
_TIG_IZ_o5Rd_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"Enter the row size:"
.LC1:
	.string	"%d"
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
	subq	$80, %rsp
	movl	%edi, -52(%rbp)
	movq	%rsi, -64(%rbp)
	movq	%rdx, -72(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_o5Rd_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_o5Rd_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_o5Rd_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 107 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-o5Rd--0
# 0 "" 2
#NO_APP
	movl	-52(%rbp), %eax
	movl	%eax, _TIG_IZ_o5Rd_argc(%rip)
	movq	-64(%rbp), %rax
	movq	%rax, _TIG_IZ_o5Rd_argv(%rip)
	movq	-72(%rbp), %rax
	movq	%rax, _TIG_IZ_o5Rd_envp(%rip)
	nop
	movq	$15, -16(%rbp)
.L31:
	cmpq	$16, -16(%rbp)
	ja	.L34
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
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L34-.L8
	.long	.L34-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L34-.L8
	.long	.L34-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L10:
	movl	-36(%rbp), %eax
	addl	%eax, %eax
	cmpl	%eax, -32(%rbp)
	jg	.L21
	movq	$0, -16(%rbp)
	jmp	.L23
.L21:
	movq	$16, -16(%rbp)
	jmp	.L23
.L9:
	movq	$3, -16(%rbp)
	jmp	.L23
.L12:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L32
	jmp	.L33
.L19:
	movl	$42, %edi
	call	putchar@PLT
	movq	$6, -16(%rbp)
	jmp	.L23
.L17:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-36(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	$1, -24(%rbp)
	movl	-36(%rbp), %eax
	addl	%eax, %eax
	subl	$1, %eax
	movl	%eax, -20(%rbp)
	movl	$1, -28(%rbp)
	movq	$13, -16(%rbp)
	jmp	.L23
.L7:
	addl	$1, -24(%rbp)
	subl	$1, -20(%rbp)
	movl	$10, %edi
	call	putchar@PLT
	addl	$1, -28(%rbp)
	movq	$13, -16(%rbp)
	jmp	.L23
.L13:
	movl	$42, %edi
	call	putchar@PLT
	movq	$6, -16(%rbp)
	jmp	.L23
.L11:
	movl	-36(%rbp), %eax
	cmpl	%eax, -28(%rbp)
	jg	.L25
	movq	$10, -16(%rbp)
	jmp	.L23
.L25:
	movq	$12, -16(%rbp)
	jmp	.L23
.L16:
	addl	$1, -32(%rbp)
	movq	$14, -16(%rbp)
	jmp	.L23
.L14:
	movl	$1, -32(%rbp)
	movq	$14, -16(%rbp)
	jmp	.L23
.L20:
	movl	-32(%rbp), %eax
	cmpl	-24(%rbp), %eax
	jne	.L27
	movq	$1, -16(%rbp)
	jmp	.L23
.L27:
	movq	$7, -16(%rbp)
	jmp	.L23
.L15:
	movl	-32(%rbp), %eax
	cmpl	-20(%rbp), %eax
	jne	.L29
	movq	$11, -16(%rbp)
	jmp	.L23
.L29:
	movq	$2, -16(%rbp)
	jmp	.L23
.L18:
	movl	$32, %edi
	call	putchar@PLT
	movq	$6, -16(%rbp)
	jmp	.L23
.L34:
	nop
.L23:
	jmp	.L31
.L33:
	call	__stack_chk_fail@PLT
.L32:
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
